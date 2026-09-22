import logging
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from trievo.checkpoint import load_checkpoint, save_checkpoint
from trievo.data import get_batch
from trievo.nn_utils import cross_entropy_loss
from trievo.optimizer import AdamW, cosine_annealing_lr_schedule, gradient_clipping
from evaluation import evaluate


def train(
    model: torch.nn.Module,
    train_dataset_path: os.PathLike,
    val_dataset_path: os.PathLike,
    max_lr: float,
    min_lr: float,
    betas: tuple,
    eps: float,
    weight_decay: float,
    train_num_iterations: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
    warm_up_steps: int,
    anneal_steps: int,
    max_l2_norm: float,
    checkpoint_dir: os.PathLike,
    save_interval: int,
    log_interval: int,
    tb_log_dir: os.PathLike,
    val_num_iterations: int,
    val_interval: int,
    resume_from: int = 0,
    rank: int = 0,
    world_size: int = 1,
    num_layers: int = 0,
    d_model: int = 0,
    gpu_peak_tflops: float = 330.0,
):
    is_main_process = rank == 0
    writer = SummaryWriter(log_dir=tb_log_dir) if tb_log_dir and is_main_process else None
    training_dataset = np.memmap(train_dataset_path, dtype="uint16", mode="r")
    val_dataset = np.memmap(val_dataset_path, dtype="uint16", mode="r")
    optimizer = AdamW(
        model.parameters(), lr=min_lr, betas=betas, eps=eps, weight_decay=weight_decay
    )
    num_parameters = sum(parameter.numel() for parameter in model.parameters())
    flops_per_token = 6 * num_parameters + 12 * num_layers * context_length * d_model

    if resume_from > 0:
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_iter_{resume_from}.pth")
        if os.path.exists(checkpoint_path):
            resume_from = load_checkpoint(checkpoint_path, model, optimizer)
            if is_main_process:
                logging.info("Resumed training from %s", checkpoint_path)
        else:
            if is_main_process:
                logging.warning("Checkpoint not found at %s", checkpoint_path)
            resume_from = 0

    model.train()
    running_loss = 0.0
    start_time = time.time()
    best_val_loss = float("inf")
    best_val_iter = -1
    final_iteration = resume_from

    for iteration in range(resume_from, train_num_iterations):
        actual_step = iteration + 1
        final_iteration = actual_step
        x, y = get_batch(training_dataset, batch_size, context_length, device)
        optimizer.zero_grad()

        current_lr = cosine_annealing_lr_schedule(
            iteration, max_lr, min_lr, warm_up_steps, anneal_steps
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        loss = cross_entropy_loss(model(x), y)
        loss.backward()
        total_norm = gradient_clipping(model.parameters(), max_l2_norm)
        optimizer.step()
        running_loss += loss.item()

        if actual_step % log_interval == 0:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            step_time = (time.time() - start_time) / log_interval
            average_loss = running_loss / log_interval
            local_tokens_per_second = batch_size * context_length / step_time
            global_tokens_per_second = local_tokens_per_second * world_size
            achieved_tflops = flops_per_token * global_tokens_per_second / 1e12
            mfu = achieved_tflops / (world_size * gpu_peak_tflops)
            if is_main_process:
                logging.info(
                    "Iteration %d/%d | Loss: %.4f | LR: %.2e | Grad Norm: %.3f | "
                    "%.0f global tokens/sec | %.1f TFLOPs | MFU: %.2f%%",
                    actual_step, train_num_iterations, average_loss, current_lr,
                    total_norm, global_tokens_per_second, achieved_tflops, mfu * 100,
                )
            if writer:
                writer.add_scalar("Train/Loss", average_loss, actual_step)
                writer.add_scalar("Train/LearningRate", current_lr, actual_step)
                writer.add_scalar("Train/GradNorm", total_norm, actual_step)
                writer.add_scalar("Train/GlobalTokensPerSecond", global_tokens_per_second, actual_step)
                writer.add_scalar("Train/AchievedTFLOPs", achieved_tflops, actual_step)
                writer.add_scalar("Train/MFU", mfu, actual_step)
                writer.add_scalar("Train/PeakMem", torch.cuda.max_memory_allocated() / 1024**2, actual_step)
                torch.cuda.reset_peak_memory_stats()
            running_loss = 0.0
            start_time = time.time()

        if val_dataset_path and actual_step % val_interval == 0:
            results = evaluate(
                model, val_dataset_path, batch_size, context_length,
                device, val_num_iterations,
            )
            if world_size > 1:
                validation_loss = torch.tensor(
                    results["avg_val_loss"], device=device
                )
                dist.all_reduce(validation_loss, op=dist.ReduceOp.SUM)
                results["avg_val_loss"] = (
                    validation_loss / world_size
                ).item()
                results["perplexity"] = float(np.exp(results["avg_val_loss"]))
            if is_main_process:
                logging.info(
                    "Iteration %d | Validation Loss: %.4f | Perplexity: %.2f",
                    actual_step, results["avg_val_loss"], results["perplexity"],
                )
            if results["avg_val_loss"] < best_val_loss:
                best_val_loss = results["avg_val_loss"]
                best_val_iter = actual_step
                if is_main_process:
                    save_checkpoint(
                        model, optimizer, actual_step,
                        os.path.join(checkpoint_dir, "best_checkpoint.pth"),
                    )
            if writer:
                writer.add_scalar("Validation/Loss", results["avg_val_loss"], actual_step)
                writer.add_scalar("Validation/Perplexity", results["perplexity"], actual_step)
            if best_val_iter >= 0 and actual_step - best_val_iter >= 5 * val_interval:
                if is_main_process:
                    logging.info("Early stopping triggered")
                break

        if actual_step % save_interval == 0:
            filename = (
                f"final_checkpoint_iter_{actual_step}.pth"
                if actual_step == train_num_iterations
                else f"checkpoint_iter_{actual_step}.pth"
            )
            if is_main_process:
                save_checkpoint(model, optimizer, actual_step, os.path.join(checkpoint_dir, filename))
            if writer:
                writer.flush()

    if writer:
        writer.close()
    return {
        "status": "completed",
        "final_iteration": final_iteration,
        "final_checkpoint_path": os.path.join(
            checkpoint_dir, f"checkpoint_iter_{final_iteration}.pth"
        ),
    }
