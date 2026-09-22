import os
import random

import hydra
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from accounting import calc_params, dataset_token_count
from evaluation import evaluate
from predict import predict_fasta
from training import train
from trievo.model import Transformer_LM
from trievo.tokenizer import build_tokenizer
from trievo.checkpoint import load_checkpoint


def _resolve_config(config: DictConfig) -> DictConfig:
    """Resolve derived values and runtime paths before constructing objects."""
    tokenizer = build_tokenizer(config.tokenizer)
    config.model.vocab_size = len(tokenizer)

    if config.data.add_reverse_complement:
        for path_key in ["train_path", "val_path", "test_path"]:
            path_str = str(config.data[path_key])
            # 在文件名或目录前插入 _rc，例如把 /train_vir.bin 变成 _rc/train_vir.bin
            if "train_vir.bin" in path_str:
                config.data.train_path = path_str.replace("/train_vir.bin", "_rc/train_vir.bin")
            elif "val_vir.bin" in path_str:
                config.data.val_path = path_str.replace("/val_vir.bin", "_rc/val_vir.bin")
            elif "test_vir.bin" in path_str:
                config.data.test_path = path_str.replace("/test_vir.bin", "_rc/test_vir.bin")
    config.data.train_path = to_absolute_path(config.data.train_path)
    config.data.val_path = to_absolute_path(config.data.val_path)
    config.data.test_path = to_absolute_path(config.data.test_path)

    config.experiment.output_dir = to_absolute_path(config.experiment.output_dir)
    config.experiment.checkpoint_dir = os.path.join(
        config.experiment.output_dir, "checkpoints"
    )
    config.experiment.log_dir = os.path.join(config.experiment.output_dir, "logs")
    config.experiment.tb_log_dir = os.path.join(
        config.experiment.output_dir, "tb_logs"
    )

    if config.training.min_lr is None:
        config.training.min_lr = config.training.max_lr * 0.01
    if config.training.log_interval is None:
        config.training.log_interval = max(1, config.training.train_num_iterations // 1000)
    if config.training.val_interval is None:
        config.training.val_interval = max(1, config.training.train_num_iterations // 10)
    if config.training.save_interval is None:
        config.training.save_interval = max(1, config.training.train_num_iterations // 10)
    if config.training.val_num_iterations is None:
        config.training.val_num_iterations = max(1, config.training.train_num_iterations // 20)
    if config.training.warm_up_steps is None:
        config.training.warm_up_steps = max(1, config.training.train_num_iterations // 10)
    if config.training.anneal_steps is None:
        config.training.anneal_steps = config.training.train_num_iterations

    if config.model.d_model % config.model.num_heads != 0:
        raise ValueError("model.d_model must be divisible by model.num_heads")
    if config.prediction.max_length > config.data.context_length:
        config.prediction.max_length = config.data.context_length

    return config


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _init_distributed():
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not distributed:
        return False, 0, 0, 1

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, local_rank, world_size


def build_model(config: DictConfig, tokenizer, device, dtype) -> Transformer_LM:
    return Transformer_LM(
        vocab_size=len(tokenizer),
        context_length=config.data.context_length,
        d_model=config.model.d_model,
        num_heads=config.model.num_heads,
        d_ff=config.model.d_ff,
        num_layers=config.model.num_layers,
        theta=config.model.theta,
        device=device,
        dtype=dtype,
        checkpointing_blocksize=config.model.checkpointing_blocksize,
    )


def _save_resolved_config(config: DictConfig) -> None:
    output_dir = config.experiment.output_dir
    os.makedirs(config.experiment.checkpoint_dir, exist_ok=True)
    os.makedirs(config.experiment.log_dir, exist_ok=True)
    os.makedirs(config.experiment.tb_log_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(output_dir, "resolved_config.yaml"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    distributed, rank, local_rank, world_size = _init_distributed()
    config = _resolve_config(config)
    _set_seed(config.runtime.seed + rank)

    runtime_device = torch.device(
        f"cuda:{local_rank}" if distributed else config.runtime.device
    )
    runtime_dtype = getattr(torch, config.runtime.dtype)
    tokenizer = build_tokenizer(config.tokenizer)
    model = build_model(config, tokenizer, runtime_device, runtime_dtype)
    model.to(runtime_device)
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if config.runtime.compile:
        model = torch.compile(model)

    if rank == 0:
        _save_resolved_config(config)
    if distributed:
        dist.barrier()

    if config.experiment.do_accounting and rank == 0:
        result = calc_params(
            batch_size=config.data.batch_size,
            vocab_size=len(tokenizer),
            context_length=config.data.context_length,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
            num_heads=config.model.num_heads,
            d_ff=config.model.d_ff,
        )
        print(f"Accounting results: {result}")
        if os.path.exists(config.data.train_path):
            print(f"Training tokens: {dataset_token_count(config.data.train_path):,}")
        if os.path.exists(config.data.val_path):
            print(f"Validation tokens: {dataset_token_count(config.data.val_path):,}")

    if config.experiment.do_train:
        train(
            model=model,
            train_dataset_path=config.data.train_path,
            val_dataset_path=config.data.val_path,
            max_lr=config.training.max_lr,
            min_lr=config.training.min_lr,
            betas=(config.training.beta1, config.training.beta2),
            eps=config.training.eps,
            weight_decay=config.training.weight_decay,
            train_num_iterations=config.training.train_num_iterations,
            batch_size=config.data.batch_size,
            context_length=config.data.context_length,
            device=runtime_device,
            warm_up_steps=config.training.warm_up_steps,
            anneal_steps=config.training.anneal_steps,
            max_l2_norm=config.training.max_l2_norm,
            checkpoint_dir=config.experiment.checkpoint_dir,
            save_interval=config.training.save_interval,
            log_interval=config.training.log_interval,
            tb_log_dir=config.experiment.tb_log_dir,
            val_num_iterations=config.training.val_num_iterations,
            val_interval=config.training.val_interval,
            resume_from=config.training.resume_from,
            rank=rank,
            world_size=world_size,
            num_layers=config.model.num_layers,
            d_model=config.model.d_model,
        )

    if config.experiment.do_eval:
        evaluation_result = evaluate(
            model=model,
            dataset_path=config.data.val_path,
            batch_size=config.data.batch_size * 4,
            context_length=config.data.context_length,
            device=runtime_device,
            num_iterations=config.training.eval_num_iterations,
        )
        if rank == 0:
            print(evaluation_result)

    if config.experiment.do_predict and rank == 0:
        checkpoint_path = os.path.join(config.experiment.checkpoint_dir, f"best_checkpoint.pth")
        print(f"Loading checkpoint from {checkpoint_path} for prediction...")
        resume_from = load_checkpoint(checkpoint_path, model)
                
        input_path = config.prediction.input_fasta_path
        output_dir = config.prediction.output_dir or config.experiment.output_dir
        predict_fasta(
            model=model,
            tokenizer=tokenizer,
            input_fasta_path=to_absolute_path(input_path),
            output_dir=to_absolute_path(output_dir),
            max_length=config.prediction.max_length,
            device=runtime_device,
        )

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
