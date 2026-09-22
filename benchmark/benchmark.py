
from trievo.model import Transformer_LM
from trievo.nn_utils import cross_entropy_loss
from trievo.data import get_batch

from timeit import default_timer as timer
import torch
import torch.distributed as dist
import numpy as np
import gc
import os
import torch.cuda.nvtx as nvtx
from contextlib import nullcontext 
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP

from trievo.optimizer import AdamW


def _init_distributed():
    if "RANK" not in os.environ:
        return False, 0, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return True, rank, local_rank


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _start_profile(enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    torch.cuda.cudart().cudaProfilerStart()


def _stop_profile(enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    if dist.is_initialized():
        dist.barrier()

def init_model(d_model, d_ff, num_layers, num_heads, vocab_size=10000, context_length=512, checkpointing_blocksize=0, compiled=False, local_rank=0, distributed=False):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    model = Transformer_LM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
        checkpointing_blocksize=checkpointing_blocksize,
    )
    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if compiled:
        model = torch.compile(model)
    return model

def init_optimizer(model, **kwargs):
    optimizer = AdamW(
        model.parameters(),
        lr=3e-3,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8
    )
    return optimizer

def bench_forward_pass(model, dataset, warmup_steps, num_steps, same_data=False, mode='timer', mem_pkl=None, context_length=512, vocab_size=10000, batch_size=4):
    
    device = _model_device(model)

    is_timer = (mode == 'timer')
    is_nsys = (mode == 'nsys')
    is_mem = (mode == 'mem')

    # prepare data
    x, y = get_batch(dataset, batch_size, context_length, device)
    
    if is_timer:
        times = []

    model.eval()
    with torch.no_grad():
        with nvtx.range("Testing forward pass"):
            with nvtx.range("warmup"):
                # warmup
                for i in range(warmup_steps):
                    with nvtx.range(f"forward"):
                        logits = model(x)
            
                # timing
                if (is_timer or is_nsys) and torch.cuda.is_available():
                    torch.cuda.synchronize()

            _start_profile(is_nsys)
            
            if is_mem:
                # Start recording memory history.
                torch.cuda.memory._record_memory_history(max_entries=1000000)

            with nvtx.range("timing"):
                for i in range(num_steps):
                    if is_timer:
                        start = timer()

                    with nvtx.range(f"forward"):
                        logits = model(x)
                
                    if is_timer and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    if is_timer:
                        end = timer()
                        times.append(end - start)
            _stop_profile(is_nsys)
            if is_mem:
                # Save a pickle file to be loaded by PyTorch's online tool.
                torch.cuda.memory._dump_snapshot(mem_pkl) if mem_pkl else None
                # Stop recording history.
                torch.cuda.memory._record_memory_history(enabled=None)

    if is_timer:
        avg = sum(times) / num_steps
        std = (sum((t - avg) ** 2 for t in times) / num_steps) ** 0.5
        return avg, std
    else:
        return 0.0, 0.0

def bench_forward_backward_pass(model, dataset, warmup_steps, num_steps, same_data=False, mode='timer', mem_pkl=None, context_length=512, vocab_size=10000, batch_size=4):
    
    device = _model_device(model)

    is_timer = (mode == 'timer')
    is_nsys = (mode == 'nsys')
    is_mem = (mode == 'mem')

    # prepare data
    x, y = get_batch(dataset, batch_size, context_length, device)
    
    if is_timer:
        times = []

    model.train()

    with nvtx.range("Testing forward-backward pass"):
        
        with nvtx.range("warmup"):
            for i in range(warmup_steps):
                with nvtx.range(f"forward"):
                    logits = model(x)
                with nvtx.range(f"calc_loss"):
                    loss = cross_entropy_loss(logits, y)
                with nvtx.range(f"backward"):
                    loss.backward()

            if (is_timer or is_nsys) and torch.cuda.is_available():
                torch.cuda.synchronize()

        _start_profile(is_nsys)
        
        if is_mem:
            # Start recording memory history.
            torch.cuda.memory._record_memory_history(max_entries=1000000)
        with nvtx.range("timing"):
            for i in range(num_steps):
                if is_timer:
                    start = timer()

                with nvtx.range(f"forward"):
                    logits = model(x)
                with nvtx.range(f"calc_loss"):
                    loss = cross_entropy_loss(logits, y)
                with nvtx.range(f"backward"):
                    loss.backward()

                if is_timer and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    end = timer()
                    times.append(end - start)
            _stop_profile(is_nsys)
        if is_mem:
            # Save a pickle file to be loaded by PyTorch's online tool.
            torch.cuda.memory._dump_snapshot(mem_pkl) if mem_pkl else None
            # Stop recording history.
            torch.cuda.memory._record_memory_history(enabled=None)

    if is_timer:
        avg = sum(times) / num_steps
        std = (sum((t - avg) ** 2 for t in times) / num_steps) ** 0.5
        return avg, std
    else:
        return 0.0, 0.0

def bench_forward_backward_optimizer_pass(model, dataset, optimizer, warmup_steps, num_steps, same_data=False, mode='timer', mem_pkl=None, context_length=512, vocab_size=10000, batch_size=4, enable_autocast=True):
    
    device = _model_device(model)

    is_timer = (mode == 'timer')
    is_nsys = (mode == 'nsys')
    is_mem = (mode == 'mem')
    
    autocast_ctx = torch.autocast(device_type=device.type, dtype=torch.float16) if enable_autocast else nullcontext()

    # prepare data
    x, y = get_batch(dataset, batch_size, context_length, device)
    
    if is_timer:
        times = []
    
    model.train()

    with nvtx.range("Testing forward-backward-optimizer pass"):
        with nvtx.range("warmup"):
            for i in range(warmup_steps):
                with nvtx.range(f"forward"):
                    with autocast_ctx:
                        logits = model(x)
                with nvtx.range(f"calc_loss"):
                    loss = cross_entropy_loss(logits, y)
                with nvtx.range(f"backward"):
                    loss.backward()
                with nvtx.range(f"optimizer"):
                    optimizer.step()
                optimizer.zero_grad()

            if (is_timer or is_nsys) and torch.cuda.is_available():
                torch.cuda.synchronize()

        _start_profile(is_nsys)

        if is_mem:
            # Start recording memory history.
            torch.cuda.memory._record_memory_history(max_entries=1000000)

        with nvtx.range("timing"):
            for i in range(num_steps):
                if is_timer:
                    start = timer()

                with nvtx.range(f"forward"):
                    with autocast_ctx:
                        logits = model(x)

                with nvtx.range(f"calc_loss"):
                    loss = cross_entropy_loss(logits, y)

                with nvtx.range(f"backward"):
                    loss.backward()

                with nvtx.range(f"optimizer"):
                    optimizer.step()
                    
                optimizer.zero_grad()

                if is_timer and torch.cuda.is_available():
                    torch.cuda.synchronize()
                    end = timer()
                    times.append(end - start)
            _stop_profile(is_nsys)
        if is_mem:
            # Save a pickle file to be loaded by PyTorch's online tool.
            torch.cuda.memory._dump_snapshot(mem_pkl) if mem_pkl else None
            # Stop recording history.
            torch.cuda.memory._record_memory_history(enabled=None)

    if is_timer:
        avg = sum(times) / num_steps
        std = (sum((t - avg) ** 2 for t in times) / num_steps) ** 0.5
        return avg, std
    else:
        return 0.0, 0.0


def get_config_sets(model=True, optimizer=True):
    config_sets = {
        'tiny':     {'d_model': 512,    'd_ff': 1344,   'num_layers': 8, 'num_heads': 8, },
        'small':    {'d_model': 768,    'd_ff': 2048,   'num_layers': 12, 'num_heads': 12, },
        'medium':   {'d_model': 1024,   'd_ff': 4096,   'num_layers': 24, 'num_heads': 16, },
        'large':    {'d_model': 1280,   'd_ff': 5120,   'num_layers': 36, 'num_heads': 20, },
        'xl':       {'d_model': 2560,   'd_ff': 10240,  'num_layers': 32, 'num_heads': 32, },
        '10B':      { 'd_model': 4608,  'd_ff': 12288,  'num_layers': 50, 'num_heads': 36, },
    }
    return config_sets

class BenchmarkRunner:
    def __init__(
        self, config_sets, dataset_path,
        warmup=5, num_steps=20, 
        context_length=512, vocab_size=10000, batch_size=4, 
        mode='timer', mem_pkl_dir=None,
        models=None, steps=None,
        enable_autocast=True,
        checkpointing_blocksize=0,
        compiled=False,
        local_rank=0,
        distributed=False,
        rank=0,
    ):
        self.config_sets = config_sets
        self.warmup = warmup
        self.num_steps = num_steps
        self.dataset_path = dataset_path
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.mode = mode
        self.mem_pkl_dir = mem_pkl_dir
        self.models = models
        self.steps = steps
        self.enable_autocast = enable_autocast
        self.checkpointing_blocksize = checkpointing_blocksize
        self.compiled = compiled
        self.local_rank = local_rank
        self.distributed = distributed
        self.rank = rank

        self.results = {
            model_name: {step: None for step in steps}
            for model_name in config_sets
        }
        
        if self.mem_pkl_dir and not os.path.exists(self.mem_pkl_dir):
            os.makedirs(self.mem_pkl_dir)
    
    def run(self):
        for model_name in self.models:
            if self.rank == 0:
                print(f"\n{'='*50}")
                print(f"Benchmarking: {model_name}")
                print(f"{'='*50}")
            
            for process in self.steps:
                self._run_single_benchmark(model_name, process)
            
            # 每个模型测试完后，彻底清理
            self._clean_memory()
        
        return self.results
    
    def _run_single_benchmark(self, model_name, process):
        """运行单个基准测试，带 OOM 处理"""
        try:
            # 初始化模型
            model = self._init_model(model_name)
            optimizer = None
            dataset = np.memmap(self.dataset_path, dtype=np.uint16, mode='r')
            
            if process == 'forward_backward_optimizer':
                optimizer = self._init_optimizer(model, model_name)
            
            # 运行基准测试
            if process == 'forward':
                avg, std = bench_forward_pass(
                    model, dataset,
                    warmup_steps=self.warmup, 
                    num_steps=self.num_steps,
                    same_data=True,
                    context_length=self.context_length,
                    vocab_size=self.vocab_size,
                    batch_size=self.batch_size,
                    mode=self.mode,
                    mem_pkl=os.path.join(self.mem_pkl_dir, f"{model_name}_context_{self.context_length}_cpt_{self.checkpointing_blocksize}_forward.pkl") if self.mem_pkl_dir else None
                )
            elif process == 'forward_backward':
                avg, std = bench_forward_backward_pass(
                    model, dataset,
                    warmup_steps=self.warmup,
                    num_steps=self.num_steps,
                    same_data=True,
                    context_length=self.context_length,
                    vocab_size=self.vocab_size,
                    batch_size=self.batch_size,
                    mode=self.mode,
                    mem_pkl=os.path.join(self.mem_pkl_dir, f"{model_name}_context_{self.context_length}_cpt_{self.checkpointing_blocksize}_forward_backward.pkl") if self.mem_pkl_dir else None
                )
            else:
                avg, std = bench_forward_backward_optimizer_pass(
                    model, dataset,
                    optimizer,
                    warmup_steps=self.warmup,
                    num_steps=self.num_steps,
                    same_data=True,
                    context_length=self.context_length,
                    vocab_size=self.vocab_size,
                    batch_size=self.batch_size,
                    mode=self.mode,
                    mem_pkl=os.path.join(self.mem_pkl_dir, f"{model_name}_context_{self.context_length}_cpt_{self.checkpointing_blocksize}_autocast_{self.enable_autocast}_forward_backward_optimizer.pkl") if self.mem_pkl_dir else None,
                    enable_autocast=self.enable_autocast
                )
            
            # 记录结果
            if self.mode == 'timer':
                self.results[model_name][process] = f"{avg:.4f} ± {std:.4f}"
                print(f"  ✅ {process}: {avg:.4f} ± {std:.4f} s")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                self.results[model_name][process] = "OOM"
                print(f"  ❌ {process}: OUT OF MEMORY")
            else:
                self.results[model_name][process] = "ERROR"
                print(f"  ❌ {process}: {str(e)}")
            
            # 清理显存
            self._clean_memory()
            
        except Exception as e:
            self.results[model_name][process] = "ERROR"
            print(f"  ❌ {process}: {str(e)}")
            self._clean_memory()
        
        finally:
            # 强制垃圾回收
            gc.collect()
    
    def _init_model(self, model_name):
        """初始化模型，带错误处理"""
        try:
            return init_model(
                context_length=self.context_length,
                vocab_size=self.vocab_size,
                checkpointing_blocksize=self.checkpointing_blocksize,
                compiled=self.compiled,
                local_rank=self.local_rank,
                distributed=self.distributed,
                **self.config_sets[model_name],
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  ⚠️  Model initialization OOM for {model_name}")
                self._clean_memory()
            raise
    
    def _init_optimizer(self, model, model_name):
        """初始化优化器"""
        return init_optimizer(model, **self.config_sets[model_name])
    
    def _clean_memory(self):
        """清理 GPU 显存"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            # 可选：打印当前显存使用情况
            # allocated = torch.cuda.memory_allocated() / 1024**3
            # print(f"    Current GPU memory: {allocated:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description="Benchmark Transformer Models")
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset (numpy memmap file)')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup steps')
    parser.add_argument('--num_steps', type=int, default=10, help='Number of steps to time')
    parser.add_argument('--context_length', type=int, default=512, help='Context length for input sequences')
    parser.add_argument('--vocab_size', type=int, default=10000, help='Vocabulary size for input sequences')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for input sequences')
    parser.add_argument('--mode', type=str, choices=['timer', 'nsys', 'mem'], help='Mode of benchmarking: timer, nsys, or mem')
    parser.add_argument('--mem_pkl_dir', type=str, default=None, help='Directory to save memory profiling pickle files (only for mem mode)')
    parser.add_argument('--models', type=str, choices=['tiny', 'small', 'medium', 'large', 'xl', '10B'], nargs='+', default=['small', 'medium', 'large', 'xl', '10B'], help='Models to benchmark')
    parser.add_argument('--steps', type=str, choices=['forward', 'forward_backward', 'forward_backward_optimizer'], nargs='+', default=['forward', 'forward_backward', 'forward_backward_optimizer'], help='Steps to benchmark')
    parser.add_argument('--enable_autocast', action='store_true', help='Enable autocast for mixed precision training')
    parser.add_argument('--checkpointing_blocksize', type=int, default=0, help='Block size for gradient checkpointing (0 means no checkpointing)')
    parser.add_argument('--compiled', action='store_true', help='Use compiled model')
    args = parser.parse_args()

    distributed, rank, local_rank = _init_distributed()
    try:
        config_sets = get_config_sets()
        runner = BenchmarkRunner(
            config_sets,
            dataset_path=args.dataset_path,
            warmup=args.warmup, num_steps=args.num_steps,
            context_length=args.context_length, vocab_size=args.vocab_size, batch_size=args.batch_size,
            mode=args.mode, mem_pkl_dir=args.mem_pkl_dir,
            steps=args.steps, models=args.models,
            enable_autocast=args.enable_autocast,
            checkpointing_blocksize=args.checkpointing_blocksize,
            compiled=args.compiled,
            local_rank=local_rank,
            distributed=distributed,
            rank=rank,
        )
        results = runner.run()

        if rank == 0:
            print("\n" + "="*50)
            print("FINAL RESULTS")
            print("="*50)
            for model_name, timings in results.items():
                print(f"{model_name}: {timings}")
    
            if args.mode == 'timer':
                results_path = (
                    f"/srv/scratch/z3543429/LLM/TriEvo/benchmark/results/"
                    f"warmup_{runner.warmup}_steps_{runner.num_steps}_context_"
                    f"{runner.context_length}_vocab_{runner.vocab_size}_batch_"
                    f"{runner.batch_size}.txt"
                )
                with open(results_path, "w") as output:
                    for model_name, timings in results.items():
                        output.write(f"{model_name}: {timings}\n")
    finally:
        if distributed:
            dist.destroy_process_group()

if __name__ == "__main__":
    main()