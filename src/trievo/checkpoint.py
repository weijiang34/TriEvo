import os
import torch
from typing import BinaryIO, IO


def _get_raw_model(model: torch.nn.Module) -> torch.nn.Module:
    """剥掉 DDP 和 torch.compile 的包裹，拿到原始模型。

    兼容任意嵌套顺序，例如 compile(DDP(raw)) 或 DDP(compile(raw))。
    """
    changed = True
    while changed:
        changed = False
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
            changed = True
            continue
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
            changed = True
    return model


def _strip_prefixes(k: str) -> str:
    """去掉参数名里所有 _orig_mod. 和 module. 前缀，得到干净名。"""
    for p in ("_orig_mod.", "module."):
        while k.startswith(p):
            k = k[len(p):]
    # 处理 _orig_mod.module.xxx 这类交错情况
    k = k.replace("_orig_mod.", "").replace("module.", "")
    return k


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    raw_model = _get_raw_model(model)
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),  # 干净 key
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    checkpoint = torch.load(src)
    state = checkpoint["model_state_dict"]

    # 按"干净名"把 checkpoint 映射到当前模型的 key 上
    model_keys = set(model.state_dict().keys())
    clean_to_ckpt = {_strip_prefixes(k): k for k in state.keys()}

    new_state = {}
    missing = []
    for mk in model_keys:
        clean = _strip_prefixes(mk)
        if clean in clean_to_ckpt:
            new_state[mk] = state[clean_to_ckpt[clean]]
        else:
            missing.append(mk)
            new_state[mk] = model.state_dict()[mk]

    if missing:
        print(f"[load_checkpoint] {len(missing)} keys not found in checkpoint, "
              f"kept current init values. e.g. {missing[:3]}")

    model.load_state_dict(new_state)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]