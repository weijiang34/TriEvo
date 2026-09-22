import numpy as np
import numpy.typing as npt
import torch


def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device):
    start_positions = np.random.randint(0, len(dataset) - context_length, size=(batch_size, 1))
    offsets = np.arange(context_length)
    indices = start_positions + offsets
    x = torch.from_numpy(dataset[indices]).long()
    y = torch.from_numpy(dataset[indices + 1]).long()
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def data_loading(dataset, batch_size, context_length, device):
    return get_batch(dataset, batch_size, context_length, device)


__all__ = ["get_batch", "data_loading"]
