import math

import numpy as np
import torch

from trievo.data import get_batch
from trievo.nn_utils import cross_entropy_loss


def evaluate(
    model: torch.nn.Module,
    dataset_path,
    batch_size: int,
    context_length: int,
    device,
    num_iterations: int = 200,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    dataset = np.memmap(dataset_path, dtype="uint16", mode="r")
    total_loss = 0.0

    with torch.no_grad():
        for _ in range(num_iterations):
            x, y = get_batch(dataset, batch_size, context_length, device)
            total_loss += cross_entropy_loss(model(x), y).item()

    if was_training:
        model.train()
    average_loss = total_loss / num_iterations
    return {
        "avg_val_loss": average_loss,
        "perplexity": math.exp(average_loss) if average_loss < 20 else float("inf"),
    }
