from collections.abc import Callable, Iterable
from math import cos, pi

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params: Iterable[torch.nn.parameter.Parameter], lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self, closure: Callable | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                grad = parameter.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter.data)
                    state["exp_avg_sq"] = torch.zeros_like(parameter.data)
                beta1, beta2 = group["betas"]
                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                denominator = exp_avg_sq.sqrt().add_(group["eps"])
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * bias2 ** 0.5 / bias1
                parameter.data.addcdiv_(exp_avg, denominator, value=-step_size)
                parameter.data.add_(parameter.data, alpha=-group["lr"] * group["weight_decay"])
        return loss


def gradient_clipping(parameters, max_l2_norm: float) -> float:
    with torch.no_grad():
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        device = grads[0].device if grads else torch.device("cpu")
        total_norm = torch.zeros([], device=device)
        for grad in grads:
            total_norm.add_(grad.pow(2).sum())
        total_norm = total_norm.sqrt()
        if total_norm > max_l2_norm:
            coefficient = max_l2_norm / (total_norm + 1e-6)
            for grad in grads:
                grad.mul_(coefficient)
    return total_norm.item()


def cosine_annealing_lr_schedule(step, max_lr, min_lr, warm_up_steps, anneal_steps):
    if step < warm_up_steps:
        return step / warm_up_steps * max_lr
    if step < anneal_steps:
        return min_lr + 0.5 * (1 + cos((step - warm_up_steps) / (anneal_steps - warm_up_steps) * pi)) * (max_lr - min_lr)
    return min_lr
