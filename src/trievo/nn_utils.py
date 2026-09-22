import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor, agg_mean: bool = True) -> torch.Tensor:
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    stabilized_logits = logits - max_logits
    log_probs = stabilized_logits - torch.logsumexp(
        stabilized_logits, dim=-1, keepdim=True
    )
    target_log_probs = log_probs.gather(
        dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)
    if agg_mean:
        return -target_log_probs.mean()
    else:
        return -target_log_probs