from dataclasses import dataclass
from time import perf_counter
from typing import Sequence

import torch

from trievo.checkpoint import load_checkpoint
from trievo.tokenizer import KmerTokenizer


@dataclass
class InferenceState:
    """Autoregressive state for the correctness-first inference baseline."""

    token_ids: torch.Tensor
    next_token_logits: torch.Tensor


@dataclass
class GenerationResult:
    token_ids: torch.Tensor
    prefill_seconds: float
    decode_seconds: float

    @property
    def generated_tokens(self) -> int:
        return self.token_ids.shape[1]


class TriEvoEngine:
    """Single-GPU inference baseline without a KV cache.

    ``decode`` intentionally recomputes the active context. It provides a
    correctness and latency baseline before introducing paged KV caching.
    """

    def __init__(self, model: torch.nn.Module, tokenizer: KmerTokenizer):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        self.context_length = model.context_length

    def load_checkpoint(self, checkpoint_path: str) -> int:
        """Load a training checkpoint into this engine's model."""
        iteration = load_checkpoint(checkpoint_path, self.model)
        self.model.eval()
        return iteration

    def prefill(self, token_ids: Sequence[int] | torch.Tensor) -> InferenceState:
        """Run the prompt and return logits for sampling its next token."""
        prompt = self._prepare_token_ids(token_ids)
        _, logits = self._run_model(prompt)
        return InferenceState(token_ids=prompt, next_token_logits=logits[:, -1, :])

    def decode(self, state: InferenceState, token_id: int) -> InferenceState:
        """Append one token and recompute the active context without a KV cache."""
        next_token = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
        token_ids = torch.cat((state.token_ids, next_token), dim=1)
        if token_ids.shape[1] > self.context_length:
            token_ids = token_ids[:, -self.context_length:]
        _, logits = self._run_model(token_ids)
        return InferenceState(token_ids=token_ids, next_token_logits=logits[:, -1, :])

    def generate(
        self,
        prompt: str | Sequence[int] | torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ) -> GenerationResult:
        """Generate token IDs from a DNA prompt using greedy or top-p sampling."""
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        prefill_start = perf_counter()
        state = self.prefill(token_ids)
        self._synchronize()
        prefill_seconds = perf_counter() - prefill_start

        generated = []
        decode_start = perf_counter()
        for _ in range(max_new_tokens):
            token_id = self._sample(state.next_token_logits, temperature, top_p)
            generated.append(token_id)
            state = self.decode(state, token_id)
        self._synchronize()
        decode_seconds = perf_counter() - decode_start

        output = torch.tensor([generated], device=self.device, dtype=torch.long)
        return GenerationResult(
            token_ids=output,
            prefill_seconds=prefill_seconds,
            decode_seconds=decode_seconds,
        )

    def _prepare_token_ids(
        self, token_ids: Sequence[int] | torch.Tensor
    ) -> torch.Tensor:
        token_tensor = torch.as_tensor(token_ids, device=self.device, dtype=torch.long)
        if token_tensor.ndim == 1:
            token_tensor = token_tensor.unsqueeze(0)
        if token_tensor.ndim != 2 or token_tensor.shape[0] != 1:
            raise ValueError("token_ids must have shape [sequence] or [1, sequence]")
        if not 0 < token_tensor.shape[1] <= self.context_length:
            raise ValueError(
                f"prompt length must be in [1, {self.context_length}], "
                f"got {token_tensor.shape[1]}"
            )
        return token_tensor

    def _run_model(self, token_ids: torch.Tensor) -> tuple[float, torch.Tensor]:
        self._synchronize()
        start = perf_counter()
        with torch.inference_mode():
            logits = self.model(token_ids)
        self._synchronize()
        return perf_counter() - start, logits

    def _sample(
        self, logits: torch.Tensor, temperature: float, top_p: float
    ) -> int:
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if temperature == 0:
            return int(torch.argmax(logits, dim=-1).item())

        sorted_logits, sorted_ids = torch.sort(logits / temperature, descending=True)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        sorted_logits = sorted_logits.masked_fill(cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p, -torch.inf)
        sampled_index = torch.multinomial(torch.softmax(sorted_logits, dim=-1), 1)
        return int(sorted_ids.gather(-1, sampled_index).item())

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
