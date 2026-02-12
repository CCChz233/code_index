from typing import Tuple

import torch


POOLING_CHOICES: Tuple[str, ...] = ("cls", "first_non_pad", "last_token", "mean")


def extract_last_hidden_state(outputs) -> torch.Tensor:
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, torch.Tensor):
        return outputs
    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def pool_hidden_states(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    pooling: str = "first_non_pad",
) -> torch.Tensor:
    if pooling not in POOLING_CHOICES:
        raise ValueError(f"Unsupported pooling: {pooling}. Choices: {POOLING_CHOICES}")
    if attention_mask.dim() == 1:
        attention_mask = attention_mask.unsqueeze(0)

    if pooling == "cls":
        return last_hidden_state[:, 0]

    mask = attention_mask.to(last_hidden_state.device)

    if pooling == "mean":
        weights = mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * weights).sum(dim=1)
        denom = weights.sum(dim=1).clamp(min=1e-6)
        return summed / denom

    if pooling == "first_non_pad":
        token_indices = (mask > 0).float().argmax(dim=1).long()
    else:  # last_token
        token_indices = mask.sum(dim=1).long().clamp(min=1) - 1

    batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
    return last_hidden_state[batch_indices, token_indices]
