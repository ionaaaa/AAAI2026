import numpy as np
import torch


def generate_block_mask(
    num_tokens: int,
    mask_ratio: float,
    min_block_tokens: int,
    max_block_tokens: int,
    mask_start: int = 0,
):
    """
    生成固定位置的 token mask。

    Returns:
        mask: [num_tokens]

    其中:
        1 表示该 token 被 mask
        0 表示该 token 保留

    说明:
        每次都会从 mask_start 开始，连续 mask target_masked 个 token。
        不再随机选择 mask 位置。
    """
    if num_tokens <= 0:
        return np.zeros((0,), dtype=np.float32)

    target_masked = int(round(num_tokens * mask_ratio))
    target_masked = max(0, min(num_tokens, target_masked))

    mask = np.zeros(num_tokens, dtype=np.float32)

    if target_masked == 0:
        return mask

    mask_start = max(0, min(mask_start, num_tokens - 1))
    mask_end = min(num_tokens, mask_start + target_masked)

    mask[mask_start:mask_end] = 1.0

    return mask


def apply_token_mask(tokens: torch.Tensor, token_mask: torch.Tensor):
    """
    在 token 维度上 mask。

    Args:
        tokens: [B, S, L]
        token_mask: [B, S]

    Returns:
        masked_tokens: [B, S, L]
    """
    masked_tokens = tokens.clone()

    masked_tokens[token_mask.bool()] = 0.0

    return masked_tokens