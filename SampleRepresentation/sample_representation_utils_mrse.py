# SampleRepresentation/sample_representation_utils_mrse.py
import torch
import torch.nn.functional as F


def split_mask_nonempty(mask: torch.Tensor, p: float = 0.5, max_tries: int = 10):
    """
    Split a boolean mask [B,N] into two disjoint submasks m1/m2 such that
    each has at least 1 True per row where possible.

    If a row has <=1 valid patch, we fall back to m1=mask, m2=mask (no split).
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()

    B, N = mask.shape
    valid_counts = mask.sum(dim=1)  # [B]

    # default fallback
    m1 = mask.clone()
    m2 = mask.clone()

    # rows eligible for splitting
    eligible = valid_counts >= 2
    if not eligible.any():
        return m1, m2

    # attempt random splits for eligible rows
    for _ in range(max_tries):
        rand = torch.rand_like(mask.float())
        cand1 = mask & (rand < p)
        cand2 = mask & (rand >= p)

        ok = eligible & (cand1.sum(dim=1) >= 1) & (cand2.sum(dim=1) >= 1)
        if ok.any():
            m1[ok] = cand1[ok]
            m2[ok] = cand2[ok]

        # stop early if all eligible rows are ok
        done = (~eligible) | ok
        if done.all():
            break

    return m1, m2


def residual_consistency_loss(r1: torch.Tensor, r2: torch.Tensor, mode: str = "mse"):
    """
    r1, r2: [B,D]
    mode: "mse" or "cosine"
    """
    if mode == "mse":
        return F.mse_loss(r1, r2)
    if mode == "cosine":
        # 1 - cosine similarity
        r1n = F.normalize(r1, dim=-1)
        r2n = F.normalize(r2, dim=-1)
        return (1.0 - (r1n * r2n).sum(dim=-1)).mean()
    raise ValueError(f"Unknown loss mode: {mode}")

def variance_regularization(z, eps=1e-4):
    # z: [B, D]
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(F.relu(1.0 - std))