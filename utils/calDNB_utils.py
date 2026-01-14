# utils/calDNB_utils.py
import torch


# =========================
# Masked statistics helpers
# =========================

def masked_mean(X, mask, dim, keepdim=False, eps=1e-8):
    """
    X:    [..., T, ...]
    mask: same shape as X along dim (broadcastable), boolean
    """
    m = mask.to(X.dtype)
    num = (X * m).sum(dim=dim, keepdim=keepdim)
    den = m.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


def masked_var(X, mask, dim, keepdim=False, eps=1e-8):
    mu = masked_mean(X, mask, dim=dim, keepdim=True, eps=eps)
    m = mask.to(X.dtype)
    num = ((X - mu) ** 2 * m).sum(dim=dim, keepdim=keepdim)
    den = m.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return num / den


# =========================
# DNB metrics (GPU batch)
# =========================

def dnb_sd_batch(emb, mask, eps=1e-8):
    """
    DNB-SD per sample

    emb:  [B, T, D]
    mask: [B, T] boolean
    return: [B]
    """
    m = mask.unsqueeze(-1)                # [B,T,1]
    var_t = masked_var(emb, m, dim=2, eps=eps)  # [B,D]
    std_t = torch.sqrt(var_t.clamp_min(eps))
    return std_t.mean(dim=1),std_t.max(dim=1).values          # [B]


def dnb_pcc_batch(emb, mask, eps=1e-6):
    """
    DNB-PCC per sample (mean absolute correlation across dimensions)

    emb:  [B, T, D]
    mask: [B, T] boolean
    return: [B]
    """
    B, T, D = emb.shape
    m = mask.unsqueeze(-1)                # [B,T,1]

    mu = masked_mean(emb, m, dim=1, keepdim=True)
    var = masked_var(emb, m, dim=1, keepdim=True)
    std = torch.sqrt(var.clamp_min(eps))

    X = (emb - mu) / std
    X = X * m.to(X.dtype)

    n = m.sum(dim=1).clamp_min(2.0).view(B, 1, 1)

    C = torch.bmm(X.transpose(1, 2), X) / (n - 1.0)
    C = C.clamp(min=-1.0, max=1.0)

    idx = torch.triu_indices(D, D, offset=1, device=emb.device)
    return C[:, idx[0], idx[1]].abs().mean(dim=1), C[:, idx[0], idx[1]].abs().max(dim=1).values


# ==========================================
# Lyapunov proxy via Jacobian / local expansion
# ==========================================

def lyap_proxy_jacobian_batch(emb, mask, ridge=1e-2, eps=1e-8):
    """
    GPU-friendly Lyapunov proxy:
        z_{t+1} ≈ A z_t
        proxy = log(sigma_max(A))

    emb:  [B, T, D]
    mask: [B, T]
    return: [B]
    """
    B, T, D = emb.shape
    if T < 2:
        return torch.full((B,), float("nan"), device=emb.device)

    z = emb[:, :-1, :]
    y = emb[:, 1:, :]
    m = (mask[:, :-1] & mask[:, 1:]).unsqueeze(-1)

    n = m.sum(dim=1).squeeze(-1)
    valid = n >= (D + 2)

    Z = z * m.to(z.dtype)
    Y = y * m.to(y.dtype)

    Szz = torch.bmm(Z.transpose(1, 2), Z)
    Szy = torch.bmm(Z.transpose(1, 2), Y)

    I = torch.eye(D, device=emb.device).unsqueeze(0)
    A = torch.linalg.solve(Szz + ridge * I, Szy)

    smax = torch.linalg.svdvals(A).max(dim=1).values.clamp_min(eps)
    out = torch.log(smax)

    return torch.where(valid, out, torch.full_like(out, float("nan")))
