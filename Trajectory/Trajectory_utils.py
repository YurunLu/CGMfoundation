# Trajectory/Trajectory_utils.py
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from SampleRepresentation.sample_representation_model import CGMSampleRepresentationModel


# -------------------------
# Patch utility
# -------------------------
def series_to_patches(x: torch.Tensor, patch_size: int = 36, stride: int = 6):
    if x.dim() != 1:
        raise ValueError(f"x must be 1D [T], got {x.shape}")
    if x.numel() < patch_size:
        pad = patch_size - x.numel()
        x = torch.cat([x, torch.full((pad,), float("nan"))], dim=0)
    patches = x.unfold(0, patch_size, stride).contiguous()  # [N,P]
    mask = ~torch.isnan(patches).any(dim=1)
    patches = torch.nan_to_num(patches, nan=0.0)
    return patches, mask


# -------------------------
# CSV loading
# -------------------------
def _read_cgm_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f'"id" column not found in {path}')
    if "starttime" not in df.columns:
        raise ValueError(f'"starttime" column not found in {path}')
    return df


def _read_info_csv(path: str) -> pd.DataFrame:
    info = pd.read_csv(path)
    required = {"id", "Userid", "Age", "Type"}
    missing = required - set(info.columns)
    if missing:
        raise ValueError(f"Info file missing columns: {missing}")
    return info


def load_visits_table(
    cgm_paths: Dict[str, str],
    info_path: str,
    allowed_types: Tuple[str, ...] = ("ND", "HRT2D", "T2D"),
) -> pd.DataFrame:
    info = _read_info_csv(info_path).copy()
    info = info.rename(columns={"id": "device_id", "Userid": "user_id", "Age": "age", "Type": "type"})
    info = info[info["type"].isin(list(allowed_types))].copy()

    all_rows = []
    for typ, p in cgm_paths.items():
        df = _read_cgm_csv(p).copy()
        df = df.rename(columns={"id": "device_id"})
        df["type_from_file"] = typ
        df["order_time"] = pd.to_datetime(df["starttime"], errors="coerce")

        m = df.merge(info, on="device_id", how="left", suffixes=("", "_info"))

        exclude = {"device_id", "type_from_file", "user_id", "age", "type", "order_time", "starttime"}
        glucose_cols = [c for c in m.columns if c not in exclude]
        g = m[glucose_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        for i in range(len(m)):
            tval = m.iloc[i]["type"] if pd.notna(m.iloc[i]["type"]) else m.iloc[i]["type_from_file"]
            all_rows.append({
                "device_id": m.iloc[i]["device_id"],
                "user_id": m.iloc[i]["user_id"],
                "age": m.iloc[i]["age"],
                "type": tval,
                "order_time": m.iloc[i]["order_time"],
                "glucose": g[i],
            })

    visits = pd.DataFrame(all_rows)
    visits = visits.dropna(subset=["user_id", "order_time"]).copy()
    return visits


LABEL_MAP = {"ND": 0, "HRT2D": 1, "PreD": 1, "T2D": 2}


def build_transitions(
    visits: pd.DataFrame,
    min_gap_months: float = 4.0,
    max_pairs_per_user: Optional[int] = 10,
) -> pd.DataFrame:
    rows = []
    for user_id, grp in visits.groupby("user_id"):
        grp = grp.sort_values("order_time").reset_index(drop=True)
        if len(grp) < 2:
            continue

        times = pd.to_datetime(grp["order_time"]).values
        candidates = []

        for i in range(len(grp) - 1):
            for j in range(i + 1, len(grp)):
                dt_days = (times[j] - times[i]) / np.timedelta64(1, "D")
                dt_months = dt_days / 30.4375
                if dt_months < min_gap_months:
                    continue

                a = grp.iloc[i]
                b = grp.iloc[j]
                candidates.append({
                    "user_id": user_id,
                    "age_a": a["age"],
                    "type": a["type"],
                    "glucose_a": a["glucose"],
                    "glucose_b": b["glucose"],
                    "dt_months": float(dt_months),
                })

        if not candidates:
            continue

        if max_pairs_per_user is not None and max_pairs_per_user > 0 and len(candidates) > max_pairs_per_user:
            idx = np.random.choice(len(candidates), size=max_pairs_per_user, replace=False)
            candidates = [candidates[k] for k in idx]

        rows.extend(candidates)

    return pd.DataFrame(rows)


class CGMTrajectoryDataset(Dataset):
    def __init__(
        self,
        transitions: pd.DataFrame,
        patch_size: int = 36,
        stride: int = 6,
        max_len: int = 3456,
        age_mean: float = 0.0,
        age_std: float = 1.0,
    ):
        self.df = transitions.reset_index(drop=True)
        self.patch_size = patch_size
        self.stride = stride
        self.max_len = max_len
        self.age_mean = float(age_mean)
        self.age_std = float(age_std) if float(age_std) > 1e-8 else 1.0

    def __len__(self):
        return len(self.df)

    def _to_tensor_series(self, arr: np.ndarray) -> torch.Tensor:
        x = torch.tensor(arr, dtype=torch.float32)
        if x.numel() > self.max_len:
            x = x[: self.max_len]
        elif x.numel() < self.max_len:
            pad = self.max_len - x.numel()
            x = torch.cat([x, torch.full((pad,), float("nan"))], dim=0)
        return x

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        x1 = self._to_tensor_series(r["glucose_a"])
        x2 = self._to_tensor_series(r["glucose_b"])
        p1, m1 = series_to_patches(x1, patch_size=self.patch_size, stride=self.stride)
        p2, m2 = series_to_patches(x2, patch_size=self.patch_size, stride=self.stride)

        age_a = r["age_a"]
        if pd.isna(age_a):
            age_missing = 1
            age_norm = 0.0
        else:
            age_missing = 0
            age_norm = (float(age_a) - self.age_mean) / self.age_std

        typ = str(r["type"])
        label_id = LABEL_MAP.get(typ, 3)

        return {
            "patches_a": p1,
            "mask_a": m1,
            "patches_b": p2,
            "mask_b": m2,
            "age_a": torch.tensor([age_norm], dtype=torch.float32),
            "age_missing": torch.tensor([bool(age_missing)]),
            "label_id": torch.tensor(label_id, dtype=torch.long),
            "dt": torch.tensor([float(r["dt_months"])], dtype=torch.float32),
        }


def pad_collate(batch: List[Dict]):
    def pad_stack(kp: str, km: str):
        Ns = [b[kp].shape[0] for b in batch]
        maxN = max(Ns)
        P = batch[0][kp].shape[1]
        out = torch.zeros((len(batch), maxN, P), dtype=torch.float32)
        out_mask = torch.zeros((len(batch), maxN), dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b[kp].shape[0]
            out[i, :n] = b[kp]
            out_mask[i, :n] = b[km]
        return out, out_mask

    pa, ma = pad_stack("patches_a", "mask_a")
    pb, mb = pad_stack("patches_b", "mask_b")

    age = torch.stack([b["age_a"] for b in batch], dim=0)                  # [B,1]
    age_miss = torch.stack([b["age_missing"] for b in batch], dim=0).bool()# [B,1]
    y = torch.stack([b["label_id"] for b in batch], dim=0)                 # [B]
    dt = torch.stack([b["dt"] for b in batch], dim=0)                      # [B,1]

    return {"patches_a": pa, "mask_a": ma, "patches_b": pb, "mask_b": mb,
            "age_a": age, "age_missing": age_miss, "label_id": y, "dt": dt}


class VisitEmbeddingExtractor(nn.Module):
    """
    Frozen sample representation model → visit embedding.
    IMPORTANT: do NOT normalize embedding output for dynamics.
    """
    def __init__(
        self,
        base_ckpt_path: str,
        sample_rep_ckpt_path: str,
        patch_size: int = 36,
        stride: int = 6,
        embed_dim: int = 128,
        max_seq_len: int = 3456,
        proj_dim: int = 128,
        device: str = "cpu",
        use_projector: bool = True,
    ):
        super().__init__()
        self.rep = CGMSampleRepresentationModel(
            checkpoint_path=base_ckpt_path,
            patch_size=patch_size,
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
            proj_dim=proj_dim,
            finetune_encoder=False,
        )

        state = torch.load(sample_rep_ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        self.rep.load_state_dict(state, strict=False)
        self.rep.to(device)
        for p in self.rep.parameters():
            p.requires_grad = False

        self.device = device
        self.stride = stride
        self.use_projector = use_projector

    @torch.no_grad()
    def encode(self, patches: torch.Tensor, patch_mask: torch.Tensor) -> torch.Tensor:
        x = self.rep.encoder(patches, stride=self.stride, return_embedding=True)  # [B,N,E]
        z = self.rep.aggregator(x, patch_mask)                                   # [B,E]
        if self.use_projector:
            return self.rep.projector(z)  # NOT normalized
        return z


# ============================================================
# Baselines (vector MSE / cosine / speed MSE)
# ============================================================

@torch.no_grad()
def compute_baselines(loader, extractor, device, max_batches: int = 50, exclude_unknown: bool = True):
    v_list, y_list = [], []
    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        for k in batch:
            batch[k] = batch[k].to(device)

        z = extractor.encode(batch["patches_a"], batch["mask_a"])
        z2 = extractor.encode(batch["patches_b"], batch["mask_b"])
        dt = torch.clamp(batch["dt"], min=1e-3)
        v_obs = (z2 - z) / dt

        v_list.append(v_obs.detach().cpu())
        y_list.append(batch["label_id"].detach().cpu())

    v = torch.cat(v_list, dim=0)                  # [N,d]
    y = torch.cat(y_list, dim=0).long().view(-1)  # [N]

    # counts
    uniq, cnt = torch.unique(y, return_counts=True)
    label_counts = {int(a): int(b) for a, b in zip(uniq, cnt)}

    if exclude_unknown:
        known = (y <= 2)
        v = v[known]
        y = y[known]

    # zero
    v0 = torch.zeros_like(v)
    zero_vec_mse = F.mse_loss(v0, v).item()
    zero_speed_mse = F.mse_loss(v0.norm(dim=1), v.norm(dim=1)).item()
    zero_cos = 0.0

    # label-mean
    v_mean = torch.zeros_like(v)
    for lab in torch.unique(y):
        m = (y == lab)
        v_mean[m] = v[m].mean(dim=0, keepdim=True)

    label_vec_mse = F.mse_loss(v_mean, v).item()
    label_speed_mse = F.mse_loss(v_mean.norm(dim=1), v.norm(dim=1)).item()

    vp = v_mean / (v_mean.norm(dim=1, keepdim=True) + 1e-8)
    vo = v / (v.norm(dim=1, keepdim=True) + 1e-8)
    label_cos = (vp * vo).sum(dim=1).mean().item()

    return {
        "label_counts": label_counts,
        "zero": {"vec_mse": zero_vec_mse, "cos": zero_cos, "speed_mse": zero_speed_mse},
        "label_mean": {"vec_mse": label_vec_mse, "cos": label_cos, "speed_mse": label_speed_mse},
        "obs": {"speed_mean": v.norm(dim=1).mean().item(), "speed_std": v.norm(dim=1).std().item()},
    }


def compute_losses_twohead(
    model_out: Dict[str, torch.Tensor],
    z: torch.Tensor,
    z_next: torch.Tensor,
    dt: torch.Tensor,
    weights: Dict[str, float],
):
    """
    Two-head losses:
      - direction loss: ((1 - cos(d_pred, v_obs_dir))^2).mean()
      - speed loss: Huber(s_pred, ||v_obs||)
      - optional vec MSE of v_pred vs v_obs (usually small weight or 0)
      - progress ordering: ND < PreD < T2D on tau = w·z
      - age anchoring: tau ~ age_value (small weight; only observed age)

    weights:
      w_dir, w_speed, w_vec, w_prog, w_age_tau
    """
    dt = torch.clamp(dt, min=1e-3)
    v_obs = (z_next - z) / dt                           # [B,d]
    speed_obs = v_obs.norm(dim=1)                       # [B]
    v_obs_dir = v_obs / (speed_obs.unsqueeze(1) + 1e-8) # unit

    d_pred = model_out["d_pred"]                         # unit
    s_pred = model_out["s_pred"]                         # [B]
    v_pred = model_out["v_pred"]                         # [B,d]

    # direction cosine
    cos = (d_pred * v_obs_dir).sum(dim=1).clamp(-1, 1)   # [B]
    l_dir = ((1.0 - cos) ** 2).mean()

    # speed loss (robust)
    # Huber is less overfit-prone than MSE for heterogeneous speeds
    l_speed = F.smooth_l1_loss(s_pred, speed_obs)

    # (optional) vector component loss
    l_vec = F.mse_loss(v_pred, v_obs)

    loss = (
        weights.get("w_dir", 1.0) * l_dir +
        weights.get("w_speed", 0.5) * l_speed +
        weights.get("w_vec", 0.0) * l_vec
    )

    # metrics
    speed_mse = F.mse_loss(s_pred, speed_obs)

    return {
        "loss": loss,
        "l_dir": l_dir.detach(),
        "l_speed_huber": l_speed.detach(),
        "l_speed_mse": speed_mse.detach(),
        "l_vec_mse": l_vec.detach(),
        "cos_mean": cos.mean().detach(),
        "speed_pred_mean": s_pred.mean().detach(),
        "speed_obs_mean": speed_obs.mean().detach(),
    }
