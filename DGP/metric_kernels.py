"""Numerical kernels copied verbatim from the local current DGP workflow.

See docs/TRAJECTORY_DGP_SOURCES.json for source locations and SHA256 hashes.
Only imports and module constants are isolated from cohort-specific runners.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

PATCH_MINUTES = 15.0
PATCHES_PER_HOUR = 4.0
FIRST_PATCH_MINUTE = 720
NIGHT_END_MINUTE = 360
PULSE_SEPARATION_PATCHES = 8
PULSE_PEAK_WINDOW_PATCHES = 8
PLE_HORIZONS = (4, 8)
EPS = 1e-8


def patch_risk(raw_patch: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(raw_patch, dtype=np.float32)
    finite = np.isfinite(raw)
    count = np.maximum(finite.sum(axis=-1), 1)
    clipped = np.where(finite, np.clip(raw, 1e-3, None), np.nan)
    mg = clipped * 18.0
    f = 1.509 * (np.log(np.maximum(mg, 1.0)) ** 1.084 - 5.381)
    risk = 10.0 * f**2
    hbgi = np.nansum(np.where(f > 0, risk, 0.0), axis=-1) / count
    lbgi = np.nansum(np.where(f < 0, risk, 0.0), axis=-1) / count
    tar = np.nansum(np.where(finite, raw > 10.0, 0.0), axis=-1) / count
    tbr = np.nansum(np.where(finite, raw < 3.9, 0.0), axis=-1) / count
    mean = np.nansum(np.where(finite, raw, 0.0), axis=-1) / count
    return mean.astype(float), hbgi.astype(float), lbgi.astype(float), tar.astype(float), tbr.astype(float)

def patch_minute(indices: np.ndarray) -> np.ndarray:
    return (FIRST_PATCH_MINUTE + indices * PATCH_MINUTES) % (24 * 60)

def collapse_starts(starts: np.ndarray, min_gap: int) -> list[int]:
    kept: list[int] = []
    for start in starts:
        value = int(start)
        if not kept or value - kept[-1] >= min_gap:
            kept.append(value)
    return kept

def individual_attractor_pulse_events(
    patch_idx: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    minute = patch_minute(patch_idx)
    night = minute < NIGHT_END_MINUTE
    reference = y[night] if int(night.sum()) >= 12 else y
    center = np.median(reference, axis=0)
    displacement = y - center
    radius = np.linalg.norm(displacement, axis=1)

    adjacent = np.diff(patch_idx) == 1
    delta_radius = np.diff(radius)
    flow = np.linalg.norm(np.diff(y, axis=0), axis=1)
    valid = adjacent & np.isfinite(delta_radius) & np.isfinite(flow)
    if int(valid.sum()) < 20:
        return displacement, []

    delta_valid = delta_radius[valid]
    flow_valid = flow[valid]
    delta_iqr = float(np.quantile(delta_valid, 0.75) - np.quantile(delta_valid, 0.25))
    flow_iqr = float(np.quantile(flow_valid, 0.75) - np.quantile(flow_valid, 0.25))
    delta_threshold = max(
        float(np.quantile(delta_valid, 0.85)),
        float(np.quantile(delta_valid, 0.75) + delta_iqr),
        0.0,
    )
    flow_threshold = float(np.quantile(flow_valid, 0.75) + 0.25 * flow_iqr)
    starts = collapse_starts(
        np.flatnonzero(
            valid
            & (delta_radius > delta_threshold)
            & (flow >= flow_threshold)
        )
        + 1,
        PULSE_SEPARATION_PATCHES,
    )

    events: list[int] = []
    max_horizon = max(PLE_HORIZONS)
    for start in starts:
        last_patch = patch_idx[start] + PULSE_PEAK_WINDOW_PATCHES
        stop = int(np.searchsorted(patch_idx, last_patch, side="right"))
        if stop <= start:
            continue
        peak = start + int(np.argmax(radius[start:stop]))
        if peak + max_horizon >= len(patch_idx):
            continue
        if not np.all(np.diff(patch_idx[peak : peak + max_horizon + 1]) == 1):
            continue
        if not events or peak != events[-1]:
            events.append(peak)
    return displacement, events

def linear_slope(values: np.ndarray, delta_hours: float = PATCH_MINUTES / 60.0) -> float:
    x = np.arange(len(values), dtype=float) * delta_hours
    x_centered = x - x.mean()
    denominator = float(np.sum(x_centered**2))
    if denominator <= 0:
        return np.nan
    return float(np.sum(x_centered * (values - values.mean())) / denominator)

def finite_time_propagator_terms(
    y: np.ndarray,
    events: list[int],
    pairs: list[tuple[int, int]],
    horizon: int,
    prefix: str,
) -> dict[str, float | int]:
    hours = horizon * PATCH_MINUTES / 60.0
    out: dict[str, float | int] = {
        f"{prefix}_event_count": len(events),
        f"{prefix}_radial_median_per_hour": np.nan,
        f"{prefix}_radial_q75_per_hour": np.nan,
        f"{prefix}_radial_q90_per_hour": np.nan,
        f"{prefix}_state_map_log_sigma_max_per_hour": np.nan,
        f"{prefix}_pair_map_log_sigma_max_per_hour": np.nan,
    }
    if events:
        y0 = np.stack([y[event] for event in events])
        yt = np.stack([y[event + horizon] for event in events])
        radius0 = np.linalg.norm(y0, axis=1)
        radiust = np.linalg.norm(yt, axis=1)
        radial = np.log((radiust + EPS) / (radius0 + EPS)) / hours
        radial = radial[np.isfinite(radial)]
    else:
        y0 = np.empty((0, y.shape[1]), dtype=float)
        yt = np.empty((0, y.shape[1]), dtype=float)
        radial = np.empty(0, dtype=float)
    if len(radial) >= 1:
        out.update(
            {
                f"{prefix}_radial_median_per_hour": float(np.median(radial)),
                f"{prefix}_radial_q75_per_hour": float(np.quantile(radial, 0.75)),
                f"{prefix}_radial_q90_per_hour": float(np.quantile(radial, 0.90)),
            }
        )

    if len(events) < 6:
        return out

    # Healthy coordinates have the fixed point at y=0. The no-intercept
    # least-squares map therefore estimates the finite-time tangent propagator.
    state_map = np.linalg.pinv(y0) @ yt
    state_sigma = float(np.linalg.svd(state_map, compute_uv=False)[0])
    out[f"{prefix}_state_map_log_sigma_max_per_hour"] = float(
        np.log(state_sigma + EPS) / hours
    )

    if len(pairs) >= 6:
        delta0 = np.stack([y[left] - y[right] for left, right in pairs])
        deltat = np.stack(
            [
                y[left + horizon] - y[right + horizon]
                for left, right in pairs
            ]
        )
        pair_map = np.linalg.pinv(delta0) @ deltat
        pair_sigma = float(np.linalg.svd(pair_map, compute_uv=False)[0])
        out[f"{prefix}_pair_map_log_sigma_max_per_hour"] = float(
            np.log(pair_sigma + EPS) / hours
        )
    return out

def robust_location_scale(values: pd.Series) -> tuple[float, float]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
    if x.size == 0:
        return 0.0, 1.0
    center = float(np.nanmedian(x))
    mad = float(np.nanmedian(np.abs(x - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-10:
        q25, q75 = np.nanpercentile(x, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-10:
        scale = float(np.nanstd(x, ddof=1))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return center, scale

def z_from_stats(values: pd.Series, center: float, scale: float) -> pd.Series:
    return ((pd.to_numeric(values, errors="coerce") - center) / scale).clip(-8, 8)

def mahalanobis_radius_terms(pc0: np.ndarray, pc1: np.ndarray, ref: dict[str, np.ndarray | float]) -> dict[str, float]:
    mu = ref["mu"]
    inv_cov = ref["inv_cov"]
    threshold = float(ref["threshold"])
    assert isinstance(mu, np.ndarray)
    assert isinstance(inv_cov, np.ndarray)

    d0 = pc0 - mu
    d1 = pc1 - mu
    v0 = np.einsum("ij,jk,ik->i", d0, inv_cov, d0)
    v1 = np.einsum("ij,jk,ik->i", d1, inv_cov, d1)
    v0 = np.maximum(v0, 0.0)
    v1 = np.maximum(v1, 0.0)
    r0 = np.sqrt(v0)
    r1 = np.sqrt(v1)
    delta_pc = pc1 - pc0
    gradient_half = d0 @ inv_cov
    outward_flow = np.maximum(np.sum(gradient_half * delta_pc, axis=1), 0.0) / (v0 + EPS)
    ell = PATCHES_PER_HOUR * np.log((r1 + EPS) / (r0 + EPS))
    delta_v = (v1 - v0) / max(threshold, EPS)
    normalized_speed = np.linalg.norm(delta_pc, axis=1) / (r0 + EPS)
    valid = np.isfinite(ell) & np.isfinite(outward_flow) & np.isfinite(delta_v)
    if int(valid.sum()) < 8:
        return {
            "sli_step_count": int(valid.sum()),
            "sli_maha_ell_q90_raw": np.nan,
            "sli_maha_ell_q75_raw": np.nan,
            "sli_maha_ell_pos_frac_raw": np.nan,
            "sli_maha_ell_pos_mean_raw": np.nan,
            "sli_maha_out_q90_raw": np.nan,
            "sli_maha_out_pos_frac_raw": np.nan,
            "sli_maha_delta_v_q90_raw": np.nan,
            "sli_maha_delta_v_pos_frac_raw": np.nan,
            "sli_maha_speed_q90_raw": np.nan,
            "sli_maha_boundary_out_q90_raw": np.nan,
            "sli_maha_outer_return_deficit_raw": np.nan,
        }
    ell = ell[valid]
    outward_flow = outward_flow[valid]
    delta_v = delta_v[valid]
    normalized_speed = normalized_speed[valid]
    v0_scaled = v0[valid] / max(threshold, EPS)
    boundary = (v0_scaled > 0.75) & (v0_scaled <= 1.25)
    outer = (v0_scaled > 1.0) & (v0_scaled <= 1.5)
    boundary_out = outward_flow[boundary]
    outer_delta = delta_v[outer]
    return {
        "sli_step_count": int(valid.sum()),
        "sli_maha_ell_q90_raw": float(np.nanquantile(ell, 0.90)),
        "sli_maha_ell_q75_raw": float(np.nanquantile(ell, 0.75)),
        "sli_maha_ell_pos_frac_raw": float(np.mean(ell > 0)),
        "sli_maha_ell_pos_mean_raw": float(np.nanmean(np.maximum(ell, 0))),
        "sli_maha_out_q90_raw": float(np.nanquantile(outward_flow, 0.90)),
        "sli_maha_out_pos_frac_raw": float(np.mean(outward_flow > 0)),
        "sli_maha_delta_v_q90_raw": float(np.nanquantile(delta_v, 0.90)),
        "sli_maha_delta_v_pos_frac_raw": float(np.mean(delta_v > 0)),
        "sli_maha_speed_q90_raw": float(np.nanquantile(normalized_speed, 0.90)),
        "sli_maha_boundary_out_q90_raw": float(np.nanquantile(boundary_out, 0.90)) if int(boundary.sum()) >= 8 else np.nan,
        "sli_maha_outer_return_deficit_raw": float(np.mean(outer_delta >= 0)) if int(outer.sum()) >= 8 else np.nan,
    }
