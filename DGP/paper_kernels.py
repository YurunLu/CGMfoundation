"""Numerical kernels for the manuscript DGP definitions."""
from __future__ import annotations
from types import SimpleNamespace
import numpy as np
import pandas as pd

PATCHES_PER_DAY = 96
FIRST_PATCH_PHASE = 48
DT_HOURS = 0.25
EPS = 1e-8
COLLAPSE_PATCHES = 8
THEILER_PATCHES = 12
MIN_NIGHT_TRANSITIONS = 24
core = SimpleNamespace(FIRST_PATCH_MINUTE=720, PATCH_MINUTES=15,
                       NIGHT_END_MINUTE=360)

def phase_and_day(patch_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    absolute = np.asarray(patch_idx, dtype=int) + FIRST_PATCH_PHASE
    return absolute % PATCHES_PER_DAY, absolute // PATCHES_PER_DAY


def metric_transform(inv_cov: np.ndarray, dimensions: int) -> np.ndarray:
    precision = np.asarray(inv_cov[:dimensions, :dimensions], dtype=float)
    precision = 0.5 * (precision + precision.T)
    values, vectors = np.linalg.eigh(precision)
    values = np.maximum(values, 1e-8)
    return vectors @ np.diag(np.sqrt(values))


def collapse(starts: np.ndarray) -> list[int]:
    kept: list[int] = []
    for value in starts:
        start = int(value)
        if not kept or start - kept[-1] >= COLLAPSE_PATCHES:
            kept.append(start)
    return kept


def detect_pulse_peaks(patch_idx: np.ndarray, pc: np.ndarray) -> list[int]:
    phase, _ = phase_and_day(patch_idx)
    night = phase < 24
    reference = pc[night] if int(night.sum()) >= 12 else pc
    center = np.nanmedian(reference, axis=0)
    radius = np.linalg.norm(pc - center, axis=1)
    adjacent = np.diff(patch_idx) == 1
    radial_step = np.diff(radius)
    flow = np.linalg.norm(np.diff(pc, axis=0), axis=1)
    valid = adjacent & np.isfinite(radial_step) & np.isfinite(flow)
    if int(valid.sum()) < 20:
        return []
    dr = radial_step[valid]
    fv = flow[valid]
    dr_iqr = float(np.percentile(dr, 75) - np.percentile(dr, 25))
    flow_iqr = float(np.percentile(fv, 75) - np.percentile(fv, 25))
    dr_threshold = max(float(np.percentile(dr, 85)), float(np.percentile(dr, 75) + dr_iqr), 0.0)
    flow_threshold = max(float(np.percentile(fv, 75)), float(np.percentile(fv, 75) + 0.25 * flow_iqr))
    starts = collapse(np.flatnonzero(valid & (radial_step > dr_threshold) & (flow >= flow_threshold)) + 1)
    peaks: list[int] = []
    last_peak = -1
    for start in starts:
        last_patch = patch_idx[start] + COLLAPSE_PATCHES
        stop = int(np.searchsorted(patch_idx, last_patch, side="right"))
        if stop <= start:
            continue
        peak = start + int(np.argmax(radius[start:stop]))
        if peak != last_peak:
            peaks.append(peak)
            last_peak = peak
    return peaks


def circular_phase_distance(left: int, right: int) -> int:
    difference = abs(left - right)
    return min(difference, 96 - difference)


def pulse_neighbor_le(
    patch_idx: np.ndarray,
    pc: np.ndarray,
    peaks: list[int],
    dimensions: int,
    horizon: int,
    tolerance: int,
    inv_cov: np.ndarray,
) -> dict[str, float]:
    transform = metric_transform(inv_cov, dimensions)
    y = pc[:, :dimensions] @ transform
    phase, day = phase_and_day(patch_idx)
    index_by_patch = {int(value): index for index, value in enumerate(patch_idx)}
    valid_peaks = [
        peak
        for peak in peaks
        if all(int(patch_idx[peak] + k) in index_by_patch for k in range(horizon + 1))
    ]
    pairs: set[tuple[int, int]] = set()
    for anchor in valid_peaks:
        candidates = [
            other
            for other in valid_peaks
            if other != anchor
            and abs(int(patch_idx[other]) - int(patch_idx[anchor])) > THEILER_PATCHES
            and circular_phase_distance(int(phase[other]), int(phase[anchor])) <= tolerance
        ]
        if not candidates:
            continue
        distances = np.asarray([np.linalg.norm(y[other] - y[anchor]) for other in candidates])
        neighbor = candidates[int(np.argmin(distances))]
        pairs.add(tuple(sorted((anchor, neighbor))))
    # A single matched response still defines a finite-time exponent. Keep it
    # and expose the pair count instead of filtering participants by trajectory
    # count; precision can be handled in sensitivity analyses.
    if len(pairs) < 1:
        return {"slope": np.nan, "mean": np.nan, "median": np.nan, "q75": np.nan, "pairs": len(pairs)}
    curves = []
    for anchor, neighbor in pairs:
        curve = []
        for k in range(horizon + 1):
            left = index_by_patch[int(patch_idx[anchor] + k)]
            right = index_by_patch[int(patch_idx[neighbor] + k)]
            curve.append(np.log(np.linalg.norm(y[left] - y[right]) + EPS))
        curves.append(curve)
    curves = np.asarray(curves, dtype=float)
    time = np.arange(horizon + 1) * DT_HOURS
    slope = float(np.polyfit(time, np.mean(curves, axis=0), 1)[0])
    endpoint = (curves[:, -1] - curves[:, 0]) / (horizon * DT_HOURS)
    return {
        "slope": slope,
        "mean": float(np.mean(endpoint)),
        "median": float(np.median(endpoint)),
        "q75": float(np.percentile(endpoint, 75)),
        "pairs": len(pairs),
    }


def night_transition_blocks(
    patch_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    adjacent = np.diff(patch_idx) == 1
    starts = np.flatnonzero(adjacent)
    absolute_minute = core.FIRST_PATCH_MINUTE + patch_idx * core.PATCH_MINUTES
    minute0 = absolute_minute[starts] % (24 * 60)
    minute1 = absolute_minute[starts + 1] % (24 * 60)
    night = (minute0 < core.NIGHT_END_MINUTE) & (
        minute1 <= core.NIGHT_END_MINUTE
    )
    starts = starts[night]
    day = np.floor(absolute_minute[starts] / (24 * 60)).astype(int)
    return starts, day


def lag_one_terms(
    values: np.ndarray,
    starts: np.ndarray,
    day: np.ndarray,
    prefix: str,
) -> dict[str, float | int]:
    output: dict[str, float | int] = {
        f"{prefix}_step_count": len(starts),
        f"{prefix}_ar1_global": np.nan,
        f"{prefix}_ac1_global": np.nan,
        f"{prefix}_ar1_nightfixed": np.nan,
        f"{prefix}_ac1_nightfixed": np.nan,
    }
    if len(starts) < MIN_NIGHT_TRANSITIONS:
        return output

    x = np.asarray(values[starts], dtype=float)
    y = np.asarray(values[starts + 1], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    day = day[finite]
    if len(x) < MIN_NIGHT_TRANSITIONS:
        return output

    def slope(left: np.ndarray, right: np.ndarray) -> float:
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denominator = float(np.sum(left_centered**2))
        if denominator <= EPS:
            return np.nan
        return float(np.sum(left_centered * right_centered) / denominator)

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if np.std(left) <= EPS or np.std(right) <= EPS:
            return np.nan
        return float(np.corrcoef(left, right)[0, 1])

    x_fixed = x.copy()
    y_fixed = y.copy()
    for night_id in np.unique(day):
        selected = day == night_id
        if int(selected.sum()) < 2:
            continue
        x_fixed[selected] -= float(x[selected].mean())
        y_fixed[selected] -= float(y[selected].mean())

    output.update(
        {
            f"{prefix}_ar1_global": slope(x, y),
            f"{prefix}_ac1_global": correlation(x, y),
            f"{prefix}_ar1_nightfixed": slope(x_fixed, y_fixed),
            f"{prefix}_ac1_nightfixed": correlation(x_fixed, y_fixed),
        }
    )
    return output


def robust_location_scale(values: pd.Series) -> tuple[float, float]:
    x = (
        pd.to_numeric(values, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(float)
    )
    if x.size == 0:
        return 0.0, 1.0
    center = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - center)))
    if not np.isfinite(scale) or scale < 1e-10:
        q25, q75 = np.percentile(x, [25, 75])
        scale = float((q75 - q25) / 1.349)
    if not np.isfinite(scale) or scale < 1e-10:
        scale = float(np.std(x, ddof=1))
    if not np.isfinite(scale) or scale < 1e-10:
        scale = 1.0
    return center, scale
