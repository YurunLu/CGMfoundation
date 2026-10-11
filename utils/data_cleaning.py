import numpy as np
import pandas as pd
import re


CGM_SENSOR_LOW = 1.11
CGM_SENSOR_HIGH = 33.3
CGM_EXTREME_HIGH = 60.0


def clean_cgm_array(values, low=CGM_SENSOR_LOW, high=CGM_SENSOR_HIGH, extreme_high=CGM_EXTREME_HIGH):
    """
    Clean CGM values before mask generation.

    Rules:
    - non-finite, <=0, and extreme-high values are treated as missing.
    - positive values below the sensor lower bound are clipped to low.
    - values above the sensor upper bound but not extreme are clipped to high.
    """
    arr = np.asarray(values, dtype=np.float32).copy()
    arr[~np.isfinite(arr)] = np.nan
    arr[arr <= 0] = np.nan
    arr[arr > extreme_high] = np.nan
    np.clip(arr, low, high, out=arr)
    return arr


def clean_cgm_dataframe(df, low=CGM_SENSOR_LOW, high=CGM_SENSOR_HIGH, extreme_high=CGM_EXTREME_HIGH):
    cleaned = df.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    cleaned = cleaned.mask(cleaned <= 0)
    cleaned = cleaned.mask(cleaned > extreme_high)
    return cleaned.clip(lower=low, upper=high)


def align_cgm_dataframe_columns(df, max_len):
    """Align CGM columns by numeric time index, accepting 0-based or 1-based CSVs."""
    col_by_idx = {}
    for col in df.columns:
        nums = re.findall(r"\d+", str(col))
        if nums:
            col_by_idx[int(nums[0])] = col

    if not col_by_idx:
        return pd.DataFrame(index=df.index, columns=[str(i) for i in range(max_len)])

    zero_based_score = sum(i in col_by_idx for i in range(max_len))
    one_based_score = sum(i in col_by_idx for i in range(1, max_len + 1))
    start = 0 if zero_based_score >= one_based_score else 1

    columns = {}
    for pos, idx in enumerate(range(start, start + max_len)):
        columns[str(pos)] = df[col_by_idx[idx]] if idx in col_by_idx else np.nan
    return pd.DataFrame(columns, index=df.index)
