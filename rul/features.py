"""Feature engineering for RUL prediction.

All features are causal: they only use the current and past readings of a
machine, never its future. That keeps them valid at prediction time and
avoids label leakage (no use of max cycle or total life).
"""
import pandas as pd

SENSORS = ("vibration", "temperature", "pressure", "rpm")
WINDOWS = (5, 15)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return one feature row per (machine, cycle), sorted by cycle."""
    df = df.sort_values(["machine", "cycle"]).reset_index(drop=True)
    out = pd.DataFrame({"machine": df["machine"], "cycle": df["cycle"]})
    grouped = df.groupby("machine")
    for sensor in SENSORS:
        out[f"{sensor}_raw"] = df[sensor]
        out[f"{sensor}_diff1"] = grouped[sensor].diff().fillna(0.0)
        for window in WINDOWS:
            rolled = grouped[sensor].rolling(window, min_periods=1)
            out[f"{sensor}_mean{window}"] = rolled.mean().reset_index(level=0, drop=True)
            out[f"{sensor}_std{window}"] = rolled.std().reset_index(level=0, drop=True).fillna(0.0)
    return out


def feature_columns() -> list[str]:
    cols = []
    for sensor in SENSORS:
        cols.append(f"{sensor}_raw")
        cols.append(f"{sensor}_diff1")
        for window in WINDOWS:
            cols.append(f"{sensor}_mean{window}")
            cols.append(f"{sensor}_std{window}")
    return cols
