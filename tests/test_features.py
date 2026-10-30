import pandas as pd

from rul.features import build_features, feature_columns
from scripts.generate_data import SENSORS


def sample_df() -> pd.DataFrame:
    rows = []
    for machine in (1, 2):
        for cycle in range(1, 21):
            rows.append(
                {
                    "machine": machine,
                    "cycle": cycle,
                    "vibration": 0.5 + 0.01 * cycle,
                    "temperature": 530 + cycle,
                    "pressure": 100.0,
                    "rpm": 1400.0,
                    "rul": 20 - cycle,
                }
            )
    return pd.DataFrame(rows)


def test_build_features_shape_and_columns():
    feats = build_features(sample_df())
    assert len(feats) == 40
    for col in feature_columns():
        assert col in feats.columns
    assert not feats[feature_columns()].isna().any().any()


def test_features_are_causal_per_machine():
    feats = build_features(sample_df())
    first = feats[(feats["machine"] == 1) & (feats["cycle"] == 1)].iloc[0]
    # first row of a machine: rolling mean over one value equals the raw value
    assert first["vibration_mean5"] == first["vibration_raw"]
    assert first["vibration_diff1"] == 0.0


def test_sensor_list_matches_generator():
    assert set(SENSORS) == {"vibration", "temperature", "pressure", "rpm"}
