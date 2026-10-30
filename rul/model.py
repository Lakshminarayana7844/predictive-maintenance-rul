"""Train and apply the RUL model.

Gradient boosting on causal rolling features. Evaluation splits by machine
(group holdout) so the score reflects generalisation to unseen machines,
not memorised cycles of machines already seen in training.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from rul.features import build_features, feature_columns


@dataclass
class TrainedModel:
    model: GradientBoostingRegressor
    features: list[str]
    metrics: dict

    def predict_frame(self, readings: pd.DataFrame) -> np.ndarray:
        """Predict RUL for each row of a raw readings frame.

        `readings` must carry machine, cycle and the four sensor columns.
        """
        feats = build_features(readings)
        return self.model.predict(feats[self.features])


def _group_split(machines: np.ndarray, holdout_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.unique(machines)
    shuffled = rng.permutation(unique)
    n_holdout = max(1, int(len(unique) * holdout_fraction))
    holdout = set(shuffled[:n_holdout].tolist())
    mask = np.array([m in holdout for m in machines])
    return ~mask, mask


def train_model(df: pd.DataFrame, holdout_fraction: float = 0.25, seed: int = 7) -> TrainedModel:
    feats = build_features(df)
    cols = feature_columns()
    train_mask, test_mask = _group_split(feats["machine"].to_numpy(), holdout_fraction, seed)
    model = GradientBoostingRegressor(random_state=seed)
    model.fit(feats.loc[train_mask, cols], df.loc[train_mask, "rul"])
    preds = model.predict(feats.loc[test_mask, cols])
    truth = df.loc[test_mask, "rul"].to_numpy()
    metrics = {
        "mae_cycles": round(float(mean_absolute_error(truth, preds)), 2),
        "rmse_cycles": round(float(mean_squared_error(truth, preds) ** 0.5), 2),
        "holdout_machines": int(test_mask.sum() and len(set(feats.loc[test_mask, "machine"]))),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
    }
    # Refit on all data so the shipped model uses every machine.
    model.fit(feats[cols], df["rul"])
    return TrainedModel(model=model, features=cols, metrics=metrics)


def risk_band(rul_cycles: float) -> str:
    if rul_cycles < 30:
        return "high"
    if rul_cycles < 80:
        return "medium"
    return "low"
