from pathlib import Path

import pandas as pd

from rul.model import risk_band, train_model

DATA = Path(__file__).resolve().parents[1] / "data" / "sensor_readings.csv"


def test_model_learns_degradation_signal():
    df = pd.read_csv(DATA)
    trained = train_model(df)
    # generous bound: the data has a clear degradation trend, so a working
    # model should land well under 25 cycles MAE on unseen machines
    assert trained.metrics["mae_cycles"] < 25.0
    assert trained.metrics["test_rows"] > 0


def test_worn_machine_scores_lower_rul_than_healthy_one():
    df = pd.read_csv(DATA)
    trained = train_model(df)
    healthy = df[(df["machine"] == 1) & (df["cycle"] <= 5)]
    worn = df[df["machine"] == 1].tail(5)
    healthy_pred = trained.predict_frame(healthy)[-1]
    worn_pred = trained.predict_frame(worn)[-1]
    assert worn_pred < healthy_pred


def test_risk_bands():
    assert risk_band(10) == "high"
    assert risk_band(50) == "medium"
    assert risk_band(120) == "low"
