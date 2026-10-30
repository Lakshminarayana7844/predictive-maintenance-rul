"""FastAPI scoring service for RUL predictions.

The model is trained at startup from the bundled CSV (a few seconds on the
compact sample). Set DATA_PATH to point at a different CSV. Keeping training
at startup means the container needs no model artifact; the dataset is the
artifact.
"""
import os
from functools import lru_cache

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rul.model import risk_band, train_model

DATA_PATH = os.environ.get("DATA_PATH", "data/sensor_readings.csv")
REQUIRED = ("cycle", "vibration", "temperature", "pressure", "rpm")

app = FastAPI(title="predictive-maintenance-rul", version="0.1.0")


class Reading(BaseModel):
    cycle: int = Field(ge=1)
    vibration: float
    temperature: float
    pressure: float
    rpm: float


class ScoreRequest(BaseModel):
    readings: list[Reading]


@lru_cache(maxsize=1)
def get_model():
    return train_model(pd.read_csv(DATA_PATH))


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> dict:
    return get_model().metrics


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    if len(req.readings) < 3:
        raise HTTPException(status_code=422, detail="at least 3 readings are required")
    frame = pd.DataFrame([r.model_dump() for r in req.readings])
    missing = [c for c in REQUIRED if c not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing fields: {missing}")
    frame["machine"] = 1
    trained = get_model()
    rul = float(trained.predict_frame(frame)[-1])
    rul = max(0.0, rul)
    return {
        "rul_cycles": round(rul, 1),
        "risk_band": risk_band(rul),
        "model_mae_cycles": trained.metrics["mae_cycles"],
    }
