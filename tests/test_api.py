from fastapi.testclient import TestClient

from rul.api import app

client = TestClient(app)


def reading(cycle: int, worn: bool) -> dict:
    return {
        "cycle": cycle,
        "vibration": 2.5 if worn else 0.5,
        "temperature": 580.0 if worn else 530.0,
        "pressure": 95.0 if worn else 102.0,
        "rpm": 1400.0,
    }


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_score_worn_machine_is_high_risk():
    payload = {"readings": [reading(c, worn=True) for c in range(1, 20)]}
    body = client.post("/score", json=payload).json()
    assert body["risk_band"] in {"high", "medium"}
    assert body["rul_cycles"] >= 0


def test_score_needs_three_readings():
    payload = {"readings": [reading(1, worn=False), reading(2, worn=False)]}
    resp = client.post("/score", json=payload)
    assert resp.status_code == 422
