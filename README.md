# predictive-maintenance-rul

Predict remaining useful life (RUL) of machines from vibration, temperature,
pressure and rpm readings. A gradient boosting model scores how many cycles a
machine has left and maps that to a risk band, served over a FastAPI endpoint.

Built as a portfolio project for industrial AI and data engineering work:
condition monitoring and predictive maintenance are core use cases when
factory equipment is instrumented with sensors.

## What it does

- Synthetic but realistic sensor data: 30 machines, each running 70-130
  cycles until failure, with vibration and temperature ramping up as the
  machine wears out (generator included, seeded and reproducible)
- Causal feature engineering: rolling means/stds over 5 and 15 cycle windows
  plus first differences, computed per machine with no future leakage
- GradientBoostingRegressor for RUL in cycles, evaluated with a group
  holdout split by machine so the score reflects unseen equipment
- FastAPI service: POST /score with recent readings, get back predicted RUL
  and a risk band (high < 30 cycles, medium < 80, low otherwise)
- Tests cover feature causality, model quality and the API

## Results on the bundled data

Evaluated on 7 held-out machines (767 cycles) the model never saw in
training:

| Metric | Value |
| --- | --- |
| MAE | 10.11 cycles |
| RMSE | 14.14 cycles |

Reproduce with:

```bash
python -m rul.train
```

## Run it

```bash
pip install -r requirements.txt
python -m rul.train            # prints holdout metrics
uvicorn rul.api:app --port 8000
```

Then score a machine from its recent readings (at least 3):

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "readings": [
    {"cycle": 90, "vibration": 2.1, "temperature": 575.0, "pressure": 96.0, "rpm": 1405.0},
    {"cycle": 91, "vibration": 2.2, "temperature": 577.0, "pressure": 95.8, "rpm": 1398.0},
    {"cycle": 92, "vibration": 2.3, "temperature": 579.0, "pressure": 95.5, "rpm": 1401.0}
  ]
}'
```

Example response:

```json
{"rul_cycles": 12.4, "risk_band": "high", "model_mae_cycles": 10.11}
```

`GET /metrics` returns the holdout evaluation of the running model.

## Docker

```bash
docker build -t rul-service .
docker run -p 8000:8000 rul-service
```

The container trains the model at startup from the bundled CSV (a few
seconds), so no model artifact needs to be shipped. Point DATA_PATH at
another CSV to use your own data.

## Project layout

```
data/sensor_readings.csv   compact synthetic dataset (30 machines, ~3.1k rows)
scripts/generate_data.py   seeded data generator, scales to any size
rul/features.py            causal rolling feature engineering
rul/model.py               training, group-holdout evaluation, risk bands
rul/train.py               CLI: train and print metrics
rul/api.py                 FastAPI scoring service
tests/                     feature, model and API tests
```

## Honest limits

- The bundled data is synthetic with a clean degradation trend; real sensor
  data is noisier, has sensor drift and failure modes this dataset does not
  model. The generator is a starting point, not a validated digital twin.
- The committed CSV is a compact 30-machine sample so the repo stays light;
  the same pipeline runs on larger output from the generator.
- RUL regression here is univariate per machine; there is no transfer across
  machine types and no uncertainty interval on predictions.
