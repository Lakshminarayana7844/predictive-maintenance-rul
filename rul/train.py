"""Train the RUL model on the bundled CSV and print evaluation metrics.

    python -m rul.train --data data/sensor_readings.csv
"""
import argparse
import json

import pandas as pd

from rul.model import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/sensor_readings.csv")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    trained = train_model(df)
    print(json.dumps(trained.metrics, indent=2))


if __name__ == "__main__":
    main()
