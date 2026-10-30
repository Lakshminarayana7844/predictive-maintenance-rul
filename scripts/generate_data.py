"""Generate a synthetic machine sensor dataset with degradation to failure.

Deterministic (seeded) so the committed CSV is reproducible:
    python scripts/generate_data.py            # writes data/sensor_readings.csv
    python scripts/generate_data.py --machines 200 --out data/big.csv

Each machine has its own sensor baselines and health. Vibration and
temperature ramp up as the machine approaches failure; pressure and rpm
stay mostly flat with noise. The label is remaining useful life (RUL) in
cycles. The patterns are inspired by public turbofan datasets (NASA
C-MAPSS) but everything here is synthetic.
"""
import argparse
import csv
import random

SENSORS = ("vibration", "temperature", "pressure", "rpm")


def machine_life(rng: random.Random) -> int:
    return rng.randint(70, 130)


def simulate_machine(machine_id: int, rng: random.Random) -> list[dict]:
    life = machine_life(rng)
    base_vib = rng.uniform(0.4, 0.7)
    base_tmp = rng.uniform(520.0, 545.0)
    base_prs = rng.uniform(95.0, 105.0)
    base_rpm = rng.uniform(1380.0, 1420.0)
    wear_rate = rng.uniform(0.7, 1.3)  # machines degrade at different speeds
    rows = []
    for cycle in range(1, life + 1):
        rul = life - cycle
        degradation = wear_rate * (cycle / life) ** 2
        vibration = base_vib + 2.4 * degradation + rng.gauss(0, 0.06)
        temperature = base_tmp + 42.0 * degradation + rng.gauss(0, 1.5)
        pressure = base_prs - 6.0 * degradation + rng.gauss(0, 0.4)
        rpm = base_rpm + rng.gauss(0, 6.0)
        rows.append(
            {
                "machine": machine_id,
                "cycle": cycle,
                "vibration": round(vibration, 3),
                "temperature": round(temperature, 2),
                "pressure": round(pressure, 2),
                "rpm": round(rpm, 1),
                "rul": rul,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/sensor_readings.csv")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    with open(args.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["machine", "cycle", *SENSORS, "rul"])
        writer.writeheader()
        for machine_id in range(1, args.machines + 1):
            writer.writerows(simulate_machine(machine_id, rng))
    print(f"wrote {args.machines} machines to {args.out}")


if __name__ == "__main__":
    main()
