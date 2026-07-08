from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a sampled LogEvol cross-system experiment.")
    parser.add_argument("--input-root", default="data/processed/logevol")
    parser.add_argument("--output-dir", default="data/processed/experiments/h2_h3_s2_to_s3")
    parser.add_argument("--sources", nargs="+", default=["hadoop2", "hadoop3", "spark2"])
    parser.add_argument("--target", default="spark3")
    parser.add_argument("--source-normal", type=int, default=5000)
    parser.add_argument("--source-anomaly", type=int, default=1000)
    parser.add_argument("--support-normal", type=int, default=1000)
    parser.add_argument("--support-anomaly", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["session_id", "log", "label", "system", "num_lines"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    positives = sum(int(row["label"]) for row in rows)
    rate = positives / len(rows) if rows else 0.0
    print(f"[build_experiment] saved {path}: rows={len(rows)} anomalies={positives} rate={rate:.4f}")


def sample_rows(rows: list[dict[str, str]], max_normal: int, max_anomaly: int, rng: random.Random) -> list[dict[str, str]]:
    normal = [row for row in rows if int(row["label"]) == 0]
    anomaly = [row for row in rows if int(row["label"]) == 1]
    normal = rng.sample(normal, min(max_normal, len(normal)))
    anomaly = rng.sample(anomaly, min(max_anomaly, len(anomaly)))
    sampled = normal + anomaly
    rng.shuffle(sampled)
    return sampled


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)

    mixed_rows: list[dict[str, str]] = []
    for source in args.sources:
        rows = read_csv(input_root / source / "train.csv")
        sampled = sample_rows(rows, args.source_normal, args.source_anomaly, rng)
        write_csv(output_dir / source / "train.csv", sampled)
        mixed_rows.extend(sampled)

    rng.shuffle(mixed_rows)
    write_csv(output_dir / "mixed" / "train.csv", mixed_rows)

    target_valid = read_csv(input_root / args.target / "valid.csv")
    support = sample_rows(target_valid, args.support_normal, args.support_anomaly, rng)
    write_csv(output_dir / args.target / "support.csv", support)

    target_test = read_csv(input_root / args.target / "test.csv")
    write_csv(output_dir / args.target / "test.csv", target_test)


if __name__ == "__main__":
    main()
