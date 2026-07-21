"""Build a deterministic, leakage-free LogEvol F1 benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="data/processed/logevol")
    parser.add_argument("--output-dir", default="data/processed/experiments/f1_optimized")
    parser.add_argument("--sources", nargs="+", default=["hadoop2", "hadoop3", "spark2"])
    parser.add_argument("--target", default="spark3")
    parser.add_argument("--source-train-normal", type=int, default=5000)
    parser.add_argument("--source-train-anomaly", type=int, default=1000)
    parser.add_argument("--source-validation-normal", type=int, default=1000)
    parser.add_argument("--source-validation-anomaly", type=int, default=200)
    parser.add_argument("--target-support-normal", type=int, default=400)
    parser.add_argument("--target-support-anomaly", type=int, default=20)
    parser.add_argument("--target-validation-normal", type=int, default=800)
    parser.add_argument("--target-validation-anomaly", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_by_class(
    rows: list[dict[str, str]], normal_count: int, anomaly_count: int, rng: random.Random
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    normal = [row for row in rows if int(row["label"]) == 0]
    anomaly = [row for row in rows if int(row["label"]) == 1]
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    selected = normal[: min(normal_count, len(normal))] + anomaly[: min(anomaly_count, len(anomaly))]
    remaining = normal[min(normal_count, len(normal)) :] + anomaly[min(anomaly_count, len(anomaly)) :]
    rng.shuffle(selected)
    rng.shuffle(remaining)
    return selected, remaining


def write_rows(path: Path, rows: list[dict[str, str]]) -> dict[str, int | float | str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty split: {path}")
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    anomalies = sum(int(row["label"]) for row in rows)
    if anomalies == 0 or anomalies == len(rows):
        raise ValueError(f"Split must contain both classes: {path}")
    summary = {
        "path": str(path),
        "rows": len(rows),
        "normal": len(rows) - anomalies,
        "anomaly": anomalies,
        "anomaly_rate": anomalies / len(rows),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return summary


def ids(rows: list[dict[str, str]]) -> set[str]:
    return {row.get("session_id", "") for row in rows}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    source_train: list[dict[str, str]] = []
    source_validation: list[dict[str, str]] = []
    for source in args.sources:
        train, _ = sample_by_class(
            read_rows(Path(args.input_root) / source / "train.csv"),
            args.source_train_normal,
            args.source_train_anomaly,
            rng,
        )
        validation, _ = sample_by_class(
            read_rows(Path(args.input_root) / source / "valid.csv"),
            args.source_validation_normal,
            args.source_validation_anomaly,
            rng,
        )
        source_train.extend(train)
        source_validation.extend(validation)
    rng.shuffle(source_train)
    rng.shuffle(source_validation)

    target_valid = read_rows(Path(args.input_root) / args.target / "valid.csv")
    target_support, remaining = sample_by_class(
        target_valid, args.target_support_normal, args.target_support_anomaly, rng
    )
    target_validation, _ = sample_by_class(
        remaining, args.target_validation_normal, args.target_validation_anomaly, rng
    )
    if ids(target_support) & ids(target_validation):
        raise RuntimeError("Target support and validation sessions overlap.")
    target_test = read_rows(Path(args.input_root) / args.target / "test.csv")
    if (ids(target_support) | ids(target_validation)) & ids(target_test):
        raise RuntimeError("Target development sessions overlap the test set.")

    output = Path(args.output_dir)
    manifest = {
        "seed": args.seed,
        "sources": args.sources,
        "target": args.target,
        "splits": {
            "source_train": write_rows(output / "source_train.csv", source_train),
            "source_validation": write_rows(output / "source_validation.csv", source_validation),
            "target_support": write_rows(output / "target_support.csv", target_support),
            "target_validation": write_rows(output / "target_validation.csv", target_validation),
            "target_test": write_rows(output / "target_test.csv", target_test),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
