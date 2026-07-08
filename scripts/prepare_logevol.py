from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LogEvol session PKL files to PA-MoELog CSV files.")
    parser.add_argument("--input-root", default=r"D:\GoogleDownload\Logevol")
    parser.add_argument("--output-dir", default="data/processed/logevol")
    parser.add_argument("--datasets", nargs="+", default=["hadoop2", "hadoop3", "spark2", "spark3"])
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    parser.add_argument("--max-lines", type=int, default=200, help="Maximum log lines kept per session.")
    parser.add_argument("--join-token", default=" [SEP] ", help="Token used to join session log lines.")
    parser.add_argument("--write-mixed", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_pickle(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in {path}, got {type(data)!r}")
    return data


def session_to_row(session_id: str, item: dict[str, Any], system: str, max_lines: int, join_token: str) -> dict[str, Any]:
    content = item.get("Content") or item.get("content") or []
    if isinstance(content, str):
        lines = [content]
    else:
        lines = [str(line) for line in content[:max_lines]]
    log = join_token.join(line.strip() for line in lines if line and line.strip())
    label = int(item.get("label", 0))
    return {
        "session_id": session_id,
        "log": log,
        "label": label,
        "system": system,
        "num_lines": len(lines),
    }


def convert_split(input_path: Path, output_path: Path, system: str, max_lines: int, join_token: str) -> None:
    data = load_pickle(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    anomalies = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["session_id", "log", "label", "system", "num_lines"])
        writer.writeheader()
        for session_id, item in data.items():
            row = session_to_row(str(session_id), item, system, max_lines=max_lines, join_token=join_token)
            if not row["log"]:
                continue
            writer.writerow(row)
            total += 1
            anomalies += int(row["label"])
    rate = anomalies / total if total else 0.0
    print(f"[prepare_logevol] {system} {input_path.name}: rows={total} anomalies={anomalies} rate={rate:.4f}")
    print(f"[prepare_logevol] saved: {output_path}")


def write_mixed_split(output_dir: Path, datasets: list[str], split: str) -> None:
    output_path = output_dir / "mixed" / f"{split}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    anomalies = 0
    with output_path.open("w", encoding="utf-8", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=["session_id", "log", "label", "system", "num_lines"])
        writer.writeheader()
        for dataset in datasets:
            input_path = output_dir / dataset / f"{split}.csv"
            if not input_path.exists():
                continue
            with input_path.open("r", encoding="utf-8", newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    writer.writerow(row)
                    total += 1
                    anomalies += int(row["label"])
    rate = anomalies / total if total else 0.0
    print(f"[prepare_logevol] mixed {split}: rows={total} anomalies={anomalies} rate={rate:.4f}")
    print(f"[prepare_logevol] saved: {output_path}")


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    for dataset in args.datasets:
        for split in args.splits:
            input_path = input_root / dataset / f"session_{split}.pkl"
            if not input_path.exists():
                print(f"[prepare_logevol][warning] missing: {input_path}")
                continue
            output_path = output_dir / dataset / f"{split}.csv"
            convert_split(input_path, output_path, dataset, max_lines=args.max_lines, join_token=args.join_token)
    if args.write_mixed:
        for split in args.splits:
            write_mixed_split(output_dir, args.datasets, split)


if __name__ == "__main__":
    main()
