"""Select HDFS adaptation on validation data and evaluate the locked test once."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    {"name": "head_lr1e3_e8", "adaptation": "head-only", "lr": 1e-3, "epochs": 8},
    {"name": "dora_lr5e4_e8", "adaptation": "dora", "lr": 5e-4, "epochs": 8},
    {"name": "dora_lr1e3_e8", "adaptation": "dora", "lr": 1e-3, "epochs": 8},
    {"name": "partial_lr1e4_e5", "adaptation": "partial", "lr": 1e-4, "epochs": 5},
    {"name": "partial_lr3e4_e5", "adaptation": "partial", "lr": 3e-4, "epochs": 5},
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--metrics-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(command):
    command = [str(item) for item in command]
    print("[hdfs-grid]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    target = args.data_root / "hdfs_target"
    candidates = []
    for config in CONFIGS:
        output = args.output_root / config["name"]
        validation_path = output / "HDFS_validation.json"
        efficiency_path = output / "HDFS_efficiency.json"
        checkpoint_path = output / "HDFS_adapted.pt"
        if not all(path.exists() for path in (validation_path, efficiency_path, checkpoint_path)):
            run([sys.executable, "-u", "scripts/adapt_target.py",
                 "--support-csv", target / "support.csv",
                 "--validation-csv", target / "validation.csv",
                 "--base-checkpoint", args.source_checkpoint,
                 "--target-system", "HDFS", "--sequence",
                 "--batch-size", args.batch_size, "--epochs", config["epochs"],
                 "--lr", config["lr"], "--fusion", "support-guided",
                 "--adaptation", config["adaptation"], "--device", args.device,
                 "--seed", args.seed, "--alpha-prior", "0.7", "--output-dir", output])
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
        candidates.append({**config, "validation_f1": validation["f1"],
            "validation_auprc": validation["auprc"], "alpha": validation["alpha"],
            "threshold": validation["threshold"],
            "trainable_parameters": efficiency["trainable_parameters"],
            "checkpoint": str(checkpoint_path)})
    selected = max(candidates, key=lambda item: (
        item["validation_f1"], item["validation_auprc"], -item["trainable_parameters"]))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection.json").write_text(
        json.dumps({"selected": selected, "candidates": candidates}, indent=2), encoding="utf-8")
    metrics = args.metrics_root / "test.json"
    if not metrics.exists():
        run([sys.executable, "-u", "scripts/evaluate.py", "--test-csv", target / "test.csv",
             "--checkpoint", selected["checkpoint"], "--sequence", "--batch-size", args.batch_size,
             "--device", args.device, "--output-json", metrics])
    result = {"selected": selected,
              "test": json.loads(metrics.read_text(encoding="utf-8"))}
    args.metrics_root.mkdir(parents=True, exist_ok=True)
    (args.metrics_root / "experiment_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
