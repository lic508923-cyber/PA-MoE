"""Run nested OpenStack target adaptation and evaluate each locked outer fold once."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    {"name": "head_lr1e3", "adaptation": "head-only", "lr": 1e-3, "epochs": 5},
    {"name": "dora_lr5e4", "adaptation": "dora", "lr": 5e-4, "epochs": 5},
    {"name": "dora_lr1e3", "adaptation": "dora", "lr": 1e-3, "epochs": 5},
    {"name": "partial_lr1e4", "adaptation": "partial", "lr": 1e-4, "epochs": 3},
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/processed/loghub_crossdomain"))
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/checkpoints/loghub_crossdomain"))
    parser.add_argument("--metrics-root", type=Path, default=Path("artifacts/outputs/loghub_crossdomain"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("[grid]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    args = parse_args()
    summaries = []
    for fold in args.folds:
        fold_data = args.data_root / f"openstack_fold_{fold}"
        candidates = []
        for config in CONFIGS:
            output_dir = args.output_root / f"fold_{fold}" / config["name"]
            validation_path = output_dir / "OpenStack_validation.json"
            efficiency_path = output_dir / "OpenStack_efficiency.json"
            checkpoint_path = output_dir / "OpenStack_adapted.pt"
            if not (validation_path.exists() and efficiency_path.exists() and checkpoint_path.exists()):
                run([
                    sys.executable, "-u", "scripts/adapt_target.py",
                    "--support-csv", str(fold_data / "support.csv"),
                    "--validation-csv", str(fold_data / "validation.csv"),
                    "--base-checkpoint", str(args.source_checkpoint),
                    "--target-system", "OpenStack", "--sequence",
                    "--batch-size", str(args.batch_size), "--epochs", str(config["epochs"]),
                    "--lr", str(config["lr"]), "--fusion", "support-guided",
                    "--adaptation", config["adaptation"], "--device", args.device,
                    "--seed", str(args.seed + fold), "--alpha-prior", "0.7",
                    "--output-dir", str(output_dir),
                ])
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            efficiency = json.loads(efficiency_path.read_text(encoding="utf-8"))
            candidates.append({
                "name": config["name"], "adaptation": config["adaptation"],
                "lr": config["lr"], "epochs": config["epochs"],
                "validation_f1": validation["f1"], "validation_auprc": validation["auprc"],
                "alpha": validation["alpha"], "beta": validation["beta"],
                "threshold": validation["threshold"],
                "f1_tied_alphas": validation["f1_tied_alphas"],
                "trainable_parameters": efficiency["trainable_parameters"],
                "checkpoint": str(checkpoint_path),
            })
        # Validation evidence controls selection. Parameter count is only the final
        # predeclared tie-break, favoring the simpler model when metrics are equal.
        selected = max(candidates, key=lambda item: (
            item["validation_f1"], item["validation_auprc"], -item["trainable_parameters"]
        ))
        selection_path = args.output_root / f"fold_{fold}" / "selection.json"
        selection_path.write_text(json.dumps({"fold": fold, "selected": selected,
            "candidates": candidates}, indent=2), encoding="utf-8")
        metrics_path = args.metrics_root / f"fold_{fold}_test.json"
        if not metrics_path.exists():
            run([
                sys.executable, "-u", "scripts/evaluate.py",
                "--test-csv", str(fold_data / "test.csv"),
                "--checkpoint", selected["checkpoint"], "--sequence",
                "--batch-size", str(args.batch_size), "--device", args.device,
                "--output-json", str(metrics_path),
            ])
        test_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        summaries.append({"fold": fold, "selected": selected, "test": test_metrics})
    args.metrics_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.metrics_root / "nested_cv_results.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[grid] saved {summary_path}")


if __name__ == "__main__":
    main()
