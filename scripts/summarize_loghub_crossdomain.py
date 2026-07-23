"""Summarize confirmatory and exploratory Loghub cross-domain experiments."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch


def aggregate(results_path: Path) -> dict:
    folds = json.loads(results_path.read_text(encoding="utf-8"))
    f1 = [float(item["test"]["f1"]) for item in folds]
    alpha = [float(item["selected"]["alpha"]) for item in folds]
    auroc = [float(item["test"]["auroc"]) for item in folds]
    auprc = [float(item["test"]["auprc"]) for item in folds]
    pooled = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    rows = []
    fusion_weights = []
    for item in folds:
        matrix = item["test"]["confusion_matrix"]
        pooled["tn"] += int(matrix[0][0]); pooled["fp"] += int(matrix[0][1])
        pooled["fn"] += int(matrix[1][0]); pooled["tp"] += int(matrix[1][1])
        checkpoint = torch.load(item["selected"]["checkpoint"], map_location="cpu", weights_only=False)
        weights = checkpoint["model_state_dict"]["fusion.weights"].tolist()
        fusion_weights.append(weights)
        rows.append({
            "fold": item["fold"], "selected": item["selected"]["name"],
            "validation_f1": item["selected"]["validation_f1"],
            "alpha": item["selected"]["alpha"], "beta": item["selected"]["beta"],
            "f1_tied_alphas": item["selected"]["f1_tied_alphas"],
            "test_f1": item["test"]["f1"], "precision": item["test"]["precision"],
            "recall": item["test"]["recall"], "auroc": item["test"]["auroc"],
            "auprc": item["test"]["auprc"], "fpr": item["test"]["fpr"],
            "confusion_matrix": matrix, "fusion_weights": weights,
            "trainable_parameters": item["selected"]["trainable_parameters"],
        })
    precision = pooled["tp"] / max(pooled["tp"] + pooled["fp"], 1)
    recall = pooled["tp"] / max(pooled["tp"] + pooled["fn"], 1)
    pooled_f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "folds": rows,
        "macro": {
            "f1_mean": statistics.mean(f1), "f1_stdev": statistics.stdev(f1),
            "alpha_mean": statistics.mean(alpha), "alpha_stdev": statistics.stdev(alpha),
            "auroc_mean": statistics.mean(auroc), "auroc_stdev": statistics.stdev(auroc),
            "auprc_mean": statistics.mean(auprc), "auprc_stdev": statistics.stdev(auprc),
        },
        "pooled": {**pooled, "precision": precision, "recall": recall, "f1": pooled_f1},
        "fusion_weights_mean": [statistics.mean(column) for column in zip(*fusion_weights)],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--confirmatory", type=Path, required=True)
    parser.add_argument("--exploratory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
    result = {
        "source": {"best_epoch": source["best_epoch"],
                   "validation_macro_auprc": source["best_validation_macro_auprc"],
                   "validation_loss": source["best_validation_loss"]},
        "data_manifest": json.loads(args.manifest.read_text(encoding="utf-8")),
        "confirmatory_whole_vm": aggregate(args.confirmatory),
        "exploratory_window5": aggregate(args.exploratory),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"source": result["source"],
        "confirmatory": result["confirmatory_whole_vm"]["macro"],
        "exploratory": result["exploratory_window5"]["macro"]}, indent=2))


if __name__ == "__main__":
    main()
