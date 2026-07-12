from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pa_moelog.data import LogDataset, collate_fn
from pa_moelog.models import PAMoELog
from pa_moelog.utils import compute_binary_metrics, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PA-MoELog on a CSV test set.")
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--backbone-name", default=None)
    parser.add_argument("--no-hash-fallback", action="store_true")
    parser.add_argument("--basic-metrics-only", action="store_true")
    return parser.parse_args()


def load_model(path: str, device: torch.device, backbone_override: str | None = None, allow_hash_fallback: bool = True) -> PAMoELog:
    checkpoint = load_checkpoint(path, map_location=device)
    config = checkpoint.get("config", {})
    hidden_dim = int(checkpoint.get("hidden_dim", config.get("hidden_dim", 128)))
    num_experts = int(checkpoint.get("num_experts", config.get("num_experts", 3)))
    backbone_name = backbone_override or str(config.get("backbone_name", "bert-base-uncased"))
    model = PAMoELog(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        backbone_name=backbone_name,
        allow_hash_fallback=allow_hash_fallback,
    ).to(device)
    load_checkpoint(path, model=model, map_location=device)
    model.eval()
    return model


def tensor_rows(tensor: torch.Tensor) -> list[str]:
    return [json.dumps(row, ensure_ascii=False) for row in tensor.detach().cpu().tolist()]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = LogDataset(args.test_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    model = load_model(
        args.checkpoint,
        device,
        backbone_override=args.backbone_name,
        allow_hash_fallback=not args.no_hash_fallback,
    )

    labels: list[torch.Tensor] = []
    final_scores: list[torch.Tensor] = []
    classifier_scores: list[torch.Tensor] = []
    energy_scores: list[torch.Tensor] = []
    prediction_rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch in loader:
            output = model(batch["semantic_texts"], batch["parameters"])
            batch_labels = batch["labels"].detach().cpu()
            batch_final = output["final_score"].detach().cpu()
            batch_classifier = output["classifier_score"].detach().cpu()
            batch_energy = output["energy_score"].detach().cpu()
            predictions = (batch_final >= args.threshold).int()
            weight_rows = tensor_rows(output["fusion_weights"])

            labels.append(batch_labels)
            final_scores.append(batch_final)
            classifier_scores.append(batch_classifier)
            energy_scores.append(batch_energy)

            for index, raw_log in enumerate(batch["raw_logs"]):
                prediction_rows.append(
                    {
                        "raw_log": raw_log,
                        "label": int(batch_labels[index].item()),
                        "prediction": int(predictions[index].item()),
                        "final_score": float(batch_final[index].item()),
                        "classifier_score": float(batch_classifier[index].item()),
                        "energy_score": float(batch_energy[index].item()),
                        "fusion_weights": weight_rows[index],
                    }
                )

    y_true = torch.cat(labels)
    y_score = torch.cat(final_scores)
    metrics = compute_binary_metrics(y_true, y_score, threshold=args.threshold)
    result = {
        **metrics,
        "threshold": args.threshold,
        "average_final_score": float(torch.cat(final_scores).mean().item()),
        "average_classifier_score": float(torch.cat(classifier_scores).mean().item()),
        "average_energy_score": float(torch.cat(energy_scores).mean().item()),
        "num_samples": len(dataset),
    }
    if args.basic_metrics_only:
        result = {
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "threshold": args.threshold,
            "num_samples": len(dataset),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    predictions_path = Path("artifacts/outputs/predictions.csv")
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "raw_log",
                "label",
                "prediction",
                "final_score",
                "classifier_score",
                "energy_score",
                "fusion_weights",
            ],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[evaluate] saved metrics: {output_json}")
    print(f"[evaluate] saved predictions: {predictions_path}")


if __name__ == "__main__":
    main()
