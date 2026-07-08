from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pa_moelog.data import LogDataset, collate_fn
from pa_moelog.models import PAMoELog
from pa_moelog.utils import compute_binary_metrics, load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PA-MoELog top-k sparse router.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--expert-checkpoints", nargs="+", required=True)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda-aux", type=float, default=0.01)
    parser.add_argument("--output-dir", default="artifacts/checkpoints/router")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backbone-name", default="bert-base-uncased")
    parser.add_argument("--no-hash-fallback", action="store_true")
    return parser.parse_args()


def load_source_experts(model: PAMoELog, checkpoint_paths: list[str], map_location: torch.device) -> None:
    for fallback_id, path in enumerate(checkpoint_paths):
        checkpoint = load_checkpoint(path, map_location=map_location)
        state = checkpoint["model_state_dict"]
        expert_id = int(checkpoint.get("expert_id", fallback_id))
        if expert_id >= len(model.expert_pool.experts):
            print(f"[router][warning] skip expert checkpoint with out-of-range expert_id={expert_id}: {path}")
            continue

        model_state = model.state_dict()
        copied = 0
        for key, value in state.items():
            if key.startswith(f"expert_pool.experts.{expert_id}.") and key in model_state:
                model_state[key].copy_(value)
                copied += 1
            elif key.startswith("text_encoder.") and key in model_state:
                model_state[key].copy_(value)
            elif key.startswith("parameter_encoder.") and key in model_state:
                model_state[key].copy_(value)
        print(f"[router] loaded expert_id={expert_id} tensors={copied} from {path}")


def set_trainable(model: PAMoELog) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.fusion_encoder, model.router):
        for parameter in module.parameters():
            parameter.requires_grad = True
    for expert in model.expert_pool.experts:
        for parameter in expert.classifier.parameters():
            parameter.requires_grad = True


def train_epoch(
    model: PAMoELog,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_aux: float,
    device: torch.device,
    num_experts: int,
) -> tuple[float, float, float, dict, list[float]]:
    model.train()
    losses: list[float] = []
    cls_losses: list[float] = []
    aux_losses: list[float] = []
    all_labels: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    usage_counter: Counter[int] = Counter()
    selected_total = 0

    for batch in loader:
        labels = batch["labels"].to(device)
        output = model(batch["semantic_texts"], batch["parameters"])
        classification_loss = criterion(output["logit"], labels)
        router_aux_loss = output["router_aux_loss"]
        loss = classification_loss + lambda_aux * router_aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        selected = output["selected_experts"].detach().cpu().reshape(-1).tolist()
        usage_counter.update(int(item) for item in selected)
        selected_total += len(selected)
        losses.append(float(loss.detach().cpu()))
        cls_losses.append(float(classification_loss.detach().cpu()))
        aux_losses.append(float(router_aux_loss.detach().cpu()))
        all_labels.append(labels.detach().cpu())
        all_scores.append(output["final_score"].detach().cpu())

    metrics = compute_binary_metrics(torch.cat(all_labels), torch.cat(all_scores))
    usage = [usage_counter.get(index, 0) / max(selected_total, 1) for index in range(num_experts)]
    return (
        sum(losses) / max(len(losses), 1),
        sum(cls_losses) / max(len(cls_losses), 1),
        sum(aux_losses) / max(len(aux_losses), 1),
        metrics,
        usage,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = LogDataset(args.train_csv)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = PAMoELog(
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        top_k=args.top_k,
        backbone_name=args.backbone_name,
        allow_hash_fallback=not args.no_hash_fallback,
    ).to(device)
    load_source_experts(model, args.expert_checkpoints, device)
    set_trainable(model)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    print(f"[router] samples={len(dataset)} num_experts={args.num_experts} top_k={args.top_k}")

    for epoch in range(1, args.epochs + 1):
        loss, cls_loss, aux_loss, metrics, usage = train_epoch(
            model, loader, criterion, optimizer, args.lambda_aux, device, args.num_experts
        )
        usage_text = ", ".join(f"e{i}={value:.3f}" for i, value in enumerate(usage))
        print(
            f"[router][epoch {epoch}] loss={loss:.4f} cls={cls_loss:.4f} aux={aux_loss:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f} "
            f"usage=[{usage_text}]"
        )

    config = vars(args).copy()
    config["backbone_name"] = model.backbone_name
    config["requested_backbone_name"] = model.requested_backbone_name
    path = Path(args.output_dir) / "router.pt"
    save_checkpoint(
        path,
        model,
        config=config,
        extra={
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "hidden_dim": args.hidden_dim,
            "lambda_aux": args.lambda_aux,
        },
    )


if __name__ == "__main__":
    main()
