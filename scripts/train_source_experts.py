from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pa_moelog.data import LogDataset, collate_fn
from pa_moelog.models import PAMoELog
from pa_moelog.utils import compute_binary_metrics, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one PA-MoELog source expert.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--system-name", required=True)
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="artifacts/checkpoints/source_experts")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--backbone-name", default="bert-base-uncased")
    parser.add_argument("--no-hash-fallback", action="store_true")
    return parser.parse_args()


def compute_pos_weight(dataset: LogDataset, override: float | None, device: torch.device) -> torch.Tensor | None:
    if override is not None:
        return torch.tensor(float(override), dtype=torch.float32, device=device)
    labels = [row["label"] for row in dataset.rows]
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    return torch.tensor(negatives / positives, dtype=torch.float32, device=device)


def set_trainable(model: PAMoELog, expert_id: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module in (model.text_encoder, model.parameter_encoder, model.expert_pool.experts[expert_id]):
        for parameter in module.parameters():
            parameter.requires_grad = True


def train_epoch(
    model: PAMoELog,
    loader: DataLoader,
    expert_id: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, dict]:
    model.train()
    losses: list[float] = []
    all_labels: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []

    for batch in loader:
        labels = batch["labels"].to(device)
        text_embeddings = model.text_encoder(batch["semantic_texts"])
        parameter_embeddings = model.parameter_encoder.encode_sequence(
            batch["parameters"],
            seq_len=text_embeddings.size(1),
            device=device,
        )
        output = model.expert_pool.experts[expert_id](
            text_embeddings=text_embeddings,
            parameter_embeddings=parameter_embeddings,
        )
        loss = criterion(output["logit"], labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        all_labels.append(labels.detach().cpu())
        all_scores.append(output["prob"].detach().cpu())

    metrics = compute_binary_metrics(torch.cat(all_labels), torch.cat(all_scores))
    return sum(losses) / max(len(losses), 1), metrics


def main() -> None:
    args = parse_args()
    if args.expert_id < 0 or args.expert_id >= args.num_experts:
        raise ValueError("--expert-id must be in [0, num_experts).")

    device = torch.device(args.device)
    dataset = LogDataset(args.train_csv, default_system=args.system_name)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = PAMoELog(
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        backbone_name=args.backbone_name,
        allow_hash_fallback=not args.no_hash_fallback,
    ).to(device)
    set_trainable(model, args.expert_id)

    pos_weight = compute_pos_weight(dataset, args.pos_weight, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    print(f"[train_source] samples={len(dataset)} system={args.system_name} expert_id={args.expert_id}")
    if pos_weight is not None:
        print(f"[train_source] pos_weight={float(pos_weight.detach().cpu()):.4f}")

    for epoch in range(1, args.epochs + 1):
        train_loss, metrics = train_epoch(model, loader, args.expert_id, criterion, optimizer, device)
        print(
            f"[train_source][epoch {epoch}] loss={train_loss:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

    config = vars(args).copy()
    config["backbone_name"] = model.backbone_name
    config["requested_backbone_name"] = model.requested_backbone_name
    path = Path(args.output_dir) / f"{args.system_name}_expert{args.expert_id}.pt"
    save_checkpoint(
        path,
        model,
        config=config,
        extra={
            "system_name": args.system_name,
            "expert_id": args.expert_id,
            "hidden_dim": args.hidden_dim,
            "num_experts": args.num_experts,
        },
    )


if __name__ == "__main__":
    main()
