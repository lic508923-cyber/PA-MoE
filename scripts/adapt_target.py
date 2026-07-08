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
from pa_moelog.models.dora_adalora import DoRALinear
from pa_moelog.utils import compute_binary_metrics, load_checkpoint, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Few-label target-system adaptation for PA-MoELog.")
    parser.add_argument("--support-csv", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--target-system", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lambda-energy", type=float, default=0.1)
    parser.add_argument("--output-dir", default="artifacts/checkpoints/target_adapt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--freeze-backbone", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--backbone-name", default=None)
    parser.add_argument("--no-hash-fallback", action="store_true")
    return parser.parse_args()


def model_from_checkpoint(
    path: str,
    device: torch.device,
    backbone_override: str | None = None,
    allow_hash_fallback: bool = True,
) -> tuple[PAMoELog, dict]:
    checkpoint = load_checkpoint(path, map_location=device)
    config = checkpoint.get("config", {})
    hidden_dim = int(checkpoint.get("hidden_dim", config.get("hidden_dim", 128)))
    num_experts = int(checkpoint.get("num_experts", config.get("num_experts", 3)))
    top_k = int(checkpoint.get("top_k", config.get("top_k", 2)))
    backbone_name = backbone_override or str(config.get("backbone_name", "bert-base-uncased"))
    model = PAMoELog(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        top_k=top_k,
        backbone_name=backbone_name,
        allow_hash_fallback=allow_hash_fallback,
    ).to(device)
    load_checkpoint(path, model=model, map_location=device)
    return model, checkpoint


def set_adaptation_trainable(model: PAMoELog, freeze_backbone: bool) -> None:
    if not freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = True
        return

    for parameter in model.parameters():
        parameter.requires_grad = False

    for module in (model.router, model.gmm_energy):
        for parameter in module.parameters():
            parameter.requires_grad = True

    for expert in model.expert_pool.experts:
        for parameter in expert.classifier.parameters():
            parameter.requires_grad = True
        for module in expert.modules():
            if isinstance(module, DoRALinear):
                for parameter in (module.lora_a, module.lora_b, module.magnitude):
                    parameter.requires_grad = True


def train_epoch(
    model: PAMoELog,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    lambda_energy: float,
    device: torch.device,
) -> tuple[float, float, float, dict]:
    model.train()
    losses: list[float] = []
    bce_losses: list[float] = []
    energy_losses: list[float] = []
    all_labels: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []

    for batch in loader:
        labels = batch["labels"].to(device)
        output = model(batch["semantic_texts"], batch["parameters"])
        bce_loss = criterion(output["logit"], labels)
        energy_regularization = torch.mean((output["energy_score"] - labels).pow(2))
        loss = bce_loss + lambda_energy * energy_regularization

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(float(loss.detach().cpu()))
        bce_losses.append(float(bce_loss.detach().cpu()))
        energy_losses.append(float(energy_regularization.detach().cpu()))
        all_labels.append(labels.detach().cpu())
        all_scores.append(output["final_score"].detach().cpu())

    metrics = compute_binary_metrics(torch.cat(all_labels), torch.cat(all_scores))
    return (
        sum(losses) / max(len(losses), 1),
        sum(bce_losses) / max(len(bce_losses), 1),
        sum(energy_losses) / max(len(energy_losses), 1),
        metrics,
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dataset = LogDataset(args.support_csv, default_system=args.target_system)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model, base_checkpoint = model_from_checkpoint(
        args.base_checkpoint,
        device,
        backbone_override=args.backbone_name,
        allow_hash_fallback=not args.no_hash_fallback,
    )
    set_adaptation_trainable(model, args.freeze_backbone)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    print(f"[adapt] samples={len(dataset)} target={args.target_system} trainable={trainable}/{total}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss, bce_loss, energy_loss, metrics = train_epoch(
            model, loader, criterion, optimizer, args.lambda_energy, device
        )
        print(
            f"[adapt][epoch {epoch}] loss={loss:.4f} bce={bce_loss:.4f} energy={energy_loss:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

    config = vars(args).copy()
    config["backbone_name"] = model.backbone_name
    config["requested_backbone_name"] = model.requested_backbone_name
    config["base_config"] = base_checkpoint.get("config", {})
    path = Path(args.output_dir) / f"{args.target_system}_adapted.pt"
    save_checkpoint(
        path,
        model,
        config=config,
        extra={
            "target_system": args.target_system,
            "hidden_dim": model.hidden_dim,
            "num_experts": model.router.num_experts,
            "top_k": model.router.top_k,
        },
    )


if __name__ == "__main__":
    main()
