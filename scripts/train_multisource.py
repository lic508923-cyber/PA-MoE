"""Train shared encoders and all source experts in one coherent checkpoint."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from pa_moelog.data import LogDataset, LogSequenceDataset, collate_fn
from pa_moelog.models import PAMoELog
from pa_moelog.utils import compute_binary_metrics, save_checkpoint

def args():
    p=argparse.ArgumentParser(description="Train one shared PA-MoELog model across source systems.")
    p.add_argument("--train-csv", required=True); p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--device", default="cpu")
    p.add_argument("--backbone-name", default="bert-base-uncased"); p.add_argument("--no-hash-fallback", action="store_true")
    p.add_argument("--sequence", action="store_true"); p.add_argument("--window-size", type=int, default=20)
    p.add_argument("--stride", type=int, default=None); p.add_argument("--output", default="artifacts/checkpoints/multisource.pt")
    return p.parse_args()

def main():
    a=args(); device=torch.device(a.device)
    dataset = LogSequenceDataset(a.train_csv,a.window_size,a.stride) if a.sequence else LogDataset(a.train_csv)
    systems=sorted({row["system"] for row in (dataset.sequences if a.sequence else dataset.rows)})
    system_to_expert={name:index for index,name in enumerate(systems)}
    model=PAMoELog(hidden_dim=a.hidden_dim,num_experts=len(systems),backbone_name=a.backbone_name,
                   allow_hash_fallback=not a.no_hash_fallback).to(device)
    model.fusion.set_trained_mask(torch.ones(len(systems),dtype=torch.bool,device=device))
    loader=DataLoader(dataset,batch_size=a.batch_size,shuffle=True,collate_fn=collate_fn)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr); criterion=nn.BCEWithLogitsLoss()
    for epoch in range(1,a.epochs+1):
        labels_all=[]; scores_all=[]; losses=[]; model.train()
        for batch in loader:
            labels=batch["labels"].to(device)
            shared=model.encode_sequences(batch["semantic_texts"],batch["parameters"],batch["event_mask"].to(device))
            expert_ids=torch.tensor([system_to_expert[x] for x in batch["systems"]],device=device)
            logits=torch.stack([expert(shared)["logit"] for expert in model.expert_pool.experts],dim=1)
            selected=logits.gather(1,expert_ids[:,None]).squeeze(1); loss=criterion(selected,labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(float(loss.detach())); labels_all.append(labels.cpu()); scores_all.append(torch.sigmoid(selected).detach().cpu())
        metrics=compute_binary_metrics(torch.cat(labels_all),torch.cat(scores_all))
        print(f"[multisource][{epoch}] loss={sum(losses)/len(losses):.4f} f1={metrics['f1']:.4f}")
    # Store one normal prototype per trained expert in that expert's representation space.
    sums=torch.zeros(len(systems),a.hidden_dim,device=device); counts=torch.zeros(len(systems),device=device)
    model.eval()
    with torch.no_grad():
        for batch in DataLoader(dataset,batch_size=a.batch_size,shuffle=False,collate_fn=collate_fn):
            labels=batch["labels"].to(device); shared=model.encode_sequences(batch["semantic_texts"],batch["parameters"],batch["event_mask"].to(device))
            for i,name in enumerate(batch["systems"]):
                if labels[i] == 0:
                    eid=system_to_expert[name]; sums[eid]+=model.expert_pool.experts[eid](shared[i:i+1])["hidden"][0]; counts[eid]+=1
    if bool((counts==0).any()): raise ValueError("every source system needs at least one normal sample for its prototype")
    prototypes=(sums/counts[:,None]).cpu(); config=vars(a).copy(); config.update({"backbone_name":model.backbone_name,"num_experts":len(systems)})
    save_checkpoint(a.output,model,config,extra={"hidden_dim":a.hidden_dim,"num_experts":len(systems),
        "system_to_expert":system_to_expert,"source_normal_prototypes":prototypes,"trained_expert_mask":torch.ones(len(systems),dtype=torch.bool)})
if __name__=="__main__": main()
