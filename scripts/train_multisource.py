"""Balanced multisource training with validation-selected checkpoints."""
from __future__ import annotations
import argparse,copy,sys
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader,Sampler
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pa_moelog.data import LogDataset,LogSequenceDataset,collate_fn
from pa_moelog.models import PAMoELog
from pa_moelog.utils import compute_binary_metrics,save_checkpoint

def args():
    p=argparse.ArgumentParser(description="Train a balanced shared PA-MoELog model across source systems.")
    p.add_argument("--train-csv",required=True); p.add_argument("--validation-csv",required=True)
    p.add_argument("--hidden-dim",type=int,default=128); p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--epochs",type=int,default=20); p.add_argument("--patience",type=int,default=5)
    p.add_argument("--scheduler-patience",type=int,default=2); p.add_argument("--scheduler-factor",type=float,default=.5)
    p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--device",default="cpu"); p.add_argument("--seed",type=int,default=7)
    p.add_argument("--backbone-name",default="bert-base-uncased"); p.add_argument("--debug-hash-encoder",action="store_true")
    p.add_argument("--sequence",action="store_true"); p.add_argument("--window-size",type=int,default=20); p.add_argument("--stride",type=int,default=None)
    p.add_argument("--max-events",type=int,default=512); p.add_argument("--sequence-layers",type=int,default=1)
    p.add_argument("--dora-rank",type=int,default=4); p.add_argument("--gmm-projection-dim",type=int,default=32)
    p.add_argument("--output",default="artifacts/checkpoints/multisource.pt"); return p.parse_args()

def make_dataset(path,a):
    return LogSequenceDataset(path,a.window_size,a.stride) if a.sequence else LogDataset(path)

def records(dataset): return dataset.sequences if isinstance(dataset,LogSequenceDataset) else dataset.rows

class SystemBalancedBatchSampler(Sampler):
    """Round-robin systems so every feasible batch contains multiple sources."""
    def __init__(self,rows,batch_size,seed=7):
        self.batch_size=batch_size; self.seed=seed; self.epoch=0; self.num_batches=max(1,(len(rows)+batch_size-1)//batch_size)
        self.by_system=defaultdict(list)
        for index,row in enumerate(rows): self.by_system[row["system"]].append(index)
        self.systems=sorted(self.by_system)
    def __len__(self): return self.num_batches
    def __iter__(self):
        generator=torch.Generator().manual_seed(self.seed+self.epoch); self.epoch+=1
        queues={name:[self.by_system[name][i] for i in torch.randperm(len(self.by_system[name]),generator=generator).tolist()] for name in self.systems}
        pointers={name:0 for name in self.systems}
        for batch_index in range(self.num_batches):
            batch=[]
            for offset in range(self.batch_size):
                name=self.systems[(batch_index*self.batch_size+offset)%len(self.systems)]
                if pointers[name]>=len(queues[name]):
                    queues[name]=[self.by_system[name][i] for i in torch.randperm(len(self.by_system[name]),generator=generator).tolist()]; pointers[name]=0
                batch.append(queues[name][pointers[name]]); pointers[name]+=1
            yield batch

def shared(model,batch,device):
    return model.encode_sequences(batch["semantic_texts"],batch["parameters"],batch["event_mask"].to(device))

@torch.no_grad()
def validate(model,loader,system_to_expert,device):
    model.eval(); by_system=defaultdict(lambda:[[],[]]); losses=[]
    for batch in loader:
        labels=batch["labels"].to(device); hidden=shared(model,batch,device)
        expert_ids=torch.tensor([system_to_expert[name] for name in batch["systems"]],device=device)
        logits=torch.stack([expert(hidden)["logit"] for expert in model.expert_pool.experts],1)
        selected=logits.gather(1,expert_ids[:,None]).squeeze(1); losses.append(float(F.binary_cross_entropy_with_logits(selected,labels)))
        for index,name in enumerate(batch["systems"]):
            by_system[name][0].append(labels[index].cpu()); by_system[name][1].append(torch.sigmoid(selected[index]).cpu())
    auprcs=[compute_binary_metrics(torch.stack(values[0]),torch.stack(values[1]))["auprc"] for values in by_system.values()]
    return sum(auprcs)/len(auprcs),sum(losses)/max(len(losses),1)

def main():
    a=args(); torch.manual_seed(a.seed); device=torch.device(a.device)
    train=make_dataset(a.train_csv,a); validation=make_dataset(a.validation_csv,a)
    train_rows=records(train); validation_rows=records(validation); systems=sorted({row["system"] for row in train_rows})
    if {row["system"] for row in validation_rows} != set(systems): raise ValueError("validation must contain exactly the training source systems")
    system_to_expert={name:index for index,name in enumerate(systems)}
    if a.backbone_name in {"hash","simple-hash-encoder"} and not a.debug_hash_encoder:
        raise ValueError("hash encoder is debug-only; pass --debug-hash-encoder explicitly")
    model=PAMoELog(hidden_dim=a.hidden_dim,num_experts=len(systems),backbone_name=a.backbone_name,
        allow_hash_fallback=a.debug_hash_encoder,max_events=a.max_events,gmm_projection_dim=a.gmm_projection_dim,
        sequence_layers=a.sequence_layers,dora_rank=a.dora_rank).to(device)
    model.fusion.set_trained_mask(torch.ones(len(systems),dtype=torch.bool,device=device))
    loader=DataLoader(train,batch_sampler=SystemBalancedBatchSampler(train_rows,a.batch_size,a.seed),collate_fn=collate_fn)
    validation_loader=DataLoader(validation,batch_size=a.batch_size,shuffle=False,collate_fn=collate_fn)
    class_counts=defaultdict(lambda:[0,0])
    for row in train_rows: class_counts[row["system"]][row["label"]]+=1
    missing_classes={name:counts for name,counts in class_counts.items() if 0 in counts}
    if missing_classes: raise ValueError(f"every source system needs both classes; counts={missing_classes}")
    validation_counts=defaultdict(lambda:[0,0])
    for row in validation_rows: validation_counts[row["system"]][row["label"]]+=1
    missing_validation={name:counts for name,counts in validation_counts.items() if 0 in counts}
    if missing_validation: raise ValueError(f"validation needs both classes per source; counts={missing_validation}")
    positive_weights=torch.tensor([class_counts[name][0]/max(class_counts[name][1],1) for name in systems],device=device)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=a.scheduler_factor,patience=a.scheduler_patience)
    best_state=None; best_score=(-1.0,float("inf")); best_epoch=0; stale=0
    for epoch in range(1,a.epochs+1):
        model.train(); losses=[]
        for batch in loader:
            labels=batch["labels"].to(device); hidden=shared(model,batch,device)
            expert_ids=torch.tensor([system_to_expert[name] for name in batch["systems"]],device=device)
            logits=torch.stack([expert(hidden)["logit"] for expert in model.expert_pool.experts],1)
            selected=logits.gather(1,expert_ids[:,None]).squeeze(1)
            system_losses=[]
            for expert_id in expert_ids.unique():
                selected_system=expert_ids==expert_id
                system_losses.append(F.binary_cross_entropy_with_logits(selected[selected_system],labels[selected_system],
                    pos_weight=positive_weights[expert_id]))
            loss=torch.stack(system_losses).mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        macro_auprc,val_loss=validate(model,validation_loader,system_to_expert,device)
        scheduler.step(macro_auprc)
        print(f"[multisource][{epoch}] train_loss={sum(losses)/len(losses):.4f} val_loss={val_loss:.4f} macro_auprc={macro_auprc:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")
        score=(macro_auprc,-val_loss)
        if score>(best_score[0],-best_score[1]):
            best_score=(macro_auprc,val_loss); best_epoch=epoch; stale=0; best_state=copy.deepcopy(model.state_dict())
        else:
            stale+=1
            if stale>=a.patience: print(f"[multisource] early stop at epoch {epoch}"); break
    if best_state is None: raise RuntimeError("no best checkpoint was selected")
    model.load_state_dict(best_state,strict=True)
    sums=torch.zeros(len(systems),a.hidden_dim,device=device); counts=torch.zeros(len(systems),device=device); model.eval()
    with torch.no_grad():
        for batch in DataLoader(train,batch_size=a.batch_size,shuffle=False,collate_fn=collate_fn):
            labels=batch["labels"].to(device); hidden=shared(model,batch,device)
            for i,name in enumerate(batch["systems"]):
                if labels[i]==0:
                    eid=system_to_expert[name]; sums[eid]+=model.expert_pool.experts[eid](hidden[i:i+1])["hidden"][0]; counts[eid]+=1
    if bool((counts==0).any()): raise ValueError("every source needs a normal sample for its prototype")
    prototypes=(sums/counts[:,None]).cpu(); config=vars(a).copy(); config.update({"backbone_name":model.backbone_name,
        "num_experts":len(systems),"num_gmm_components":model.gmm_energy.num_components,"best_epoch":best_epoch})
    save_checkpoint(a.output,model,config,extra={"checkpoint_schema_version":2,"hidden_dim":a.hidden_dim,"num_experts":len(systems),
        "system_to_expert":system_to_expert,"source_normal_prototypes":prototypes,"trained_expert_mask":torch.ones(len(systems),dtype=torch.bool),
        "best_epoch":best_epoch,"best_validation_macro_auprc":best_score[0],"best_validation_loss":best_score[1]})
if __name__=="__main__": main()
