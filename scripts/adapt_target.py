"""Few-shot adaptation with automatic support-guided fusion and post-fit GMM."""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pa_moelog.data import LogDataset, LogSequenceDataset, collate_fn
from pa_moelog.models import PAMoELog
from pa_moelog.models.dora import DoRALinear
from pa_moelog.utils import load_checkpoint, restore_checkpoint, save_checkpoint, select_best_f1_threshold

def parse_args():
    p=argparse.ArgumentParser(description="Few-shot target adaptation for PA-MoELog.")
    p.add_argument("--support-csv",required=True); p.add_argument("--validation-csv",default=None)
    p.add_argument("--base-checkpoint",required=True); p.add_argument("--target-system",required=True)
    p.add_argument("--batch-size",type=int,default=32); p.add_argument("--epochs",type=int,default=5)
    p.add_argument("--lr",type=float,default=5e-5); p.add_argument("--fusion-temperature",type=float,default=1.0)
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--fusion-shrinkage",type=float,default=None)
    p.add_argument("--fusion",choices=["uniform","support-guided"],default="support-guided")
    p.add_argument("--adaptation",choices=["head-only","dora","full"],default="dora")
    p.add_argument("--disable-parameters",action=argparse.BooleanOptionalAction,default=None)
    p.add_argument("--disable-gmm",action=argparse.BooleanOptionalAction,default=None)
    p.add_argument("--output-dir",default="artifacts/checkpoints/target_adapt"); p.add_argument("--device",default="cpu")
    p.add_argument("--backbone-name",default=None); p.add_argument("--debug-hash-encoder",action="store_true")
    p.add_argument("--sequence",action="store_true"); p.add_argument("--window-size",type=int,default=20); p.add_argument("--stride",type=int,default=None)
    return p.parse_args()

def dataset(path,a): return LogSequenceDataset(path,a.window_size,a.stride) if a.sequence else LogDataset(path,default_system=a.target_system)

def load_model(path,device,backbone=None,allow=False,checkpoint=None):
    ck=checkpoint or load_checkpoint(path,map_location=device); cfg=ck.get("config",{})
    model=PAMoELog(hidden_dim=int(ck.get("hidden_dim",cfg.get("hidden_dim",128))),
        num_experts=int(ck.get("num_experts",cfg.get("num_experts",3))),
        num_gmm_components=int(cfg.get("num_gmm_components",4)),
        backbone_name=backbone or str(cfg.get("backbone_name","bert-base-uncased")),allow_hash_fallback=allow,
        max_events=int(cfg.get("max_events",512)),gmm_projection_dim=int(cfg.get("gmm_projection_dim",32)),
        fusion_shrinkage_strength=float(cfg.get("fusion_shrinkage_strength",16.0)),
        sequence_layers=int(cfg.get("sequence_layers",1)),dora_rank=int(cfg.get("dora_rank",4)),
        disable_parameters=bool(cfg.get("disable_parameters",False)),disable_gmm=bool(cfg.get("disable_gmm",False))).to(device)
    restore_checkpoint(ck,model,strict=True)
    if "trained_expert_mask" in ck: model.fusion.set_trained_mask(ck["trained_expert_mask"].to(device))
    return model,ck

def set_adaptation_trainable(model,mode):
    for p in model.parameters(): p.requires_grad=False
    for p in model.target_classifier.parameters(): p.requires_grad=True
    if mode in {"dora","full"}:
        for p in model.target_norm.parameters(): p.requires_grad=True
    if mode=="dora":
        for module in model.target_adapter.modules():
            if isinstance(module,DoRALinear):
                module.lora_a.requires_grad=True; module.lora_b.requires_grad=True; module.magnitude.requires_grad=True
    if mode=="full":
        for p in model.parameters(): p.requires_grad=True

def forward(model,batch,device):
    return model(batch["semantic_texts"],batch["parameters"],batch["event_mask"].to(device))

@torch.no_grad()
def calibrate_fusion(model,loader,prototypes,temperature,device,label_budget):
    sums=torch.zeros_like(prototypes,device=device); count=0; model.eval()
    for batch in loader:
        labels=batch["labels"].to(device); normal=labels==0
        if bool(normal.any()):
            out=forward(model,batch,device); sums+=out["expert_hiddens"][normal].sum(0); count+=int(normal.sum())
    if count==0: raise ValueError("support set needs at least one normal sample for fusion calibration")
    target=F.normalize(sums/count,dim=1); source=F.normalize(prototypes.to(device),dim=1)
    distances=1.0-(target*source).sum(dim=1)
    model.fusion.calibrate_from_distances(distances,temperature,label_budget=label_budget)
    print(f"[adapt] fusion distances={distances.cpu().tolist()} weights={model.fusion.weights.cpu().tolist()}")

@torch.no_grad()
def collect_normal_hidden(model,loader,device):
    result=[]; model.eval()
    for batch in loader:
        labels=batch["labels"].to(device); out=forward(model,batch,device)
        if bool((labels==0).any()): result.append(out["target_hidden"][labels==0])
    return torch.cat(result) if result else None

@torch.no_grad()
def tune_validation(model,loader,device):
    labels=[]; cls=[]; energy=[]
    for batch in loader:
        out=forward(model,batch,device); labels.append(batch["labels"]); cls.append(out["classifier_score"].cpu()); energy.append(out["energy_score"].cpu())
    labels=torch.cat(labels); cls=torch.cat(cls); energy=torch.cat(energy)
    best=(-1.0,0.7,0.3,0.5,0.0,0.0)
    alphas=[1.0] if model.disable_gmm else [i/20 for i in range(21)]
    for alpha in alphas:
        scores=alpha*cls+(1-alpha)*energy
        operating_point=select_best_f1_threshold(labels,scores)
        candidate=(operating_point["f1"],alpha,1-alpha,operating_point["threshold"],
                   operating_point["precision"],operating_point["recall"])
        if candidate[0]>best[0]: best=candidate
    model.alpha,model.beta=best[1],best[2]
    print(f"[adapt] validation f1={best[0]:.4f} precision={best[4]:.4f} recall={best[5]:.4f} "
          f"alpha={best[1]:.2f} beta={best[2]:.2f} threshold={best[3]:.8f}")
    return best[3]

def main():
    a=parse_args(); torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    device=torch.device(a.device); support=dataset(a.support_csv,a)
    loader=DataLoader(support,batch_size=a.batch_size,shuffle=False,collate_fn=collate_fn)
    base_metadata=load_checkpoint(a.base_checkpoint,map_location=device); base_config=base_metadata.get("config",{})
    if bool(base_config.get("sequence",False)) != bool(a.sequence):
        raise ValueError("target adaptation mode must match the source checkpoint sequence mode")
    backbone=a.backbone_name or str(base_config.get("backbone_name","bert-base-uncased"))
    if backbone in {"hash","simple-hash-encoder"} and not a.debug_hash_encoder:
        raise ValueError("hash encoder is debug-only; pass --debug-hash-encoder explicitly")
    model,base=load_model(a.base_checkpoint,device,a.backbone_name,a.debug_hash_encoder,base_metadata)
    source_disable_parameters=bool(base_config.get("disable_parameters",False))
    if a.disable_parameters is not None and a.disable_parameters!=source_disable_parameters:
        raise ValueError("--disable-parameters must match the source checkpoint; retrain the source ablation")
    model.disable_parameters=source_disable_parameters
    model.disable_gmm=bool(base_config.get("disable_gmm",False) if a.disable_gmm is None else a.disable_gmm)
    adaptation_start=time.perf_counter()
    if device.type=="cuda": torch.cuda.reset_peak_memory_stats(device)
    if a.fusion_shrinkage is not None: model.fusion.shrinkage_strength=float(a.fusion_shrinkage)
    prototypes=base.get("source_normal_prototypes")
    if a.fusion=="support-guided":
        if prototypes is None: raise ValueError("support-guided fusion requires source_normal_prototypes")
        calibrate_fusion(model,loader,prototypes,a.fusion_temperature,device,len(support))
    else:
        model.fusion.set_weights(model.fusion.trained_mask.to(model.fusion.weights.dtype))
    set_adaptation_trainable(model,a.adaptation)
    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_parameters=sum(p.numel() for p in model.parameters())
    support_rows=support.sequences if isinstance(support,LogSequenceDataset) else support.rows
    positives=sum(row["label"] for row in support_rows); negatives=len(support_rows)-positives
    pos_weight=(torch.tensor(negatives/positives,device=device) if positives and negatives else None)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr)
    criterion=nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    train_loader=DataLoader(support,batch_size=a.batch_size,shuffle=True,collate_fn=collate_fn)
    for epoch in range(1,a.epochs+1):
        losses=[]; model.train()
        for batch in train_loader:
            labels=batch["labels"].to(device); out=forward(model,batch,device); loss=criterion(out["logit"],labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        print(f"[adapt][{epoch}] bce={sum(losses)/len(losses):.4f}")
    if not model.disable_gmm:
        normal=collect_normal_hidden(model,loader,device)
        if normal is None or normal.size(0)<2: raise ValueError("at least two normal support samples are required to fit GMM")
        model.gmm_energy.fit_normal(normal); model.fit_energy_statistics(normal)
    threshold=0.5
    if a.validation_csv:
        threshold=tune_validation(model,DataLoader(dataset(a.validation_csv,a),batch_size=a.batch_size,shuffle=False,collate_fn=collate_fn),device)
    adaptation_seconds=time.perf_counter()-adaptation_start
    peak_memory_bytes=(torch.cuda.max_memory_allocated(device) if device.type=="cuda" else 0)
    config=vars(a).copy(); config.update({"backbone_name":model.backbone_name,"alpha":model.alpha,"beta":model.beta,"threshold":threshold,
        "num_gmm_components":model.gmm_energy.num_components,"gmm_projection_dim":model.gmm_energy.projection_dim,
        "max_events":model.max_events,"fusion_shrinkage_strength":model.fusion.shrinkage_strength,
        "sequence_layers":model.sequence_layers,"dora_rank":model.target_adapter.rank,
        "disable_parameters":model.disable_parameters,"disable_gmm":model.disable_gmm})
    path=Path(a.output_dir)/f"{a.target_system}_adapted.pt"
    save_checkpoint(path,model,config,extra={"checkpoint_schema_version":2,"target_system":a.target_system,"hidden_dim":model.hidden_dim,"num_experts":model.num_experts,
        "system_to_expert":base.get("system_to_expert"),"source_normal_prototypes":prototypes,"trained_expert_mask":model.fusion.trained_mask.cpu(),"threshold":threshold,
        "trainable_parameters":trainable_parameters,"total_parameters":total_parameters,
        "trainable_parameter_ratio":trainable_parameters/max(total_parameters,1),"adaptation_seconds":adaptation_seconds,
        "peak_memory_bytes":peak_memory_bytes})
    efficiency={"trainable_parameters":trainable_parameters,"total_parameters":total_parameters,
        "trainable_parameter_ratio":trainable_parameters/max(total_parameters,1),"adaptation_seconds":adaptation_seconds,
        "peak_memory_bytes":peak_memory_bytes,"checkpoint_size_bytes":path.stat().st_size}
    (Path(a.output_dir)/f"{a.target_system}_efficiency.json").write_text(json.dumps(efficiency,indent=2),encoding="utf-8")
if __name__=="__main__": main()
