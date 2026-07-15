"""Generate class-aware nested few-shot supports for multiple random seeds."""
from __future__ import annotations
import argparse,csv,json,random,sys
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pa_moelog.data import parse_timestamp

def parse_args():
    p=argparse.ArgumentParser(description="Generate S_B nested support sets for each seed.")
    p.add_argument("--input-csv",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--budgets",type=int,nargs="+",default=[10,50,100,500,2000])
    p.add_argument("--seeds",type=int,nargs="+",default=[1,2,3,4,5]); p.add_argument("--sequence",action="store_true")
    p.add_argument("--window-size",type=int,default=20); p.add_argument("--stride",type=int,default=None)
    p.add_argument("--session-field",default="session_id"); p.add_argument("--timestamp-field",default="timestamp")
    return p.parse_args()

def read_samples(path,args):
    with path.open("r",encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle); fields=reader.fieldnames or []; rows=list(reader)
    if not args.sequence:
        return fields,[{"id":index,"rows":[row],"label":int(float(row["label"]))} for index,row in enumerate(rows)]
    if args.timestamp_field not in fields: raise ValueError("sequence support generation requires timestamp")
    sessions=[(row.get(args.session_field) or "").strip() for row in rows]
    if any(sessions) and not all(sessions): raise ValueError("session ids must be all present or all absent")
    samples=[]
    if all(sessions) and sessions:
        groups=defaultdict(list)
        for index,row in enumerate(rows):
            groups[((row.get("system") or "unknown"),sessions[index])].append((parse_timestamp(row[args.timestamp_field],path,index+2),row))
        for sample_id,items in enumerate(groups.values()):
            ordered=[row for _,row in sorted(items,key=lambda item:item[0])]
            samples.append({"id":sample_id,"rows":ordered,"label":max(int(float(row["label"])) for row in ordered)})
    else:
        stride=args.stride or args.window_size
        if stride!=args.window_size: print("[support][warning] overlapping support windows reduce effective independent budget")
        groups=defaultdict(list)
        for index,row in enumerate(rows): groups[row.get("system") or "unknown"].append((parse_timestamp(row[args.timestamp_field],path,index+2),row))
        sample_id=0
        for items in groups.values():
            ordered=[row for _,row in sorted(items,key=lambda item:item[0])]
            for start in range(0,len(ordered),stride):
                window=ordered[start:start+args.window_size]
                if len(window)<args.window_size: continue
                samples.append({"id":sample_id,"rows":window,"label":max(int(float(row["label"])) for row in window)})
                sample_id+=1
    return fields,samples

def nested_order(samples,seed):
    rng=random.Random(seed); queues={label:[sample for sample in samples if sample["label"]==label] for label in (0,1)}
    for queue in queues.values(): rng.shuffle(queue)
    positive_ratio=len(queues[1])/max(len(samples),1); selected=[]; counts={0:0,1:0}
    while len(selected)<len(samples):
        target_positive=round((len(selected)+1)*positive_ratio)
        label=1 if queues[1] and (counts[1]<target_positive or not queues[0]) else 0
        if not queues[label]: label=1-label
        selected.append(queues[label].pop()); counts[label]+=1
    return selected

def write_support(path,fields,samples,args,seed):
    output_fields=list(fields)
    if args.sequence and args.session_field not in output_fields: output_fields.append(args.session_field)
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=output_fields); writer.writeheader()
        for sample in samples:
            session=f"support-seed{seed}-sample{sample['id']}"
            for row in sample["rows"]:
                item={key:row.get(key,"") for key in output_fields}
                if args.sequence: item[args.session_field]=session
                writer.writerow(item)

def main():
    args=parse_args(); path=Path(args.input_csv); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    budgets=sorted(set(args.budgets))
    if not budgets or budgets[0]<=0: raise ValueError("budgets must be positive")
    fields,samples=read_samples(path,args)
    if budgets[-1]>len(samples): raise ValueError(f"largest budget {budgets[-1]} exceeds available samples {len(samples)}")
    manifest={"input":str(path),"sequence":args.sequence,"budgets":budgets,"seeds":{}}
    for seed in args.seeds:
        ordered=nested_order(samples,seed); seed_dir=output/f"seed_{seed}"; seed_dir.mkdir(parents=True,exist_ok=True)
        manifest["seeds"][str(seed)]={}
        for budget in budgets:
            chosen=ordered[:budget]; write_support(seed_dir/f"support_B{budget}.csv",fields,chosen,args,seed)
            manifest["seeds"][str(seed)][str(budget)]=[sample["id"] for sample in chosen]
    (output/"nested_support_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"[support] generated {len(args.seeds)} nested families from {len(samples)} samples")
if __name__=="__main__": main()
