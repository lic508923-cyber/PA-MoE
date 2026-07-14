"""Strict chronological split before session/window construction."""
from __future__ import annotations
import argparse,csv,json,sys
import math
from collections import defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pa_moelog.data import parse_timestamp

def parse_args():
    p=argparse.ArgumentParser(description="Split raw events chronologically without session/window leakage.")
    p.add_argument("--input-csv",required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--train-ratio",type=float,default=.6); p.add_argument("--support-ratio",type=float,default=.1)
    p.add_argument("--validation-ratio",type=float,default=.1); p.add_argument("--session-field",default="session_id")
    p.add_argument("--timestamp-field",default="timestamp"); p.add_argument("--window-size",type=int,default=20)
    p.add_argument("--stride",type=int,default=None); p.add_argument("--keep-incomplete",action="store_true")
    return p.parse_args()

def write_rows(path,fields,items):
    with path.open("w",encoding="utf-8",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for row in items: writer.writerow({key:row.get(key,"") for key in fields})

def build_split_sequences(name,items,fields,output,args,has_sessions):
    sequence_fields=list(fields)
    if args.session_field not in sequence_fields: sequence_fields.append(args.session_field)
    emitted=[]; sequence_count=0
    if has_sessions:
        emitted=items; sequence_count=len({row.get(args.session_field) for row in items})
    else:
        by_system=defaultdict(list)
        for row in items: by_system[(row.get("system") or "unknown")].append(row)
        stride=args.stride or args.window_size
        for system,events in by_system.items():
            events.sort(key=lambda row:row["__timestamp__"])
            for start in range(0,len(events),stride):
                window=events[start:start+args.window_size]
                if len(window)<args.window_size and not args.keep_incomplete: continue
                if not window: continue
                sequence_id=f"{name}:{system}:window:{sequence_count}"
                for row in window:
                    copied=dict(row); copied[args.session_field]=sequence_id; emitted.append(copied)
                sequence_count+=1
    write_rows(output/f"{name}_sequences.csv",sequence_fields,emitted)
    return sequence_count

def main():
    a=parse_args(); path=Path(a.input_csv); ratios=[a.train_ratio,a.support_ratio,a.validation_ratio]
    if any(x<0 for x in ratios) or sum(ratios)>=1: raise ValueError("ratios must be non-negative and sum to less than 1")
    with path.open("r",encoding="utf-8-sig",newline="") as handle:
        reader=csv.DictReader(handle); fields=reader.fieldnames or []; rows=list(reader)
    if a.timestamp_field not in fields: raise ValueError(f"CSV must contain {a.timestamp_field!r}")
    sessions=[(row.get(a.session_field) or "").strip() for row in rows]
    has_sessions=any(sessions)
    if has_sessions and not all(sessions): raise ValueError("session ids must be complete before strict splitting")
    units=defaultdict(list)
    for index,row in enumerate(rows):
        system=(row.get("system") or "unknown").strip() or "unknown"
        timestamp=parse_timestamp(row.get(a.timestamp_field),path,index+2)
        row=dict(row); row["__timestamp__"]=timestamp; row["__index__"]=index
        unit=sessions[index] if has_sessions else f"row:{index}"
        units[(system,unit)].append(row)
    per_system=defaultdict(list)
    for (system,unit),items in units.items():
        per_system[system].append((min(x["__timestamp__"] for x in items),max(x["__timestamp__"] for x in items),unit,items))
    split_rows={name:[] for name in ("train","support","validation","test")}; manifest={"systems":{},"session_mode":has_sessions}
    for system,system_units in per_system.items():
        system_units.sort(key=lambda x:(x[0],x[2])); n=len(system_units)
        all_ratios=[a.train_ratio,a.support_ratio,a.validation_ratio,1-sum(ratios)]
        exact=[n*ratio for ratio in all_ratios]; counts=[math.floor(value) for value in exact]
        for index in sorted(range(4),key=lambda i:(exact[i]-counts[i],all_ratios[i]),reverse=True)[:n-sum(counts)]: counts[index]+=1
        cuts=[counts[0],counts[0]+counts[1],sum(counts[:3])]
        buckets=[system_units[:cuts[0]],system_units[cuts[0]:cuts[1]],system_units[cuts[1]:cuts[2]],system_units[cuts[2]:]]
        nonempty=[bucket for bucket in buckets if bucket]
        for earlier,later in zip(nonempty,nonempty[1:]):
            if max(unit[1] for unit in earlier) >= min(unit[0] for unit in later):
                raise ValueError(f"overlapping sessions prevent a strict chronological split for system {system!r}")
        manifest["systems"][system]={}
        for name,bucket in zip(split_rows,buckets):
            flat=[row for _,_,_,items in bucket for row in items]; split_rows[name].extend(flat)
            manifest["systems"][system][name]={"units":len(bucket),"events":len(flat),
                "start":min((x[0] for x in bucket),default=None),"end":max((x[0] for x in bucket),default=None)}
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    for name,items in split_rows.items():
        items.sort(key=lambda row:(row["__timestamp__"],row["__index__"]))
        write_rows(out/f"{name}.csv",fields,items)
        manifest.setdefault("sequence_counts",{})[name]=build_split_sequences(name,items,fields,out,a,has_sessions)
    (out/"split_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps({name:len(items) for name,items in split_rows.items()},ensure_ascii=False))
if __name__=="__main__": main()
