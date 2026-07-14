"""Summarize repeated-seed evaluation JSON files as mean and sample std."""
from __future__ import annotations
import argparse,json,statistics
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description="Report mean/std across 3-5 seeded experiment results.")
    parser.add_argument("results",nargs="+"); parser.add_argument("--output",default=None); args=parser.parse_args()
    if len(args.results)<3: raise ValueError("at least three seeded results are required")
    rows=[json.loads(Path(path).read_text(encoding="utf-8")) for path in args.results]
    metrics={}
    for key in ("precision","recall","f1","auroc","auprc"):
        values=[float(row[key]) for row in rows if key in row]
        if len(values)==len(rows): metrics[key]={"mean":statistics.fmean(values),"std":statistics.stdev(values)}
    result={"num_seeds":len(rows),"metrics":metrics}; rendered=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        path=Path(args.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(rendered,encoding="utf-8")
    print(rendered)
if __name__=="__main__": main()
