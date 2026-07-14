"""Event and leakage-safe chronological sequence datasets."""
from __future__ import annotations
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import torch
from torch.utils.data import Dataset
from .preprocess import LogPreprocessor

def _label(value:Any,path:Path,line:int)->int:
    try: result=int(float(str(value).strip()))
    except (TypeError,ValueError) as exc: raise ValueError(f"Invalid label at {path}:{line}: {value!r}") from exc
    if result not in (0,1): raise ValueError(f"Label must be 0 or 1 at {path}:{line}")
    return result

def parse_timestamp(value:Any,path:Path,line:int)->float:
    text=str(value or "").strip()
    if not text: raise ValueError(f"Missing timestamp at {path}:{line}")
    try: return float(text)
    except ValueError: pass
    try:
        parsed=datetime.fromisoformat(text.replace("Z","+00:00"))
        if parsed.tzinfo is None: parsed=parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except ValueError as exc: raise ValueError(f"Invalid timestamp at {path}:{line}: {text!r}") from exc

class LogDataset(Dataset):
    def __init__(self,csv_path,log_field="log",label_field="label",system_field="system",default_system="unknown"):
        self.csv_path=Path(csv_path); self.rows=[]
        with self.csv_path.open("r",encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle)
            if not reader.fieldnames or log_field not in reader.fieldnames or label_field not in reader.fieldnames:
                raise ValueError(f"CSV must contain {log_field!r} and {label_field!r}")
            for line,row in enumerate(reader,2):
                raw=(row.get(log_field) or "").strip()
                if not raw: raise ValueError(f"Empty log at {self.csv_path}:{line}")
                self.rows.append({"raw_log":raw,"label":_label(row.get(label_field),self.csv_path,line),
                    "system":(row.get(system_field) or default_system).strip() or default_system})
        if not self.rows: raise ValueError(f"CSV file has no samples: {self.csv_path}")
    def __len__(self): return len(self.rows)
    def __getitem__(self,index): return self.rows[index]

class LogSequenceDataset(Dataset):
    """Use whole sessions, or full chronological windows when sessions are absent."""
    def __init__(self,csv_path,window_size=20,stride=None,session_field="session_id",timestamp_field="timestamp",
                 label_mode="any",default_system="unknown",drop_incomplete=True):
        self.csv_path=Path(csv_path); self.window_size=window_size; self.stride=stride or window_size
        if window_size<1 or self.stride<1: raise ValueError("window_size and stride must be positive")
        with self.csv_path.open("r",encoding="utf-8-sig",newline="") as handle:
            reader=csv.DictReader(handle); rows=list(reader); fields=reader.fieldnames or []
        if timestamp_field not in fields: raise ValueError(f"sequence CSV must contain {timestamp_field!r}")
        session_values=[(row.get(session_field) or "").strip() for row in rows]
        has_sessions=any(session_values)
        if has_sessions and not all(session_values):
            raise ValueError("session_id must be present for every row or absent for every row")
        groups=defaultdict(list)
        for index,row in enumerate(rows):
            line=index+2; raw=(row.get("log") or "").strip()
            if not raw: raise ValueError(f"Empty log at {self.csv_path}:{line}")
            system=(row.get("system") or default_system).strip() or default_system
            session=session_values[index] if has_sessions else None
            groups[(system,session)].append({"raw_log":raw,"label":_label(row.get("label"),self.csv_path,line),
                "system":system,"session_id":session,"timestamp":parse_timestamp(row.get(timestamp_field),self.csv_path,line)})
        self.sequences=[]
        for (system,session),items in groups.items():
            items.sort(key=lambda item:item["timestamp"])
            if has_sessions:
                windows=[items]
            else:
                windows=[]
                for start in range(0,len(items),self.stride):
                    window=items[start:start+window_size]
                    if len(window)<window_size and drop_incomplete: continue
                    if window: windows.append(window)
            for window_index,window in enumerate(windows):
                labels=[item["label"] for item in window]
                self.sequences.append({"raw_logs":[item["raw_log"] for item in window],
                    "timestamps":[item["timestamp"] for item in window],"label":max(labels) if label_mode=="any" else labels[-1],
                    "system":system,"session_id":session,"sequence_id":session or f"{system}:window:{window_index}"})
        self.sequences.sort(key=lambda sequence:sequence["timestamps"][0])
        if not self.sequences: raise ValueError("no complete sequences were created; reduce window_size or keep incomplete windows")
    def __len__(self): return len(self.sequences)
    def __getitem__(self,index): return self.sequences[index]

def collate_fn(batch:Iterable[dict[str,Any]])->dict[str,Any]:
    items=list(batch); preprocessor=LogPreprocessor()
    raw_sequences=[item["raw_logs"] if "raw_logs" in item else [item["raw_log"]] for item in items]
    parsed=[preprocessor.parse_sequence(sequence) for sequence in raw_sequences]
    max_events=max(map(len,raw_sequences)); event_mask=torch.zeros((len(items),max_events),dtype=torch.bool)
    for index,sequence in enumerate(raw_sequences): event_mask[index,:len(sequence)]=True
    return {"raw_logs":[sequence[0] if len(sequence)==1 else sequence for sequence in raw_sequences],
        "semantic_texts":[item["semantic_texts"] for item in parsed],"parameters":[item["parameters"] for item in parsed],
        "event_mask":event_mask,"labels":torch.tensor([item["label"] for item in items],dtype=torch.float32),
        "systems":[item["system"] for item in items],"sequence_ids":[item.get("sequence_id") for item in items]}
