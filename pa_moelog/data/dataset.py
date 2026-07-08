"""CSV dataset utilities for PA-MoELog experiments."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from .preprocess import LogPreprocessor


class LogDataset(Dataset):
    """Read raw log anomaly detection samples from a CSV file."""

    def __init__(
        self,
        csv_path: str | Path,
        log_field: str = "log",
        label_field: str = "label",
        system_field: str = "system",
        default_system: str = "unknown",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.log_field = log_field
        self.label_field = label_field
        self.system_field = system_field
        self.default_system = default_system
        self.rows = self._read_rows()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "raw_log": row["raw_log"],
            "label": row["label"],
            "system": row["system"],
        }

    def _read_rows(self) -> list[dict[str, Any]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        rows: list[dict[str, Any]] = []
        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"CSV file has no header: {self.csv_path}")
            missing = [name for name in (self.log_field, self.label_field) if name not in reader.fieldnames]
            if missing:
                raise ValueError(f"CSV missing required field(s): {missing}")

            for line_number, row in enumerate(reader, start=2):
                raw_log = (row.get(self.log_field) or "").strip()
                if not raw_log:
                    raise ValueError(f"Empty log at {self.csv_path}:{line_number}")
                label = self._parse_label(row.get(self.label_field), line_number)
                system = (row.get(self.system_field) or self.default_system).strip() or self.default_system
                rows.append({"raw_log": raw_log, "label": label, "system": system})

        if not rows:
            raise ValueError(f"CSV file has no samples: {self.csv_path}")
        return rows

    def _parse_label(self, value: Any, line_number: int) -> int:
        try:
            label = int(float(str(value).strip()))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid label at {self.csv_path}:{line_number}: {value!r}") from exc
        if label not in (0, 1):
            raise ValueError(f"Label must be 0 or 1 at {self.csv_path}:{line_number}: {label}")
        return label


def collate_fn(batch: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Collate raw logs and run parser-free preprocessing."""

    items = list(batch)
    raw_logs = [item["raw_log"] for item in items]
    labels = torch.tensor([item["label"] for item in items], dtype=torch.float32)
    systems = [item["system"] for item in items]

    parsed = LogPreprocessor().parse_sequence(raw_logs)
    return {
        "raw_logs": raw_logs,
        "semantic_texts": parsed["semantic_texts"],
        "parameters": parsed["parameters"],
        "labels": labels,
        "systems": systems,
    }
