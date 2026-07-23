"""Prepare leakage-safe Loghub cross-domain data for PA-MoELog.

Sources are HDFS, BGL and Hadoop; OpenStack is the few-shot target.  The
conversion deliberately preserves the native ground-truth unit of each data
set.  Caps select whole parent units and are intended to make exploratory runs
feasible without changing test membership after looking at model results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


FIELDS = ["log", "label", "system", "session_id", "timestamp", "parent_id"]
HDFS_LINE = re.compile(
    r"^(?P<date>\d{6})\s+(?P<time>\d{6})\s+\d+\s+"
    r"(?P<level>\S+)\s+(?P<component>[^:]+):\s*(?P<message>.*)$"
)
BLOCK_ID = re.compile(r"blk_-?\d+")
HADOOP_LINE = re.compile(
    r"^(?P<stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+"
    r"(?P<level>\S+)\s+(?P<message>.*)$"
)
OPENSTACK_LINE = re.compile(
    r"^\S+\s+(?P<stamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"\d+\s+(?P<level>TRACE|DEBUG|INFO|AUDIT|WARNING|WARN|ERROR|CRITICAL)\s+(?P<message>.*)$"
)
INSTANCE_ID = re.compile(r"\[instance:\s*([0-9a-fA-F-]{36})\]")
UUID = re.compile(r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


def _stable_int(seed: int, value: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\0{value}".encode()).digest()[:8], "big")


def _timestamp(text: str, fmt: str) -> float:
    return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()


def _write_csv(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
            count += 1
    return count


def _assignment_summary(assignment: dict[str, str]) -> dict:
    """Compact, reproducible audit record for potentially large parent lists."""
    result = {}
    for split in sorted(set(assignment.values())):
        parents = sorted(parent for parent, part in assignment.items() if part == split)
        digest = hashlib.sha256("\n".join(parents).encode()).hexdigest()
        result[split] = {"parents": len(parents), "parent_ids_sha256": digest}
    return result


def _allocate_cap(counts: dict[int, int], cap: int) -> dict[int, int]:
    total = sum(counts.values())
    if cap <= 0 or cap >= total:
        return dict(counts)
    exact = {label: cap * count / total for label, count in counts.items()}
    result = {label: min(counts[label], math.floor(value)) for label, value in exact.items()}
    if cap >= len([n for n in counts.values() if n]):
        for label, count in counts.items():
            if count and result[label] == 0:
                result[label] = 1
    while sum(result.values()) < cap:
        candidates = [label for label in counts if result[label] < counts[label]]
        label = max(candidates, key=lambda x: (exact[x] - result[x], counts[x], -x))
        result[label] += 1
    while sum(result.values()) > cap:
        candidates = [label for label in counts if result[label] > 1]
        label = min(candidates, key=lambda x: (exact[x] - result[x], counts[x], -x))
        result[label] -= 1
    return result


def stratified_cap(labels: dict[str, int], cap: int, seed: int) -> dict[str, int]:
    by_label: dict[int, list[str]] = defaultdict(list)
    for unit, label in labels.items():
        by_label[label].append(unit)
    allocation = _allocate_cap({label: len(items) for label, items in by_label.items()}, cap)
    selected: dict[str, int] = {}
    for label, items in by_label.items():
        items.sort(key=lambda x: (_stable_int(seed, x), x))
        for unit in items[: allocation[label]]:
            selected[unit] = label
    return selected


def stratified_split(labels: dict[str, int], validation_ratio: float, seed: int) -> dict[str, str]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    by_label: dict[int, list[str]] = defaultdict(list)
    for unit, label in labels.items():
        by_label[label].append(unit)
    assignment: dict[str, str] = {}
    for label, items in by_label.items():
        items.sort(key=lambda x: (_stable_int(seed, x), x))
        n_val = round(len(items) * validation_ratio)
        if len(items) > 1:
            n_val = min(max(n_val, 1), len(items) - 1)
        for unit in items[:n_val]:
            assignment[unit] = "validation"
        for unit in items[n_val:]:
            assignment[unit] = "train"
    return assignment


def parse_hdfs_line(line: str) -> tuple[float, str, list[str]] | None:
    match = HDFS_LINE.match(line.rstrip("\r\n"))
    if not match:
        return None
    blocks = list(dict.fromkeys(BLOCK_ID.findall(match.group("message"))))
    if not blocks:
        return None
    stamp = _timestamp(match.group("date") + " " + match.group("time"), "%y%m%d %H%M%S")
    log = f"{match.group('level')} {match.group('component')} {match.group('message')}"
    return stamp, log, blocks


def load_hdfs_labels(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {
            row["BlockId"].strip(): int(row["Label"].strip().lower() == "anomaly")
            for row in csv.DictReader(handle)
        }


def prepare_hdfs(root: Path, cap: int, validation_ratio: float, seed: int):
    labels = stratified_cap(load_hdfs_labels(root / "preprocessed" / "anomaly_label.csv"), cap, seed)
    assignment = stratified_split(labels, validation_ratio, seed + 1)
    events: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    malformed = 0
    with (root / "HDFS.log").open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            parsed = parse_hdfs_line(line)
            if parsed is None:
                malformed += 1
                continue
            stamp, log, blocks = parsed
            for block in blocks:
                if block in labels:
                    events[block].append((stamp, index, log))
    rows = {"train": [], "validation": []}
    missing = []
    for block in sorted(labels):
        if not events[block]:
            missing.append(block)
            continue
        split = assignment[block]
        for stamp, _, log in sorted(events[block]):
            rows[split].append({"log": log, "label": labels[block], "system": "HDFS",
                                "session_id": f"HDFS:{block}", "timestamp": stamp, "parent_id": block})
    return rows, {"selected_parents": len(labels), "split_assignment": _assignment_summary(assignment),
                  "missing_parents": missing, "ignored_lines": malformed}


def parse_bgl_line(line: str) -> tuple[float, int, str] | None:
    parts = line.rstrip("\r\n").split(None, 9)
    if len(parts) < 10:
        return None
    try:
        stamp = float(parts[1])
    except ValueError:
        return None
    label = int(parts[0] != "-")
    return stamp, label, " ".join(parts[6:])


class BottomK:
    """Deterministic bounded selection without retaining every BGL window."""
    def __init__(self, cap: int, seed: int):
        self.cap, self.seed, self.heap = cap, seed, []

    def add(self, key: str, value):
        rank = _stable_int(self.seed, key)
        item = (-rank, key, value)
        if self.cap <= 0:
            self.heap.append(item)
        elif len(self.heap) < self.cap:
            heapq.heappush(self.heap, item)
        elif item > self.heap[0]:
            heapq.heapreplace(self.heap, item)

    def values(self):
        return [item[2] for item in sorted(self.heap, key=lambda x: (-x[0], x[1]))]


def prepare_bgl(root: Path, window_size: int, cap_per_split: int,
                validation_ratio: float, seed: int):
    path = root / "BGL.log"
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        total = sum(1 for _ in handle)
    boundary = math.floor(total * (1 - validation_ratio))
    collectors = {name: BottomK(cap_per_split, seed + i) for i, name in enumerate(("train", "validation"))}
    buffers = {"train": [], "validation": []}
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for index, line in enumerate(handle):
            split = "train" if index < boundary else "validation"
            parsed = parse_bgl_line(line)
            if parsed is None:
                malformed += 1
                continue
            buffers[split].append(parsed)
            if len(buffers[split]) == window_size:
                window = buffers[split]
                parent = f"BGL:{split}:{index - window_size + 1}"
                collectors[split].add(parent, (parent, window))
                buffers[split] = []
    result = {"train": [], "validation": []}
    selected_windows = {}
    for split, collector in collectors.items():
        chosen = collector.values()
        selected_windows[split] = [parent for parent, _ in chosen]
        for parent, window in chosen:
            label = max(item[1] for item in window)
            for stamp, _, log in window:
                result[split].append({"log": log, "label": label, "system": "BGL",
                                      "session_id": parent, "timestamp": stamp, "parent_id": parent})
    window_assignment = {parent: split for split, parents in selected_windows.items() for parent in parents}
    return result, {"raw_lines": total, "boundary_line": boundary, "ignored_lines": malformed,
                    "window_size": window_size,
                    "split_assignment": _assignment_summary(window_assignment)}


def load_hadoop_labels(path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    mode: int | None = None
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if line == "Normal:":
            mode = 0
        elif line.endswith(":") and not line.startswith("###"):
            mode = 1
        match = re.match(r"\+\s+(application_\d+_\d+)$", line)
        if match and mode is not None:
            labels[match.group(1)] = mode
    return labels


def parse_hadoop_file(path: Path) -> list[tuple[float, str]]:
    events: list[list] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            match = HADOOP_LINE.match(line)
            if match:
                stamp = _timestamp(match.group("stamp"), "%Y-%m-%d %H:%M:%S,%f")
                events.append([stamp, f"{match.group('level')} {match.group('message')}"])
            elif line and events:
                events[-1][1] += " " + line.strip()
    return [(stamp, log) for stamp, log in events]


def _evenly_spaced(items: list, limit: int) -> list:
    """Keep deterministic coverage across a long parent without duplicating chunks."""
    if limit <= 0 or len(items) <= limit:
        return items
    if limit == 1:
        return [items[len(items) // 2]]
    indices = [round(index * (len(items) - 1) / (limit - 1)) for index in range(limit)]
    return [items[index] for index in indices]


def prepare_hadoop(root: Path, chunk_size: int, cap: int, validation_ratio: float, seed: int,
                   max_chunks_per_application: int = 0):
    labels = load_hadoop_labels(root / "abnormal_label.txt")
    available = {path.name for path in root.glob("application_*") if path.is_dir()}
    labels = {app: label for app, label in labels.items() if app in available}
    labels = stratified_cap(labels, cap, seed)
    assignment = stratified_split(labels, validation_ratio, seed + 1)
    result = {"train": [], "validation": []}
    chunk_counts: dict[str, int] = {}
    for app in sorted(labels):
        events = []
        for path in sorted((root / app).glob("*.log")):
            events.extend(parse_hadoop_file(path))
        events.sort(key=lambda x: x[0])
        chunks = [events[start:start + chunk_size] for start in range(0, len(events), chunk_size)]
        chunks = _evenly_spaced(chunks, max_chunks_per_application)
        chunk_counts[app] = len(chunks)
        split = assignment[app]
        for chunk_index, chunk in enumerate(chunks):
            session = f"Hadoop:{app}:chunk:{chunk_index}"
            for stamp, log in chunk:
                result[split].append({"log": log, "label": labels[app], "system": "Hadoop",
                                      "session_id": session, "timestamp": stamp, "parent_id": app})
    return result, {"selected_parents": len(labels), "split_assignment": _assignment_summary(assignment),
                    "chunk_size": chunk_size, "max_chunks_per_application": max_chunks_per_application,
                    "chunks": chunk_counts}


def load_openstack_anomalies(path: Path) -> set[str]:
    return {match.group(0).lower() for match in UUID.finditer(path.read_text(encoding="utf-8", errors="replace"))}


def prepare_openstack(root: Path, output: Path, normal_cap: int, seed: int,
                      support_normal_ratio: float, validation_normal_ratio: float,
                      window_size: int = 0):
    anomalies = load_openstack_anomalies(root / "anomaly_labels.txt")
    events: dict[str, list[tuple[float, int, str]]] = defaultdict(list)
    ignored = malformed = 0
    for path in sorted(root.glob("*.log")):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for index, line in enumerate(handle):
                instance = INSTANCE_ID.search(line)
                if not instance:
                    ignored += 1
                    continue
                parsed = OPENSTACK_LINE.match(line.rstrip("\r\n"))
                if not parsed:
                    malformed += 1
                    continue
                instance_id = instance.group(1).lower()
                stamp = _timestamp(parsed.group("stamp"), "%Y-%m-%d %H:%M:%S.%f")
                events[instance_id].append((stamp, index, f"{parsed.group('level')} {parsed.group('message')}"))
    anomaly_ids = sorted(anomalies & events.keys(), key=lambda x: (_stable_int(seed, x), x))
    if len(anomaly_ids) < 3:
        raise ValueError("OpenStack needs at least three labeled anomaly instances for support/validation/test folds")
    normal_labels = {instance: 0 for instance in events if instance not in anomalies}
    normal_ids = list(stratified_cap(normal_labels, normal_cap, seed + 1))
    normal_ids.sort(key=lambda x: (_stable_int(seed + 2, x), x))
    n_support = math.floor(len(normal_ids) * support_normal_ratio)
    n_validation = math.floor(len(normal_ids) * validation_normal_ratio)
    normal_parts = {
        "support": normal_ids[:n_support],
        "validation": normal_ids[n_support:n_support + n_validation],
        "test": normal_ids[n_support + n_validation:],
    }
    manifests = []
    for fold, test_anomaly in enumerate(anomaly_ids):
        remaining = [item for item in anomaly_ids if item != test_anomaly]
        validation_anomaly = remaining[fold % len(remaining)]
        anomaly_parts = {
            "support": [item for item in remaining if item != validation_anomaly],
            "validation": [validation_anomaly],
            "test": [test_anomaly],
        }
        fold_dir = output / f"openstack_fold_{fold}"
        manifest = {"fold": fold, "seed": seed, "anomaly_instances": anomaly_parts,
                    "normal_instances": normal_parts,
                    "normal_counts": {name: len(ids) for name, ids in normal_parts.items()}}
        for split in ("support", "validation", "test"):
            ids = normal_parts[split] + anomaly_parts[split]
            rows = []
            for instance in ids:
                label = int(instance in anomalies)
                ordered = sorted(events[instance])
                windows = ([ordered] if window_size <= 0 else
                           [ordered[start:start + window_size]
                            for start in range(0, len(ordered), window_size)])
                for window_index, window in enumerate(windows):
                    session = f"OpenStack:{instance}"
                    if window_size > 0:
                        session += f":window:{window_index}"
                    for stamp, _, log in window:
                        rows.append({"log": log, "label": label, "system": "OpenStack",
                                     "session_id": session, "timestamp": stamp,
                                     "parent_id": instance})
            _write_csv(fold_dir / f"{split}.csv", rows)
        (fold_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        manifests.append(manifest)
    return manifests, {"instances": len(events), "normal_instances": len(normal_ids),
                       "anomaly_instances": len(anomaly_ids), "ignored_unassigned_lines": ignored,
                       "malformed_assigned_lines": malformed, "window_size": window_size}


def parse_args(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-validation-ratio", type=float, default=0.1)
    parser.add_argument("--hdfs-cap", type=int, default=20000, help="Whole blocks; 0 means unlimited")
    parser.add_argument("--bgl-cap-per-split", type=int, default=20000, help="Whole windows; 0 means unlimited")
    parser.add_argument("--bgl-window-size", type=int, default=20)
    parser.add_argument("--hadoop-cap", type=int, default=55, help="Whole applications; 0 means unlimited")
    parser.add_argument("--hadoop-chunk-size", type=int, default=256)
    parser.add_argument("--hadoop-max-chunks-per-application", type=int, default=0,
                        help="Deterministic evenly spaced chunks per application; 0 means unlimited")
    parser.add_argument("--openstack-normal-cap", type=int, default=0, help="Whole VM instances; 0 means unlimited")
    parser.add_argument("--openstack-window-size", type=int, default=0,
                        help="Non-overlapping events per VM window; 0 keeps each VM whole")
    parser.add_argument("--support-normal-ratio", type=float, default=0.2)
    parser.add_argument("--validation-normal-ratio", type=float, default=0.2)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    if args.bgl_window_size < 1 or args.hadoop_chunk_size < 1:
        raise ValueError("window and chunk sizes must be positive")
    if args.support_normal_ratio < 0 or args.validation_normal_ratio < 0 or \
            args.support_normal_ratio + args.validation_normal_ratio >= 1:
        raise ValueError("OpenStack normal ratios must be non-negative and sum to less than one")
    raw, output = args.raw_root, args.output_dir
    hdfs, hdfs_meta = prepare_hdfs(raw / "HDFS_v1", args.hdfs_cap,
                                   args.source_validation_ratio, args.seed)
    bgl, bgl_meta = prepare_bgl(raw / "BGL", args.bgl_window_size, args.bgl_cap_per_split,
                               args.source_validation_ratio, args.seed + 10)
    hadoop, hadoop_meta = prepare_hadoop(raw / "Hadoop", args.hadoop_chunk_size, args.hadoop_cap,
        args.source_validation_ratio, args.seed + 20, args.hadoop_max_chunks_per_application)
    source_counts = {}
    for split in ("train", "validation"):
        rows = hdfs[split] + bgl[split] + hadoop[split]
        rows.sort(key=lambda row: (row["system"], row["timestamp"], row["session_id"]))
        source_counts[split] = _write_csv(output / f"source_{split}.csv", rows)
    folds, openstack_meta = prepare_openstack(
        raw / "OpenStack", output, args.openstack_normal_cap, args.seed + 30,
        args.support_normal_ratio, args.validation_normal_ratio, args.openstack_window_size,
    )
    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_event_counts": source_counts,
        "datasets": {"HDFS": hdfs_meta, "BGL": bgl_meta, "Hadoop": hadoop_meta,
                     "OpenStack": openstack_meta},
        "openstack_folds": folds,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "crossdomain_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"source_event_counts": source_counts, "openstack_folds": len(folds)}))


if __name__ == "__main__":
    main()
