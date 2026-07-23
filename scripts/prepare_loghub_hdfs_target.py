"""Prepare BGL + Hadoop + OpenStack sources and an HDFS target split.

Every native parent unit is assigned to exactly one split.  OpenStack uses
non-overlapping windows inside a VM, but all windows from one VM remain in the
same source split.  HDFS is split by Block ID before any event rows are written.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from prepare_loghub_crossdomain import (
    _assignment_summary,
    _stable_int,
    _timestamp,
    _write_csv,
    INSTANCE_ID,
    OPENSTACK_LINE,
    load_hdfs_labels,
    load_openstack_anomalies,
    parse_hdfs_line,
    prepare_bgl,
    prepare_hadoop,
    stratified_cap,
)


def three_way_stratified_split(labels: dict[str, int], support_ratio: float,
                               validation_ratio: float, seed: int) -> dict[str, str]:
    if support_ratio <= 0 or validation_ratio <= 0 or support_ratio + validation_ratio >= 1:
        raise ValueError("support and validation ratios must be positive and sum to less than one")
    by_label: dict[int, list[str]] = defaultdict(list)
    for parent, label in labels.items():
        by_label[label].append(parent)
    assignment: dict[str, str] = {}
    for label, parents in by_label.items():
        parents.sort(key=lambda item: (_stable_int(seed, item), item))
        n_support = max(1, round(len(parents) * support_ratio))
        n_validation = max(1, round(len(parents) * validation_ratio))
        if n_support + n_validation >= len(parents):
            raise ValueError(f"class {label} is too small for a three-way split")
        for parent in parents[:n_support]:
            assignment[parent] = "support"
        for parent in parents[n_support:n_support + n_validation]:
            assignment[parent] = "validation"
        for parent in parents[n_support + n_validation:]:
            assignment[parent] = "test"
    return assignment


def prepare_hdfs_target(root: Path, output: Path, cap: int, support_ratio: float,
                        validation_ratio: float, seed: int):
    labels = stratified_cap(load_hdfs_labels(root / "preprocessed" / "anomaly_label.csv"), cap, seed)
    assignment = three_way_stratified_split(labels, support_ratio, validation_ratio, seed + 1)
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
    counts = {}
    missing = []
    for split in ("support", "validation", "test"):
        rows = []
        for block in sorted(parent for parent, part in assignment.items() if part == split):
            if not events[block]:
                missing.append(block)
                continue
            for stamp, _, log in sorted(events[block]):
                rows.append({"log": log, "label": labels[block], "system": "HDFS",
                             "session_id": f"HDFS:{block}", "timestamp": stamp,
                             "parent_id": block})
        counts[split] = _write_csv(output / "hdfs_target" / f"{split}.csv", rows)
    return counts, {"selected_parents": len(labels), "split_assignment": _assignment_summary(assignment),
                    "missing_parents": missing, "ignored_lines": malformed}


def prepare_openstack_source(root: Path, normal_cap: int, validation_ratio: float,
                             window_size: int, seed: int):
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
                parent = instance.group(1).lower()
                stamp = _timestamp(parsed.group("stamp"), "%Y-%m-%d %H:%M:%S.%f")
                events[parent].append((stamp, index, f"{parsed.group('level')} {parsed.group('message')}"))
    anomaly_ids = sorted(anomalies & events.keys(), key=lambda item: (_stable_int(seed, item), item))
    normal_ids = list(stratified_cap({item: 0 for item in events if item not in anomalies}, normal_cap, seed + 1))
    normal_ids.sort(key=lambda item: (_stable_int(seed + 2, item), item))
    if len(anomaly_ids) < 2:
        raise ValueError("OpenStack needs at least two anomalous VMs for source train/validation")
    normal_cut = max(1, round(len(normal_ids) * validation_ratio))
    anomaly_cut = max(1, round(len(anomaly_ids) * validation_ratio))
    anomaly_cut = min(anomaly_cut, len(anomaly_ids) - 1)
    assignment = {item: "validation" for item in normal_ids[:normal_cut] + anomaly_ids[:anomaly_cut]}
    assignment.update({item: "train" for item in normal_ids[normal_cut:] + anomaly_ids[anomaly_cut:]})
    result = {"train": [], "validation": []}
    for parent, split in assignment.items():
        ordered = sorted(events[parent])
        windows = [ordered] if window_size <= 0 else [
            ordered[start:start + window_size] for start in range(0, len(ordered), window_size)
        ]
        for window_index, window in enumerate(windows):
            session = f"OpenStack:{parent}" + (f":window:{window_index}" if window_size > 0 else "")
            for stamp, _, log in window:
                result[split].append({"log": log, "label": int(parent in anomalies),
                                      "system": "OpenStack", "session_id": session,
                                      "timestamp": stamp, "parent_id": parent})
    return result, {"instances": len(assignment), "normal_instances": len(normal_ids),
                    "anomaly_instances": len(anomaly_ids), "window_size": window_size,
                    "split_assignment": _assignment_summary(assignment),
                    "ignored_unassigned_lines": ignored, "malformed_assigned_lines": malformed}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--source-validation-ratio", type=float, default=0.1)
    parser.add_argument("--bgl-cap-per-split", type=int, default=1000)
    parser.add_argument("--bgl-window-size", type=int, default=20)
    parser.add_argument("--hadoop-cap", type=int, default=55)
    parser.add_argument("--hadoop-chunk-size", type=int, default=32)
    parser.add_argument("--hadoop-max-chunks-per-application", type=int, default=4)
    parser.add_argument("--openstack-normal-cap", type=int, default=500)
    parser.add_argument("--openstack-window-size", type=int, default=5)
    parser.add_argument("--hdfs-cap", type=int, default=10000)
    parser.add_argument("--hdfs-support-ratio", type=float, default=0.4)
    parser.add_argument("--hdfs-validation-ratio", type=float, default=0.2)
    return parser.parse_args()


def main():
    args = parse_args()
    raw, output = args.raw_root, args.output_dir
    bgl, bgl_meta = prepare_bgl(raw / "BGL", args.bgl_window_size, args.bgl_cap_per_split,
                                args.source_validation_ratio, args.seed + 10)
    hadoop, hadoop_meta = prepare_hadoop(raw / "Hadoop", args.hadoop_chunk_size, args.hadoop_cap,
        args.source_validation_ratio, args.seed + 20, args.hadoop_max_chunks_per_application)
    openstack, openstack_meta = prepare_openstack_source(raw / "OpenStack", args.openstack_normal_cap,
        args.source_validation_ratio, args.openstack_window_size, args.seed + 30)
    source_counts = {}
    for split in ("train", "validation"):
        rows = bgl[split] + hadoop[split] + openstack[split]
        rows.sort(key=lambda row: (row["system"], row["timestamp"], row["session_id"]))
        source_counts[split] = _write_csv(output / f"source_{split}.csv", rows)
    target_counts, hdfs_meta = prepare_hdfs_target(raw / "HDFS_v1", output, args.hdfs_cap,
        args.hdfs_support_ratio, args.hdfs_validation_ratio, args.seed + 40)
    manifest = {"schema_version": 1, "seed": args.seed,
        "design": {"sources": ["BGL", "Hadoop", "OpenStack"], "target": "HDFS"},
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "source_event_counts": source_counts, "target_event_counts": target_counts,
        "datasets": {"BGL": bgl_meta, "Hadoop": hadoop_meta,
                     "OpenStack": openstack_meta, "HDFS": hdfs_meta}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "crossdomain_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"source_event_counts": source_counts, "target_event_counts": target_counts}))


if __name__ == "__main__":
    main()
