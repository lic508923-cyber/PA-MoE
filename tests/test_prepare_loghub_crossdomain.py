import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_loghub_crossdomain import (  # noqa: E402
    parse_bgl_line,
    parse_hdfs_line,
    prepare_bgl,
    prepare_hadoop,
    prepare_openstack,
    stratified_cap,
    stratified_split,
)
from prepare_loghub_hdfs_target import three_way_stratified_split  # noqa: E402


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ParserTests(unittest.TestCase):
    def test_three_way_target_split_is_stratified_and_disjoint(self):
        labels = {f"n{i}": 0 for i in range(20)} | {f"a{i}": 1 for i in range(10)}
        split = three_way_stratified_split(labels, 0.4, 0.2, 9)
        self.assertEqual(set(split), set(labels))
        self.assertEqual({split[item] for item in labels}, {"support", "validation", "test"})
        for label in (0, 1):
            self.assertEqual(
                {split[item] for item, item_label in labels.items() if item_label == label},
                {"support", "validation", "test"},
            )

    def test_hdfs_parser_returns_all_distinct_blocks(self):
        parsed = parse_hdfs_line(
            "081109 203518 143 INFO dfs.DataNode: moved blk_-12 to blk_34 and blk_-12\n"
        )
        self.assertIsNotNone(parsed)
        stamp, log, blocks = parsed
        self.assertEqual(blocks, ["blk_-12", "blk_34"])
        self.assertIn("dfs.DataNode", log)
        self.assertGreater(stamp, 0)

    def test_bgl_parser_uses_native_label_and_epoch(self):
        normal = parse_bgl_line(
            "- 1117838570 2005.06.03 node 2005-06-03-15.42.50 node RAS KERNEL INFO corrected error\n"
        )
        abnormal = parse_bgl_line(
            "KERN 1117838571 2005.06.03 node 2005-06-03-15.42.51 node RAS KERNEL FATAL machine failed\n"
        )
        self.assertEqual(normal[:2], (1117838570.0, 0))
        self.assertEqual(abnormal[:2], (1117838571.0, 1))
        self.assertNotIn("1117838570", normal[2])

    def test_seeded_stratification_is_reproducible_and_group_disjoint(self):
        labels = {f"n{i}": 0 for i in range(20)} | {f"a{i}": 1 for i in range(5)}
        capped = stratified_cap(labels, 10, 7)
        self.assertEqual(capped, stratified_cap(labels, 10, 7))
        self.assertEqual(len(capped), 10)
        self.assertEqual(set(capped.values()), {0, 1})
        split = stratified_split(capped, 0.2, 8)
        self.assertEqual(set(split), set(capped))
        self.assertTrue({unit for unit, part in split.items() if part == "train"}.isdisjoint(
            {unit for unit, part in split.items() if part == "validation"}
        ))


class SmallConversionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_bgl_windows_do_not_cross_chronological_boundary(self):
        lines = []
        for index in range(10):
            label = "ERR" if index in {3, 9} else "-"
            lines.append(
                f"{label} {1000 + index} 2005.01.01 node time node RAS KERNEL INFO message-{index}\n"
            )
        write(self.root / "BGL.log", "".join(lines))
        rows, meta = prepare_bgl(self.root, window_size=2, cap_per_split=0,
                                 validation_ratio=0.2, seed=1)
        self.assertEqual(meta["boundary_line"], 8)
        self.assertEqual({row["timestamp"] for row in rows["train"]}, set(range(1000, 1008)))
        self.assertEqual({row["timestamp"] for row in rows["validation"]}, {1008.0, 1009.0})
        labels = {row["session_id"]: int(row["label"]) for row in rows["train"]}
        self.assertIn(1, labels.values())

    def test_hadoop_chunks_keep_parent_in_one_split_and_attach_continuations(self):
        label_text = """### WordCount
Normal:
+ application_1_0001
+ application_1_0002
Machine down:
+ application_1_0003
+ application_1_0004
"""
        write(self.root / "abnormal_label.txt", label_text)
        for app_index in range(1, 5):
            app = f"application_1_{app_index:04d}"
            body = ""
            for event in range(5):
                body += f"2015-10-17 15:37:5{event},000 INFO [main] event {event}\n"
                if event == 0:
                    body += "stack trace continuation\n"
            write(self.root / app / "container.log", body)
        rows, meta = prepare_hadoop(self.root, chunk_size=2, cap=0,
                                    validation_ratio=0.5, seed=3)
        parent_splits = {}
        for split, split_rows in rows.items():
            for row in split_rows:
                parent_splits.setdefault(row["parent_id"], set()).add(split)
        self.assertTrue(all(len(parts) == 1 for parts in parent_splits.values()))
        self.assertTrue(any("stack trace continuation" in row["log"]
                            for split_rows in rows.values() for row in split_rows))
        self.assertTrue(all(count == 3 for count in meta["chunks"].values()))

    def test_hadoop_chunk_cap_keeps_even_parent_coverage(self):
        label_text = """### WordCount
Normal:
+ application_1_0001
+ application_1_0002
Machine down:
+ application_1_0003
+ application_1_0004
"""
        write(self.root / "abnormal_label.txt", label_text)
        for app_index in range(1, 5):
            app = f"application_1_{app_index:04d}"
            body = "".join(
                f"2015-10-17 15:37:{event:02d},000 INFO [main] event {event}\n"
                for event in range(10)
            )
            write(self.root / app / "container.log", body)
        rows, meta = prepare_hadoop(self.root, chunk_size=2, cap=0,
            validation_ratio=0.5, seed=3, max_chunks_per_application=3)
        self.assertTrue(all(count == 3 for count in meta["chunks"].values()))
        by_parent = {}
        for split_rows in rows.values():
            for row in split_rows:
                by_parent.setdefault(row["parent_id"], []).append(row["log"])
        self.assertTrue(all(len(logs) == 6 for logs in by_parent.values()))
        self.assertTrue(all(any("event 0" in log for log in logs) for logs in by_parent.values()))
        self.assertTrue(all(any("event 9" in log for log in logs) for logs in by_parent.values()))

    def test_openstack_folds_have_disjoint_units_and_each_anomaly_is_test_once(self):
        anomaly_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
            "00000000-0000-0000-0000-000000000004",
        ]
        normal_ids = [f"10000000-0000-0000-0000-{index:012d}" for index in range(10)]
        write(self.root / "anomaly_labels.txt", "\n".join(anomaly_ids))
        lines = []
        for index, instance in enumerate(anomaly_ids + normal_ids):
            lines.append(
                f"nova.log 2017-05-14 19:39:{index:02d}.000 1 INFO nova.compute "
                f"[instance: {instance}] event {index}\n"
            )
        lines.append("nova.log 2017-05-14 19:40:00.000 1 INFO nova.api unassigned\n")
        write(self.root / "openstack.log", "".join(lines))
        output = self.root / "out"
        manifests, meta = prepare_openstack(self.root, output, normal_cap=0, seed=9,
                                            support_normal_ratio=0.2,
                                            validation_normal_ratio=0.2)
        self.assertEqual(len(manifests), 4)
        self.assertEqual(meta["ignored_unassigned_lines"], 1)
        test_anomalies = []
        for fold in range(4):
            groups = {}
            for split in ("support", "validation", "test"):
                rows = csv_rows(output / f"openstack_fold_{fold}" / f"{split}.csv")
                groups[split] = {row["parent_id"] for row in rows}
            self.assertTrue(groups["support"].isdisjoint(groups["validation"]))
            self.assertTrue(groups["support"].isdisjoint(groups["test"]))
            self.assertTrue(groups["validation"].isdisjoint(groups["test"]))
            manifest = json.loads((output / f"openstack_fold_{fold}" / "manifest.json").read_text())
            test_anomalies.extend(manifest["anomaly_instances"]["test"])
        self.assertEqual(set(test_anomalies), set(anomaly_ids))
        self.assertEqual(len(test_anomalies), 4)

    def test_openstack_windows_never_cross_vm_parent(self):
        anomaly_ids = [
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
        ]
        normal_ids = [f"10000000-0000-0000-0000-{index:012d}" for index in range(6)]
        write(self.root / "anomaly_labels.txt", "\n".join(anomaly_ids))
        lines = []
        for instance_index, instance in enumerate(anomaly_ids + normal_ids):
            for event in range(7):
                lines.append(
                    f"nova.log 2017-05-14 19:{instance_index:02d}:{event:02d}.000 1 INFO nova.compute "
                    f"[instance: {instance}] event {event}\n"
                )
        write(self.root / "openstack.log", "".join(lines))
        output = self.root / "windowed"
        prepare_openstack(self.root, output, normal_cap=0, seed=9,
            support_normal_ratio=0.2, validation_normal_ratio=0.2, window_size=5)
        for split in ("support", "validation", "test"):
            rows = csv_rows(output / "openstack_fold_0" / f"{split}.csv")
            sessions = {}
            for row in rows:
                sessions.setdefault(row["session_id"], []).append(row)
            self.assertTrue(all(len(items) <= 5 for items in sessions.values()))
            self.assertTrue(all(len({item["parent_id"] for item in items}) == 1
                                for items in sessions.values()))


if __name__ == "__main__":
    unittest.main()
