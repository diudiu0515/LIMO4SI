import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch
from generate_qa import ADT_REQUIRED_FILES, BUNDLE_SCHEMA, build_task5_adt, detect_kind


class AnnotationRoutingTests(unittest.TestCase):
    def test_detects_adt_before_generic_directory_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sequence = root / "sequence_01"
            sequence.mkdir()
            for name in ADT_REQUIRED_FILES:
                (sequence / name).touch()
            self.assertEqual(detect_kind(root), "task5_adt")

    def test_detects_egoexo_only_from_its_explicit_schema_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "annotations").mkdir()
            (root / "takes.json").write_text("{}")
            (root / "annotations" / "relations_val.json").write_text("{}")
            self.assertEqual(detect_kind(root), "task5_egoexo")

    def test_plain_directory_is_not_silently_treated_as_task5(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(detect_kind(Path(directory)), "task4_raw")

    def test_bundle_schema_remains_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.json"
            path.write_text(json.dumps({
                "schema": BUNDLE_SCHEMA,
                "task4_annotations": "task4.json",
                "task5_annotations": "adt",
            }))
            self.assertEqual(detect_kind(path), "bundle")


    def test_adt_builder_uses_balanced_fifteen_second_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotations, output, site = root / "adt", root / "output", root / "site"
            annotations.mkdir()
            output.mkdir()
            site.mkdir()

            def fake_run(name, *arguments):
                self.assertEqual(name, "scale_task5_adt.py")
                values = list(arguments)
                self.assertEqual(values[values.index("--target-per-category") + 1], "1")
                category_targets = [values[index + 1] for index, value in enumerate(values) if value == "--category-target"]
                self.assertEqual(sorted(category_targets), ["after_gaze_turns_to_object=2", "between_repeated_gaze_events=2", "last_gaze_annotated_object=2"])
                self.assertEqual(values[values.index("--target-window-sec") + 1], "15")
                self.assertIn("--allow-subset", values)
                report = output / "task5_adt_scale" / "pipeline_report.json"
                report.parent.mkdir(parents=True)
                report.write_text(json.dumps({
                    "selection": {"selected_count": 6, "selected_counts": {"a": 2, "b": 2, "c": 2}},
                    "rejected_sequence_count": 0,
                }))

            with patch("generate_qa.run_script", side_effect=fake_run):
                report = build_task5_adt(annotations, output, site / "data.js", site, 6, 2, 1, None)
            self.assertEqual((report["backend"], report["accepted_count"]), ("adt", 6))

if __name__ == "__main__":
    unittest.main()
