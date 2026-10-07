import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from limo4si.adt_media import (
    closest_timestamp_index,
    index_gaze_rows_by_device_timestamp,
    load_sequence_calibration,
    pixel_inside_box,
    preview_frame_timestamps,
    resolve_rgb_media,
)


class AdtMediaTests(unittest.TestCase):
    def test_gaze_rows_are_indexed_by_timestamp_not_filtered_state_offset(self):
        leading = {"tracking_timestamp_us": "100", "yaw_rads_cpf": "9"}
        retained = {"tracking_timestamp_us": "200", "yaw_rads_cpf": "1"}
        indexed = index_gaze_rows_by_device_timestamp([leading, retained])
        self.assertIs(indexed[200_000], retained)

    def test_duplicate_gaze_timestamps_are_rejected(self):
        rows = [
            {"tracking_timestamp_us": "100"},
            {"tracking_timestamp_us": "100"},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate ADT gaze timestamp"):
            index_gaze_rows_by_device_timestamp(rows)

    def test_projected_pixel_must_fall_inside_target_box_margin(self):
        box = (100.0, 200.0, 140.0, 240.0)
        self.assertTrue(pixel_inside_box((112.0, 220.0), box, margin=12.0))
        self.assertTrue(pixel_inside_box((88.0, 252.0), box, margin=12.0))
        self.assertFalse(pixel_inside_box((87.9, 220.0), box, margin=12.0))
        self.assertFalse(pixel_inside_box(None, box, margin=12.0))

    def test_official_preview_timestamps_are_device_time_and_frame_exact(self):
        probe = {
            "streams": [{"codec_type": "video", "nb_frames": "4"}],
            "format": {"tags": {"description": json.dumps([10, 20, 31, 40])}},
        }
        with patch("limo4si.adt_media.subprocess.check_output", return_value=json.dumps(probe).encode()):
            self.assertEqual(preview_frame_timestamps(Path("preview_rgb.mp4")), [10, 20, 31, 40])
        self.assertEqual(closest_timestamp_index([10, 20, 31, 40], 29), 2)

    def test_preview_timestamp_count_mismatch_fails_closed(self):
        probe = {
            "streams": [{"codec_type": "video", "nb_frames": "3"}],
            "format": {"tags": {"description": json.dumps([10, 20])}},
        }
        with patch("limo4si.adt_media.subprocess.check_output", return_value=json.dumps(probe).encode()):
            with self.assertRaisesRegex(ValueError, "timestamp/frame count mismatch"):
                preview_frame_timestamps(Path("preview_rgb.mp4"))

    def test_preview_allows_duplicate_encoded_frames_but_not_backward_time(self):
        duplicate = {
            "streams": [{"codec_type": "video", "nb_frames": "3"}],
            "format": {"tags": {"description": json.dumps([10, 10, 20])}},
        }
        backward = {
            "streams": [{"codec_type": "video", "nb_frames": "3"}],
            "format": {"tags": {"description": json.dumps([10, 9, 20])}},
        }
        with patch("limo4si.adt_media.subprocess.check_output", return_value=json.dumps(duplicate).encode()):
            self.assertEqual(preview_frame_timestamps(Path("preview_rgb.mp4")), [10, 10, 20])
        with patch("limo4si.adt_media.subprocess.check_output", return_value=json.dumps(backward).encode()):
            with self.assertRaisesRegex(ValueError, "move backward"):
                preview_frame_timestamps(Path("preview_rgb.mp4"))

    def test_sequence_calibration_is_bound_to_metadata_serial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata.json").write_text(json.dumps({"serial": "device-a"}))
            registry = root / "registry.json"
            registry.write_text(json.dumps({
                "schema_version": "limo4si.adt_device_calibrations.v1",
                "devices": {
                    "device-a": {
                        "source_sequence": "source-a",
                        "source_api": "official",
                        "transform_device_cpf": [
                            [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
                        ],
                        "rgb_camera": {"projection_params": list(range(15))},
                    },
                },
            }))
            result = load_sequence_calibration(root, registry)
            self.assertEqual(result["device_serial"], "device-a")

    def test_preview_is_preferred_over_vrs_for_release_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video.vrs").touch()
            (root / "preview_rgb.mp4").touch()
            path, kind = resolve_rgb_media(root)
            self.assertEqual(path.name, "preview_rgb.mp4")
            self.assertEqual(kind, "official_preview_mp4")


if __name__ == "__main__":
    unittest.main()
