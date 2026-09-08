#!/usr/bin/env python3
"""Render hidden-by-default Task 5 gaze/object localization panels."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task5_release_cases.json")
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/qa/task5_adt_analysis.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "site/qa_benchmark/task5_media")
    args = parser.parse_args()
    import cv2
    import numpy as np
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    config = json.loads(args.config.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    provider = data_provider.create_vrs_data_provider(str(args.sequence / "video.vrs"))
    stream = StreamId("214-1")
    device_calibration = provider.get_device_calibration()
    camera_calibration = device_calibration.get_camera_calib("camera-rgb")
    transform_camera_cpf = device_calibration.get_transform_cpf_sensor("camera-rgb").inverse().to_matrix()
    with (args.sequence / "eyegaze.csv").open(newline="", encoding="utf-8") as handle:
        gaze_rows = list(csv.DictReader(handle))
    boxes: dict[tuple[int, str], list[int]] = {}
    with (args.sequence / "2d_bounding_box.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] != "214-1":
                continue
            boxes[(int(row["timestamp[ns]"]), row["object_uid"])] = [
                int(float(row["x_min[pixel]"])), int(float(row["y_min[pixel]"])),
                int(float(row["x_max[pixel]"])), int(float(row["y_max[pixel]"])),
            ]
    box_times: dict[str, list[int]] = {}
    for timestamp, uid in boxes:
        box_times.setdefault(uid, []).append(timestamp)
    for values in box_times.values():
        values.sort()
    objects_by_name: dict[str, list[str]] = {}
    for uid, row in analysis["objects"].items():
        objects_by_name.setdefault(row["instance_name"], []).append(uid)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for spec in config["cases"]:
        if spec.get("object_id"):
            uid = str(spec["object_id"])
        else:
            matches = objects_by_name.get(spec["object_name"], [])
            if len(matches) != 1:
                raise ValueError(f"{spec['id']} object name does not uniquely resolve; provide object_id")
            uid = matches[0]
        if uid not in box_times:
            raise ValueError(f"{spec['id']} target has no RGB 2D bounding-box annotation")
        if spec["question_type"] == "relation_change_between_gazes":
            events = [next(event for event in analysis["gaze_events"] if str(event["object_id"]) == uid and event["start_index"] == frame) for frame in spec["event_start_frames"]]
            frame_ids = [event["start_index"] + event["state_count"] // 2 for event in events]
        elif spec["question_type"] == "gaze_onset_side_change":
            event = next(event for event in analysis["gaze_events"] if str(event["object_id"]) == uid and event["start_index"] == spec["event_start_frames"][0])
            frame_ids = [spec["pre_frame"], event["start_index"] + event["state_count"] // 2]
        else:
            frame_ids = list(spec["window_frames"])
        panels = []
        for frame_index in frame_ids:
            image, _ = provider.get_image_data_by_index(stream, int(frame_index))
            rgb = image.to_numpy_array()
            canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            state = analysis["states"][frame_index]
            timestamp = int(state["timestamp_ns"])
            nearest = min(box_times[uid], key=lambda value: abs(value - timestamp))
            skew_ms = abs(nearest - timestamp) / 1_000_000
            if skew_ms > 50.0:
                raise ValueError(f"{spec['id']} nearest 2D box is stale by {skew_ms:.1f} ms")
            x1, y1, x2, y2 = boxes[(nearest, uid)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 210, 40), 8)
            gaze = gaze_rows[frame_index]
            depth = float(gaze["depth_m"]) or float(state.get("gaze_hit_distance_m") or 1.0)
            point_cpf = np.array([
                math.tan(float(gaze["yaw_rads_cpf"])) * depth,
                math.tan(float(gaze["pitch_rads_cpf"])) * depth, depth, 1.0,
            ])
            point_camera = transform_camera_cpf @ point_cpf
            pixel = camera_calibration.project(point_camera[:3] / point_camera[3])
            if pixel is not None:
                px, py = (int(round(value)) for value in pixel)
                cv2.drawMarker(canvas, (px, py), (20, 20, 240), cv2.MARKER_CROSS, 55, 8)
                cv2.circle(canvas, (px, py), 18, (20, 20, 240), 5)
            canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
            relation = state["object_relations"][uid]["label"]
            target_name = analysis["objects"][uid]["instance_name"]
            label = f"t={state['time_s']:.1f}s  {target_name}  {relation}"
            cv2.rectangle(canvas, (0, 0), (1408, 80), (15, 23, 42), -1)
            cv2.putText(canvas, label, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)
            panels.append(cv2.resize(canvas, (448, 448), interpolation=cv2.INTER_AREA))
        combined = cv2.hconcat(panels)
        output = args.output_dir / f"{spec['id']}_gaze_evidence.jpg"
        cv2.imwrite(str(output), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(output)


if __name__ == "__main__":
    main()
