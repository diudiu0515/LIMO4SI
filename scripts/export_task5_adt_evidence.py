#!/usr/bin/env python3
"""Render hidden-by-default Task 5 gaze/object localization panels."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.adt_media import (  # noqa: E402
    closest_timestamp_index,
    index_gaze_rows_by_device_timestamp,
    load_sequence_calibration,
    pixel_inside_box,
    preview_frame_timestamps,
    project_gaze_to_raw_rgb,
    resolve_rgb_media,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/task5_release_cases.json")
    parser.add_argument("--analysis", type=Path, default=ROOT / "outputs/qa/task5_adt_analysis.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "site/qa_benchmark/task5_media")
    args = parser.parse_args()
    import cv2
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain, TimeQueryOptions
    from projectaria_tools.core.stream_id import StreamId

    config = json.loads(args.config.read_text(encoding="utf-8"))
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    rgb_media, media_kind = resolve_rgb_media(args.sequence)
    stream = StreamId("214-1")
    provider = None
    preview = None
    preview_timestamps: list[int] = []
    if media_kind == "vrs_rgb_stream":
        provider = data_provider.create_vrs_data_provider(str(rgb_media))
    else:
        load_sequence_calibration(args.sequence)
        preview_timestamps = preview_frame_timestamps(rgb_media)
        preview = cv2.VideoCapture(str(rgb_media))
        if not preview.isOpened():
            raise RuntimeError(f"could not open {rgb_media}")
    with (args.sequence / "eyegaze.csv").open(newline="", encoding="utf-8") as handle:
        gaze_by_timestamp = index_gaze_rows_by_device_timestamp(csv.DictReader(handle))
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
        if spec["question_type"] == "gaze_target_at_evidence_anchor":
            frame_ids = [int(frame) for frame in spec["anchor_frames"]]
            show_target_gaze = [
                str(analysis["states"][frame].get("gazed_object_id")) == uid for frame in frame_ids
            ]
        elif spec["question_type"] == "relation_change_between_gazes":
            events = [next(event for event in analysis["gaze_events"] if str(event["object_id"]) == uid and event["start_index"] == frame) for frame in spec["event_start_frames"]]
            frame_ids = [(event["start_index"] + event["end_index"]) // 2 for event in events]
            show_target_gaze = [True] * len(frame_ids)
        elif spec["question_type"] == "gaze_onset_side_change":
            event = next(event for event in analysis["gaze_events"] if str(event["object_id"]) == uid and event["start_index"] == spec["event_start_frames"][0])
            frame_ids = [spec["pre_frame"], (event["start_index"] + event["end_index"]) // 2]
            show_target_gaze = [False, True]
        else:
            lo, hi = (int(value) for value in spec["window_frames"])
            minimum_run = int((config.get("selection_policy") or {}).get("minimum_relation_run_states", 6))
            runs = []
            for frame_index in range(lo, hi + 1):
                label = analysis["states"][frame_index]["object_relations"][uid]["label"]
                if not runs or runs[-1][0] != label:
                    runs.append([label, frame_index, frame_index])
                else:
                    runs[-1][2] = frame_index
            stable_runs = [run for run in runs if run[2] - run[1] + 1 >= minimum_run]
            collapsed = []
            for label, start, end in stable_runs:
                if collapsed and collapsed[-1][0] == label:
                    collapsed[-1][2] = end
                else:
                    collapsed.append([label, start, end])
            def annotation_skew(frame_index: int) -> int:
                timestamp = int(analysis["states"][frame_index]["timestamp_ns"])
                return min(abs(value - timestamp) for value in box_times[uid])

            frame_ids = [
                min(range(start, end + 1), key=annotation_skew)
                for _, start, end in collapsed
            ]
            if len(frame_ids) < 2:
                raise ValueError(f"{spec['id']} has fewer than two sustained relation evidence stages")
            show_target_gaze = [False] * len(frame_ids)
        panels = []
        for panel_index, frame_index in enumerate(frame_ids):
            state = analysis["states"][frame_index]
            timestamp = int(state["timestamp_ns"])
            if provider is not None:
                image, record = provider.get_image_data_by_time_ns(
                    stream, timestamp, TimeDomain.DEVICE_TIME, TimeQueryOptions.CLOSEST,
                )
                rgb_timestamp = int(record.capture_timestamp_ns)
                canvas = cv2.cvtColor(image.to_numpy_array(), cv2.COLOR_RGB2BGR)
            else:
                frame = closest_timestamp_index(preview_timestamps, timestamp)
                preview.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ok, encoded_canvas = preview.read()
                if not ok:
                    raise RuntimeError(f"could not decode preview frame {frame} for {spec['id']}")
                rgb_timestamp = preview_timestamps[frame]
                # ADT preview frames are already rotated clockwise for display.
                # Restore sensor orientation while applying raw 2D annotations;
                # the common path below rotates the completed panel back.
                canvas = cv2.rotate(encoded_canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)
            rgb_skew_ms = abs(rgb_timestamp - timestamp) / 1_000_000
            if rgb_skew_ms > 50.0:
                raise ValueError(
                    f"{spec['id']} nearest RGB frame is stale by {rgb_skew_ms:.1f} ms"
                )
            nearest = min(box_times[uid], key=lambda value: abs(value - timestamp))
            skew_ms = abs(nearest - timestamp) / 1_000_000
            if skew_ms > 50.0:
                raise ValueError(f"{spec['id']} nearest 2D box is stale by {skew_ms:.1f} ms")
            x1, y1, x2, y2 = boxes[(nearest, uid)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 210, 40), 8)
            gaze = gaze_by_timestamp.get(timestamp)
            if gaze is None:
                raise ValueError(
                    f"{spec['id']} evidence frame {frame_index} has no exact gaze row at {timestamp}"
                )
            depth = float(gaze["depth_m"]) or float(state.get("gaze_hit_distance_m") or 1.0)
            pixel = (
                project_gaze_to_raw_rgb(
                    args.sequence, float(gaze["yaw_rads_cpf"]),
                    float(gaze["pitch_rads_cpf"]), depth,
                )
                if show_target_gaze[panel_index] else None
            )
            if pixel is not None:
                if not pixel_inside_box(pixel, (x1, y1, x2, y2), margin=12.0):
                    raise ValueError(
                        f"{spec['id']} gaze projection falls outside the target RGB box"
                    )
                px, py = (int(round(value)) for value in pixel)
                cv2.drawMarker(canvas, (px, py), (20, 20, 240), cv2.MARKER_CROSS, 55, 8)
                cv2.circle(canvas, (px, py), 18, (20, 20, 240), 5)
            canvas = cv2.rotate(canvas, cv2.ROTATE_90_CLOCKWISE)
            target_name = analysis["objects"][uid]["instance_name"]
            if spec["question_type"] == "gaze_target_at_evidence_anchor":
                gaze_target = state.get("gazed_object_name") or "no annotated object"
                label = (
                    f"anchor {panel_index + 1}  t={state['time_s']:.1f}s  "
                    f"gaze -> {gaze_target}  target: {target_name}"
                )
            else:
                relation = state["object_relations"][uid]["label"]
                label = f"t={state['time_s']:.1f}s  {target_name}  {relation.replace('-', ' ')}"
            cv2.rectangle(canvas, (0, 0), (1408, 80), (15, 23, 42), -1)
            cv2.putText(canvas, label, (24, 54), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 3, cv2.LINE_AA)
            panels.append(cv2.resize(canvas, (448, 448), interpolation=cv2.INTER_AREA))
        combined = cv2.hconcat(panels)
        output = args.output_dir / f"{spec['id']}_gaze_evidence.jpg"
        cv2.imwrite(str(output), combined, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(output)
    if preview is not None:
        preview.release()


if __name__ == "__main__":
    main()
