#!/usr/bin/env python3
"""Mine metric gaze/object/wearer states from an Aria Digital Twin sequence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from bisect import bisect_left
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.task5_human_state import (
    body_centric_relation,
    mat_vec,
    quaternion_rotation_xyzw,
    ray_aabb_interval,
    supported_gaze_events,
    transpose,
    unit,
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def matrix_column(matrix: list[list[float]], index: int) -> list[float]:
    return [row[index] for row in matrix]


def nearest_timed(entries: list[tuple[int, Any]], timestamp: int) -> tuple[Any, int]:
    """Return the nearest value and absolute timestamp skew without copying timestamps."""
    if not entries:
        raise ValueError("cannot align against an empty timestamp stream")
    index = bisect_left(entries, (timestamp,))
    candidates = [i for i in (index - 1, index) if 0 <= i < len(entries)]
    chosen = min(candidates, key=lambda i: abs(entries[i][0] - timestamp))
    return entries[chosen][1], abs(entries[chosen][0] - timestamp)


def parse_bounds(path: Path) -> dict[str, list[list[float]]]:
    out = {}
    for row in rows(path):
        if int(row["timestamp[ns]"]) != -1:
            continue
        out[row["object_uid"]] = [
            [float(row[f"p_local_obj_{axis}min[m]"]), float(row[f"p_local_obj_{axis}max[m]"])]
            for axis in "xyz"
        ]
    return out


def parse_object_poses(path: Path) -> tuple[dict[str, Any], dict[str, list[tuple[int, Any]]]]:
    static: dict[str, Any] = {}
    dynamic: dict[str, list[tuple[int, Any]]] = {}
    for row in rows(path):
        uid, timestamp = row["object_uid"], int(row["timestamp[ns]"])
        value = {
            "translation": [float(row[f"t_wo_{axis}[m]"]) for axis in "xyz"],
            "rotation": quaternion_rotation_xyzw([
                float(row["q_wo_x"]), float(row["q_wo_y"]),
                float(row["q_wo_z"]), float(row["q_wo_w"]),
            ]),
        }
        if timestamp < 0:
            static[uid] = value
        else:
            dynamic.setdefault(uid, []).append((timestamp, value))
    for values in dynamic.values():
        values.sort(key=lambda item: item[0])
    return static, dynamic


def object_pose(
    uid: str, timestamp_ns: int, static: dict[str, Any],
    dynamic: dict[str, list[tuple[int, Any]]], maximum_skew_ns: int,
) -> tuple[Any | None, int]:
    # A dynamic track takes precedence if a dataset also provides a canonical static pose.
    if uid in dynamic:
        pose, skew = nearest_timed(dynamic[uid], timestamp_ns)
        return (pose, skew) if skew <= maximum_skew_ns else (None, skew)
    return (static.get(uid), 0)


def object_center(pose: dict[str, Any], bounds: list[list[float]]) -> list[float]:
    local = [(minimum + maximum) / 2 for minimum, maximum in bounds]
    return add(pose["translation"], mat_vec(pose["rotation"], local))


def gaze_hit(
    origin_world: list[float], direction_world: list[float], gaze_depth_m: float, timestamp_ns: int,
    bounds: dict[str, list[list[float]]], static: dict[str, Any], dynamic: dict[str, list[tuple[int, Any]]],
    maximum_distance_m: float, maximum_object_pose_skew_ns: int, maximum_depth_residual_m: float,
    eligible_object_ids: set[str],
) -> tuple[str | None, float | None, float | None, float | None]:
    """Assign gaze only when its measured fixation depth lies inside an object OBB."""
    if not math.isfinite(gaze_depth_m) or not 0 < gaze_depth_m <= maximum_distance_m:
        return None, None, None, None
    candidates = []
    for uid, box in bounds.items():
        if uid not in eligible_object_ids or (uid not in static and uid not in dynamic):
            continue
        pose, _ = object_pose(uid, timestamp_ns, static, dynamic, maximum_object_pose_skew_ns)
        if pose is None:
            continue
        inverse = transpose(pose["rotation"])
        origin_local = mat_vec(inverse, [x - y for x, y in zip(origin_world, pose["translation"])])
        direction_local = mat_vec(inverse, direction_world)
        interval = ray_aabb_interval(origin_local, direction_local, box)
        if interval is None:
            continue
        entry, exit_distance = interval
        residual = max(entry - gaze_depth_m, gaze_depth_m - exit_distance, 0.0)
        if entry > maximum_distance_m or residual > maximum_depth_residual_m:
            continue
        volume = math.prod(maximum - minimum for minimum, maximum in box)
        midpoint_error = abs((entry + exit_distance) / 2 - gaze_depth_m)
        candidates.append((residual, volume, midpoint_error, entry, uid, exit_distance))
    if not candidates:
        return None, None, None, None
    residual, _, _, entry, uid, exit_distance = min(candidates)
    return uid, entry, exit_distance, residual

def calibration_device_from_cpf(vrs_path: Path) -> list[list[float]]:
    try:
        from projectaria_tools.core import data_provider
    except ImportError as exc:
        raise RuntimeError("Install projectaria-tools to read the VRS device/CPF calibration") from exc
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    return provider.get_device_calibration().get_transform_device_cpf().to_matrix().tolist()


def mine(
    sequence: Path, minimum_gaze_run: int = 4, maximum_hit_distance_m: float = 8.0,
    maximum_wearer_skew_ms: float = 10.0, maximum_object_pose_skew_ms: float = 50.0,
    maximum_internal_gaze_gap_states: int = 1, maximum_gaze_depth_residual_m: float = 0.05,
) -> dict[str, Any]:
    required = {
        "video.vrs", "eyegaze.csv", "aria_trajectory.csv", "scene_objects.csv",
        "3d_bounding_box.csv", "instances.json",
    }
    missing = sorted(name for name in required if not (sequence / name).is_file())
    if missing:
        raise FileNotFoundError(f"{sequence} is missing required ADT files: {missing}")
    instances = json.loads((sequence / "instances.json").read_text(encoding="utf-8"))
    gaze_rows = rows(sequence / "eyegaze.csv")
    trajectory_rows = rows(sequence / "aria_trajectory.csv")
    if len(gaze_rows) < minimum_gaze_run or not trajectory_rows:
        raise ValueError("gaze and wearer trajectory streams must be non-empty")
    gaze_stamps_us = [int(row["tracking_timestamp_us"]) for row in gaze_rows]
    if any(right <= left for left, right in zip(gaze_stamps_us, gaze_stamps_us[1:])):
        raise ValueError("gaze timestamps must be strictly increasing")
    trajectory_entries = sorted(
        ((int(row["tracking_timestamp_us"]), row) for row in trajectory_rows),
        key=lambda item: item[0],
    )
    bounds = parse_bounds(sequence / "3d_bounding_box.csv")
    eligible_object_ids = {
        uid for uid, row in instances.items()
        if row.get("instance_type") == "object" and row.get("category") != "shelter"
    }
    static, dynamic = parse_object_poses(sequence / "scene_objects.csv")
    transform = calibration_device_from_cpf(sequence / "video.vrs")
    rotation_device_cpf = [row[:3] for row in transform[:3]]
    translation_device_cpf = [row[3] for row in transform[:3]]
    first_timestamp_us = gaze_stamps_us[0]
    states = []
    all_centers: list[dict[str, list[float]]] = []
    all_center_skews_ms: list[dict[str, float]] = []
    wearer_skews_ms: list[float] = []
    object_skews_ms: list[float] = []
    maximum_wearer_skew_us = int(maximum_wearer_skew_ms * 1000)
    maximum_object_pose_skew_ns = int(maximum_object_pose_skew_ms * 1_000_000)
    for frame_index, gaze in enumerate(gaze_rows):
        gaze_timestamp_us = int(gaze["tracking_timestamp_us"])
        wearer, wearer_skew_us = nearest_timed(trajectory_entries, gaze_timestamp_us)
        if wearer_skew_us > maximum_wearer_skew_us:
            raise ValueError(
                f"gaze/wearer timestamp mismatch {wearer_skew_us / 1000:.3f} ms exceeds "
                f"{maximum_wearer_skew_ms:.3f} ms"
            )
        wearer_skews_ms.append(wearer_skew_us / 1000)
        timestamp_ns = gaze_timestamp_us * 1000
        rotation_world_device = quaternion_rotation_xyzw([
            float(wearer["qx_world_device"]), float(wearer["qy_world_device"]),
            float(wearer["qz_world_device"]), float(wearer["qw_world_device"]),
        ])
        wearer_world = [float(wearer[f"t{axis}_world_device"]) for axis in "xyz"]
        cpf_origin_world = add(wearer_world, mat_vec(rotation_world_device, translation_device_cpf))
        gaze_cpf = unit([
            math.tan(float(gaze["yaw_rads_cpf"])),
            math.tan(float(gaze["pitch_rads_cpf"])), 1.0,
        ])
        assert gaze_cpf is not None
        gaze_world = mat_vec(rotation_world_device, mat_vec(rotation_device_cpf, gaze_cpf))
        right_world = mat_vec(rotation_world_device, matrix_column(rotation_device_cpf, 0))
        forward_world = mat_vec(rotation_world_device, matrix_column(rotation_device_cpf, 2))
        gaze_depth_m = float(gaze["depth_m"])
        hit_uid, hit_distance, hit_exit_distance, hit_depth_residual = gaze_hit(
            cpf_origin_world, gaze_world, gaze_depth_m, timestamp_ns, bounds, static, dynamic,
            maximum_hit_distance_m, maximum_object_pose_skew_ns, maximum_gaze_depth_residual_m,
            eligible_object_ids,
        )
        centers = {}
        center_skews_ms: dict[str, float] = {}
        for uid, box in bounds.items():
            if uid not in static and uid not in dynamic:
                continue
            pose, skew_ns = object_pose(uid, timestamp_ns, static, dynamic, maximum_object_pose_skew_ns)
            if uid in dynamic:
                object_skews_ms.append(skew_ns / 1_000_000)
            if pose is not None:
                centers[uid] = object_center(pose, box)
                center_skews_ms[uid] = skew_ns / 1_000_000
        all_centers.append(centers)
        all_center_skews_ms.append(center_skews_ms)
        states.append({
            "frame_index": frame_index,
            "time_s": (gaze_timestamp_us - first_timestamp_us) / 1_000_000,
            "timestamp_ns": timestamp_ns,
            "wearer_world_m": wearer_world,
            "gaze_origin_world_m": cpf_origin_world,
            "gaze_direction_world_unit": gaze_world,
            "gaze_depth_m": gaze_depth_m,
            "gazed_object_id": hit_uid,
            "gazed_object_name": instances.get(str(hit_uid), {}).get("instance_name") if hit_uid else None,
            "gaze_hit_distance_m": hit_distance,
            "gaze_hit_exit_distance_m": hit_exit_distance,
            "gaze_depth_obb_residual_m": hit_depth_residual,
            "right_world": right_world,
            "forward_world": forward_world,
            "wearer_timestamp_skew_ms": wearer_skew_us / 1000,
        })
    events = supported_gaze_events(
        states, minimum_run=minimum_gaze_run, maximum_internal_gap=maximum_internal_gaze_gap_states,
    )
    target_ids = sorted({str(event["object_id"]) for event in events})
    for state, centers, center_skews in zip(states, all_centers, all_center_skews_ms):
        state["object_relations"] = {
            uid: body_centric_relation(
                state["gaze_origin_world_m"], centers[uid], state["right_world"], state["forward_world"],
            )
            for uid in target_ids if uid in centers
        }
        state["object_pose_skew_ms"] = {uid: center_skews[uid] for uid in target_ids if uid in center_skews}
    for event in events:
        middle = int(event["median_state"]["frame_index"])
        event["median_relation"] = states[middle]["object_relations"][str(event["object_id"])]
        del event["median_state"]
    intervals = sorted((right - left) / 1_000_000 for left, right in zip(gaze_stamps_us, gaze_stamps_us[1:]))
    median_interval_s = intervals[len(intervals) // 2] if intervals else 0.0
    return {
        "schema_version": 4,
        "dataset": "Aria Digital Twin v2",
        "sequence_name": sequence.name,
        "source_files": [
            "video.vrs", "eyegaze.csv", "aria_trajectory.csv", "scene_objects.csv",
            "3d_bounding_box.csv", "instances.json",
        ],
        "coordinate_frame": "gravity-aligned wearer CPF: +right and +forward, metric world positions",
        "gaze_definition": "smallest eligible same-time object OBB containing the measured fixation depth along the gaze ray",
        "minimum_gaze_run_states": minimum_gaze_run,
        "maximum_internal_gaze_gap_states": maximum_internal_gaze_gap_states,
        "maximum_gaze_hit_distance_m": maximum_hit_distance_m,
        "maximum_gaze_depth_obb_residual_m": maximum_gaze_depth_residual_m,
        "state_rate_hz": round(1.0 / median_interval_s, 3) if median_interval_s > 0 else 0,
        "alignment_diagnostics": {
            "maximum_allowed_wearer_skew_ms": maximum_wearer_skew_ms,
            "maximum_observed_wearer_skew_ms": max(wearer_skews_ms, default=0.0),
            "maximum_allowed_dynamic_object_skew_ms": maximum_object_pose_skew_ms,
            "maximum_observed_dynamic_object_skew_ms": max(object_skews_ms, default=0.0),
            "gaze_state_count": len(gaze_rows),
            "wearer_state_count": len(trajectory_rows),
        },
        "states": states,
        "gaze_events": events,
        "objects": {
            uid: {
                "instance_name": instances[uid]["instance_name"],
                "category": instances[uid]["category"],
                "motion_type": instances[uid]["motion_type"],
            }
            for uid in target_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/qa/task5_adt_analysis.json"))
    parser.add_argument("--minimum-gaze-run", type=int, default=4)
    parser.add_argument("--maximum-hit-distance-m", type=float, default=8.0)
    parser.add_argument("--maximum-wearer-skew-ms", type=float, default=10.0)
    parser.add_argument("--maximum-object-pose-skew-ms", type=float, default=50.0)
    parser.add_argument("--maximum-internal-gaze-gap-states", type=int, default=1)
    parser.add_argument("--maximum-gaze-depth-residual-m", type=float, default=0.05)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = mine(
        args.sequence, args.minimum_gaze_run, args.maximum_hit_distance_m,
        args.maximum_wearer_skew_ms, args.maximum_object_pose_skew_ms,
        args.maximum_internal_gaze_gap_states, args.maximum_gaze_depth_residual_m,
    )
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "states": len(result["states"]),
        "gaze_events": len(result["gaze_events"]), "gazed_objects": len(result["objects"]),
    }, indent=2))


if __name__ == "__main__":
    main()
