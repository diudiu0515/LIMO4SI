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
    ray_aabb_distance,
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


def nearest_timed(entries: list[tuple[int, Any]], timestamp_ns: int) -> Any:
    stamps = [entry[0] for entry in entries]
    index = bisect_left(stamps, timestamp_ns)
    candidates = [i for i in (index - 1, index) if 0 <= i < len(entries)]
    chosen = min(candidates, key=lambda i: abs(entries[i][0] - timestamp_ns))
    return entries[chosen][1]


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


def object_pose(uid: str, timestamp_ns: int, static: dict[str, Any], dynamic: dict[str, list[tuple[int, Any]]]) -> Any:
    return static.get(uid) or nearest_timed(dynamic[uid], timestamp_ns)


def object_center(pose: dict[str, Any], bounds: list[list[float]]) -> list[float]:
    local = [(minimum + maximum) / 2 for minimum, maximum in bounds]
    return add(pose["translation"], mat_vec(pose["rotation"], local))


def gaze_hit(
    origin_world: list[float], direction_world: list[float], timestamp_ns: int,
    bounds: dict[str, list[list[float]]], static: dict[str, Any], dynamic: dict[str, list[tuple[int, Any]]],
    maximum_distance_m: float,
) -> tuple[str | None, float | None]:
    candidates = []
    for uid, box in bounds.items():
        if uid not in static and uid not in dynamic:
            continue
        pose = object_pose(uid, timestamp_ns, static, dynamic)
        inverse = transpose(pose["rotation"])
        origin_local = mat_vec(inverse, [x - y for x, y in zip(origin_world, pose["translation"])])
        direction_local = mat_vec(inverse, direction_world)
        distance = ray_aabb_distance(origin_local, direction_local, box)
        if distance is not None and distance <= maximum_distance_m:
            candidates.append((distance, uid))
    if not candidates:
        return None, None
    distance, uid = min(candidates)
    return uid, distance


def calibration_device_from_cpf(vrs_path: Path) -> list[list[float]]:
    try:
        from projectaria_tools.core import data_provider
    except ImportError as exc:
        raise RuntimeError("Install projectaria-tools to read the VRS device/CPF calibration") from exc
    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    return provider.get_device_calibration().get_transform_device_cpf().to_matrix().tolist()


def mine(sequence: Path, minimum_gaze_run: int = 4, maximum_hit_distance_m: float = 8.0) -> dict[str, Any]:
    instances = json.loads((sequence / "instances.json").read_text(encoding="utf-8"))
    gaze_rows = rows(sequence / "eyegaze.csv")
    trajectory_rows = rows(sequence / "aria_trajectory.csv")
    if len(gaze_rows) != len(trajectory_rows) or len(gaze_rows) < minimum_gaze_run:
        raise ValueError("gaze and wearer trajectory must be non-empty and frame aligned")
    bounds = parse_bounds(sequence / "3d_bounding_box.csv")
    static, dynamic = parse_object_poses(sequence / "scene_objects.csv")
    transform = calibration_device_from_cpf(sequence / "video.vrs")
    rotation_device_cpf = [row[:3] for row in transform[:3]]
    translation_device_cpf = [row[3] for row in transform[:3]]
    first_timestamp_us = int(gaze_rows[0]["tracking_timestamp_us"])
    states = []
    all_centers: list[dict[str, list[float]]] = []
    for frame_index, (gaze, wearer) in enumerate(zip(gaze_rows, trajectory_rows)):
        gaze_timestamp_us = int(gaze["tracking_timestamp_us"])
        wearer_timestamp_us = int(wearer["tracking_timestamp_us"])
        if abs(gaze_timestamp_us - wearer_timestamp_us) > 10_000:
            raise ValueError("gaze/wearer timestamp mismatch exceeds 10 ms")
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
        hit_uid, hit_distance = gaze_hit(
            cpf_origin_world, gaze_world, timestamp_ns, bounds, static, dynamic, maximum_hit_distance_m,
        )
        centers = {
            uid: object_center(object_pose(uid, timestamp_ns, static, dynamic), box)
            for uid, box in bounds.items() if uid in static or uid in dynamic
        }
        all_centers.append(centers)
        states.append({
            "frame_index": frame_index,
            "time_s": (gaze_timestamp_us - first_timestamp_us) / 1_000_000,
            "timestamp_ns": timestamp_ns,
            "wearer_world_m": wearer_world,
            "gaze_origin_world_m": cpf_origin_world,
            "gaze_direction_world_unit": gaze_world,
            "gaze_depth_m": float(gaze["depth_m"]),
            "gazed_object_id": hit_uid,
            "gazed_object_name": instances.get(str(hit_uid), {}).get("instance_name") if hit_uid else None,
            "gaze_hit_distance_m": hit_distance,
            "right_world": right_world,
            "forward_world": forward_world,
        })
    events = supported_gaze_events(states, minimum_run=minimum_gaze_run)
    target_ids = sorted({str(event["object_id"]) for event in events})
    for state, centers in zip(states, all_centers):
        state["object_relations"] = {
            uid: body_centric_relation(
                state["gaze_origin_world_m"], centers[uid], state["right_world"], state["forward_world"],
            )
            for uid in target_ids
        }
    for event in events:
        middle = int(event["median_state"]["frame_index"])
        event["median_relation"] = states[middle]["object_relations"][str(event["object_id"])]
        del event["median_state"]
    return {
        "schema_version": 1,
        "dataset": "Aria Digital Twin v2",
        "sequence_name": sequence.name,
        "source_files": [
            "video.vrs", "eyegaze.csv", "aria_trajectory.csv", "scene_objects.csv",
            "3d_bounding_box.csv", "instances.json",
        ],
        "coordinate_frame": "gravity-aligned wearer CPF: +right and +forward, metric world positions",
        "gaze_definition": "nearest positive intersection of the measured gaze ray with a same-time object OBB",
        "minimum_gaze_run_states": minimum_gaze_run,
        "maximum_gaze_hit_distance_m": maximum_hit_distance_m,
        "state_rate_hz": 30,
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
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result = mine(args.sequence, args.minimum_gaze_run)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "states": len(result["states"]),
        "gaze_events": len(result["gaze_events"]), "gazed_objects": len(result["objects"]),
    }, indent=2))


if __name__ == "__main__":
    main()
