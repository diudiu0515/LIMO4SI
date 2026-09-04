"""Metric human-trajectory / static-landmark topology for Task 3."""
from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping, Sequence

Vec = Sequence[float]


def dot(a: Vec, b: Vec) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def sub(a: Vec, b: Vec) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def norm(a: Vec) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec) -> list[float] | None:
    length = norm(a)
    return None if length < 1e-9 else [float(value) / length for value in a]


def cross(a: Vec, b: Vec) -> list[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def horizontal(vector: Vec, up: Vec) -> list[float]:
    vertical = dot(vector, up)
    return [float(vector[i]) - vertical * float(up[i]) for i in range(3)]


def smooth_points(points: Sequence[Vec], radius: int = 2) -> list[list[float]]:
    rows = []
    for index in range(len(points)):
        window = points[max(0, index - radius):min(len(points), index + radius + 1)]
        rows.append([sum(float(row[axis]) for row in window) / len(window) for axis in range(3)])
    return rows


def clean_landmark_name(value: Any) -> str:
    text = str(value or "landmark").replace("_", " ").strip()
    parts = text.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        text = parts[0]
    return " ".join(text.split()).lower()


def trajectory_topology(
    timeline: Sequence[Mapping[str, Any]],
    landmarks: Sequence[Mapping[str, Any]],
    *,
    smoothing_radius: int = 2,
    side_dead_zone_m: float = 0.25,
    max_landmark_distance_m: float = 2.0,
    min_local_travel_m: float = 0.35,
    min_order_gap_sec: float = 2.0,
) -> dict[str, Any]:
    """Analyze topology using the full smoothed trajectory and local path tangent."""
    if len(timeline) < 8:
        return {"status": "missing_evidence", "missing_evidence": ["at least eight human trajectory states"]}
    raw_points = []
    times = []
    frames = []
    ups = []
    for row in timeline:
        frame = row.get("human_frame")
        origin = getattr(frame, "origin", None) if frame is not None else row.get("origin_world_m")
        up = getattr(frame, "up", None) if frame is not None else row.get("up_world_unit")
        time = row.get("t_sec_from_center", row.get("t"))
        if origin is None or up is None or not isinstance(time, (int, float)):
            continue
        raw_points.append([float(value) for value in origin])
        ups.append([float(value) for value in up])
        times.append(float(time))
        frames.append(row.get("frame", row.get("frame_id")))
    if len(raw_points) < 8:
        return {"status": "missing_evidence", "missing_evidence": ["eight finite metric trajectory states"]}
    reference_up = unit(ups[len(ups) // 2])
    if reference_up is None:
        return {"status": "missing_evidence", "missing_evidence": ["valid scene-up direction"]}
    points = smooth_points(raw_points, smoothing_radius)
    segment_lengths = [norm(horizontal(sub(right, left), reference_up)) for left, right in zip(points, points[1:])]
    path_length = sum(segment_lengths)
    net_displacement = norm(horizontal(sub(points[-1], points[0]), reference_up))
    speeds = [
        distance / (times[index + 1] - times[index])
        for index, distance in enumerate(segment_lengths)
        if times[index + 1] > times[index]
    ]
    states = [
        {"frame": frames[index], "t": times[index], "origin_world_m": points[index]}
        for index in range(len(points))
    ]

    landmark_rows = []
    for landmark in landmarks:
        xyz = landmark.get("object_xyz_world_m") or landmark.get("center")
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            continue
        landmark_id = str(landmark.get("object_id") or landmark.get("id") or "landmark")
        display_name = str(landmark.get("display_name") or clean_landmark_name(landmark_id))
        distances = [norm(horizontal(sub(xyz, point), reference_up)) for point in points]
        closest_index = min(range(len(distances)), key=distances.__getitem__)
        lo = max(0, closest_index - 2)
        hi = min(len(points) - 1, closest_index + 2)
        while hi - lo < 12 and norm(horizontal(sub(points[hi], points[lo]), reference_up)) < min_local_travel_m:
            if lo > 0:
                lo -= 1
            if hi < len(points) - 1:
                hi += 1
            if lo == 0 and hi == len(points) - 1:
                break
        tangent = horizontal(sub(points[hi], points[lo]), reference_up)
        tangent_length = norm(tangent)
        right_axis = unit(cross(reference_up, tangent)) if tangent_length >= min_local_travel_m else None
        lateral = dot(horizontal(sub(xyz, points[closest_index]), reference_up), right_axis) if right_axis else 0.0
        side = "right" if lateral > side_dead_zone_m else "left" if lateral < -side_dead_zone_m else "near_centerline"
        neighborhood = range(max(0, closest_index - 1), min(len(points), closest_index + 2))
        stable = []
        if right_axis:
            for index in neighborhood:
                value = dot(horizontal(sub(xyz, points[index]), reference_up), right_axis)
                stable.append("right" if value > side_dead_zone_m else "left" if value < -side_dead_zone_m else "near_centerline")
        side_support = sum(value == side for value in stable) / len(stable) if stable else 0.0
        valid_pass = (
            1 < closest_index < len(points) - 2
            and distances[closest_index] <= max_landmark_distance_m
            and tangent_length >= min_local_travel_m
            and side in {"left", "right"}
            and side_support >= 2 / 3
        )
        visit_prominence = min(distances[0], distances[-1]) - distances[closest_index]
        valid_visit = (
            1 < closest_index < len(points) - 2
            and distances[closest_index] <= max_landmark_distance_m
            and visit_prominence >= 0.15
        )
        landmark_rows.append({
            "landmark_id": landmark_id,
            "display_name": display_name,
            "center_world_m": [float(value) for value in xyz],
            "static_scene_landmark": landmark.get("static_scene_landmark") is True,
            "grounding": landmark.get("grounding"),
            "closest_index": closest_index,
            "closest_frame": frames[closest_index],
            "closest_time_sec": times[closest_index],
            "closest_horizontal_distance_m": distances[closest_index],
            "local_window_indices": [lo, hi],
            "local_travel_m": tangent_length,
            "signed_lateral_m": lateral,
            "pass_side": side,
            "side_support_ratio": side_support,
            "valid_local_pass": valid_pass,
            "visit_prominence_m": visit_prominence,
            "valid_visit": valid_visit,
        })

    order_pairs = []
    nearby = [row for row in landmark_rows if row["valid_visit"]]
    for left, right in combinations(nearby, 2):
        gap = abs(left["closest_time_sec"] - right["closest_time_sec"] )
        if gap < min_order_gap_sec:
            continue
        first, second = sorted((left, right), key=lambda row: row["closest_time_sec"] )
        center_separation = norm(horizontal(sub(first["center_world_m"], second["center_world_m"]), reference_up))
        order_pairs.append({
            "first_landmark": first["landmark_id"],
            "second_landmark": second["landmark_id"],
            "first_time_sec": first["closest_time_sec"],
            "second_time_sec": second["closest_time_sec"],
            "time_gap_sec": gap,
            "center_separation_m": center_separation,
            "max_closest_distance_m": max(first["closest_horizontal_distance_m"], second["closest_horizontal_distance_m"]),
        })
    order_pairs.sort(key=lambda row: (-row["time_gap_sec"], row["max_closest_distance_m"]))

    proximity_order_pairs = []
    for left, right in combinations(landmark_rows, 2):
        center_separation = norm(horizontal(sub(left["center_world_m"], right["center_world_m"]), reference_up))
        gap = abs(left["closest_time_sec"] - right["closest_time_sec"] )
        if (
            left["display_name"] == right["display_name"]
            or center_separation < 0.25
            or gap < min_order_gap_sec
            or max(left["closest_horizontal_distance_m"], right["closest_horizontal_distance_m"]) > max_landmark_distance_m
        ):
            continue
        first, second = sorted((left, right), key=lambda row: row["closest_time_sec"] )
        proximity_order_pairs.append({
            "first_landmark": first["landmark_id"],
            "second_landmark": second["landmark_id"],
            "first_time_sec": first["closest_time_sec"],
            "second_time_sec": second["closest_time_sec"],
            "time_gap_sec": gap,
            "center_separation_m": center_separation,
            "max_closest_distance_m": max(first["closest_horizontal_distance_m"], second["closest_horizontal_distance_m"]),
        })
    proximity_order_pairs.sort(key=lambda row: (-row["time_gap_sec"], row["max_closest_distance_m"]))

    flank_pairs = []
    valid_passes = [row for row in landmark_rows if row["valid_local_pass"]]
    for left, right in combinations(valid_passes, 2):
        if {left["pass_side"], right["pass_side"]} != {"left", "right"}:
            continue
        left_side = left if left["pass_side"] == "left" else right
        right_side = right if right["pass_side"] == "right" else left
        flank_pairs.append({
            "left_landmark": left_side["landmark_id"],
            "right_landmark": right_side["landmark_id"],
            "time_gap_sec": abs(left["closest_time_sec"] - right["closest_time_sec"]),
            "minimum_side_margin_m": min(abs(left["signed_lateral_m"]), abs(right["signed_lateral_m"])),
        })
    flank_pairs.sort(key=lambda row: (-row["minimum_side_margin_m"], row["time_gap_sec"]))

    nearest_states = []
    for index, point in enumerate(points):
        ranked = sorted(
            ({
                "landmark_id": row["landmark_id"],
                "distance_m": norm(horizontal(sub(row["center_world_m"], point), reference_up)),
            } for row in landmark_rows),
            key=lambda row: row["distance_m"],
        )
        if ranked:
            nearest_states.append({
                "frame": frames[index],
                "t": times[index],
                "nearest_landmark": ranked[0]["landmark_id"],
                "distance_m": ranked[0]["distance_m"],
                "runner_up_margin_m": ranked[1]["distance_m"] - ranked[0]["distance_m"] if len(ranked) > 1 else None,
            })
    endpoint_run = max(2, min(5, len(nearest_states) // 5))
    start_ids = [row["nearest_landmark"] for row in nearest_states[:endpoint_run]]
    end_ids = [row["nearest_landmark"] for row in nearest_states[-endpoint_run:]]
    nearest_change = None
    if nearest_states and len(set(start_ids)) == 1 and len(set(end_ids)) == 1:
        start_state, end_state = nearest_states[0], nearest_states[-1]
        margins = [row["runner_up_margin_m"] for row in nearest_states[:endpoint_run] + nearest_states[-endpoint_run:] if row["runner_up_margin_m"] is not None]
        nearest_change = {
            "start_landmark": start_ids[0],
            "end_landmark": end_ids[0],
            "changed": start_ids[0] != end_ids[0],
            "endpoint_run_length": endpoint_run,
            "minimum_endpoint_margin_m": min(margins, default=0.0),
            "start_distance_m": start_state["distance_m"],
            "end_distance_m": end_state["distance_m"],
            "valid": start_ids[0] != end_ids[0] and min(margins, default=0.0) >= 0.10,
        }

    return {
        "status": "ok",
        "coordinate_frame": "metric world trajectory projected onto the local horizontal plane",
        "side_definition": "left/right uses the local smoothed path tangent at closest approach, not image coordinates or the start-end chord",
        "trajectory_states": states,
        "trajectory_state_count": len(states),
        "temporal_span_sec": times[-1] - times[0],
        "path_length_m": path_length,
        "net_displacement_m": net_displacement,
        "max_smoothed_speed_mps": max(speeds, default=0.0),
        "reference_up_world_unit": reference_up,
        "landmarks": landmark_rows,
        "valid_pass_landmarks": [row["landmark_id"] for row in valid_passes],
        "valid_visit_landmarks": [row["landmark_id"] for row in landmark_rows if row["valid_visit"]],
        "order_pairs": order_pairs,
        "proximity_order_pairs": proximity_order_pairs,
        "flank_pairs": flank_pairs,
        "nearest_landmark_states": nearest_states,
        "route_landmark_ranking": sorted(
            ({
                "landmark_id": row["landmark_id"],
                "display_name": row["display_name"],
                "static_scene_landmark": row["static_scene_landmark"],
                "grounding": row["grounding"],
                "minimum_horizontal_distance_m": row["closest_horizontal_distance_m"],
                "closest_time_sec": row["closest_time_sec"],
            } for row in landmark_rows),
            key=lambda row: row["minimum_horizontal_distance_m"],
        ),
        "nearest_landmark_change": nearest_change,
        "thresholds": {
            "smoothing_radius": smoothing_radius,
            "side_dead_zone_m": side_dead_zone_m,
            "max_landmark_distance_m": max_landmark_distance_m,
            "min_local_travel_m": min_local_travel_m,
            "min_order_gap_sec": min_order_gap_sec,
        },
    }
