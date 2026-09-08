"""Geometry and temporal event mining for Task 5 human-state reasoning.

The module is dataset-agnostic: callers provide metric, time-aligned wearer
poses, gaze rays, and 3D object boxes.  Natural-language answers are never
used to infer labels.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


Vec3 = Sequence[float]


def dot(a: Vec3, b: Vec3) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def sub(a: Vec3, b: Vec3) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def norm(a: Vec3) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec3) -> list[float] | None:
    length = norm(a)
    return [float(x) / length for x in a] if length > 1e-9 else None


def mat_vec(matrix: Sequence[Sequence[float]], vector: Vec3) -> list[float]:
    return [dot(row, vector) for row in matrix]


def transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[j][i]) for j in range(3)] for i in range(3)]


def quaternion_rotation_xyzw(quaternion: Vec3) -> list[list[float]]:
    """Return a 3x3 rotation for an (x, y, z, w) quaternion."""
    x, y, z, w = (float(value) for value in quaternion)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1e-9:
        raise ValueError("degenerate quaternion")
    x, y, z, w = x / length, y / length, z / length, w / length
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def ray_aabb_distance(origin: Vec3, direction: Vec3, bounds: Sequence[Sequence[float]]) -> float | None:
    """Distance parameter to the first positive ray/AABB intersection."""
    lo, hi = 0.0, math.inf
    for axis in range(3):
        value = float(direction[axis])
        minimum, maximum = (float(v) for v in bounds[axis])
        if abs(value) <= 1e-12:
            if float(origin[axis]) < minimum or float(origin[axis]) > maximum:
                return None
            continue
        first = (minimum - float(origin[axis])) / value
        second = (maximum - float(origin[axis])) / value
        if first > second:
            first, second = second, first
        lo, hi = max(lo, first), min(hi, second)
        if hi < lo:
            return None
    return lo if lo > 0 else hi if hi > 0 else None


def body_centric_relation(
    wearer_world: Vec3,
    object_world: Vec3,
    right_world: Vec3,
    forward_world: Vec3,
    up_world: Vec3 = (0.0, 1.0, 0.0),
    lateral_deadband_m: float = 0.12,
    forward_deadband_m: float = 0.12,
) -> dict[str, Any]:
    """Classify an object in a gravity-aligned wearer frame."""
    relative = sub(object_world, wearer_world)
    up = unit(up_world)
    if up is None:
        raise ValueError("invalid up vector")
    right = unit(sub(right_world, [dot(right_world, up) * x for x in up]))
    forward = unit(sub(forward_world, [dot(forward_world, up) * x for x in up]))
    if right is None or forward is None:
        raise ValueError("wearer axes collapse after gravity alignment")
    lateral, longitudinal, vertical = dot(relative, right), dot(relative, forward), dot(relative, up)
    side = "left" if lateral < -lateral_deadband_m else "right" if lateral > lateral_deadband_m else "center"
    depth = "behind" if longitudinal < -forward_deadband_m else "front" if longitudinal > forward_deadband_m else "level"
    label = depth if side == "center" else side if depth == "level" else f"{side}-{depth}"
    return {
        "label": label,
        "side": side,
        "depth": depth,
        "right_m": lateral,
        "forward_m": longitudinal,
        "up_m": vertical,
        "distance_m": norm(relative),
    }


def contiguous_runs(states: Sequence[Mapping[str, Any]], key: str = "gazed_object_id") -> list[list[Mapping[str, Any]]]:
    runs: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    for state in states:
        if current and state.get(key) != current[-1].get(key):
            runs.append(current)
            current = []
        current.append(state)
    if current:
        runs.append(current)
    return runs


def supported_gaze_events(
    states: Sequence[Mapping[str, Any]], minimum_run: int = 4, maximum_internal_gap: int = 1,
) -> list[dict[str, Any]]:
    """Return sustained gaze events, merging only tiny no-hit gaps."""
    labels = [state.get("gazed_object_id") for state in states]
    merged = labels[:]
    index = 0
    while index < len(labels):
        if labels[index] is not None:
            index += 1
            continue
        gap_start = index
        while index < len(labels) and labels[index] is None:
            index += 1
        gap_end = index
        left = labels[gap_start - 1] if gap_start else None
        right = labels[gap_end] if gap_end < len(labels) else None
        if 0 < gap_end - gap_start <= maximum_internal_gap and left is not None and left == right:
            merged[gap_start:gap_end] = [left] * (gap_end - gap_start)
    enriched = [dict(state, _event_object=label) for state, label in zip(states, merged)]
    events = []
    for run in contiguous_runs(enriched, "_event_object"):
        object_id = run[0].get("_event_object")
        if object_id is None or len(run) < minimum_run:
            continue
        middle = run[len(run) // 2]
        direct_hit_count = sum(state.get("gazed_object_id") == object_id for state in run)
        events.append({
            "object_id": object_id,
            "object_name": middle.get("gazed_object_name"),
            "start_index": int(run[0]["frame_index"]),
            "end_index": int(run[-1]["frame_index"]),
            "start_time_s": float(run[0]["time_s"]),
            "end_time_s": float(run[-1]["time_s"]),
            "state_count": len(run),
            "direct_hit_count": direct_hit_count,
            "merged_gap_count": len(run) - direct_hit_count,
            "hit_support_ratio": direct_hit_count / len(run),
            "median_state": dict(middle),
        })
    return events


def circular_yaw_change_deg(first_forward: Vec3, second_forward: Vec3, up_world: Vec3 = (0.0, 1.0, 0.0)) -> float:
    up = unit(up_world)
    a = unit(sub(first_forward, [dot(first_forward, up) * x for x in up])) if up else None
    b = unit(sub(second_forward, [dot(second_forward, up) * x for x in up])) if up else None
    if a is None or b is None:
        return 0.0
    cosine = max(-1.0, min(1.0, dot(a, b)))
    return math.degrees(math.acos(cosine))


def relation_transition(states: Sequence[Mapping[str, Any]], object_id: str, start: int, end: int) -> dict[str, Any]:
    selected = [state for state in states if start <= int(state["frame_index"]) <= end]
    if not selected:
        raise ValueError("empty transition window")
    relations = [state["object_relations"][object_id] for state in selected]
    return {
        "object_id": object_id,
        "start": dict(selected[0], relation=relations[0]),
        "end": dict(selected[-1], relation=relations[-1]),
        "relation_sequence": [row["label"] for row in relations],
        "wearer_displacement_m": norm(sub(selected[-1]["wearer_world_m"], selected[0]["wearer_world_m"])),
        "wearer_turn_deg": circular_yaw_change_deg(selected[0]["forward_world"], selected[-1]["forward_world"]),
    }
