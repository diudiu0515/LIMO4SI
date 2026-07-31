"""Independent checks for metric person-to-object distances."""

from __future__ import annotations

from math import dist, sqrt
from typing import Mapping, Sequence


def _xyz(point: Mapping[str, float]) -> list[float]:
    return [float(point[axis]) for axis in "xyz"]


def _midpoint(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [(float(x) + float(y)) * 0.5 for x, y in zip(a, b)]


def validate_metric_distance(
    object_xyz_world: Sequence[float],
    human_frame: Mapping[str, Sequence[float]],
    human_xyz: Mapping[str, float],
    joints_3d: Mapping[str, Mapping[str, float]],
    *,
    tolerance_m: float = 1e-6,
) -> dict:
    """Cross-check distance computation and the world frame's metric scale."""

    world_distance = dist(object_xyz_world, human_frame["origin"])
    human_distance = sqrt(
        float(human_xyz["right"]) ** 2
        + float(human_xyz["up"]) ** 2
        + float(human_xyz["forward"]) ** 2
    )
    residual = abs(world_distance - human_distance)

    left_shoulder = _xyz(joints_3d["left-shoulder"])
    right_shoulder = _xyz(joints_3d["right-shoulder"])
    left_hip = _xyz(joints_3d["left-hip"])
    right_hip = _xyz(joints_3d["right-hip"])
    shoulder_width = dist(left_shoulder, right_shoulder)
    hip_width = dist(left_hip, right_hip)
    torso_length = dist(
        _midpoint(left_shoulder, right_shoulder),
        _midpoint(left_hip, right_hip),
    )
    skeleton_scale_plausible = (
        0.20 <= shoulder_width <= 0.60
        and 0.10 <= hip_width <= 0.50
        and 0.35 <= torso_length <= 0.80
    )
    return {
        "world_direct_m": world_distance,
        "human_components_m": human_distance,
        "agreement_residual_m": residual,
        "agreement_pass": residual <= tolerance_m,
        "skeleton_scale": {
            "shoulder_width_m": shoulder_width,
            "hip_width_m": hip_width,
            "torso_length_m": torso_length,
            "plausible": skeleton_scale_plausible,
        },
        "validated": residual <= tolerance_m and skeleton_scale_plausible,
    }
