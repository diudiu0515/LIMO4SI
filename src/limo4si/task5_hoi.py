"""Annotation-only geometry for Task 5C human-object interaction.

BEHAVE's packed 30 fps files contain time-aligned SMPL-H root translations
and registered object poses.  This module deliberately does not infer contact
or gaze: both require annotations that are not present in the compact pack.
"""
from __future__ import annotations

import math
from typing import Any, Sequence


Vec3 = Sequence[float]


def _sub(a: Vec3, b: Vec3) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def _dot(a: Vec3, b: Vec3) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a: Vec3) -> float:
    return math.sqrt(_dot(a, a))


def rodrigues(axis_angle: Vec3) -> list[list[float]]:
    """Convert an axis-angle vector to a rotation matrix."""
    x, y, z = (float(value) for value in axis_angle)
    theta = math.sqrt(x * x + y * y + z * z)
    if theta <= 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    x, y, z = x / theta, y / theta, z / theta
    c, s, t = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return [
        [t*x*x+c, t*x*y-s*z, t*x*z+s*y],
        [t*x*y+s*z, t*y*y+c, t*y*z-s*x],
        [t*x*z-s*y, t*y*z+s*x, t*z*z+c],
    ]


def _transpose(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    return [[float(matrix[j][i]) for j in range(3)] for i in range(3)]


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Vec3) -> list[float]:
    return [_dot(row, vector) for row in matrix]


def body_relative_vector(human_translation: Vec3, root_axis_angle: Vec3, object_center: Vec3) -> list[float]:
    """Object center in the fitted SMPL-H root frame (+X right, +Z front)."""
    return _mat_vec(_transpose(rodrigues(root_axis_angle)), _sub(object_center, human_translation))


def relation_label(relative: Vec3, deadband_m: float = 0.15) -> str:
    right, _, front = (float(value) for value in relative)
    side = "left" if right < -deadband_m else "right" if right > deadband_m else "center"
    depth = "behind" if front < -deadband_m else "front" if front > deadband_m else "level"
    return depth if side == "center" else side if depth == "level" else f"{side}-{depth}"


def synchronized_indices(human_times: Sequence[str], object_times: Sequence[str]) -> list[tuple[int, int, str]]:
    """Return exact timestamp matches, rejecting duplicates rather than guessing."""
    human = {str(value): index for index, value in enumerate(human_times)}
    objects = {str(value): index for index, value in enumerate(object_times)}
    if len(human) != len(human_times) or len(objects) != len(object_times):
        raise ValueError("duplicate BEHAVE frame timestamp")
    return [(human[key], objects[key], key) for key in human if key in objects]


def analyze_window(
    human_translations: Sequence[Vec3], root_axis_angles: Sequence[Vec3],
    object_centers: Sequence[Vec3], *, fps: float = 30.0,
) -> dict[str, Any]:
    """Summarize an aligned window without claiming gaze or mesh contact."""
    count = len(human_translations)
    if count < 2 or len(root_axis_angles) != count or len(object_centers) != count:
        raise ValueError("aligned Task 5C window must contain at least two complete states")
    relatives = [
        body_relative_vector(human_translations[i], root_axis_angles[i], object_centers[i])
        for i in range(count)
    ]
    distances = [_norm(_sub(object_centers[i], human_translations[i])) for i in range(count)]
    labels = [relation_label(value) for value in relatives]
    reduced_labels: list[str] = []
    for value in labels:
        if not reduced_labels or reduced_labels[-1] != value:
            reduced_labels.append(value)
    edge = max(1, min(count // 3, round(fps)))
    start_distance = sorted(distances[:edge])[edge // 2]
    end_distance = sorted(distances[-edge:])[edge // 2]
    return {
        "state_count": count,
        "duration_s": (count - 1) / fps,
        "start_distance_m": start_distance,
        "end_distance_m": end_distance,
        "distance_change_m": end_distance - start_distance,
        "human_displacement_m": _norm(_sub(human_translations[-1], human_translations[0])),
        "object_displacement_m": _norm(_sub(object_centers[-1], object_centers[0])),
        "relative_relation_sequence": reduced_labels,
        "start_relative_xyz_m": relatives[0],
        "end_relative_xyz_m": relatives[-1],
        "evidence_scope": "BEHAVE fitted SMPL-H root and registered object 6DoF; no gaze/contact claim",
    }
