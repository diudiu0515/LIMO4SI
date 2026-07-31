"""Geometry for expressing world-space objects in a human-centric frame.

The expected convention is:
    +X: the person's right
    +Y: up
    +Z: the person's front

All input points must already be expressed in the same metric world frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Mapping, Sequence

Vec3 = tuple[float, float, float]


def _vec(value: Sequence[float]) -> Vec3:
    if len(value) != 3:
        raise ValueError(f"Expected xyz with length 3, got {value!r}")
    return float(value[0]), float(value[1]), float(value[2])


def _add(a: Vec3, b: Vec3) -> Vec3:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def _sub(a: Vec3, b: Vec3) -> Vec3:
    return a[0] - b[0], a[1] - b[1], a[2] - b[2]


def _scale(a: Vec3, s: float) -> Vec3:
    return a[0] * s, a[1] * s, a[2] * s


def _dot(a: Vec3, b: Vec3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a: Vec3) -> float:
    return sqrt(_dot(a, a))


def _unit(a: Vec3, name: str) -> Vec3:
    length = _norm(a)
    if length < 1e-8:
        raise ValueError(f"Cannot normalize degenerate {name} axis")
    return _scale(a, 1.0 / length)


def _midpoint(a: Vec3, b: Vec3) -> Vec3:
    return _scale(_add(a, b), 0.5)


@dataclass(frozen=True)
class HumanFrame:
    """Rigid frame with axes stored in world coordinates."""

    origin: Vec3
    right: Vec3
    up: Vec3
    forward: Vec3

    def world_to_human(self, point_world: Sequence[float]) -> Vec3:
        delta = _sub(_vec(point_world), self.origin)
        return (
            _dot(delta, self.right),
            _dot(delta, self.up),
            _dot(delta, self.forward),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def build_human_frame(
    joints: Mapping[str, Sequence[float]],
    *,
    origin_mode: str = "pelvis",
) -> HumanFrame:
    """Build a body frame from world-space 3D joints.

    Required joints are left/right shoulder and left/right hip. A nose point is
    strongly recommended because it resolves the front/back sign ambiguity.
    """

    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    missing = [name for name in required if name not in joints]
    if missing:
        raise ValueError(f"Missing required joints: {', '.join(missing)}")

    ls, rs = _vec(joints["left_shoulder"]), _vec(joints["right_shoulder"])
    lh, rh = _vec(joints["left_hip"]), _vec(joints["right_hip"])
    shoulder_center = _midpoint(ls, rs)
    pelvis = _midpoint(lh, rh)

    right = _unit(_sub(rs, ls), "right")
    up_hint = _unit(_sub(shoulder_center, pelvis), "up")
    up = _unit(_sub(up_hint, _scale(right, _dot(up_hint, right))), "up")
    forward = _unit(_cross(right, up), "forward")

    if "nose" in joints:
        face_hint = _sub(_vec(joints["nose"]), shoulder_center)
        face_hint = _sub(face_hint, _scale(up, _dot(face_hint, up)))
        if _norm(face_hint) > 1e-8 and _dot(forward, face_hint) < 0:
            forward = _scale(forward, -1.0)

    if origin_mode == "pelvis":
        origin = pelvis
    elif origin_mode == "shoulders":
        origin = shoulder_center
    else:
        raise ValueError("origin_mode must be 'pelvis' or 'shoulders'")

    return HumanFrame(origin=origin, right=right, up=up, forward=forward)


def describe_relation(
    object_xyz_human: Sequence[float],
    *,
    dead_zone_m: float = 0.15,
    vertical_dead_zone_m: float = 0.05,
) -> dict:
    """Describe lateral, longitudinal, and vertical relations independently."""

    x, y, z = _vec(object_xyz_human)
    horizontal_distance = sqrt(x * x + z * z)
    distance = sqrt(x * x + y * y + z * z)

    if x > dead_zone_m:
        lateral = "right"
    elif x < -dead_zone_m:
        lateral = "left"
    else:
        lateral = "same_lateral_position"

    if z > dead_zone_m:
        longitudinal = "front"
    elif z < -dead_zone_m:
        longitudinal = "behind"
    else:
        longitudinal = "same_longitudinal_position"

    if y > dead_zone_m:
        vertical = "above"
    elif y > vertical_dead_zone_m:
        vertical = "slightly_above"
    elif y < -dead_zone_m:
        vertical = "below"
    elif y < -vertical_dead_zone_m:
        vertical = "slightly_below"
    else:
        vertical = "same_height"

    return {
        "human_xyz_m": {"right": x, "up": y, "forward": z},
        "distance_m": distance,
        "horizontal_distance_m": horizontal_distance,
        "lateral_relation": lateral,
        "longitudinal_relation": longitudinal,
        "vertical_relation": vertical,
        "text_zh": (
            f"物体在人的{_ZH[lateral]}、{_ZH[longitudinal]}、{_ZH[vertical]}，"
            f"前向分量 {z:.2f} m，右向分量 {x:.2f} m，"
            f"高度差 {y:.2f} m，直线距离 {distance:.2f} m"
        ),
    }


_ZH = {
    "front": "前方",
    "behind": "后方",
    "left": "左侧",
    "right": "右侧",
    "same_lateral_position": "左右中线附近",
    "same_longitudinal_position": "前后原点附近",
    "above": "上方",
    "slightly_above": "略偏上",
    "same_height": "近似同高",
    "slightly_below": "略偏下",
    "below": "下方",
}
