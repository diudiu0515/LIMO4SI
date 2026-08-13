"""Perspective-grounded QA geometry helpers.

The functions in this module turn the existing spatial-relation outputs into
richer perspective-taking answers. They are deliberately evidence-aware: every
answer includes the evidence that was used, approximations, and missing fields
instead of silently hallucinating unavailable observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import acos, degrees, sqrt
from typing import Mapping, Sequence

import numpy as np

from .human_frame import HumanFrame, build_human_frame, describe_relation

Vec3 = tuple[float, float, float]


def _arr(value: Sequence[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    if not np.isfinite(arr).all():
        raise ValueError(f"Expected finite xyz, got {value!r}")
    return arr


def _dist(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(_arr(a) - _arr(b)))


def _unit(v: Sequence[float], fallback: Sequence[float] | None = None) -> np.ndarray:
    arr = _arr(v)
    n = np.linalg.norm(arr)
    if n < 1e-8:
        if fallback is None:
            raise ValueError("Cannot normalize degenerate vector")
        return _unit(fallback)
    return arr / n


def _joint_xyz(joints: Mapping[str, Mapping[str, float] | Sequence[float]], name: str) -> np.ndarray | None:
    variants = (name, name.replace("_", "-"), name.replace("-", "_"))
    for key in variants:
        value = joints.get(key)
        if value is None:
            continue
        if isinstance(value, Mapping):
            if all(axis in value for axis in "xyz"):
                return _arr([value[axis] for axis in "xyz"])
        else:
            try:
                return _arr(value)  # type: ignore[arg-type]
            except Exception:
                pass
    return None


def _joint_dict(joints: Mapping[str, Mapping[str, float] | Sequence[float]]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key, value in joints.items():
        p = _joint_xyz({key: value}, key)
        if p is not None:
            out[key.replace("-", "_")] = p.tolist()
    return out


def relation_phrase(relation: Mapping) -> str:
    parts = []
    lateral = relation.get("lateral_relation") or relation.get("lateral")
    longitudinal = relation.get("longitudinal_relation") or relation.get("longitudinal")
    vertical = relation.get("vertical_relation") or relation.get("vertical")
    if lateral and lateral != "same_lateral_position":
        parts.append(str(lateral))
    if longitudinal and longitudinal != "same_longitudinal_position":
        parts.append(str(longitudinal))
    if vertical and vertical != "same_height":
        parts.append(str(vertical))
    return " and ".join(parts) if parts else "roughly aligned with the person"


@dataclass(frozen=True)
class ObserverPose:
    """Minimal observer geometry for perspective-taking."""

    origin: Vec3
    forward: Vec3
    right: Vec3
    up: Vec3
    source: str
    approximations: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "origin": list(self.origin),
            "forward": list(self.forward),
            "right": list(self.right),
            "up": list(self.up),
            "source": self.source,
            "approximations": list(self.approximations),
        }


def observer_from_joints(joints: Mapping[str, Mapping[str, float] | Sequence[float]]) -> ObserverPose:
    """Estimate eye/head origin and viewing direction from EgoPose joints."""

    normalized = _joint_dict(joints)
    frame = build_human_frame(normalized)
    approx: list[str] = []

    nose = _joint_xyz(joints, "nose")
    left_eye = _joint_xyz(joints, "left_eye")
    right_eye = _joint_xyz(joints, "right_eye")
    left_ear = _joint_xyz(joints, "left_ear")
    right_ear = _joint_xyz(joints, "right_ear")
    left_shoulder = _joint_xyz(joints, "left_shoulder")
    right_shoulder = _joint_xyz(joints, "right_shoulder")

    if left_eye is not None and right_eye is not None:
        origin = (left_eye + right_eye) / 2.0
        source = "eye_midpoint"
    elif nose is not None:
        origin = nose
        source = "nose"
        approx.append("eye origin approximated by nose joint")
    elif left_shoulder is not None and right_shoulder is not None:
        origin = (left_shoulder + right_shoulder) / 2.0 + 0.22 * _arr(frame.up)
        source = "shoulder_center_plus_up_offset"
        approx.append("head origin approximated from shoulder center")
    else:
        origin = _arr(frame.origin) + 0.55 * _arr(frame.up)
        source = "pelvis_plus_up_offset"
        approx.append("head origin approximated from pelvis")

    if nose is not None and left_eye is not None and right_eye is not None:
        eye_mid = (left_eye + right_eye) / 2.0
        forward = _unit(nose - eye_mid, frame.forward)
        direction_source = "eyes_to_nose"
    elif nose is not None and left_ear is not None and right_ear is not None:
        ear_mid = (left_ear + right_ear) / 2.0
        forward = _unit(nose - ear_mid, frame.forward)
        direction_source = "ears_to_nose"
    else:
        forward = _unit(frame.forward)
        direction_source = "body_forward"
        approx.append("view direction approximated by body-forward axis")

    return ObserverPose(
        origin=tuple(float(x) for x in origin),
        forward=tuple(float(x) for x in forward),
        right=frame.right,
        up=frame.up,
        source=direction_source + "/" + source,
        approximations=tuple(approx),
    )


def human_centric_answer(object_xyz_world: Sequence[float], human_frame: HumanFrame | Mapping, *, dead_zone_m: float = 0.15) -> dict:
    """Return person-centric relation and distance."""

    frame = human_frame if isinstance(human_frame, HumanFrame) else HumanFrame(
        origin=tuple(human_frame["origin"]),
        right=tuple(human_frame["right"]),
        up=tuple(human_frame["up"]),
        forward=tuple(human_frame["forward"]),
    )
    human_xyz = frame.world_to_human(object_xyz_world)
    relation = describe_relation(human_xyz, dead_zone_m=dead_zone_m)
    return {
        "status": "ok",
        "answer_type": "human_centric_spatial",
        "relation": relation,
        "answer": f"Object is {relation_phrase(relation)} relative to the person; distance {relation['distance_m']:.2f} m.",
    }


def nearest_reachable_object(
    objects: Sequence[Mapping],
    joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    *,
    reach_radius_m: float | None = None,
    margin_m: float = 0.20,
) -> dict:
    """Choose the nearest object that is plausibly reachable by either wrist.

    If wrists are missing, falls back to pelvis distance and reports the
    approximation. If reach_radius_m is not supplied, it is estimated from
    shoulder width as max(0.55, 2.4 * shoulder_width) plus margin.
    """

    approx: list[str] = []
    normalized = _joint_dict(joints)
    frame = build_human_frame(normalized)
    left_wrist = _joint_xyz(joints, "left_wrist")
    right_wrist = _joint_xyz(joints, "right_wrist")
    wrists = [p for p in (left_wrist, right_wrist) if p is not None]
    ls = _joint_xyz(joints, "left_shoulder")
    rs = _joint_xyz(joints, "right_shoulder")
    if reach_radius_m is None:
        if ls is not None and rs is not None:
            reach_radius_m = max(0.55, 2.4 * float(np.linalg.norm(rs - ls))) + margin_m
        else:
            reach_radius_m = 0.80 + margin_m
            approx.append("reach radius defaulted because shoulder width is missing")

    scored = []
    for obj in objects:
        xyz = obj.get("object_xyz_world_m") or obj.get("object_xyz_world")
        if xyz is None:
            continue
        center = _arr(xyz)
        pelvis_d = _dist(center, frame.origin)
        if wrists:
            hand_d = min(float(np.linalg.norm(center - wrist)) for wrist in wrists)
            distance_source = "nearest_wrist"
        else:
            hand_d = pelvis_d
            distance_source = "pelvis_fallback"
            approx.append("wrist joints missing; reachability approximated by pelvis distance")
        scored.append({
            "object_id": obj.get("object_id") or obj.get("object_name") or "object",
            "distance_to_nearest_hand_m": hand_d,
            "distance_to_pelvis_m": pelvis_d,
            "reachable": bool(hand_d <= reach_radius_m),
            "distance_source": distance_source,
            "raw_object": dict(obj),
        })
    scored.sort(key=lambda row: (not row["reachable"], row["distance_to_nearest_hand_m"]))
    chosen = scored[0] if scored else None
    return {
        "status": "ok" if chosen else "missing_evidence",
        "answer_type": "reachability",
        "reach_radius_m": float(reach_radius_m),
        "chosen": chosen,
        "candidates": scored,
        "approximations": sorted(set(approx)),
        "answer": (
            f"{chosen['object_id']} is the nearest reachable candidate "
            f"({chosen['distance_to_nearest_hand_m']:.2f} m from nearest hand; reach radius {reach_radius_m:.2f} m)."
            if chosen and chosen["reachable"] else
            f"No candidate is within the estimated reach radius {reach_radius_m:.2f} m; nearest is {chosen['object_id']} at {chosen['distance_to_nearest_hand_m']:.2f} m."
            if chosen else "No object with 3D center was provided."
        ),
    }


def line_of_sight_occluders(
    observer_origin: Sequence[float],
    target_xyz: Sequence[float],
    candidates: Sequence[Mapping],
    *,
    tube_radius_m: float = 0.18,
    min_depth_gap_m: float = 0.05,
) -> list[dict]:
    """Return candidate objects lying between observer and target line segment."""

    origin = _arr(observer_origin)
    target = _arr(target_xyz)
    ray = target - origin
    length = float(np.linalg.norm(ray))
    if length < 1e-8:
        return []
    unit = ray / length
    hits = []
    for cand in candidates:
        xyz = cand.get("object_xyz_world_m") or cand.get("object_xyz_world")
        if xyz is None:
            continue
        center = _arr(xyz)
        if np.linalg.norm(center - target) < 1e-6:
            continue
        rel = center - origin
        depth = float(np.dot(rel, unit))
        if depth <= min_depth_gap_m or depth >= length - min_depth_gap_m:
            continue
        perp = float(np.linalg.norm(rel - depth * unit))
        quality = cand.get("quality") or {}
        radius_hint = float(quality.get("object_radius_m", tube_radius_m)) if isinstance(quality, Mapping) else tube_radius_m
        effective_radius = max(tube_radius_m, radius_hint)
        if perp <= effective_radius:
            hits.append({
                "object_id": cand.get("object_id") or cand.get("object_name") or "object",
                "depth_from_observer_m": depth,
                "target_depth_m": length,
                "perpendicular_distance_to_sightline_m": perp,
                "occlusion_margin_m": effective_radius - perp,
                "raw_object": dict(cand),
            })
    hits.sort(key=lambda row: (row["depth_from_observer_m"], -row["occlusion_margin_m"]))
    return hits


def visibility_answer(
    target: Mapping,
    joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    *,
    candidates: Sequence[Mapping] = (),
    fov_degrees: float = 110.0,
    tube_radius_m: float = 0.18,
) -> dict:
    """Estimate if target is visible from observer head direction and occluders."""

    target_xyz = target.get("object_xyz_world_m") or target.get("object_xyz_world")
    if target_xyz is None:
        return {"status": "missing_evidence", "answer_type": "visibility", "missing_evidence": ["target 3D center"]}
    observer = observer_from_joints(joints)
    origin = _arr(observer.origin)
    target_arr = _arr(target_xyz)
    to_target = target_arr - origin
    dist = float(np.linalg.norm(to_target))
    if dist < 1e-8:
        angle = 0.0
    else:
        cosang = float(np.clip(np.dot(_unit(observer.forward), to_target / dist), -1.0, 1.0))
        angle = degrees(acos(cosang))
    inside_fov = angle <= fov_degrees / 2.0
    occluders = line_of_sight_occluders(origin, target_arr, candidates, tube_radius_m=tube_radius_m)
    visible = bool(inside_fov and not occluders)
    return {
        "status": "ok",
        "answer_type": "visibility_occlusion",
        "target_object_id": target.get("object_id") or target.get("object_name") or "target",
        "observer": observer.to_dict(),
        "angle_to_view_direction_deg": angle,
        "fov_degrees": fov_degrees,
        "inside_fov": inside_fov,
        "occluders": occluders,
        "visible": visible,
        "approximations": list(observer.approximations) + ["occlusion uses object centroids/tube unless dense masks are supplied"],
        "answer": (
            f"Likely visible: target is {angle:.1f}° from the viewing direction and no listed object blocks the sightline."
            if visible else
            f"Likely not visible: target is {angle:.1f}° from the viewing direction, outside the {fov_degrees:.0f}° field of view."
            if not inside_fov else
            f"Likely occluded by {occluders[0]['object_id']} along the observer-to-target sightline."
        ),
    }


def level2_occlusion_answer(
    observer_joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    target: Mapping,
    candidates: Sequence[Mapping],
    *,
    tube_radius_m: float = 0.18,
) -> dict:
    """Answer which object blocks target from another observer's viewpoint."""

    target_xyz = target.get("object_xyz_world_m") or target.get("object_xyz_world")
    if target_xyz is None:
        return {"status": "missing_evidence", "answer_type": "level2_occlusion", "missing_evidence": ["target 3D center"]}
    observer = observer_from_joints(observer_joints)
    occluders = line_of_sight_occluders(observer.origin, target_xyz, candidates, tube_radius_m=tube_radius_m)
    blocker = occluders[0] if occluders else None
    return {
        "status": "ok",
        "answer_type": "level2_perspective_occlusion",
        "observer": observer.to_dict(),
        "target_object_id": target.get("object_id") or target.get("object_name") or "target",
        "blocker": blocker,
        "occluders": occluders,
        "approximations": list(observer.approximations) + ["single-observer demo when only one person pose is available"],
        "answer": (
            f"From the observer's perspective, {blocker['object_id']} is the first listed object blocking the target."
            if blocker else "No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates."
        ),
    }


def reference_frame_switching_answer(
    object_xyz_world: Sequence[float],
    human_frame: HumanFrame | Mapping,
    *,
    camera_intrinsics: Sequence[Sequence[float]] | None = None,
    camera_extrinsics: Sequence[Sequence[float]] | None = None,
    world_axes: Mapping[str, Sequence[float]] | None = None,
) -> dict:
    """Describe one object in human, camera/egocentric, and world frames."""

    frame = human_frame if isinstance(human_frame, HumanFrame) else HumanFrame(
        origin=tuple(human_frame["origin"]),
        right=tuple(human_frame["right"]),
        up=tuple(human_frame["up"]),
        forward=tuple(human_frame["forward"]),
    )
    obj = _arr(object_xyz_world)
    human = human_centric_answer(obj, frame)
    result = {
        "status": "ok",
        "answer_type": "reference_frame_switching",
        "human_centric": human["relation"],
        "allocentric_world_xyz_m": obj.tolist(),
        "allocentric_note": "Raw Ego-Exo4D world coordinates. Semantic room axes require a scene/world-axis declaration.",
        "egocentric_camera": None,
        "missing_evidence": [],
    }
    if camera_extrinsics is not None:
        ext = np.asarray(camera_extrinsics, dtype=np.float64).reshape(3, 4)
        cam_xyz = obj @ ext[:, :3].T + ext[:, 3]
        result["egocentric_camera"] = {
            "camera_xyz_m": cam_xyz.tolist(),
            "right_m": float(cam_xyz[0]),
            "down_or_up_depends_on_camera_y_m": float(cam_xyz[1]),
            "forward_depth_m": float(cam_xyz[2]),
        }
        if camera_intrinsics is not None:
            K = np.asarray(camera_intrinsics, dtype=np.float64).reshape(3, 3)
            pix_h = cam_xyz @ K.T
            if abs(float(pix_h[2])) > 1e-8:
                result["egocentric_camera"]["pixel_xy"] = (pix_h[:2] / pix_h[2]).tolist()
    else:
        result["missing_evidence"].append("camera_extrinsics for egocentric answer")
    if world_axes:
        rel = {}
        origin = _arr(world_axes.get("origin", [0, 0, 0]))
        delta = obj - origin
        for name, axis in world_axes.items():
            if name == "origin":
                continue
            rel[name] = float(np.dot(delta, _unit(axis)))
        result["allocentric_axis_components_m"] = rel
    else:
        result["missing_evidence"].append("semantic world_axes for room-level allocentric labels")
    result["answer"] = {
        "human_centric": human["answer"],
        "egocentric": result["egocentric_camera"] or "missing camera extrinsics",
        "allocentric": result.get("allocentric_axis_components_m") or result["allocentric_world_xyz_m"],
    }
    return result
