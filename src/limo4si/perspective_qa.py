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





FINGERTIP_SUFFIXES = ("thumb_4", "index_4", "middle_4", "ring_4", "pinky_4")
FINGER_BASE_SUFFIXES = ("thumb_1", "index_1", "middle_1", "ring_1", "pinky_1")


def _side_points(joints: Mapping[str, Mapping[str, float] | Sequence[float]], side: str, suffixes: Sequence[str]) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for suffix in suffixes:
        point = _joint_xyz(joints, f"{side}_{suffix}")
        if point is not None:
            points.append(point)
    return points


def _merge_joints(*sources: Mapping[str, Mapping[str, float] | Sequence[float]] | None) -> dict:
    merged: dict = {}
    for source in sources:
        if source:
            merged.update(source)
    return merged


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> tuple[float, float]:
    seg = end - start
    length2 = float(np.dot(seg, seg))
    if length2 < 1e-10:
        return float(np.linalg.norm(point - start)), 0.0
    t = float(np.clip(np.dot(point - start, seg) / length2, 0.0, 1.0))
    closest = start + t * seg
    return float(np.linalg.norm(point - closest)), t


def _reach_obstacle_candidates(
    wrist: np.ndarray | None,
    target_center: np.ndarray,
    candidates: Sequence[Mapping],
    target_id: str,
    *,
    tube_radius_m: float = 0.12,
) -> list[dict]:
    if wrist is None:
        return []
    hits = []
    for cand in candidates:
        cand_id = cand.get("object_id") or cand.get("object_name") or "object"
        if cand_id == target_id:
            continue
        xyz = cand.get("object_xyz_world_m") or cand.get("object_xyz_world")
        if xyz is None:
            continue
        d, t = _segment_distance(_arr(xyz), wrist, target_center)
        if 0.05 < t < 0.95 and d <= tube_radius_m:
            hits.append({
                "object_id": cand_id,
                "distance_to_hand_target_segment_m": d,
                "segment_fraction_from_hand": t,
            })
    hits.sort(key=lambda row: row["distance_to_hand_target_segment_m"])
    return hits



def spatial_relation_quality_answer(
    sample: Mapping,
    *,
    min_distance_m: float = 0.60,
    dead_zone_m: float = 0.15,
) -> dict:
    """Audit a precomputed human-object spatial relation for QA confidence."""

    xyz = sample.get("object_xyz_world_m") or sample.get("object_xyz_world")
    frame = sample.get("human_frame")
    obj_id = sample.get("object_id") or sample.get("object_name") or "object"
    if xyz is None or frame is None:
        return {"status": "missing_evidence", "answer_type": "human_centric_spatial_quality", "target_object_id": obj_id, "missing_evidence": ["object 3D center or human frame"]}
    base = human_centric_answer(xyz, frame, dead_zone_m=dead_zone_m)
    rel = base["relation"]
    distance = float(rel["distance_m"])
    quality = sample.get("quality") or {}
    validation = sample.get("distance_validation") or {}
    points = int(quality.get("points_in_mask", 0) or 0) if isinstance(quality, Mapping) else 0
    inliers = int(quality.get("robust_inliers", 0) or 0) if isinstance(quality, Mapping) else 0
    validated = bool(validation.get("validated", True)) if isinstance(validation, Mapping) else True
    near_threshold = distance < min_distance_m
    relation_in_dead_zone = any(rel.get(k) in (None, "same_lateral_position", "same_longitudinal_position", "same_height") for k in ["lateral_relation", "longitudinal_relation"])
    confidence = "high"
    reasons = []
    if near_threshold:
        confidence = "low"
        reasons.append("object is too close to the person for stable left/right/front/back labels")
    if not validated:
        confidence = "low"
        reasons.append("metric distance validation failed")
    if points and points < 8:
        confidence = "medium_low" if confidence == "high" else confidence
        reasons.append("few point-cloud points fell inside the object mask")
    if relation_in_dead_zone:
        confidence = "medium" if confidence == "high" else confidence
        reasons.append("one relation axis is near the dead zone / roughly aligned")
    if not reasons:
        reasons.append("distance validation and relation thresholds are acceptable")
    return {
        "status": "ok",
        "answer_type": "human_centric_spatial_quality",
        "target_object_id": obj_id,
        "relation": rel,
        "distance_m": distance,
        "direction_confidence": confidence,
        "near_threshold": near_threshold,
        "min_distance_m": min_distance_m,
        "dead_zone_m": dead_zone_m,
        "quality_reasons": reasons,
        "point_cloud_points_in_mask": points,
        "robust_inliers": inliers,
        "distance_validation": validation,
        "answer": f"Object is {relation_phrase(rel)} relative to the person at {distance:.2f} m; direction confidence is {confidence}.",
    }

def nearest_object_analysis(
    objects: Sequence[Mapping],
    joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    *,
    hand_joints: Mapping[str, Mapping[str, float] | Sequence[float]] | None = None,
) -> dict:
    """Separate nearest-by-distance from easiest-to-reach.

    The dataset question "nearest" is answered by pelvis-to-object distance.
    A separate reachability-aware ranking uses current arm/hand geometry so the
    benchmark does not conflate physical nearness with accessibility.
    """

    if not objects:
        return {"status": "missing_evidence", "answer_type": "nearest_object_analysis", "missing_evidence": ["candidate objects"], "answer": "No candidate objects were provided."}
    distance_rows = []
    reach_rows = []
    for obj in objects:
        xyz = obj.get("object_xyz_world_m") or obj.get("object_xyz_world")
        if xyz is None:
            continue
        obj_id = obj.get("object_id") or obj.get("object_name") or "object"
        distance = float(obj.get("distance_m") or human_centric_answer(xyz, obj.get("human_frame") or build_human_frame(_joint_dict(joints)))["relation"]["distance_m"])
        distance_rows.append({"object_id": obj_id, "distance_m": distance, "raw_object": dict(obj)})
        reach = static_reachability_answer(obj, joints, hand_joints=hand_joints, candidates=objects)
        best_arm = reach.get("best_arm") or {}
        reach_score = 0.0
        if reach.get("reachable"):
            reach_score += 1.0
        if reach.get("hand_already_close"):
            reach_score += 0.35
        if reach.get("grasp_cue"):
            reach_score += 0.50
        if not reach.get("obstacle_free", True):
            reach_score -= 0.30
        reach_score += max(0.0, 0.4 - float(best_arm.get("wrist_to_target_m", 9.0)))
        reach_rows.append({
            "object_id": obj_id,
            "distance_m": distance,
            "reach_score": float(max(0.0, reach_score)),
            "reachable": bool(reach.get("reachable")),
            "hand_already_close": bool(reach.get("hand_already_close")),
            "grasp_cue": bool(reach.get("grasp_cue")),
            "obstacle_free": bool(reach.get("obstacle_free", True)),
            "static_reachability": reach,
        })
    if not distance_rows:
        return {"status": "missing_evidence", "answer_type": "nearest_object_analysis", "missing_evidence": ["candidate 3D centers"], "answer": "No candidate object has a 3D center."}
    distance_rows.sort(key=lambda row: row["distance_m"])
    reach_rows.sort(key=lambda row: (-row["reach_score"], row["distance_m"]))
    nearest = distance_rows[0]
    easiest = reach_rows[0] if reach_rows else None
    same = bool(easiest and nearest["object_id"] == easiest["object_id"])
    answer = (
        f"The nearest listed object is {nearest['object_id']} at {nearest['distance_m']:.2f} m. "
        f"The easiest-to-reach object by current arm/hand geometry is {easiest['object_id']}" +
        ("; these are the same object." if same else "; this is different from the nearest-by-distance object.")
        if easiest else f"The nearest listed object is {nearest['object_id']} at {nearest['distance_m']:.2f} m."
    )
    return {
        "status": "ok",
        "answer_type": "nearest_object_analysis",
        "nearest_by_distance": nearest,
        "easiest_to_reach": easiest,
        "same_object": same,
        "distance_ranked_candidates": distance_rows,
        "reachability_ranked_candidates": reach_rows,
        "approximations": ["easiest-to-reach is a geometric proxy from current pose, not an action label"],
        "answer": answer,
    }

def static_reachability_answer(
    target: Mapping,
    joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    *,
    hand_joints: Mapping[str, Mapping[str, float] | Sequence[float]] | None = None,
    candidates: Sequence[Mapping] = (),
    hand_margin_m: float = 0.12,
    close_hand_threshold_m: float = 0.25,
    fingertip_close_threshold_m: float = 0.12,
    grasp_fingertip_threshold_m: float = 0.16,
    obstacle_tube_radius_m: float = 0.12,
) -> dict:
    """Estimate static reachability, fingertip proximity, grasp cue, and obstacles.

    Evidence hierarchy:
    - arm span: shoulder + elbow + wrist from body pose;
    - fingertip proximity / grasp cue: hand pose fingertips when available;
    - obstacle cue: other candidate object centers near the hand-to-target segment.

    This still does not prove a real grasp: it is a geometric cue, not contact or
    object-state recognition.
    """

    xyz = target.get("object_xyz_world_m") or target.get("object_xyz_world")
    target_id = target.get("object_id") or target.get("object_name") or "target"
    if xyz is None:
        return {
            "status": "missing_evidence",
            "answer_type": "static_reachability",
            "target_object_id": target_id,
            "missing_evidence": ["target 3D center"],
            "answer": "Reachability cannot be computed because the target has no 3D center.",
        }
    center = _arr(xyz)
    merged = _merge_joints(joints, hand_joints)
    approximations: list[str] = []
    missing: list[str] = []
    arms = []
    for side in ("left", "right"):
        shoulder = _joint_xyz(merged, f"{side}_shoulder")
        elbow = _joint_xyz(merged, f"{side}_elbow")
        wrist = _joint_xyz(merged, f"{side}_wrist")
        if shoulder is None:
            missing.append(f"{side}_shoulder")
            continue
        if wrist is None:
            missing.append(f"{side}_wrist")
            continue
        shoulder_to_target = float(np.linalg.norm(center - shoulder))
        wrist_to_target = float(np.linalg.norm(center - wrist))
        if elbow is not None:
            arm_length = float(np.linalg.norm(elbow - shoulder) + np.linalg.norm(wrist - elbow) + hand_margin_m)
            length_source = "shoulder_elbow_wrist"
        else:
            arm_length = float(np.linalg.norm(wrist - shoulder) + hand_margin_m)
            length_source = "shoulder_wrist_without_elbow"
            approximations.append(f"{side} elbow missing; arm span uses shoulder-wrist distance")
        fingertips = _side_points(merged, side, FINGERTIP_SUFFIXES)
        finger_bases = _side_points(merged, side, FINGER_BASE_SUFFIXES)
        fingertip_distances = [float(np.linalg.norm(center - p)) for p in fingertips]
        finger_base_distances = [float(np.linalg.norm(center - p)) for p in finger_bases]
        min_fingertip_distance = min(fingertip_distances) if fingertip_distances else None
        close_fingertip_count = sum(d <= grasp_fingertip_threshold_m for d in fingertip_distances)
        hand_center = np.mean(fingertips + [wrist], axis=0) if fingertips else wrist
        hand_center_to_target = float(np.linalg.norm(center - hand_center))
        grasp_cue = bool(close_fingertip_count >= 2 or (min_fingertip_distance is not None and min_fingertip_distance <= fingertip_close_threshold_m))
        if not fingertips:
            missing.append(f"{side}_fingertips")
        obstacles = _reach_obstacle_candidates(wrist, center, candidates, target_id, tube_radius_m=obstacle_tube_radius_m)
        arms.append({
            "side": side,
            "arm_length_m": arm_length,
            "arm_length_source": length_source,
            "shoulder_to_target_m": shoulder_to_target,
            "wrist_to_target_m": wrist_to_target,
            "within_arm_span": bool(shoulder_to_target <= arm_length),
            "hand_already_close": bool(wrist_to_target <= close_hand_threshold_m),
            "reach_margin_m": arm_length - shoulder_to_target,
            "fingertip_count": len(fingertips),
            "min_fingertip_distance_m": min_fingertip_distance,
            "close_fingertip_count": int(close_fingertip_count),
            "finger_base_distance_min_m": min(finger_base_distances) if finger_base_distances else None,
            "hand_center_to_target_m": hand_center_to_target,
            "grasp_cue": grasp_cue,
            "obstacle_candidates": obstacles,
            "obstacle_count": len(obstacles),
        })
    if not arms:
        return {
            "status": "missing_evidence",
            "answer_type": "static_reachability",
            "target_object_id": target_id,
            "missing_evidence": sorted(set(missing)),
            "approximations": approximations,
            "answer": "Reachability cannot be computed because shoulder/wrist joints are missing.",
        }
    arms.sort(key=lambda row: (not row["within_arm_span"], row["obstacle_count"], row["wrist_to_target_m"], -row["reach_margin_m"]))
    best = arms[0]
    reachable = any(row["within_arm_span"] for row in arms)
    hand_close = any(row["hand_already_close"] for row in arms)
    grasp_cue = any(row["grasp_cue"] for row in arms)
    obstacle_count = min(row["obstacle_count"] for row in arms)
    obstacle_free = obstacle_count == 0
    if reachable and grasp_cue and obstacle_free:
        answer = f"The {target_id} is reachable, and the fingertips provide a possible grasp/contact cue."
    elif reachable and hand_close and obstacle_free:
        answer = f"The {target_id} is reachable, and the {best['side']} hand is already close to it ({best['wrist_to_target_m']:.2f} m)."
    elif reachable and not obstacle_free:
        answer = f"The {target_id} is within arm span, but another candidate object may lie between the hand and target."
    elif reachable:
        answer = f"The {target_id} is within arm span from the {best['side']} shoulder, but the hand/fingers are not yet very close."
    else:
        answer = f"The {target_id} is likely not reachable from the current pose; even the best arm is short by {-best['reach_margin_m']:.2f} m."
    return {
        "status": "ok",
        "answer_type": "static_reachability",
        "target_object_id": target_id,
        "reachable": bool(reachable),
        "hand_already_close": bool(hand_close),
        "grasp_cue": bool(grasp_cue),
        "obstacle_free": bool(obstacle_free),
        "best_arm": best,
        "arms": arms,
        "thresholds": {
            "hand_margin_m": hand_margin_m,
            "close_hand_threshold_m": close_hand_threshold_m,
            "fingertip_close_threshold_m": fingertip_close_threshold_m,
            "grasp_fingertip_threshold_m": grasp_fingertip_threshold_m,
            "obstacle_tube_radius_m": obstacle_tube_radius_m,
        },
        "missing_evidence": sorted(set(missing)),
        "approximations": sorted(set(approximations)),
        "answer": answer,
    }

def reach_for_intent(
    objects: Sequence[Mapping],
    pose_sequence: Sequence[Mapping[str, Mapping[str, float] | Sequence[float]]],
    *,
    current_joints: Mapping[str, Mapping[str, float] | Sequence[float]] | None = None,
    hand_joints: Mapping[str, Mapping[str, float] | Sequence[float]] | None = None,
    use_visibility_gate: bool = True,
    reach_radius_m: float | None = None,
    margin_m: float = 0.20,
    min_approach_m: float = 0.08,
) -> dict:
    """Infer which object a hand is reaching for from motion + current geometry.

    Evidence hierarchy:
    - short wrist trajectory: whether the hand is moving closer to each object;
    - static reachability: current arm span + wrist/hand proximity;
    - optional hand/finger pose: grasp/contact cue when present;
    - optional visibility: object should be at least geometrically visible/in FOV.

    This is still an intent proxy, not an action/contact label. When motion is
    weak, the result is explicitly marked as a proxy instead of pretending a true
    interaction was observed.
    """

    approx: list[str] = ["object centers are held fixed at the key frame during the reach-for window"]
    missing: list[str] = []
    valid_poses = [pose for pose in pose_sequence if pose]
    if not valid_poses:
        return {
            "status": "missing_evidence",
            "answer_type": "reach_for_intent",
            "missing_evidence": ["temporal wrist pose sequence"],
            "candidates": [],
            "chosen": None,
            "answer": "Reach-for intent cannot be estimated because no temporal wrist poses were provided.",
        }

    evidence_joints = current_joints or valid_poses[-1]
    first_pose = valid_poses[0]
    ls = _joint_xyz(first_pose, "left_shoulder")
    rs = _joint_xyz(first_pose, "right_shoulder")
    if reach_radius_m is None:
        if ls is not None and rs is not None:
            reach_radius_m = max(0.55, 2.4 * float(np.linalg.norm(rs - ls))) + margin_m
        else:
            reach_radius_m = 1.0
            approx.append("reach radius defaulted because shoulder width is missing")

    scored = []
    for obj in objects:
        xyz = obj.get("object_xyz_world_m") or obj.get("object_xyz_world")
        if xyz is None:
            missing.append(f"{obj.get('object_id') or obj.get('object_name') or 'object'} 3D center")
            continue
        center = _arr(xyz)
        left_series: list[float] = []
        right_series: list[float] = []
        nearest_series: list[float] = []
        for pose in valid_poses:
            lw = _joint_xyz(pose, "left_wrist")
            rw = _joint_xyz(pose, "right_wrist")
            distances = []
            if lw is not None:
                d = float(np.linalg.norm(center - lw)); left_series.append(d); distances.append(d)
            if rw is not None:
                d = float(np.linalg.norm(center - rw)); right_series.append(d); distances.append(d)
            if distances:
                nearest_series.append(min(distances))
        obj_id = obj.get("object_id") or obj.get("object_name") or "object"
        if not nearest_series:
            missing.append("temporal wrist joints")
            approx.append(f"wrist joints missing in temporal window; {obj_id} skipped")
            continue
        start_d = float(nearest_series[0])
        end_d = float(nearest_series[-1])
        min_d = float(min(nearest_series))
        approach_m = start_d - end_d
        best_hand = "left_wrist" if left_series and min(left_series) <= (min(right_series) if right_series else float("inf")) else "right_wrist"
        entering_reach = bool(start_d > reach_radius_m and min_d <= reach_radius_m)
        close_and_approaching = bool(min_d <= reach_radius_m and approach_m >= min_approach_m)
        already_close = bool(min_d <= max(0.25, 0.35 * reach_radius_m))

        static = static_reachability_answer(obj, evidence_joints, hand_joints=hand_joints, candidates=objects)
        visible_result = visibility_answer(obj, evidence_joints, candidates=objects) if use_visibility_gate else None
        static_reachable = bool(static.get("reachable"))
        hand_close = bool(static.get("hand_already_close"))
        grasp_cue = bool(static.get("grasp_cue"))
        obstacle_free = bool(static.get("obstacle_free", True))
        visible = True if visible_result is None else bool(visible_result.get("visible"))
        inside_fov = True if visible_result is None else bool(visible_result.get("inside_fov"))

        score = 0.0
        score += max(0.0, approach_m)
        score += max(0.0, reach_radius_m - min_d) * 0.45
        if entering_reach:
            score += 0.30
        if close_and_approaching:
            score += 0.25
        if already_close:
            score += 0.10
        if static_reachable:
            score += 0.18
        if hand_close:
            score += 0.12
        if grasp_cue:
            score += 0.30
        if not obstacle_free:
            score -= 0.20
        if use_visibility_gate and not inside_fov:
            score -= 0.25
        elif use_visibility_gate and not visible:
            score -= 0.12
        score = float(max(0.0, score))

        if grasp_cue:
            intent_state = "possible_contact_or_grasp"
        elif close_and_approaching or entering_reach:
            intent_state = "active_reach_for"
        elif static_reachable and already_close:
            intent_state = "near_hand_reachable_proxy"
        elif approach_m >= min_approach_m:
            intent_state = "approaching_but_not_close"
        else:
            intent_state = "no_clear_reach_motion"

        candidate_missing = sorted(set((static.get("missing_evidence") or []) + ([] if visible_result is None else (visible_result.get("missing_evidence") or []))))
        candidate_approx = sorted(set((static.get("approximations") or []) + ([] if visible_result is None else (visible_result.get("approximations") or []))))
        scored.append({
            "object_id": obj_id,
            "best_hand": best_hand,
            "start_distance_m": start_d,
            "end_distance_m": end_d,
            "min_distance_m": min_d,
            "approach_m": approach_m,
            "approaching": bool(approach_m >= min_approach_m),
            "entering_reach": entering_reach,
            "reachable_during_window": bool(min_d <= reach_radius_m),
            "static_reachable": static_reachable,
            "hand_already_close": hand_close,
            "grasp_cue": grasp_cue,
            "obstacle_free": obstacle_free,
            "visible_to_person": visible,
            "inside_fov": inside_fov,
            "intent_state": intent_state,
            "score": score,
            "distance_series_m": nearest_series,
            "static_reachability": static,
            "visibility": visible_result,
            "missing_evidence": candidate_missing,
            "approximations": candidate_approx,
            "raw_object": dict(obj),
        })
    scored.sort(key=lambda row: (-row["score"], row["min_distance_m"]))
    chosen = scored[0] if scored else None
    if chosen and chosen["score"] > 0:
        if chosen["intent_state"] == "possible_contact_or_grasp":
            answer = f"The strongest interaction cue is {chosen['object_id']}: the hand/fingers are close enough to give a possible grasp/contact cue."
        elif chosen["intent_state"] == "active_reach_for":
            answer = (
                f"The person appears to reach for {chosen['object_id']}: the nearest hand moves "
                f"{chosen['approach_m']:.2f} m closer over the window, with minimum hand-object distance "
                f"{chosen['min_distance_m']:.2f} m."
            )
        else:
            answer = (
                f"No strong reach-for motion is detected; {chosen['object_id']} is the best geometric proxy "
                f"based on hand distance, static reachability, and visibility evidence."
            )
        status = "ok"
    elif chosen:
        answer = (
            f"No clear reach-for motion is detected; the closest candidate is {chosen['object_id']} "
            f"with minimum hand-object distance {chosen['min_distance_m']:.2f} m."
        )
        status = "ok"
        approx.append("no positive interaction evidence; answer is closest-hand proxy")
    else:
        answer = "No object with both 3D center and temporal wrist evidence was available."
        status = "missing_evidence"
    return {
        "status": status,
        "answer_type": "reach_for_intent",
        "reach_radius_m": float(reach_radius_m),
        "window_pose_count": len(valid_poses),
        "min_approach_m": min_approach_m,
        "evidence_used": ["temporal_wrist_trajectory", "static_reachability", "hand_fingertips_if_available"] + (["visibility_gate"] if use_visibility_gate else []),
        "chosen": chosen,
        "candidates": scored,
        "missing_evidence": sorted(set(missing)),
        "approximations": sorted(set(approx)),
        "answer": answer,
    }


def unified_reach_analysis(
    target: Mapping,
    objects: Sequence[Mapping],
    current_joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    pose_sequence: Sequence[Mapping[str, Mapping[str, float] | Sequence[float]]] | None = None,
    *,
    hand_margin_m: float = 0.12,
    close_hand_threshold_m: float = 0.25,
    min_approach_m: float = 0.08,
) -> dict:
    """Unified reachability + reach-for analysis using arm geometry and motion.

    Static part: shoulder-elbow-wrist arm span, shoulder-to-object distance,
    wrist-to-object distance.

    Dynamic part: short temporal wrist trajectory and hand-object distance trend.

    The final state separates "can reach" from "is reaching" but exposes both
    under one result object so QA does not confuse static reachability with
    action intent.
    """

    target_id = target.get("object_id") or target.get("object_name") or "target"
    static = static_reachability_answer(
        target,
        current_joints,
        hand_margin_m=hand_margin_m,
        close_hand_threshold_m=close_hand_threshold_m,
    )
    seq = list(pose_sequence or [])
    dynamic = reach_for_intent(
        objects,
        seq if seq else [current_joints],
        min_approach_m=min_approach_m,
    )
    chosen = dynamic.get("chosen") or {}
    dynamic_target_match = chosen.get("object_id") == target.get("object_id")
    reaching_motion = bool(dynamic_target_match and chosen.get("approaching"))
    entering_reach = bool(dynamic_target_match and chosen.get("entering_reach"))
    static_reachable = bool(static.get("reachable"))
    hand_close = bool(static.get("hand_already_close"))

    if static_reachable and reaching_motion:
        reach_state = "reachable_and_reaching"
        answer = f"The {target_id} is reachable from the current arm pose, and the hand is moving toward it."
    elif static_reachable and hand_close:
        reach_state = "reachable_hand_close"
        answer = f"The {target_id} is reachable, and a hand is already close to it."
    elif static_reachable:
        reach_state = "reachable_not_reaching"
        answer = f"The {target_id} is within arm span, but the short hand trajectory does not clearly show reaching toward it."
    elif reaching_motion:
        reach_state = "reaching_but_not_yet_reachable"
        answer = f"The hand is moving toward the {target_id}, but the object is not yet within the current arm span."
    else:
        reach_state = "not_reachable_not_reaching"
        answer = f"The {target_id} is not reachable from the current arm pose, and the hand trajectory does not clearly target it."

    approximations = sorted(set((static.get("approximations") or []) + (dynamic.get("approximations") or [])))
    missing = sorted(set((static.get("missing_evidence") or []) + (dynamic.get("missing_evidence") or [])))
    return {
        "status": "missing_evidence" if static.get("status") == "missing_evidence" else "ok",
        "answer_type": "unified_reach_analysis",
        "target_object_id": target_id,
        "reach_state": reach_state,
        "static_reachable": static_reachable,
        "hand_already_close": hand_close,
        "reaching_motion": reaching_motion,
        "entering_reach": entering_reach,
        "static": static,
        "dynamic": dynamic,
        "approximations": approximations,
        "missing_evidence": missing,
        "answer": answer,
    }

def _candidate_radius_hint(candidate: Mapping, default_radius_m: float) -> tuple[float, str]:
    """Estimate a conservative object radius for centroid-based visibility tests."""

    quality = candidate.get("quality") or {}
    if isinstance(quality, Mapping):
        for key in ("object_radius_m", "radius_m", "extent_radius_m"):
            if quality.get(key) is not None:
                try:
                    return max(default_radius_m, float(quality[key])), f"quality.{key}"
                except Exception:
                    pass
        inliers = quality.get("robust_inliers") or quality.get("points_in_mask")
        if inliers is not None:
            try:
                # More inlier points usually means a larger/clearer mask. This is only a
                # radius hint, capped so it cannot dominate the geometry.
                radius = min(0.45, max(default_radius_m, 0.05 + 0.018 * sqrt(float(inliers))))
                return radius, "quality.point_count_radius_hint"
            except Exception:
                pass
    return default_radius_m, "default_tube_radius"


def line_of_sight_occluders(
    observer_origin: Sequence[float],
    target_xyz: Sequence[float],
    candidates: Sequence[Mapping],
    *,
    tube_radius_m: float = 0.18,
    min_depth_gap_m: float = 0.05,
    angular_block_threshold_deg: float = 3.0,
) -> list[dict]:
    """Return candidate objects lying between observer and target line segment.

    This is still geometric occlusion, not dense ray-casting. Compared with a
    raw centroid check, it now estimates each candidate's angular radius and
    reports an occlusion score/margin, which is easier to audit and less brittle.
    """

    origin = _arr(observer_origin)
    target = _arr(target_xyz)
    ray = target - origin
    length = float(np.linalg.norm(ray))
    if length < 1e-8:
        return []
    unit = ray / length
    target_id = None
    hits = []
    for cand in candidates:
        xyz = cand.get("object_xyz_world_m") or cand.get("object_xyz_world")
        if xyz is None:
            continue
        center = _arr(xyz)
        if np.linalg.norm(center - target) < 1e-6:
            target_id = cand.get("object_id") or cand.get("object_name") or target_id
            continue
        rel = center - origin
        depth = float(np.dot(rel, unit))
        if depth <= min_depth_gap_m or depth >= length - min_depth_gap_m:
            continue
        perp_vec = rel - depth * unit
        perp = float(np.linalg.norm(perp_vec))
        radius_hint, radius_source = _candidate_radius_hint(cand, tube_radius_m)
        # Angular separation between target ray and blocker center, and blocker apparent radius.
        angle_sep = degrees(acos(float(np.clip(depth / max(float(np.linalg.norm(rel)), 1e-8), -1.0, 1.0))))
        angular_radius = degrees(acos(float(np.clip(depth / max(sqrt(depth * depth + radius_hint * radius_hint), 1e-8), -1.0, 1.0))))
        metric_margin = radius_hint - perp
        angular_margin = angular_radius - angle_sep
        blocks = metric_margin >= 0.0 or angular_margin >= angular_block_threshold_deg
        if blocks:
            # 1.0 roughly means centerline fully covered by the radius hint; values near 0 are grazing.
            occlusion_score = max(0.0, min(1.0, (radius_hint - perp) / max(radius_hint, 1e-8)))
            hits.append({
                "object_id": cand.get("object_id") or cand.get("object_name") or "object",
                "depth_from_observer_m": depth,
                "target_depth_m": length,
                "depth_fraction_to_target": depth / length,
                "perpendicular_distance_to_sightline_m": perp,
                "occlusion_radius_m": radius_hint,
                "radius_source": radius_source,
                "occlusion_margin_m": metric_margin,
                "angular_separation_deg": angle_sep,
                "angular_radius_deg": angular_radius,
                "angular_margin_deg": angular_margin,
                "occlusion_score": occlusion_score,
                "raw_object": dict(cand),
            })
    hits.sort(key=lambda row: (row["depth_from_observer_m"], -row["occlusion_margin_m"], -row["angular_margin_deg"]))
    return hits


def visibility_answer(
    target: Mapping,
    joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    *,
    candidates: Sequence[Mapping] = (),
    fov_degrees: float = 110.0,
    central_fov_degrees: float = 70.0,
    peripheral_fov_degrees: float | None = None,
    tube_radius_m: float = 0.18,
) -> dict:
    """Estimate if target is visible from observer pose and listed occluders.

    Evidence hierarchy:
    - head/eye/nose direction when available, otherwise body-forward fallback;
    - field-of-view tier: central / peripheral / outside;
    - listed-object line-of-sight occlusion with radius/depth scoring.

    This does not claim true pixel visibility unless dense depth/mask ray-casting is
    supplied by a later module.
    """

    target_xyz = target.get("object_xyz_world_m") or target.get("object_xyz_world")
    target_id = target.get("object_id") or target.get("object_name") or "target"
    if target_xyz is None:
        return {"status": "missing_evidence", "answer_type": "visibility", "target_object_id": target_id, "missing_evidence": ["target 3D center"]}
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
    peripheral_fov_degrees = fov_degrees if peripheral_fov_degrees is None else peripheral_fov_degrees
    central_half = central_fov_degrees / 2.0
    peripheral_half = peripheral_fov_degrees / 2.0
    hard_half = fov_degrees / 2.0
    inside_central_fov = angle <= central_half
    inside_peripheral_fov = angle <= peripheral_half
    inside_fov = angle <= hard_half
    if inside_central_fov:
        fov_zone = "central"
    elif inside_peripheral_fov:
        fov_zone = "peripheral"
    elif inside_fov:
        fov_zone = "edge"
    else:
        fov_zone = "outside"
    occluders = line_of_sight_occluders(origin, target_arr, candidates, tube_radius_m=tube_radius_m)
    blocker = occluders[0] if occluders else None
    visible = bool(inside_fov and blocker is None)
    if not inside_fov:
        visibility_state = "outside_field_of_view"
    elif blocker:
        visibility_state = "occluded_by_listed_object"
    elif fov_zone == "central":
        visibility_state = "visible_central"
    else:
        visibility_state = "visible_peripheral"
    confidence = "medium"
    if observer.source.startswith(("eyes_to_nose", "ears_to_nose")) and fov_zone == "central" and not blocker:
        confidence = "medium_high"
    if observer.source.startswith("body_forward"):
        confidence = "medium_low"
    approximations = list(observer.approximations) + [
        "visibility uses geometric FOV and listed-object sightline occlusion",
        "occlusion is radius-scored from object centers, not dense pixel/depth ray-casting",
    ]
    answer = (
        f"Likely visible: the {target_id} is in the person's {fov_zone} field of view ({angle:.1f}° from view direction), and no listed object blocks the sightline."
        if visible else
        f"Likely not visible: the {target_id} is {angle:.1f}° from the view direction, outside the {fov_degrees:.0f}° field of view."
        if not inside_fov else
        f"Likely occluded: {blocker['object_id']} lies between the person and the {target_id} along the sightline."
    )
    return {
        "status": "ok",
        "answer_type": "visibility_occlusion",
        "target_object_id": target_id,
        "observer": observer.to_dict(),
        "observer_origin_world_m": origin.tolist(),
        "target_distance_from_observer_m": dist,
        "angle_to_view_direction_deg": angle,
        "fov_degrees": fov_degrees,
        "central_fov_degrees": central_fov_degrees,
        "peripheral_fov_degrees": peripheral_fov_degrees,
        "inside_fov": inside_fov,
        "inside_central_fov": inside_central_fov,
        "inside_peripheral_fov": inside_peripheral_fov,
        "fov_zone": fov_zone,
        "visibility_state": visibility_state,
        "blocker": blocker,
        "occluders": occluders,
        "visible": visible,
        "visibility_confidence": confidence,
        "approximations": approximations,
        "answer": answer,
    }


def level2_occlusion_answer(
    observer_joints: Mapping[str, Mapping[str, float] | Sequence[float]],
    target: Mapping,
    candidates: Sequence[Mapping],
    *,
    tube_radius_m: float = 0.18,
) -> dict:
    """Answer which object blocks target from another observer's viewpoint.

    Uses the same radius-scored sightline blocker logic as visibility_answer, but
    packages it for Level-2 perspective taking: what blocks the target from the
    person's perspective.
    """

    target_xyz = target.get("object_xyz_world_m") or target.get("object_xyz_world")
    target_id = target.get("object_id") or target.get("object_name") or "target"
    if target_xyz is None:
        return {"status": "missing_evidence", "answer_type": "level2_occlusion", "target_object_id": target_id, "missing_evidence": ["target 3D center"]}
    observer = observer_from_joints(observer_joints)
    occluders = line_of_sight_occluders(observer.origin, target_xyz, candidates, tube_radius_m=tube_radius_m)
    blocker = occluders[0] if occluders else None
    if blocker is None:
        occlusion_state = "no_listed_blocker"
        confidence = "medium"
        answer = f"No listed object blocks the {target_id} on the observer-to-target sightline."
    else:
        score = float(blocker.get("occlusion_score", 0.0))
        if score >= 0.5 or float(blocker.get("angular_margin_deg", 0.0)) >= 5.0:
            confidence = "medium_high"
        else:
            confidence = "medium"
        occlusion_state = "blocked_by_listed_object"
        answer = (
            f"From the observer's perspective, {blocker['object_id']} is the first listed blocker in front of the {target_id}; "
            f"it lies {blocker['depth_from_observer_m']:.2f} m from the observer and has occlusion score {score:.2f}."
        )
    return {
        "status": "ok",
        "answer_type": "level2_perspective_occlusion",
        "observer": observer.to_dict(),
        "target_object_id": target_id,
        "target_xyz_world_m": list(target_xyz),
        "occlusion_state": occlusion_state,
        "blocker": blocker,
        "occluders": occluders,
        "blocker_confidence": confidence,
        "approximations": list(observer.approximations) + [
            "single-observer demo when only one person pose is available",
            "blocker detection uses listed object centers with radius/depth scoring, not dense pixel ray-casting",
        ],
        "answer": answer,
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
        "semantic_room_frame": {
            "status": "unavailable",
            "reason": "room semantic axes were not provided; raw world xyz must not be interpreted as room left/right/front/back",
        },
        "egocentric_camera": None,
        "missing_evidence": [],
        "approximations": [],
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
        result["semantic_room_frame"] = {
            "status": "available",
            "axis_components_m": rel,
            "note": "Room labels are valid only relative to the supplied semantic world_axes.",
        }
    else:
        result["missing_evidence"].append("semantic world_axes for room-level allocentric labels")
        result["approximations"].append("allocentric answer is raw world xyz only, not room-semantic left/right/front/back")
    result["answer"] = {
        "human_centric": human["answer"],
        "egocentric": result["egocentric_camera"] or "missing camera extrinsics",
        "allocentric": result.get("allocentric_axis_components_m") or result["allocentric_world_xyz_m"],
        "semantic_room_frame": result["semantic_room_frame"],
    }
    return result
