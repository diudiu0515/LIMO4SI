"""Normalize Aria Digital Twin multi-person ground truth for Task 4.

ADT supplies a metric Aria trajectory and, for selected recordings, one or
more motion-capture skeletons in the same scene frame.  The wearer's
face-forward/right axes come from the official per-device CPF calibration,
not raw device axes.  Skeleton torso orientation is kept as evidence, but
categories that require the other person's *face* direction are explicitly
disabled because the 51-joint suit has no facial landmark.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class AdtTask4Error(ValueError):
    """An ADT sequence cannot support a release-grade Task 4 scene."""


@dataclass(frozen=True)
class AdtTask4Policy:
    window_sec: float = 15.0
    window_stride_sec: float = 7.5
    samples_per_window: int = 16
    maximum_windows_per_sequence: int = 24
    minimum_trajectory_quality: float = 0.5
    maximum_sync_error_sec: float = 0.025
    wearer_head_to_pelvis_m: float = 0.65


_JOINT = {
    "pelvis": 0,
    "chest": 2,
    "head": 4,
    "left_upper_arm": 6,
    "right_upper_arm": 25,
}


def _sub(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) - float(b) for a, b in zip(left, right)]


def _add(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [float(a) + float(b) for a, b in zip(left, right)]


def _scale(value: Sequence[float], factor: float) -> list[float]:
    return [float(axis) * factor for axis in value]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


def _cross(left: Sequence[float], right: Sequence[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def _unit(value: Sequence[float]) -> list[float]:
    length = math.sqrt(_dot(value, value))
    if not math.isfinite(length) or length < 1e-8:
        raise AdtTask4Error("degenerate annotated axis")
    return [float(axis) / length for axis in value]


def _horizontal(value: Sequence[float]) -> list[float]:
    return _unit([float(value[0]), 0.0, float(value[2])])


def _quat_rotate(quaternion_xyzw: Sequence[float], vector: Sequence[float]) -> list[float]:
    x, y, z, w = _unit4(quaternion_xyzw)
    qv = [x, y, z]
    twice_cross = _scale(_cross(qv, vector), 2.0)
    return _add(vector, _add(_scale(twice_cross, w), _cross(qv, twice_cross)))


def _unit4(value: Sequence[float]) -> list[float]:
    if len(value) != 4:
        raise AdtTask4Error("quaternion must have four components")
    length = math.sqrt(sum(float(axis) ** 2 for axis in value))
    if not math.isfinite(length) or length < 1e-8:
        raise AdtTask4Error("invalid trajectory quaternion")
    return [float(axis) / length for axis in value]


def _nearest_index(timestamps: Sequence[int], timestamp_ns: int, maximum_error_ns: int) -> int:
    position = bisect.bisect_left(timestamps, timestamp_ns)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(timestamps)]
    if not candidates:
        raise AdtTask4Error("timestamp lies outside evidence")
    index = min(candidates, key=lambda value: abs(timestamps[value] - timestamp_ns))
    if abs(timestamps[index] - timestamp_ns) > maximum_error_ns:
        raise AdtTask4Error("time-aligned evidence exceeds the synchronization gate")
    return index


def _load_trajectory(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "tracking_timestamp_us", "tx_world_device", "ty_world_device", "tz_world_device",
        "qx_world_device", "qy_world_device", "qz_world_device", "qw_world_device",
        "gravity_x_world", "gravity_y_world", "gravity_z_world", "quality_score",
    }
    if not rows or not required.issubset(rows[0]):
        raise AdtTask4Error("aria_trajectory.csv is missing required pose fields")
    parsed = []
    for row in rows:
        parsed.append({
            "timestamp_ns": int(row["tracking_timestamp_us"]) * 1000,
            "translation": [float(row[f"t{axis}_world_device"]) for axis in "xyz"],
            "quaternion": [float(row[f"q{axis}_world_device"]) for axis in "xyzw"],
            "gravity": [float(row[f"gravity_{axis}_world"]) for axis in "xyz"],
            "quality": float(row["quality_score"]),
        })
    if any(left["timestamp_ns"] >= right["timestamp_ns"] for left, right in zip(parsed, parsed[1:])):
        raise AdtTask4Error("trajectory timestamps are not strictly increasing")
    return parsed


def _load_skeletons(sequence_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(sequence_dir.glob("Skeleton_*.json"))
    skeletons = []
    for path in paths:
        raw = json.loads(path.read_text(encoding="utf-8"))
        frames = raw.get("frames") if isinstance(raw, Mapping) else None
        if not isinstance(frames, list) or not frames:
            continue
        parsed = []
        for frame in frames:
            joints = frame.get("joints")
            if not isinstance(joints, list) or len(joints) <= max(_JOINT.values()):
                continue
            try:
                values = [[float(axis) for axis in point[:3]] for point in joints]
                timestamp_ns = int(frame["timestamp_ns"])
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(axis) for point in values for axis in point):
                parsed.append({"timestamp_ns": timestamp_ns, "joints": values})
        if parsed:
            skeletons.append({
                "name": path.stem,
                "path": path,
                "frames": parsed,
                "timestamps": [frame["timestamp_ns"] for frame in parsed],
            })
    return skeletons


def _wearer_skeleton_names(
    sequence_dir: Path, serial: str, *, require_association: bool,
) -> set[str]:
    """Return mocap tracks belonging to the current Aria wearer.

    ADT multiskeleton recordings include a skeleton for each Aria wearer.  The
    current wearer's skeleton must not be added beside the trajectory-derived
    wearer track, or one human would be counted twice.
    """
    path = sequence_dir / "skeleton_aria_association.json"
    if not path.is_file():
        if require_association:
            raise AdtTask4Error("multi-skeleton sequence lacks wearer association metadata")
        return set()
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("SkeletonMetadata") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise AdtTask4Error("invalid skeleton-to-Aria association metadata")
    return {
        str(row["SkeletonName"])
        for row in rows
        if isinstance(row, Mapping)
        and str(row.get("AssociatedDeviceSerial")) == serial
        and str(row.get("SkeletonName") or "NONE") != "NONE"
    }


def _mat_vec(matrix: Sequence[Sequence[float]], vector: Sequence[float]) -> list[float]:
    if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
        raise AdtTask4Error("device-to-CPF rotation must be 3x3")
    return [sum(float(matrix[i][j]) * float(vector[j]) for j in range(3)) for i in range(3)]


def _quat_wxyz_to_matrix(value: Sequence[float]) -> list[list[float]]:
    if len(value) != 4:
        raise AdtTask4Error("object quaternion must have four components")
    w, x, y, z = (float(axis) for axis in value)
    length = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(length) or length < 1e-8:
        raise AdtTask4Error("invalid object quaternion")
    w, x, y, z = w / length, x / length, y / length, z / length
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _load_object_blockers(sequence_dir: Path) -> dict[str, Any]:
    """Load exact ADT oriented boxes and their annotated world poses."""
    boxes_path = sequence_dir / "3d_bounding_box.csv"
    poses_path = sequence_dir / "scene_objects.csv"
    instances_path = sequence_dir / "instances.json"
    if not boxes_path.is_file() or not poses_path.is_file() or not instances_path.is_file():
        return {"static": [], "dynamic": [], "evidence_files": []}
    instances = json.loads(instances_path.read_text(encoding="utf-8"))
    if not isinstance(instances, Mapping):
        raise AdtTask4Error("instances.json must map object IDs to annotations")
    bounds: dict[str, dict[str, list[float]]] = {}
    with boxes_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = str(row["object_uid"])
            try:
                bounds[uid] = {
                    "local_min": [float(row[f"p_local_obj_{axis}min[m]"]) for axis in "xyz"],
                    "local_max": [float(row[f"p_local_obj_{axis}max[m]"]) for axis in "xyz"],
                }
            except (KeyError, TypeError, ValueError):
                continue
    static: list[dict[str, Any]] = []
    dynamic_by_id: dict[str, list[dict[str, Any]]] = {}
    with poses_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uid = str(row["object_uid"])
            annotation = instances.get(uid)
            if annotation is None and uid.isdigit():
                annotation = instances.get(int(uid))
            if uid not in bounds or not isinstance(annotation, Mapping):
                continue
            if annotation.get("instance_type") != "object" or annotation.get("rigidity") != "rigid":
                continue
            try:
                timestamp_ns = int(row["timestamp[ns]"])
                translation = [float(row[f"t_wo_{axis}[m]"]) for axis in "xyz"]
                quaternion = [float(row[f"q_wo_{axis}"]) for axis in "wxyz"]
                rotation = _quat_wxyz_to_matrix(quaternion)
            except (KeyError, TypeError, ValueError, AdtTask4Error):
                continue
            blocker = {
                "id": uid,
                "name": str(annotation.get("instance_name") or uid),
                "category": str(annotation.get("category") or "object"),
                **bounds[uid],
                "translation": translation,
                "rotation_world_from_local": rotation,
                "timestamp_ns": timestamp_ns,
                "geometry_source": "ADT annotated T_world_object and local 3D bounding box",
            }
            if timestamp_ns < 0:
                static.append(blocker)
            else:
                dynamic_by_id.setdefault(uid, []).append(blocker)
    dynamic = []
    for uid, rows in dynamic_by_id.items():
        rows.sort(key=lambda row: row["timestamp_ns"])
        dynamic.append({
            "id": uid,
            "timestamps": [row["timestamp_ns"] for row in rows],
            "poses": rows,
        })
    return {
        "static": static,
        "dynamic": dynamic,
        "evidence_files": [str(boxes_path), str(poses_path), str(instances_path)],
    }


def _blockers_at(
    object_geometry: Mapping[str, Any], timestamp_ns: int, maximum_error_ns: int,
) -> list[dict[str, Any]]:
    """Return only moving boxes; static boxes live once at scene level."""
    blockers: list[dict[str, Any]] = []
    for track in object_geometry.get("dynamic", []):
        try:
            index = _nearest_index(track["timestamps"], timestamp_ns, maximum_error_ns)
        except AdtTask4Error:
            continue
        blockers.append(dict(track["poses"][index]))
    return blockers


def _wearer(
    row: Mapping[str, Any], policy: AdtTask4Policy,
    device_cpf_rotation: Sequence[Sequence[float]],
) -> dict[str, Any]:
    if float(row["quality"]) < policy.minimum_trajectory_quality:
        raise AdtTask4Error("Aria trajectory quality is below threshold")
    head = list(row["translation"])
    down = _unit(row["gravity"])
    forward = _horizontal(_quat_rotate(
        row["quaternion"], _mat_vec(device_cpf_rotation, [0.0, 0.0, 1.0])
    ))
    # Gen1 CPF +X points to the wearer's left; -X is the explicit human-right
    # axis.  Keeping both calibrated axes avoids an implicit handedness flip.
    right = _horizontal(_quat_rotate(
        row["quaternion"], _mat_vec(device_cpf_rotation, [-1.0, 0.0, 0.0])
    ))
    if abs(_dot(right, forward)) > 0.05:
        raise AdtTask4Error("calibrated CPF face-forward/right axes are not orthogonal")
    return {
        "id": "A",
        "pelvis": _add(head, _scale(down, policy.wearer_head_to_pelvis_m)),
        "head": head,
        "forward": forward,
        "right": right,
    }


def _skeleton(frame: Mapping[str, Any], person_id: str, source_name: str) -> dict[str, Any]:
    joints = frame["joints"]
    pelvis = joints[_JOINT["pelvis"]]
    head = joints[_JOINT["head"]]
    up = _unit(_sub(joints[_JOINT["chest"]], pelvis))
    anatomical_right = _unit(_sub(
        joints[_JOINT["right_upper_arm"]], joints[_JOINT["left_upper_arm"]]
    ))
    forward = _horizontal(_cross(anatomical_right, up))
    right = _horizontal(_cross([0.0, 1.0, 0.0], forward))
    if _dot(right, _horizontal(anatomical_right)) < 0.8:
        raise AdtTask4Error("skeleton anatomical-right calibration is inconsistent")
    return {
        "id": person_id,
        "pelvis": list(pelvis),
        "head": list(head),
        "forward": forward,
        "right": right,
        "orientation_scope": "torso-forward only",
        "source_skeleton": source_name,
    }


def load_adt_task4_scenes(
    sequence_dir: Path, *, device_cpf_rotations: Mapping[str, Sequence[Sequence[float]]],
    policy: AdtTask4Policy | None = None,
) -> list[dict[str, Any]]:
    """Return deterministic 15-second candidate windows from one ADT source video."""
    policy = policy or AdtTask4Policy()
    metadata_path = sequence_dir / "metadata.json"
    trajectory_path = sequence_dir / "aria_trajectory.csv"
    if not metadata_path.is_file() or not trajectory_path.is_file():
        raise AdtTask4Error("sequence is missing metadata or the Aria trajectory")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not metadata.get("is_multi_person"):
        raise AdtTask4Error("ADT metadata does not mark this as a multi-person sequence")
    serial = str(metadata.get("serial") or "")
    device_cpf_rotation = device_cpf_rotations.get(serial)
    if device_cpf_rotation is None:
        raise AdtTask4Error(f"missing official device-to-CPF calibration for {serial or '<unknown>'}")
    trajectory = _load_trajectory(trajectory_path)
    skeletons = _load_skeletons(sequence_dir)
    wearer_skeletons = _wearer_skeleton_names(
        sequence_dir, serial,
        require_association=int(metadata.get("num_skeletons") or 0) > 1,
    )
    skeletons = [
        skeleton for skeleton in skeletons
        if skeleton["name"] not in wearer_skeletons
    ]
    if not skeletons:
        raise AdtTask4Error("sequence has no motion-capture skeleton track")
    trajectory_timestamps = [row["timestamp_ns"] for row in trajectory]
    object_geometry = _load_object_blockers(sequence_dir)
    start_ns = max(trajectory_timestamps[0], *(skeleton["timestamps"][0] for skeleton in skeletons))
    end_ns = min(trajectory_timestamps[-1], *(skeleton["timestamps"][-1] for skeleton in skeletons))
    window_ns = round(policy.window_sec * 1e9)
    stride_ns = round(policy.window_stride_sec * 1e9)
    if stride_ns <= 0 or policy.maximum_windows_per_sequence < 2:
        raise AdtTask4Error("window stride must be positive and maximum windows at least two")
    if end_ns - start_ns < window_ns:
        raise AdtTask4Error("shared multi-person evidence is shorter than 15 seconds")
    available_starts = list(range(start_ns, end_ns - window_ns + 1, stride_ns))
    if len(available_starts) > policy.maximum_windows_per_sequence:
        last = len(available_starts) - 1
        available_starts = sorted({
            available_starts[round(index * last / (policy.maximum_windows_per_sequence - 1))]
            for index in range(policy.maximum_windows_per_sequence)
        })
    max_error_ns = round(policy.maximum_sync_error_sec * 1e9)
    sequence_name = sequence_dir.name
    output = []
    for window_index, window_start_ns in enumerate(available_starts, 1):
        frames = []
        try:
            for sample_index in range(policy.samples_per_window):
                fraction = sample_index / (policy.samples_per_window - 1)
                timestamp_ns = window_start_ns + round(fraction * window_ns)
                trajectory_index = _nearest_index(
                    trajectory_timestamps, timestamp_ns, max_error_ns
                )
                people = [_wearer(
                    trajectory[trajectory_index], policy, device_cpf_rotation
                )]
                for skeleton_index, skeleton in enumerate(skeletons):
                    evidence_index = _nearest_index(
                        skeleton["timestamps"], timestamp_ns, max_error_ns
                    )
                    people.append(_skeleton(
                        skeleton["frames"][evidence_index],
                        chr(ord("B") + skeleton_index),
                        skeleton["name"],
                    ))
                frames.append({
                    "t": fraction * policy.window_sec,
                    "timestamp_ns": timestamp_ns,
                    "people": people,
                    "blockers": _blockers_at(object_geometry, timestamp_ns, max_error_ns),
                })
        except AdtTask4Error:
            continue
        identities = {"A": "the camera wearer"}
        if len(skeletons) == 1:
            identities["B"] = "the other motion-capture participant"
        else:
            initial_distances = []
            for person in frames[0]["people"][1:]:
                delta = _sub(person["pelvis"], frames[0]["people"][0]["pelvis"])
                initial_distances.append((math.sqrt(_dot(delta, delta)), person["id"]))
            ordered = sorted(initial_distances)
            if any(abs(right[0] - left[0]) < 0.25 for left, right in zip(ordered, ordered[1:])):
                continue
            for rank, (_, person_id) in enumerate(ordered):
                identities[person_id] = (
                    "the initially nearer motion-capture participant"
                    if rank == 0 else "the initially farther motion-capture participant"
                )
        source_video = str(sequence_dir / "preview_rgb.mp4")
        if not (sequence_dir / "preview_rgb.mp4").is_file():
            source_video = str(sequence_dir / "video.vrs")
        output.append({
            "scene_id": f"adt_{sequence_name}_win{window_index:02d}",
            "title": f"ADT · {sequence_name} · window {window_index:02d}",
            "dataset": "Aria Digital Twin",
            "source_video": source_video,
            "source_file": source_video,
            "start_sec": (window_start_ns - trajectory_timestamps[0]) / 1e9,
            "duration_sec": policy.window_sec,
            "frames": frames,
            "blockers": [dict(row) for row in object_geometry["static"]],
            "metric_person_ids": [person["id"] for person in frames[0]["people"]],
            "person_identities": identities,
            # Passing is evaluated as B moving around A, so every side/front
            # claim is anchored on the camera wearer's calibrated face frame.
            "passing_relation_key": "b_relative_to_a",
            "face_forward_person_ids": ["A"],
            "dominant_relation_mode": "body_centric_position",
            "supported_task4_capabilities": [
                "distance_evolution",
                "passing_side_and_final_position",
                "dominant_interaction_relation",
                "reunion_relation_restoration",
                "relation_change_cause",
                *(["group_reorganization"] if len(frames[0]["people"]) >= 3 else []),
                *(["physical_visibility_occlusion_timeline"] if (
                    object_geometry["static"] or object_geometry["dynamic"]
                ) else []),
            ],
            "human_coordinate_frame": {
                "forward_axis": "A face-forward = Aria Gen1 device +Z projected to the gravity-horizontal plane; other tracks retain torso-forward only",
                "right_axis": "explicit official Gen1 CPF -X human-right axis for A; explicit labeled anatomical-right shoulders for B/C",
                "right_sign": 1,
                "orientation_calibration": {
                    "source": "ADT T_world_device quaternion, per-device T_device_CPF calibration, and official 51-joint labels from AriaDigitalTwinSkeletonProvider",
                    "directional_scope": "face-centered claims are anchored on A only; B/C face-dependent categories are disabled",
                },
            },
            "evidence_source": [
                "ADT aria_trajectory.csv metric T_world_device",
                *[f"ADT {skeleton['path'].name} official motion-capture joints" for skeleton in skeletons],
                "ADT skeleton_aria_association.json excludes the current wearer's duplicate mocap track",
                *object_geometry["evidence_files"],
                "nearest-timestamp alignment with a 25 ms rejection gate",
            ],
        })
    return output
