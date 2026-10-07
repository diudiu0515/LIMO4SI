"""Normalize CMU Panoptic metric multi-person tracks for Task 4 groups."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


class PanopticTask4Error(ValueError):
    """A Panoptic sequence cannot support release-grade group reasoning."""


@dataclass(frozen=True)
class PanopticTask4Policy:
    fps: float = 29.97
    window_sec: float = 15.0
    samples_per_window: int = 16
    window_stride_sec: float = 7.5
    minimum_track_coverage: float = 0.90
    minimum_joint_confidence: float = 0.20
    minimum_people: int = 3
    maximum_windows_per_sequence: int = 64
    camera_name: str = "00_00"
    minimum_projection_coverage: float = 0.80


_FRAME_PATTERN = re.compile(r"body3DScene_(\d+)\.json$")
_PELVIS_INDEX = 2
_HEAD_INDEX = 1
_DISPLAY_MARKERS = (
    "red", "blue", "green", "yellow", "magenta", "cyan", "orange", "white",
)


def _point(joints: Sequence[float], index: int) -> tuple[list[float], float]:
    offset = index * 4
    if len(joints) < offset + 4:
        raise PanopticTask4Error("Panoptic joints19 row is truncated")
    # Official Panoptic world coordinates are centimeters.
    point = [float(joints[offset + axis]) / 100.0 for axis in range(3)]
    confidence = float(joints[offset + 3])
    if not all(math.isfinite(value) for value in [*point, confidence]):
        raise PanopticTask4Error("Panoptic joint contains a non-finite value")
    return point, confidence


def _body(raw: Mapping[str, Any], minimum_confidence: float) -> dict[str, Any] | None:
    joints = raw.get("joints19")
    if not isinstance(joints, list):
        return None
    try:
        pelvis, pelvis_confidence = _point(joints, _PELVIS_INDEX)
        head, head_confidence = _point(joints, _HEAD_INDEX)
        track_id = int(raw["id"])
    except (KeyError, TypeError, ValueError, PanopticTask4Error):
        return None
    if min(pelvis_confidence, head_confidence) < minimum_confidence:
        return None
    return {
        "track_id": track_id,
        "pelvis": pelvis,
        "head": head,
        "joint_confidence": {
            "pelvis": pelvis_confidence,
            "head": head_confidence,
        },
    }


def _stable_person_ids(
    frame_people: Sequence[Mapping[int, Mapping[str, Any]]], minimum_coverage: float,
) -> list[int]:
    counts: Counter[int] = Counter()
    for people in frame_people:
        counts.update(people.keys())
    required = math.ceil(len(frame_people) * minimum_coverage)
    return sorted(track_id for track_id, count in counts.items() if count >= required)


def _load_frames(sequence_dir: Path, policy: PanopticTask4Policy) -> list[dict[str, Any]]:
    pose_root = sequence_dir / "hdPose3d_stage1_coco19"
    paths = sorted(pose_root.rglob("body3DScene_*.json"))
    frames = []
    for path in paths:
        match = _FRAME_PATTERN.search(path.name)
        if not match:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # Some official archives contain isolated zero-byte frames.  They
            # remain missing evidence; the later 95% coverage gate decides
            # whether any enclosing window is usable.
            continue
        people = {}
        for raw in payload.get("bodies") or []:
            if not isinstance(raw, Mapping):
                continue
            parsed = _body(raw, policy.minimum_joint_confidence)
            if parsed is not None:
                people[parsed["track_id"]] = parsed
        frames.append({"frame_index": int(match.group(1)), "people": people})
    if not frames:
        raise PanopticTask4Error("sequence has no extracted Coco19 body frames")
    return frames


def _camera(sequence_dir: Path, name: str) -> dict[str, Any]:
    path = sequence_dir / f"calibration_{sequence_dir.name}.json"
    if not path.is_file():
        raise PanopticTask4Error("sequence lacks official camera calibration")
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in payload.get("cameras") or [] if str(row.get("name")) == name]
    if len(matches) != 1:
        raise PanopticTask4Error(f"camera calibration {name} is unavailable or ambiguous")
    return matches[0]


def _projected_inside(point_m: Sequence[float], camera: Mapping[str, Any]) -> bool:
    rotation = camera["R"]
    translation = [float(row[0]) for row in camera["t"]]
    # Panoptic calibration translation is in centimeters.
    point_cm = [100.0 * float(value) for value in point_m]
    xyz = [
        sum(float(rotation[axis][component]) * point_cm[component] for component in range(3))
        + translation[axis]
        for axis in range(3)
    ]
    if xyz[2] <= 0:
        return False
    x, y = xyz[0] / xyz[2], xyz[1] / xyz[2]
    k1, k2, p1, p2, k3 = (float(value) for value in camera["distCoef"][:5])
    radius = x * x + y * y
    radial = 1 + k1 * radius + k2 * radius * radius + k3 * radius * radius * radius
    distorted_x = x * radial + 2 * p1 * x * y + p2 * (radius + 2 * x * x)
    distorted_y = y * radial + 2 * p2 * x * y + p1 * (radius + 2 * y * y)
    intrinsics = camera["K"]
    u = float(intrinsics[0][0]) * distorted_x + float(intrinsics[0][1]) * distorted_y + float(intrinsics[0][2])
    v = float(intrinsics[1][0]) * distorted_x + float(intrinsics[1][1]) * distorted_y + float(intrinsics[1][2])
    width, height = (int(value) for value in camera["resolution"])
    return 0 <= u < width and 0 <= v < height


def load_panoptic_task4_scenes(
    sequence_dir: Path, policy: PanopticTask4Policy | None = None,
) -> list[dict[str, Any]]:
    """Return evidence-closed 15-second group windows from one source video."""
    policy = policy or PanopticTask4Policy()
    if policy.samples_per_window < 8 or policy.minimum_people < 3:
        raise PanopticTask4Error("group windows require at least 8 samples and 3 people")
    frames = _load_frames(sequence_dir, policy)
    camera = _camera(sequence_dir, policy.camera_name)
    by_index = {int(frame["frame_index"]): frame for frame in frames}
    minimum_index, maximum_index = min(by_index), max(by_index)
    window_frames = round(policy.window_sec * policy.fps)
    stride = max(1, round(policy.window_stride_sec * policy.fps))
    starts = list(range(minimum_index, maximum_index - window_frames + 1, stride))
    if len(starts) > policy.maximum_windows_per_sequence:
        last = len(starts) - 1
        starts = sorted({
            starts[round(index * last / (policy.maximum_windows_per_sequence - 1))]
            for index in range(policy.maximum_windows_per_sequence)
        })
    video = sequence_dir / "hd_00_00.mp4"
    output = []
    for start in starts:
        end = start + window_frames
        full = [by_index[index] for index in range(start, end + 1) if index in by_index]
        if len(full) < 0.95 * (window_frames + 1):
            continue
        stable_ids = _stable_person_ids(
            [frame["people"] for frame in full], policy.minimum_track_coverage,
        )
        if len(stable_ids) < policy.minimum_people:
            continue
        projection_coverage = {
            track_id: sum(
                track_id in frame["people"]
                and _projected_inside(frame["people"][track_id]["head"], camera)
                for frame in full
            ) / len(full)
            for track_id in stable_ids
        }
        if min(projection_coverage.values(), default=0.0) < policy.minimum_projection_coverage:
            continue
        # Use every stable track so group topology is not computed from a
        # hand-picked subset.  Every released label is intended for a
        # deterministic video overlay generated from the same track IDs.
        sample_indices = sorted({
            round(start + index * window_frames / (policy.samples_per_window - 1))
            for index in range(policy.samples_per_window)
        })
        if any(index not in by_index for index in sample_indices):
            continue
        if any(track_id not in by_index[index]["people"] for index in sample_indices for track_id in stable_ids):
            continue
        aliases = {
            track_id: chr(ord("A") + position)
            for position, track_id in enumerate(stable_ids)
        }
        display_markers = {
            track_id: (
                _DISPLAY_MARKERS[position]
                if position < len(_DISPLAY_MARKERS)
                else f"number-{position + 1}"
            )
            for position, track_id in enumerate(stable_ids)
        }
        normalized_frames = []
        for index in sample_indices:
            people = []
            for track_id in stable_ids:
                raw = by_index[index]["people"][track_id]
                label = aliases[track_id]
                people.append({
                    "id": label,
                    "source_track_id": track_id,
                    "pelvis": list(raw["pelvis"]),
                    "head": list(raw["head"]),
                    "forward": [0.0, 1.0, 0.0],
                    "joint_confidence": dict(raw["joint_confidence"]),
                })
            normalized_frames.append({
                "t": (index - start) / policy.fps,
                "frame_id": index,
                "people": people,
            })
        sequence_name = sequence_dir.name
        output.append({
            "scene_id": f"panoptic_{sequence_name}_f{start:08d}",
            "title": f"CMU Panoptic · {sequence_name} · {start / policy.fps:.1f}s",
            "dataset": "CMU Panoptic Studio",
            "source_video": str(video),
            "source_file": str(video),
            "start_sec": start / policy.fps,
            "duration_sec": (sample_indices[-1] - sample_indices[0]) / policy.fps,
            "frames": normalized_frames,
            "metric_person_ids": [aliases[track_id] for track_id in stable_ids],
            "person_identities": {
                aliases[track_id]: f"the {display_markers[track_id]}-marked person"
                for track_id in stable_ids
            },
            "supported_task4_capabilities": ["group_reorganization"],
            "video_identity_overlay": {
                "required": True,
                "camera": "hd_00_00",
                "projection_coverage": {
                    aliases[track_id]: projection_coverage[track_id]
                    for track_id in stable_ids
                },
                "minimum_projection_coverage": policy.minimum_projection_coverage,
                "person_to_marker": {
                    aliases[track_id]: display_markers[track_id].upper()
                    for track_id in stable_ids
                },
                "source_track_to_label": {
                    str(track_id): display_markers[track_id].upper()
                    for track_id in stable_ids
                },
            },
            "evidence_source": [
                "CMU Panoptic hdPose3d_stage1_coco19 metric 3D body tracks",
                "official persistent body IDs within the selected 15-second window",
                "Coco19 mid-hip positions with confidence gating",
                "official camera calibration with per-track in-frame head projection coverage gating",
            ],
        })
    return output
