#!/usr/bin/env python3
"""Convert CMU Panoptic coco19 3D bodies to the Task 4 scene schema."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Sequence


def sub(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [float(x) - float(y) for x, y in zip(a, b)]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(float(x) * float(y) for x, y in zip(a, b))


def unit(vector: Sequence[float]) -> list[float] | None:
    length = math.sqrt(dot(vector, vector))
    return None if length < 1e-8 else [float(value) / length for value in vector]


def cross(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def joint(body: dict, index: int, min_confidence: float) -> list[float] | None:
    values = body.get("joints19") or []
    offset = index * 4
    if len(values) < offset + 4 or float(values[offset + 3]) < min_confidence:
        return None
    # Panoptic world coordinates are stored in centimetres.
    return [float(value) / 100.0 for value in values[offset:offset + 3]]


def convert_body(body: dict, min_confidence: float) -> dict | None:
    neck = joint(body, 0, min_confidence)
    nose = joint(body, 1, min_confidence)
    pelvis = joint(body, 2, min_confidence)
    left_shoulder = joint(body, 3, min_confidence)
    right_shoulder = joint(body, 9, min_confidence)
    if not all((neck, pelvis, left_shoulder, right_shoulder)):
        return None
    right = unit(sub(right_shoulder, left_shoulder))
    up = unit(sub(neck, pelvis))
    if right is None or up is None:
        return None
    forward = unit(cross(right, up))
    if forward is None:
        return None
    if nose is not None:
        hint = sub(nose, neck)
        hint = [hint[i] - up[i] * dot(hint, up) for i in range(3)]
        if dot(forward, hint) < 0:
            forward = [-value for value in forward]
    return {
        "id": str(body.get("id")),
        "pelvis": pelvis,
        "head": nose or neck,
        "forward": forward,
        "source": "CMU Panoptic coco19 metric 3D skeleton",
    }


def public_identity_map(source_ids: Sequence[str]) -> dict[str, str]:
    """Map dataset IDs to compact internal IDs while retaining provenance."""
    ordered = sorted((str(value) for value in source_ids), key=lambda value: int(value))
    return {source_id: (chr(65 + index) if index < 26 else f"P{index + 1}") for index, source_id in enumerate(ordered)}


def frame_number(path: Path) -> int:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequence_dir", type=Path)
    parser.add_argument("--pose-dir", type=Path, help="Defaults to sequence_dir/hdPose3d_stage1_coco19")
    parser.add_argument("--fps", type=float, default=29.97)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--min-people", type=int, default=3)
    parser.add_argument("--min-id-coverage", type=float, default=0.80)
    parser.add_argument("--min-joint-confidence", type=float, default=0.10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pose_dir = args.pose_dir or args.sequence_dir / "hdPose3d_stage1_coco19"
    paths = sorted(pose_dir.glob("body3DScene_*.json"), key=frame_number)[::max(1, args.stride)]
    if not paths:
        raise SystemExit(f"No Panoptic pose files found in {pose_dir}")

    raw_frames = []
    identity_counts: Counter[str] = Counter()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        people = []
        for body in payload.get("bodies") or []:
            converted = convert_body(body, args.min_joint_confidence)
            if converted:
                people.append(converted)
                identity_counts[converted["id"]] += 1
        raw_frames.append((frame_number(path), people))
    keep = {pid for pid, count in identity_counts.items() if count / len(raw_frames) >= args.min_id_coverage}
    if len(keep) < args.min_people:
        raise SystemExit(
            f"Only {len(keep)} identities meet {args.min_id_coverage:.0%} coverage; need {args.min_people}. "
            "This sequence is not eligible for three-plus-person metric QA."
        )
    first_frame = raw_frames[0][0]
    identity_map = public_identity_map(keep)
    frames = []
    for frame_id, people in raw_frames:
        converted_people = []
        for person in people:
            if person["id"] not in keep:
                continue
            person = dict(person)
            person["source_person_id"] = person["id"]
            person["id"] = identity_map[person["id"]]
            converted_people.append(person)
        frames.append({
            "frame_id": frame_id,
            "t": (frame_id - first_frame) / args.fps,
            "people": converted_people,
        })
    scene = {
        "scene_id": args.sequence_dir.name,
        "dataset": "CMU Panoptic Studio",
        "duration_sec": frames[-1]["t"] - frames[0]["t"],
        "frames": frames,
        "metric_person_ids": list(identity_map.values()),
        "source_identity_map": identity_map,
        "evidence_source": [
            "CMU Panoptic time-associated coco19 metric 3D skeletons",
            "pelvis-to-neck and shoulder geometry for body-forward direction",
        ],
        "conversion": {
            "source_pose_dir": str(pose_dir),
            "source_fps": args.fps,
            "stride": args.stride,
            "min_id_coverage": args.min_id_coverage,
            "coordinate_unit": "metres (converted from Panoptic centimetres)",
            "forward_axis": "shoulder-right cross pelvis-to-neck up, nose-sign disambiguated",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"scenes": [scene]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"scene_id": scene["scene_id"], "frames": len(frames), "metric_people": len(keep)}, indent=2))


if __name__ == "__main__":
    main()
