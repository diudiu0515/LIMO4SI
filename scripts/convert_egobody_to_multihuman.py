#!/usr/bin/env python3
"""Convert paired EgoBody SMPL-X annotations to the shared Task 4 schema."""
from __future__ import annotations

import argparse
import bisect
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.egobody_task4 import pv_face_axes, smplx_forward  # noqa: E402


def _frames(root: Path, recording: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in (root / recording).glob("body_idx_*/results/frame_*/000.pkl"):
        match = re.search(r"frame_(\d+)", str(path))
        if match:
            found[int(match.group(1))] = path
    return found


def _person(path: Path, person_id: str, *, forward: list[float] | None = None, right: list[float] | None = None) -> dict[str, object]:
    with path.open("rb") as handle:
        row = pickle.load(handle)
    pelvis = [float(value) for value in row["transl"][0]]
    forward = forward or smplx_forward(row["global_orient"][0])
    # Head is deliberately a documented proxy until SMPL-X joints are loaded.
    head = [pelvis[0], pelvis[1] - 1.6, pelvis[2]]
    return {"id": person_id, "pelvis": pelvis, "head": head, "forward": forward, **({"right": right} if right is not None else {})}


def _pv_timestamps(root: Path, recording: str) -> dict[int, int]:
    values = {}
    for path in (root / recording / "PV").glob("*_frame_*.jpg"):
        match = re.match(r"(\d+)_frame_(\d+)\.jpg$", path.name)
        if match:
            values[int(match.group(2))] = int(match.group(1))
    return values


def _pv_poses(root: Path, recording: str) -> tuple[list[int], list[np.ndarray]]:
    matches = list((root / recording).glob("*/*_pv.txt"))
    if len(matches) != 1:
        raise ValueError(f"expected one PV pose file for {recording}, got {len(matches)}")
    timestamps, poses = [], []
    with matches[0].open() as handle:
        next(handle, None)  # intrinsics header
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) != 19:
                continue
            timestamps.append(int(fields[0]))
            poses.append(np.asarray([float(value) for value in fields[3:19]]).reshape(4, 4))
    if not timestamps:
        raise ValueError(f"no PV poses for {recording}")
    return timestamps, poses


def _timestamp_for_frame(frame: int, pv_timestamps: dict[int, int]) -> int:
    frames = sorted(pv_timestamps)
    if not frames:
        raise ValueError("no synchronized PV timestamps")
    index = bisect.bisect_left(frames, frame)
    candidates = [value for value in (index - 1, index) if 0 <= value < len(frames)]
    anchor = min((frames[value] for value in candidates), key=lambda value: abs(value - frame))
    if abs(anchor - frame) > 15:
        raise ValueError("nearest PV timestamp is more than 15 frames away")
    return int(round(pv_timestamps[anchor] + (frame - anchor) * 1e7 / 30.0))


def _nearest_pv_pose(timestamp: int, timestamps: list[int], poses: list[np.ndarray]) -> np.ndarray:
    index = bisect.bisect_left(timestamps, timestamp)
    candidates = [value for value in (index - 1, index) if 0 <= value < len(timestamps)]
    best = min(candidates, key=lambda value: abs(timestamps[value] - timestamp))
    if abs(timestamps[best] - timestamp) > 500_000:
        raise ValueError("nearest PV pose is more than 50 ms from the PV timestamp")
    return poses[best]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/EgoBody/extracted"))
    parser.add_argument("--pv-root", type=Path, default=Path("data/EgoBody/media"))
    parser.add_argument("--pv-metadata-root", type=Path, default=Path("data/EgoBody/egocentric_color_meta"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-frames", type=int, default=450)
    parser.add_argument("--sample-stride", type=int, default=15)
    args = parser.parse_args()

    wearer_root = args.dataset_root / "smplx_camera_wearer_val"
    partner_root = args.dataset_root / "smplx_interactee_val"
    recordings = sorted({path.name for path in wearer_root.iterdir() if path.is_dir()} &
                        {path.name for path in partner_root.iterdir() if path.is_dir()})
    scenes = []
    for recording in recordings:
        wearer, partner = _frames(wearer_root, recording), _frames(partner_root, recording)
        pv_timestamps = _pv_timestamps(args.pv_root, recording)
        if not pv_timestamps:
            continue
        try:
            pose_timestamps, pv_poses = _pv_poses(args.pv_metadata_root, recording)
            calibration_path = args.dataset_root / "calibrations" / recording / "cal_trans" / "holo_to_kinect12.json"
            holo_to_kinect = np.asarray(json.loads(calibration_path.read_text())["trans"], dtype=float)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        common = sorted(set(wearer) & set(partner))
        if len(common) < 8:
            continue
        for start_pos in range(0, len(common), args.window_frames):
            window = common[start_pos:start_pos + args.window_frames]
            if len(window) < args.window_frames * 0.85:
                continue
            sampled = window[::args.sample_stride]
            first = sampled[0]
            frames = []
            try:
                for frame in sampled:
                    timestamp = _timestamp_for_frame(frame, pv_timestamps)
                    pv_pose = _nearest_pv_pose(timestamp, pose_timestamps, pv_poses)
                    wearer_forward, wearer_right = pv_face_axes(pv_pose, holo_to_kinect)
                    frames.append({
                        "frame_id": frame,
                        "t": (frame - first) / 30.0,
                        "people": [
                            _person(wearer[frame], "A", forward=wearer_forward, right=wearer_right),
                            _person(partner[frame], "B"),
                        ],
                    })
            except ValueError:
                continue
            scenes.append({
                "scene_id": f"egobody_{recording}_{window[0]}_{window[-1]}",
                "title": f"EgoBody {recording}", "dataset": "EgoBody",
                "duration_sec": (window[-1] - window[0]) / 30.0,
                "frames": frames,
                "person_identities": {"A": "the camera wearer", "B": "the interaction partner"},
                "human_coordinate_frame": {
                    "forward_axis": "camera-wearer face-forward = synchronized HoloLens PV optical -Z transformed to master-Kinect and projected to ground",
                    "right_axis": "explicit synchronized HoloLens PV optical +X (human right) transformed to master-Kinect",
                    "right_sign": 1,
                    "orientation_calibration": {
                        "source": "EgoBody official per-frame PV camera-to-world pose + holo_to_kinect12 calibration",
                        "scope": "camera-wearer face-centered directional claims",
                    },
                },
                "evidence_source": [
                    "EgoBody paired camera-wearer/interactee SMPL-X transl",
                    "EgoBody paired SMPL-X global_orient for interaction-partner body orientation",
                    "EgoBody per-frame PV camera-to-world pose for camera-wearer face orientation",
                    "PV image timestamp synchronized to official PV pose",
                ],
                "source_file": str(args.dataset_root),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "limo4si.task4_annotations.v1", "scenes": scenes}, indent=2) + "\n")
    print(json.dumps({"recordings": len(recordings), "scenes": len(scenes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
