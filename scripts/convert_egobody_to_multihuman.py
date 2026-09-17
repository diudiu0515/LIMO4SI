#!/usr/bin/env python3
"""Convert paired EgoBody SMPL-X annotations to the shared Task 4 schema."""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.egobody_task4 import smplx_forward  # noqa: E402


def _frames(root: Path, recording: str) -> dict[int, Path]:
    found: dict[int, Path] = {}
    for path in (root / recording).glob("body_idx_*/results/frame_*/000.pkl"):
        match = re.search(r"frame_(\d+)", str(path))
        if match:
            found[int(match.group(1))] = path
    return found


def _person(path: Path, person_id: str) -> dict[str, object]:
    with path.open("rb") as handle:
        row = pickle.load(handle)
    pelvis = [float(value) for value in row["transl"][0]]
    forward = smplx_forward(row["global_orient"][0])
    # Head is deliberately a documented proxy until SMPL-X joints are loaded.
    head = [pelvis[0], pelvis[1] - 1.6, pelvis[2]]
    return {"id": person_id, "pelvis": pelvis, "head": head, "forward": forward}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/EgoBody/extracted"))
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
        common = sorted(set(wearer) & set(partner))
        if len(common) < 8:
            continue
        for start_pos in range(0, len(common), args.window_frames):
            window = common[start_pos:start_pos + args.window_frames]
            if len(window) < args.window_frames * 0.85:
                continue
            sampled = window[::args.sample_stride]
            first = sampled[0]
            frames = [{
                "frame_id": frame,
                "t": (frame - first) / 30.0,
                "people": [_person(wearer[frame], "A"), _person(partner[frame], "B")],
            } for frame in sampled]
            scenes.append({
                "scene_id": f"egobody_{recording}_{window[0]}_{window[-1]}",
                "title": f"EgoBody {recording}", "dataset": "EgoBody",
                "duration_sec": (window[-1] - window[0]) / 30.0,
                "frames": frames,
                "person_identities": {"A": "the camera wearer", "B": "the interaction partner"},
                "human_coordinate_frame": {
                    "forward_axis": "body-forward = SMPL-X canonical +Z transformed by global_orient and projected to the ground plane",
                    "right_axis": "scene-up cross forward in the master-Kinect camera frame",
                    "right_sign": -1,
                    "orientation_calibration": {
                        "source": "EgoBody official SMPL-X global_orient in master-Kinect coordinates; canonical +Z checked against synchronized first-person visibility",
                        "validation_recording": "recording_20210921_S11_S10_02",
                        "validation_frame": 2555,
                    },
                },
                "evidence_source": [
                    "EgoBody paired camera-wearer/interactee SMPL-X transl",
                    "EgoBody paired SMPL-X global_orient",
                    "30 fps synchronized frame ids",
                ],
                "source_file": str(args.dataset_root),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"schema": "limo4si.task4_annotations.v1", "scenes": scenes}, indent=2) + "\n")
    print(json.dumps({"recordings": len(recordings), "scenes": len(scenes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
