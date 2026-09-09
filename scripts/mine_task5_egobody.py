#!/usr/bin/env python3
"""Mine real eye-gaze-on-interactee events from EgoBody validation data."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import re
from pathlib import Path

import numpy as np


IMAGE_RE = re.compile(r"(?P<timestamp>\d+)_frame_(?P<frame>\d+)\.jpg$")


def transform_point(matrix: np.ndarray, point: np.ndarray) -> np.ndarray:
    value = matrix @ np.r_[point, 1.0]
    return value[:3] / value[3]


def load_gaze(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",")
    return data[:, 0].astype(np.int64), data[:, 851] == 1, data[:, 852:855], data[:, 856:859], data[:, 860]


def runs(frames: list[int], maximum_gap: int = 2) -> list[list[int]]:
    output: list[list[int]] = []
    for frame in frames:
        if not output or frame - output[-1][-1] > maximum_gap:
            output.append([frame])
        else:
            output[-1].append(frame)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/qa/task5_egobody_candidates.json"))
    parser.add_argument("--minimum-run", type=int, default=8)
    parser.add_argument("--bbox-margin", type=float, default=0.08)
    parser.add_argument("--maximum-gaze-skew-ms", type=float, default=20.0)
    args = parser.parse_args()
    root = args.root
    info = {row["recording_name"]: row for row in csv.DictReader((root / "data_info_release.csv").open())}
    candidates, rejected = [], {}
    color_root = root / "egocentric_color"
    for keypoints_path in sorted(color_root.rglob("keypoints.npz")):
        recording = next(part for part in keypoints_path.parts if part.startswith("recording_"))
        try:
            mapping = np.load(keypoints_path, allow_pickle=True)
            valid_start = int(info[recording]["start_frame"])
            valid_end = int(info[recording]["end_frame"])
            gaze_path = next((root / "egocentric_gaze" / recording).rglob("*_head_hand_eye.csv"))
            pv_path = next(keypoints_path.parent.glob("*_pv.txt"))
            gaze_t, gaze_ok, origins, directions, depths = load_gaze(gaze_path)
            pv_lines = pv_path.read_text().splitlines()
            cx, cy, width, height = (float(x) for x in pv_lines[0].split(","))
            pv_rows = {}
            for line in pv_lines[1:]:
                row = line.split(",")
                pv_rows[int(row[0])] = (float(row[1]), float(row[2]), np.asarray(row[3:19], float).reshape(4, 4))
            hits, evidence = [], {}
            for index, image_name in enumerate(mapping["imgname"]):
                match = IMAGE_RE.search(str(image_name))
                if not match:
                    continue
                timestamp, frame = int(match["timestamp"]), int(match["frame"])
                if not valid_start <= frame <= valid_end:
                    continue
                nearest = int(np.argmin(np.abs(gaze_t - timestamp)))
                skew_ms = abs(int(gaze_t[nearest]) - timestamp) / 10_000.0
                if skew_ms > args.maximum_gaze_skew_ms or not gaze_ok[nearest] or timestamp not in pv_rows:
                    continue
                direction = directions[nearest]
                length = np.linalg.norm(direction)
                if length <= 1e-8:
                    continue
                point_world = origins[nearest] + direction / length * max(float(depths[nearest]), 1.0)
                fx, fy, pv_to_world = pv_rows[timestamp]
                point_camera = transform_point(np.linalg.inv(pv_to_world), point_world)
                if abs(point_camera[2]) <= 1e-8:
                    continue
                uv = np.array([width - (fx * point_camera[0] / point_camera[2] + cx), fy * point_camera[1] / point_camera[2] + cy])
                keypoints = np.asarray(mapping["keypoints"][index]).reshape(-1, 3)
                visible = keypoints[keypoints[:, 2] > 0.2, :2]
                if len(visible) < 6:
                    continue
                lo, hi = visible.min(0), visible.max(0)
                margin = args.bbox_margin * max(*(hi - lo), 1.0)
                if np.all(uv >= lo - margin) and np.all(uv <= hi + margin):
                    hits.append(frame)
                    evidence[frame] = {"timestamp": timestamp, "gaze_skew_ms": skew_ms, "gaze_uv": uv.tolist(), "bbox_xyxy": [*lo.tolist(), *hi.tolist()]}
            for run in runs(hits):
                if len(run) < args.minimum_run:
                    continue
                candidates.append({
                    "id": f"task5a_egobody_{recording}_{run[0]}_{run[-1]}", "dataset": "EgoBody",
                    "track": "5A_gaze_grounded_human_relation", "recording_name": recording,
                    "scene_name": info[recording]["scene_name"], "start_frame": run[0], "end_frame": run[-1],
                    "direct_hit_count": len(run), "duration_s": (run[-1] - run[0]) / 30.0,
                    "start_evidence": evidence[run[0]], "end_evidence": evidence[run[-1]],
                    "gaze_definition": "measured HoloLens gaze projects inside the visible interactee keypoint box",
                })
        except Exception as exc:
            rejected[recording] = f"{type(exc).__name__}: {exc}"
    candidates.sort(key=lambda row: (-row["direct_hit_count"], row["id"]))
    payload = {"schema_version": 1, "status": "candidates_require_rgb_and_visual_audit", "candidate_count": len(candidates), "rejected": rejected, "candidates": candidates}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate_count": len(candidates), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
