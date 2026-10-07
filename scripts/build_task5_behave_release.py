#!/usr/bin/env python3
"""Append one signed 15-second BEHAVE human-object motion case."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.scale_quality import require_release_quality
from limo4si.semantic_gt import seal_deterministic_question

TASK_ID = "task5_human_state_grounded_spatial_reasoning"
CASE_ID = "task5_behave_suitcase_net_displacement_15s"
SEQUENCE = "Date05_Sub06_suitcase_lift"
START_FRAME, END_FRAME, FPS = 360, 810, 30.0
ENDPOINT_SUPPORT = 30


def load_site(path: Path) -> dict:
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", path.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError(f"cannot parse {path}")
    return json.loads(match.group(1))


def save_site(path: Path, data: dict) -> None:
    path.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def robust_endpoint_displacement(values: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    start = np.median(values[:ENDPOINT_SUPPORT], axis=0)
    end = np.median(values[-ENDPOINT_SUPPORT:], axis=0)
    return float(np.linalg.norm(end - start)), start, end


def render_video(path: Path, human: np.ndarray, suitcase: np.ndarray) -> None:
    points = np.concatenate((human[:, [0, 2]], suitcase[:, [0, 2]]), axis=0)
    low, high = points.min(axis=0), points.max(axis=0)
    span = np.maximum(high - low, 0.5)
    margin = span * 0.18
    low, high = low - margin, high + margin

    def pixel(point: np.ndarray) -> tuple[int, int]:
        normalized = (point[[0, 2]] - low) / (high - low)
        return int(80 + normalized[0] * 800), int(450 - normalized[1] * 350)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix="limo4si_behave_")
    staging = Path(temporary.name) / "trajectory.mp4"
    writer = cv2.VideoWriter(str(staging), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (960, 540))
    if not writer.isOpened():
        raise ValueError(f"cannot create {path}")
    human_pixels = [pixel(row) for row in human]
    object_pixels = [pixel(row) for row in suitcase]
    for index in range(len(human)):
        canvas = np.full((540, 960, 3), (248, 246, 240), np.uint8)
        cv2.putText(canvas, "BEHAVE annotation-space motion (top view)", (34, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (30, 40, 55), 2)
        cv2.putText(canvas, f"t = {index / FPS:4.1f}s / 15.0s", (34, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 90, 105), 2)
        if index > 1:
            cv2.polylines(canvas, [np.asarray(human_pixels[:index + 1], np.int32)], False, (35, 125, 145), 4)
            cv2.polylines(canvas, [np.asarray(object_pixels[:index + 1], np.int32)], False, (55, 80, 220), 4)
        cv2.circle(canvas, human_pixels[index], 17, (35, 125, 145), -1)
        ox, oy = object_pixels[index]
        cv2.rectangle(canvas, (ox - 15, oy - 13), (ox + 15, oy + 13), (55, 80, 220), -1)
        cv2.putText(canvas, "person root", (720, 475), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (35, 125, 145), 2)
        cv2.putText(canvas, "suitcase center", (720, 505), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (55, 80, 220), 2)
        writer.write(canvas)
    writer.release()
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-i", str(staging),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], check=True)
    temporary.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-data", type=Path, default=ROOT / "site/qa_benchmark/data.js")
    parser.add_argument("--media-dir", type=Path, default=ROOT / "site/qa_benchmark/task5_media")
    args = parser.parse_args()
    sequence = ROOT / "data/BEHAVE/params" / SEQUENCE
    human_all = np.load(sequence / "smpl_fit_all.npz", allow_pickle=True)["trans"]
    object_all = np.load(sequence / "object_fit_all.npz", allow_pickle=True)["trans"]
    human = np.asarray(human_all[START_FRAME:END_FRAME + 1], dtype=float)
    suitcase = np.asarray(object_all[START_FRAME:END_FRAME + 1], dtype=float)
    if len(human) != 451 or len(suitcase) != 451:
        raise ValueError("BEHAVE 15-second window is incomplete")
    human_move, human_start, human_end = robust_endpoint_displacement(human)
    object_move, object_start, object_end = robust_endpoint_displacement(suitcase)
    ratio = object_move / human_move
    video = args.media_dir / f"{CASE_ID}.mp4"
    render_video(video, human, suitcase)
    result = {
        "status": "ok", "answer_type": "human_object_net_displacement_ratio",
        "T_Q": True, "H_Q": True, "S_Q": True,
        "annotation_direct": True, "annotation_source": "BEHAVE",
        "sequence_name": SEQUENCE, "frame_range": [START_FRAME, END_FRAME],
        "state_count": len(human), "duration_sec": 15.0,
        "human_net_displacement_m": human_move,
        "object_net_displacement_m": object_move,
        "object_to_human_displacement_ratio": ratio,
        "robust_endpoint_policy": {"method": "componentwise_median", "support_frames": ENDPOINT_SUPPORT},
        "robust_endpoints": {
            "human_start_xyz_m": human_start.tolist(), "human_end_xyz_m": human_end.tolist(),
            "object_start_xyz_m": object_start.tolist(), "object_end_xyz_m": object_end.tolist(),
        },
        "coordinate_frame": "BEHAVE registered metric world frame",
        "evidence_scope": "time-aligned fitted SMPL-H root and registered object 6DoF; net displacement only; no gaze/contact claim",
    }
    rounded = round(ratio, 1)
    correct = f"About {rounded:.1f} times the person's net displacement."
    alternatives = [
        "About 2.0 times the person's net displacement.",
        "About 1.0 times the person's net displacement.",
        "About 0.5 times the person's net displacement.",
    ]
    values = [alternatives[0], alternatives[1], correct, alternatives[2]]
    question = seal_deterministic_question({
        "task_id": TASK_ID,
        "task_name": "Task 5 · Human-State–Grounded Spatial Reasoning",
        "question_type": "human_object_net_displacement_ratio",
        "question_categories": ["human_object_coupled_motion"],
        "question": "Over this 15-second window, about how many times farther is the suitcase's net displacement than the person's?",
        "options": [{"label": label, "text": text} for label, text in zip("ABCD", values)],
        "correct_option": "C", "correct_answer": correct, "answer": correct,
        "explanation": (
            f"Robust endpoint positions give {object_move:.2f} m for the suitcase and "
            f"{human_move:.2f} m for the person; {object_move:.2f}/{human_move:.2f} = {ratio:.2f}."
        ),
        "method": "Uses componentwise median positions over the first and last annotated second, then compares metric net displacements.",
        "status": "ok", "release_eligible": True, "result_json": result,
    }, case_id=CASE_ID, provenance={"generator": "scripts/build_task5_behave_release.py"})
    group = {
        "name": CASE_ID, "title": "Task 5 · BEHAVE · person carrying a suitcase",
        "video_clip": "./task5_media/" + video.name,
        "video_window": {"source_sequence": SEQUENCE, "start_sec": START_FRAME / FPS, "duration_sec": 15.0},
        "qa": [question], "case_policy": "signed annotation-derived 15-second human-object motion question",
    }
    data = load_site(args.site_data)
    data["groups"] = [group_ for group_ in data.get("groups", []) if group_.get("name") != CASE_ID] + [group]
    require_release_quality({"groups": [group]})
    save_site(args.site_data, data)
    print(json.dumps({"status": "ok", "case_id": CASE_ID, "ratio": ratio, "video": str(video)}, indent=2))


if __name__ == "__main__":
    main()
