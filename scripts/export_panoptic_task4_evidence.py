#!/usr/bin/env python3
"""Render deterministic identity markers onto selected Panoptic Task 4 clips."""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np


FPS = 29.97
HEAD_INDEX = 1
ASS_COLORS = {
    "RED": "&H000000FF", "BLUE": "&H00FF0000", "GREEN": "&H0000FF00",
    "YELLOW": "&H0000FFFF", "MAGENTA": "&H00FF00FF", "CYAN": "&H00FFFF00",
    "ORANGE": "&H000080FF", "WHITE": "&H00FFFFFF",
}
LABEL_OFFSETS = {
    "RED": (-58, -48),
    "BLUE": (58, -48),
    "GREEN": (0, -78),
    "YELLOW": (-58, 48),
    "MAGENTA": (58, 48),
    "CYAN": (0, 78),
}


def load_site(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    if not match:
        raise ValueError(f"cannot parse QA site data: {path}")
    return json.loads(match.group(1))


def save_site(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        "window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def camera(sequence_dir: Path) -> dict[str, np.ndarray]:
    sequence = sequence_dir.name
    payload = json.loads(
        (sequence_dir / f"calibration_{sequence}.json").read_text(encoding="utf-8")
    )
    matches = [row for row in payload.get("cameras") or [] if row.get("panel") == 0 and row.get("node") == 0]
    if len(matches) != 1:
        raise ValueError(f"{sequence}: expected exactly one HD camera 00_00")
    row = matches[0]
    return {
        "K": np.asarray(row["K"], dtype=float),
        "R": np.asarray(row["R"], dtype=float),
        "t": np.asarray(row["t"], dtype=float).reshape(3),
        "dist": np.asarray(row["distCoef"], dtype=float),
        "resolution": np.asarray(row["resolution"], dtype=int),
    }


def project(point_cm: np.ndarray, calibration: Mapping[str, np.ndarray]) -> tuple[float, float, float]:
    xyz = calibration["R"] @ point_cm + calibration["t"]
    depth = float(xyz[2])
    if depth <= 0:
        return math.nan, math.nan, depth
    x, y = float(xyz[0] / depth), float(xyz[1] / depth)
    k1, k2, p1, p2, k3 = calibration["dist"][:5]
    radius = x * x + y * y
    radial = 1 + k1 * radius + k2 * radius * radius + k3 * radius * radius * radius
    distorted_x = x * radial + 2 * p1 * x * y + p2 * (radius + 2 * x * x)
    distorted_y = y * radial + 2 * p2 * x * y + p1 * (radius + 2 * y * y)
    pixel = calibration["K"] @ np.asarray([distorted_x, distorted_y, 1.0])
    return float(pixel[0]), float(pixel[1]), depth


def timestamp(value: float) -> str:
    centiseconds = max(0, round(value * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, fraction = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{fraction:02d}"


def body_rows(path: Path) -> dict[int, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(body["id"]): body["joints19"]
        for body in payload.get("bodies") or []
        if isinstance(body, dict) and "id" in body and isinstance(body.get("joints19"), list)
    }


def write_ass(
    path: Path, sequence_dir: Path, start_frame: int, end_frame: int,
    track_to_label: Mapping[str, str], calibration: Mapping[str, np.ndarray],
    minimum_coverage: float = 0.80,
) -> dict[str, Any]:
    width, height = map(int, calibration["resolution"])
    pose_root = sequence_dir / "hdPose3d_stage1_coco19"
    pose_paths = {
        int(match.group(1)): candidate
        for candidate in pose_root.rglob("body3DScene_*.json")
        if (match := re.search(r"body3DScene_(\d+)\.json$", candidate.name))
    }
    header = (
        "[Script Info]\nScriptType: v4.00+\n"
        f"PlayResX: {width}\nPlayResY: {height}\n"
        "ScaledBorderAndShadow: yes\n\n[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Base,DejaVu Sans,34,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,1,5,10,10,10,1\n\n"
        "[Events]\nFormat: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    events = []
    visible_counts = {track_id: 0 for track_id in track_to_label}
    total_frames = end_frame - start_frame + 1
    for frame_index in range(start_frame, end_frame + 1):
        pose_path = pose_paths.get(frame_index)
        if pose_path is None:
            continue
        rows = body_rows(pose_path)
        start = (frame_index - start_frame) / FPS
        end = (frame_index - start_frame + 1) / FPS
        for track_id, label in track_to_label.items():
            joints = rows.get(int(track_id))
            offset = HEAD_INDEX * 4
            if joints is None or len(joints) < offset + 4 or float(joints[offset + 3]) < 0.2:
                continue
            u, v, depth = project(np.asarray(joints[offset:offset + 3], dtype=float), calibration)
            if not (depth > 0 and 0 <= u < width and 0 <= v < height):
                continue
            visible_counts[track_id] += 1
            color = ASS_COLORS.get(label, ASS_COLORS["WHITE"])
            marker = (
                f"{{\\pos({round(u)},{round(v)})\\fs26\\bord2\\1c{color}}}●"
            )
            dx, dy = LABEL_OFFSETS.get(label, (0, -52))
            label_x = round(min(max(42, u + dx), width - 42))
            label_y = round(min(max(24, v + dy), height - 24))
            text = f"{{\\pos({label_x},{label_y})\\1c{color}}}{label}"
            events.append(
                f"Dialogue: 0,{timestamp(start)},{timestamp(end)},Base,,0,0,0,,{marker}"
            )
            events.append(
                f"Dialogue: 0,{timestamp(start)},{timestamp(end)},Base,,0,0,0,,{text}"
            )
    coverage = {
        track_id: visible_counts[track_id] / total_frames
        for track_id in track_to_label
    }
    if min(coverage.values(), default=0.0) < minimum_coverage:
        raise ValueError(
            f"{sequence_dir.name}: identity projection coverage below "
            f"{minimum_coverage:.0%}: {coverage}"
        )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return {"frame_count": total_frames, "projection_coverage": coverage, "event_count": len(events)}


def export_group(group: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    overlay = group.get("video_identity_overlay") or {}
    source = Path(str((group.get("video_window") or {}).get("source") or ""))
    if not source.is_file():
        raise FileNotFoundError(source)
    question = (group.get("qa") or [])[0]
    multi = (question.get("result_json") or {}).get("multi_person_timeline") or {}
    states = multi.get("states") or []
    if not states:
        raise ValueError(f"{group.get('name')}: missing multi-person timeline")
    start_frame, end_frame = int(states[0]["frame_id"]), int(states[-1]["frame_id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{group['name']}_marked_15s.mp4"
    ass_path = output.with_suffix(".ass")
    calibration = camera(source.parent)
    overlay_audit = write_ass(
        ass_path, source.parent, start_frame, end_frame,
        overlay.get("source_track_to_label") or {}, calibration,
    )
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_frame / FPS:.9f}", "-i", str(source),
        "-t", f"{(end_frame - start_frame) / FPS:.9f}",
        "-vf", f"ass={ass_path}", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "20", "-movflags", "+faststart", str(output),
    ], check=True)
    group["video_clip"] = str(output)
    group["video_identity_overlay"] = {
        **overlay, "status": "rendered_and_projection_audited", **overlay_audit,
    }
    return {"case_id": group["name"], "output": str(output), **overlay_audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()
    data = load_site(args.site_data)
    selected = [
        group for group in data.get("groups") or []
        if group.get("dataset") == "CMU Panoptic Studio"
    ]
    if not selected:
        raise ValueError("release contains no selected Panoptic groups")
    rows = [export_group(group, args.output_dir) for group in selected]
    save_site(args.site_data, data)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps({
        "status": "ok", "case_count": len(rows), "cases": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "case_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
