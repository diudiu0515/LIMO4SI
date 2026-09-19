#!/usr/bin/env python3
"""Publish selected deterministic EgoBody Task 4 cases with auditable media."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.task4_annotation import generate_task4_release  # noqa: E402


def load_site(path: Path) -> dict:
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", path.read_text(), re.S)
    if not match:
        raise ValueError(f"cannot parse {path}")
    return json.loads(match.group(1))


def encode_h264(raw: Path, output: Path) -> None:
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw),
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ], check=True)
    raw.unlink()


def recording_id(case_id: str) -> str:
    match = re.match(r"egobody_(recording_.*)_\d+_\d+$", case_id)
    if not match:
        raise ValueError(f"cannot parse EgoBody recording from {case_id}")
    return match.group(1)


def render_pv_clip(scene: dict, pv_root: Path, output: Path) -> None:
    """Build a time-faithful 30 fps clip from synchronized local PV frames."""
    recording = recording_id(str(scene["scene_id"]))
    indexed = {}
    for path in (pv_root / recording / "PV").glob("*_frame_*.jpg"):
        match = re.search(r"_frame_(\d+)\.jpg$", path.name)
        if match:
            indexed[int(match.group(1))] = path
    start = int(scene["frames"][0]["frame_id"]); end = int(scene["frames"][-1]["frame_id"])
    available = sorted(indexed)
    if not available:
        raise ValueError(f"no raw PV frames for {recording}")
    first = cv2.imread(str(indexed[min(available, key=lambda value: abs(value - start))]))
    if first is None:
        raise ValueError(f"cannot read raw PV frames for {recording}")
    height, width = first.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(output.stem + ".raw.mp4")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
    for frame_id in range(start, end + 1):
        nearest = min(available, key=lambda value: abs(value - frame_id))
        if abs(nearest - frame_id) > 15:
            writer.release()
            raise ValueError(f"raw PV evidence gap exceeds 0.5 s at frame {frame_id}")
        image = cv2.imread(str(indexed[nearest]))
        if image is None:
            writer.release(); raise ValueError(f"cannot read {indexed[nearest]}")
        writer.write(image)
    writer.release()
    encode_h264(raw_output, output)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to render raw PV clip {output}")


def render(scene: dict, output: Path) -> None:
    frames = scene["frames"]
    points = [person["pelvis"] for frame in frames for person in frame["people"]]
    xs, zs = [p[0] for p in points], [p[2] for p in points]
    xmin, xmax, zmin, zmax = min(xs), max(xs), min(zs), max(zs)
    span = max(xmax - xmin, zmax - zmin, 1.0)
    def pixel(p):
        return int(100 + (p[0] - xmin) / span * 700), int(500 - (p[2] - zmin) / span * 400)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(output.stem + ".raw.mp4")
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), 15, (900, 600))
    colors = {"A": (210, 110, 35), "B": (55, 70, 220)}
    for index, frame in enumerate(frames):
        canvas = np.full((600, 900, 3), 246, np.uint8)
        cv2.putText(canvas, "EgoBody annotation trajectory evidence (not camera video)", (25, 35), cv2.FONT_HERSHEY_SIMPLEX, .68, (30, 30, 30), 2)
        cv2.putText(canvas, f"t={float(frame['t']):.1f}s   body-forward arrows", (25, 65), cv2.FONT_HERSHEY_SIMPLEX, .58, (70, 70, 70), 1)
        for pid in ("A", "B"):
            trail = []
            for old in frames[:index + 1]:
                person = next(value for value in old["people"] if value["id"] == pid)
                trail.append(pixel(person["pelvis"]))
            if len(trail) > 1:
                cv2.polylines(canvas, [np.asarray(trail, np.int32)], False, colors[pid], 3)
            start_person = next(value for value in frames[0]["people"] if value["id"] == pid)
            start_center = pixel(start_person["pelvis"])
            cv2.circle(canvas, start_center, 10, colors[pid], 2)
            cv2.putText(canvas, f"{pid} start", (start_center[0] + 12, start_center[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, .42, colors[pid], 1)
            person = next(value for value in frame["people"] if value["id"] == pid)
            center = pixel(person["pelvis"])
            tip3 = [person["pelvis"][0] + person["forward"][0] * .45, 0, person["pelvis"][2] + person["forward"][2] * .45]
            cv2.circle(canvas, center, 14, colors[pid], -1)
            cv2.arrowedLine(canvas, center, pixel(tip3), colors[pid], 4, tipLength=.3)
            label = "A: camera wearer" if pid == "A" else "B: interaction partner"
            cv2.putText(canvas, label, (center[0] + 18, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, colors[pid], 2)
        for _ in range(8):
            writer.write(canvas)
    writer.release()
    encode_h264(raw_output, output)
    cv2.imwrite(str(output.with_suffix(".jpg")), canvas)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"failed to render {output}")


def render_endpoint_sheet(video: Path, output: Path, scene: dict) -> None:
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    images = []
    endpoint_times = (float(scene["frames"][0]["t"]), float(scene["frames"][-1]["t"]))
    for label, time_sec in zip(("START", "END"), endpoint_times):
        capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, time_sec) * 1000.0)
        ok, image = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(f"cannot read {label} frame from {video}")
        image = cv2.resize(image, (480, 270))
        cv2.rectangle(image, (0, 0), (125, 32), (255, 255, 255), -1)
        cv2.putText(image, label, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 2)
        images.append(image)
    capture.release()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), cv2.hconcat(images)):
        raise RuntimeError(f"cannot write endpoint sheet {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument("--media-dir", type=Path, default=Path("site/qa_benchmark/multihuman_media"))
    parser.add_argument("--pv-root", type=Path, default=Path("data/EgoBody/media"))
    parser.add_argument("--max-per-recording", type=int, default=1)
    parser.add_argument("--question-type", action="append", dest="question_types")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--media-map", type=Path, help="JSON mapping from case ID to site-relative original-video URL")
    parser.add_argument("--replace-selected-types", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.annotations.read_text())
    data, _ = generate_task4_release(payload["scenes"])
    question_types = set(args.question_types or ["passing_side_and_final_position"])
    selected = [group for group in data["groups"] if group["qa"][0]["question_type"] in question_types]
    if args.max_per_recording < 1:
        raise ValueError("--max-per-recording must be positive")
    selected_by_recording = []
    recording_counts = {}
    for group in selected:
        source = recording_id(group["name"])
        if recording_counts.get(source, 0) < args.max_per_recording:
            selected_by_recording.append(group)
            recording_counts[source] = recording_counts.get(source, 0) + 1
    selected = selected_by_recording
    if args.case_ids:
        requested = set(args.case_ids)
        selected = [group for group in selected if group["name"] in requested]
        missing = requested - {group["name"] for group in selected}
        if missing:
            raise ValueError(f"requested cases are unavailable for selected question types: {sorted(missing)}")
    media_map = json.loads(args.media_map.read_text()) if args.media_map else {}
    scenes = {scene["scene_id"]: scene for scene in payload["scenes"]}
    site = load_site(args.site_data)
    names = {group["name"] for group in selected}
    if args.replace_selected_types:
        site["groups"] = [
            group for group in site["groups"]
            if not (
                group.get("dataset") == "EgoBody"
                and (group.get("qa") or [{}])[0].get("question_type") in question_types
            )
        ]
    else:
        site["groups"] = [group for group in site["groups"] if group.get("name") not in names]
    for group in selected:
        filename = group["name"] + "_trajectory.mp4"
        render(scenes[group["name"]], args.media_dir / filename)
        group["metric_evidence_video"] = "./multihuman_media/" + filename
        group["topdown_image"] = "./multihuman_media/" + Path(filename).with_suffix(".jpg").name
        original_url = media_map.get(group["name"])
        if not original_url:
            original_name = group["name"] + "_original.mp4"
            render_pv_clip(scenes[group["name"]], args.pv_root, args.media_dir / original_name)
            original_url = "./multihuman_media/" + original_name
        if original_url:
            group["video_clip"] = original_url
            group["media_scope"] = "original EgoBody HoloLens PV RGB; official synchronized frame window"
            endpoint_name = group["name"] + "_endpoints.jpg"
            original_path = args.site_data.parent / original_url.removeprefix("./")
            render_endpoint_sheet(original_path, args.media_dir / endpoint_name, scenes[group["name"]])
            group["original_image"] = "./multihuman_media/" + endpoint_name
            group["original_caption"] = "Original HoloLens PV start and end frames used to audit left/right claims"
        else:
            group["video_clip"] = group["metric_evidence_video"]
            group["media_scope"] = "deterministic annotation trajectory; not raw camera video"
        site["groups"].append(group)
    args.site_data.write_text("window.QA_DATA = " + json.dumps(site, ensure_ascii=False, indent=2) + ";\n")
    referenced = {
        Path(value).name
        for item in site.get("groups", [])
        for key in ("video_clip", "metric_evidence_video", "topdown_image", "original_image")
        for value in [item.get(key)] if isinstance(value, str)
    }
    stale = [path for path in args.media_dir.glob("egobody_*") if path.name not in referenced]
    for path in stale:
        path.unlink()
    print(json.dumps({"published": len(selected), "case_ids": sorted(names), "pruned_stale_egobody_media": len(stale)}))


if __name__ == "__main__":
    main()
