#!/usr/bin/env python3
"""Publish selected deterministic EgoBody Task 4 cases with auditable media."""
from __future__ import annotations

import argparse
import json
import re
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


def render(scene: dict, output: Path) -> None:
    frames = scene["frames"]
    points = [person["pelvis"] for frame in frames for person in frame["people"]]
    xs, zs = [p[0] for p in points], [p[2] for p in points]
    xmin, xmax, zmin, zmax = min(xs), max(xs), min(zs), max(zs)
    span = max(xmax - xmin, zmax - zmin, 1.0)
    def pixel(p):
        return int(100 + (p[0] - xmin) / span * 700), int(500 - (p[2] - zmin) / span * 400)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), 15, (900, 600))
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
    parser.add_argument("--question-type", action="append", dest="question_types")
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--media-map", type=Path, help="JSON mapping from case ID to site-relative original-video URL")
    parser.add_argument("--replace-selected-types", action="store_true")
    args = parser.parse_args()
    payload = json.loads(args.annotations.read_text())
    data, _ = generate_task4_release(payload["scenes"])
    question_types = set(args.question_types or ["passing_side_and_final_position"])
    selected = [group for group in data["groups"] if group["qa"][0]["question_type"] in question_types]
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
    print(json.dumps({"published": len(selected), "case_ids": sorted(names)}))


if __name__ == "__main__":
    main()
