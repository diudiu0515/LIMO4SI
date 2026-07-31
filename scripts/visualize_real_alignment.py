#!/usr/bin/env python3
"""Visualize a real Ego-Exo4D relation mask and EgoPose on one exo frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ego4d.research.util.masks import decode_mask


SKELETON = (
    ("nose", "left-eye"),
    ("nose", "right-eye"),
    ("left-eye", "left-ear"),
    ("right-eye", "right-ear"),
    ("left-shoulder", "right-shoulder"),
    ("left-shoulder", "left-elbow"),
    ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument(
        "--take-uid",
        default="35bfade9-8ead-46a4-b2f0-cdcfb86df1d6",
    )
    parser.add_argument("--camera", default="cam02")
    parser.add_argument("--frame", type=int, default=750)
    parser.add_argument("--object-id", default="stainless salt container_0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/validation/real_alignment.jpg"),
    )
    return parser.parse_args()


def load_take(root: Path, take_uid: str) -> dict:
    takes = json.loads((root / "takes.json").read_text())
    for take in takes:
        if take["take_uid"] == take_uid:
            return take
    raise KeyError(f"Unknown take_uid: {take_uid}")


def read_frame(video_path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_number} from {video_path}")
    return frame


def scaled_point(point: dict, sx: float, sy: float) -> tuple[int, int]:
    return round(float(point["x"]) * sx), round(float(point["y"]) * sy)


def main() -> None:
    args = parse_args()
    take = load_take(args.root, args.take_uid)
    take_name = take["take_name"]

    relation_path = args.root / "annotations" / "relations_val.json"
    relations = json.loads(relation_path.read_text())["annotations"][args.take_uid]
    track = relations["object_masks"][args.object_id][args.camera]
    mask_record = track["annotation"][str(args.frame)]
    mask = decode_mask(mask_record).astype(bool)

    body_path = (
        args.root
        / "annotations"
        / "ego_pose"
        / "val"
        / "body"
        / "automatic"
        / f"{args.take_uid}.json"
    )
    body = json.loads(body_path.read_text())
    frame_pose = body[str(args.frame)][0]
    joints_2d = frame_pose["annotation2D"][args.camera]

    video_path = (
        args.root
        / "takes"
        / take_name
        / "frame_aligned_videos"
        / "downscaled"
        / "448"
        / f"{args.camera}.mp4"
    )
    image = read_frame(video_path, args.frame)
    height, width = image.shape[:2]
    mask_small = cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    overlay = image.copy()
    overlay[mask_small] = (
        0.25 * overlay[mask_small] + 0.75 * np.array([0, 255, 0])
    ).astype(np.uint8)
    image = cv2.addWeighted(image, 0.25, overlay, 0.75, 0)

    sx, sy = width / mask.shape[1], height / mask.shape[0]
    points = {
        name: scaled_point(point, sx, sy)
        for name, point in joints_2d.items()
        if point and "x" in point and "y" in point
    }
    for start, end in SKELETON:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], (255, 180, 0), 3)
    for name, point in points.items():
        color = (0, 0, 255) if name == "nose" else (255, 255, 0)
        cv2.circle(image, point, 5, color, -1)

    ys, xs = np.nonzero(mask_small)
    if len(xs):
        center = (round(float(np.median(xs))), round(float(np.median(ys))))
        cv2.drawMarker(
            image,
            center,
            (0, 0, 255),
            cv2.MARKER_CROSS,
            24,
            3,
        )

    lines = (
        f"take: {take_name}",
        f"frame: {args.frame}  camera: {args.camera}",
        f"object: {args.object_id}",
        "green=Relations mask  cyan=EgoPose  red-cross=mask center",
        "This validates 2D temporal/view alignment, not object 3D yet.",
    )
    banner_h = 28 * len(lines) + 14
    canvas = cv2.copyMakeBorder(
        image, banner_h, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
    )
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (12, 28 + 27 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"Failed to write {args.output}")

    report = {
        "take_uid": args.take_uid,
        "take_name": take_name,
        "frame": args.frame,
        "camera": args.camera,
        "object_id": args.object_id,
        "video_path": str(video_path),
        "video_size": [width, height],
        "mask_original_size": [mask.shape[1], mask.shape[0]],
        "mask_pixels_original": int(mask.sum()),
        "pose_joint_count": len(points),
        "output_image": str(args.output),
        "validated": [
            "video_frame_read",
            "relation_mask_decode",
            "relation_mask_video_alignment",
            "egopose_frame_camera_alignment",
        ],
        "not_yet_validated": [
            "object_3d_position",
            "human_centric_direction",
            "metric_distance",
        ],
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
    print(report_path)


if __name__ == "__main__":
    main()
