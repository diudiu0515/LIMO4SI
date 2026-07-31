#!/usr/bin/env python3
"""Create top-down and exo reprojection acceptance plots for spatial results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ego4d.research.util.masks import decode_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from limo4si.spatial_real import project_world_points  # noqa: E402
from limo4si.spatial_visuals import draw_forward_axis, draw_pose_skeleton  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument(
        "--results", type=Path, default=Path("outputs/spatial/val_3")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/spatial/val_3/acceptance"),
    )
    parser.add_argument("--dead-zone-m", type=float, default=0.15)
    return parser.parse_args()


def read_frame(path: Path, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_number}: {path}")
    return image


def safe_stem(result: dict) -> str:
    object_name = result["object_id"].replace(" ", "_").replace("/", "_")
    return f"{result['take_name']}_frame{result['frame']}_{object_name}"


def draw_topdown(result: dict, output: Path, dead_zone_m: float) -> None:
    size = 760
    margin = 90
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    x = float(result["human_xyz_m"]["right"])
    z = float(result["human_xyz_m"]["forward"])
    extent = max(0.75, abs(x) * 1.25, abs(z) * 1.25)
    scale = (size / 2 - margin) / extent
    origin = np.array([size // 2, size // 2])

    def pixel(x_m: float, z_m: float) -> tuple[int, int]:
        value = origin + np.array([x_m * scale, -z_m * scale])
        return int(round(value[0])), int(round(value[1]))

    cv2.rectangle(
        image,
        pixel(-dead_zone_m, dead_zone_m),
        pixel(dead_zone_m, -dead_zone_m),
        (226, 226, 226),
        -1,
    )
    cv2.line(image, (margin, origin[1]), (size - margin, origin[1]), (170, 170, 170), 2)
    cv2.line(image, (origin[0], margin), (origin[0], size - margin), (170, 170, 170), 2)
    cv2.arrowedLine(image, tuple(origin), pixel(0, extent * 0.42), (40, 120, 220), 5)
    cv2.arrowedLine(image, tuple(origin), pixel(extent * 0.42, 0), (220, 120, 40), 5)
    cv2.circle(image, tuple(origin), 15, (30, 30, 30), -1)
    object_pixel = pixel(x, z)
    cv2.line(image, tuple(origin), object_pixel, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.circle(image, object_pixel, 18, (30, 40, 230), -1)
    cv2.circle(image, object_pixel, 24, (255, 255, 255), 3)

    labels = [
        f"{result['take_name']}  frame {result['frame']}  {result['camera']}",
        f"object: {result['object_id']}",
        f"lateral={result['lateral_relation']}  longitudinal={result['longitudinal_relation']}",
        f"right={x:+.2f} m  forward={z:+.2f} m  distance={result['distance_m']:.2f} m",
    ]
    for index, text in enumerate(labels):
        cv2.putText(
            image, text, (18, 30 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.60, (25, 25, 25), 2, cv2.LINE_AA
        )
    cv2.putText(image, "FRONT", (origin[0] + 12, margin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 120, 220), 2)
    cv2.putText(image, "RIGHT", (size - margin - 75, origin[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 120, 40), 2)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to write {output}")


def draw_reprojection(
    result: dict,
    *,
    root: Path,
    relations: dict,
    takes: dict,
    output: Path,
) -> dict:
    uid, camera, frame = result["take_uid"], result["camera"], int(result["frame"])
    calibration = json.loads(Path(result["inputs"]["camera_pose"]).read_text())[camera]
    point = np.asarray(result["object_xyz_world_m"], dtype=np.float64).reshape(1, 3)
    projected, depth = project_world_points(
        point, calibration["camera_intrinsics"], calibration["camera_extrinsics"]
    )
    u, v = projected[0]
    mask_record = relations[uid]["object_masks"][result["object_id"]][camera][
        "annotation"
    ][str(frame)]
    mask = decode_mask(mask_record).astype(bool)
    pose_record = json.loads(Path(result["inputs"]["body_pose"]).read_text())[str(frame)][0]
    rounded = np.rint([u, v]).astype(int)
    in_bounds = 0 <= rounded[0] < mask.shape[1] and 0 <= rounded[1] < mask.shape[0]
    inside_mask = bool(in_bounds and mask[rounded[1], rounded[0]])

    take_name = takes[uid]["take_name"]
    video_path = (
        root / "takes" / take_name / "frame_aligned_videos"
        / "downscaled" / "448" / f"{camera}.mp4"
    )
    image = read_frame(video_path, frame)
    height, width = image.shape[:2]
    mask_small = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    overlay = image.copy()
    overlay[mask_small] = (
        overlay[mask_small].astype(np.float32) * 0.25
        + np.array([0, 220, 40], dtype=np.float32) * 0.75
    ).astype(np.uint8)
    image = cv2.addWeighted(image, 0.35, overlay, 0.65, 0)
    pose_joint_count = draw_pose_skeleton(
        image,
        pose_record["annotation2D"][camera],
        annotation_width=mask.shape[1],
        annotation_height=mask.shape[0],
    )
    forward_axis_drawn = draw_forward_axis(
        image,
        result["human_frame"],
        calibration,
        annotation_width=mask.shape[1],
        annotation_height=mask.shape[0],
    )
    point_small = (
        int(round(u * width / mask.shape[1])),
        int(round(v * height / mask.shape[0])),
    )
    color = (40, 40, 240) if inside_mask else (0, 180, 255)
    cv2.drawMarker(image, point_small, color, cv2.MARKER_CROSS, 34, 4)
    cv2.circle(image, point_small, 18, color, 3)

    lines = [
        f"{take_name}  frame {frame}  {camera}",
        f"object: {result['object_id']}",
        f"3D centroid -> ({u:.1f}, {v:.1f}) px  depth={depth[0]:.2f} m",
        f"ACCEPT={'PASS' if inside_mask else 'FAIL'}: projected centroid inside Relations mask",
        "green=mask  red=3D centroid  cyan=skeleton  magenta=body front",
    ]
    banner = 150
    canvas = cv2.copyMakeBorder(
        image, banner, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)
    )
    for index, text in enumerate(lines):
        cv2.putText(
            canvas, text, (12, 27 + index * 27), cv2.FONT_HERSHEY_SIMPLEX,
            0.62, (235, 235, 235), 2, cv2.LINE_AA
        )
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError(f"Failed to write {output}")
    return {
        "take_uid": uid,
        "frame": frame,
        "camera": camera,
        "object_id": result["object_id"],
        "projected_centroid_px": [float(u), float(v)],
        "camera_depth_m": float(depth[0]),
        "inside_image": bool(in_bounds),
        "inside_relation_mask": inside_mask,
        "pose_joint_count": pose_joint_count,
        "forward_axis_drawn": forward_axis_drawn,
        "orientation": result.get("orientation", {}),
        "accepted": bool(inside_mask and depth[0] > 0),
        "topdown_image": str(output.with_name(output.name.replace("_reprojection", "_topdown"))),
        "reprojection_image": str(output),
    }


def main() -> None:
    args = parse_args()
    result_files = sorted(
        path for path in args.results.glob("*.json") if path.name != "summary.json"
    )
    results = [json.loads(path.read_text()) for path in result_files]
    results = [row for row in results if row.get("status") == "ok"]
    takes = {
        row["take_uid"]: row
        for row in json.loads((args.root / "takes.json").read_text())
    }
    relations = json.loads(
        (args.root / "annotations" / "relations_val.json").read_text()
    )["annotations"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    for result in results:
        stem = safe_stem(result)
        topdown = args.output_dir / f"{stem}_topdown.jpg"
        reprojection = args.output_dir / f"{stem}_reprojection.jpg"
        draw_topdown(result, topdown, args.dead_zone_m)
        reports.append(
            draw_reprojection(
                result,
                root=args.root,
                relations=relations,
                takes=takes,
                output=reprojection,
            )
        )
        print(topdown)
        print(reprojection)
    report = {
        "sample_count": len(reports),
        "accepted_count": sum(row["accepted"] for row in reports),
        "all_accepted": all(row["accepted"] for row in reports),
        "samples": reports,
    }
    report_path = args.output_dir / "acceptance.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(report_path)
    if not report["all_accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
