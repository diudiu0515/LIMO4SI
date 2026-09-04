#!/usr/bin/env python3
"""Ground manually reviewed, scene-fixed Task 3 landmarks in metric world coordinates."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.spatial_real import PointSelection, iter_semidense_points, project_world_points, robust_object_center


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fps_for_video(video: Path) -> float:
    raw = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", str(video)],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    left, *right = raw.split("/")
    return float(left) / float(right[0]) if right else float(left)


def extract_review_frame(sample: dict[str, Any], output: Path, width: int, height: int) -> None:
    if output.exists() and output.stat().st_size > 1000:
        return
    video = ROOT / "data/egoexo4d/takes" / sample["take_name"] / "frame_aligned_videos/downscaled/448" / f"{sample['camera']}.mp4"
    fps = fps_for_video(video)
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-ss", f"{int(sample['frame']) / fps:.6f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:{height}", str(output)],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def collect_box_points(
    point_cloud: Path,
    calibration: dict[str, Any],
    boxes: list[list[float]],
    frame_size: tuple[int, int],
    max_dist_std_m: float,
) -> tuple[list[PointSelection], dict[str, int]]:
    width, height = frame_size
    selected_xyz: list[list[np.ndarray]] = [[] for _ in boxes]
    selected_depth: list[list[np.ndarray]] = [[] for _ in boxes]
    scanned_total = quality_total = projected_total = 0
    intrinsics = calibration["camera_intrinsics"]
    cal_width = 2.0 * float(intrinsics[0][2])
    cal_height = 2.0 * float(intrinsics[1][2])
    for xyz, scanned, quality in iter_semidense_points(point_cloud, max_dist_std_m=max_dist_std_m):
        scanned_total += scanned
        quality_total += quality
        if not len(xyz):
            continue
        pixels, depth = project_world_points(xyz, intrinsics, calibration["camera_extrinsics"])
        px = pixels[:, 0] * width / cal_width
        py = pixels[:, 1] * height / cal_height
        visible = np.isfinite(px) & np.isfinite(py) & (depth > 0) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        projected_total += int(visible.sum())
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            inside = visible & (px >= x1) & (px <= x2) & (py >= y1) & (py <= y2)
            if inside.any():
                selected_xyz[index].append(xyz[inside])
                selected_depth[index].append(depth[inside])
    selections = []
    for xyz_parts, depth_parts in zip(selected_xyz, selected_depth):
        xyz = np.concatenate(xyz_parts) if xyz_parts else np.empty((0, 3), dtype=np.float64)
        depth = np.concatenate(depth_parts) if depth_parts else np.empty((0,), dtype=np.float64)
        selections.append(PointSelection(xyz, depth, scanned_total, quality_total, projected_total))
    return selections, {"scanned_points": scanned_total, "quality_points": quality_total, "projected_points": projected_total}


def project_to_review(center: np.ndarray, calibration: dict[str, Any], frame_size: tuple[int, int]) -> tuple[list[float], float]:
    pixels, depth = project_world_points(center.reshape(1, 3), calibration["camera_intrinsics"], calibration["camera_extrinsics"])
    width, height = frame_size
    intrinsics = calibration["camera_intrinsics"]
    cal_width = 2.0 * float(intrinsics[0][2])
    cal_height = 2.0 * float(intrinsics[1][2])
    return [float(pixels[0, 0] * width / cal_width), float(pixels[0, 1] * height / cal_height)], float(depth[0])


def ground_scene(scene: dict[str, Any], policy: dict[str, Any], frame_size: tuple[int, int], media_dir: Path) -> dict[str, Any]:
    summary_path = ROOT / scene["source_summary"]
    summary = load_json(summary_path)
    sample = next(row for row in summary["samples"] if row.get("object_xyz_world_m"))
    calibration_path = Path(sample["inputs"]["camera_pose"])
    point_cloud_path = Path(sample["inputs"]["point_cloud"])
    calibration_path = calibration_path if calibration_path.is_absolute() else ROOT / calibration_path
    point_cloud_path = point_cloud_path if point_cloud_path.is_absolute() else ROOT / point_cloud_path
    calibration = load_json(calibration_path)[sample["camera"]]
    boxes = [row["box_xyxy"] for row in scene["landmarks"]]
    selections, scan = collect_box_points(
        point_cloud_path, calibration, boxes, frame_size,
        float(policy["max_point_dist_std_m"]),
    )
    results = []
    for spec, selection in zip(scene["landmarks"], selections):
        center, inliers = robust_object_center(selection, min_points=int(policy["minimum_robust_inliers"]))
        projected, camera_depth = project_to_review(center, calibration, frame_size)
        x1, y1, x2, y2 = spec["box_xyxy"]
        inside = x1 <= projected[0] <= x2 and y1 <= projected[1] <= y2
        if len(selection.xyz_world) < int(policy["minimum_selected_points"]):
            raise ValueError(f"{sample['take_name']}:{spec['id']} has only {len(selection.xyz_world)} selected points")
        if not inside or camera_depth <= 0:
            raise ValueError(f"{sample['take_name']}:{spec['id']} failed centroid reprojection audit")
        radius = np.linalg.norm(inliers - center, axis=1)
        results.append({
            "object_id": spec["id"],
            "display_name": spec["display_name"],
            "object_xyz_world_m": [float(value) for value in center],
            "static_scene_landmark": True,
            "grounding": {
                "method": "human-reviewed fixed-scene box plus semidense metric point-cloud centroid",
                "source_frame": int(sample["frame"]),
                "source_camera": sample["camera"],
                "box_xyxy_796x448": [float(value) for value in spec["box_xyxy"]],
                "selected_points": int(len(selection.xyz_world)),
                "robust_inlier_points": int(len(inliers)),
                "inlier_radius_p90_m": float(np.quantile(radius, 0.90)),
                "reprojected_centroid_796x448": projected,
                "camera_depth_m": camera_depth,
                "centroid_reprojects_inside_box": inside,
                "manual_static_review": True
            }
        })
    stem = f"{sample['take_name']}_{sample['camera']}_frame{sample['frame']}"
    review_frame = media_dir / f"{stem}_static_landmarks.jpg"
    extract_review_frame(sample, review_frame, *frame_size)
    image = Image.open(review_frame).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = ["#ffb000", "#00b7ff", "#ff4fb3", "#6ee36e"]
    for index, (spec, result) in enumerate(zip(scene["landmarks"], results)):
        color = colors[index % len(colors)]
        box = tuple(spec["box_xyxy"])
        draw.rectangle(box, outline=color, width=4)
        point = result["grounding"]["reprojected_centroid_796x448"]
        x, y = int(round(point[0])), int(round(point[1]))
        draw.line((x - 8, y, x + 8, y), fill=color, width=4)
        draw.line((x, y - 8, x, y + 8), fill=color, width=4)
        draw.text((box[0] + 4, box[1] + 4), spec["display_name"], fill=color, font=font, stroke_width=2, stroke_fill="black")
    image.save(review_frame, quality=92)
    return {
        "source_summary": scene["source_summary"],
        "take_uid": sample["take_uid"],
        "take_name": sample["take_name"],
        "camera": sample["camera"],
        "frame": int(sample["frame"]),
        "review_image": str(review_frame.relative_to(ROOT)),
        "landmark_count": len(results),
        "landmarks": results,
        "point_cloud_scan": scan,
        "status": "ok"
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task3_static_landmarks.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qa/task3_static_landmarks.json"))
    parser.add_argument("--media-dir", type=Path, default=Path("outputs/qa/task3_media"))
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    config = load_json(resolve(args.config))
    frame_size = tuple(int(value) for value in config["frame_size"])
    scenes = []
    for index, scene in enumerate(config["scenes"], 1):
        print(f"[{index}/{len(config['scenes'])}] grounding {scene['source_summary']}", flush=True)
        scenes.append(ground_scene(scene, config["grounding_policy"], frame_size, resolve(args.media_dir)))
    result = {
        "status": "ok",
        "coordinate_frame": "Ego-Exo4D metric world coordinates",
        "frame_size": list(frame_size),
        "grounding_policy": config["grounding_policy"],
        "scene_count": len(scenes),
        "scenes": scenes
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
