#!/usr/bin/env python3
"""Select an unannotated object from language and measure its human relation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.distance_validation import validate_metric_distance
from limo4si.human_frame import HumanFrame, build_human_frame, describe_relation
from limo4si.open_vocab import Candidate, OpenVocabularyGrounder, filter_candidates, parse_referent, resolve_candidate, target_terms
from limo4si.spatial_real import project_world_points, robust_object_center, select_mask_points
from limo4si.spatial_visuals import draw_forward_axis, draw_pose_skeleton


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--take-uid", required=True)
    p.add_argument("--camera", required=True)
    p.add_argument("--frame", required=True, type=int)
    p.add_argument("--query", required=True)
    p.add_argument("--root", type=Path, default=Path("data/egoexo4d"))
    p.add_argument("--output-dir", type=Path, default=Path("outputs/spatial/open_vocab"))
    p.add_argument("--detector", type=Path, default=Path("models/grounding-dino-tiny"))
    p.add_argument("--segmenter", type=Path, default=Path("models/sam2.1-hiera-tiny"))
    p.add_argument("--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    p.add_argument("--box-threshold", type=float, default=0.25)
    p.add_argument("--text-threshold", type=float, default=0.20)
    p.add_argument("--max-box-area-frac", type=float, default=0.12)
    p.add_argument("--forward-sign", type=int, choices=(-1, 1))
    p.add_argument("--orientation-overrides", type=Path, default=Path("configs/spatial_orientation_overrides.json"))
    p.add_argument("--min-distance-m", type=float, default=0.60)
    p.add_argument("--dead-zone-m", type=float, default=0.15)
    p.add_argument("--skip-sam", action="store_true", help="Use the selected box as a rectangular mask.")
    p.add_argument("--candidate-index", type=int, help="Resolve a needs-confirmation case by candidate index.")
    return p.parse_args()


def xyz_dict(annotation: dict) -> dict[str, list[float]]:
    return {
        name.replace("-", "_"): [float(point[a]) for a in "xyz"]
        for name, point in annotation.items()
        if point and all(a in point for a in "xyz")
    }


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot read frame {index} from {path}")
    return frame


def calibration_size(calibration: dict) -> tuple[int, int]:
    k = np.asarray(calibration["camera_intrinsics"])
    return int(round(2 * k[0, 2])), int(round(2 * k[1, 2]))


def draw_overlay(image: np.ndarray, candidates: list[Candidate], selected: Candidate | None) -> np.ndarray:
    canvas = image.copy()
    for c in candidates:
        x1, y1, x2, y2 = map(lambda v: int(round(v)), c.box_xyxy)
        color = (0, 255, 0) if c is selected else (0, 180, 255)
        if c.mask is not None:
            tint = np.zeros_like(canvas)
            tint[c.mask] = color
            canvas = cv2.addWeighted(canvas, 1.0, tint, 0.30, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        tag = f"#{c.index} L{c.level_from_top or '-'} R{c.ordinal_in_level or '-'} {c.score:.2f}"
        cv2.putText(canvas, tag, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
    return canvas





def draw_topdown(result: dict, output: Path, dead_zone_m: float) -> None:
    size = 760
    margin = 90
    image = np.full((size, size, 3), 248, dtype=np.uint8)
    x = float(result["human_xyz_m"]["right"])
    z = float(result["human_xyz_m"]["forward"])
    extent = max(0.75, abs(x) * 1.25, abs(z) * 1.25, result["min_distance_m"] * 1.35)
    scale = (size / 2 - margin) / extent
    origin = np.array([size // 2, size // 2])

    def pixel(x_m: float, z_m: float) -> tuple[int, int]:
        value = origin + np.array([x_m * scale, -z_m * scale])
        return int(round(value[0])), int(round(value[1]))

    cv2.rectangle(image, pixel(-dead_zone_m, dead_zone_m), pixel(dead_zone_m, -dead_zone_m), (226, 226, 226), -1)
    cv2.circle(image, tuple(origin), int(round(result["min_distance_m"] * scale)), (210, 210, 210), 2)
    cv2.line(image, (margin, origin[1]), (size - margin, origin[1]), (170, 170, 170), 2)
    cv2.line(image, (origin[0], margin), (origin[0], size - margin), (170, 170, 170), 2)
    cv2.arrowedLine(image, tuple(origin), pixel(0, extent * 0.42), (40, 120, 220), 5)
    cv2.arrowedLine(image, tuple(origin), pixel(extent * 0.42, 0), (220, 120, 40), 5)
    cv2.circle(image, tuple(origin), 15, (30, 30, 30), -1)
    object_pixel = pixel(x, z)
    color = (30, 40, 230) if result["recognition_status"] == "eligible" else (0, 160, 255)
    cv2.line(image, tuple(origin), object_pixel, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.circle(image, object_pixel, 18, color, -1)
    cv2.circle(image, object_pixel, 24, (255, 255, 255), 3)
    labels = [
        f"{result['take_name']}  frame {result['frame']}  {result['camera']}",
        f"query: {result['query']['raw']}",
        f"status={result['recognition_status']}  distance={result['distance_m']:.2f} m",
        f"right={x:+.2f} m  forward={z:+.2f} m  up={result['human_xyz_m']['up']:+.2f} m",
        f"relation={result.get('lateral_relation')}/{result.get('longitudinal_relation')}/{result.get('vertical_relation')}",
    ]
    for index, text in enumerate(labels):
        cv2.putText(image, text, (18, 30 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(image, "FRONT", (origin[0] + 12, margin + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 120, 220), 2)
    cv2.putText(image, "RIGHT", (size - margin - 75, origin[1] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 120, 40), 2)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"Failed to write {output}")


def draw_reprojection(image: np.ndarray, result: dict, selected: Candidate, calibration: dict, pose_record: dict, output: Path) -> dict:
    mask = selected.mask.astype(bool)
    point = np.asarray(result["object_xyz_world_m"], dtype=np.float64).reshape(1, 3)
    projected, depth = project_world_points(point, calibration["camera_intrinsics"], calibration["camera_extrinsics"])
    annotation_width, annotation_height = calibration_size(calibration)
    height, width = image.shape[:2]
    u, v = projected[0]
    point_small = (int(round(u * width / annotation_width)), int(round(v * height / annotation_height)))
    overlay = image.copy()
    overlay[mask] = (overlay[mask].astype(np.float32) * 0.25 + np.array([0, 220, 40], dtype=np.float32) * 0.75).astype(np.uint8)
    canvas = cv2.addWeighted(image, 0.35, overlay, 0.65, 0)
    pose_joint_count = draw_pose_skeleton(
        canvas,
        pose_record["annotation2D"][result["camera"]],
        annotation_width=annotation_width,
        annotation_height=annotation_height,
    )
    forward_axis_drawn = draw_forward_axis(
        canvas,
        result["human_frame"],
        calibration,
        annotation_width=annotation_width,
        annotation_height=annotation_height,
    )
    in_bounds = 0 <= point_small[0] < width and 0 <= point_small[1] < height
    inside_mask = bool(in_bounds and mask[point_small[1], point_small[0]])
    color = (40, 40, 240) if inside_mask else (0, 180, 255)
    cv2.drawMarker(canvas, point_small, color, cv2.MARKER_CROSS, 34, 4)
    cv2.circle(canvas, point_small, 18, color, 3)
    lines = [
        f"{result['take_name']}  frame {result['frame']}  {result['camera']}",
        f"query: {result['query']['raw']}",
        f"3D centroid -> ({u:.1f}, {v:.1f}) px  depth={depth[0]:.2f} m",
        f"status={result['recognition_status']}  distance={result['distance_m']:.2f} m",
        "green=selected mask  red=3D centroid  yellow=skeleton  magenta=body front",
    ]
    banner = 150
    output_image = cv2.copyMakeBorder(canvas, banner, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    for index, text in enumerate(lines):
        cv2.putText(output_image, text, (12, 27 + index * 27), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output), output_image):
        raise RuntimeError(f"Failed to write {output}")
    return {
        "projected_centroid_px": [float(u), float(v)],
        "camera_depth_m": float(depth[0]),
        "inside_video_image": in_bounds,
        "inside_selected_mask": inside_mask,
        "pose_joint_count": pose_joint_count,
        "forward_axis_drawn": forward_axis_drawn,
    }


def main() -> None:
    a = args()
    takes = {r["take_uid"]: r for r in json.loads((a.root / "takes.json").read_text())}
    take = takes[a.take_uid]
    video = a.root / "takes" / take["take_name"] / "frame_aligned_videos" / "downscaled" / "448" / f"{a.camera}.mp4"
    bgr = read_frame(video, a.frame)
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    query = parse_referent(a.query)
    if a.box:
        box = list(a.box)
        candidates = [Candidate(0, query.target, 1.0, box, [(box[0]+box[2])/2, (box[1]+box[3])/2])]
        grounder = None if a.skip_sam else OpenVocabularyGrounder(a.detector, a.segmenter)
    else:
        grounder = OpenVocabularyGrounder(a.detector, a.segmenter)
        candidates = []
        for term in target_terms(query.target):
            candidates.extend(
                grounder.detect(
                    image,
                    term,
                    box_threshold=a.box_threshold,
                    text_threshold=a.text_threshold,
                    start_index=len(candidates),
                )
            )
        candidates = filter_candidates(candidates, image.size, max_box_area_frac=a.max_box_area_frac)
    if grounder is not None:
        grounder.segment(image, candidates)
    elif a.box:
        x1, y1, x2, y2 = [int(round(v)) for v in a.box]
        mask = np.zeros(bgr.shape[:2], dtype=bool)
        mask[max(0, y1):min(mask.shape[0], y2), max(0, x1):min(mask.shape[1], x2)] = True
        candidates[0].mask = mask
    selected, resolution = resolve_candidate(candidates, query)
    if selected is None and a.candidate_index is not None:
        by_index = {c.index: c for c in candidates}
        if a.candidate_index not in by_index:
            raise ValueError(f"candidate index {a.candidate_index} not in detected candidates")
        selected = by_index[a.candidate_index]
        resolution = {
            **resolution,
            "status": "resolved",
            "reason": "resolved_by_candidate_index",
            "matching_candidate_indices": [selected.index],
            "manual_confirmation": True,
        }

    a.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{take['take_name']}_{a.camera}_frame{a.frame}"
    overlay = draw_overlay(bgr, candidates, selected)
    overlay_path = a.output_dir / f"{stem}_candidates.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    result = {
        "status": resolution["status"],
        "take_uid": a.take_uid, "take_name": take["take_name"],
        "camera": a.camera, "frame": a.frame,
        "query": query.to_dict(),
        "selection": resolution,
        "candidates": [c.to_dict() for c in candidates],
        "mask_provenance": "manual_box_plus_sam2" if a.box else "grounding_dino_plus_sam2",
        "overlay": str(overlay_path),
    }
    if selected is not None and selected.mask is not None:
        camera_path = a.root / "annotations" / "ego_pose" / "val" / "camera_pose" / f"{a.take_uid}.json"
        if not camera_path.exists():
            for candidate_path in (
                ROOT / "outputs" / "calibration" / "val_12" / f"{a.take_uid}.json",
                ROOT / "outputs" / "calibration" / "val_3" / f"{a.take_uid}.json",
            ):
                if candidate_path.exists():
                    camera_path = candidate_path
                    break
        calibration = json.loads(camera_path.read_text())[a.camera]
        width, height = calibration_size(calibration)
        mask = cv2.resize(selected.mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        body_path = a.root / "annotations" / "ego_pose" / "val" / "body" / "automatic" / f"{a.take_uid}.json"
        pose = json.loads(body_path.read_text())[str(a.frame)][0]
        frame3d = build_human_frame(xyz_dict(pose["annotation3D"]))
        orientation_source = "nose_or_default"
        forward_sign = a.forward_sign
        if forward_sign is None and a.orientation_overrides.exists():
            overrides = json.loads(a.orientation_overrides.read_text())
            row = overrides.get(a.take_uid, {})
            forward_sign = row.get("forward_sign")
            orientation_source = row.get("source", "orientation_overrides") if forward_sign is not None else orientation_source
        if forward_sign is None:
            forward_sign = 1
        if forward_sign == -1:
            frame3d = HumanFrame(frame3d.origin, frame3d.right, frame3d.up, tuple(-x for x in frame3d.forward))
        cloud = a.root / "takes" / take["take_name"] / "trajectory" / "semidense_points.csv.gz"
        points = select_mask_points(cloud, mask, calibration["camera_intrinsics"], calibration["camera_extrinsics"])
        center, inliers = robust_object_center(points)
        relation = describe_relation(frame3d.world_to_human(center), dead_zone_m=a.dead_zone_m)
        check = validate_metric_distance(center, frame3d.to_dict(), relation["human_xyz_m"], pose["annotation3D"])
        eligible = relation["distance_m"] >= a.min_distance_m and check["validated"]
        raw_relation = dict(relation)
        if not eligible:
            for key in ("lateral_relation", "longitudinal_relation", "vertical_relation", "text_zh"):
                relation[key] = None
        result.update({
            "status": "ok" if eligible else "filtered_near_or_invalid",
            "recognition_status": "eligible" if eligible else "filtered_near_or_invalid",
            "filter_reason": None if eligible else "distance_below_threshold_or_distance_validation_failed",
            "min_distance_m": a.min_distance_m,
            "selected_candidate_index": selected.index,
            "object_xyz_world_m": center.tolist(),
            "human_frame": frame3d.to_dict(),
            "orientation": {"forward_sign": forward_sign, "source": orientation_source},
            "distance_definition": {
                "distance_m": "Euclidean distance from pelvis midpoint to robust 3D object centroid",
                "horizontal_distance_m": "Euclidean norm of right and forward components",
                "minimum_eligible_m": a.min_distance_m,
            },
            "distance_validation": check,
            "raw_relation_before_filter": raw_relation,
            **relation,
            "quality": {
                "mask_pixels_video": int(selected.mask.sum()),
                "mask_pixels_calibration": int(mask.sum()),
                "points_in_mask": int(len(points.xyz_world)),
                "robust_inliers": int(len(inliers)),
            },
            "inputs": {"video": str(video), "point_cloud": str(cloud), "camera_pose": str(camera_path), "body_pose": str(body_path)},
        })
        topdown_path = a.output_dir / f"{stem}_topdown.jpg"
        reprojection_path = a.output_dir / f"{stem}_reprojection.jpg"
        draw_topdown(result, topdown_path, a.dead_zone_m)
        result["acceptance_topdown"] = str(topdown_path)
        result["acceptance_reprojection"] = str(reprojection_path)
        result["reprojection_check"] = draw_reprojection(bgr, result, selected, calibration, pose, reprojection_path)
    output = a.output_dir / f"{stem}.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(output)
    if result["status"] in ("needs_confirmation",):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
