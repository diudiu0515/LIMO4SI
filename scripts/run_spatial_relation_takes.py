#!/usr/bin/env python3
"""Run metric human-centric spatial relations on selected Ego-Exo4D takes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from ego4d.research.util.masks import decode_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from limo4si.human_frame import HumanFrame, build_human_frame, describe_relation  # noqa: E402
from limo4si.distance_validation import validate_metric_distance  # noqa: E402
from limo4si.spatial_real import (  # noqa: E402
    project_world_points,
    robust_object_center,
    select_mask_points,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples",
        type=Path,
        default=Path("examples/val_3_spatial_samples.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/spatial/val_3")
    )
    parser.add_argument("--max-dist-std-m", type=float, default=0.10)
    parser.add_argument("--min-points", type=int, default=8)
    parser.add_argument("--dead-zone-m", type=float, default=0.15)
    parser.add_argument("--min-distance-m", type=float, default=0.60)
    return parser.parse_args()


def xyz_dict(annotation: dict) -> dict[str, list[float]]:
    result = {}
    for name, point in annotation.items():
        if point and all(axis in point for axis in "xyz"):
            result[name.replace("-", "_")] = [float(point[axis]) for axis in "xyz"]
    return result


def pose_reprojection_error(
    annotation_3d: dict,
    annotation_2d: dict,
    calibration: dict,
) -> dict:
    names = [
        name
        for name, point in annotation_3d.items()
        if point and annotation_2d.get(name)
    ]
    if not names:
        return {"joint_count": 0}
    world = np.asarray(
        [[annotation_3d[name][axis] for axis in "xyz"] for name in names]
    )
    expected = np.asarray(
        [[annotation_2d[name][axis] for axis in "xy"] for name in names]
    )
    projected, depth = project_world_points(
        world,
        calibration["camera_intrinsics"],
        calibration["camera_extrinsics"],
    )
    errors = np.linalg.norm(projected - expected, axis=1)
    return {
        "joint_count": len(names),
        "positive_depth_joint_count": int((depth > 0).sum()),
        "median_px": float(np.median(errors)),
        "mean_px": float(np.mean(errors)),
        "max_px": float(np.max(errors)),
    }


def process_sample(
    sample: dict,
    *,
    root: Path,
    takes: dict,
    relations: dict,
    output_dir: Path,
    max_dist_std_m: float,
    min_points: int,
    dead_zone_m: float,
    min_distance_m: float,
) -> dict:
    uid = sample["take_uid"]
    camera = sample["camera"]
    frame = int(sample["frame"])
    object_id = sample["object_id"]
    take = takes[uid]
    take_name = take["take_name"]

    mask_record = relations[uid]["object_masks"][object_id][camera]["annotation"][
        str(frame)
    ]
    mask = decode_mask(mask_record).astype(bool)
    camera_pose_path = Path(
        sample.get(
            "camera_pose_path",
            root / "annotations" / "ego_pose" / "val" / "camera_pose" / f"{uid}.json",
        )
    )
    calibration = json.loads(camera_pose_path.read_text())[camera]
    body_path = (
        root
        / "annotations"
        / "ego_pose"
        / "val"
        / "body"
        / "automatic"
        / f"{uid}.json"
    )
    pose_record = json.loads(body_path.read_text())[str(frame)][0]
    joints = xyz_dict(pose_record["annotation3D"])
    frame_3d = build_human_frame(joints)
    forward_sign = int(sample.get("forward_sign", 1))
    if forward_sign not in (-1, 1):
        raise ValueError("forward_sign must be -1 or 1")
    if forward_sign == -1:
        frame_3d = HumanFrame(
            origin=frame_3d.origin,
            right=frame_3d.right,
            up=frame_3d.up,
            forward=tuple(-value for value in frame_3d.forward),
        )

    point_cloud_path = (
        root / "takes" / take_name / "trajectory" / "semidense_points.csv.gz"
    )
    selection = select_mask_points(
        point_cloud_path,
        mask,
        calibration["camera_intrinsics"],
        calibration["camera_extrinsics"],
        max_dist_std_m=max_dist_std_m,
    )
    object_center, inliers = robust_object_center(selection, min_points=min_points)
    human_xyz = frame_3d.world_to_human(object_center)
    relation = describe_relation(human_xyz, dead_zone_m=dead_zone_m)
    distance_validation = validate_metric_distance(
        object_center,
        frame_3d.to_dict(),
        relation["human_xyz_m"],
        pose_record["annotation3D"],
    )
    eligible = (
        relation["distance_m"] >= min_distance_m
        and distance_validation["validated"]
    )
    if not eligible:
        relation["lateral_relation"] = None
        relation["longitudinal_relation"] = None
        relation["vertical_relation"] = None
        relation["text_zh"] = None

    result = {
        "status": "ok" if eligible else "filtered_near_or_invalid",
        "recognition_status": "eligible" if eligible else "filtered_near_or_invalid",
        "take_uid": uid,
        "take_name": take_name,
        "frame": frame,
        "camera": camera,
        "object_id": object_id,
        "object_xyz_world_m": object_center.tolist(),
        "human_frame": frame_3d.to_dict(),
        "orientation": {
            "forward_sign": forward_sign,
            "source": sample.get(
                "orientation_source",
                "nose" if "nose" in joints else "body_cross_product",
            ),
        },
        "distance_policy": {"min_distance_m": min_distance_m},
        "distance_validation": distance_validation,
        **relation,
        "quality": {
            "mask_pixels": int(mask.sum()),
            "point_cloud_scanned": selection.scanned_points,
            "point_cloud_quality": selection.quality_points,
            "point_cloud_in_image": selection.projected_points,
            "points_in_mask": int(len(selection.xyz_world)),
            "robust_inliers": int(len(inliers)),
            "max_point_dist_std_m": max_dist_std_m,
            "pose_reprojection": pose_reprojection_error(
                pose_record["annotation3D"],
                pose_record["annotation2D"][camera],
                calibration,
            ),
        },
        "inputs": {
            "point_cloud": str(point_cloud_path),
            "camera_pose": str(camera_pose_path),
            "body_pose": str(body_path),
        },
    }
    output_path = output_dir / f"{take_name}_frame{frame}_{object_id}.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    result["output_path"] = str(output_path)
    return result


def main() -> None:
    args = parse_args()
    samples = json.loads(args.samples.read_text())
    take_rows = json.loads((args.root / "takes.json").read_text())
    takes = {row["take_uid"]: row for row in take_rows}
    relation_file = json.loads(
        (args.root / "annotations" / "relations_val.json").read_text()
    )
    relations = relation_file["annotations"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sample in samples:
        label = f"{sample['take_uid']} frame={sample['frame']} {sample['object_id']}"
        print(f"Processing {label}", flush=True)
        try:
            result = process_sample(
                sample,
                root=args.root,
                takes=takes,
                relations=relations,
                output_dir=args.output_dir,
                max_dist_std_m=args.max_dist_std_m,
                min_points=args.min_points,
                dead_zone_m=args.dead_zone_m,
                min_distance_m=args.min_distance_m,
            )
        except Exception as error:
            result = {"status": "error", **sample, "error": str(error)}
            print(f"ERROR: {error}", file=sys.stderr, flush=True)
        results.append(result)

    summary = {
        "sample_count": len(results),
        "success_count": sum(row["status"] == "ok" for row in results),
        "filtered_count": sum(
            row["status"] == "filtered_near_or_invalid" for row in results
        ),
        "error_count": sum(row["status"] == "error" for row in results),
        "samples": results,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(summary_path)
    if summary["error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
