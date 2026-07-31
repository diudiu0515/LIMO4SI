#!/usr/bin/env python3
"""Select 12 additional relation samples and derive missing exo calibrations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ego4d.research.util.masks import decode_mask

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from limo4si.projective_calibration import (  # noqa: E402
    collect_pose_correspondences,
    decompose_projection_matrix,
    estimate_projection_matrix,
    reprojection_statistics,
)

ROOT = Path("data/egoexo4d")
BODY_ROOT = ROOT / "annotations" / "ego_pose" / "val" / "body" / "automatic"
CALIBRATION_DIR = Path("outputs/calibration/val_12")
SAMPLE_PATH = Path("examples/val_12_spatial_samples.json")
REPORT_PATH = Path("outputs/selection/val_12_spatial_report.json")
REQUIRED = ("left-shoulder", "right-shoulder", "left-hip", "right-hip", "nose")


def valid_pose(body: dict, frame: str) -> bool:
    records = body.get(frame)
    if not records:
        return False
    joints = records[0].get("annotation3D", {})
    return all(joints.get(name) for name in REQUIRED)


def derive_calibrations(body: dict) -> tuple[dict, dict]:
    first_record = next(records[0] for records in body.values() if records)
    cameras = [
        camera
        for camera in first_record.get("annotation2D", {})
        if camera.startswith(("cam", "gp"))
    ]
    calibrations, reports = {}, {}
    for camera in cameras:
        world, image = collect_pose_correspondences(body, camera)
        if len(world) < 100:
            continue
        projection = estimate_projection_matrix(world, image)
        intrinsics, extrinsics = decompose_projection_matrix(projection)
        stats = reprojection_statistics(world, image, intrinsics, extrinsics)
        stats["accepted"] = stats["median_px"] <= 3.0 and stats["p95_px"] <= 10.0
        reports[camera] = stats
        if stats["accepted"]:
            calibrations[camera] = {
                "camera_intrinsics": intrinsics.tolist(),
                "camera_extrinsics": extrinsics.tolist(),
                "distortion_coeffs": [],
                "source": "egopose_3d2d_normalized_dlt",
                "fit": stats,
            }
    return calibrations, reports


def object_candidates(uid_relations: dict, body: dict, cameras: set[str]) -> list[dict]:
    candidates = []
    for object_id, tracks in uid_relations["object_masks"].items():
        best = None
        for camera, track in tracks.items():
            if camera not in cameras:
                continue
            annotations = track.get("annotation", {})
            ranked = sorted(
                (
                    (len(record.get("encodedMask", "")), frame, record)
                    for frame, record in annotations.items()
                    if valid_pose(body, frame)
                ),
                reverse=True,
            )
            for _, frame, record in ranked[:3]:
                mask_pixels = int(decode_mask(record).sum())
                row = {
                    "camera": camera,
                    "frame": int(frame),
                    "object_id": object_id,
                    "mask_pixels": mask_pixels,
                }
                if best is None or row["mask_pixels"] > best["mask_pixels"]:
                    best = row
        if best is not None:
            candidates.append(best)
    return sorted(candidates, key=lambda row: row["mask_pixels"], reverse=True)


def main() -> None:
    uids = Path("outputs/selection/val_15_uids.txt").read_text().split()
    existing = set(Path("outputs/selection/val_3_uids.txt").read_text().split())
    takes = {row["take_uid"]: row for row in json.loads((ROOT / "takes.json").read_text())}
    relations = json.loads((ROOT / "annotations" / "relations_val.json").read_text())[
        "annotations"
    ]
    no_pose_mask_overlap = {"6eb10b39-5171-4293-afba-4084f5825748"}
    available = [
        uid for uid in uids
        if uid not in existing
        and uid not in no_pose_mask_overlap
        and (BODY_ROOT / f"{uid}.json").exists()
    ]
    excluded = [uid for uid in uids if uid not in existing and uid not in available]
    # Five takes contribute two objects and two contribute one: 12 total.
    allocation = {uid: 2 if index < 5 else 1 for index, uid in enumerate(available)}
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    samples, take_reports = [], []
    for uid in available:
        take = takes[uid]
        body = json.loads((BODY_ROOT / f"{uid}.json").read_text())
        calibrations, calibration_reports = derive_calibrations(body)
        calibration_path = CALIBRATION_DIR / f"{uid}.json"
        calibration_path.write_text(
            json.dumps(
                {"metadata": {"take_uid": uid, "take_name": take["take_name"]}, **calibrations},
                indent=2,
            )
            + "\n"
        )
        candidates = object_candidates(relations[uid], body, set(calibrations))
        selected = candidates[: allocation[uid]]
        if len(selected) != allocation[uid]:
            raise RuntimeError(f"Not enough valid objects for {take['take_name']}")
        for row in selected:
            samples.append(
                {
                    "take_uid": uid,
                    "camera": row["camera"],
                    "frame": row["frame"],
                    "object_id": row["object_id"],
                    "camera_pose_path": str(calibration_path),
                    "orientation_source": "nose",
                }
            )
        take_reports.append(
            {
                "take_uid": uid,
                "take_name": take["take_name"],
                "requested_count": allocation[uid],
                "selected": selected,
                "calibrations": calibration_reports,
            }
        )
    if len(samples) != 12:
        raise RuntimeError(f"Expected 12 samples, selected {len(samples)}")
    SAMPLE_PATH.write_text(json.dumps(samples, ensure_ascii=False, indent=2) + "\n")
    report = {
        "selected_sample_count": len(samples),
        "available_skeleton_take_count": len(available),
        "excluded_unusable_for_spatial": [
            {"take_uid": uid, "take_name": takes[uid]["take_name"]} for uid in excluded
        ],
        "takes": take_reports,
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(SAMPLE_PATH)
    print(REPORT_PATH)


if __name__ == "__main__":
    main()
