#!/usr/bin/env python3
"""Audit existing spatial results and gate near-object direction labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from limo4si.distance_validation import validate_metric_distance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/spatial/val_3"),
            Path("outputs/spatial/val_12"),
        ],
    )
    parser.add_argument("--min-distance-m", type=float, default=0.60)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/spatial/distance_audit.json"),
    )
    args = parser.parse_args()

    rows = []
    for directory in args.results:
        summary = json.loads((directory / "summary.json").read_text())
        for result in summary["samples"]:
            pose = json.loads(Path(result["inputs"]["body_pose"]).read_text())
            joints = pose[str(result["frame"])][0]["annotation3D"]
            validation = validate_metric_distance(
                result["object_xyz_world_m"],
                result["human_frame"],
                result["human_xyz_m"],
                joints,
            )
            eligible = (
                result["distance_m"] >= args.min_distance_m
                and validation["validated"]
            )
            rows.append(
                {
                    "recognition_status": (
                        "eligible" if eligible else "filtered_near_or_invalid"
                    ),
                    "take_uid": result["take_uid"],
                    "take_name": result["take_name"],
                    "frame": result["frame"],
                    "camera": result["camera"],
                    "object_id": result["object_id"],
                    "distance_m": result["distance_m"],
                    "min_distance_m": args.min_distance_m,
                    "spatial_relation": (
                        {
                            "lateral": result["lateral_relation"],
                            "longitudinal": result["longitudinal_relation"],
                            "vertical": result["vertical_relation"],
                        }
                        if eligible
                        else None
                    ),
                    "distance_validation": validation,
                }
            )
    report = {
        "policy": {
            "min_distance_m": args.min_distance_m,
            "near_objects_have_no_direction_label": True,
        },
        "sample_count": len(rows),
        "eligible_count": sum(row["recognition_status"] == "eligible" for row in rows),
        "filtered_count": sum(
            row["recognition_status"] != "eligible" for row in rows
        ),
        "distance_validation_pass_count": sum(
            row["distance_validation"]["validated"] for row in rows
        ),
        "samples": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(args.output)
    print(
        f"eligible={report['eligible_count']} filtered={report['filtered_count']} "
        f"distance_valid={report['distance_validation_pass_count']}/{report['sample_count']}"
    )


if __name__ == "__main__":
    main()
