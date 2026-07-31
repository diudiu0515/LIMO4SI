#!/usr/bin/env python3
"""Compute an object's relation to a person from common-frame 3D points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from limo4si import build_human_frame, describe_relation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--origin", choices=("pelvis", "shoulders"), default="pelvis")
    parser.add_argument("--dead-zone-m", type=float, default=0.15)
    args = parser.parse_args()

    sample = json.loads(args.input.read_text())
    frame = build_human_frame(sample["person_joints_world"], origin_mode=args.origin)
    human_xyz = frame.world_to_human(sample["object_xyz_world"])
    result = {
        "object_name": sample.get("object_name", "object"),
        "frame": frame.to_dict(),
        **describe_relation(human_xyz, dead_zone_m=args.dead_zone_m),
    }
    text = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(args.output)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
