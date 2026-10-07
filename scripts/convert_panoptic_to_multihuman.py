#!/usr/bin/env python3
"""Convert extracted CMU Panoptic sequences to normalized Task 4 scenes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.panoptic_task4 import (  # noqa: E402
    PanopticTask4Error, load_panoptic_task4_scenes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/panoptic"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scenes = []
    rejected = []
    for sequence_dir in sorted(path for path in args.dataset_root.iterdir() if path.is_dir()):
        try:
            converted = load_panoptic_task4_scenes(sequence_dir)
        except (PanopticTask4Error, json.JSONDecodeError, OSError) as exc:
            rejected.append({"sequence": sequence_dir.name, "reason": str(exc)})
            continue
        scenes.extend(converted)
        if not converted:
            rejected.append({"sequence": sequence_dir.name, "reason": "no stable three-person 15-second window"})
    payload = {
        "schema": "limo4si.task4_annotations.v1",
        "dataset": "CMU Panoptic Studio",
        "reasoning_owner": "deterministic_code",
        "scenes": scenes,
        "audit": {
            "accepted_scene_count": len(scenes),
            "accepted_source_video_count": len({scene["source_video"] for scene in scenes}),
            "rejected": rejected,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["audit"], indent=2))


if __name__ == "__main__":
    main()
