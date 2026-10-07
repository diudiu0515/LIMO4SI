#!/usr/bin/env python3
"""Convert ADT multi-person ground truth into normalized Task 4 scenes."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.adt_task4 import AdtTask4Error, load_adt_task4_scenes  # noqa: E402


def _convert_sequence(
    work: tuple[Path, dict[str, list[list[float]]]],
) -> tuple[str, list[dict], str | None]:
    """Convert one source in an isolated process and return an audit result."""
    sequence_dir, device_cpf_rotations = work
    try:
        converted = load_adt_task4_scenes(
            sequence_dir, device_cpf_rotations=device_cpf_rotations,
        )
    except (AdtTask4Error, json.JSONDecodeError, OSError) as exc:
        return sequence_dir.name, [], str(exc)
    if not converted:
        return sequence_dir.name, [], "no window passed all evidence gates"
    return sequence_dir.name, converted, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/adt"))
    parser.add_argument(
        "--cpf-calibrations", type=Path,
        default=Path("configs/adt_device_cpf_calibrations.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--output-shards-dir", type=Path,
        help="Write one normalized JSON shard per source sequence plus _manifest.json.",
    )
    parser.add_argument(
        "--workers", type=int, default=min(8, os.cpu_count() or 1),
        help="Independent sequence-conversion processes (default: up to 8).",
    )
    args = parser.parse_args()
    if args.output is None and args.output_shards_dir is None:
        parser.error("one of --output or --output-shards-dir is required")
    if args.workers < 1:
        parser.error("--workers must be positive")

    calibration_payload = json.loads(args.cpf_calibrations.read_text(encoding="utf-8"))
    device_cpf_rotations = {
        serial: row["rotation_device_cpf"]
        for serial, row in calibration_payload.get("devices", {}).items()
    }
    scenes = []
    shard_sources = []
    rejected = []
    accepted_scene_count = 0
    accepted_source_videos: set[str] = set()
    if args.output_shards_dir is not None:
        args.output_shards_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted(path for path in args.dataset_root.iterdir() if path.is_dir())
    work = [(path, device_cpf_rotations) for path in sequence_dirs]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for sequence_name, converted, reason in executor.map(_convert_sequence, work):
            accepted_scene_count += len(converted)
            accepted_source_videos.update(
                str(scene["source_video"]) for scene in converted
            )
            if args.output is not None:
                scenes.extend(converted)
            if converted and args.output_shards_dir is not None:
                shard_name = f"{sequence_name}.json"
                (args.output_shards_dir / shard_name).write_text(
                    json.dumps({
                        "schema": "limo4si.task4_annotations.v1",
                        "dataset": "Aria Digital Twin",
                        "reasoning_owner": "deterministic_code",
                        "source_sequence": sequence_name,
                        "scenes": converted,
                    }, indent=2) + "\n",
                    encoding="utf-8",
                )
                shard_sources.append(shard_name)
            if reason is not None:
                rejected.append({"sequence": sequence_name, "reason": reason})
    audit = {
        "accepted_scene_count": accepted_scene_count,
        "accepted_source_video_count": len(accepted_source_videos),
        "rejected": rejected,
    }
    payload = {
        "schema": "limo4si.task4_annotations.v1",
        "dataset": "Aria Digital Twin",
        "reasoning_owner": "deterministic_code",
        "scenes": scenes,
        "audit": audit,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.output_shards_dir is not None:
        (args.output_shards_dir / "_manifest.json").write_text(
            json.dumps({
                "schema": "limo4si.task4_annotation_manifest.v1",
                "sources": shard_sources,
                "audit": audit,
            }, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
