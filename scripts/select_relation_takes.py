#!/usr/bin/env python3
"""Rank Ego-Exo4D relation takes for a small spatial-reasoning prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HAND_OR_PERSON = (
    "hand",
    "arm",
    "person",
    "body",
    "head",
    "foot",
    "leg",
)


def camera_stats(object_masks: dict) -> tuple[set[str], set[str], int]:
    ego, exo, visible_frames = set(), set(), 0
    for cameras in object_masks.values():
        for camera, track in cameras.items():
            target = ego if camera.startswith("aria") else exo
            target.add(camera)
            visible_frames += len(track.get("annotation", {}))
    return ego, exo, visible_frames


def score_take(entry: dict, take_meta: dict) -> dict:
    object_ids = list(entry.get("object_masks", {}))
    scene_objects = [
        name for name in object_ids
        if not any(token in name.lower() for token in HAND_OR_PERSON)
    ]
    ego, exo, visible_frames = camera_stats(entry.get("object_masks", {}))
    has_trajectory = bool(take_meta.get("has_trimmed_trajectory"))
    duration = float(take_meta.get("duration_sec") or 0.0)

    score = (
        min(len(scene_objects), 20) * 3
        + min(len(exo), 4) * 4
        + min(visible_frames / 100.0, 20)
        + (8 if ego and exo else 0)
        + (5 if has_trajectory else 0)
        - max(0.0, duration - 900.0) / 100.0
    )
    return {
        "score": round(score, 3),
        "scenario": entry.get("scenario"),
        "take_name": entry.get("take_name"),
        "duration_sec": duration,
        "scene_object_count": len(scene_objects),
        "scene_objects": scene_objects,
        "ego_cameras": sorted(ego),
        "exo_cameras": sorted(exo),
        "annotated_mask_frames": visible_frames,
        "has_trimmed_trajectory": has_trajectory,
        "best_exo": take_meta.get("best_exo"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/egoexo4d"),
        help="Ego-Exo4D download root",
    )
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--count", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selection"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    relation_path = args.root / "annotations" / f"relations_{args.split}.json"
    if not relation_path.exists():
        raise SystemExit(f"Missing completed annotation file: {relation_path}")

    with (args.root / "takes.json").open() as handle:
        takes = json.load(handle)
    take_by_uid = {take["take_uid"]: take for take in takes}

    with relation_path.open() as handle:
        relation_data = json.load(handle)

    ranked = []
    for take_uid, entry in relation_data["annotations"].items():
        meta = take_by_uid.get(take_uid, {})
        result = score_take(entry, meta)
        result["take_uid"] = take_uid
        ranked.append(result)

    ranked.sort(key=lambda item: item["score"], reverse=True)
    selected = ranked[: args.count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    uid_path = args.output_dir / f"{args.split}_{args.count}_uids.txt"
    report_path = args.output_dir / f"{args.split}_{args.count}_report.json"
    uid_path.write_text("\n".join(row["take_uid"] for row in selected) + "\n")
    report_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2) + "\n")

    print(f"Selected {len(selected)} takes from {len(ranked)} candidates")
    print(f"UIDs:   {uid_path}")
    print(f"Report: {report_path}")
    for row in selected:
        print(
            f"{row['score']:6.2f}  {row['take_uid']}  "
            f"{row['scenario']}  objects={row['scene_object_count']} "
            f"ego/exo={len(row['ego_cameras'])}/{len(row['exo_cameras'])}"
        )


if __name__ == "__main__":
    main()
