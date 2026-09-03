#!/usr/bin/env python3
"""Inventory metric human annotations before mining Task 4 clips."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

PERSON_FILE = re.compile(r"^(?P<sequence>.+)_person(?P<person>\d+)(?:_meta)?\.(?:npz|json)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/HOI-M3"))
    parser.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument("--output", type=Path, default=Path("outputs/qa/multihuman_annotation_coverage_audit.json"))
    args = parser.parse_args()

    people: dict[str, set[str]] = defaultdict(set)
    files: dict[str, list[str]] = defaultdict(list)
    for directory in (args.root / "smplx_with_distortion", args.root / "smplx"):
        if not directory.exists():
            continue
        for path in directory.glob("*_person*"):
            match = PERSON_FILE.match(path.name)
            if match and path.suffix == ".npz":
                sequence, person_id = match.group("sequence"), match.group("person")
                people[sequence].add(person_id)
                files[sequence].append(str(path))

    video_views: dict[str, int] = {}
    video_root = args.root / "videos"
    if video_root.exists():
        for directory in video_root.iterdir():
            if directory.is_dir():
                video_views[directory.name] = sum(path.suffix.lower() == ".mp4" for path in directory.iterdir())

    release_scenes = []
    if args.site_data.exists():
        text = args.site_data.read_text(encoding="utf-8")
        payload = text[text.index("=") + 1:].strip().rstrip(";")
        data = json.loads(payload)
        for group in data.get("groups", []):
            audit = group.get("visual_person_audit") or {}
            if audit:
                question = (group.get("qa") or [{}])[0]
                persistent = int(audit.get("persistent_visible_person_count") or 0)
                maximum = int(audit.get("max_visible_person_count") or persistent)
                metric = int(audit.get("metric_3d_track_count") or 0)
                qtype = question.get("question_type")
                release_scenes.append({
                    "scene_id": group.get("name"),
                    "question_type": qtype,
                    "reasoning_scope": "three-person 2D topology" if qtype in {
                        "visible_pair_topology_change_2d", "visible_pair_topology_consistency_2d",
                    } else "explicit metric 3D pair",
                    "persistent_visible_person_count": persistent,
                    "max_visible_person_count": maximum,
                    "metric_3d_track_count": metric,
                    "coverage_status": audit.get("status"),
                    "has_persistent_visible_metric_mismatch": persistent > metric,
                    "has_intermittent_visible_metric_mismatch": maximum > metric,
                })

    sequences = []
    for sequence in sorted(set(video_views) | set(people)):
        person_ids = sorted(people.get(sequence, set()), key=int)
        sequences.append({
            "sequence": sequence,
            "metric_person_ids": person_ids,
            "metric_person_count": len(person_ids),
            "available_video_views": video_views.get(sequence, 0),
            "metric_annotation_files": sorted(files.get(sequence, [])),
            "eligible_for_two_person_metric_qa": len(person_ids) >= 2,
            "eligible_for_three_plus_metric_qa": len(person_ids) >= 3,
        })

    report = {
        "status": "ok",
        "root": str(args.root),
        "sequence_count": len(sequences),
        "two_person_metric_sequence_count": sum(row["eligible_for_two_person_metric_qa"] for row in sequences),
        "three_plus_metric_sequence_count": sum(row["eligible_for_three_plus_metric_qa"] for row in sequences),
        "policy": {
            "two_person_metric_qa": "requires at least two per-person metric annotation files",
            "three_plus_metric_qa": "requires at least three per-person metric annotation files; 2D detections never fill missing metric identities",
            "intermittent_unannotated_people": "pairwise metric questions must explicitly name the annotated pair",
            "three_person_2d_topology": "allowed only as an explicitly labeled 2D topology scope",
        },
        "dataset_decision": {
            "local_hoim3": "use for audited two-person metric QA and explicitly scoped three-person 2D topology only",
            "cmu_panoptic": {
                "role": "primary fallback for true three-plus-person metric 3D QA",
                "official_source": "https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox",
                "adapter": "scripts/convert_panoptic_multihuman.py",
            },
            "egohumans": {
                "role": "secondary egocentric multi-human complement when ego-view coverage is required",
                "official_source": "https://github.com/rawalkhirodkar/egohumans",
            },
        },
        "sequences": sequences,
        "release_scene_coverage": release_scenes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("sequence_count", "two_person_metric_sequence_count", "three_plus_metric_sequence_count")}, indent=2))


if __name__ == "__main__":
    main()
