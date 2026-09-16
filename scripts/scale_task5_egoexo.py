#!/usr/bin/env python3
"""Mine and build deterministic 15-second EgoExo4D Task 5 release cases."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.task5_egoexo import (  # noqa: E402
    RELEASE_MAX_WINDOW_SEC,
    RELEASE_MIN_WINDOW_SEC,
    display_object_name,
    gaze_row_for_video_frame,
    unique_encoded_mask_hit,
)

EXCLUDED_TARGET_WORDS = {
    "table", "counter", "countertop", "floor", "wall", "ceiling", "cabinet",
    "shelf", "stove", "sink", "oven", "fridge", "refrigerator", "background",
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def load_gaze(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or [int(row["frame_num"]) for row in rows] != list(range(len(rows))):
        raise ValueError(f"non-contiguous or empty gaze stream: {path}")
    return rows


def target_allowed(object_id: str) -> bool:
    words = set(display_object_name(object_id).replace("-", " ").split())
    return not bool(words & EXCLUDED_TARGET_WORDS)


def camera_frame_records(entry: dict[str, Any], camera: str) -> dict[int, dict[str, dict[str, Any]]]:
    output: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for object_id, tracks in entry["object_masks"].items():
        for frame, record in tracks.get(camera, {}).get("annotation", {}).items():
            output[int(frame)][str(object_id)] = record
    return dict(output)


def discover_local_takes(
    dataset_root: Path, relations: dict[str, Any], takes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    take_by_uid = {str(row["take_uid"]): row for row in takes}
    discovered = []
    for uid, entry in relations.items():
        take = take_by_uid.get(str(uid))
        if not take:
            continue
        name = str(take["take_name"])
        take_root = dataset_root / "takes" / name
        gaze = take_root / "eye_gaze" / "general_eye_gaze_2d.csv"
        video_root = take_root / "frame_aligned_videos" / "downscaled" / "448"
        cameras = sorted({
            camera
            for tracks in entry["object_masks"].values()
            for camera in tracks
            if camera.startswith("aria") and (video_root / f"{camera}.mp4").is_file()
        })
        if gaze.is_file() and len(cameras) == 1:
            discovered.append({
                "take_uid": str(uid), "take_name": name,
                "camera": cameras[0], "gaze": gaze,
            })
    return sorted(discovered, key=lambda row: row["take_name"])


def mine_hits(
    take: dict[str, Any], entry: dict[str, Any], minimum_boundary_margin_px: float,
) -> tuple[list[dict[str, Any]], Counter]:
    rows = load_gaze(take["gaze"])
    frames = camera_frame_records(entry, take["camera"])
    hits: list[dict[str, Any]] = []
    rejects: Counter = Counter()
    for frame, records in sorted(frames.items()):
        try:
            gaze, skew_ms = gaze_row_for_video_frame(rows, frame)
            hit = unique_encoded_mask_hit(
                records, float(gaze["x"]), float(gaze["y"]),
                minimum_boundary_margin_px=minimum_boundary_margin_px,
            )
            hits.append({"frame": frame, "alignment_skew_ms": round(skew_ms, 6), **hit})
        except ValueError as exc:
            message = str(exc)
            if "intersects 0" in message:
                rejects["no_mask_hit"] += 1
            elif "intersects" in message:
                rejects["ambiguous_mask_hit"] += 1
            elif "margin" in message:
                rejects["insufficient_margin"] += 1
            else:
                rejects["invalid_alignment_or_mask"] += 1
    return hits, rejects


def case_candidates(
    take: dict[str, Any], hits: list[dict[str, Any]], *,
    minimum_anchor_gap_frames: int,
    minimum_anchor_span_frames: int,
    maximum_anchor_span_frames: int,
) -> list[dict[str, Any]]:
    candidates = []
    for first_index, first in enumerate(hits):
        for second_index in range(first_index + 1, len(hits)):
            second = hits[second_index]
            if second["frame"] - first["frame"] < minimum_anchor_gap_frames:
                continue
            if second["frame"] - first["frame"] > maximum_anchor_span_frames:
                break
            for third in hits[second_index + 1:]:
                anchor_span = third["frame"] - first["frame"]
                if anchor_span > maximum_anchor_span_frames:
                    break
                if (
                    anchor_span < minimum_anchor_span_frames
                    or third["frame"] - second["frame"] < minimum_anchor_gap_frames
                ):
                    continue
                anchors = [first, second, third]
                object_ids = [str(row["object_id"]) for row in anchors]
                names = [display_object_name(value) for value in object_ids]
                if len(set(object_ids)) != 3 or len(set(names)) != 3:
                    continue
                for target_index, target in enumerate(anchors):
                    if not target_allowed(str(target["object_id"])):
                        continue
                    identity = "|".join([
                        str(take["take_uid"]), *(str(row["frame"]) for row in anchors),
                        str(target["object_id"]),
                    ])
                    digest = hashlib.sha1(identity.encode()).hexdigest()[:10]
                    candidates.append({
                        "id": f"task5_egoexo_{take['take_name']}_{digest}",
                        "take_uid": take["take_uid"],
                        "take_name": take["take_name"],
                        "anchor_frames": [int(row["frame"]) for row in anchors],
                        "target_object_id": str(target["object_id"]),
                        "target_anchor_index": target_index,
                        "target_object_name": display_object_name(str(target["object_id"])),
                        "score": round(
                            min(float(row["boundary_margin_px"]) for row in anchors)
                            + float(target["boundary_margin_px"]), 6,
                        ),
                        "anchor_objects": names,
                    })
                break
    return candidates


def select_balanced(
    candidates: list[dict[str, Any]], target_count: int, max_cases_per_take: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    take_counts: Counter = Counter()
    anchor_counts: Counter = Counter()
    target_names: set[str] = set()
    remaining = sorted(candidates, key=lambda row: (-float(row["score"]), row["id"]))
    while len(selected) < target_count:
        minimum_anchor_count = min(anchor_counts[index] for index in range(3))
        eligible = []
        for row in remaining:
            take_name = str(row["take_name"])
            target_name = str(row["target_object_name"])
            target_anchor = int(row["target_anchor_index"])
            if take_counts[take_name] >= max_cases_per_take or target_name in target_names:
                continue
            if anchor_counts[target_anchor] != minimum_anchor_count:
                continue
            if any(
                take_name == item["take_name"]
                and set(row["anchor_frames"]) & set(item["anchor_frames"])
                for item in selected
            ):
                continue
            eligible.append(row)
        if not eligible:
            break
        row = eligible[0]
        selected.append(row)
        take_counts[str(row["take_name"])] += 1
        anchor_counts[int(row["target_anchor_index"])] += 1
        target_names.add(str(row["target_object_name"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--target-count", type=int, default=12)
    parser.add_argument("--max-takes", type=int, default=0)
    parser.add_argument("--max-cases-per-take", type=int, default=1)
    parser.add_argument("--minimum-boundary-margin-px", type=float, default=10.0)
    parser.add_argument("--minimum-anchor-gap-sec", type=float, default=2.0)
    parser.add_argument("--minimum-anchor-span-sec", type=float, default=10.0)
    parser.add_argument("--maximum-anchor-span-sec", type=float, default=14.0)
    parser.add_argument("--window-duration-sec", type=float, default=15.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qa/task5_egoexo_scale"))
    parser.add_argument("--site-dir", type=Path, default=Path("site/qa_benchmark"))
    parser.add_argument(
        "--language-client-factory",
        help="Optional module:function language-only client factory.",
    )
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.target_count <= 0 or args.max_cases_per_take <= 0:
        raise SystemExit("target count and max cases per take must be positive")
    if not (
        0 < args.minimum_anchor_gap_sec
        and 2 * args.minimum_anchor_gap_sec <= args.minimum_anchor_span_sec
        <= args.maximum_anchor_span_sec <= args.window_duration_sec
    ):
        raise SystemExit("anchor/window timing policy is invalid")
    if not RELEASE_MIN_WINDOW_SEC <= args.window_duration_sec <= RELEASE_MAX_WINDOW_SEC:
        raise SystemExit(
            f"release window must be {RELEASE_MIN_WINDOW_SEC:g}–{RELEASE_MAX_WINDOW_SEC:g} seconds"
        )

    dataset_root = resolve(args.dataset_root)
    output_root = resolve(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    takes = json.loads((dataset_root / "takes.json").read_text(encoding="utf-8"))
    relations = json.loads(
        (dataset_root / "annotations" / "relations_val.json").read_text(encoding="utf-8")
    )["annotations"]
    discovered = discover_local_takes(dataset_root, relations, takes)
    if args.max_takes:
        discovered = discovered[:args.max_takes]
    if not discovered:
        raise SystemExit("no local take has Relations, 2D gaze, and one frame-aligned Aria RGB video")

    all_candidates: list[dict[str, Any]] = []
    take_reports = []
    for take in discovered:
        hits, rejects = mine_hits(
            take, relations[take["take_uid"]], args.minimum_boundary_margin_px,
        )
        candidates = case_candidates(
            take, hits,
            minimum_anchor_gap_frames=round(args.minimum_anchor_gap_sec * 30),
            minimum_anchor_span_frames=round(args.minimum_anchor_span_sec * 30),
            maximum_anchor_span_frames=round(args.maximum_anchor_span_sec * 30),
        )
        all_candidates.extend(candidates)
        take_reports.append({
            "take_uid": take["take_uid"], "take_name": take["take_name"],
            "camera": take["camera"], "accepted_anchor_hits": len(hits),
            "candidate_count": len(candidates), "rejections": dict(rejects),
        })
        print(
            f"{take['take_name']}: {len(hits)} anchor hits, {len(candidates)} candidates",
            flush=True,
        )

    selected = select_balanced(all_candidates, args.target_count, args.max_cases_per_take)
    report = {
        "status": "ok" if len(selected) == args.target_count else "insufficient_candidates",
        "pipeline": "deterministic Ego-Exo4D Relations-mask plus synchronized-2D-gaze mining",
        "dataset_root": portable(dataset_root),
        "discovered_take_count": len(discovered),
        "candidate_count": len(all_candidates),
        "selected_count": len(selected),
        "requested_count": args.target_count,
        "selection_constraints": {
            "max_cases_per_take": args.max_cases_per_take,
            "unique_target_display_names": True,
            "non_overlapping_anchors_within_take": True,
            "balanced_target_anchor_positions": True,
            "minimum_anchor_gap_sec": args.minimum_anchor_gap_sec,
            "minimum_anchor_span_sec": args.minimum_anchor_span_sec,
            "maximum_anchor_span_sec": args.maximum_anchor_span_sec,
            "window_duration_sec": args.window_duration_sec,
        },
        "take_reports": take_reports,
        "selected": selected,
    }
    report_path = output_root / "pipeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(selected) != args.target_count:
        raise SystemExit(f"only selected {len(selected)}/{args.target_count}; inspect {report_path}")

    config_path = output_root / "generated_release_cases.json"
    config_path.write_text(
        json.dumps({
            "schema_version": 1,
            "generated_by": "scripts/scale_task5_egoexo.py",
            "release_status": "signed_semantic_gt_egoexo_primary",
            "minimum_boundary_margin_px": args.minimum_boundary_margin_px,
            "window_duration_sec": args.window_duration_sec,
            "cases": [
                {key: row[key] for key in ("id", "take_uid", "anchor_frames", "target_object_id")}
                for row in selected
            ],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report["generated_config"] = portable(config_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print(json.dumps({
            "status": "planned", "report": str(report_path), "config": str(config_path),
        }, indent=2))
        return

    command = [
        sys.executable, str(ROOT / "scripts" / "build_task5_egoexo.py"),
        "--config", str(config_path),
        "--dataset-root", str(dataset_root),
        "--site-dir", str(resolve(args.site_dir)),
    ]
    if args.language_client_factory:
        command.extend(["--language-client-factory", args.language_client_factory])
    subprocess.run(command, cwd=ROOT, check=True)
    report["status"] = "built_release"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
