#!/usr/bin/env python3
"""Build standard Task 1 / Task 3 QA JSONL records.

This is the benchmark-oriented generator. It is intentionally independent from
website rendering: it produces normalized JSONL records with evidence paths,
computed result JSON, confidence, approximations, and missing evidence.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_task1_task3_site_data import DEFAULT_SUMMARIES, build_group  # noqa: E402

LOWER_CONFIDENCE_TYPES = {
    "visibility",
    "current_interaction_object",
    "perspective_visibility_occlusion",
    "level2_perspective_taking",
}
MEDIUM_CONFIDENCE_TYPES = {
    "reachability",
    "perspective_reachable_nearest",
    "reference_frame_switching",
}
HIGH_CONFIDENCE_TYPES = {
    "quantitative_distance_and_direction",
    "nearest_referring_object",
    "person_perspective_left_right_front_back",
}


def strip_site_prefix(path: str | None) -> str | None:
    if not path:
        return None
    return path[2:] if path.startswith("./") else path


def collect_approximations(result: Any) -> list[str]:
    found: list[str] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            approx = value.get("approximations")
            if isinstance(approx, list):
                found.extend(str(x) for x in approx)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(result)
    return sorted(set(x for x in found if x))


def collect_missing(result: Any) -> list[str]:
    found: list[str] = []
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            missing = value.get("missing_evidence")
            if isinstance(missing, list):
                found.extend(str(x) for x in missing)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(result)
    return sorted(set(x for x in found if x))


def confidence_for(question_type: str, result: dict, approximations: list[str], missing: list[str]) -> str:
    if result.get("status") == "missing_evidence":
        return "reject"
    nonfatal_missing = {"semantic world_axes for room-level allocentric labels", "left_fingertips", "right_fingertips"}
    fatal_missing = [item for item in missing if item not in nonfatal_missing]
    if fatal_missing:
        return "reject"
    if question_type in HIGH_CONFIDENCE_TYPES and not approximations and not missing:
        return "high"
    if question_type in MEDIUM_CONFIDENCE_TYPES:
        return "medium"
    if question_type in LOWER_CONFIDENCE_TYPES:
        return "medium" if result.get("status") == "ok" else "low"
    return "medium"


def evidence_for(group: dict, qa: dict) -> dict:
    raw = qa.get("raw_json") or {}
    sample = raw.get("target") if isinstance(raw, dict) and isinstance(raw.get("target"), dict) else raw
    if not isinstance(sample, dict):
        sample = {}
    return {
        "take_uid": sample.get("take_uid") or (group.get("raw_summary", {}).get("samples") or [{}])[0].get("take_uid"),
        "take_name": sample.get("take_name") or (group.get("raw_summary", {}).get("samples") or [{}])[0].get("take_name"),
        "camera": sample.get("camera") or (group.get("raw_summary", {}).get("samples") or [{}])[0].get("camera"),
        "frame": sample.get("frame") or (group.get("raw_summary", {}).get("samples") or [{}])[0].get("frame"),
        "video_clip": strip_site_prefix(group.get("video_clip")),
        "video_window": group.get("video_window"),
        "image_with_skeleton_and_objects": strip_site_prefix(group.get("original_image")),
        "topdown_human_frame": strip_site_prefix(group.get("topdown_image")),
        "summary_json": group.get("summary_path"),
        "raw_object_json_available": bool(qa.get("raw_json")),
    }


def flatten_group(group: dict) -> list[dict]:
    records = []
    for idx, qa in enumerate(group["qa"], 1):
        result = qa.get("result_json") or {}
        approximations = collect_approximations(result)
        missing = collect_missing(result)
        confidence = confidence_for(qa["question_type"], result, approximations, missing)
        ev = evidence_for(group, qa)
        record = {
            "qa_uid": f"{group['name']}::{idx:02d}::{qa['question_type']}",
            "task_id": qa["task_id"],
            "task_name": qa["task_name"],
            "question_type": qa["question_type"],
            "case_id": group["name"],
            "case_title": group["title"],
            "question": qa["question"],
            "answer": qa["answer"],
            "method": qa["method"],
            "status": qa.get("status", result.get("status", "ok")),
            "confidence": confidence,
            "approximations": approximations,
            "missing_evidence": missing,
            "evidence": ev,
            "result_json": result,
            "raw_json": qa.get("raw_json"),
        }
        records.append(record)
    return records


def summarize(records: list[dict]) -> dict:
    by_type = Counter((r["task_id"], r["question_type"]) for r in records)
    by_conf = Counter(r["confidence"] for r in records)
    by_status = Counter(r["status"] for r in records)
    type_quality: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in records:
        key = f"{r['task_id']}::{r['question_type']}"
        type_quality[key][r["confidence"]] += 1
    return {
        "record_count": len(records),
        "task_counts": dict(Counter(r["task_id"] for r in records)),
        "question_type_counts": {f"{k[0]}::{k[1]}": v for k, v in sorted(by_type.items())},
        "confidence_counts": dict(by_conf),
        "status_counts": dict(by_status),
        "quality_by_type": {k: dict(v) for k, v in sorted(type_quality.items())},
        "note": "Website export is downstream; these JSONL records are the reusable benchmark artifact.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="*", default=DEFAULT_SUMMARIES)
    ap.add_argument("--output", type=Path, default=Path("outputs/qa/task1_task3_qa.jsonl"))
    ap.add_argument("--summary-output", type=Path, default=Path("outputs/qa/task1_task3_qa_summary.json"))
    ap.add_argument("--clip-seconds", type=float, default=3.0)
    args = ap.parse_args()

    records: list[dict] = []
    for summary in args.summaries:
        group = build_group(ROOT, ROOT / summary, args.clip_seconds)
        records.extend(flatten_group(group))

    out = ROOT / args.output if not args.output.is_absolute() else args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")

    summary = summarize(records)
    summary_out = ROOT / args.summary_output if not args.summary_output.is_absolute() else args.summary_output
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(out)
    print(summary_out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
