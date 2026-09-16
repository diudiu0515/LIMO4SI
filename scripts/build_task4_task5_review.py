#!/usr/bin/env python3
"""Build a fail-closed final-review page containing every released Task 4/5 case."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_static_qa_site import build_static_html, load_site_data

ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT / "site" / "qa_benchmark"

from limo4si.scale_quality import TASK4_ID, TASK5_ID, validate_release

REVIEW_TASK_IDS = {TASK4_ID, TASK5_ID}
MEDIA_FIELDS = (
    "video_clip",
    "localization_video",
    "original_video",
    "original_image",
    "localization_image",
    "topdown_image",
)


def select_review_data(data: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select Task 4/5 without changing any question, option, answer, or evidence."""
    review = copy.deepcopy(data)
    groups = []
    for group in review.get("groups", []):
        questions = group.get("qa") or []
        selected = [question for question in questions if question.get("task_id") in REVIEW_TASK_IDS]
        if selected:
            if len(questions) != 1 or len(selected) != 1:
                raise ValueError(f"review case {group.get('name')} is not one-question atomic")
            groups.append(group)
    review["groups"] = groups
    review["tasks"] = [
        task for task in review.get("tasks", []) if task.get("id") in REVIEW_TASK_IDS
    ]
    review["title"] = "Task 4 + Task 5 · Final Pre-Scale Review"
    review["subtitle"] = (
        "Every currently released Task 4 and Task 5 case, with signed deterministic evidence."
    )
    review["release_policy"] = {
        **dict(review.get("release_policy") or {}),
        "task_scope": [TASK4_ID, TASK5_ID],
        "review_projection_only": True,
        "semantic_mutation": False,
    }

    counts = Counter(
        question["task_id"] for group in groups for question in group.get("qa", [])
    )
    source_counts = Counter(
        question["task_id"]
        for group in data.get("groups", [])
        for question in (group.get("qa") or [])
        if question.get("task_id") in REVIEW_TASK_IDS
    )
    if not all(source_counts.get(task_id, 0) > 0 for task_id in REVIEW_TASK_IDS):
        raise ValueError(f"source release is missing Task 4 or Task 5: {dict(source_counts)}")
    if counts != source_counts:
        raise ValueError(
            f"Task 4/5 review omitted source cases: {dict(counts)} != {dict(source_counts)}"
        )

    quality = validate_release({"groups": groups})
    if quality["status"] != "ok":
        rejected = [
            f"{row['case_id']}: {', '.join(row['errors'])}"
            for row in quality["cases"] if row["status"] == "rejected"
        ]
        raise ValueError("Task 4/5 review quality gate failed: " + "; ".join(rejected))

    windows = {
        (
            (group.get("video_window") or {}).get("source_sequence")
            or (group.get("video_window") or {}).get("source_video")
            or group["name"],
            (group.get("video_window") or {}).get("start_sec"),
            (group.get("video_window") or {}).get("duration_sec"),
        )
        for group in groups
    }
    if len(windows) != len(groups):
        raise ValueError("Task 4/5 review contains duplicate case windows")

    missing_media = []
    checked_media = []
    for group in groups:
        for field in MEDIA_FIELDS:
            value = group.get(field)
            if not isinstance(value, str) or not value.startswith("./"):
                continue
            path = SITE_ROOT / value[2:]
            checked_media.append(str(path.relative_to(ROOT)))
            if not path.is_file():
                missing_media.append(str(path.relative_to(ROOT)))
    if missing_media:
        raise ValueError("Task 4/5 review media is missing: " + ", ".join(missing_media))

    signed = sum(
        bool(question.get("answer_signature"))
        for group in groups for question in group.get("qa", [])
    )
    audit = {
        "status": "ok",
        "case_count": len(groups),
        "qa_count": sum(counts.values()),
        "task_counts": dict(counts),
        "unique_case_windows": len(windows),
        "signed_question_count": signed,
        "media_reference_count": len(checked_media),
        "quality_warning_count": quality["warning_count"],
        "quality": quality,
    }
    return review, audit


def review_html(data: dict[str, Any], audit: dict[str, Any]) -> str:
    output = build_static_html(data)
    output = output.replace(
        "<title>Task 4 + Task 5 Spatial QA</title>",
        "<title>Task 4 + Task 5 Final Pre-Scale Review</title>",
        1,
    )
    output = output.replace(
        "<h1>Task 4 + Task 5 · Spatial QA</h1>",
        "<h1>Task 4 + Task 5 · Final Pre-Scale Review</h1>",
        1,
    )
    output = output.replace(
        "This site contains only Task 4 multi-human reasoning and Task 5 gaze-grounded reasoning. Every answer is computed and signed before any optional language API call. Submit an answer first; localization, gaze/mask, and computed evidence remain collapsed until requested.",
        (
            f"All {audit['case_count']} current release cases are shown here: "
            f"{audit['task_counts'][TASK4_ID]} Task 4 and "
            f"{audit['task_counts'][TASK5_ID]} Task 5. "
            "Choose and submit an answer before opening localization, gaze/mask, "
            "computed evidence, and semantic signatures."
        ),
        1,
    )
    for index, group in enumerate(data["groups"], 1):
        task = "Task 4" if group["qa"][0]["task_id"] == TASK4_ID else "Task 5"
        output = output.replace(
            f'<a href="#case-{index}">Case {index}</a>',
            f'<a href="#case-{index}">{task} · {index}</a>',
            1,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-js", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument(
        "--output", type=Path, default=Path("site/qa_benchmark/task4_task5_review.html")
    )
    parser.add_argument(
        "--audit-output", type=Path, default=Path("outputs/qa/task4_task5_review_audit.json")
    )
    args = parser.parse_args()

    data_path = args.data_js if args.data_js.is_absolute() else ROOT / args.data_js
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    audit_path = (
        args.audit_output
        if args.audit_output.is_absolute()
        else ROOT / args.audit_output
    )
    review, audit = select_review_data(load_site_data(data_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(review_html(review, audit), encoding="utf-8")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "ok",
        "output": str(output_path),
        "audit": str(audit_path),
        "task_counts": audit["task_counts"],
        "media_reference_count": audit["media_reference_count"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
