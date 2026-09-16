#!/usr/bin/env python3
"""Project a mixed intermediate into the public Task 4 + Task 5 release."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.scale_quality import (  # noqa: E402
    TASK4_ID,
    TASK5_ID,
    validate_release,
)

PUBLIC_TASK_IDS = {TASK4_ID, TASK5_ID}


def load_data(path: Path) -> dict[str, Any]:
    """Load QA_DATA JavaScript or plain JSON."""

    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"window\.QA_DATA\s*=\s*(.*);\s*$",
        text,
        re.S,
    )
    return json.loads(match.group(1) if match else text)


def save_data(
    path: Path,
    data: dict[str, Any],
) -> None:
    """Write the browser QA_DATA assignment."""

    path.write_text(
        "window.QA_DATA = "
        + json.dumps(data, ensure_ascii=False, indent=2)
        + ";\n",
        encoding="utf-8",
    )


def filter_public_data(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Remove every non-public task without mutating question semantics."""

    result = dict(data)
    groups = []
    for group in data.get("groups", []):
        questions = group.get("qa") or []
        if len(questions) != 1:
            raise ValueError(
                f"case {group.get('name')} is not "
                "one-question atomic"
            )
        if questions[0].get("task_id") in PUBLIC_TASK_IDS:
            groups.append(group)

    result.update(
        {
            "title": "Task 4 + Task 5 Spatial QA",
            "subtitle": (
                "Deterministic multi-human and "
                "gaze-grounded spatial reasoning."
            ),
            "tasks": [
                task
                for task in data.get("tasks", [])
                if task.get("id") in PUBLIC_TASK_IDS
            ],
            "groups": groups,
            "release_policy": {
                **dict(data.get("release_policy") or {}),
                "task_scope": [TASK4_ID, TASK5_ID],
                "public_tasks_only": True,
            },
        }
    )
    return result


def project(
    data: dict[str, Any],
    *,
    require_both: bool = False,
) -> dict[str, Any]:
    """Filter and independently validate the public projection."""

    result = filter_public_data(data)
    groups = result["groups"]
    present = {
        group["qa"][0]["task_id"]
        for group in groups
    }
    if require_both and present != PUBLIC_TASK_IDS:
        raise ValueError(
            "public release must contain exactly "
            f"Task 4 and Task 5, got {sorted(present)}"
        )

    quality = validate_release({"groups": groups})
    if quality["status"] != "ok":
        rejected = [
            f"{row['case_id']}: {', '.join(row['errors'])}"
            for row in quality["cases"]
            if row["status"] == "rejected"
        ]
        raise ValueError(
            "Task4/5 projection failed quality gate: "
            + "; ".join(rejected)
        )
    return result


def task_counts(
    data: dict[str, Any],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for group in data["groups"]:
        task_id = group["qa"][0]["task_id"]
        counts[task_id] = counts.get(task_id, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-js",
        type=Path,
        default=Path("site/qa_benchmark/data.js"),
    )
    parser.add_argument("--require-both", action="store_true")
    args = parser.parse_args()

    path = (
        args.data_js
        if args.data_js.is_absolute()
        else ROOT / args.data_js
    )
    projected = project(
        load_data(path),
        require_both=args.require_both,
    )
    save_data(path, projected)
    print(
        json.dumps(
            {
                "status": "ok",
                "case_count": len(projected["groups"]),
                "task_counts": task_counts(projected),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
