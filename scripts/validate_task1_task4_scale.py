#!/usr/bin/env python3
"""Validate a Task 1/4 release and emit per-case accept/reject reasons."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.scale_quality import ScaleQualityPolicy, validate_release  # noqa: E402


def load_release(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    return json.loads(match.group(1) if match else text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="QA_DATA JavaScript or JSON release")
    parser.add_argument("--policy", type=Path, help="JSON overrides for ScaleQualityPolicy")
    parser.add_argument("--output", type=Path, help="Write the full per-case report")
    args = parser.parse_args()
    overrides = json.loads(args.policy.read_text(encoding="utf-8")) if args.policy else {}
    report = validate_release(load_release(args.input), ScaleQualityPolicy(**overrides))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "case_count", "accepted_count", "rejected_count", "warning_count")}, ensure_ascii=False, indent=2))
    for case in report["cases"]:
        if case["status"] == "rejected":
            print(f"REJECT {case['case_id']}: {'; '.join(case['errors'])}")
    raise SystemExit(0 if report["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
