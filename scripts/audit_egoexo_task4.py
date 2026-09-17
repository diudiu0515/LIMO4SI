#!/usr/bin/env python3
"""Audit one local EgoExo4D capture for missing Task 4 capabilities."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.egoexo_task4 import audit_task4_evidence  # noqa: E402


def _trajectory_sessions(path: Path) -> list[str]:
    if not path.is_file():
        return []
    sessions: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get("session_uid")
            if value:
                sessions.add(value)
    return sorted(sessions)


def _stable_subject_ids(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    for candidates in rows.values():
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            value = candidate.get("person_id") or candidate.get("track_id") or candidate.get("subject_id")
            if value is not None:
                ids.add(str(value))
    return sorted(ids)


def _relations_contains(path: Path, take_name: str) -> bool:
    if not path.is_file():
        return False
    needle = json.dumps(take_name).encode("utf-8")
    tail = b""
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            data = tail + chunk
            if needle in data:
                return True
            tail = data[-len(needle):]
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_name")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    captures = json.loads((args.dataset_root / "captures.json").read_text(encoding="utf-8"))
    capture = next((row for row in captures if row.get("capture_name") == args.capture_name), None)
    if capture is None:
        raise SystemExit(f"unknown capture: {args.capture_name}")
    takes = json.loads((args.dataset_root / "takes.json").read_text(encoding="utf-8"))
    capture_takes = [row for row in takes if row.get("capture_uid") == capture.get("capture_uid")]

    body_counts: dict[str, int] = {}
    relations_takes: list[str] = []
    for take in capture_takes:
        uid, name = str(take["take_uid"]), str(take["take_name"])
        matches = list((args.dataset_root / "annotations" / "ego_pose").glob(f"*/body/annotation/{uid}.json"))
        if matches:
            body_counts[name] = len(_stable_subject_ids(matches[0]))
        if any(
            _relations_contains(args.dataset_root / "annotations" / f"relations_{split}.json", name)
            for split in ("train", "val", "test")
        ):
            relations_takes.append(name)

    trajectory_dir = args.dataset_root / str(capture["root_dir"]) / "trajectory"
    files = [path.name for path in trajectory_dir.glob("*") if path.is_file()]
    sessions = _trajectory_sessions(trajectory_dir / "open_loop_trajectory.csv")
    report = audit_task4_evidence(
        capture=capture,
        trajectory_session_uids=sessions,
        body_subject_counts=body_counts,
        relations_take_names=relations_takes,
        available_files=files,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
