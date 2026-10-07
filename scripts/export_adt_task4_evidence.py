#!/usr/bin/env python3
"""Export the exact selected 15-second ADT Task 4 video windows."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import re
import subprocess
from pathlib import Path
from typing import Any


def load_site(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    if not match:
        raise ValueError(f"cannot parse QA site data: {path}")
    return json.loads(match.group(1))


def save_site(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        "window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def duration(path: Path) -> float:
    value = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path),
    ], check=True, capture_output=True, text=True).stdout.strip()
    return float(value)


def export(group: dict[str, Any], dataset_root: Path, output_dir: Path) -> dict[str, Any]:
    window = group.get("video_window") or {}
    source = str(window.get("source") or "")
    match = re.search(r"(?:^|/)adt/([^/]+)/(?:video\.vrs|preview_rgb\.mp4)$", source)
    if not match:
        raise ValueError(f"{group.get('name')}: unrecognized ADT source {source}")
    sequence = match.group(1)
    preview = dataset_root / sequence / "preview_rgb.mp4"
    if not preview.is_file():
        raise FileNotFoundError(preview)
    start = float(window.get("start_sec") or 0.0)
    requested_duration = float(window.get("duration_sec") or 0.0)
    if not 14.5 <= requested_duration <= 15.5:
        raise ValueError(f"{group.get('name')}: invalid public duration {requested_duration}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{group['name']}_15s.mp4"
    if not output.is_file() or not 14.5 <= duration(output) <= 15.5:
        subprocess.run([
            "ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.9f}",
            "-i", str(preview), "-t", f"{requested_duration:.9f}", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-movflags", "+faststart", str(output),
        ], check=True)
    actual_duration = duration(output)
    if not 14.5 <= actual_duration <= 15.5:
        raise ValueError(f"{group.get('name')}: encoded duration {actual_duration} is outside policy")
    group["video_clip"] = str(output)
    return {
        "case_id": group["name"], "sequence": sequence, "source": str(preview),
        "start_sec": start, "duration_sec": actual_duration, "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-data", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/adt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    data = load_site(args.site_data)
    groups = [
        group for group in data.get("groups") or []
        if group.get("dataset") == "Aria Digital Twin"
    ]
    if not groups:
        raise ValueError("release contains no selected ADT groups")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(
            lambda group: export(group, args.dataset_root, args.output_dir), groups,
        ))
    save_site(args.site_data, data)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps({
        "status": "ok", "case_count": len(rows), "cases": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "case_count": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
