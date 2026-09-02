#!/usr/bin/env python3
"""One-command HOI-M3 -> Humans in Space Task4 pipeline.

Usage after preparing HOI-M3 with the official toolbox/export:

  python scripts/build_hoim3_multihuman_qa.py --hoi-m3-root data/HOI-M3

If the dataset is not present, the script writes an explicit missing_dataset
status instead of generating fake HOI-M3 answers.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print(' '.join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_clips(meta: dict) -> None:
    """Cut clips from the same source-frame interval used by the 3D timeline."""
    for scene in meta.get("scenes", []):
        source = ROOT / scene["source_video"]
        rel = str(scene.get("video_clip") or "").removeprefix("./")
        if not source.exists() or not rel:
            continue
        output = ROOT / "site/qa_benchmark" / rel
        output.parent.mkdir(parents=True, exist_ok=True)
        run(["ffmpeg", "-y", "-v", "error", "-ss", f"{float(scene.get('start_sec', 0)):.6f}", "-i", str(source), "-t", f"{float(scene.get('duration_sec', 15)):.3f}", "-vf", "scale=720:-2", "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--hoi-m3-root', type=Path, default=Path('data/HOI-M3'))
    ap.add_argument('--converted-scenes', type=Path, default=Path('outputs/qa/hoim3_multihuman_scenes.json'))
    ap.add_argument('--output-jsonl', type=Path, default=Path('outputs/qa/hoim3_multihuman_qa.jsonl'))
    ap.add_argument('--site-data', type=Path, default=Path('site/qa_benchmark/data.js'))
    ap.add_argument('--exclude-sequences', default='bedroom_data04', help='Comma-separated HOI-M3 sequences to skip; default excludes bedroom_data04 because the visible video contains an untracked third person in the current local subset.')
    ap.add_argument('--windows-per-sequence', type=int, default=2, help='Keep a small verified set by default; exported clips currently exist for the first two windows.')
    ap.add_argument('--samples-per-window', type=int, default=16, help='Metric pose samples in each 15-second window.')
    args = ap.parse_args()

    run([sys.executable, 'scripts/convert_hoim3_to_multihuman.py', '--input', str(args.hoi_m3_root), '--output', str(args.converted_scenes), '--exclude-sequences', args.exclude_sequences, '--windows-per-sequence', str(args.windows_per_sequence), '--samples-per-window', str(args.samples_per_window)])
    converted = ROOT / args.converted_scenes if not args.converted_scenes.is_absolute() else args.converted_scenes
    meta = json.loads(converted.read_text())
    if meta.get('status') != 'ok' or not meta.get('scenes'):
        print('HOI-M3 scenes not available; no real HOI-M3 QA generated.')
        print(converted)
        return
    build_clips(meta)
    run([sys.executable, 'scripts/build_multihuman_dynamic_qa.py', '--scenes-json', str(args.converted_scenes), '--source-label', 'HOI-M3 converted multihuman trajectory', '--replace-prefix', 'hoi_m3_', '--output-jsonl', str(args.output_jsonl), '--site-data', str(args.site_data)])
    run([sys.executable, 'scripts/calibrate_multihuman_video_evidence.py', '--site-data', str(args.site_data)])
    run([sys.executable, 'scripts/apply_precision_gate.py', '--input', str(args.site_data), '--output', str(args.site_data)])
    run([sys.executable, 'scripts/refine_humans_in_space_site_data.py', '--input', str(args.site_data), '--output', str(args.site_data)])
    run([sys.executable, 'scripts/build_static_qa_site.py', '--data-js', str(args.site_data)])


if __name__ == '__main__':
    main()
