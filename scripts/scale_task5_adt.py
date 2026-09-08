#!/usr/bin/env python3
"""End-to-end, fail-closed Task 5 scaling pipeline for ADT sequences."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from limo4si.task5_scaling import (  # noqa: E402
    Task5CandidatePolicy,
    generate_task5_candidates,
    select_balanced_candidates,
)
from mine_task5_adt import mine  # noqa: E402

REQUIRED_SEQUENCE_FILES = {
    "video.vrs", "eyegaze.csv", "aria_trajectory.csv", "scene_objects.csv",
    "3d_bounding_box.csv", "2d_bounding_box.csv", "instances.json",
}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def discover_sequences(dataset_root: Path) -> list[Path]:
    candidates = {path.parent for path in dataset_root.rglob("eyegaze.csv")}
    return sorted(path for path in candidates if REQUIRED_SEQUENCE_FILES.issubset({item.name for item in path.iterdir()}))


def run_script(name: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src") + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    subprocess.run([sys.executable, str(ROOT / "scripts" / name), *arguments], cwd=ROOT, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Directory containing one or more unpacked ADT sequences")
    parser.add_argument("--target-per-category", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument("--max-cases-per-sequence", type=int, default=0)
    parser.add_argument("--minimum-gaze-run", type=int, default=4)
    parser.add_argument("--maximum-hit-distance-m", type=float, default=8.0)
    parser.add_argument("--maximum-wearer-skew-ms", type=float, default=10.0)
    parser.add_argument("--maximum-object-pose-skew-ms", type=float, default=50.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qa/task5_scale"))
    parser.add_argument("--site-media-dir", type=Path, default=Path("site/qa_benchmark/task5_media"))
    parser.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument("--site-index", type=Path, default=Path("site/qa_benchmark/index.html"))
    parser.add_argument("--reuse-analysis", action="store_true")
    parser.add_argument("--reuse-media", action="store_true")
    parser.add_argument("--plan-only", action="store_true", help="Mine/select/write config, but do not export media or update the site")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_root = resolve(args.output_root)
    analysis_dir = output_root / "analysis"
    selection_dir = output_root / "selection"
    media_dir = resolve(args.site_media_dir)
    for directory in (analysis_dir, selection_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sequences = discover_sequences(dataset_root)
    if args.max_sequences > 0:
        sequences = sequences[:args.max_sequences]
    if not sequences:
        raise SystemExit(f"no complete ADT sequences found below {dataset_root}")
    try:
        import projectaria_tools  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "projectaria-tools is required for mining. Run this pipeline with the project virtualenv, "
            "for example: .venv/bin/python scripts/scale_task5_adt.py <ADT_ROOT>"
        ) from exc
    if not args.plan_only and shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required for Task 5 media export")

    all_candidates: list[dict[str, Any]] = []
    sequence_records: list[dict[str, Any]] = []
    sequence_sources: dict[str, Path] = {}
    analysis_paths: dict[str, Path] = {}
    for sequence in sequences:
        sequence_name = sequence.relative_to(dataset_root).as_posix()
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sequence_name)[:80]
        digest = hashlib.sha1(sequence_name.encode()).hexdigest()[:8]
        analysis_path = analysis_dir / f"{slug}_{digest}.json"
        record: dict[str, Any] = {"sequence_name": sequence_name, "sequence_path": str(sequence), "analysis": portable(analysis_path)}
        try:
            use_cache = False
            if args.reuse_analysis and analysis_path.is_file():
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                use_cache = int(analysis.get("schema_version") or 0) >= 4 and analysis.get("sequence_name") == sequence_name
            if use_cache:
                record["analysis_status"] = "reused"
            else:
                analysis = mine(
                    sequence, args.minimum_gaze_run, args.maximum_hit_distance_m,
                    args.maximum_wearer_skew_ms, args.maximum_object_pose_skew_ms,
                )
                analysis["sequence_directory_name"] = sequence.name
                analysis["sequence_name"] = sequence_name
                analysis_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                record["analysis_status"] = "mined" if not args.reuse_analysis else "remined_stale_cache"
            candidates, diagnostics = generate_task5_candidates(analysis, Task5CandidatePolicy())
            for candidate in candidates:
                candidate["analysis"] = portable(analysis_path)
                candidate["sequence_path"] = str(sequence)
            all_candidates.extend(candidates)
            record["candidate_diagnostics"] = diagnostics
            sequence_sources[sequence_name] = sequence
            analysis_paths[sequence_name] = analysis_path
            record["status"] = "ok"
        except Exception as exc:  # Fail one bad sequence without losing the whole batch audit.
            record.update({
                "status": "rejected_sequence", "error_type": type(exc).__name__,
                "error": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-5:],
            })
        sequence_records.append(record)

    selected, selection = select_balanced_candidates(
        all_candidates, args.target_per_category, args.max_cases_per_sequence,
    )
    report: dict[str, Any] = {
        "pipeline": "annotation-only ADT Task 5 scale pipeline",
        "dataset_root": str(dataset_root),
        "sequence_count": len(sequences),
        "accepted_sequence_count": sum(row["status"] == "ok" for row in sequence_records),
        "rejected_sequence_count": sum(row["status"] != "ok" for row in sequence_records),
        "candidate_count": len(all_candidates),
        "candidate_counts": dict(Counter(row["category"] for row in all_candidates)),
        "selection": selection,
        "sequences": sequence_records,
    }
    report_path = output_root / "pipeline_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if selection["status"] != "ok":
        raise SystemExit(
            f"Task 5 selection is incomplete: {selection['deficits']}; inspect {report_path}. "
            "Nothing was published."
        )

    # Export one browser video per selected source sequence, then attach per-case sources.
    selected_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in selected:
        selected_by_sequence[str(candidate["sequence_name"])].append(candidate)
    for sequence_name, cases in selected_by_sequence.items():
        sequence = sequence_sources[sequence_name]
        media_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sequence_name)[:96]
        source_media = media_dir / f"{media_slug}_rgb.mp4"
        for case in cases:
            case["media_source"] = portable(source_media)
        if args.plan_only:
            continue
        media_dir.mkdir(parents=True, exist_ok=True)
        if not (args.reuse_media and source_media.is_file()):
            run_script("export_task5_adt_video.py", str(sequence / "video.vrs"), str(source_media))
        evidence_config = selection_dir / f"{media_slug}_evidence_cases.json"
        evidence_config.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        run_script(
            "export_task5_adt_evidence.py", str(sequence), "--config", str(evidence_config),
            "--analysis", str(analysis_paths[sequence_name]), "--output-dir", str(media_dir),
        )

    release_config = output_root / "generated_release_cases.json"
    release_config.write_text(json.dumps({
        "schema_version": 1,
        "generated_by": "scripts/scale_task5_adt.py",
        "minimum_examples_per_category": args.target_per_category,
        "selection_summary": selection,
        "cases": selected,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["release_config"] = portable(release_config)
    report["selected_case_ids"] = [row["id"] for row in selected]
    if args.plan_only:
        report["status"] = "planned"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "planned", "report": str(report_path), "selection": selection}, indent=2))
        return

    try:
        media_url_prefix = os.path.relpath(media_dir, resolve(args.site_data).parent).replace(os.sep, "/")
        if not media_url_prefix.startswith("."):
            media_url_prefix = "./" + media_url_prefix
        task5_jsonl = output_root / "task5_scaled_qa.jsonl"
        task5_audit = output_root / "task5_scale_audit.json"
        task5_quality = output_root / "task5_scale_quality.json"
        run_script(
            "build_task5_scaled.py", "--config", str(release_config),
            "--site-data", str(resolve(args.site_data)),
            "--media-output-dir", str(media_dir), "--media-url-prefix", media_url_prefix,
            "--output-jsonl", str(task5_jsonl), "--audit-output", str(task5_audit),
            "--quality-output", str(task5_quality),
        )
        report["task5_outputs"] = {
            "qa_jsonl": portable(task5_jsonl), "audit": portable(task5_audit),
            "quality": portable(task5_quality),
        }
        run_script(
            "build_static_qa_site.py", "--data-js", str(resolve(args.site_data)),
            "--output", str(resolve(args.site_index)),
        )
        combined_quality = output_root / "combined_scale_quality.json"
        run_script(
            "validate_task1_task4_scale.py", str(resolve(args.site_data)),
            "--output", str(combined_quality),
        )
        quality = json.loads(combined_quality.read_text(encoding="utf-8"))
        report.update({
            "status": "ok" if quality.get("status") == "ok" else "failed",
            "combined_quality": portable(combined_quality),
            "combined_quality_summary": {
                key: quality.get(key) for key in ("status", "case_count", "accepted_count", "rejected_count", "warning_count")
            },
        })
    except Exception as exc:
        report.update({"status": "publication_failed", "publication_error": f"{type(exc).__name__}: {exc}"})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        raise
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "ok":
        raise SystemExit(f"combined release gate failed; inspect {combined_quality}")
    print(json.dumps({"status": "ok", "report": str(report_path), "selection": selection}, indent=2))


if __name__ == "__main__":
    main()
