#!/usr/bin/env python3
"""End-to-end, fail-closed Task 5 scaling pipeline for ADT sequences."""
from __future__ import annotations

import argparse
import csv
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
from limo4si.adt_media import (  # noqa: E402
    index_gaze_rows_by_device_timestamp,
    pixel_inside_box,
    project_gaze_to_raw_rgb,
    rgb_window_alignment,
)
from mine_task5_adt import mine  # noqa: E402

REQUIRED_SEQUENCE_FILES = {
    "metadata.json", "eyegaze.csv", "aria_trajectory.csv", "scene_objects.csv",
    "3d_bounding_box.csv", "2d_bounding_box.csv", "instances.json",
}

def _evidence_frames(candidate: dict[str, Any], analysis: dict[str, Any]) -> list[int]:
    """Resolve every RGB panel frame required to audit one candidate."""
    uid = str(candidate["object_id"])
    question_type = candidate["question_type"]
    if question_type == "gaze_target_at_evidence_anchor":
        return [int(frame) for frame in candidate["anchor_frames"]]
    if question_type == "relation_change_between_gazes":
        events = [
            next(
                event for event in analysis["gaze_events"]
                if str(event["object_id"]) == uid and int(event["start_index"]) == int(start)
            )
            for start in candidate["event_start_frames"]
        ]
        return [(int(event["start_index"]) + int(event["end_index"])) // 2 for event in events]
    if question_type == "gaze_onset_side_change":
        event = next(
            event for event in analysis["gaze_events"]
            if str(event["object_id"]) == uid
            and int(event["start_index"]) == int(candidate["event_start_frames"][0])
        )
        return [int(candidate["pre_frame"]), (int(event["start_index"]) + int(event["end_index"])) // 2]
    if question_type == "last_gaze_annotated_object_relation_change":
        lo, hi = (int(value) for value in candidate["window_frames"])
        runs: list[list[Any]] = []
        for frame_index in range(lo, hi + 1):
            label = analysis["states"][frame_index]["object_relations"][uid]["label"]
            if not runs or runs[-1][0] != label:
                runs.append([label, frame_index, frame_index])
            else:
                runs[-1][2] = frame_index
        stable = [run for run in runs if int(run[2]) - int(run[1]) + 1 >= 6]
        collapsed: list[list[Any]] = []
        for label, start, end in stable:
            if collapsed and collapsed[-1][0] == label:
                collapsed[-1][2] = end
            else:
                collapsed.append([label, start, end])
        return [(int(start) + int(end)) // 2 for _, start, end in collapsed]
    raise ValueError(f"unsupported release question type: {question_type}")


def _filter_rgb_auditable_candidates(
    candidates: list[dict[str, Any]], analysis: dict[str, Any], sequence: Path,
    maximum_skew_ms: float = 50.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Reject candidates before selection unless every evidence panel has a fresh RGB box."""
    boxes: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    box_times: dict[str, list[int]] = defaultdict(list)
    with (sequence / "2d_bounding_box.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["stream_id"] == "214-1":
                uid, timestamp = str(row["object_uid"]), int(row["timestamp[ns]"])
                box_times[uid].append(timestamp)
                boxes[(uid, timestamp)] = tuple(
                    float(row[key]) for key in (
                        "x_min[pixel]", "y_min[pixel]", "x_max[pixel]", "y_max[pixel]",
                    )
                )
    for values in box_times.values():
        values.sort()
    with (sequence / "eyegaze.csv").open(newline="", encoding="utf-8") as handle:
        gaze_by_timestamp = index_gaze_rows_by_device_timestamp(csv.DictReader(handle))
    kept: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for candidate in candidates:
        uid = str(candidate["object_id"])
        times = box_times.get(uid)
        if not times:
            rejected["missing_rgb_2d_box"] += 1
            continue
        frames = _evidence_frames(candidate, analysis)
        if len(frames) < 2:
            rejected["insufficient_evidence_panels"] += 1
            continue
        skews_ms = [
            min(abs(timestamp - int(analysis["states"][frame]["timestamp_ns"])) for timestamp in times) / 1_000_000
            for frame in frames
        ]
        if any(skew > maximum_skew_ms for skew in skews_ms):
            rejected["stale_rgb_2d_box"] += 1
            continue
        if candidate["question_type"] == "relation_change_between_gazes":
            target_gaze_frames = frames
        elif candidate["question_type"] == "gaze_onset_side_change":
            target_gaze_frames = frames[1:]
        else:
            target_gaze_frames = []
        projection_failed = False
        for frame in target_gaze_frames:
            state = analysis["states"][frame]
            timestamp = int(state["timestamp_ns"])
            gaze = gaze_by_timestamp.get(timestamp)
            if gaze is None or str(state.get("gazed_object_id")) != uid:
                projection_failed = True
                break
            pixel = project_gaze_to_raw_rgb(
                sequence, float(gaze["yaw_rads_cpf"]), float(gaze["pitch_rads_cpf"]),
                float(gaze["depth_m"]),
            )
            nearest = min(times, key=lambda value: abs(value - timestamp))
            x1, y1, x2, y2 = boxes[(uid, nearest)]
            if not pixel_inside_box(pixel, (x1, y1, x2, y2), margin=12.0):
                projection_failed = True
                break
        if projection_failed:
            rejected["gaze_projection_outside_target_box"] += 1
            continue
        candidate["evidence_frame_indices"] = frames
        candidate["maximum_rgb_box_skew_ms"] = round(max(skews_ms), 6)
        candidate["gaze_projection_validated_frames"] = target_gaze_frames
        kept.append(candidate)
    return kept, dict(rejected)


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
    complete = []
    for path in candidates:
        names = {item.name for item in path.iterdir()}
        if REQUIRED_SEQUENCE_FILES.issubset(names) and {"video.vrs", "preview_rgb.mp4"} & names:
            complete.append(path)
    return sorted(complete)


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
    parser.add_argument(
        "--category-target", action="append", default=[], metavar="CATEGORY=COUNT",
        help="Override the requested count for one canonical Task 5 category.",
    )
    parser.add_argument("--max-sequences", type=int, default=0)
    parser.add_argument(
        "--sequence-name", action="append", default=[],
        help="Restrict the run to an exact sequence directory name; repeat for multiple sequences.",
    )
    parser.add_argument(
        "--max-cases-per-sequence", type=int, default=1,
        help="Release invariant; exactly one question is selected from each source sequence.",
    )
    parser.add_argument("--minimum-gaze-run", type=int, default=4)
    parser.add_argument("--minimum-repeated-gaze-gap-sec", type=float, default=2.0)
    parser.add_argument("--target-window-sec", type=float, default=15.0)
    parser.add_argument("--minimum-window-sec", type=float, default=14.5)
    parser.add_argument("--maximum-window-sec", type=float, default=15.5)
    parser.add_argument("--maximum-hit-distance-m", type=float, default=8.0)
    parser.add_argument("--maximum-wearer-skew-ms", type=float, default=10.0)
    parser.add_argument("--maximum-object-pose-skew-ms", type=float, default=50.0)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/qa/task5_scale"))
    parser.add_argument("--site-media-dir", type=Path, default=Path("site/qa_benchmark/task5_media"))
    parser.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument("--site-index", type=Path, default=Path("site/qa_benchmark/index.html"))
    parser.add_argument("--reuse-analysis", action="store_true")
    parser.add_argument("--reuse-media", action="store_true")
    parser.add_argument("--allow-subset", action="store_true", help="Allow Task 5-only intermediate publication.")
    parser.add_argument("--language-client-factory", help="Optional module:function language-only client factory.")
    parser.add_argument("--plan-only", action="store_true", help="Mine/select/write config, but do not export media or update the site")
    args = parser.parse_args()
    if args.max_cases_per_sequence != 1:
        raise SystemExit("Task 5 release requires --max-cases-per-sequence 1")
    language_args = (
        ["--language-client-factory", args.language_client_factory]
        if args.language_client_factory else []
    )

    category_targets: dict[str, int] = {}
    for value in args.category_target:
        name, separator, count = value.partition("=")
        if not separator or not count.isdigit() or int(count) < 1:
            raise SystemExit(f"invalid --category-target {value!r}; expected CATEGORY=positive_integer")
        category_targets[name] = int(count)

    dataset_root = args.dataset_root.resolve()
    output_root = resolve(args.output_root)
    analysis_dir = output_root / "analysis"
    selection_dir = output_root / "selection"
    media_dir = resolve(args.site_media_dir)
    for directory in (analysis_dir, selection_dir):
        directory.mkdir(parents=True, exist_ok=True)
    sequences = discover_sequences(dataset_root)
    if args.sequence_name:
        requested_names = set(args.sequence_name)
        sequences = [path for path in sequences if path.name in requested_names]
        missing_names = sorted(requested_names - {path.name for path in sequences})
        if missing_names:
            raise SystemExit(f"requested ADT sequences are unavailable or incomplete: {missing_names}")
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

    if not 0 < args.minimum_window_sec <= args.target_window_sec <= args.maximum_window_sec:
        raise SystemExit("Task 5 window thresholds must satisfy 0 < minimum <= target <= maximum")
    candidate_policy = Task5CandidatePolicy(
        min_event_gap_sec=args.minimum_repeated_gaze_gap_sec,
        target_window_sec=args.target_window_sec,
        min_window_sec=args.minimum_window_sec,
        max_window_sec=args.maximum_window_sec,
    )

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
                use_cache = int(analysis.get("schema_version") or 0) >= 6 and analysis.get("sequence_name") == sequence_name
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
            candidates, diagnostics = generate_task5_candidates(analysis, candidate_policy)
            candidates, evidence_rejections = _filter_rgb_auditable_candidates(
                candidates, analysis, sequence, args.maximum_object_pose_skew_ms,
            )
            diagnostics["rgb_evidence_rejection_counts"] = evidence_rejections
            diagnostics["rgb_auditable_candidate_count"] = len(candidates)
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
        all_candidates, args.target_per_category, args.max_cases_per_sequence, category_targets,
    )
    report: dict[str, Any] = {
        "pipeline": "annotation-only ADT Task 5 scale pipeline",
        "dataset_root": str(dataset_root),
        "candidate_policy": candidate_policy.__dict__,
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

    # Export only the selected annotation window from each source.  Gaze time_s
    # starts at the first valid gaze row, which need not match the first RGB
    # frame, so device timestamps are the sole media-alignment authority.
    selected_by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in selected:
        selected_by_sequence[str(candidate["sequence_name"])].append(candidate)
    for sequence_name, cases in selected_by_sequence.items():
        if len(cases) != 1:
            raise RuntimeError(
                f"release invariant violated: {sequence_name} has {len(cases)} selected cases"
            )
        sequence = sequence_sources[sequence_name]
        media_slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", sequence_name)[:96]
        case = cases[0]
        analysis = json.loads(analysis_paths[sequence_name].read_text(encoding="utf-8"))
        states_by_frame = {
            int(state["frame_index"]): state for state in analysis["states"]
        }
        lo, hi = (int(value) for value in case["window_frames"])
        start_device_time_ns = int(states_by_frame[lo]["timestamp_ns"])
        end_device_time_ns = int(states_by_frame[hi]["timestamp_ns"])
        rgb_input, alignment = rgb_window_alignment(
            sequence, start_device_time_ns, end_device_time_ns,
        )
        source_media = media_dir / f"{case['id']}_rgb_window_source.mp4"
        case.update({
            "media_source": portable(source_media),
            "media_source_is_exact_window": True,
            "media_alignment": alignment,
        })
        if args.plan_only:
            continue
        media_dir.mkdir(parents=True, exist_ok=True)
        if not (args.reuse_media and source_media.is_file()):
            run_script(
                "export_task5_adt_video.py", str(rgb_input), str(source_media),
                "--start-device-time-ns", str(start_device_time_ns),
                "--end-device-time-ns", str(end_device_time_ns),
            )
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
        "minimum_examples_per_category": min(selection["target_by_category"].values()),
        "minimum_examples_by_category": selection["target_by_category"],
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
            "--quality-output", str(task5_quality), *language_args,
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
        validation_arguments = [
            str(resolve(args.site_data)), "--output", str(combined_quality),
        ]
        if args.allow_subset:
            validation_arguments.append("--allow-subset")
        run_script("validate_task4_task5_release.py", *validation_arguments)
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
