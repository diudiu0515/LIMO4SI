#!/usr/bin/env python3
"""Generate audited Task 4/5 QA from one annotation input or bundle manifest."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.scale_quality import require_release_quality
from limo4si.semantic_gt import load_language_realizer
from limo4si.task4_annotation import generate_task4_release

BUNDLE_SCHEMA = "limo4si.annotation_bundle.v1"


def resolve(path: Path, base: Path = ROOT) -> Path:
    return path if path.is_absolute() else (base / path).resolve()


def run_script(name: str, *arguments: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [
        str(ROOT / "src"), str(ROOT / "scripts"), environment.get("PYTHONPATH", ""),
    ]))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / name), *arguments],
        cwd=ROOT, env=environment, check=True,
    )


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_site(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def load_site(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    if not match:
        raise ValueError(f"cannot parse {path}")
    return json.loads(match.group(1))


def detect_kind(path: Path) -> str:
    if path.is_file():
        value = load_json(path)
        if isinstance(value, dict) and value.get("schema") == BUNDLE_SCHEMA:
            return "bundle"
        if isinstance(value, dict) and isinstance(value.get("scenes"), list):
            return "task4"
    if path.is_dir() and (path / "takes.json").is_file() and (
        path / "annotations" / "relations_val.json"
    ).is_file():
        return "task5"
    if path.is_dir():
        return "task4_raw"
    raise ValueError(f"unsupported annotation input: {path}")


def normalized_task4(path: Path, output_dir: Path) -> Path:
    kind = detect_kind(path)
    if kind == "task4":
        return path
    if kind != "task4_raw":
        raise ValueError(f"not a Task 4 annotation source: {path}")
    normalized = output_dir / "normalized_task4_annotations.json"
    run_script(
        "convert_hoim3_to_multihuman.py",
        "--input", str(path), "--output", str(normalized),
    )
    return normalized


def build_task4(
    annotations: Path, output_dir: Path, site_data: Path,
    language_client_factory: str | None,
) -> dict[str, Any]:
    source = normalized_task4(annotations, output_dir)
    payload = load_json(source)
    scenes = payload.get("scenes") if isinstance(payload, dict) else None
    if not isinstance(scenes, list):
        raise ValueError("Task 4 annotation payload must contain a scenes list")
    realizer = load_language_realizer(language_client_factory)
    data, audit = generate_task4_release(scenes, realizer=realizer)
    if not data["groups"]:
        raise ValueError("Task 4 produced no accepted cases; inspect annotation audit")
    quality = require_release_quality(data)
    save_site(site_data, data)
    rows = [
        {"case_index": index, "case_id": group["name"], "video_clip": group.get("video_clip"), **group["qa"][0]}
        for index, group in enumerate(data["groups"], 1)
    ]
    (output_dir / "task4_qa.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8",
    )
    (output_dir / "task4_annotation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    (output_dir / "task4_quality.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return {"task": "task4", **audit}


def empty_site(path: Path) -> None:
    save_site(path, {
        "title": "Task 4 + Task 5 Spatial QA", "subtitle": "Annotation-driven deterministic QA.",
        "tasks": [], "groups": [],
        "release_policy": {"annotation_only": True, "reasoning_owner": "deterministic_code"},
    })


def build_task5(
    annotations: Path, output_dir: Path, site_data: Path, site_dir: Path,
    target_count: int, language_client_factory: str | None,
) -> dict[str, Any]:
    scale_dir = output_dir / "task5_scale"
    arguments = [
        "--dataset-root", str(annotations), "--target-count", str(target_count),
        "--output-root", str(scale_dir), "--site-dir", str(site_dir), "--plan-only",
    ]
    if language_client_factory:
        arguments += ["--language-client-factory", language_client_factory]
    run_script("scale_task5_egoexo.py", *arguments)
    if not site_data.exists():
        empty_site(site_data)
    build_arguments = [
        "--config", str(scale_dir / "generated_release_cases.json"),
        "--dataset-root", str(annotations),
        "--output-jsonl", str(output_dir / "task5_qa.jsonl"),
        "--audit-output", str(output_dir / "task5_annotation_audit.json"),
        "--quality-output", str(output_dir / "task5_quality.json"),
        "--site-dir", str(site_dir), "--site-data", str(site_data),
    ]
    if language_client_factory:
        build_arguments += ["--language-client-factory", language_client_factory]
    run_script("build_task5_egoexo.py", *build_arguments)
    report = load_json(scale_dir / "pipeline_report.json")
    return {"task": "task5", "accepted_count": report["selected_count"], "rejected_count": 0}


def bundle_sources(path: Path) -> tuple[Path | None, Path | None]:
    value = load_json(path)
    if value.get("schema") != BUNDLE_SCHEMA:
        raise ValueError(f"bundle schema must be {BUNDLE_SCHEMA}")
    base = path.parent
    task4 = resolve(Path(value["task4_annotations"]), base) if value.get("task4_annotations") else None
    task5 = resolve(Path(value["task5_annotations"]), base) if value.get("task5_annotations") else None
    if task4 is None and task5 is None:
        raise ValueError("bundle contains no task annotation source")
    return task4, task5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--task", choices=("auto", "task4", "task5"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/annotation_release"))
    parser.add_argument("--site-dir", type=Path, default=Path("site/annotation_release"))
    parser.add_argument("--task5-target-count", type=int, default=12)
    parser.add_argument("--language-client-factory")
    args = parser.parse_args()

    annotations = resolve(args.annotations)
    output_dir = resolve(args.output_dir)
    site_dir = resolve(args.site_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)
    site_data = site_dir / "data.js"
    kind = detect_kind(annotations) if args.task == "auto" else args.task
    task4_source = task5_source = None
    if kind == "bundle":
        task4_source, task5_source = bundle_sources(annotations)
    elif kind in {"task4", "task4_raw"} or args.task == "task4":
        task4_source = annotations
    elif kind == "task5" or args.task == "task5":
        task5_source = annotations
    else:
        raise ValueError(f"cannot route annotation kind {kind}")

    reports = []
    if task4_source:
        reports.append(build_task4(task4_source, output_dir, site_data, args.language_client_factory))
    if task5_source:
        reports.append(build_task5(
            task5_source, output_dir, site_data, site_dir,
            args.task5_target_count, args.language_client_factory,
        ))
    run_script("project_task4_task5_release.py", "--data-js", str(site_data))
    run_script("build_static_qa_site.py", "--data-js", str(site_data), "--output", str(site_dir / "index.html"))
    quality_path = output_dir / "release_quality.json"
    validation_args = [str(site_data), "--output", str(quality_path)]
    if not (task4_source and task5_source):
        validation_args.append("--allow-subset")
    run_script("validate_task4_task5_release.py", *validation_args)
    summary = {"status": "ok", "annotations": str(annotations), "reports": reports,
               "site_data": str(site_data), "quality": str(quality_path)}
    (output_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
