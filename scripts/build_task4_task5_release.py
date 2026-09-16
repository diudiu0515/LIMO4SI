#!/usr/bin/env python3
"""Rebuild the public Task 4 + Task 5 audited release."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_stage(script: str, *arguments: str) -> None:
    """Run one release stage with the repository source on PYTHONPATH."""

    environment = dict(os.environ)
    existing_path = environment.get("PYTHONPATH")
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_path
        else source_path + os.pathsep + existing_path
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / script),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language-client-factory",
        help=(
            "Optional language-only structured-output "
            "client factory"
        ),
    )
    args = parser.parse_args()
    language_arguments = (
        [
            "--language-client-factory",
            args.language_client_factory,
        ]
        if args.language_client_factory
        else []
    )

    # Source extraction is followed by a fail-closed public-task projection.
    run_stage("build_task4_curated.py")
    run_stage(
        "build_task4_scaled.py",
        *language_arguments,
    )
    run_stage("project_task4_task5_release.py")
    run_stage("build_task5_egoexo.py", *language_arguments)
    run_stage(
        "project_task4_task5_release.py",
        "--require-both",
    )
    run_stage("build_static_qa_site.py")
    run_stage("build_task4_task5_review.py")
    run_stage(
        "validate_task4_task5_release.py",
        "site/qa_benchmark/data.js",
        "--output",
        "outputs/qa/task4_task5_scale_quality.json",
    )


if __name__ == "__main__":
    main()
