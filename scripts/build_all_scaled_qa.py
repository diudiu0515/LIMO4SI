#!/usr/bin/env python3
"""Rebuild the combined Task 1 + Task 3 + Task 4 audited release."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script: str, *args: str) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], cwd=ROOT, env=env, check=True)


def main() -> None:
    run("build_task1_task4_curated.py")
    run("build_task1_task4_scaled.py")
    run("build_task3_scaled.py")
    run("build_static_qa_site.py")
    run("validate_task1_task4_scale.py", "site/qa_benchmark/data.js", "--output", "outputs/qa/task1_task3_task4_scale_quality.json")


if __name__ == "__main__":
    main()
