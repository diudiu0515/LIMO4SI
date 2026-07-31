#!/usr/bin/env python3
"""Download only the takes selected by select_relation_takes.py."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_PARTS = (
    "downscaled_takes/448",
    "take_trajectory",
    "take_point_cloud",
    "ego_pose_pseudo_gt",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("uid_file", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument("--profile", default="egoexo")
    parser.add_argument("--release", default="v2")
    parser.add_argument("--parts", nargs="+", default=list(DEFAULT_PARTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    uids = [
        line.strip()
        for line in args.uid_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not uids:
        raise SystemExit(f"No UIDs found in {args.uid_file}")

    executable = Path(".venv/bin/egoexo").resolve()
    command = [
        str(executable),
        "-o",
        str(args.output.resolve()),
        "--release",
        args.release,
        "--parts",
        *args.parts,
        "--uids",
        *uids,
        "--s3_profile",
        args.profile,
        "-y",
    ]
    print("Command:")
    print(" ".join(command))
    if args.dry_run:
        return

    env = os.environ.copy()
    for key in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy",
    ):
        env.pop(key, None)
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
