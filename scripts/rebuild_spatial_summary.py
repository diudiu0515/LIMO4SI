#!/usr/bin/env python3
"""Rebuild a spatial output summary from the configured per-sample JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    samples = json.loads(args.samples.read_text())
    results = []
    for sample in samples:
        matches = [
            path
            for path in args.results.glob("*.json")
            if path.name != "summary.json"
            and (row := json.loads(path.read_text()))["take_uid"] == sample["take_uid"]
            and row["camera"] == sample["camera"]
            and row["frame"] == sample["frame"]
            and row["object_id"] == sample["object_id"]
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one result for {sample}, found {len(matches)}")
        result = json.loads(matches[0].read_text())
        result["output_path"] = str(matches[0])
        results.append(result)
    summary = {
        "sample_count": len(results),
        "success_count": sum(row["status"] == "ok" for row in results),
        "error_count": sum(row["status"] != "ok" for row in results),
        "samples": results,
    }
    output = args.results / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
