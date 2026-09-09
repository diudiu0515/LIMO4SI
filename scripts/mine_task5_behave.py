#!/usr/bin/env python3
"""Mine geometry-grounded Task 5C candidates from BEHAVE's compact pack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.task5_hoi import analyze_window, synchronized_indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("params_root", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/qa/task5_behave_candidates.json")
    parser.add_argument("--window-sec", type=float, default=9.0)
    parser.add_argument("--stride-sec", type=float, default=3.0)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()
    width, stride = round(args.window_sec * args.fps) + 1, round(args.stride_sec * args.fps)
    candidates, rejected = [], {}
    for sequence in sorted(path for path in args.params_root.iterdir() if path.is_dir()):
        try:
            human = np.load(sequence / "smpl_fit_all.npz", allow_pickle=False)
            obj = np.load(sequence / "object_fit_all.npz", allow_pickle=False)
            info = json.loads((sequence / "info.json").read_text(encoding="utf-8"))
            aligned = synchronized_indices(human["frame_times"], obj["frame_times"])
            if len(aligned) < width:
                rejected[sequence.name] = "short_or_unaligned"
                continue
            hi = np.asarray([row[0] for row in aligned]); oi = np.asarray([row[1] for row in aligned])
            for start in range(0, len(aligned) - width + 1, stride):
                selection = slice(start, start + width)
                metrics = analyze_window(
                    human["trans"][hi[selection]], human["poses"][hi[selection], :3],
                    obj["trans"][oi[selection]], fps=args.fps,
                )
                distance_signal = abs(metrics["distance_change_m"])
                relation_signal = max(0, len(metrics["relative_relation_sequence"]) - 1)
                object_signal = metrics["object_displacement_m"]
                if distance_signal < 0.35 and relation_signal == 0 and object_signal < 0.40:
                    continue
                candidates.append({
                    "id": f"task5c_{sequence.name}_{start:05d}_{start+width-1:05d}",
                    "track": "5C_human_object_interaction_state",
                    "dataset": "BEHAVE",
                    "sequence_name": sequence.name,
                    "object_category": info["cat"],
                    "frame_times": [aligned[start][2], aligned[start + width - 1][2]],
                    "aligned_indices": [start, start + width - 1],
                    "candidate_score": distance_signal + 0.5 * object_signal + 0.2 * relation_signal,
                    "metrics": metrics,
                })
        except Exception as exc:
            rejected[sequence.name] = f"{type(exc).__name__}: {exc}"
    candidates.sort(key=lambda row: (-row["candidate_score"], row["id"]))
    payload = {
        "schema_version": 1,
        "status": "candidates_only_media_not_yet_validated",
        "dataset": "BEHAVE",
        "track": "5C_human_object_interaction_state",
        "coordinate_note": "relative vectors use the fitted SMPL-H root frame; metric distances are rotation invariant",
        "claim_limits": ["no gaze claim", "no contact claim without mesh/contact evidence"],
        "candidate_count": len(candidates), "rejected_sequences": rejected, "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "candidate_count": len(candidates), "rejected": len(rejected)}))


if __name__ == "__main__":
    main()
