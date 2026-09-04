#!/usr/bin/env python3
"""Build audited Task 3 human-trajectory / scene-topology QA."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_humans_in_space_site_data import body_timeline, build_clip
from limo4si.scale_quality import TASK3_ID, ScaleQualityPolicy, require_release_quality
from limo4si.task3_topology import cross, dot, horizontal, norm, sub, trajectory_topology, unit

TASK3_NAME = "Task 3 · Human–Scene Topological Reasoning"
CATEGORY_BY_TYPE = {
    "local_landmark_pass_side": "local_path_side",
    "landmark_closest_approach_order": "temporal_landmark_order",
    "closest_landmark_to_full_trajectory": "full_route_proximity",
}


def load_js(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    return json.loads(match.group(1) if match else text)


def save_js(path: Path, data: dict[str, Any]) -> None:
    path.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def rounded(value: float, decimals: int = 1) -> str:
    quantum = Decimal(1).scaleb(-decimals)
    return f"{Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP):.{decimals}f}"


def name(value: str) -> str:
    text = str(value).replace("_", " ").strip()
    parts = text.rsplit(" ", 1)
    if len(parts) == 2 and parts[1].isdigit():
        text = parts[0]
    return " ".join(text.lower().split())


def make_options(correct: str, distractors: list[str], seed: str) -> tuple[list[dict[str, str]], str]:
    values = [correct, *distractors]
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError(f"Task 3 option template did not produce four unique choices: {seed}")
    offset = int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % 4
    ordered = distractors[:]
    ordered.insert(offset, correct)
    labels = ["A", "B", "C", "D"]
    return [{"label": label, "text": text} for label, text in zip(labels, ordered)], labels[offset]


def question_for(case_id: str, qtype: str, topology: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "ok", "answer_type": qtype, "T_Q": True, "H_Q": True, "S_Q": True,
        "topology": topology,
    }
    if qtype == "local_landmark_pass_side":
        event = next(row for row in topology["landmarks"] if row["landmark_id"] == spec["landmark_id"])
        target = name(event["display_name"])
        side = event["pass_side"]
        correct = f"Around closest approach, the {target} stays on the {side} side of the person's local travel direction."
        distractors = [
            f"Around closest approach, the {target} stays on the {'right' if side == 'left' else 'left'} side of the person's local travel direction.",
            f"Around closest approach, the {target} changes from the left side to the right side of the person's local travel direction.",
            f"Around closest approach, the {target} changes from the right side to the left side of the person's local travel direction.",
        ]
        question = f"As the person moves past the {target}, which side of the person's local direction of travel is it on around closest approach?"
        explanation = (
            f"The local-tangent signed offset is about {rounded(abs(event['signed_lateral_m']))} m to the {side}; "
            f"the local trajectory covers about {rounded(event['local_travel_m'])} m and keeps the same side around closest approach."
        )
        method = "Uses the full smoothed metric pelvis trajectory. Side is measured against the local path tangent at closest approach, not image left/right or the start-to-end chord."
        result["pass_event"] = event
    elif qtype == "landmark_closest_approach_order":
        wanted = set(spec["landmark_ids"])
        event = next(row for row in topology["order_pairs"] if {row["first_landmark"], row["second_landmark"]} == wanted)
        by_id = {row["landmark_id"]: row for row in topology["landmarks"]}
        first = name(by_id[event["first_landmark"]]["display_name"])
        second = name(by_id[event["second_landmark"]]["display_name"])
        correct = f"The person reaches the closest point to the {first} first and to the {second} later."
        distractors = [
            f"The person reaches the closest point to the {second} first and to the {first} later.",
            f"The person reaches the closest points to the {first} and the {second} at the same time.",
            f"The person reaches no distinct closest point to either the {first} or the {second} during the clip.",
        ]
        question = f"During the clip, in what order does the person's trajectory reach its closest points to the {first} and the {second}?"
        explanation = f"The two closest-approach times are separated by about {rounded(event['time_gap_sec'])} seconds, with the {first} occurring first."
        method = "Computes each landmark's minimum horizontal distance to every state of the full smoothed metric trajectory, then compares the associated times."
        result["order_event"] = event
    elif qtype == "closest_landmark_to_full_trajectory":
        ranking = topology["route_landmark_ranking"]
        if len(ranking) < 3:
            raise ValueError(f"{case_id}: route ranking needs three landmarks for balanced choices")
        shown = ranking[:3]
        winner = name(shown[0]["display_name"])
        correct = f"The {winner} has the smallest minimum distance to the person's full trajectory."
        distractors = [
            f"The {name(row['display_name'])} has the smallest minimum distance to the person's full trajectory."
            for row in shown[1:]
        ] + ["All listed landmarks have the same minimum distance to the person's full trajectory."]
        question = "Which listed landmark does the person's full trajectory come closest to at any point in the clip?"
        explanation = (
            f"The route comes within about {rounded(shown[0]['minimum_horizontal_distance_m'])} m of the {winner}; "
            f"the next-smallest minimum distance is about {rounded(shown[1]['minimum_horizontal_distance_m'])} m."
        )
        method = "Ranks static 3D landmarks by their minimum horizontal distance to the full smoothed metric pelvis trajectory, rather than using one frame."
        result["route_landmark_ranking"] = ranking
    else:
        raise ValueError(f"unsupported Task 3 question type: {qtype}")
    options, correct_option = make_options(correct, distractors, case_id + qtype)
    return {
        "task_id": TASK3_ID,
        "task_name": TASK3_NAME,
        "question_type": qtype,
        "question_categories": [CATEGORY_BY_TYPE[qtype]],
        "question": question,
        "options": options,
        "correct_option": correct_option,
        "correct_answer": correct,
        "answer": correct,
        "explanation": explanation,
        "status": "ok",
        "method": method,
        "result_json": result,
    }


def topology_svg(case_id: str, topology: dict[str, Any], output_dir: Path) -> str:
    points = [row["origin_world_m"] for row in topology["trajectory_states"]]
    up = topology["reference_up_world_unit"]
    best = max(((norm(horizontal(sub(b, a), up)), a, b) for a in points for b in points), key=lambda row: row[0])
    axis_x = unit(horizontal(sub(best[2], best[1]), up)) or [1.0, 0.0, 0.0]
    axis_y = unit(cross(up, axis_x)) or [0.0, 0.0, 1.0]
    origin = points[0]
    landmark_rows = topology["landmarks"]
    coordinates = [(dot(sub(point, origin), axis_x), dot(sub(point, origin), axis_y)) for point in points]
    landmark_xy = [(row, dot(sub(row["center_world_m"], origin), axis_x), dot(sub(row["center_world_m"], origin), axis_y)) for row in landmark_rows]
    all_x = [x for x, _ in coordinates] + [x for _, x, _ in landmark_xy]
    all_y = [y for _, y in coordinates] + [y for _, _, y in landmark_xy]
    min_x, max_x = min(all_x) - 0.3, max(all_x) + 0.3
    min_y, max_y = min(all_y) - 0.3, max(all_y) + 0.3
    scale = min(720 / max(max_x - min_x, 0.1), 420 / max(max_y - min_y, 0.1))
    def xy(x: float, y: float) -> tuple[float, float]:
        return (50 + (x - min_x) * scale, 450 - (y - min_y) * scale)
    path_points = " ".join(f"{xy(x, y)[0]:.1f},{xy(x, y)[1]:.1f}" for x, y in coordinates)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="820" height="500" viewBox="0 0 820 500">',
        '<rect width="820" height="500" fill="#f8fafc"/>',
        f'<polyline points="{path_points}" fill="none" stroke="#2563eb" stroke-width="5" stroke-linejoin="round" opacity="0.9"/>',
    ]
    sx, sy = xy(*coordinates[0]); ex, ey = xy(*coordinates[-1])
    parts += [f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="9" fill="#16a34a"/><text x="{sx+12:.1f}" y="{sy-10:.1f}" font-size="15">start</text>', f'<circle cx="{ex:.1f}" cy="{ey:.1f}" r="9" fill="#dc2626"/><text x="{ex+12:.1f}" y="{ey-10:.1f}" font-size="15">end</text>']
    colors = ["#f59e0b", "#7c3aed", "#0891b2", "#db2777", "#65a30d"]
    for index, (row, lx, ly) in enumerate(landmark_xy):
        x, y = xy(lx, ly); label = html.escape(row["display_name"])
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{colors[index % len(colors)]}"/><text x="{x+13:.1f}" y="{y+5:.1f}" font-size="15" font-weight="700">{label}</text>')
    parts.append('<text x="22" y="486" font-size="14" fill="#475569">Blue: full smoothed metric pelvis trajectory · landmark coordinates are scene-fixed</text></svg>')
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{case_id}_trajectory.svg"
    output.write_text("\n".join(parts), encoding="utf-8")
    return "./" + str(output.relative_to(ROOT / "site/qa_benchmark")) if output.is_relative_to(ROOT / "site/qa_benchmark") else "./outputs/qa/task3_media/" + output.name


def build_group(spec: dict[str, Any], landmark_scenes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary_path = ROOT / spec["source_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scene = landmark_scenes.get(spec["source_summary"])
    if not scene or not scene.get("landmarks"):
        raise ValueError(f"no audited static metric landmarks: {summary_path}")
    sample = dict(next(row for row in summary["samples"] if row.get("object_xyz_world_m")))
    sample["frame"] = int(spec["center_frame"])
    duration = float(spec["duration_sec"])
    timeline = body_timeline(ROOT, sample, duration)
    topology = trajectory_topology(timeline, scene["landmarks"])
    case_id = f"task3_{sample['take_name']}_frame{sample['frame']}_{int(duration)}s"
    question = question_for(case_id, spec["question_type"], topology, spec)
    release_summary = {**summary, "samples": [sample]}
    group = {
        "name": case_id,
        "title": f"Task 3 · {sample.get('take_name')} · {spec['question_type'].replace('_', ' ')}",
        "original_image": "./" + scene["review_image"],
        "topdown_image": topology_svg(case_id, topology, ROOT / "outputs/qa/task3_media"),
        "summary_path": spec["source_summary"],
        "raw_summary": release_summary,
        "static_landmark_audit": scene,
        "clip_filename": f"task3_clip_frame{sample['frame']}_{int(duration)}s.mp4",
        "qa": [question],
        "case_policy": "one temporal question per unique video window",
    }
    build_clip(ROOT, group, duration)
    return group


def category_audit(groups: list[dict[str, Any]], minimum: int) -> dict[str, Any]:
    counts = Counter(group["qa"][0]["question_categories"][0] for group in groups)
    missing = [category for category in CATEGORY_BY_TYPE.values() if counts[category] < minimum]
    if missing:
        raise ValueError(f"Task 3 categories below {minimum} examples: {missing}")
    windows = [(group["video_window"]["source_video"], group["video_window"]["start_sec"], group["video_window"]["duration_sec"]) for group in groups]
    if len(windows) != len(set(windows)):
        raise ValueError("Task 3 release contains duplicate video windows")
    return {
        "status": "ok", "case_count": len(groups), "minimum_examples_per_category": minimum,
        "category_counts": dict(counts), "unique_video_windows": len(windows),
        "coordinate_policy": "metric world trajectory projected onto scene horizontal plane; local path side is never image left/right",
        "scope_limit": "no room-entry, doorway, or obstacle-bypass claim is generated without semantic region/footprint annotations",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task3_release_cases.json"))
    parser.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument("--static-landmarks", type=Path, default=Path("outputs/qa/task3_static_landmarks.json"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/qa/task3_scaled_qa.jsonl"))
    parser.add_argument("--audit-output", type=Path, default=Path("outputs/qa/task3_scale_audit.json"))
    parser.add_argument("--quality-output", type=Path, default=Path("outputs/qa/task3_scale_quality.json"))
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    landmark_data = json.loads(resolve(args.static_landmarks).read_text(encoding="utf-8"))
    landmark_scenes = {scene["source_summary"]: scene for scene in landmark_data["scenes"]}
    groups = [build_group(spec, landmark_scenes) for spec in config["cases"]]
    audit = category_audit(groups, int(config.get("minimum_examples_per_category", 2)))
    quality = require_release_quality({"groups": groups}, ScaleQualityPolicy())

    site_path = resolve(args.site_data)
    data = load_js(site_path)
    data["groups"] = [group for group in data.get("groups", []) if not str(group.get("name", "")).startswith("task3_")]
    data["groups"].extend(groups)
    task = {"id": TASK3_ID, "name": TASK3_NAME, "description": "How the full human trajectory passes, approaches, and is arranged relative to static scene landmarks."}
    tasks = [row for row in data.get("tasks", []) if row.get("id") != TASK3_ID]
    insert_at = next((index + 1 for index, row in enumerate(tasks) if row.get("id", "").startswith("task1_")), len(tasks))
    tasks.insert(insert_at, task)
    data["tasks"] = tasks
    data["title"] = "Humans in Space · Task 1 + Task 3 + Task 4 QA"
    data["subtitle"] = "One evidence-grounded temporal question per unique video window."
    policy = data.setdefault("release_policy", {})
    policy["task_scope"] = [row["id"] for row in tasks]
    policy["task3_scope"] = "full metric human trajectory versus static 3D landmarks"
    require_release_quality(data, ScaleQualityPolicy())
    save_js(site_path, data)

    rows = [{"case_index": index, "case_id": group["name"], "video_clip": group.get("video_clip"), **group["qa"][0]} for index, group in enumerate(groups, 1)]
    output = resolve(args.output_jsonl); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    resolve(args.audit_output).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resolve(args.quality_output).write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
