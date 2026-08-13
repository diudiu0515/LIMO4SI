#!/usr/bin/env python3
"""Build implemented Perspective-Grounded QA examples from spatial summaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.perspective_qa import (  # noqa: E402
    human_centric_answer,
    level2_occlusion_answer,
    nearest_reachable_object,
    reference_frame_switching_answer,
    visibility_answer,
)

TASK_REQUIREMENTS = [
    {
        "id": "person_centric_spatial",
        "name": "Human-centric left/right/front/back",
        "goal": "Answer where an object is relative to a named person, using the person body frame rather than image coordinates.",
        "evidence": ["3D human skeleton / pelvis origin", "body right/front/up axes", "object mask or open-vocabulary segmentation", "object 3D centroid"],
        "example": "From the blue-shirt person's view, is the sofa on his left or right?",
        "implementation_status": "implemented",
    },
    {
        "id": "visibility_occlusion",
        "name": "Visibility and occlusion reasoning",
        "goal": "Decide whether the target object is visible from the person's head/gaze direction, considering occluders.",
        "evidence": ["head or face direction from EgoPose", "target 3D position", "candidate object 3D centers", "optional dense masks/depth for stronger occlusion"],
        "example": "Can the cook see the painting on the wall? Consider head direction and occlusion.",
        "implementation_status": "implemented with head/body direction and centroid-line occlusion approximation",
    },
    {
        "id": "reachability",
        "name": "Reachable nearest object",
        "goal": "Find the nearest visible object that lies inside the person's reachable region.",
        "evidence": ["wrist / hand keypoints", "estimated reach radius from shoulder width", "object 3D centroid"],
        "example": "Which toy is closest and reachable by the mother?",
        "implementation_status": "implemented with wrist distance and estimated reach radius",
    },
    {
        "id": "level2_perspective",
        "name": "Level-2 perspective taking",
        "goal": "Reason about what one person would perceive as occluding another object from that person's viewpoint.",
        "evidence": ["observer head position", "target object", "candidate occluder centers", "depth ordering along sightline"],
        "example": "From the child's perspective, which object blocks the TV?",
        "implementation_status": "implemented as line-of-sight blocker detection; single-person fallback if only one pose exists",
    },
    {
        "id": "reference_frame_switching",
        "name": "Reference-frame switching",
        "goal": "Answer the same spatial question under egocentric, allocentric, and human-centric coordinate frames.",
        "evidence": ["camera extrinsics/intrinsics", "world xyz", "human skeleton frame"],
        "example": "Give egocentric, allocentric, and human-centric answers for the same object relation.",
        "implementation_status": "implemented for human/world/camera coordinates; semantic room labels require declared world axes",
    },
]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_body_pose(root: Path, sample: dict) -> dict:
    path = sample.get("inputs", {}).get("body_pose")
    if path:
        p = Path(path)
    else:
        p = root / "data/egoexo4d/annotations/ego_pose/val/body/automatic" / f"{sample['take_uid']}.json"
    body = load_json(p)
    return body[str(sample["frame"])][0]


def find_camera_calibration(root: Path, sample: dict) -> dict | None:
    path = sample.get("inputs", {}).get("camera_pose")
    if path and Path(path).exists():
        all_cal = load_json(Path(path))
        return all_cal.get(sample["camera"])
    p = root / "data/egoexo4d/annotations/ego_pose/val/camera_pose" / f"{sample['take_uid']}.json"
    if p.exists():
        return load_json(p).get(sample["camera"])
    return None


def object_label(sample: dict) -> str:
    return str(sample.get("object_id", "target object")).replace("_0", "").replace("_1", "")


def build_group_examples(root: Path, summary: dict) -> list[dict]:
    samples = summary.get("samples", [])
    ok = [s for s in samples if s.get("object_xyz_world_m")]
    if not ok:
        return []
    first = ok[0]
    second = ok[1] if len(ok) > 1 else ok[0]
    body_pose = find_body_pose(root, first)
    joints = body_pose["annotation3D"]
    cal = find_camera_calibration(root, first)

    hc = human_centric_answer(first["object_xyz_world_m"], first["human_frame"])
    vis = visibility_answer(second, joints, candidates=ok)
    reach = nearest_reachable_object(ok, joints)
    lvl2 = level2_occlusion_answer(joints, first, ok)
    ref = reference_frame_switching_answer(
        first["object_xyz_world_m"],
        first["human_frame"],
        camera_intrinsics=cal.get("camera_intrinsics") if cal else None,
        camera_extrinsics=cal.get("camera_extrinsics") if cal else None,
    )

    return [
        {
            "task_id": "person_centric_spatial",
            "task_name": "Human-centric left/right/front/back",
            "query": f"From the person's view, where is the {object_label(first)}?",
            "answer": hc["answer"],
            "status": hc["status"],
            "method": "Computed from human_frame.world_to_human(object_centroid) and independent lateral/longitudinal/vertical thresholds.",
            "result_json": hc,
            "raw_json": first,
        },
        {
            "task_id": "visibility_occlusion",
            "task_name": "Visibility and occlusion reasoning",
            "query": f"Can the person see the {object_label(second)}? Consider head/body direction and occlusion.",
            "answer": vis["answer"],
            "status": vis["status"],
            "method": "Head/face direction if available; otherwise body-forward. Occlusion is approximated by listed object centers inside an observer-to-target tube.",
            "result_json": vis,
            "raw_json": {"target": second, "candidates": ok},
        },
        {
            "task_id": "reachability",
            "task_name": "Reachable nearest object",
            "query": "Which listed object is closest and reachable by the person?",
            "answer": reach["answer"],
            "status": reach["status"],
            "method": "Uses nearest wrist-to-object distance with reach radius estimated from shoulder width; falls back to pelvis distance if wrists are absent.",
            "result_json": reach,
            "raw_json": {"candidates": ok},
        },
        {
            "task_id": "level2_perspective",
            "task_name": "Level-2 perspective taking",
            "query": f"From the observer's perspective, which listed object blocks the {object_label(first)}?",
            "answer": lvl2["answer"],
            "status": lvl2["status"],
            "method": "Casts a line segment from observer head to target and selects candidate objects between them with smallest depth and tube distance.",
            "result_json": lvl2,
            "raw_json": {"target": first, "candidates": ok},
        },
        {
            "task_id": "reference_frame_switching",
            "task_name": "Reference-frame switching",
            "query": f"Describe the {object_label(first)} in human-centric, camera/egocentric, and allocentric/world frames.",
            "answer": ref["answer"],
            "status": ref["status"],
            "method": "Human-centric uses body axes; egocentric uses camera extrinsics; allocentric reports world xyz unless semantic room axes are supplied.",
            "result_json": ref,
            "raw_json": first,
        },
    ]


def update_site_data(root: Path, summary_root: Path, site_data: Path, output: Path | None) -> None:
    text = site_data.read_text(encoding="utf-8")
    data = json.loads(re.search(r"window\.QA_DATA\s*=\s*(\{.*\});\s*$", text, re.S).group(1))
    data["task_requirements"] = TASK_REQUIREMENTS
    for group in data["groups"]:
        summary_path = root / group.get("summary_path", "")
        if not summary_path.exists():
            summary_path = summary_root / group["name"] / "summary.json"
        summary = load_json(summary_path)
        examples = build_group_examples(root, summary)
        group["perspective_examples"] = examples
        group["raw_summary"] = summary
        out_dir = summary_path.parent
        (out_dir / "perspective_qa.json").write_text(json.dumps({
            "group": group["name"],
            "task_requirements": TASK_REQUIREMENTS,
            "perspective_examples": examples,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    target = output or site_data
    target.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    print(target)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-root", type=Path, default=Path("outputs/spatial/showcase_queries"))
    ap.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    update_site_data(ROOT, args.summary_root, args.site_data, args.output)


if __name__ == "__main__":
    main()
