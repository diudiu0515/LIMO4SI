#!/usr/bin/env python3
"""Publish license-safe Task 5 examples without dataset RGB or raw annotations."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_ID = "task5_human_state_grounded_spatial_reasoning"


def read_site(path: Path) -> dict:
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", path.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError(f"cannot parse {path}")
    return json.loads(match.group(1))


def write_svg(path: Path, title: str, subtitle: str, body: str) -> None:
    path.write_text(
        f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="540" viewBox="0 0 960 540">
<rect width="960" height="540" fill="#f5f1e8"/><text x="48" y="58" font-family="sans-serif" font-size="28" font-weight="700" fill="#17202a">{title}</text>
<text x="48" y="88" font-family="sans-serif" font-size="17" fill="#52606d">{subtitle}</text>{body}
<text x="48" y="508" font-family="sans-serif" font-size="15" fill="#6b7280">Annotation-derived schematic; no dataset RGB or identifiable imagery.</text></svg>''',
        encoding="utf-8",
    )


def ego_group(media: Path, qa_path: Path) -> dict:
    qa = json.loads(qa_path.read_text(encoding="utf-8"))
    measurements = qa["evidence"]["measurements"]
    distance = sum(x["distance_m"] for x in measurements) / len(measurements)
    svg = media / "task5a_egobody_gaze_interactee_schematic.svg"
    write_svg(svg, "EgoBody · gaze-grounded human relation", "Measured gaze on interactee + scene-aligned SMPL-X", '''
<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#d1495b"/></marker></defs>
<circle cx="480" cy="410" r="32" fill="#267c8c"/><text x="480" y="460" text-anchor="middle" font-family="sans-serif" font-size="18">camera wearer</text>
<circle cx="480" cy="190" r="32" fill="#e09f3e"/><text x="480" y="145" text-anchor="middle" font-family="sans-serif" font-size="18">visible interactee</text>
<line x1="480" y1="375" x2="480" y2="230" stroke="#d1495b" stroke-width="5" marker-end="url(#arrow)"/><text x="500" y="305" font-family="sans-serif" font-size="18" fill="#d1495b">measured gaze</text>
<line x1="545" y1="410" x2="545" y2="190" stroke="#334e68" stroke-width="2" stroke-dasharray="8 6"/><text x="565" y="307" font-family="sans-serif" font-size="20">front · %.2f m</text>
''' % distance)
    result = {
        "status": "ok", "answer_type": "gaze_grounded_interactee_relation", "T_Q": True, "H_Q": True, "S_Q": True,
        "annotation_direct": True, "annotation_source": "EgoBody", "coordinate_frame": "camera wearer's scene-aligned SMPL-X root frame",
        "gaze_grounding_method": "measured HoloLens gaze projected inside visible interactee keypoint box",
        "measurements": measurements,
    }
    return {
        "name": "task5a_egobody_gaze_interactee_front", "title": "Task 5A · EgoBody · gaze on interactee",
        "original_image": "./task5_media/" + svg.name,
        "original_caption": "License-safe annotation-derived schematic; raw EgoBody RGB is not distributed.",
        "video_window": {"source_sequence": qa["recording_name"], "start_frame": qa["frame_range"][0], "end_frame": qa["frame_range"][1], "public_media": "schematic_only"},
        "qa": [{"task_id": TASK_ID, "task_name": "Task 5 · Human-State–Grounded Spatial Reasoning", "question_type": "gaze_grounded_interactee_relation",
            "question_categories": ["gaze_on_visible_interactee"], "question": "While the measured gaze remains on the visible interactee, where is that person relative to the camera wearer?",
            "options": [{"label": "A", "text": "In front of the camera wearer."}, {"label": "B", "text": "Behind the camera wearer."}, {"label": "C", "text": "Clearly to the camera wearer's left."}, {"label": "D", "text": "Clearly to the camera wearer's right."}],
            "correct_option": "A", "correct_answer": "In front of the camera wearer.", "answer": "In front of the camera wearer.",
            "explanation": "At three audited frames the interactee is approximately 1.92 m away and remains in the front sector of the wearer's SMPL-X root frame.",
            "status": "ok", "method": "Measured HoloLens gaze and scene-aligned SMPL-X fits; no LLM label judgment.", "result_json": result}],
        "case_policy": "No EgoBody RGB or raw annotation is included in the public repository.",
    }


def behave_group(media: Path, qa_path: Path) -> dict:
    row = json.loads(qa_path.read_text(encoding="utf-8").splitlines()[0])
    ev = row["result_json"]
    object_move = float(ev["object_center_endpoint_displacement_m"])
    human_move = float(ev["human_root_endpoint_displacement_m"])
    svg = media / "task5c_behave_backpack_motion_schematic.svg"
    write_svg(svg, "BEHAVE · human-object motion", "Registered backpack 6DoF + fitted SMPL-H root", f'''
<defs><marker id="a" markerWidth="10" markerHeight="10" refX="7" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#267c8c"/></marker></defs>
<circle cx="260" cy="330" r="31" fill="#e09f3e"/><circle cx="350" cy="310" r="24" fill="#d1495b"/><text x="260" y="385" text-anchor="middle" font-family="sans-serif" font-size="18">human start</text><text x="350" y="270" text-anchor="middle" font-family="sans-serif" font-size="18">backpack start</text>
<line x1="290" y1="325" x2="420" y2="300" stroke="#e09f3e" stroke-width="8" marker-end="url(#a)"/><line x1="375" y1="300" x2="735" y2="205" stroke="#d1495b" stroke-width="10" marker-end="url(#a)"/>
<circle cx="440" cy="296" r="31" fill="#e09f3e"/><circle cx="760" cy="198" r="24" fill="#d1495b"/><text x="440" y="355" text-anchor="middle" font-family="sans-serif" font-size="18">human end · {human_move:.2f} m</text><text x="760" y="155" text-anchor="middle" font-family="sans-serif" font-size="18">backpack end · {object_move:.2f} m</text>
''')
    result = {"status": "ok", "answer_type": "human_object_motion_comparison", "T_Q": True, "H_Q": True, "S_Q": True,
              "annotation_direct": True, "annotation_source": "BEHAVE", "human_displacement_m": human_move, "object_displacement_m": object_move,
              "evidence_scope": "time-aligned fitted SMPL-H root and registered backpack 6DoF; no contact or gaze claim"}
    return {
        "name": "task5c_behave_backpack_motion", "title": "Task 5C · BEHAVE · backpack motion",
        "original_image": "./task5_media/" + svg.name, "original_caption": "License-safe annotation-derived motion schematic; no BEHAVE source frames are distributed.",
        "video_window": {"source_sequence": ev["sequence_name"], "start_sec": 24.0, "duration_sec": 9.0, "public_media": "schematic_only"},
        "qa": [{"task_id": TASK_ID, "task_name": "Task 5 · Human-State–Grounded Spatial Reasoning", "question_type": "human_object_motion_comparison",
            "question_categories": ["human_object_state"], "question": "Across this interaction window, which moved farther in the registered 3D scene: the person or the backpack?",
            "options": [{"label": "A", "text": "The backpack moved farther."}, {"label": "B", "text": "The person moved farther."}, {"label": "C", "text": "They moved the same distance."}, {"label": "D", "text": "The annotations cannot distinguish them."}],
            "correct_option": "A", "correct_answer": "The backpack moved farther.", "answer": "The backpack moved farther.",
            "explanation": f"The registered backpack moved about {object_move:.2f} m while the fitted human root moved about {human_move:.2f} m.",
            "status": "ok", "method": "Time-aligned BEHAVE SMPL-H and registered object fits; no LLM label judgment.", "result_json": result}],
        "case_policy": "No BEHAVE source frames or raw annotation is included in the public repository.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site-data", type=Path, default=ROOT / "site/qa_benchmark/data.js")
    args = parser.parse_args()
    data = read_site(args.site_data)
    media = ROOT / "site/qa_benchmark/task5_media"
    media.mkdir(parents=True, exist_ok=True)
    replacements = [ego_group(media, ROOT / "outputs/qa/task5_egobody_pilot/qa.json"), behave_group(media, ROOT / "outputs/qa/task5_behave_pilot_qa.jsonl")]
    names = {x["name"] for x in replacements}
    data["groups"] = [x for x in data["groups"] if x.get("name") not in names] + replacements
    data["release_policy"]["task5_scope"] = "ADT gaze-object, EgoBody gaze-human, and BEHAVE human-object relations; public real-data additions use annotation-derived schematics only"
    args.site_data.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")
    print(json.dumps({"groups": len(data["groups"]), "added": sorted(names)}))


if __name__ == "__main__":
    main()
