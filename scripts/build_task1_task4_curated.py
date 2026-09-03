#!/usr/bin/env python3
"""Build the curated Task 1 + Task 4 benchmark slice.

Policy:
* exactly one temporal question per case;
* Task 1 and Task 4 only;
* metric multi-human QA requires visually aligned A/B SMPL-X tracks;
* missing blocker geometry is never interpreted as a clear line of sight;
* answers and distractors are derived from stored evidence, not guessed labels.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_multihuman_dynamic_qa import write_svg  # noqa: E402
from limo4si.multihuman import pair_timeline  # noqa: E402

TASK1_ID = "task1_dynamic_human_referenced_relations"
TASK4_ID = "task4_multi_human_relational_dynamics"
TASK1_NAME = "Task 1 · Dynamic Human-Referenced Relations"
TASK4_NAME = "Task 4 · Multi-Human Relational Dynamics"

TASKS = [
    {
        "id": TASK1_ID,
        "name": TASK1_NAME,
        "description": "How object relations and body-forward visibility change or remain stable as the human moves or turns.",
    },
    {
        "id": TASK4_ID,
        "name": TASK4_NAME,
        "description": "Temporal position, orientation, distance, topology, visibility and relation change between people.",
    },
]


def load_js(path: Path) -> dict[str, Any]:
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", path.read_text(encoding="utf-8"), re.S)
    if not match:
        raise ValueError(f"Cannot parse {path}")
    return json.loads(match.group(1))


def save_js(path: Path, data: dict[str, Any]) -> None:
    path.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def options(correct: str, distractors: list[str], seed: str) -> tuple[list[dict[str, str]], str]:
    values: list[str] = []
    for value in [correct, *distractors]:
        if value and value not in values:
            values.append(value)
    if len(values) != 4:
        raise ValueError(f"Expected four unique options for {seed}, got {values}")
    correct_index = sum(ord(c) for c in seed) % 4
    ordered = values[1:]
    ordered.insert(correct_index, values[0])
    labels = list("ABCD")
    return ([{"label": label, "text": text} for label, text in zip(labels, ordered)], labels[correct_index])


def qa(
    *, task_id: str, task_name: str, qtype: str, question: str,
    correct: str, distractors: list[str], explanation: str,
    method: str, result: dict[str, Any], quality: str = "high",
) -> dict[str, Any]:
    opts, label = options(correct, distractors, qtype + question)
    result = copy.deepcopy(result)
    result.update({"answer_type": qtype, "T_Q": True, "H_Q": True, "S_Q": True})
    return {
        "task_id": task_id,
        "task_name": task_name,
        "question_type": qtype,
        "question": question,
        "options": opts,
        "correct_option": label,
        "correct_answer": correct,
        "answer": correct,
        "explanation": explanation,
        "status": "ok",
        "quality": quality,
        "method": method,
        "result_json": result,
    }


def source_qa(group: dict[str, Any], qtype: str) -> dict[str, Any]:
    return next(q for q in group.get("qa", []) if q.get("question_type") == qtype)


def task1_groups(ego: dict[str, Any]) -> list[dict[str, Any]]:
    by_name = {g["name"]: g for g in ego["groups"]}
    specs = []

    # General human-referenced relation change.
    group = copy.deepcopy(by_name["query_04_iiith145_frame11250"])
    old = source_qa(group, "relation_change_over_video")
    result = copy.deepcopy(old["result_json"])
    track = result["object_track"]
    sample_count = len(track["states"])
    q = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="relation_change_over_video",
        question="Across the 15-second clip, what is the large steel bowl's overall left/right transition relative to the person?",
        correct="It begins on the person's left and ends on the person's right.",
        distractors=[
            "It begins on the person's right and ends on the person's left.",
            "It stays on the person's left for the entire clip.",
            "It stays centered relative to the person for the entire clip.",
        ],
        explanation=f"Across {sample_count} valid 3D body-pose samples, the scene-fixed bowl is left at the first sample and right at the final sample in the person's body-centric frame.",
        method="Rebuilds the person's body-centric frame at every valid pose sample and transforms the fixed 3D object center into that changing frame.",
        result=result,
    )
    specs.append((group, q, "Task 1 · overall relation change"))

    # Rotation-dominant relation change.
    group = copy.deepcopy(by_name["query_02_iiith30_frame4440"])
    old = source_qa(group, "relation_change_over_video")
    result = copy.deepcopy(old["result_json"])
    motion = result["human_motion"]
    q = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="turn_induced_relation_change_over_video",
        question="The person turns substantially during the clip while moving little overall. How does the paprika container's lateral position relative to the person change?",
        correct="It changes from the person's right side to the person's left side.",
        distractors=[
            "It changes from the person's left side to the person's right side.",
            "It remains on the person's right side throughout.",
            "It remains on the person's left side throughout.",
        ],
        explanation=f"The body-centric relation changes right → left while the measured body turn is {motion['body_turn_deg']:.1f}° and net displacement is {motion['displacement_m']:.2f} m.",
        method="Uses all valid poses in the clip; the rotation-dominant label requires body turn ≥45° and net displacement <0.35 m.",
        result=result,
    )
    specs.append((group, q, "Task 1 · turn-induced relation change"))

    # Relation consistency requires every sampled instant.
    group = copy.deepcopy(by_name["query_01_iiith32_frame5280"])
    old = source_qa(group, "relation_consistency_over_video")
    result = copy.deepcopy(old["result_json"])
    relation_track = source_qa(group, "relation_change_over_video")["result_json"]["object_track"]
    result["sampled_frame_count"] = len(relation_track["states"])
    result["consistency_track"] = [
        {
            "frame": x["frame"],
            "time_s": x["t_sec_from_center"],
            "lateral_relation": x["relation"]["lateral_relation"],
        }
        for x in relation_track["states"]
    ]
    correct_obj = "stainless salt container"
    q = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="relation_consistency_over_video",
        question="Which listed object stays on the person's left side at every valid sampled time in the entire clip?",
        correct=f"The {correct_obj} stays on the person's left throughout.",
        distractors=[
            "The plastic bowl stays on the person's left throughout.",
            "The tawa pan stays on the person's left throughout.",
            "No listed object stays on the person's left throughout.",
        ],
        explanation=f"The {correct_obj} is classified left in all {result['sampled_frame_count']} valid body-centric pose samples.",
        method="Applies an all-samples consistency gate: a candidate is accepted only if its lateral label is left at every valid pose sample.",
        result=result,
    )
    specs.append((group, q, "Task 1 · relation consistency"))

    # Body-forward visibility change. This is deliberately not dense occlusion.
    group = copy.deepcopy(by_name["sfu0083_cam04_3450"])
    old = source_qa(group, "visibility_change_cause_over_video")
    result = copy.deepcopy(old["result_json"])
    result["visibility_scope"] = "body/head-forward FOV proxy plus listed-object blocker test; not gaze ground truth or dense ray casting"
    q = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="body_forward_visibility_change_cause_over_video",
        question="Near the end of the clip, the knife enters the person's central body/head-forward field. Which measured change best explains this?",
        correct="The person's body/head direction changes; no listed blocker is detected on the final sightline.",
        distractors=[
            "A listed object moves into the sightline and blocks the knife.",
            "The knife remains outside the person's forward field for the whole clip.",
            "The camera viewpoint alone determines the person's visibility state.",
        ],
        explanation="The first samples place the knife outside the forward field, the final four samples place it in the central zone, body turn is about 74.6°, and the listed-object blocker field remains empty.",
        method="Compares the target direction with the person's body/head-forward direction over the full pose timeline and checks only explicitly listed blocker geometry.",
        result=result,
        quality="audited_proxy",
    )
    specs.append((group, q, "Task 1 · body-forward visibility change"))

    out = []
    for group, question, title in specs:
        group["qa"] = [question]
        group["title"] = title + " · " + str(group.get("title") or group["name"])
        group["case_policy"] = "one temporal question per unique video window"
        out.append(group)
    return out


def compact_metric_timeline(timeline: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(timeline)
    for state in out.get("states", []):
        # Keep exact evidence vectors, but make the declared approximation explicit.
        state.setdefault("evidence", {}).setdefault("limitations", [
            "pelvis uses SMPL-X transl/root proxy",
            "body forward uses SMPL-X global orientation",
            "head uses pelvis + 1.6 m when fitted joints are not loaded",
        ])
    return out


def metric_group(scene: dict[str, Any], audit: dict[str, Any], question: dict[str, Any]) -> dict[str, Any]:
    sid = scene["scene_id"]
    if audit.get("status") != "complete_and_identity_aligned":
        raise ValueError(f"Metric group {sid} failed identity gate: {audit.get('status')}")
    seq = sid.removeprefix("hoi_m3_").rsplit("_win", 1)[0]
    win = sid.rsplit("_win", 1)[1]
    svg_scene = copy.deepcopy(scene)
    frames = svg_scene["frames"]
    svg_scene["frames"] = [frames[0], frames[len(frames)//2], frames[-1]]
    svg_scene["title"] = sid + " · start / middle / end metric tracks"
    svg_path = ROOT / "site/qa_benchmark/multihuman_media" / f"{sid}_curated.svg"
    topdown = write_svg(svg_scene, svg_path)
    return {
        "name": sid,
        "title": f"Task 4 · HOI-M3 · {seq} · window {win}",
        "dataset": "HOI-M3",
        "video_clip": f"./outputs/hoim3/{seq}/win{win}_view0_15s.mp4",
        "original_video": f"./hoim3_data/videos/{seq}/0.mp4",
        "localization_video": f"./multihuman_media/{sid}_localized.mp4",
        "localization_image": f"./multihuman_media/{sid}_localized.jpg",
        "original_image": f"./multihuman_media/{sid}_localized.jpg",
        "topdown_image": topdown,
        "video_window": {
            "start_sec": scene.get("start_sec"),
            "duration_sec": scene.get("duration_sec"),
            "metric_sample_count": len(scene["frames"]),
            "visual_sample_fps": audit.get("sample_fps"),
            "source": scene.get("source_video"),
        },
        "visual_person_audit": audit,
        "case_policy": "one temporal question per unique video window",
        "qa": [question],
    }


def task4_groups(
    dense: dict[str, Any], candidate_audits: dict[str, Any],
    existing_audits: dict[str, Any], base_site: dict[str, Any],
) -> list[dict[str, Any]]:
    scenes = {x["scene_id"]: x for x in dense["scenes"]}
    audits = {x["scene_id"]: x for x in [*candidate_audits["groups"], *existing_audits["groups"]]}
    out: list[dict[str, Any]] = []

    def timeline(sid: str) -> dict[str, Any]:
        return compact_metric_timeline(pair_timeline(scenes[sid]))

    # Position: must hold over the entire timeline, not only one endpoint.
    sid = "hoi_m3_bedroom_data02_win08"
    tl = timeline(sid)
    states = tl["states"]
    rel_counts = Counter(x["b_relative_to_a"] for x in states)
    correct = f"B stays right-and-in-front of A in all {len(states)} metric samples."
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="position_consistency_between_people",
        question="Which A-centered position statement remains true for B across the entire 15-second clip?",
        correct=correct,
        distractors=[
            "B stays left-and-in-front of A throughout.",
            "B crosses from A's right side to A's left side.",
            "B remains behind A throughout.",
        ],
        explanation=f"The dense metric timeline contains {len(states)} samples and every sample is right_front relative to A.",
        method="Uses A's ground-plane body-forward axis and right axis at every sampled time; all-samples consistency is required.",
        result={"scene_id": sid, "pair_timeline": tl, "relation_counts": dict(rel_counts), "visual_person_audit": audits[sid]},
    )
    out.append(metric_group(scenes[sid], audits[sid], q))

    # Orientation: dominant state across the whole clip.
    sid = "hoi_m3_bedroom_data01_win06"
    tl = timeline(sid)
    counts = Counter(x["facing_state"] for x in tl["states"])
    n = len(tl["states"])
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="dominant_facing_relation_over_video",
        question="What is the dominant body-facing relation between A and B over the full clip?",
        correct=f"They face each other in {counts['facing_each_other']} of {n} metric samples, so facing each other is dominant.",
        distractors=[
            "They are side-by-side or oblique at every sampled time.",
            "They remain back-to-back for most of the clip.",
            "No dominant orientation can be determined because only one frame is used.",
        ],
        explanation=f"Facing-score classification gives {dict(counts)} across the {n}-sample timeline.",
        method="Projects both SMPL-X root-forward vectors and the A↔B direction onto the ground plane, then aggregates the facing state over all samples.",
        result={"scene_id": sid, "pair_timeline": tl, "facing_counts": dict(counts), "visual_person_audit": audits[sid]},
    )
    out.append(metric_group(scenes[sid], audits[sid], q))

    # Distance: strong full-trajectory pattern.
    sid = "hoi_m3_bedroom_data03_win04"
    tl = timeline(sid)
    distances = [x["distance_m"] for x in tl["states"]]
    start, end = distances[0], distances[-1]
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="metric_distance_pattern_over_video",
        question="Which description best matches how the pelvis-to-pelvis distance between A and B evolves over the clip?",
        correct=f"They move substantially farther apart: about {start:.2f} m at the start, above 3 m in the second half, and {end:.2f} m at the end.",
        distractors=[
            "They steadily approach until they are less than 1 m apart at the end.",
            "Their distance stays within a narrow 0.1 m band throughout.",
            "They separate briefly but return to approximately their starting distance by the end.",
        ],
        explanation=f"Across {len(distances)} metric samples, distance ranges from {min(distances):.2f} m to {max(distances):.2f} m; net change is {end-start:+.2f} m.",
        method="Computes Euclidean distance between aligned SMPL-X pelvis/root translations at approximately 1 Hz across the full 15 seconds.",
        result={"scene_id": sid, "pair_timeline": tl, "distance_series_m": distances, "visual_person_audit": audits[sid]},
    )
    out.append(metric_group(scenes[sid], audits[sid], q))

    # Visibility: directional body-forward field, explicitly not occlusion.
    sid = "hoi_m3_bedroom_data02_win05"
    tl = timeline(sid)
    field_counts = Counter(x["body_forward_field"]["state"] for x in tl["states"])
    n = len(tl["states"])
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="body_forward_visibility_consistency",
        question="Ignoring physical occlusion, what body-forward visibility relation holds between A and B throughout the clip?",
        correct=f"Each person stays inside the other's ±60° body-forward field in all {n} metric samples.",
        distractors=[
            "Only A keeps B inside the body-forward field throughout.",
            "Only B keeps A inside the body-forward field throughout.",
            "Neither person contains the other in the body-forward field at any sampled time.",
        ],
        explanation=f"The directional field state is mutual_body_forward_field in all {n} samples. This does not claim gaze contact or an unoccluded sightline.",
        method="Measures the ground-plane angle from each SMPL-X body-forward vector to the other person's pelvis; accepts inside-field at ≤60°. Dense occlusion is deliberately not inferred.",
        result={"scene_id": sid, "pair_timeline": tl, "body_forward_field_counts": dict(field_counts), "occlusion_status": "not_evaluated_no_blocker_geometry", "visual_person_audit": audits[sid]},
        quality="audited_proxy",
    )
    out.append(metric_group(scenes[sid], audits[sid], q))

    # Relation change: robust side crossing sustained after the transition.
    sid = "hoi_m3_bedroom_data03_win03"
    tl = timeline(sid)
    relations = [x["b_relative_to_a"] for x in tl["states"]]
    left_prefix = sum(x.startswith("left_") for x in relations[:4])
    right_suffix = sum(x.startswith("right_") for x in relations[4:])
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="body_centric_relation_change_over_video",
        question="How does B's left/right relation in A's body-centric frame change and then persist over the clip?",
        correct=f"B is left-front for the first four samples, then moves to A's right-front side for the remaining twelve samples.",
        distractors=[
            "B starts right-front and then remains left-front for the rest of the clip.",
            "B alternates left and right at nearly every sampled second.",
            "B remains directly behind A throughout the clip.",
        ],
        explanation=f"The lateral sequence has {left_prefix}/4 left samples before the transition and {right_suffix}/12 right samples after it.",
        method="Transforms B's pelvis into A's ground-plane body frame at each of 16 samples and requires the new side to persist after the crossing.",
        result={"scene_id": sid, "pair_timeline": tl, "relation_sequence": relations, "visual_person_audit": audits[sid]},
    )
    out.append(metric_group(scenes[sid], audits[sid], q))

    # Multi-person topology uses all three visible 2D tracks and makes no metric identity claim.
    sid = "hoi_m3_bedroom_data05_win02"
    base_group = next(g for g in base_site["groups"] if g.get("name") == sid)
    old = next(
        q for q in base_group["qa"]
        if q.get("question_type") in {"closest_visible_pair_change_2d", "visible_pair_topology_change_2d"}
    )
    result = copy.deepcopy(old["result_json"])
    result["visual_person_audit"] = audits[sid]
    correct = old["correct_answer"]
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="visible_pair_topology_change_2d",
        question="How does the closest pair among the three persistently tracked people change from the start to the end of the clip?",
        correct=correct,
        distractors=[
            "V1–V2 is the closest pair at both the start and the end.",
            "V1–V3 is the closest pair at the start and V2–V3 at the end.",
            "The same pair remains closest at both endpoints.",
        ],
        explanation="All three persistent visible tracks are included; normalized image-plane pair distances change the closest pair from V2–V3 to V1–V3.",
        method="Uses Grounding DINO person detections plus temporal association for all three visible people; compares pairwise box-center separation normalized by frame diagonal. It does not claim metric 3D distance.",
        result=result,
        quality="high_2d_topology",
    )
    group = copy.deepcopy(base_group)
    group["qa"] = [q]
    group["visual_person_audit"] = audits[sid]
    group["qa"][0]["result_json"]["visual_person_audit"] = audits[sid]
    group["title"] = "Task 4 · HOI-M3 · three-person topology change"
    group["case_policy"] = "one temporal question per unique video window"
    out.append(group)

    return out



def task1_expansion_groups(expansion: dict[str, Any]) -> list[dict[str, Any]]:
    """Select six additional high-signal Ego-Exo4D temporal windows."""
    by_name = {g["name"]: copy.deepcopy(g) for g in expansion["groups"]}
    out: list[dict[str, Any]] = []

    def relation_data(group: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        relation_q = source_qa(group, "relation_change_over_video")
        consistency_q = source_qa(group, "relation_consistency_over_video")
        return copy.deepcopy(relation_q["result_json"]), copy.deepcopy(consistency_q["result_json"]["object_tracks"])

    def nearest_sequence(tracks: list[dict[str, Any]]) -> list[str]:
        n = min(len(track["states"]) for track in tracks)
        return [
            min(tracks, key=lambda track: track["states"][i]["relation"]["distance_m"])["object_id"]
            for i in range(n)
        ]

    def finish(group: dict[str, Any], question: dict[str, Any], title: str) -> None:
        group["qa"] = [question]
        group["title"] = title + " · " + str(group.get("title") or group["name"])
        group["case_policy"] = "one temporal question per unique video window"
        out.append(group)

    group = by_name["diverse_iiith_145_2_frame7380"]
    result, tracks = relation_data(group)
    sequence = nearest_sequence(tracks)
    result.update({"nearest_sequence": sequence, "sampled_frame_count": len(sequence)})
    question = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="nearest_object_consistency_over_video",
        question="Which listed object remains nearest to the person at every valid sampled time in this clip?",
        correct=f"The white chopping board remains nearest in all {len(sequence)} samples.",
        distractors=[
            "The steel tomato bowl remains nearest throughout.",
            "The small steel bowl remains nearest throughout.",
            "The nearest listed object changes during the clip.",
        ],
        explanation=f"Per-sample pelvis-to-object 3D distances select the white chopping board {Counter(sequence)['White chopping board_0']}/{len(sequence)} times.",
        method="Ranks every listed scene-fixed 3D object by distance from the changing human pelvis at every valid pose sample; all-samples agreement is required.",
        result=result,
    )
    finish(group, question, "Task 1 · nearest-object consistency")

    group = by_name["query_03_iiith29_frame4350"]
    result, tracks = relation_data(group)
    n = min(len(track["states"]) for track in tracks)
    front_counts = {
        track["object_id"]: sum(state["relation"]["longitudinal_relation"] == "front" for state in track["states"])
        for track in tracks
    }
    result.update({"front_counts": front_counts, "sampled_frame_count": n, "object_tracks": tracks})
    question = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="multi_object_front_consistency_over_video",
        question="Which statement about the three listed objects remains true across every valid time sample?",
        correct=f"The steel bowl, oil container and steel plate all remain in front of the person in all {n} samples.",
        distractors=[
            "All three objects remain behind the person throughout.",
            "Only the oil container remains in front throughout.",
            "At least one listed object crosses from front to behind during the clip.",
        ],
        explanation=f"The front counts are {front_counts}; each equals the {n}-sample timeline length.",
        method="Checks the longitudinal body-centric relation of every listed object at every valid human pose, not only the endpoints.",
        result=result,
    )
    finish(group, question, "Task 1 · multi-object relation consistency")

    group = by_name["diverse_iiith_31_3_frame1620"]
    result, tracks = relation_data(group)
    sequence = nearest_sequence(tracks)
    tomato = next(track for track in tracks if track["object_id"] == "Chopped tomato_0")
    below_count = sum(state["relation"]["vertical_relation"] == "below" for state in tomato["states"])
    result.update({"nearest_sequence": sequence, "below_count": below_count, "sampled_frame_count": len(sequence), "object_tracks": tracks})
    question = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="nearest_and_vertical_consistency_over_video",
        question="Which object stays both nearest to the person and below the person's body origin throughout the clip?",
        correct=f"The chopped tomato is nearest and below the body origin in all {len(sequence)} samples.",
        distractors=[
            "The chopping board is nearest and below throughout.",
            "The knife is nearest and below throughout.",
            "No object satisfies both conditions for the whole clip.",
        ],
        explanation=f"The chopped tomato wins all {len(sequence)} distance rankings and has below in {below_count}/{len(sequence)} vertical-relation samples.",
        method="Combines the all-samples nearest-distance ranking with the all-samples vertical body-centric relation.",
        result=result,
    )
    finish(group, question, "Task 1 · nearest plus vertical consistency")

    group = by_name["sfu0101_cam05_5460"]
    result, _ = relation_data(group)
    track = result["object_track"]
    n = len(track["states"])
    result["sampled_frame_count"] = n
    question = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="relation_change_over_video",
        question="As the person moves across the scene, how does the white plate's lateral relation to the person change over the clip?",
        correct="The white plate changes from the person's left side to the person's right side.",
        distractors=[
            "It changes from the person's right side to the person's left side.",
            "It remains on the person's left throughout.",
            "It remains centered relative to the person throughout.",
        ],
        explanation=f"The first body-centric sample is left and the final sample is right; {n} valid poses cover the 15-second window.",
        method="Transforms the fixed white-plate 3D center into the person's changing body-centric frame across the full timeline.",
        result=result,
    )
    finish(group, question, "Task 1 · lateral relation change")

    group = by_name["val_3"]
    result, _ = relation_data(group)
    track = result["object_track"]
    n = len(track["states"])
    result["sampled_frame_count"] = n
    question = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="relation_change_over_video",
        question="How does the oyster-sauce bottle change relative to the moving person from the beginning to the end?",
        correct="It begins behind the person and ends on the person's right-and-front side.",
        distractors=[
            "It begins on the person's left and ends behind the person.",
            "It remains behind the person throughout.",
            "It remains on the person's left throughout.",
        ],
        explanation=f"Across {n} valid poses, the corrected body-centric labels change from right-behind at the first sample to right-front at the final sample.",
        method="Uses the full body-frame timeline, the validated scene-fixed bottle center, and the take-level audited forward-sign calibration at every sample.",
        result=result,
    )
    finish(group, question, "Task 1 · longitudinal relation change")

    group = by_name["val_12_replacement_egg_whisk"]
    result, _ = relation_data(group)
    track = result["object_track"]
    distances = [state["relation"]["distance_m"] for state in track["states"]]
    result.update({"distance_series_m": distances, "sampled_frame_count": len(distances)})
    question = qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="human_object_distance_pattern_over_video",
        question="How does the person's 3D distance to the egg whisk change over this clip?",
        correct=f"The egg whisk is very close to the person at the beginning and about {distances[-1]:.2f} m away at the end.",
        distractors=[
            "The distance stays nearly constant throughout.",
            "The person moves steadily closer and ends within 0.3 m.",
            "The distance increases briefly but returns to its starting value by the end.",
        ],
        explanation=f"The pelvis-to-object Euclidean distance changes by {distances[-1]-distances[0]:+.2f} m across {len(distances)} valid pose samples.",
        method="Computes Euclidean distance from the human pelvis origin to the fixed 3D object center at every valid pose sample.",
        result=result,
    )
    finish(group, question, "Task 1 · human-object distance change")
    return out


def task4_expansion_groups(dense: dict[str, Any], expansion_audits: dict[str, Any]) -> list[dict[str, Any]]:
    """Add six distinct, identity-gated HOI-M3 temporal windows."""
    scenes = {x["scene_id"]: x for x in dense["scenes"]}
    audits = {x["scene_id"]: x for x in expansion_audits["groups"]}
    out: list[dict[str, Any]] = []

    def timeline(sid: str) -> dict[str, Any]:
        return compact_metric_timeline(pair_timeline(scenes[sid]))

    def add_metric(sid: str, question: dict[str, Any]) -> None:
        out.append(metric_group(scenes[sid], audits[sid], question))

    sid = "hoi_m3_bedroom_data01_win02"
    tl = timeline(sid); distances = [x["distance_m"] for x in tl["states"]]
    question = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="metric_separation_over_video",
        question="After a short early fluctuation, what sustained distance trend develops between A and B?",
        correct=f"They separate and remain far apart: distance grows from {distances[0]:.2f} m to about {distances[-1]:.2f} m.",
        distractors=[
            "They steadily converge and finish less than 1 m apart.",
            "They remain within a narrow 0.1 m distance band.",
            "They separate briefly but return to the starting distance by the end.",
        ],
        explanation=f"The 16-sample series rises above 3 m from t≈7 s onward and ends at {distances[-1]:.2f} m; net change is {distances[-1]-distances[0]:+.2f} m.",
        method="Uses all 16 aligned SMPL-X pelvis/root samples and checks that the late high-distance state persists rather than relying only on two frames.",
        result={"scene_id": sid, "pair_timeline": tl, "distance_series_m": distances, "visual_person_audit": audits[sid]},
    )
    add_metric(sid, question)

    sid = "hoi_m3_bedroom_data01_win04"
    tl = timeline(sid); relations = [x["b_relative_to_a"] for x in tl["states"]]
    counts = Counter(relations)
    question = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="dominant_body_centric_position",
        question="Which body-centric position of B relative to A dominates this clip despite a few brief deviations?",
        correct=f"B is on A's right-and-behind side in {counts['right_behind']} of 16 samples, so right-behind dominates.",
        distractors=[
            "Left-front dominates the clip.",
            "Right-front dominates the clip.",
            "The four position quadrants occur equally often.",
        ],
        explanation=f"The complete position counts are {dict(counts)}; right_behind is the clear temporal majority.",
        method="Transforms B into A's body-centric frame at each sampled second and aggregates the full position sequence.",
        result={"scene_id": sid, "pair_timeline": tl, "relation_counts": dict(counts), "visual_person_audit": audits[sid]},
    )
    add_metric(sid, question)

    sid = "hoi_m3_bedroom_data02_win04"
    tl = timeline(sid); distances = [x["distance_m"] for x in tl["states"]]
    peak_i = max(range(len(distances)), key=distances.__getitem__)
    question = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="nonmonotonic_distance_pattern",
        question="Which full-clip distance pattern best describes A and B?",
        correct=f"They first separate from about {distances[0]:.2f} m to {distances[peak_i]:.2f} m, then partially approach and finish near {distances[-1]:.2f} m.",
        distractors=[
            "They approach continuously for the entire clip.",
            "They separate continuously with no later approach.",
            "Their distance is constant throughout.",
        ],
        explanation=f"The maximum occurs near t={tl['states'][peak_i]['t']:.1f} s, so start/end alone would miss the out-then-partway-back pattern.",
        method="Finds the peak and endpoint over all 16 metric pelvis-distance samples; this question requires the middle of the video.",
        result={"scene_id": sid, "pair_timeline": tl, "distance_series_m": distances, "peak_sample_index": peak_i, "visual_person_audit": audits[sid]},
    )
    add_metric(sid, question)

    sid = "hoi_m3_bedroom_data02_win06"
    tl = timeline(sid); distances = [x["distance_m"] for x in tl["states"]]
    counts = Counter(x["facing_state"] for x in tl["states"]); min_i = min(range(len(distances)), key=distances.__getitem__)
    question = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="approach_while_facing",
        question="What combined distance-and-orientation pattern occurs between A and B?",
        correct=f"They approach to about {distances[min_i]:.2f} m while facing each other is dominant ({counts['facing_each_other']}/16 samples).",
        distractors=[
            "They move farther apart while remaining back-to-back throughout.",
            "Their distance stays constant and their facing relation cannot be determined.",
            "They approach, but side-by-side orientation dominates all 16 samples.",
        ],
        explanation=f"The minimum distance occurs near t={tl['states'][min_i]['t']:.1f} s, and the facing-state aggregate is {dict(counts)}.",
        method="Jointly aggregates the 16-sample pelvis-distance series and the ground-plane SMPL-X body-facing score.",
        result={"scene_id": sid, "pair_timeline": tl, "distance_series_m": distances, "facing_counts": dict(counts), "visual_person_audit": audits[sid]},
    )
    add_metric(sid, question)

    sid = "hoi_m3_bedroom_data03_win09"
    tl = timeline(sid); distances = [x["distance_m"] for x in tl["states"]]; relations = [x["b_relative_to_a"] for x in tl["states"]]
    question = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="coupled_distance_relation_change",
        question="As B approaches A during the clip, how does B's lateral relation in A's frame change?",
        correct=f"B approaches from about {distances[0]:.2f} m to {distances[-1]:.2f} m and crosses from A's right side to A's left side.",
        distractors=[
            "B moves farther away while crossing from left to right.",
            "B approaches but remains on A's right side throughout.",
            "B stays at the same distance and directly behind A throughout.",
        ],
        explanation="The first seven samples are right-front; later samples become center/left-front while the metric distance drops by more than 1.5 m.",
        method="Combines all-sample metric distance with B's per-sample position in A's body-centric frame.",
        result={"scene_id": sid, "pair_timeline": tl, "distance_series_m": distances, "relation_sequence": relations, "visual_person_audit": audits[sid]},
    )
    add_metric(sid, question)

    sid = "hoi_m3_bedroom_data03_win08"
    tl = timeline(sid); distances = [x["distance_m"] for x in tl["states"]]; max_i = max(range(len(distances)), key=distances.__getitem__)
    question = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="distance_out_and_back_over_video",
        question="Which temporal distance pattern occurs between A and B across this clip?",
        correct=f"They are close early, move apart to about {distances[max_i]:.2f} m near the middle, then come close again at about {distances[-1]:.2f} m.",
        distractors=[
            "They move steadily farther apart from start to finish.",
            "They move steadily closer from start to finish.",
            "Their distance remains almost unchanged throughout.",
        ],
        explanation=f"The 16-sample series has an interior maximum near t={tl['states'][max_i]['t']:.1f} s and low values at both the early and final phases.",
        method="Uses the full metric distance curve and requires a middle maximum with lower distances on both sides.",
        result={"scene_id": sid, "pair_timeline": tl, "distance_series_m": distances, "peak_sample_index": max_i, "visual_person_audit": audits[sid]},
    )
    add_metric(sid, question)
    return out

def validate(data: dict[str, Any]) -> dict[str, Any]:
    groups = data["groups"]
    errors: list[str] = []
    if any(len(g.get("qa", [])) != 1 for g in groups):
        errors.append("every case must contain exactly one question")
    names = [g.get("name") for g in groups]
    if len(names) != len(set(names)):
        errors.append("duplicate case/window names")
    qas = [g["qa"][0] for g in groups]
    if any(q.get("task_id") not in {TASK1_ID, TASK4_ID} for q in qas):
        errors.append("non Task1/Task4 question present")
    if any(len(q.get("options", [])) != 4 or len({x["text"] for x in q["options"]}) != 4 for q in qas):
        errors.append("all questions need four unique options")
    for g in groups:
        q = g["qa"][0]
        if q["task_id"] == TASK4_ID and q["question_type"] != "visible_pair_topology_change_2d":
            audit = g.get("visual_person_audit") or {}
            if audit.get("status") != "complete_and_identity_aligned":
                errors.append(f"metric Task4 identity gate failed: {g['name']}")
            if len((q.get("result_json") or {}).get("pair_timeline", {}).get("states", [])) != 16:
                errors.append(f"metric Task4 must use 16 samples: {g['name']}")
    if errors:
        raise ValueError("; ".join(errors))
    type_counts = Counter(q["question_type"] for q in qas)
    task_counts = Counter(q["task_id"] for q in qas)
    return {
        "status": "ok",
        "case_count": len(groups),
        "qa_count": len(qas),
        "one_question_per_case": True,
        "unique_case_windows": len(names),
        "task_counts": dict(task_counts),
        "question_type_counts": dict(type_counts),
        "hard_gates": [
            "Task 1 answers come from multi-frame 3D body/object timelines",
            "metric Task 4 requires complete_and_identity_aligned visual-person audit",
            "metric Task 4 uses 16 pose samples over 15 seconds",
            "three-person topology includes all three visible tracks and remains explicitly 2D",
            "missing blocker geometry produces unknown, never clear line of sight",
            "four unique options and exactly one question per case",
        ],
        "honest_limitations": [
            "Task 1 visibility and Task 4 body-forward visibility are directional FOV proxies, not gaze ground truth",
            "no physical occlusion/partition QA is released because the local subset lacks blocker geometry",
            "SMPL-X transl/global_orient are pelvis/root and body-forward proxies; fitted head joints are not loaded",
            "three-person topology is image-plane topology, not metric 3D distance",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ego-data", type=Path, default=Path("outputs/qa/ego_new_data.js"))
    ap.add_argument("--dense-scenes", type=Path, default=Path("outputs/qa/hoim3_multihuman_scenes_dense_all.json"))
    ap.add_argument("--candidate-audits", type=Path, default=Path("outputs/qa/multihuman_curated_candidate_calibration.json"))
    ap.add_argument("--existing-audits", type=Path, default=Path("outputs/qa/multihuman_visual_calibration.json"))
    ap.add_argument("--ego-expansion-data", type=Path, default=Path("outputs/qa/ego_temporal_expansion_candidates.js"))
    ap.add_argument("--expansion-audits", type=Path, default=Path("outputs/qa/multihuman_expansion_candidate_calibration.json"))
    ap.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    ap.add_argument("--output-jsonl", type=Path, default=Path("outputs/qa/task1_task4_curated_qa.jsonl"))
    ap.add_argument("--audit-output", type=Path, default=Path("outputs/qa/task1_task4_curated_audit.json"))
    args = ap.parse_args()

    resolve = lambda p: p if p.is_absolute() else ROOT / p
    ego = load_js(resolve(args.ego_data))
    dense = json.loads(resolve(args.dense_scenes).read_text(encoding="utf-8"))
    candidate_audits = json.loads(resolve(args.candidate_audits).read_text(encoding="utf-8"))
    existing_audits = json.loads(resolve(args.existing_audits).read_text(encoding="utf-8"))
    ego_expansion = load_js(resolve(args.ego_expansion_data))
    expansion_audits = json.loads(resolve(args.expansion_audits).read_text(encoding="utf-8"))
    base_site = load_js(resolve(args.site_data))

    core_task1 = task1_groups(ego)
    extra_task1 = task1_expansion_groups(ego_expansion)
    core_task4 = task4_groups(dense, candidate_audits, existing_audits, base_site)
    extra_task4 = task4_expansion_groups(dense, expansion_audits)
    data = {
        "title": "Humans in Space · Task 1 + Task 4 Curated QA",
        "subtitle": "One evidence-grounded temporal question per unique case.",
        "tasks": TASKS,
        "groups": [*core_task1, *extra_task1, *core_task4, *extra_task4],
        "release_policy": {
            "task_scope": [TASK1_ID, TASK4_ID],
            "one_question_per_case": True,
            "no_guessed_answers": True,
        },
    }
    audit = validate(data)
    output_js = resolve(args.site_data)
    save_js(output_js, data)
    jsonl = resolve(args.output_jsonl)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for case_index, group in enumerate(data["groups"], 1):
        rows.append({
            "case_index": case_index,
            "case_id": group["name"],
            "video_clip": group.get("video_clip"),
            **group["qa"][0],
        })
    jsonl.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    audit_path = resolve(args.audit_output)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
