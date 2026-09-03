#!/usr/bin/env python3
"""Build the scale-ready Task 1 / Task 4 release from curated candidates.

This release layer:
- makes question capability categories explicit;
- requires at least two independent cases per category;
- adds only candidates that pass temporal and identity/coverage gates;
- rewrites ambiguous human-relative wording before publication.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from decimal import Decimal, ROUND_HALF_UP
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_task1_task4_curated import (
    TASK1_ID, TASK1_NAME, TASK4_ID, TASK4_NAME, compact_metric_timeline,
    load_js, metric_group, options, qa, save_js, source_qa,
)
from limo4si.multihuman import pair_timeline
from limo4si.scale_quality import ScaleQualityPolicy, require_release_quality, validate_release


REQUIRED_CATEGORIES = {
    TASK1_ID: {
        "spatial_relation_change",
        "relation_consistency",
        "body_forward_visibility",
        "metric_distance_change",
    },
    TASK4_ID: {
        "body_centric_position",
        "body_orientation",
        "metric_distance",
        "body_forward_visibility",
        "relation_change",
        "image_plane_topology",
    },
}


def categories(group: dict[str, Any], question: dict[str, Any]) -> list[str]:
    qtype = question["question_type"]
    if question["task_id"] == TASK1_ID:
        mapping = {
            "relation_change_over_video": ["spatial_relation_change"],
            "turn_induced_relation_change_over_video": ["spatial_relation_change"],
            "relation_consistency_over_video": ["relation_consistency"],
            "nearest_object_consistency_over_video": ["relation_consistency"],
            "multi_object_front_consistency_over_video": ["relation_consistency"],
            "nearest_and_vertical_consistency_over_video": ["relation_consistency"],
            "body_forward_visibility_change_cause_over_video": ["body_forward_visibility"],
            "human_object_distance_pattern_over_video": ["metric_distance_change"],
        }
        out = list(mapping[qtype])
        if group["name"] == "sfu0101_cam05_5460":
            out.append("metric_distance_change")
        return out
    mapping = {
        "position_consistency_between_people": ["body_centric_position"],
        "dominant_body_centric_position": ["body_centric_position"],
        "dominant_facing_relation_over_video": ["body_orientation"],
        "approach_while_facing": ["body_orientation", "metric_distance"],
        "metric_distance_pattern_over_video": ["metric_distance"],
        "metric_separation_over_video": ["metric_distance"],
        "nonmonotonic_distance_pattern": ["metric_distance"],
        "distance_out_and_back_over_video": ["metric_distance"],
        "body_forward_visibility_consistency": ["body_forward_visibility"],
        "body_forward_field_transition_over_video": ["body_forward_visibility"],
        "body_centric_relation_change_over_video": ["relation_change"],
        "coupled_distance_relation_change": ["relation_change", "metric_distance"],
        "visible_pair_topology_change_2d": ["image_plane_topology"],
        "visible_pair_topology_consistency_2d": ["image_plane_topology"],
    }
    return mapping[qtype]


def rewrite_case1(group: dict[str, Any]) -> None:
    old = group["qa"][0]
    result = copy.deepcopy(old["result_json"])
    group["qa"] = [qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="relation_change_over_video",
        question="The large steel bowl stays fixed in the scene while the person moves and turns. How does the bowl's left/right relation in the person's changing body frame differ between the start and end?",
        correct="It is on the person's left at the start and on the person's right at the end; this is a change in human-relative coordinates, not bowl motion.",
        distractors=[
            "The bowl itself moves from the person's right side to the left side.",
            "It remains on the person's left in the body frame for the entire clip.",
            "It remains centered in the person's body frame at both endpoints.",
        ],
        explanation=old["explanation"] + " The object world coordinate is held fixed; only the human origin/orientation changes.",
        method=old["method"],
        result=result,
    )]


def rewrite_joint_relation_distance(group: dict[str, Any]) -> None:
    old = group["qa"][0]
    result = copy.deepcopy(old["result_json"])
    states = result["object_track"]["states"]
    start = states[0]["relation"]["distance_m"]
    end = states[-1]["relation"]["distance_m"]
    result["distance_series_m"] = [x["relation"]["distance_m"] for x in states]
    group["qa"] = [qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="relation_change_over_video",
        question="As the person approaches the scene-fixed white plate, what combined distance and left/right change occurs?",
        correct=f"The pelvis-to-plate distance decreases from about {start:.2f} m to {end:.2f} m, while the plate changes from the person's left to the person's right.",
        distractors=[
            "The distance increases while the plate changes from right to left.",
            "The distance decreases, but the plate stays on the person's left throughout.",
            "The distance stays nearly constant and the plate remains centered.",
        ],
        explanation=f"Across {len(states)} valid body poses, distance changes by {end-start:+.2f} m and the endpoint lateral labels are left then right.",
        method="Uses the same fixed 3D plate center with every human pelvis/body-frame sample; jointly checks Euclidean distance and body-centric lateral relation.",
        result=result,
    )]


def add_task1_visibility(data: dict[str, Any], ego: dict[str, Any]) -> None:
    name = "diverse_sfu_008_3_frame6960"
    if any(g["name"] == name for g in data["groups"]):
        return
    source = copy.deepcopy(next(g for g in ego["groups"] if g["name"] == name))
    old = source_qa(source, "visibility_change_cause_over_video")
    result = copy.deepcopy(old["result_json"])
    states = result["visibility_track"]["states"]
    source["qa"] = [qa(
        task_id=TASK1_ID, task_name=TASK1_NAME,
        qtype="body_forward_visibility_change_cause_over_video",
        question="The scene-fixed cucumber peel begins inside the person's peripheral body-forward field and ends outside it. Which measured human change explains this transition?",
        correct="The person's body orientation changes substantially, rotating the body-forward field away from the scene-fixed peel.",
        distractors=[
            "The cucumber peel moves behind a measured physical blocker.",
            "The peel stays in the central body-forward field throughout.",
            "Only the external camera viewpoint changes the human-relative field label.",
        ],
        explanation=f"The endpoint forward angles are {states[0]['angle_deg']:.1f}° and {states[-1]['angle_deg']:.1f}°, while the human body-turn estimate is about {result['human_motion']['body_turn_deg']:.1f}°.",
        method="Transforms the fixed 3D target direction into each valid human body frame. This is a body-forward directional proxy, not gaze or physical-occlusion truth.",
        result=result,
        quality="audited_proxy",
    )]
    source["title"] = "Task 1 · second body-forward visibility example · " + str(source.get("title") or name)
    source["case_policy"] = "one temporal question per unique video window"
    data["groups"].insert(10, source)


def add_task4_visibility(data: dict[str, Any], dense: dict[str, Any], audits: dict[str, Any]) -> None:
    sid = "hoi_m3_bedroom_data01_win01"
    if any(g["name"] == sid for g in data["groups"]):
        return
    scene = next(x for x in dense["scenes"] if x["scene_id"] == sid)
    audit = next(x for x in audits["groups"] if x["scene_id"] == sid)
    timeline = compact_metric_timeline(pair_timeline(scene))
    sequence = [x["body_forward_field"]["state"] for x in timeline["states"]]
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="body_forward_field_transition_over_video",
        question="Ignoring physical occlusion, how does the one-sided body-forward field relation change over this clip?",
        correct="Early, only B keeps A inside the ±60° body-forward field; later, only A keeps B inside the field for most samples.",
        distractors=[
            "The relation stays mutual for all 16 samples.",
            "Neither person enters the other's body-forward field at any time.",
            "Only A contains B early, then only B contains A later.",
        ],
        explanation=f"The 16-sample state sequence is {sequence}; the first six are B-only and the final segment is predominantly A-only.",
        method="Uses ground-plane SMPL-X root-forward directions at all 16 samples. It does not claim eye gaze or an unobstructed physical sightline.",
        result={"scene_id": sid, "pair_timeline": timeline, "body_forward_field_sequence": sequence, "visual_person_audit": audit},
        quality="audited_proxy",
    )
    data["groups"].append(metric_group(scene, audit, q))


def add_task4_topology(data: dict[str, Any], evidence: dict[str, Any]) -> None:
    sid = evidence["scene_id"]
    if any(g["name"] == sid for g in data["groups"]):
        return
    pairs = evidence["pairs"]
    closest_start = min(pairs, key=lambda x: x["start"])["pair"]
    closest_end = min(pairs, key=lambda x: x["end"])["pair"]
    assert closest_start == closest_end == "V2–V3"
    audit = {
        "scene_id": sid,
        "status": "complete_visible_2d_tracks",
        "detector": "IDEA Research Grounding DINO tiny (local weights)",
        "tracking": "motion + box overlap + HSV appearance Hungarian association",
        "sample_fps": 2.0,
        "sample_count": evidence["sample_count"],
        "persistent_visible_person_count": 3,
        "metric_3d_track_count": 0,
        "geometry_scope": "all three visible people have endpoint-valid 2D tracks; no metric 3D claim",
        "visible_2d_tracks": evidence["visible_tracks"],
        "endpoint_gate": evidence["endpoint_gate"],
    }
    q = qa(
        task_id=TASK4_ID, task_name=TASK4_NAME,
        qtype="visible_pair_topology_consistency_2d",
        question="Among the three endpoint-valid visible tracks, which pair is closest in the camera plane at both the start and the end?",
        correct="V2–V3 is the closest visible pair at both endpoints.",
        distractors=[
            "V1–V2 is closest at both endpoints.",
            "V1–V3 is closest at both endpoints.",
            "The closest pair changes from V1–V2 to V2–V3.",
        ],
        explanation=f"Normalized start distances are {[x['start'] for x in pairs]} and end distances are {[x['end'] for x in pairs]} in pair order {[x['pair'] for x in pairs]}.",
        method="Detects people at 2 Hz, associates all three tracks across the clip, requires a real observation within 0.55 s of both endpoints, then compares box-center separation normalized by frame diagonal.",
        result={
            "scene_id": sid,
            "start_pair_distances_normalized": [{"pair": x["pair"], "distance": x["start"]} for x in pairs],
            "end_pair_distances_normalized": [{"pair": x["pair"], "distance": x["end"]} for x in pairs],
            "visual_person_audit": audit,
            "T_Q": True, "H_Q": True, "S_Q": True,
        },
        quality="high_2d_topology",
    )
    data["groups"].append({
        "name": sid,
        "title": "Task 4 · HOI-M3 · second three-person 2D topology example",
        "video_clip": "./outputs/hoim3/bedroom_data05/win_extra_90s_view0_15s.mp4",
        "original_image": "./multihuman_media/hoi_m3_bedroom_data05_win_extra_90s_localized.jpg",
        "localization_video": "./multihuman_media/hoi_m3_bedroom_data05_win_extra_90s_localized.mp4",
        "duration_sec": evidence["duration_sec"],
        "visual_person_audit": audit,
        "qa": [q],
        "case_policy": "one temporal question per unique video window",
    })



PERSON_ALIAS_PATH = ROOT / "configs/person_display_aliases.json"
PERSON_ALIAS_CONFIG = json.loads(PERSON_ALIAS_PATH.read_text(encoding="utf-8"))


def visible_person_descriptors(group: dict[str, Any]) -> dict[str, str]:
    name = str(group.get("name", ""))
    if name in PERSON_ALIAS_CONFIG:
        return dict(PERSON_ALIAS_CONFIG[name])
    sequence = name.removeprefix("hoi_m3_").split("_win", 1)[0]
    return dict(PERSON_ALIAS_CONFIG.get(sequence) or {})


def _replace_person_ids(text: str, aliases: dict[str, str]) -> str:
    """Replace display IDs token-safely; metric IDs remain intact in result_json."""
    centered = re.fullmatch(r"Which (A|B)-centered position statement remains true for (A|B) across (.+)\?", text)
    if centered and centered.group(1) in aliases and centered.group(2) in aliases:
        text = f"Which statement about {centered.group(2)}, expressed in {centered.group(1)}'s body-centered frame, remains true across {centered.group(3)}?"
    malformed = re.fullmatch(r"Which position statement remains true for (.+) across (.+)\?, centered on (.+)", text)
    if malformed:
        text = f"Which statement about {malformed.group(1)}, expressed in {malformed.group(3)}'s body-centered frame, remains true across {malformed.group(2)}?"
    for left, right in re.findall(r"\b(A|B|V1|V2|V3)–(A|B|V1|V2|V3)\b", text):
        if left in aliases and right in aliases:
            text = text.replace(f"{left}–{right}", f"{aliases[left]} and {aliases[right]}")
    for source in sorted(aliases, key=len, reverse=True):
        shown = aliases[source]
        text = re.sub(rf"\b{re.escape(source)}'s\b", shown + "'s", text)
        text = re.sub(rf"\b{re.escape(source)}\b", shown, text)
    # Normalize pair names even when rebuilding an already generated release.
    text = re.sub(r"(the [a-z-]+(?: [a-z-]+)* (?:man|woman))–(the [a-z-]+(?: [a-z-]+)* (?:man|woman))", r"\1 and \2", text)
    text = text.replace(" man–the ", " man and the ").replace(" woman–the ", " woman and the ")
    if text.startswith("the "):
        text = "The " + text[4:]
    return text


def apply_person_descriptions(data: dict[str, Any]) -> None:
    """Resolve per-clip metric identities to stable, visually audited descriptions."""
    display_fields = ("question", "correct_answer", "explanation", "method")
    for group in data["groups"]:
        if not group.get("qa") or group["qa"][0].get("task_id") != TASK4_ID:
            continue
        alignment = (group.get("visual_person_audit") or {}).get("metric_identity_alignment") or {}
        metric_to_visible = alignment.get("mapping") or {}
        descriptors = visible_person_descriptors(group)
        required_metric_ids = set(metric_to_visible)
        required_visible_ids = set(metric_to_visible.values())
        metric_configured = required_metric_ids and required_metric_ids.issubset(descriptors)
        visible_configured = required_visible_ids.issubset(descriptors)
        if not descriptors or not (metric_configured or visible_configured):
            group["person_display_alias_status"] = {
                "status": "missing",
                "required_metric_ids": sorted(required_metric_ids),
                "required_visible_ids": sorted(required_visible_ids),
                "configured_ids": sorted(descriptors),
            }
            continue
        aliases = dict(descriptors)
        for metric_id, visible_id in metric_to_visible.items():
            if metric_configured:
                aliases[visible_id] = descriptors[metric_id]
            else:
                aliases[metric_id] = descriptors[visible_id]
        group["person_display_aliases"] = aliases
        group["person_display_alias_status"] = {"status": "complete", "source": str(PERSON_ALIAS_PATH.relative_to(ROOT))}
        for question in group.get("qa", []):
            for field in display_fields:
                if isinstance(question.get(field), str):
                    question[field] = _replace_person_ids(question[field], aliases)
            for option in question.get("options", []):
                option["text"] = _replace_person_ids(option["text"], aliases)



MULTIHUMAN_GENERIC_QUESTIONS = {
    "position_consistency_between_people": "Which body-centered position relation remains true between the two people in the video throughout the clip?",
    "dominant_facing_relation_over_video": "What is the dominant body-facing relation between the two people in the video?",
    "metric_distance_pattern_over_video": "How does the distance between the two people in the video evolve over the clip?",
    "body_forward_visibility_consistency": "Ignoring physical occlusion, what body-forward visibility relation holds between the two people in the video throughout the clip?",
    "body_centric_relation_change_over_video": "How does the left/right relation between the two people in the video change in a body-centric frame?",
    "metric_separation_over_video": "After a short early fluctuation, what sustained distance trend develops between the two people in the video?",
    "dominant_body_centric_position": "Which body-centric position relation between the two people in the video dominates the clip?",
    "nonmonotonic_distance_pattern": "Which full-clip distance pattern best describes the two people in the video?",
    "approach_while_facing": "What combined distance-and-orientation pattern occurs between the two people in the video?",
    "coupled_distance_relation_change": "As the two people in the video approach each other, how does their lateral relation change in a body-centric frame?",
    "distance_out_and_back_over_video": "Which temporal distance pattern occurs between the two people in the video?",
    "body_forward_field_transition_over_video": "Ignoring physical occlusion, how does the one-sided body-forward field relation between the two people in the video change?",
}

def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _set_options(group: dict[str, Any], correct: str, distractors: list[str]) -> None:
    question = group["qa"][0]
    opts, label = options(correct, distractors, group["name"] + question["question_type"])
    question["options"] = opts
    question["correct_option"] = label
    question["correct_answer"] = correct
    question["answer"] = correct


def _person(group: dict[str, Any], person_id: str) -> str:
    aliases = group.get("person_display_aliases") or {}
    return aliases.get(person_id, person_id)


def _relation_words(value: str) -> str:
    return value.replace("_", "-")


def _pair_name(group: dict[str, Any], pair: str) -> str:
    ids = re.split(r"[–-]", pair)
    return " and ".join(_person(group, value) for value in ids)


def _distance_values(question: dict[str, Any]) -> list[float]:
    result = question.get("result_json") or {}
    values = result.get("distance_series_m")
    if values:
        return [float(value) for value in values]
    return [float(state["distance_m"]) for state in (result.get("pair_timeline") or {}).get("states", [])]


def apply_multihuman_copy_edits(data: dict[str, Any]) -> None:
    """Generate symmetric counterfactual options from structured Task 4 evidence."""
    for group in data["groups"]:
        if not group.get("qa") or group["qa"][0].get("task_id") != TASK4_ID:
            continue
        q = group["qa"][0]
        qtype = q["question_type"]
        if qtype in MULTIHUMAN_GENERIC_QUESTIONS:
            q["question"] = MULTIHUMAN_GENERIC_QUESTIONS[qtype]
        result = q.get("result_json") or {}
        a, b = _person(group, "A"), _person(group, "B")
        B = _cap(b)
        audit = group.get("visual_person_audit") or result.get("visual_person_audit") or {}
        if qtype not in {"visible_pair_topology_change_2d", "visible_pair_topology_consistency_2d"} and int(audit.get("max_visible_person_count") or 0) > 2:
            q["question"] = q["question"].replace("the two people in the video", f"{a} and {b}")

        if qtype == "position_consistency_between_people":
            _set_options(group,
                f"{B} remains right-front relative to {a} throughout the clip.",
                [f"{B} remains left-front relative to {a} throughout the clip.",
                 f"{B} remains right-behind relative to {a} throughout the clip.",
                 f"{B} remains left-behind relative to {a} throughout the clip."])
        elif qtype == "dominant_facing_relation_over_video":
            _set_options(group,
                "Facing each other is the dominant relation for most of the clip.",
                ["Standing side by side is the dominant relation for most of the clip.",
                 "Facing away from each other is the dominant relation for most of the clip.",
                 "Standing at an oblique angle is the dominant relation for most of the clip."])
        elif qtype == "metric_distance_pattern_over_video":
            values = _distance_values(q); start, peak, end = values[0], max(values), values[-1]
            _set_options(group,
                f"They begin near {start:.3f} m and finish near {end:.3f} m after moving farther apart.",
                [f"They begin near {end:.3f} m and finish near {start:.3f} m after moving closer together.",
                 f"They begin near {start:.3f} m and finish near {start:.3f} m after remaining stable.",
                 f"They begin near {start:.3f} m and finish near {start:.3f} m after an out-and-back change."])
        elif qtype == "body_forward_visibility_consistency":
            _set_options(group,
                f"Both {a} and {b} remain inside each other's body-forward fields throughout the clip.",
                [f"Only {a} remains inside the body-forward field of {b} throughout the clip.",
                 f"Only {b} remains inside the body-forward field of {a} throughout the clip.",
                 f"Neither {_cap(a)} nor {b} remains inside the other's body-forward field throughout the clip."])
        elif qtype == "body_centric_relation_change_over_video":
            _set_options(group,
                f"{B} is left-front at the start and right-front at the end relative to {a}.",
                [f"{B} is right-front at the start and left-front at the end relative to {a}.",
                 f"{B} is left-front at the start and left-front at the end relative to {a}.",
                 f"{B} is right-front at the start and right-front at the end relative to {a}."])
        elif qtype == "visible_pair_topology_change_2d":
            starts = {row["pair"]: row["distance"] for row in result["start_pair_distances_normalized"]}
            ends = {row["pair"]: row["distance"] for row in result["end_pair_distances_normalized"]}
            start_pair, end_pair = min(starts, key=starts.get), min(ends, key=ends.get)
            other = next(pair for pair in starts if pair not in {start_pair, end_pair})
            sp, ep, op = (_pair_name(group, value) for value in (start_pair, end_pair, other))
            _set_options(group,
                f"The closest pair changes from {sp} at the start to {ep} at the end.",
                [f"The closest pair changes from {ep} at the start to {sp} at the end.",
                 f"The closest pair changes from {sp} at the start to {sp} at the end.",
                 f"The closest pair changes from {op} at the start to {sp} at the end."])
        elif qtype == "metric_separation_over_video":
            values = _distance_values(q); start, end = values[0], values[-1]
            _set_options(group,
                f"They separate, increasing from about {start:.3f} m initially to {end:.3f} m at the end.",
                [f"They approach, decreasing from about {end:.3f} m initially to {start:.3f} m at the end.",
                 f"They remain stable, staying near {start:.3f} m initially and {start:.3f} m at the end.",
                 f"They return, increasing from about {start:.3f} m initially to {start:.3f} m at the end."])
        elif qtype == "dominant_body_centric_position":
            counts = result.get("relation_counts") or {}
            dominant = max(counts, key=counts.get)
            alternatives = [value for value in ("left_front", "right_front", "left_behind", "right_behind") if value != dominant][:3]
            _set_options(group,
                f"{B} is predominantly {_relation_words(dominant)} relative to {a}.",
                [f"{B} is predominantly {_relation_words(value)} relative to {a}." for value in alternatives])
        elif qtype == "nonmonotonic_distance_pattern":
            values = _distance_values(q); start, peak, end = values[0], max(values), values[-1]
            _set_options(group,
                f"They begin near {start:.3f} m, reach {peak:.3f} m midway, and finish near {end:.3f} m.",
                [f"They begin near {start:.3f} m, reach {end:.3f} m midway, and finish near {peak:.3f} m.",
                 f"They begin near {end:.3f} m, reach {peak:.3f} m midway, and finish near {start:.3f} m.",
                 f"They begin near {peak:.3f} m, reach {start:.3f} m midway, and finish near {end:.3f} m."])
        elif qtype == "approach_while_facing":
            values = _distance_values(q); closest = min(values)
            farthest = max(values)
            _set_options(group,
                f"They move closer to about {closest:.3f} m while predominantly facing each other.",
                [f"They move closer to about {closest:.3f} m while predominantly facing away from each other.",
                 f"They move farther to about {farthest:.3f} m while predominantly facing each other.",
                 f"They move farther to about {farthest:.3f} m while predominantly facing away from each other."])
        elif qtype == "coupled_distance_relation_change":
            values = _distance_values(q); sequence = result["relation_sequence"]
            start_rel, end_rel = _relation_words(sequence[0]), _relation_words(sequence[-1])
            _set_options(group,
                f"{B} approaches from {values[0]:.3f} m to {values[-1]:.3f} m and changes from {start_rel} to {end_rel} relative to {a}.",
                [f"{B} approaches from {values[0]:.3f} m to {values[-1]:.3f} m and changes from {end_rel} to {start_rel} relative to {a}.",
                 f"{B} separates from {values[-1]:.3f} m to {values[0]:.3f} m and changes from {start_rel} to {end_rel} relative to {a}.",
                 f"{B} approaches from {values[0]:.3f} m to {values[-1]:.3f} m and changes from {start_rel} to {start_rel} relative to {a}."])
        elif qtype == "distance_out_and_back_over_video":
            values = _distance_values(q); start, peak, end = values[0], max(values), values[-1]
            _set_options(group,
                f"They begin near {start:.3f} m, reach {peak:.3f} m midway, and finish near {end:.3f} m.",
                [f"They begin near {start:.3f} m, reach {end:.3f} m midway, and finish near {peak:.3f} m.",
                 f"They begin near {end:.3f} m, reach {peak:.3f} m midway, and finish near {start:.3f} m.",
                 f"They begin near {peak:.3f} m, reach {start:.3f} m midway, and finish near {end:.3f} m."])
        elif qtype == "body_forward_field_transition_over_video":
            _set_options(group,
                f"Early, only {b} contains {a}; later, only {a} contains {b} for most of the clip.",
                [f"Early, only {a} contains {b}; later, only {b} contains {a} for most of the clip.",
                 f"Early, both {a} and {b} contain each other; later, neither {a} nor {b} contains the other.",
                 f"Early, neither {a} nor {b} contains the other; later, both {a} and {b} contain each other."])
        elif qtype == "visible_pair_topology_consistency_2d":
            starts = {row["pair"]: row["distance"] for row in result["start_pair_distances_normalized"]}
            ends = {row["pair"]: row["distance"] for row in result["end_pair_distances_normalized"]}
            closest = min(starts, key=starts.get); pairs = list(starts)
            cp = _pair_name(group, closest); alternatives = [_pair_name(group, pair) for pair in pairs if pair != closest]
            _set_options(group,
                f"At the start, {cp} are closest; at the end, {cp} are closest.",
                [f"At the start, {pair} are closest; at the end, {pair} are closest." for pair in alternatives] +
                [f"At the start, {cp} are closest; at the end, {alternatives[0]} are closest."])


TASK1_BALANCED_OPTIONS = {
    "query_02_iiith30_frame4440": [
        "The paprika container is right of the person at the start and left at the end.",
        "The paprika container is left of the person at the start and right at the end.",
        "The paprika container is right of the person at the start and right at the end.",
        "The paprika container is left of the person at the start and left at the end."],
    "sfu0083_cam04_3450": [
        "Near the end, changing body direction brings the knife into the person's forward field.",
        "Near the end, changing blocker position brings the knife into the person's forward field.",
        "Near the end, changing knife position brings the knife into the person's forward field.",
        "Near the end, changing camera viewpoint brings the knife into the person's forward field."],
    "diverse_iiith_145_2_frame7380": [
        "The white chopping board remains nearest to the person throughout the clip.",
        "The steel tomato bowl remains nearest to the person throughout the clip.",
        "The small steel bowl remains nearest to the person throughout the clip.",
        "No single listed object remains nearest to the person throughout the clip."],
    "query_04_iiith145_frame11250": [
        "The bowl changes from the person's left side at the start to the right side at the end.",
        "The bowl changes from the person's right side at the start to the left side at the end.",
        "The bowl remains on the person's left side from the start through the end.",
        "The bowl remains on the person's right side from the start through the end."],
    "query_03_iiith29_frame4350": [
        "All three objects remain in front of the person throughout the clip.",
        "All three objects remain behind the person throughout the clip.",
        "Only the steel bowl remains in front of the person throughout the clip.",
        "Only the steel plate remains behind the person throughout the clip."],
    "diverse_iiith_31_3_frame1620": [
        "The chopped tomato remains the nearest object and stays below the person throughout.",
        "The chopping board remains the nearest object and stays below the person throughout.",
        "The chopped tomato remains the nearest object but stays above the person throughout.",
        "The steel bowl remains the nearest object and stays below the person throughout."],
    "sfu0101_cam05_5460": [
        "The plate gets closer, from about 3.05 m to 1.86 m, and changes from left to right.",
        "The plate gets farther, from about 1.86 m to 3.05 m, and changes from right to left.",
        "The plate gets closer, from about 3.05 m to 1.86 m, and changes from left to left.",
        "The plate stays stable, from about 3.05 m to 3.05 m, and changes from right to right."],
    "val_3": [
        "The bottle is behind the person at the start and right-front at the end.",
        "The bottle is in front of the person at the start and left-behind at the end.",
        "The bottle is right-front of the person at the start and behind at the end.",
        "The bottle is left-behind the person at the start and in front at the end."],
    "diverse_sfu_008_3_frame6960": [
        "Changing the person's body orientation moves the fixed peel outside the body-forward field.",
        "Changing a physical blocker's position moves the fixed peel outside the body-forward field.",
        "Changing the fixed peel's scene position moves it outside the body-forward field.",
        "Changing the external camera viewpoint moves the fixed peel outside the body-forward field."],
    "val_12_replacement_egg_whisk": [
        "The whisk begins very close, at about 0.55 m, and ends farther away, at about 1.86 m.",
        "The whisk begins farther away, at about 1.86 m, and ends very close, at about 0.55 m.",
        "The whisk begins farther away, at about 1.86 m, and ends farther away, at about 1.86 m.",
        "The whisk begins very close, at about 0.55 m, and ends very close, at about 0.55 m."],
}


def apply_task1_balanced_options(data: dict[str, Any]) -> None:
    for group in data["groups"]:
        values = TASK1_BALANCED_OPTIONS.get(group.get("name"))
        if values:
            _set_options(group, values[0], values[1:])


def approximate_metric_distances(text: str, decimals: int = 1) -> str:
    """Round reviewer-facing metre values; exact measurements remain in result_json."""
    def replace(match: re.Match[str]) -> str:
        quantum = Decimal(1).scaleb(-decimals)
        value = Decimal(match.group(1)).quantize(quantum, rounding=ROUND_HALF_UP)
        return f"{value:.{decimals}f} m"
    return re.sub(r"\b(\d+(?:\.\d+)?)\s*m\b", replace, text)


def apply_release_distance_rounding(data: dict[str, Any]) -> None:
    for group in data["groups"]:
        for q in group.get("qa", []):
            for field in ("question", "correct_answer", "answer", "explanation"):
                if isinstance(q.get(field), str):
                    q[field] = approximate_metric_distances(q[field])
            for option in q.get("options", []):
                option["text"] = approximate_metric_distances(option["text"])
            correct = next(option["text"] for option in q["options"] if option["label"] == q["correct_option"])
            q["correct_answer"] = q["answer"] = correct


def validate_scale(data: dict[str, Any]) -> dict[str, Any]:
    names = [g["name"] for g in data["groups"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate case/window names")
    counts: dict[str, Counter[str]] = {TASK1_ID: Counter(), TASK4_ID: Counter()}
    for group in data["groups"]:
        if len(group.get("qa", [])) != 1:
            raise ValueError(f"one-question policy failed: {group['name']}")
        q = group["qa"][0]
        q["question_categories"] = categories(group, q)
        if len(q.get("options", [])) != 4 or len({x["text"] for x in q["options"]}) != 4:
            raise ValueError(f"four-option gate failed: {group['name']}")
        if q["task_id"] == TASK4_ID:
            visible_text = " ".join([q.get("question", ""), q.get("correct_answer", ""), q.get("explanation", ""), q.get("method", ""), *[x["text"] for x in q["options"]]])
            if re.search(r"\b(?:A|B|V1|V2|V3)\b", visible_text):
                raise ValueError(f"raw person ID leaked into display text: {group['name']}")
        counts[q["task_id"]].update(q["question_categories"])
    errors = []
    for task_id, required in REQUIRED_CATEGORIES.items():
        for category in required:
            if counts[task_id][category] < 2:
                errors.append(f"{task_id}::{category} has {counts[task_id][category]} examples; need >=2")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "ok",
        "case_count": len(data["groups"]),
        "qa_count": len(data["groups"]),
        "unique_case_windows": len(names),
        "minimum_examples_per_category": 2,
        "category_counts": {k: dict(v) for k, v in counts.items()},
        "hard_gates": [
            "one question per unique video window",
            "four unique options",
            "at least two independent cases per capability category",
            "Task 1 object world coordinates are scene-fixed unless motion is explicitly measured",
            "metric Task 4 uses temporally covered, identity-aligned timelines",
            "2D topology requires real observations near both endpoints; nearest-time substitution is forbidden",
            "body-forward visibility is labeled as a directional proxy, never gaze or physical occlusion",
        ],
    }



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    ap.add_argument("--ego-data", type=Path, default=Path("outputs/qa/ego_new_data.js"))
    ap.add_argument("--dense-scenes", type=Path, default=Path("outputs/qa/hoim3_multihuman_scenes_dense_all.json"))
    ap.add_argument("--visual-audits", type=Path, default=Path("outputs/qa/multihuman_visual_calibration.json"))
    ap.add_argument("--topology-evidence", type=Path, default=Path("outputs/qa/topology_extra_90s_calibration.json"))
    ap.add_argument("--output-jsonl", type=Path, default=Path("outputs/qa/task1_task4_curated_qa.jsonl"))
    ap.add_argument("--audit-output", type=Path, default=Path("outputs/qa/task1_task4_curated_audit.json"))
    ap.add_argument("--quality-report", type=Path, default=Path("outputs/qa/task1_task4_scale_quality.json"))
    ap.add_argument("--quality-policy", type=Path, help="Optional JSON overrides for ScaleQualityPolicy")
    ap.add_argument("--fail-on-rejected", action="store_true", help="Fail instead of filtering candidates rejected by scale quality gates")
    args = ap.parse_args()
    resolve = lambda p: p if p.is_absolute() else ROOT / p
    data = load_js(resolve(args.site_data))
    ego = load_js(resolve(args.ego_data))
    dense = json.loads(resolve(args.dense_scenes).read_text())
    audits = json.loads(resolve(args.visual_audits).read_text())
    evidence = json.loads(resolve(args.topology_evidence).read_text())
    policy_values = json.loads(resolve(args.quality_policy).read_text()) if args.quality_policy else {}
    policy = ScaleQualityPolicy(**policy_values)

    by_name = {g["name"]: g for g in data["groups"]}
    rewrite_case1(by_name["query_04_iiith145_frame11250"])
    rewrite_joint_relation_distance(by_name["sfu0101_cam05_5460"])
    add_task1_visibility(data, ego)
    add_task4_visibility(data, dense, audits)
    add_task4_topology(data, evidence)
    apply_person_descriptions(data)
    apply_multihuman_copy_edits(data)
    apply_task1_balanced_options(data)
    apply_release_distance_rounding(data)

    # Evaluate every candidate before category counting.  Large-scale runs keep
    # rejection reasons while publishing only cases that pass all hard gates.
    candidate_quality = validate_release(data, policy)
    rejected_indices = {case["case_index"] for case in candidate_quality["cases"] if case["status"] == "rejected"}
    if rejected_indices and args.fail_on_rejected:
        require_release_quality(data, policy)
    data["groups"] = [group for index, group in enumerate(data["groups"], 1) if index not in rejected_indices]
    release_quality = require_release_quality(data, policy)
    audit = validate_scale(data)
    audit["scale_quality_gate"] = {
        "candidate_count": candidate_quality["case_count"],
        "accepted_count": candidate_quality["accepted_count"],
        "rejected_count": candidate_quality["rejected_count"],
        "rejected_cases": [case for case in candidate_quality["cases"] if case["status"] == "rejected"],
        "release_warning_count": release_quality["warning_count"],
        "policy": candidate_quality["policy"],
    }

    save_js(resolve(args.site_data), data)
    rows = []
    for case_index, group in enumerate(data["groups"], 1):
        rows.append({"case_index": case_index, "case_id": group["name"], "video_clip": group.get("video_clip"), **group["qa"][0]})
    resolve(args.output_jsonl).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n")
    resolve(args.audit_output).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    quality_path = resolve(args.quality_report)
    quality_path.parent.mkdir(parents=True, exist_ok=True)
    quality_path.write_text(json.dumps(candidate_quality, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
