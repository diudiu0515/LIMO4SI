#!/usr/bin/env python3
"""Build audited Task 5 human-state-grounded spatial QA."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.scale_quality import TASK5_ID, ScaleQualityPolicy, require_release_quality
from limo4si.semantic_gt import (
    LanguageRealizer, compute_result_evidence_signature, load_language_realizer,
    make_semantic_gt, realize_question,
)
from limo4si.task5_human_state import circular_yaw_change_deg
from limo4si.task5_scaling import CATEGORY_BY_TYPE, sustained_relation_sequence

TASK5_NAME = "Task 5 · Human-State–Grounded Spatial Reasoning"
RELATIONS = ["left-front", "front", "right-front", "right", "right-behind", "behind", "left-behind", "left"]


def load_js(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    return json.loads(match.group(1) if match else text)


def save_js(path: Path, data: dict[str, Any]) -> None:
    path.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def shown_name(value: str) -> str:
    aliases = {"KitchIsland": "kitchen island", "WhiteVase": "white vase", "WoodenBowl": "wooden bowl"}
    if value in aliases:
        return aliases[value]
    return " ".join(re.sub(r"(?<!^)(?=[A-Z])", " ", value).lower().split())


def event_by_start(events: list[dict[str, Any]], object_id: str, frame: int) -> dict[str, Any]:
    matches = [event for event in events if str(event["object_id"]) == object_id and int(event["start_index"]) == frame]
    if len(matches) != 1:
        raise ValueError(f"expected one gaze event for {object_id} at frame {frame}, found {len(matches)}")
    return matches[0]


def reduced(values: list[str]) -> list[str]:
    out = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def make_options(
    correct: str, alternatives: list[str], seed: str, correct_index: int | None = None,
) -> tuple[list[dict[str, str]], str]:
    values = [correct, *alternatives]
    if len(values) != 4 or len(set(values)) != 4:
        raise ValueError(f"Task 5 options are not four unique parallel choices: {seed}")
    offset = correct_index if correct_index is not None else int(hashlib.sha256(seed.encode("utf-8")).hexdigest()[:8], 16) % 4
    if not 0 <= offset < 4:
        raise ValueError(f"invalid correct option index {offset}")
    ordered = alternatives[:]
    ordered.insert(offset, correct)
    labels = ["A", "B", "C", "D"]
    return [{"label": label, "text": text} for label, text in zip(labels, ordered)], labels[offset]


def _parallel_pair_distractors(start: str, end: str) -> list[tuple[str, str]]:
    mirror = {
        "left-front": "right-front", "right-front": "left-front",
        "left": "right", "right": "left", "left-behind": "right-behind",
        "right-behind": "left-behind", "front": "behind", "behind": "front",
    }
    preferred = [(end, start), (start, mirror.get(end, end)), (mirror.get(start, start), end)]
    pool = preferred + [(left, right) for left in RELATIONS for right in RELATIONS]
    correct = (start, end)
    unique: list[tuple[str, str]] = []
    for pair in pool:
        if pair != correct and pair not in unique:
            unique.append(pair)
        if len(unique) == 3:
            return unique
    raise ValueError("could not construct three parallel relation-pair distractors")


def transition_options(
    object_name: str, start: str, end: str, seed: str, correct_index: int | None = None,
) -> tuple[list[dict[str, str]], str, str]:
    template = f"The {object_name} changes from {{}} to {{}}."
    correct = template.format(start, end)
    alternatives = [template.format(*pair) for pair in _parallel_pair_distractors(start, end)]
    options, label = make_options(correct, alternatives, seed, correct_index)
    return options, label, correct


def sequence_options(
    object_name: str, sequence: list[str], seed: str, correct_index: int | None = None,
) -> tuple[list[dict[str, str]], str, str]:
    arrow = " → "
    template = f"The {object_name} follows: {{}}."
    correct = template.format(arrow.join(sequence))
    candidates: list[list[str]] = []
    if len(sequence) > 1:
        candidates.append(list(reversed(sequence)))
    for offset in range(1, len(RELATIONS)):
        shifted = [RELATIONS[(RELATIONS.index(value) + offset) % len(RELATIONS)] if value in RELATIONS else RELATIONS[offset] for value in sequence]
        candidates.append(shifted)
    alternatives: list[str] = []
    for values in candidates:
        text = template.format(arrow.join(values))
        if text != correct and text not in alternatives:
            alternatives.append(text)
        if len(alternatives) == 3:
            break
    if len(alternatives) != 3:
        raise ValueError("could not construct three equal-slot relation-sequence distractors")
    options, label = make_options(correct, alternatives, seed, correct_index)
    return options, label, correct

def compact_state(state: dict[str, Any], object_id: str) -> dict[str, Any]:
    return {
        "frame": state["frame_index"], "time_s": state["time_s"],
        "gazed_object_id": state["gazed_object_id"], "gazed_object_name": state["gazed_object_name"],
        "gaze_depth_m": state["gaze_depth_m"],
        "gaze_hit_distance_m": state["gaze_hit_distance_m"],
        "gaze_hit_exit_distance_m": state["gaze_hit_exit_distance_m"],
        "gaze_depth_obb_residual_m": state["gaze_depth_obb_residual_m"],
        "gaze_origin_world_m": state["gaze_origin_world_m"],
        "gaze_direction_world_unit": state["gaze_direction_world_unit"],
        "wearer_world_m": state["wearer_world_m"], "right_world": state["right_world"],
        "forward_world": state["forward_world"],
        "wearer_timestamp_skew_ms": state.get("wearer_timestamp_skew_ms"),
        "object_pose_skew_ms": (state.get("object_pose_skew_ms") or {}).get(object_id),
        "relation": state["object_relations"][object_id],
    }


def _semantic_options(texts: list[str], correct_text: str, prefix: str) -> tuple[list[dict[str, Any]], str]:
    options = []
    correct_id = ""
    for index, value in enumerate(texts):
        option_id = f"{prefix}_{index + 1}"
        options.append({"id": option_id, "statement": value})
        if value == correct_text:
            correct_id = option_id
    if not correct_id:
        raise ValueError("code-computed correct answer is absent from semantic options")
    return options, correct_id


def _ordinal(index: int) -> str:
    words = ["first", "second", "third", "fourth"]
    if not 0 <= index < len(words):
        raise ValueError(f"unsupported evidence-anchor index {index}")
    return words[index]


def build_question(
    spec: dict[str, Any], analysis: dict[str, Any], realizer: LanguageRealizer | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute signed GT first, then perform constrained language realization."""
    states, events = analysis["states"], analysis["gaze_events"]
    states_by_frame = {int(state["frame_index"]): state for state in states}
    object_id = str(spec.get("object_id") or next(
        uid for uid, row in analysis["objects"].items() if row["instance_name"] == spec["object_name"]
    ))
    if object_id not in analysis["objects"]:
        raise ValueError(f"configured object_id {object_id!r} is absent from analysis")
    canonical_object_name = analysis["objects"][object_id]["instance_name"]
    display = shown_name(canonical_object_name)
    lo, hi = (int(value) for value in spec["window_frames"])
    timeline = [compact_state(state, object_id) for state in states if lo <= int(state["frame_index"]) <= hi]
    qtype = str(spec["question_type"])
    correct_index = spec.get("_correct_option_index")
    event_evidence: list[dict[str, Any]] = []
    semantic_facts: list[dict[str, Any]] = []
    evidence_refs: list[dict[str, Any]] = []
    anchor_frames: list[int]

    if qtype == "relation_change_between_gazes":
        selected = [event_by_start(events, object_id, int(frame)) for frame in spec["event_start_frames"]]
        anchor_frames = [(int(event["start_index"]) + int(event["end_index"])) // 2 for event in selected]
        start_relation, end_relation = (
            states_by_frame[frame]["object_relations"][object_id]["label"] for frame in anchor_frames
        )
        rendered, _, correct_text = transition_options(display, start_relation, end_relation, spec["id"], 0)
        option_specs, correct_option_id = _semantic_options(
            [option["text"] for option in rendered], correct_text, "relation_pair",
        )
        question_focus = f"Between the two sustained gazes at the {display}, how does its wearer-relative position change?"
        evidence_statement = (
            f"At the first gaze event the {display} is {start_relation}; "
            f"at the later event it is {end_relation}."
        )
        event_evidence = selected
        transition = {
            "start_frame": anchor_frames[0], "end_frame": anchor_frames[1],
            "start_relation": start_relation, "end_relation": end_relation,
        }
        semantic_facts = [
            {"id": "first_relation", "value": start_relation, "frame": anchor_frames[0]},
            {"id": "second_relation", "value": end_relation, "frame": anchor_frames[1]},
        ]
        evidence_refs = [{"kind": "gaze_event", "start_index": int(event["start_index"])} for event in selected]
    elif qtype == "gaze_onset_side_change":
        selected = [event_by_start(events, object_id, int(spec["event_start_frames"][0]))]
        before = int(spec["pre_frame"])
        after = (int(selected[0]["start_index"]) + int(selected[0]["end_index"])) // 2
        anchor_frames = [before, after]
        start_relation = states_by_frame[before]["object_relations"][object_id]["label"]
        end_relation = states_by_frame[after]["object_relations"][object_id]["label"]
        rendered, _, correct_text = transition_options(display, start_relation, end_relation, spec["id"], 0)
        option_specs, correct_option_id = _semantic_options(
            [option["text"] for option in rendered], correct_text, "relation_pair",
        )
        question_focus = f"As gaze turns to the {display}, how does it shift in the wearer's body-relative view?"
        evidence_statement = (
            f"Immediately before the gaze onset the {display} is {start_relation}; "
            f"during the sustained gaze it is {end_relation}."
        )
        event_evidence = selected
        transition = {
            "start_frame": before, "end_frame": after,
            "start_relation": start_relation, "end_relation": end_relation,
            "wearer_turn_deg": circular_yaw_change_deg(
                states_by_frame[before]["forward_world"], states_by_frame[after]["forward_world"],
            ),
            "pre_gazed_object_id": states_by_frame[before]["gazed_object_id"],
        }
        semantic_facts = [
            {"id": "pre_onset_relation", "value": start_relation, "frame": before},
            {"id": "sustained_gaze_relation", "value": end_relation, "frame": after},
            {"id": "wearer_turn_deg", "value": transition["wearer_turn_deg"]},
        ]
        evidence_refs = [{"kind": "gaze_event", "start_index": int(selected[0]["start_index"])}]
    elif qtype == "last_gaze_annotated_object_relation_change":
        in_window = [event for event in events if int(event["start_index"]) >= lo and int(event["end_index"]) <= hi]
        if not in_window:
            raise ValueError(f"{spec['id']} has no supported gaze event")
        last_event = max(in_window, key=lambda event: int(event["end_index"]))
        if str(last_event["object_id"]) != object_id:
            raise ValueError(f"{spec['id']} configured object is not the last gaze-annotated object")
        relation_sequence = sustained_relation_sequence(
            [row["relation"]["label"] for row in timeline], ScaleQualityPolicy().min_task5_relation_run_states,
        )
        rendered, _, correct_text = sequence_options(display, relation_sequence, spec["id"], 0)
        option_specs, correct_option_id = _semantic_options(
            [option["text"] for option in rendered], correct_text, "relation_sequence",
        )
        question_focus = (
            f"The {display} is the last gaze-annotated object in this window. "
            "How does its wearer-relative position evolve?"
        )
        evidence_statement = (
            f"Across the full window, the annotation-derived relation sequence is {' → '.join(relation_sequence)}."
        )
        event_evidence = [last_event]
        anchor_frames = [lo, hi]
        transition = {
            "start_frame": lo, "end_frame": hi, "relation_sequence": relation_sequence,
            "last_supported_event_end_frame": int(last_event["end_index"]),
            "last_supported_event_object_id": str(last_event["object_id"]),
        }
        semantic_facts = [{"id": "relation_sequence", "value": relation_sequence, "window_frames": [lo, hi]}]
        evidence_refs = [{"kind": "gaze_event", "start_index": int(last_event["start_index"])}]
    elif qtype == "gaze_target_at_evidence_anchor":
        anchor_frames = [int(frame) for frame in spec["anchor_frames"]]
        if len(anchor_frames) not in {2, 3} or len(set(anchor_frames)) != len(anchor_frames):
            raise ValueError(f"{spec['id']} requires two or three unique evidence anchors")
        if anchor_frames != sorted(anchor_frames) or anchor_frames[0] < lo or anchor_frames[-1] > hi:
            raise ValueError(f"{spec['id']} evidence anchors must be ordered inside the public window")
        anchor_targets = []
        for frame in anchor_frames:
            state = states_by_frame.get(frame)
            if state is None or not state.get("gazed_object_name"):
                raise ValueError(f"{spec['id']} frame {frame} has no annotation-grounded gaze target")
            anchor_targets.append({
                "frame": frame,
                "time_s": float(state["time_s"]),
                "gaze_target": str(state["gazed_object_name"]),
                "gaze_target_id": str(state["gazed_object_id"]),
            })
        hit_indices = [index for index, anchor in enumerate(anchor_targets) if anchor["gaze_target_id"] == object_id]
        if len(hit_indices) != 1:
            raise ValueError(f"{spec['id']} must have exactly one anchor whose measured gaze hits the target")
        target_anchor = hit_indices[0]
        containing_events = [
            event for event in events
            if str(event["object_id"]) == object_id
            and int(event["start_index"]) <= anchor_frames[target_anchor] <= int(event["end_index"])
        ]
        if len(containing_events) != 1:
            raise ValueError(f"{spec['id']} target anchor does not resolve to one supported gaze event")
        supporting_event = containing_events[0]
        if len(anchor_frames) == 2:
            option_specs = [
                {"id": "anchor_1", "statement": "Only at the first marked anchor."},
                {"id": "anchor_2", "statement": "Only at the second marked anchor."},
                {"id": "multiple_anchors", "statement": "At each of the two marked anchors."},
                {"id": "no_anchor", "statement": "At neither of the two marked anchors."},
            ]
        else:
            option_specs = [
                {"id": "anchor_1", "statement": "Only at the first marked anchor."},
                {"id": "anchor_2", "statement": "Only at the second marked anchor."},
                {"id": "anchor_3", "statement": "Only at the third marked anchor."},
                {"id": "no_anchor", "statement": "At none of the three marked anchors."},
            ]
        correct_option_id = f"anchor_{target_anchor + 1}"
        question_focus = f"At which marked evidence anchor is the wearer looking at the {display}?"
        other_targets = ", ".join(
            f"{_ordinal(index)}: {shown_name(anchor['gaze_target'])}"
            for index, anchor in enumerate(anchor_targets) if index != target_anchor
        )
        evidence_statement = (
            f"The measured gaze ray hits the {display} at the {_ordinal(target_anchor)} marked anchor "
            f"({anchor_targets[target_anchor]['time_s']:.1f} s); the other annotated targets are {other_targets}."
        )
        event_evidence = [supporting_event]
        transition = {
            "start_frame": anchor_frames[0], "end_frame": anchor_frames[-1],
            "target_anchor_index": target_anchor,
        }
        semantic_facts = [
            {"id": f"anchor_{index + 1}_gaze_target", "frame": anchor["frame"], "value": anchor["gaze_target"]}
            for index, anchor in enumerate(anchor_targets)
        ]
        evidence_refs = [
            {"kind": "annotated_gaze_anchor", "frame": anchor["frame"], "gaze_target_id": anchor["gaze_target_id"]}
            for anchor in anchor_targets
        ]
    else:
        raise ValueError(f"unsupported Task 5 question type {qtype}")

    result = {
        "status": "ok", "answer_type": qtype, "T_Q": True, "H_Q": True, "S_Q": True,
        "annotation_direct": True,
        "annotation_source": analysis["dataset"],
        "analysis_schema_version": analysis.get("schema_version"),
        "coordinate_frame": analysis["coordinate_frame"],
        "alignment_diagnostics": analysis.get("alignment_diagnostics"),
        "maximum_internal_gaze_gap_states": analysis.get("maximum_internal_gaze_gap_states"),
        "gaze_definition": analysis["gaze_definition"],
        "gaze_grounding_method": "depth_consistent_ray_obb_intersection",
        "maximum_gaze_depth_obb_residual_m": analysis.get("maximum_gaze_depth_obb_residual_m"),
        "object_id": object_id, "object_name": canonical_object_name,
        "object_category": analysis["objects"][object_id].get("category"),
        "timeline": timeline, "gaze_events": event_evidence, "transition": transition,
    }
    if qtype == "gaze_target_at_evidence_anchor":
        result.update({
            "anchor_gaze_targets": anchor_targets,
            "supporting_gaze_event": event_evidence[0],
            "answer_provenance": "measured gaze ray/depth intersection with annotated 3D objects; no LLM target judgment",
            "claim_limits": ["gaze target only at the displayed anchors", "no body-axis direction claim"],
        })
    evidence_signature = compute_result_evidence_signature(result)
    semantic_facts.append({"id": "result_evidence_signature", "value": evidence_signature})
    evidence_refs.append({"kind": "result_json_sha256", "sha256": evidence_signature})
    semantic_gt = make_semantic_gt(
        case_id=spec["id"], task_id=TASK5_ID, question_type=qtype,
        question_focus=question_focus, options=option_specs, correct_option_id=correct_option_id,
        evidence_statement=evidence_statement, semantic_facts=semantic_facts, evidence_refs=evidence_refs,
        provenance={
            "dataset": analysis["dataset"], "sequence_name": analysis["sequence_name"],
            "analysis_schema_version": analysis.get("schema_version"),
            "coordinate_frame": analysis["coordinate_frame"],
            "computation": "deterministic_annotation_geometry",
        },
    )
    language = realize_question(
        semantic_gt, realizer=realizer, variant_key=str(spec.get("language_variant", "default")),
        correct_index=int(correct_index) if correct_index is not None else None,
        fallback_on_error=False,
    )
    result.update({
        "semantic_gt_id": semantic_gt["semantic_gt_id"],
        "answer_signature": semantic_gt["answer_signature"],
        "evidence_signature": evidence_signature,
        "reasoning_owner": "deterministic_code",
        "language_model_role": "wording_only",
    })
    question = {
        "task_id": TASK5_ID, "task_name": TASK5_NAME, "question_type": qtype,
        "question_categories": [CATEGORY_BY_TYPE[qtype]], **language, "status": "ok",
        "method": (
            "Deterministic ADT gaze/object computation produces signed semantic GT; "
            "the language layer cannot choose or modify the answer."
        ),
        "result_json": result,
    }
    return question, {"object_id": object_id, "anchor_frames": anchor_frames}

def media_url(prefix: str, filename: str) -> str:
    return f"{prefix.rstrip('/')}/{filename}"


def ensure_clip(source: Path, destination: Path, start_s: float, duration_s: float) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start_s:.6f}", "-i", str(source),
        "-t", f"{duration_s:.6f}", "-c:v", "libx264", "-preset", "fast", "-crf", "27",
        "-an", "-movflags", "+faststart", str(destination),
    ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/task5_release_cases.json"))
    parser.add_argument("--site-data", type=Path, default=Path("site/qa_benchmark/data.js"))
    parser.add_argument("--output-jsonl", type=Path, default=Path("outputs/qa/task5_scaled_qa.jsonl"))
    parser.add_argument("--audit-output", type=Path, default=Path("outputs/qa/task5_scale_audit.json"))
    parser.add_argument("--quality-output", type=Path, default=Path("outputs/qa/task5_scale_quality.json"))
    parser.add_argument("--media-output-dir", type=Path, default=Path("site/qa_benchmark/task5_media"))
    parser.add_argument("--media-url-prefix", default="./task5_media")
    parser.add_argument(
        "--language-client-factory",
        help="Optional module:function returning a StructuredOutputClient; omitted means deterministic templates",
    )
    args = parser.parse_args()
    resolve = lambda path: path if path.is_absolute() else ROOT / path
    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    analysis_cache: dict[Path, dict[str, Any]] = {}
    language_realizer = load_language_realizer(args.language_client_factory)
    groups = []
    for case_index, raw_spec in enumerate(config["cases"]):
        spec = dict(raw_spec, _correct_option_index=case_index % 4)
        analysis_value = spec.get("analysis") or config.get("analysis")
        media_value = spec.get("media_source") or config.get("media_source")
        if not analysis_value or not media_value:
            raise ValueError(f"{spec.get('id')} must resolve analysis and media_source")
        analysis_path = resolve(Path(analysis_value))
        analysis = analysis_cache.setdefault(
            analysis_path, json.loads(analysis_path.read_text(encoding="utf-8")),
        )
        source_media = resolve(Path(media_value))
        question, media = build_question(spec, analysis, language_realizer)
        states_by_frame = {int(state["frame_index"]): state for state in analysis["states"]}
        lo, hi = (int(value) for value in spec["window_frames"])
        start_s, end_s = float(states_by_frame[lo]["time_s"]), float(states_by_frame[hi]["time_s"])
        duration = end_s - start_s
        window_policy = ScaleQualityPolicy()
        if not window_policy.min_task5_window_sec <= duration <= window_policy.max_task5_window_sec:
            raise ValueError(
                f"{spec['id']} spans {duration:.2f}s; Task 5 public videos must be "
                f"{window_policy.min_task5_window_sec:g}–{window_policy.max_task5_window_sec:g}s"
            )
        media_output_dir = resolve(args.media_output_dir)
        destination = media_output_dir / f"{spec['id']}.mp4"
        ensure_clip(source_media, destination, start_s, duration)
        source_label = str(source_media.relative_to(ROOT)) if source_media.is_relative_to(ROOT) else str(source_media)
        groups.append({
            "name": spec["id"], "title": f"Task 5 · {shown_name(question['result_json']['object_name'])}",
            "video_clip": media_url(args.media_url_prefix, f"{spec['id']}.mp4"),
            "original_image": media_url(args.media_url_prefix, f"{spec['id']}_gaze_evidence.jpg"),
            "original_caption": "ADT RGB anchor frames · green: target 2D box · red: measured gaze projection",
            "video_window": {
                "source_video": source_label, "source_sequence": analysis["sequence_name"],
                "start_sec": start_s, "duration_sec": duration,
            },
            "qa": [question], "task5_media": media,
            "case_policy": "one annotation-derived question per unique gaze/object window",
        })
    coordinate_frames = {str(row["coordinate_frame"]) for row in analysis_cache.values()}
    if len(coordinate_frames) != 1:
        raise ValueError(f"Task 5 analyses disagree on coordinate policy: {sorted(coordinate_frames)}")
    counts = Counter(group["qa"][0]["question_categories"][0] for group in groups)
    minimum = int(config.get("minimum_examples_per_category", 2))
    category_minimums = {
        category: int((config.get("minimum_examples_by_category") or {}).get(category, minimum))
        for category in CATEGORY_BY_TYPE.values()
    }
    missing = {category: required - counts[category] for category, required in category_minimums.items() if counts[category] < required}
    if missing:
        raise ValueError(f"Task 5 categories below configured pilot minimums: {missing}")
    quality = require_release_quality({"groups": groups}, ScaleQualityPolicy())
    audit = {
        "status": "ok", "case_count": len(groups), "minimum_examples_per_category": minimum,
        "minimum_examples_by_category": category_minimums, "category_counts": dict(counts), "unique_video_windows": len({
            (g['video_window']['source_sequence'], g['video_window']['start_sec'], g['video_window']['duration_sec'])
            for g in groups
        }),
        "answer_provenance": "directly computed from gaze ray + fixation depth / same-time 3D OBB / wearer pose annotations",
        "coordinate_policy": next(iter(coordinate_frames)),
        "semantic_gt_schema": "limo4si.semantic_gt.v1",
        "reasoning_owner": "deterministic_code",
        "language_realizer": language_realizer.name if language_realizer else "deterministic_template",
    }
    data_path = resolve(args.site_data)
    data = load_js(data_path)
    data["groups"] = [group for group in data.get("groups", []) if not str(group.get("name", "")).startswith("task5_")]
    data["groups"].extend(groups)
    task = {"id": TASK5_ID, "name": TASK5_NAME, "description": "How gaze-anchored object relations change with the wearer's spatial state."}
    data["tasks"] = [row for row in data.get("tasks", []) if row.get("id") != TASK5_ID] + [task]
    data["title"] = "Task 4 + Task 5 Spatial QA"
    data["subtitle"] = "One evidence-grounded temporal question per unique video window."
    policy = data.setdefault("release_policy", {})
    policy["task_scope"] = [row["id"] for row in data["tasks"]]
    policy["task5_scope"] = "measured gaze rays + same-time object 3D boxes + gravity-aligned wearer coordinates"
    # This builder owns only the groups generated above.  Unrelated site groups
    # are validated by the combined-release command, not allowed to block or
    # weaken this batch's fail-closed Task 5 gate.
    save_js(data_path, data)
    output = resolve(args.output_jsonl); output.parent.mkdir(parents=True, exist_ok=True)
    rows_out = [{"case_index": i, "case_id": group["name"], "video_clip": group["video_clip"], **group["qa"][0]} for i, group in enumerate(groups, 1)]
    output.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows_out) + "\n", encoding="utf-8")
    resolve(args.audit_output).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    resolve(args.quality_output).write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
