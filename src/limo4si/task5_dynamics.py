"""Deterministic Human-State–Grounded Spatial Reasoning analyzers."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


def _relation(state: Mapping[str, Any], object_id: str) -> Mapping[str, Any]:
    relation = (state.get("object_relations") or {}).get(object_id)
    if not isinstance(relation, Mapping) or not relation.get("label"):
        raise ValueError(f"state lacks relation for object {object_id}")
    return relation


def _event_anchor(states: Sequence[Mapping[str, Any]], event: Mapping[str, Any]) -> Mapping[str, Any]:
    start, end = int(event["start_index"]), int(event["end_index"])
    if not 0 <= start <= end < len(states):
        raise ValueError("gaze event lies outside the state timeline")
    return states[(start + end) // 2]


def relation_change_between_gazes(
    states: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], object_id: str,
    *, minimum_gap_s: float = 2.0, minimum_lateral_shift_m: float = 0.2,
) -> dict[str, Any]:
    target = [event for event in events if str(event.get("object_id")) == object_id]
    if len(target) != 2:
        raise ValueError("target must have exactly two supported gaze events")
    first, second = target
    gap = float(second["start_time_s"]) - float(first["end_time_s"])
    if gap < minimum_gap_s:
        raise ValueError("repeated gaze events are not sufficiently separated")
    a = _relation(_event_anchor(states, first), object_id)
    b = _relation(_event_anchor(states, second), object_id)
    shift = abs(float(a["right_m"]) - float(b["right_m"]))
    if a["label"] == b["label"] or shift < minimum_lateral_shift_m:
        raise ValueError("object relation does not change saliently between gazes")
    return {
        "kind": "relation_change_between_gazes", "object_id": object_id,
        "start_relation": str(a["label"]), "end_relation": str(b["label"]),
        "start_frame": int(first["start_index"]), "end_frame": int(second["end_index"]),
        "event_gap_s": round(gap, 6), "lateral_shift_m": round(shift, 6),
    }


def gaze_onset_side_change(
    states: Sequence[Mapping[str, Any]], event: Mapping[str, Any], object_id: str,
    *, pre_offset_states: int = 6, minimum_turn_deg: float = 8.0,
) -> dict[str, Any]:
    onset = int(event["start_index"])
    before = onset - pre_offset_states
    if before < 0 or onset >= len(states):
        raise ValueError("gaze onset lacks a pre-onset state")
    if str(states[before].get("gazed_object_id")) == object_id:
        raise ValueError("pre-onset state already gazes at the target")
    start, end = _relation(states[before], object_id), _relation(states[onset], object_id)
    f0, f1 = states[before].get("forward_world"), states[onset].get("forward_world")
    if not isinstance(f0, list) or not isinstance(f1, list) or len(f0) != 3 or len(f1) != 3:
        raise ValueError("wearer forward vectors are missing")
    cosine = max(-1.0, min(1.0, sum(float(a) * float(b) for a, b in zip(f0, f1))))
    turn = math.degrees(math.acos(cosine))
    if start["label"] == end["label"] or turn < minimum_turn_deg:
        raise ValueError("gaze onset lacks a salient body-frame relation change")
    return {
        "kind": "gaze_onset_side_change", "object_id": object_id,
        "start_relation": str(start["label"]), "end_relation": str(end["label"]),
        "start_frame": before, "end_frame": onset, "wearer_turn_deg": round(turn, 6),
        "pre_gazed_object_id": states[before].get("gazed_object_id"),
    }


def last_gaze_object_relation_change(
    states: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]],
    *, minimum_run_states: int = 6,
) -> dict[str, Any]:
    if not events:
        raise ValueError("timeline has no supported gaze event")
    event = max(events, key=lambda value: int(value["end_index"]))
    object_id = str(event["object_id"])
    labels = [str(_relation(state, object_id)["label"]) for state in states]
    runs: list[list[Any]] = []
    for label in labels:
        if not runs or runs[-1][0] != label:
            runs.append([label, 1])
        else:
            runs[-1][1] += 1
    sequence = [label for label, count in runs if count >= minimum_run_states]
    sequence = [label for index, label in enumerate(sequence) if index == 0 or label != sequence[index - 1]]
    if len(sequence) < 2:
        raise ValueError("last gaze object lacks a sustained relation change")
    return {
        "kind": "last_gaze_annotated_object_relation_change", "object_id": object_id,
        "start_relation": sequence[0], "end_relation": sequence[-1],
        "relation_sequence": sequence,
        "last_supported_event_start_frame": int(event["start_index"]),
        "last_supported_event_end_frame": int(event["end_index"]),
    }
