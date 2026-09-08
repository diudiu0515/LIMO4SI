"""Automatic high-quality candidate mining and balanced selection for Task 5."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .task5_human_state import circular_yaw_change_deg

CATEGORY_BY_TYPE = {
    "relation_change_between_gazes": "between_repeated_gaze_events",
    "gaze_onset_side_change": "after_gaze_turns_to_object",
    "last_gaze_annotated_object_relation_change": "last_gaze_annotated_object",
}


@dataclass(frozen=True)
class Task5CandidatePolicy:
    min_event_gap_sec: float = 0.30
    min_event_direct_hits: int = 4
    min_event_hit_support: float = 0.80
    max_internal_gap_states: int = 1
    min_lateral_shift_m: float = 0.05
    min_onset_turn_deg: float = 8.0
    min_pre_gaze_gap_sec: float = 0.40
    max_pre_gaze_gap_sec: float = 1.50
    max_repeated_window_sec: float = 6.0
    max_last_object_window_sec: float = 4.0
    max_relation_sequence_length: int = 4
    max_candidates_per_object_category: int = 8


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return cleaned[:48] or "sequence"


def _reduced(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for value in values:
        if not output or output[-1] != value:
            output.append(value)
    return output


def _anchor(event: Mapping[str, Any]) -> int:
    return (int(event["start_index"]) + int(event["end_index"])) // 2


def _case_id(sequence: str, category: str, object_id: str, start: int, end: int) -> str:
    category_slug = {
        "between_repeated_gaze_events": "repeated",
        "after_gaze_turns_to_object": "onset",
        "last_gaze_annotated_object": "last",
    }[category]
    digest = hashlib.sha1(f"{sequence}|{category}|{object_id}|{start}|{end}".encode()).hexdigest()[:8]
    return f"task5_{category_slug}_{_slug(sequence)}_{start}_{end}_{digest}"


def _relation(states: Mapping[int, Mapping[str, Any]], frame: int, object_id: str) -> Mapping[str, Any] | None:
    return (states.get(frame, {}).get("object_relations") or {}).get(object_id)


def _candidate(
    analysis: Mapping[str, Any], category: str, object_id: str, start: int, end: int,
    score: float, **fields: Any,
) -> dict[str, Any]:
    sequence = str(analysis["sequence_name"])
    obj = (analysis.get("objects") or {})[object_id]
    return {
        "id": _case_id(sequence, category, object_id, start, end),
        "question_type": next(qtype for qtype, value in CATEGORY_BY_TYPE.items() if value == category),
        "category": category,
        "object_id": object_id,
        "object_name": obj["instance_name"],
        "sequence_name": sequence,
        "window_frames": [start, end],
        "candidate_score": round(float(score), 6),
        **fields,
    }


def generate_task5_candidates(
    analysis: Mapping[str, Any], policy: Task5CandidatePolicy | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate annotation-only Task 5 candidates; no language model is used."""
    policy = policy or Task5CandidatePolicy()
    states_list = analysis.get("states") or []
    states = {int(state["frame_index"]): state for state in states_list}
    raw_events = sorted(analysis.get("gaze_events") or [], key=lambda event: int(event["start_index"]))
    candidates: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    events = []
    for event in raw_events:
        direct_hits = int(event.get("direct_hit_count") or 0)
        support = float(event.get("hit_support_ratio") or 0.0)
        gaps = int(event.get("merged_gap_count") or 0)
        if direct_hits < policy.min_event_direct_hits or support < policy.min_event_hit_support or gaps > policy.max_internal_gap_states:
            rejection_counts["event_support"] += 1
            continue
        events.append(event)

    by_object: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_object[str(event["object_id"])].append(event)

    # 1. Compare two temporally distinct sustained gazes at the same object.
    for object_id, object_events in by_object.items():
        for first, second in zip(object_events, object_events[1:]):
            gap = float(second["start_time_s"]) - float(first["end_time_s"])
            duration = float(second["end_time_s"]) - float(first["start_time_s"])
            if gap < policy.min_event_gap_sec or duration > policy.max_repeated_window_sec:
                rejection_counts["repeated_gap_or_window"] += 1
                continue
            first_frame, second_frame = _anchor(first), _anchor(second)
            left, right = _relation(states, first_frame, object_id), _relation(states, second_frame, object_id)
            if not left or not right:
                rejection_counts["missing_relation"] += 1
                continue
            shift = abs(float(left["right_m"]) - float(right["right_m"]))
            if left["label"] == right["label"] or shift < policy.min_lateral_shift_m:
                rejection_counts["repeated_not_salient"] += 1
                continue
            score = shift + min(gap, 2.0) * 0.05 + min(int(first["state_count"]), int(second["state_count"])) / 100
            candidates.append(_candidate(
                analysis, "between_repeated_gaze_events", object_id,
                int(first["start_index"]), int(second["end_index"]), score,
                event_start_frames=[int(first["start_index"]), int(second["start_index"])],
            ))

    # 2. Find a real pre-onset state where the wearer turns and the target changes side.
    for event in events:
        object_id = str(event["object_id"])
        after_frame = _anchor(event)
        after = _relation(states, after_frame, object_id)
        if not after:
            continue
        viable: list[tuple[float, int]] = []
        for before_frame, before_state in states.items():
            if before_frame >= int(event["start_index"]):
                continue
            gap = float(event["start_time_s"]) - float(before_state["time_s"])
            if not policy.min_pre_gaze_gap_sec <= gap <= policy.max_pre_gaze_gap_sec:
                continue
            if str(before_state.get("gazed_object_id")) == object_id:
                continue
            before = _relation(states, before_frame, object_id)
            if not before or before["label"] == after["label"]:
                continue
            shift = abs(float(before["right_m"]) - float(after["right_m"]))
            turn = circular_yaw_change_deg(before_state["forward_world"], states[after_frame]["forward_world"])
            if shift < policy.min_lateral_shift_m or turn < policy.min_onset_turn_deg:
                continue
            viable.append((shift + math.radians(turn) * 0.15 - abs(gap - 0.8) * 0.02, before_frame))
        if not viable:
            rejection_counts["onset_no_valid_pre_state"] += 1
            continue
        score, before_frame = max(viable)
        candidates.append(_candidate(
            analysis, "after_gaze_turns_to_object", object_id, before_frame,
            int(event["end_index"]), score,
            event_start_frames=[int(event["start_index"])], pre_frame=before_frame,
        ))

    # 3. End a window on a sustained gaze and recompute the target's full relation sequence.
    for event in events:
        object_id = str(event["object_id"])
        end = int(event["end_index"])
        end_time = float(states[end]["time_s"])
        viable_last: list[tuple[float, int, list[str]]] = []
        for start, state in states.items():
            if start >= int(event["start_index"]):
                continue
            duration = end_time - float(state["time_s"])
            if duration <= 0 or duration > policy.max_last_object_window_sec:
                continue
            window = [states[index] for index in sorted(states) if start <= index <= end]
            relations = [_relation(states, int(item["frame_index"]), object_id) for item in window]
            if any(relation is None for relation in relations):
                continue
            sequence = _reduced([str(relation["label"]) for relation in relations if relation])
            lateral = [float(relation["right_m"]) for relation in relations if relation]
            shift = max(lateral) - min(lateral)
            if not 2 <= len(sequence) <= policy.max_relation_sequence_length or shift < policy.min_lateral_shift_m:
                continue
            score = shift + min(duration, 2.0) * 0.03 - max(0, len(sequence) - 3) * 0.05
            viable_last.append((score, start, sequence))
        if not viable_last:
            rejection_counts["last_no_salient_window"] += 1
            continue
        score, start, sequence = max(viable_last)
        candidates.append(_candidate(
            analysis, "last_gaze_annotated_object", object_id, start, end, score,
            expected_relation_sequence=sequence,
        ))

    # Bound highly repetitive candidates before cross-sequence balancing.
    kept: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        groups[(candidate["category"], candidate["object_id"])].append(candidate)
    for rows in groups.values():
        kept.extend(sorted(rows, key=lambda row: (-row["candidate_score"], row["id"]))[:policy.max_candidates_per_object_category])
    kept.sort(key=lambda row: (row["category"], -row["candidate_score"], row["id"]))
    diagnostics = {
        "policy": asdict(policy),
        "sequence_name": analysis.get("sequence_name"),
        "state_count": len(states),
        "raw_gaze_event_count": len(raw_events),
        "eligible_gaze_event_count": len(events),
        "candidate_count": len(kept),
        "candidate_counts": dict(Counter(row["category"] for row in kept)),
        "rejection_counts": dict(rejection_counts),
    }
    return kept, diagnostics


def select_balanced_candidates(
    candidates: Sequence[Mapping[str, Any]], target_per_category: int,
    max_cases_per_sequence: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select every category with deterministic sequence/object diversity."""
    if target_per_category < 1:
        raise ValueError("target_per_category must be positive")
    categories = list(CATEGORY_BY_TYPE.values())
    sequence_names = sorted({str(row["sequence_name"]) for row in candidates})
    total_target = target_per_category * len(categories)
    if max_cases_per_sequence <= 0:
        max_cases_per_sequence = max(1, math.ceil(total_target / max(1, len(sequence_names))) + 1)
    selected: list[dict[str, Any]] = []
    sequence_counts: Counter[str] = Counter()
    object_category_counts: Counter[tuple[str, str]] = Counter()
    used_windows: set[tuple[str, int, int]] = set()
    deficits: dict[str, int] = {}
    for category in categories:
        pool = [dict(row) for row in candidates if row["category"] == category]
        while sum(row["category"] == category for row in selected) < target_per_category:
            eligible = []
            for row in pool:
                key = (str(row["sequence_name"]), int(row["window_frames"][0]), int(row["window_frames"][1]))
                if key in used_windows or sequence_counts[str(row["sequence_name"])] >= max_cases_per_sequence:
                    continue
                diversity = sequence_counts[str(row["sequence_name"])] * 0.20
                diversity += object_category_counts[(category, str(row["object_id"]))] * 0.10
                eligible.append((float(row["candidate_score"]) - diversity, row["id"], row, key))
            if not eligible:
                break
            _, _, chosen, window_key = max(eligible, key=lambda item: (item[0], item[1]))
            selected.append(chosen)
            used_windows.add(window_key)
            sequence_counts[str(chosen["sequence_name"])] += 1
            object_category_counts[(category, str(chosen["object_id"]))] += 1
            pool = [row for row in pool if row["id"] != chosen["id"]]
        count = sum(row["category"] == category for row in selected)
        if count < target_per_category:
            deficits[category] = target_per_category - count
    selected.sort(key=lambda row: (categories.index(row["category"]), row["sequence_name"], row["id"]))
    return selected, {
        "status": "ok" if not deficits else "insufficient_candidates",
        "target_per_category": target_per_category,
        "selected_count": len(selected),
        "selected_counts": dict(Counter(row["category"] for row in selected)),
        "selected_sequence_counts": dict(sequence_counts),
        "max_cases_per_sequence": max_cases_per_sequence,
        "deficits": deficits,
    }
