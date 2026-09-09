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
    min_event_gap_sec: float = 2.0
    min_event_direct_hits: int = 4
    min_repeated_event_direct_hits: int = 10
    min_onset_event_direct_hits: int = 10
    min_last_event_direct_hits: int = 4
    min_repeated_event_duration_sec: float = 0.30
    min_onset_event_duration_sec: float = 0.30
    min_last_event_duration_sec: float = 0.10
    min_event_hit_support: float = 0.80
    max_internal_gap_states: int = 1
    min_lateral_shift_m: float = 0.20
    min_relation_run_states: int = 6
    excluded_target_categories: tuple[str, ...] = (
        "table", "shelter", "floor", "wall", "ceiling", "part of a cabinet/wardrobe",
    )
    min_onset_turn_deg: float = 8.0
    min_pre_gaze_gap_sec: float = 0.40
    max_pre_gaze_gap_sec: float = 1.50
    max_repeated_window_sec: float = 12.0
    target_window_sec: float = 9.0
    min_window_sec: float = 8.5
    max_window_sec: float = 10.0
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


def sustained_relation_sequence(values: Sequence[str], minimum_run: int = 6) -> list[str]:
    """Drop brief relation-label flicker and retain only sustained transitions."""
    runs: list[list[Any]] = []
    for value in values:
        if not runs or runs[-1][0] != value:
            runs.append([value, 1])
        else:
            runs[-1][1] += 1
    return _reduced([str(value) for value, count in runs if count >= minimum_run])


def _event_duration(event: Mapping[str, Any]) -> float:
    return float(event["end_time_s"]) - float(event["start_time_s"])


def _relation_run_length(
    states: Mapping[int, Mapping[str, Any]], frame: int, object_id: str,
) -> int:
    frames = sorted(states)
    if frame not in states:
        return 0
    position = frames.index(frame)
    relation = _relation(states, frame, object_id)
    if not relation:
        return 0
    label = relation["label"]
    left = position
    while left > 0 and (_relation(states, frames[left - 1], object_id) or {}).get("label") == label:
        left -= 1
    right = position
    while right + 1 < len(frames) and (_relation(states, frames[right + 1], object_id) or {}).get("label") == label:
        right += 1
    return right - left + 1


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


def _expanded_window(
    states: Mapping[int, Mapping[str, Any]], core_start: int, core_end: int, policy: Task5CandidatePolicy,
    *, align_end: bool = False, allowed_start_frame: int | None = None, allowed_end_frame: int | None = None,
) -> tuple[int, int] | None:
    """Return a real annotation-covered ~9 s window containing the core event."""
    ordered = sorted(states.values(), key=lambda state: float(state["time_s"]))
    if allowed_start_frame is not None:
        ordered = [state for state in ordered if int(state["frame_index"]) >= allowed_start_frame]
    if allowed_end_frame is not None:
        ordered = [state for state in ordered if int(state["frame_index"]) <= allowed_end_frame]
    if not ordered or core_start not in states or core_end not in states:
        return None
    times = [float(state["time_s"]) for state in ordered]
    sequence_start, sequence_end = times[0], times[-1]
    if sequence_end - sequence_start < policy.min_window_sec:
        return None
    core_start_time = float(states[core_start]["time_s"])
    core_end_time = float(states[core_end]["time_s"])
    if not sequence_start <= core_start_time <= core_end_time <= sequence_end:
        return None
    if core_end_time - core_start_time > policy.target_window_sec:
        return None
    if align_end:
        desired_end = core_end_time
        desired_start = desired_end - policy.target_window_sec
        if desired_start < sequence_start:
            return None
    else:
        center = (core_start_time + core_end_time) / 2.0
        desired_start = center - policy.target_window_sec / 2.0
        desired_end = desired_start + policy.target_window_sec
        if desired_start < sequence_start:
            desired_end += sequence_start - desired_start
            desired_start = sequence_start
        if desired_end > sequence_end:
            desired_start -= desired_end - sequence_end
            desired_end = sequence_end
        if desired_start < sequence_start:
            desired_start = sequence_start
    if desired_start < sequence_start or desired_end > sequence_end:
        return None
    start_pos = min(range(len(times)), key=lambda index: abs(times[index] - desired_start))
    end_pos = min(range(len(times)), key=lambda index: abs(times[index] - desired_end))
    start_frame = int(ordered[start_pos]["frame_index"])
    end_frame = int(ordered[end_pos]["frame_index"])
    actual_duration = times[end_pos] - times[start_pos]
    if not policy.min_window_sec <= actual_duration <= policy.max_window_sec:
        return None
    if start_frame > core_start or end_frame < core_end:
        return None
    return start_frame, end_frame


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
        object_id = str(event["object_id"])
        category = str((analysis.get("objects") or {}).get(object_id, {}).get("category", "")).lower()
        if category in policy.excluded_target_categories:
            rejection_counts["excluded_support_or_scene_target"] += 1
            continue
        if direct_hits < policy.min_event_direct_hits or support < policy.min_event_hit_support or gaps > policy.max_internal_gap_states:
            rejection_counts["event_support"] += 1
            continue
        events.append(event)

    by_object: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for event in events:
        by_object[str(event["object_id"])].append(event)

    # 1. Compare two temporally distinct sustained gazes at the same object.
    for object_id, object_events in by_object.items():
        for pair_index, (first, second) in enumerate(zip(object_events, object_events[1:])):
            if (
                int(first["direct_hit_count"]) < policy.min_repeated_event_direct_hits
                or int(second["direct_hit_count"]) < policy.min_repeated_event_direct_hits
                or _event_duration(first) < policy.min_repeated_event_duration_sec
                or _event_duration(second) < policy.min_repeated_event_duration_sec
            ):
                rejection_counts["repeated_event_too_short"] += 1
                continue
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
            if (
                left["label"] == right["label"]
                or shift < policy.min_lateral_shift_m
                or _relation_run_length(states, first_frame, object_id) < policy.min_relation_run_states
                or _relation_run_length(states, second_frame, object_id) < policy.min_relation_run_states
            ):
                rejection_counts["repeated_not_salient"] += 1
                continue
            score = shift + min(gap, 2.0) * 0.05 + min(int(first["state_count"]), int(second["state_count"])) / 100
            allowed_start = (
                int(object_events[pair_index - 1]["end_index"]) + 1 if pair_index > 0 else None
            )
            allowed_end = (
                int(object_events[pair_index + 2]["start_index"]) - 1
                if pair_index + 2 < len(object_events) else None
            )
            window = _expanded_window(
                states, int(first["start_index"]), int(second["end_index"]), policy,
                allowed_start_frame=allowed_start, allowed_end_frame=allowed_end,
            )
            if window is None:
                rejection_counts["repeated_no_target_duration_annotation_window"] += 1
                continue
            target_events_in_window = [
                event for event in object_events
                if int(event["start_index"]) >= window[0] and int(event["end_index"]) <= window[1]
            ]
            if target_events_in_window != [first, second]:
                rejection_counts["repeated_extra_target_gaze_in_public_window"] += 1
                continue
            candidates.append(_candidate(
                analysis, "between_repeated_gaze_events", object_id,
                window[0], window[1], score,
                event_start_frames=[int(first["start_index"]), int(second["start_index"])],
            ))

    # 2. Find a real pre-onset state where the wearer turns and the target changes side.
    for event in events:
        object_id = str(event["object_id"])
        if (
            int(event["direct_hit_count"]) < policy.min_onset_event_direct_hits
            or _event_duration(event) < policy.min_onset_event_duration_sec
        ):
            rejection_counts["onset_event_too_short"] += 1
            continue
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
            if (
                shift < policy.min_lateral_shift_m
                or turn < policy.min_onset_turn_deg
                or _relation_run_length(states, before_frame, object_id) < policy.min_relation_run_states
                or _relation_run_length(states, after_frame, object_id) < policy.min_relation_run_states
            ):
                continue
            viable.append((shift + math.radians(turn) * 0.15 - abs(gap - 0.8) * 0.02, before_frame))
        if not viable:
            rejection_counts["onset_no_valid_pre_state"] += 1
            continue
        score, before_frame = max(viable)
        window = _expanded_window(states, before_frame, int(event["end_index"]), policy)
        if window is None:
            rejection_counts["onset_no_target_duration_annotation_window"] += 1
            continue
        candidates.append(_candidate(
            analysis, "after_gaze_turns_to_object", object_id, window[0], window[1], score,
            event_start_frames=[int(event["start_index"])], pre_frame=before_frame,
        ))

    # 3. End a real ~9 s window on a sustained gaze, so the target is still the last gaze object.
    for event in events:
        object_id = str(event["object_id"])
        if (
            int(event["direct_hit_count"]) < policy.min_last_event_direct_hits
            or _event_duration(event) < policy.min_last_event_duration_sec
        ):
            rejection_counts["last_event_too_short"] += 1
            continue
        end = int(event["end_index"])
        expanded = _expanded_window(
            states, int(event["start_index"]), end, policy, align_end=True,
        )
        if expanded is None:
            rejection_counts["last_no_target_duration_annotation_window"] += 1
            continue
        start, end = expanded
        window = [states[index] for index in sorted(states) if start <= index <= end]
        relations = [_relation(states, int(item["frame_index"]), object_id) for item in window]
        if any(relation is None for relation in relations):
            rejection_counts["last_missing_relation"] += 1
            continue
        sequence = sustained_relation_sequence(
            [str(relation["label"]) for relation in relations if relation], policy.min_relation_run_states,
        )
        lateral = [float(relation["right_m"]) for relation in relations if relation]
        shift = max(lateral) - min(lateral)
        if not 2 <= len(sequence) <= policy.max_relation_sequence_length or shift < policy.min_lateral_shift_m:
            rejection_counts["last_no_salient_window"] += 1
            continue
        duration = float(states[end]["time_s"]) - float(states[start]["time_s"])
        score = shift + min(duration, policy.target_window_sec) * 0.01 - max(0, len(sequence) - 3) * 0.05
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
