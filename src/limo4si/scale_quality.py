"""Reusable release gates for scaling Task 1, Task 3, and Task 4 QA generation.

The validator is deliberately independent of the curated-case builders.  A
large-scale miner can pass any release-shaped dictionary and receive per-case
accept/reject reasons instead of silently publishing weak or inconsistent QA.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

TASK1_ID = "task1_dynamic_human_referenced_relations"
TASK3_ID = "task3_human_scene_topological_reasoning"
TASK4_ID = "task4_multi_human_relational_dynamics"
TOPOLOGY_TYPES = {
    "visible_pair_topology_change_2d",
    "visible_pair_topology_consistency_2d",
}
DOMINANCE_TYPES = {
    "dominant_facing_relation_over_video": "facing_counts",
    "dominant_body_centric_position": "relation_counts",
    "approach_while_facing": "facing_counts",
}
CONSISTENCY_TYPES = {
    "position_consistency_between_people": "relation_counts",
    "body_forward_visibility_consistency": "body_forward_field_counts",
}
SEQUENCE_TYPES = {
    "body_centric_relation_change_over_video": "relation_sequence",
    "coupled_distance_relation_change": "relation_sequence",
    "body_forward_field_transition_over_video": "body_forward_field_sequence",
}
SAMPLE_WORDING = re.compile(r"\b(?:sample|samples|sampled)\b|\b\d+\s*/\s*\d+\b", re.I)


@dataclass(frozen=True)
class ScaleQualityPolicy:
    min_temporal_states: int = 8
    min_task1_span_ratio: float = 0.60
    min_task4_span_ratio: float = 0.85
    min_task3_span_ratio: float = 0.85
    min_task3_path_length_m: float = 0.75
    min_task3_route_margin_m: float = 0.15
    min_task3_order_gap_sec: float = 2.0
    min_task3_side_margin_m: float = 0.25
    min_task3_local_travel_m: float = 0.35
    min_task3_grounding_inliers: int = 12
    min_visible_track_coverage: float = 0.80
    min_identity_assignment_margin: float = 0.18
    min_dominance_ratio: float = 0.65
    min_dominance_margin: float = 0.20
    max_radial_speed_mps: float = 3.0
    max_temporal_gap_ratio: float = 0.30
    min_transition_run_length: int = 2
    min_distance_pattern_range_m: float = 0.25
    min_topology_margin_normalized: float = 0.03
    endpoint_tolerance_sec: float = 0.60
    forbid_sample_count_in_answers: bool = True
    # Options are released only when they have comparable semantic payload.
    # Length is a proxy; the numeric and temporal/relation slot checks below
    # catch more direct answer cues.
    max_option_information_ratio: float = 1.25
    max_option_word_count_ratio: float = 1.30
    max_option_numeric_count_gap: int = 0
    max_option_temporal_marker_gap: int = 1
    max_option_relation_marker_gap: int = 2
    max_published_distance_decimals: int = 1


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _walk_nonfinite(value: Any, path: str = "result_json") -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_nonfinite(item, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        yield path


def _time(state: Mapping[str, Any]) -> float | None:
    for key in ("t", "t_sec_from_center", "time_s"):
        value = state.get(key)
        if _finite(value):
            return float(value)
    return None


def _best_task1_states(result: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    candidates: list[list[Mapping[str, Any]]] = []
    for key in ("object_track", "visibility_track"):
        track = result.get(key)
        if isinstance(track, Mapping) and isinstance(track.get("states"), list):
            candidates.append(track["states"])
    if isinstance(result.get("consistency_track"), list):
        candidates.append(result["consistency_track"])
    for track in result.get("object_tracks") or []:
        if isinstance(track, Mapping) and isinstance(track.get("states"), list):
            candidates.append(track["states"])
    return max(candidates, key=len, default=[])


def _validate_time_series(
    states: Sequence[Mapping[str, Any]], duration: float | None,
    min_span_ratio: float, policy: ScaleQualityPolicy, errors: list[str], metrics: dict[str, Any],
) -> None:
    metrics["temporal_state_count"] = len(states)
    if len(states) < policy.min_temporal_states:
        errors.append(f"temporal coverage has {len(states)} states; need at least {policy.min_temporal_states}")
        return
    times = [_time(state) for state in states]
    if any(value is None for value in times):
        errors.append("temporal states contain missing/non-finite timestamps")
        return
    numeric_times = [float(value) for value in times if value is not None]
    if any(b <= a for a, b in zip(numeric_times, numeric_times[1:])):
        errors.append("temporal timestamps are not strictly increasing")
    span = numeric_times[-1] - numeric_times[0]
    metrics["temporal_span_sec"] = round(span, 6)
    gaps = [b - a for a, b in zip(numeric_times, numeric_times[1:])]
    if span > 0 and gaps:
        gap_ratio = max(gaps) / span
        metrics["max_temporal_gap_ratio"] = round(gap_ratio, 6)
        if gap_ratio > policy.max_temporal_gap_ratio:
            errors.append(f"largest temporal gap ratio {gap_ratio:.3f} exceeds {policy.max_temporal_gap_ratio:.3f}")
    if duration and duration > 0:
        ratio = span / duration
        metrics["temporal_span_ratio"] = round(ratio, 6)
        if ratio < min_span_ratio:
            errors.append(f"temporal span ratio {ratio:.3f} is below {min_span_ratio:.3f}")
    frames = [state.get("frame", state.get("frame_id")) for state in states]
    frames = [frame for frame in frames if frame is not None]
    if len(frames) != len(set(map(str, frames))):
        errors.append("temporal evidence contains duplicate frame identifiers")



def option_information_profile(text: str) -> dict[str, int]:
    """Approximate answer-choice information density, not just character length."""
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*|\d+(?:\.\d+)?", text)
    numbers = re.findall(r"(?<![A-Za-z])\d+(?:\.\d+)?", text)
    lowered = text.lower()
    temporal_markers = sum(lowered.count(token) for token in (
        "start", "begin", "early", "middle", "then", "later", "end",
        "throughout", "entire clip", "most of", "remain", "while",
    ))
    relation_markers = sum(lowered.count(token) for token in (
        "left", "right", "front", "behind", "closer", "farther", "approach",
        "separate", "facing", "back-to-back", "side-by-side", "inside", "outside",
    ))
    information_units = len(words) + 2 * len(numbers) + temporal_markers + relation_markers
    return {
        "words": len(words),
        "numbers": len(numbers),
        "temporal_markers": temporal_markers,
        "relation_markers": relation_markers,
        "information_units": information_units,
    }


def _distance_decimal_places(text: str) -> list[int]:
    return [len(match.group(1) or "") for match in re.finditer(r"\b\d+(?:\.(\d+))?\s*m\b", text)]

def _validate_common(case_id: str, question: Mapping[str, Any], policy: ScaleQualityPolicy, errors: list[str], metrics: dict[str, Any]) -> None:
    options = question.get("options") or []
    if len(options) != 4:
        errors.append("question must have exactly four options")
        return
    labels = [option.get("label") for option in options]
    texts = [option.get("text") for option in options]
    if len(set(labels)) != 4 or len(set(texts)) != 4:
        errors.append("option labels and texts must be unique")
    profiles = [option_information_profile(str(text)) for text in texts]
    units = [profile["information_units"] for profile in profiles]
    word_counts = [profile["words"] for profile in profiles]
    numeric_counts = [profile["numbers"] for profile in profiles]
    temporal_counts = [profile["temporal_markers"] for profile in profiles]
    relation_counts = [profile["relation_markers"] for profile in profiles]
    metrics["option_information_profiles"] = profiles
    if units and min(units) > 0 and max(units) / min(units) > policy.max_option_information_ratio:
        errors.append(f"option information ratio {max(units) / min(units):.2f} exceeds {policy.max_option_information_ratio:.2f}")
    if word_counts and min(word_counts) > 0 and max(word_counts) / min(word_counts) > policy.max_option_word_count_ratio:
        errors.append(f"option word-count ratio {max(word_counts) / min(word_counts):.2f} exceeds {policy.max_option_word_count_ratio:.2f}")
    if numeric_counts and max(numeric_counts) - min(numeric_counts) > policy.max_option_numeric_count_gap:
        errors.append("answer choices expose unequal numeric detail")
    if temporal_counts and max(temporal_counts) - min(temporal_counts) > policy.max_option_temporal_marker_gap:
        errors.append("answer choices expose unequal temporal detail")
    if relation_counts and max(relation_counts) - min(relation_counts) > policy.max_option_relation_marker_gap:
        errors.append("answer choices expose unequal relation detail")
    decimals = [places for text in texts for places in _distance_decimal_places(str(text))]
    decimals.extend(_distance_decimal_places(str(question.get("correct_answer", ""))))
    if decimals and max(decimals) > policy.max_published_distance_decimals:
        errors.append("published distance uses more precision than the release policy allows")
    correct_label = question.get("correct_option")
    matches = [option for option in options if option.get("label") == correct_label]
    if len(matches) != 1:
        errors.append("correct_option does not resolve to exactly one option")
    else:
        correct = question.get("correct_answer")
        if matches[0].get("text") != correct:
            errors.append("correct option text is stale or differs from correct_answer")
        if question.get("answer") != correct:
            errors.append("answer differs from correct_answer")
    result = question.get("result_json")
    if not isinstance(result, Mapping):
        errors.append("result_json is missing")
        return
    if result.get("answer_type") != question.get("question_type"):
        errors.append("result_json.answer_type differs from question_type")
    for flag in ("T_Q", "H_Q", "S_Q"):
        if result.get(flag) is not True:
            errors.append(f"result_json.{flag} must be true")
    nonfinite = list(_walk_nonfinite(result))
    if nonfinite:
        errors.append("non-finite numeric evidence at " + ", ".join(nonfinite[:3]))
    if question.get("task_id") == TASK4_ID:
        published = [
            question.get("question", ""), question.get("answer", ""),
            question.get("correct_answer", ""), *texts,
        ]
        if policy.forbid_sample_count_in_answers and any(SAMPLE_WORDING.search(str(value)) for value in published[1:]):
            errors.append("published Task 4 answer/options contain sample-count wording")
        if any(re.search(r"\b(?:A|B|V1|V2|V3)\b", str(value)) for value in published):
            errors.append("published Task 4 text leaks an internal person ID")


def _validate_task1(group: Mapping[str, Any], question: Mapping[str, Any], policy: ScaleQualityPolicy, errors: list[str], warnings: list[str], metrics: dict[str, Any]) -> None:
    result = question["result_json"]
    states = _best_task1_states(result)
    duration = (group.get("video_window") or {}).get("duration_sec")
    _validate_time_series(states, float(duration) if _finite(duration) else None, policy.min_task1_span_ratio, policy, errors, metrics)
    relations = [state.get("relation") for state in states if isinstance(state.get("relation"), Mapping)]
    for relation in relations:
        distance = relation.get("distance_m")
        if not _finite(distance) or float(distance) < 0:
            errors.append("Task 1 relation contains invalid distance")
            break
        xyz = relation.get("human_xyz_m") or {}
        components = [xyz.get("right"), xyz.get("up"), xyz.get("forward")]
        if not all(_finite(value) for value in components):
            errors.append("Task 1 relation lacks finite human-frame coordinates")
            break
        reconstructed = math.sqrt(sum(float(value) ** 2 for value in components))
        if abs(reconstructed - float(distance)) > 0.02:
            errors.append("Task 1 distance is inconsistent with human-frame coordinates")
            break
    forward_sensitive = question.get("question_type") in {"multi_object_front_consistency_over_video"} or any(
        word in (str(question.get("question", "")) + " " + str(question.get("correct_answer", ""))).lower()
        for word in ("front", "behind")
    )
    if forward_sensitive:
        signs = [(state.get("orientation") or {}).get("forward_sign") for state in states]
        if not signs or any(sign not in (-1, 1) for sign in signs):
            errors.append("front/behind Task 1 QA lacks a valid orientation sign on every state")
        elif len(set(signs)) != 1:
            errors.append("orientation sign changes within a single take")
        else:
            metrics["orientation_forward_sign"] = signs[0]
    elif states and not any(state.get("orientation") for state in states):
        warnings.append("orientation audit metadata is absent but this QA is not front/behind-sensitive")


def _validate_person_descriptions(aliases: Mapping[str, Any], person_ids: Sequence[str], errors: list[str]) -> list[str]:
    descriptions = [str(aliases.get(person_id, "")).strip() for person_id in person_ids]
    if any(not value for value in descriptions) or len(set(value.lower() for value in descriptions)) != len(descriptions):
        errors.append("public person descriptions must be present and pairwise distinct")
        return descriptions
    gender_words = [re.findall(r"\b(?:man|woman)\b", value.lower()) for value in descriptions]
    if any(not words for words in gender_words):
        errors.append("each public person description must include an unambiguous man/woman noun")
        return descriptions
    genders = [words[-1] for words in gender_words]
    if len(set(genders)) < len(genders):
        # Same-gender people need an additional visible cue.  Clothing is the
        # current release convention; future extractors may emit another cue
        # after "with" or "wearing".
        if any(not re.search(r"\b(?:in|wearing|with)\b", value.lower()) for value in descriptions):
            errors.append("same-gender people require distinct visible appearance cues")
    return descriptions


def _validate_identity_audit(group: Mapping[str, Any], errors: list[str], metrics: dict[str, Any], policy: ScaleQualityPolicy) -> None:
    audit = group.get("visual_person_audit") or {}
    if audit.get("status") != "complete_and_identity_aligned":
        errors.append(f"metric Task 4 identity status is {audit.get('status')!r}")
        return
    alignment = audit.get("metric_identity_alignment") or {}
    mapping = alignment.get("mapping") or {}
    if set(mapping) != {"A", "B"} or len(set(mapping.values())) != 2:
        errors.append("metric-to-visible identity mapping must contain distinct A and B tracks")
    margin = alignment.get("margin")
    if not _finite(margin) or float(margin) < policy.min_identity_assignment_margin:
        errors.append("visual/metric identity assignment margin is below threshold")
    else:
        metrics["identity_assignment_margin"] = round(float(margin), 6)
    # Calibration output may relabel a matched V-track to its metric A/B ID.
    visible = {track.get("id"): track for track in audit.get("visible_2d_tracks") or []}
    for metric_id, visible_id in mapping.items():
        track = visible.get(visible_id) or visible.get(metric_id)
        if not track:
            errors.append(f"identity mapping references missing visible track {visible_id}/{metric_id}")
            continue
        coverage = track.get("coverage")
        if not _finite(coverage) or float(coverage) < policy.min_visible_track_coverage:
            errors.append(f"visible track {visible_id} coverage is below threshold")
    metrics["identity_mapping"] = mapping


def _validate_metric_task4(group: Mapping[str, Any], question: Mapping[str, Any], policy: ScaleQualityPolicy, errors: list[str], warnings: list[str], metrics: dict[str, Any]) -> None:
    _validate_identity_audit(group, errors, metrics, policy)
    audit = group.get("visual_person_audit") or {}
    aliases = group.get("person_display_aliases") or {}
    pair_descriptions = _validate_person_descriptions(aliases, ("A", "B"), errors)
    persistent_count = int(audit.get("persistent_visible_person_count") or 0)
    metric_count = int(audit.get("metric_3d_track_count") or 0)
    max_visible_count = int(audit.get("max_visible_person_count") or persistent_count)
    metrics["visible_person_counts"] = {
        "persistent": persistent_count, "maximum": max_visible_count, "metric_3d": metric_count,
    }
    if persistent_count > metric_count:
        errors.append("persistent visible people outnumber metric 3D annotations")
    if max_visible_count > metric_count:
        public_question = str(question.get("question", "")).lower()
        if not all(description.lower() in public_question for description in pair_descriptions):
            errors.append("an intermittent unannotated person makes the unnamed metric pair ambiguous")
        else:
            warnings.append("an intermittent extra person is excluded; the question explicitly names the annotated pair")
    result = question["result_json"]
    timeline = result.get("pair_timeline") or {}
    if timeline.get("status") != "ok":
        errors.append("pair_timeline status is not ok")
    states = timeline.get("states") or []
    duration = (group.get("video_window") or {}).get("duration_sec")
    _validate_time_series(states, float(duration) if _finite(duration) else None, policy.min_task4_span_ratio, policy, errors, metrics)
    if timeline.get("pair") != ["A", "B"]:
        errors.append("metric Task 4 timeline must use the aligned A/B pair")
    distances = []
    speeds = []
    for state in states:
        distance = state.get("distance_m")
        score = state.get("facing_score")
        if not _finite(distance) or float(distance) < 0:
            errors.append("pair timeline contains invalid distance")
            break
        distances.append(float(distance))
        if not _finite(score) or not -1.000001 <= float(score) <= 1.000001:
            errors.append("pair timeline contains invalid facing score")
            break
        evidence = state.get("evidence") or {}
        people = [evidence.get("person_a") or {}, evidence.get("person_b") or {}]
        pelvises = [person.get("pelvis_xyz_m") for person in people]
        forwards = [person.get("forward_unit") for person in people]
        if not all(isinstance(vector, list) and len(vector) == 3 and all(_finite(value) for value in vector) for vector in pelvises + forwards):
            errors.append("pair timeline lacks finite pelvis/forward evidence vectors")
            break
        if any(not 0.95 <= math.sqrt(sum(float(value) ** 2 for value in vector)) <= 1.05 for vector in forwards):
            errors.append("pair timeline contains a degenerate or non-unit forward vector")
            break
        reconstructed = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(pelvises[0], pelvises[1])))
        if abs(reconstructed - float(distance)) > 0.02:
            errors.append("pair distance is inconsistent with pelvis evidence")
            break
    for left, right in zip(states, states[1:]):
        ta, tb = _time(left), _time(right)
        if ta is not None and tb is not None and tb > ta and _finite(left.get("distance_m")) and _finite(right.get("distance_m")):
            speeds.append(abs(float(right["distance_m"]) - float(left["distance_m"])) / (tb - ta))
    if speeds:
        metrics["max_radial_speed_mps"] = round(max(speeds), 6)
        if max(speeds) > policy.max_radial_speed_mps:
            errors.append(f"radial distance jump {max(speeds):.2f} m/s exceeds threshold")
    stored_distances = result.get("distance_series_m")
    if stored_distances is not None:
        if len(stored_distances) != len(distances) or any(not _finite(value) for value in stored_distances):
            errors.append("distance_series_m is missing values or differs in length from timeline")
        elif any(abs(float(a) - b) > 1e-6 for a, b in zip(stored_distances, distances)):
            errors.append("distance_series_m is stale and differs from pair_timeline")
    qtype = question.get("question_type")
    distance_pattern_types = {
        "metric_distance_pattern_over_video", "metric_separation_over_video",
        "nonmonotonic_distance_pattern", "approach_while_facing",
        "coupled_distance_relation_change", "distance_out_and_back_over_video",
    }
    if qtype in distance_pattern_types and distances:
        distance_range = max(distances) - min(distances)
        metrics["distance_range_m"] = round(distance_range, 6)
        if distance_range < policy.min_distance_pattern_range_m:
            errors.append("distance change is too small for a salient temporal-pattern question")
    count_key = DOMINANCE_TYPES.get(qtype)
    if count_key:
        counts = result.get(count_key) or {}
        values = sorted((int(v) for v in counts.values()), reverse=True)
        if not values or sum(values) != len(states):
            errors.append(f"{count_key} does not account for every timeline state")
        else:
            dominance = values[0] / len(states)
            runner_up = values[1] / len(states) if len(values) > 1 else 0.0
            metrics["dominance_ratio"] = round(dominance, 6)
            metrics["dominance_margin"] = round(dominance - runner_up, 6)
            if dominance < policy.min_dominance_ratio or dominance - runner_up < policy.min_dominance_margin:
                errors.append("dominant relation is too ambiguous for a high-quality question")
    count_key = CONSISTENCY_TYPES.get(qtype)
    if count_key:
        counts = result.get(count_key) or {}
        if sum(int(v) for v in counts.values()) != len(states) or max((int(v) for v in counts.values()), default=0) != len(states):
            errors.append("consistency claim is not supported by every timeline state")
    sequence_key = SEQUENCE_TYPES.get(qtype)
    if sequence_key:
        sequence = result.get(sequence_key) or []
        if len(sequence) != len(states):
            errors.append(f"{sequence_key} length differs from pair_timeline")
        elif len(set(sequence)) < 2:
            errors.append("relation-change question contains no actual state change")
        else:
            segment = max(policy.min_transition_run_length, len(sequence) // 3)
            start_support = sum(value == sequence[0] for value in sequence[:segment])
            end_support = sum(value == sequence[-1] for value in sequence[-segment:])
            metrics["transition_endpoint_segment_support"] = [start_support, end_support]
            if min(start_support, end_support) < policy.min_transition_run_length:
                errors.append("relation transition lacks repeated support in an endpoint segment")


def _validate_task3(group: Mapping[str, Any], question: Mapping[str, Any], policy: ScaleQualityPolicy, errors: list[str], metrics: dict[str, Any]) -> None:
    result = question["result_json"]
    topology = result.get("topology") or {}
    if topology.get("status") != "ok":
        errors.append("Task 3 topology status is not ok")
        return
    states = topology.get("trajectory_states") or []
    duration = (group.get("video_window") or {}).get("duration_sec")
    _validate_time_series(states, float(duration) if _finite(duration) else None, policy.min_task3_span_ratio, policy, errors, metrics)
    path_length = topology.get("path_length_m")
    if not _finite(path_length) or float(path_length) < policy.min_task3_path_length_m:
        errors.append("Task 3 path is too short for trajectory-topology reasoning")
    else:
        metrics["path_length_m"] = round(float(path_length), 6)
    speed = topology.get("max_smoothed_speed_mps")
    if not _finite(speed) or float(speed) > policy.max_radial_speed_mps:
        errors.append("Task 3 smoothed trajectory contains an implausible speed")
    landmark_rows = {row.get("landmark_id"): row for row in topology.get("landmarks") or []}

    def validate_static_landmarks(ids: Sequence[str]) -> None:
        for landmark_id in ids:
            row = landmark_rows.get(landmark_id) or {}
            grounding = row.get("grounding") or {}
            if row.get("static_scene_landmark") is not True:
                errors.append(f"Task 3 landmark {landmark_id!r} is not audited as scene-fixed")
            if grounding.get("manual_static_review") is not True or grounding.get("centroid_reprojects_inside_box") is not True:
                errors.append(f"Task 3 landmark {landmark_id!r} lacks complete visual/reprojection audit")
            inliers = grounding.get("robust_inlier_points")
            if not _finite(inliers) or int(inliers) < policy.min_task3_grounding_inliers:
                errors.append(f"Task 3 landmark {landmark_id!r} has insufficient metric grounding inliers")

    qtype = question.get("question_type")
    if qtype == "local_landmark_pass_side":
        event = result.get("pass_event") or {}
        validate_static_landmarks([str(event.get("landmark_id"))])
        if event.get("valid_local_pass") is not True:
            errors.append("Task 3 local-pass event does not pass its geometry gate")
        if not _finite(event.get("signed_lateral_m")) or abs(float(event["signed_lateral_m"])) < policy.min_task3_side_margin_m:
            errors.append("Task 3 local-pass side margin is too small")
        if not _finite(event.get("local_travel_m")) or float(event["local_travel_m"]) < policy.min_task3_local_travel_m:
            errors.append("Task 3 local-pass window contains too little travel")
        if not _finite(event.get("side_support_ratio")) or float(event["side_support_ratio"]) < 2 / 3:
            errors.append("Task 3 local-pass side lacks repeated temporal support")
    elif qtype == "landmark_closest_approach_order":
        event = result.get("order_event") or {}
        ordered_ids = [str(event.get("first_landmark")), str(event.get("second_landmark"))]
        validate_static_landmarks(ordered_ids)
        if any((landmark_rows.get(landmark_id) or {}).get("valid_visit") is not True for landmark_id in ordered_ids):
            errors.append("Task 3 landmark-order evidence lacks two prominent interior visits")
        if not _finite(event.get("time_gap_sec")) or float(event["time_gap_sec"]) < policy.min_task3_order_gap_sec:
            errors.append("Task 3 landmark-order events are not sufficiently separated in time")
        if not _finite(event.get("center_separation_m")) or float(event["center_separation_m"]) < policy.min_task3_side_margin_m:
            errors.append("Task 3 ordered landmarks are not spatially distinct")
        threshold = (topology.get("thresholds") or {}).get("max_landmark_distance_m", 2.0)
        if not _finite(event.get("max_closest_distance_m")) or float(event["max_closest_distance_m"]) > float(threshold):
            errors.append("Task 3 ordered landmark is too far from the trajectory")
    elif qtype == "closest_landmark_to_full_trajectory":
        ranking = result.get("route_landmark_ranking") or []
        validate_static_landmarks([str(row.get("landmark_id")) for row in ranking[:3]])
        if len(ranking) < 3:
            errors.append("Task 3 full-route ranking needs at least three static landmarks")
        else:
            margin = float(ranking[1]["minimum_horizontal_distance_m"]) - float(ranking[0]["minimum_horizontal_distance_m"])
            metrics["route_landmark_margin_m"] = round(margin, 6)
            if margin < policy.min_task3_route_margin_m:
                errors.append("Task 3 closest-route landmark margin is too small")
    else:
        errors.append(f"unsupported Task 3 question type {qtype!r}")


def _validate_topology(group: Mapping[str, Any], question: Mapping[str, Any], policy: ScaleQualityPolicy, errors: list[str], metrics: dict[str, Any]) -> None:
    result = question["result_json"]
    start = result.get("start_pair_distances_normalized") or []
    end = result.get("end_pair_distances_normalized") or []
    def parsed(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        out: dict[str, float] = {}
        for row in rows:
            pair, value = row.get("pair"), row.get("distance")
            if not isinstance(pair, str) or not _finite(value) or float(value) < 0 or pair in out:
                errors.append("topology evidence contains invalid or duplicate pair distances")
                continue
            out[pair] = float(value)
        return out
    start_map, end_map = parsed(start), parsed(end)
    if len(start_map) < 3 or set(start_map) != set(end_map):
        errors.append("topology start/end evidence must contain the same three or more pairs")
    topology_ids = sorted({person_id for pair in start_map for person_id in re.split(r"[–-]", pair)})
    _validate_person_descriptions(group.get("person_display_aliases") or {}, topology_ids, errors)
    for label, values in (("start", start_map), ("end", end_map)):
        ordered = sorted(values.values())
        if len(ordered) >= 2:
            margin = ordered[1] - ordered[0]
            metrics[f"{label}_closest_pair_margin"] = round(margin, 6)
            if margin < policy.min_topology_margin_normalized:
                errors.append(f"{label} closest pair is ambiguous (margin {margin:.3f})")
    audit = group.get("visual_person_audit") or {}
    tracks = audit.get("visible_2d_tracks") or []
    if int(audit.get("persistent_visible_person_count") or 0) < 3 or len(tracks) < 3:
        errors.append("topology QA requires at least three persistent visible tracks")
    duration = audit.get("duration_sec") or (group.get("video_window") or {}).get("duration_sec") or group.get("duration_sec")
    endpoint_gate = audit.get("endpoint_gate") or {}
    explicit_ok = endpoint_gate.get("all_tracks_have_start_observation") is True and endpoint_gate.get("all_tracks_have_end_observation") is True
    inferred_ok = bool(tracks) and _finite(duration) and all(
        _finite(track.get("coverage"))
        and float(track["coverage"]) >= policy.min_visible_track_coverage
        and _finite(track.get("first_seen_sec"))
        and float(track["first_seen_sec"]) <= policy.endpoint_tolerance_sec
        and _finite(track.get("last_seen_sec"))
        and float(duration) - float(track["last_seen_sec"]) <= policy.endpoint_tolerance_sec
        for track in tracks
    )
    if not (explicit_ok or inferred_ok):
        errors.append("topology tracks lack real observations near both endpoints")


def validate_release(data: Mapping[str, Any], policy: ScaleQualityPolicy | None = None) -> dict[str, Any]:
    """Return a structured per-case gate report without mutating the release."""
    policy = policy or ScaleQualityPolicy()
    reports = []
    seen: set[str] = set()
    groups = data.get("groups") or []
    for index, group in enumerate(groups, 1):
        case_id = str(group.get("name") or f"case_{index}")
        errors: list[str] = []
        warnings: list[str] = []
        metrics: dict[str, Any] = {}
        if case_id in seen:
            errors.append("duplicate case/window name")
        seen.add(case_id)
        questions = group.get("qa") or []
        if len(questions) != 1:
            errors.append("case must contain exactly one question")
            question: Mapping[str, Any] = {}
        else:
            question = questions[0]
            _validate_common(case_id, question, policy, errors, metrics)
            task_id = question.get("task_id")
            if isinstance(question.get("result_json"), Mapping):
                if task_id == TASK1_ID:
                    _validate_task1(group, question, policy, errors, warnings, metrics)
                elif task_id == TASK3_ID:
                    _validate_task3(group, question, policy, errors, metrics)
                elif task_id == TASK4_ID:
                    if question.get("question_type") in TOPOLOGY_TYPES:
                        _validate_topology(group, question, policy, errors, metrics)
                    else:
                        _validate_metric_task4(group, question, policy, errors, warnings, metrics)
                else:
                    errors.append(f"unsupported task_id {task_id!r}")
        reports.append({
            "case_index": index,
            "case_id": case_id,
            "task_id": question.get("task_id"),
            "question_type": question.get("question_type"),
            "status": "rejected" if errors else "accepted",
            "errors": errors,
            "warnings": warnings,
            "metrics": metrics,
        })
    rejected = [report for report in reports if report["status"] == "rejected"]
    return {
        "status": "ok" if not rejected else "failed",
        "policy": asdict(policy),
        "case_count": len(reports),
        "accepted_count": len(reports) - len(rejected),
        "rejected_count": len(rejected),
        "warning_count": sum(len(report["warnings"]) for report in reports),
        "cases": reports,
    }


def require_release_quality(data: Mapping[str, Any], policy: ScaleQualityPolicy | None = None) -> dict[str, Any]:
    """Validate a release and raise with concise case-specific rejection reasons."""
    report = validate_release(data, policy)
    if report["status"] != "ok":
        reasons = []
        for case in report["cases"]:
            if case["status"] == "rejected":
                reasons.append(f"{case['case_id']}: {', '.join(case['errors'])}")
        raise ValueError("scale quality gate failed; " + "; ".join(reasons))
    return report
