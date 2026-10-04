"""Balanced, annotation-only scaling for the seven canonical Task 4 capabilities."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .multihuman import (
    derive_task4_answer_semantics,
    distance_evolution_pattern,
    multi_person_metric_timeline,
    pair_timeline,
)
from .scale_quality import TASK4_ID
from .semantic_gt import LanguageRealizer, seal_deterministic_question
from .task4_annotation import (
    TASK4_NAME,
    AnnotationEvidenceError,
    Task4GenerationPolicy,
    annotation_identity_evidence,
    build_balanced_options,
    dominant_relation,
    facing_relation_text,
    validate_human_coordinate_frame,
)
from .task4_contract import TASK4_CAPABILITIES, capability_for
from .task4_dynamics import (
    passing_side_and_final_position,
    relation_change_cause,
    reunion_relation_restoration,
)


@dataclass(frozen=True)
class Task4ScalePolicy:
    target_count: int = 40
    minimum_window_sec: float = 14.5
    maximum_window_sec: float = 15.5
    min_distance_change_m: float = 0.25
    min_group_pair_margin_m: float = 0.05
    min_transition_run_states: int = 2
    generation: Task4GenerationPolicy = Task4GenerationPolicy()


def balanced_category_targets(total: int) -> dict[str, int]:
    """Split a total target across all seven categories with at most one-count skew."""
    if total < len(TASK4_CAPABILITIES):
        raise ValueError(f"Task 4 target must be at least {len(TASK4_CAPABILITIES)}")
    quotient, remainder = divmod(total, len(TASK4_CAPABILITIES))
    return {
        category: quotient + (index < remainder)
        for index, category in enumerate(TASK4_CAPABILITIES)
    }


def _sentence(value: str) -> str:
    return value[:1].upper() + value[1:]


def _relation(value: str) -> str:
    return value.replace("_", "-")


def _base(scene: Mapping[str, Any], policy: Task4ScalePolicy) -> tuple[dict[str, Any], list[Mapping[str, Any]], dict[str, str], dict[str, Any], dict[str, Any]]:
    duration = float(scene.get("duration_sec") or 0)
    if not policy.minimum_window_sec <= duration <= policy.maximum_window_sec:
        raise AnnotationEvidenceError(
            f"public window duration {duration:g}s is outside "
            f"{policy.minimum_window_sec:g}–{policy.maximum_window_sec:g}s"
        )
    timeline = pair_timeline(scene)
    states = timeline.get("states") or []
    if timeline.get("status") != "ok" or len(states) < policy.generation.min_states:
        raise AnnotationEvidenceError("insufficient aligned person-pair temporal states")
    times = [float(state["t"]) for state in states]
    if (times[-1] - times[0]) / duration < policy.generation.min_span_ratio:
        raise AnnotationEvidenceError("annotation time coverage is below threshold")
    aliases, audit, alias_status = annotation_identity_evidence(scene)
    return timeline, states, aliases, audit, alias_status




def _identity_for_group(
    scene: Mapping[str, Any], person_ids: Sequence[str], aliases: dict[str, str],
    audit: dict[str, Any], alias_status: dict[str, Any],
) -> None:
    configured = scene.get("person_identities") or {}
    ordinals = ("first", "second", "third", "fourth", "fifth", "sixth")
    for index, person_id in enumerate(person_ids):
        if person_id not in aliases:
            aliases[person_id] = str(
                configured.get(person_id)
                or f"the {ordinals[index] if index < len(ordinals) else str(index + 1) + 'th'} annotated person"
            )
        if aliases[person_id].lower() in {
            value.lower() for key, value in aliases.items() if key != person_id
        }:
            raise AnnotationEvidenceError("group person identities are not pairwise distinct")
    audit["persistent_visible_person_count"] = len(person_ids)
    audit["metric_3d_track_count"] = len(person_ids)
    audit["visible_2d_tracks"] = [
        {
            "id": person_id,
            "coverage": sum(
                any(str(person.get("id")) == person_id for person in frame.get("people", []))
                for frame in scene.get("frames", [])
            ) / len(scene.get("frames", [])),
        }
        for person_id in person_ids
    ]
    alias_status["configured_descriptors"] = dict(aliases)


def _pair_text(pair: str, aliases: Mapping[str, str]) -> str:
    left, right = pair.split("–", 1)
    return f"{aliases[left]} and {aliases[right]}"


def _seal(
    scene: Mapping[str, Any], qtype: str, question_text: str, correct: str,
    alternatives: list[str], result: dict[str, Any], explanation: str, method: str,
    aliases: dict[str, str], audit: dict[str, Any], alias_status: dict[str, Any],
    realizer: LanguageRealizer | None,
    compound_parts_by_text: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    options, label = build_balanced_options(correct, alternatives, str(scene["scene_id"]) + qtype)
    result.update({
        "scene_id": scene["scene_id"],
        "answer_type": qtype,
        "T_Q": True,
        "H_Q": True,
        "S_Q": True,
    })
    if compound_parts_by_text:
        result["compound_option_parts"] = {
            option["label"]: list(compound_parts_by_text[option["text"]])
            for option in options
        }
    result["answer_semantics"] = derive_task4_answer_semantics(qtype, result)
    question = seal_deterministic_question({
        "task_id": TASK4_ID,
        "task_name": TASK4_NAME,
        "question_type": qtype,
        "question": question_text,
        "options": options,
        "correct_option": label,
        "correct_answer": correct,
        "answer": correct,
        "explanation": explanation,
        "method": method,
        "status": "ok",
        "release_eligible": True,
        "result_json": result,
    }, case_id=f"{scene['scene_id']}::{qtype}", realizer=realizer, provenance={
        "generator": "limo4si.task4_scaling.generate_task4_candidates",
        "reasoning_owner": "deterministic_code",
    })
    return {
        "name": str(scene["scene_id"]),
        "title": str(scene.get("title") or scene["scene_id"]),
        "dataset": str(scene.get("dataset") or "annotation bundle"),
        "video_clip": scene.get("video_clip"),
        "video_window": {
            "start_sec": float(scene.get("start_sec") or 0),
            "duration_sec": float(scene["duration_sec"]),
            "metric_sample_count": len((result.get("pair_timeline") or {}).get("states") or []),
            "source": scene.get("source_file") or scene.get("source_video"),
        },
        "visual_person_audit": audit,
        "person_display_aliases": aliases,
        "person_display_alias_status": alias_status,
        "case_policy": "one deterministic question per non-overlapping annotation window",
        "qa": [question],
        "_candidate_score": 0.0,
    }


def _distance_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, states, aliases, audit, alias_status = base
    analysis = distance_evolution_pattern(states, policy.min_distance_change_m)
    pattern, score = str(analysis["pattern"]), float(analysis["salience_m"])
    texts = {
        "continuous_closer": "They move closer overall across the clip.",
        "continuous_farther": "They move farther apart across the clip.",
        "closer_then_farther": "They first move closer, then farther apart.",
        "farther_then_closer": "They first move farther, then closer together.",
    }
    correct = texts[pattern]
    candidate = _seal(
        scene, "metric_distance_pattern_over_video",
        f"Across the clip, how does the distance between {aliases['A']} and {aliases['B']} change?",
        correct, [text for key, text in texts.items() if key != pattern],
        {
            "pair_timeline": timeline,
            "distance_series_m": [float(state["distance_m"]) for state in states],
            "distance_pattern": pattern,
        },
        "The pattern is computed from the complete metric person-pair distance timeline.",
        "Uses only annotation-derived pelvis distances and sustained temporal extrema.",
        aliases, audit, alias_status, realizer,
    )
    candidate["_candidate_score"] = score
    return candidate


def _passing_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, states, aliases, audit, alias_status = base
    validate_human_coordinate_frame(scene)
    analysis = passing_side_and_final_position(states)
    a, b = aliases["A"], aliases["B"]
    side = str(analysis["passing_side"])
    other_side = "left" if side == "right" else "right"
    final_relation = _relation(str(analysis["final_relation"]))
    start_relation = _relation(str(analysis["start_relation"]))

    def text(pass_side: str, relation: str) -> str:
        return f"{_sentence(a)} passes on {b}'s {pass_side} side and finishes {relation} relative to {b}."

    correct = text(side, final_relation)
    alternatives = [
        text(side, start_relation),
        text(other_side, final_relation),
        text(other_side, start_relation),
    ]
    compound_parts = {
        text(pass_side, relation): [pass_side, relation]
        for pass_side, relation in (
            (side, final_relation),
            (side, start_relation),
            (other_side, final_relation),
            (other_side, start_relation),
        )
    }
    candidate = _seal(
        scene, "passing_side_and_final_position",
        f"As {a} passes {b}, on which of {b}'s body-centered sides does the pass occur, and where does {a} finish?",
        correct, alternatives, {"pair_timeline": timeline, "passing_analysis": analysis},
        "The signed body-frame path crosses sides at an interior closest approach.",
        "Uses the annotated face-forward human frame, approach minimum, side crossing, and final relation.",
        aliases, audit, alias_status, realizer, compound_parts,
    )
    candidate["_candidate_score"] = min(float(states[0]["distance_m"]), float(states[-1]["distance_m"])) - float(analysis["closest_distance_m"])
    return candidate


def _dominant_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, states, aliases, audit, alias_status = base
    validate_human_coordinate_frame(scene)
    facing = [str(state["facing_state"]) for state in states]
    winner, dominance, margin = dominant_relation(facing)
    if dominance < policy.generation.min_dominance_ratio or margin < policy.generation.min_dominance_margin:
        raise ValueError("dominant facing relation is ambiguous")
    a, b = aliases["A"], aliases["B"]
    correct = facing_relation_text(winner, a, b)
    alternatives = [
        facing_relation_text(value, a, b)
        for value in ("facing_each_other", "back_to_back_or_away", "side_by_side_or_oblique")
        if value != winner
    ]
    alternatives.append(f"For most of the clip, {a} and {b} have no dominant facing relation.")
    candidate = _seal(
        scene, "dominant_facing_relation_over_video",
        f"Despite brief deviations, what body-facing relation do {a} and {b} maintain for most of the clip?",
        correct, alternatives, {"pair_timeline": timeline, "facing_counts": dict(Counter(facing))},
        "The answer is the temporally dominant relation after applying dominance and margin gates.",
        "Aggregates annotation-derived body-forward vectors over the complete video window.",
        aliases, audit, alias_status, realizer,
    )
    candidate["_candidate_score"] = dominance + margin
    return candidate


def _reunion_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, states, aliases, audit, alias_status = base
    validate_human_coordinate_frame(scene)
    analysis = reunion_relation_restoration(states)
    a, b = aliases["A"], aliases["B"]
    start, end = _relation(str(analysis["start_relation"])), _relation(str(analysis["end_relation"]))
    restored = bool(analysis["restored"])
    texts = [
        f"Yes; it begins {start} and returns to {start}.",
        f"No; it begins {start} and finishes {end}.",
        f"Yes; it begins {end} and returns to {end}.",
        f"No; it begins {end} and finishes {start}.",
    ]
    correct = texts[0] if restored else texts[1]
    candidate = _seal(
        scene, "reunion_relation_restoration",
        f"After {a} and {b} separate and come together again, is {b}'s final body-centered position relative to {a} restored?",
        correct, [text for text in texts if text != correct],
        {"pair_timeline": timeline, "reunion_analysis": analysis},
        "An interior separation peak establishes the split and reunion; endpoint relations determine restoration.",
        "Uses metric distance and the face-forward body-centered relation at both endpoints.",
        aliases, audit, alias_status, realizer,
    )
    candidate["_candidate_score"] = float(analysis["maximum_distance_m"]) - max(float(states[0]["distance_m"]), float(states[-1]["distance_m"]))
    return candidate


def _cause_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, states, aliases, audit, alias_status = base
    validate_human_coordinate_frame(scene)
    analysis = relation_change_cause(states, right_sign=int(timeline["coordinate_frame"]["right_sign"]))
    a, b = aliases["A"], aliases["B"]
    labels = {
        "position_movement": f"Relative position movement is sufficient, while {_sentence(a)}'s body turn is not sufficient.",
        "anchor_body_turn": f"{_sentence(a)}'s body turn is sufficient, while relative position movement is not sufficient.",
        "combined_motion": f"Relative position movement and {_sentence(a)}'s body turn are both required together.",
        "either_component_suffices": f"Relative position movement and {_sentence(a)}'s body turn are each sufficient alone.",
    }
    cause = str(analysis["cause"])
    candidate = _seal(
        scene, "relation_change_cause",
        f"{_sentence(b)} changes from {_relation(str(analysis['start_relation']))} to {_relation(str(analysis['end_relation']))} relative to {a}. Which motion causes that change?",
        labels[cause], [text for key, text in labels.items() if key != cause],
        {"pair_timeline": timeline, "causal_decomposition": analysis},
        "Translation-only and anchor-turn-only counterfactuals are evaluated independently.",
        "Deterministically freezes either starting orientation or starting relative positions.",
        aliases, audit, alias_status, realizer,
    )
    candidate["_candidate_score"] = 1.0
    return candidate


def _group_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, _, aliases, audit, alias_status = base
    multi = multi_person_metric_timeline(scene)
    states = multi.get("states") or []
    if multi.get("status") != "ok" or len(states) < policy.generation.min_states:
        raise ValueError("three stable metric person tracks are unavailable")
    person_ids = [str(value) for value in multi["metric_person_ids"]]
    _identity_for_group(scene, person_ids, aliases, audit, alias_status)
    span = max(policy.min_transition_run_states, len(states) // 4)
    start_pair, end_pair = str(states[0]["closest_pair"]), str(states[-1]["closest_pair"])
    if start_pair == end_pair:
        raise ValueError("closest-pair group structure does not change")
    if (
        sum(str(row["closest_pair"]) == start_pair for row in states[:span]) < policy.min_transition_run_states
        or sum(str(row["closest_pair"]) == end_pair for row in states[-span:]) < policy.min_transition_run_states
        or min(float(row["closest_pair_margin_m"]) for row in states[:span] + states[-span:])
        < policy.min_group_pair_margin_m
    ):
        raise ValueError("group reorganization lacks stable endpoint support or pair margin")
    start_text, end_text = _pair_text(start_pair, aliases), _pair_text(end_pair, aliases)
    correct = f"At first, {start_text} are closest; at the end, {end_text} are closest."
    alternatives = [
        f"At first, {end_text} are closest; at the end, {start_text} are closest.",
        f"At first, {start_text} are closest; at the end, {start_text} are closest.",
        f"At first, {end_text} are closest; at the end, {end_text} are closest.",
    ]
    candidate = _seal(
        scene, "metric_group_reorganization_over_video",
        f"Among {', '.join(aliases[person_id] for person_id in person_ids)}, which pair forms the closest subgroup at the start, and which pair does so at the end?",
        correct, alternatives,
        {"pair_timeline": timeline, "multi_person_timeline": multi},
        "All pairwise pelvis distances identify a stable closest pair near each endpoint.",
        "Uses every stable annotated person track; no person identity or grouping is inferred by language.",
        aliases, audit, alias_status, realizer,
    )
    candidate["_candidate_score"] = min(float(row["closest_pair_margin_m"]) for row in states[:span] + states[-span:])
    return candidate


def _visibility_candidate(scene: Mapping[str, Any], base: tuple[Any, ...], policy: Task4ScalePolicy, realizer: LanguageRealizer | None) -> dict[str, Any]:
    timeline, states, aliases, audit, alias_status = base
    blocked = [state.get("line_of_sight_blocked") for state in states]
    statuses = [state.get("line_of_sight_status") for state in states]
    if any(status != "evaluated" for status in statuses) or any(value is None for value in blocked):
        raise ValueError("physical blocker geometry is unavailable")
    span = max(policy.min_transition_run_states, len(states) // 4)
    if blocked[0] == blocked[-1]:
        raise ValueError("physical visibility does not change between stable endpoints")
    if (
        sum(value == blocked[0] for value in blocked[:span]) < policy.min_transition_run_states
        or sum(value == blocked[-1] for value in blocked[-span:]) < policy.min_transition_run_states
    ):
        raise ValueError("visibility transition lacks stable endpoint support")
    texts = {
        (False, False): "Their physical line of sight stays clear throughout.",
        (True, True): "Their physical line of sight stays blocked throughout.",
        (False, True): "Their physical line of sight changes from clear to blocked.",
        (True, False): "Their physical line of sight changes from blocked to clear.",
    }
    key = (bool(blocked[0]), bool(blocked[-1]))
    candidate = _seal(
        scene, "physical_visibility_occlusion_timeline",
        f"How does the physical line of sight between {aliases['A']} and {aliases['B']} change across the clip?",
        texts[key], [text for state, text in texts.items() if state != key],
        {"pair_timeline": timeline},
        "The annotated blocker geometry intersects the head-to-head segment at one stable endpoint but not the other.",
        "Evaluates head-to-head segment intersection against annotated blocker geometry in every temporal state.",
        aliases, audit, alias_status, realizer,
    )
    candidate["_candidate_score"] = sum(left != right for left, right in zip(blocked, blocked[1:]))
    return candidate


_BUILDERS: tuple[Callable[..., dict[str, Any]], ...] = (
    _distance_candidate,
    _passing_candidate,
    _dominant_candidate,
    _reunion_candidate,
    _cause_candidate,
    _group_candidate,
    _visibility_candidate,
)


def generate_task4_candidates(
    scene: Mapping[str, Any], *, policy: Task4ScalePolicy | None = None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Generate every evidence-supported category for one annotation window."""
    policy = policy or Task4ScalePolicy()
    base = _base(scene, policy)
    candidates: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    for builder in _BUILDERS:
        try:
            candidate = builder(scene, copy.deepcopy(base), policy, None)
            capability = capability_for(candidate["qa"][0]["question_type"])
            if capability is None:
                raise RuntimeError("candidate question type is outside the Task 4 contract")
            candidate["_capability"] = capability
            candidates.append(candidate)
        except (AnnotationEvidenceError, KeyError, TypeError, ValueError) as exc:
            rejected[builder.__name__.removeprefix("_").removesuffix("_candidate")] = str(exc)
    return candidates, rejected


def select_task4_candidates(
    candidates: Sequence[Mapping[str, Any]], targets: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily preserve scarce categories and low-versatility source windows."""
    unknown = sorted(set(targets) - set(TASK4_CAPABILITIES))
    if unknown or any(int(value) < 1 for value in targets.values()):
        raise ValueError(f"invalid Task 4 category targets: unknown={unknown}, targets={targets}")
    remaining = {category: int(targets[category]) for category in TASK4_CAPABILITIES}
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    scene_capabilities: dict[str, set[str]] = defaultdict(set)
    for source in candidates:
        candidate = dict(source)
        category = str(candidate["_capability"])
        scene_id = str(candidate["name"])
        by_category[category].append(candidate)
        scene_capabilities[scene_id].add(category)
    selected: list[dict[str, Any]] = []
    used_scenes: set[str] = set()
    while any(value > 0 for value in remaining.values()):
        viable = {
            category: [
                row for row in by_category[category]
                if str(row["name"]) not in used_scenes
            ]
            for category, needed in remaining.items() if needed > 0
        }
        viable = {category: rows for category, rows in viable.items() if rows}
        if not viable:
            break
        category = min(
            viable,
            key=lambda value: (
                len(viable[value]) / remaining[value],
                TASK4_CAPABILITIES.index(value),
            ),
        )
        chosen = max(
            viable[category],
            key=lambda row: (
                -len(scene_capabilities[str(row["name"])]),
                float(row.get("_candidate_score") or 0),
                str(row["name"]),
            ),
        )
        selected.append(chosen)
        used_scenes.add(str(chosen["name"]))
        remaining[category] -= 1
    selected.sort(key=lambda row: (
        TASK4_CAPABILITIES.index(str(row["_capability"])),
        str(row["name"]),
    ))
    counts = Counter(str(row["_capability"]) for row in selected)
    deficits = {
        category: int(targets[category]) - counts[category]
        for category in TASK4_CAPABILITIES
        if counts[category] < int(targets[category])
    }
    clean = []
    for row in selected:
        value = dict(row)
        value.pop("_candidate_score", None)
        value.pop("_capability", None)
        clean.append(value)
    return clean, {
        "status": "ok" if not deficits else "insufficient_candidates",
        "target_count": sum(int(value) for value in targets.values()),
        "target_by_category": dict(targets),
        "selected_count": len(clean),
        "selected_counts": dict(counts),
        "deficits": deficits,
        "one_question_per_window": True,
    }

def realize_task4_release(
    data: dict[str, Any], realizer: LanguageRealizer | None,
) -> None:
    """Apply wording only after deterministic selection; never during mining."""
    if realizer is None:
        return
    for group in data.get("groups") or []:
        questions = group.get("qa") or []
        if len(questions) != 1:
            raise ValueError("selected Task 4 group must contain exactly one question")
        question = questions[0]
        question_type = str(question["question_type"])
        case_id = f"{group['name']}::{question_type}"
        group["qa"] = [seal_deterministic_question(
            question,
            case_id=case_id,
            realizer=realizer,
            variant_key=case_id,
            provenance={
                "generator": "limo4si.task4_scaling.realize_task4_release",
                "reasoning_owner": "deterministic_code",
                "language_stage": "post_selection_only",
            },
        )]




def generate_task4_scale_release(
    scenes: Sequence[Mapping[str, Any]], *, target_count: int = 40,
    allow_partial: bool = False, policy: Task4ScalePolicy | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mine seven categories, balance them to the target, and fail closed on deficits."""
    policy = policy or Task4ScalePolicy(target_count=target_count)
    targets = balanced_category_targets(target_count)
    all_candidates: list[dict[str, Any]] = []
    scene_audits: list[dict[str, Any]] = []
    for scene in scenes:
        try:
            candidates, rejected = generate_task4_candidates(scene, policy=policy)
            all_candidates.extend(candidates)
            scene_audits.append({
                "scene_id": str(scene.get("scene_id") or "<missing>"),
                "candidate_categories": [
                    str(candidate["_capability"]) for candidate in candidates
                ],
                "rejected_categories": rejected,
            })
        except (AnnotationEvidenceError, KeyError, TypeError, ValueError) as exc:
            scene_audits.append({
                "scene_id": str(scene.get("scene_id") or "<missing>"),
                "candidate_categories": [],
                "rejected_scene": str(exc),
            })
    groups, selection = select_task4_candidates(all_candidates, targets)
    if selection["status"] != "ok" and not allow_partial:
        raise AnnotationEvidenceError(
            f"Task 4 selection cannot reach {target_count} release cases: "
            f"{selection['deficits']}"
        )
    data = {
        "title": "Task 4 Annotation-Driven Spatial QA",
        "subtitle": "Seven deterministic multi-human reasoning families from normalized annotations.",
        "tasks": [{
            "id": TASK4_ID,
            "name": TASK4_NAME,
            "description": "Multi-human temporal spatial reasoning.",
        }],
        "groups": groups,
        "release_policy": {
            "task_scope": [TASK4_ID],
            "annotation_only": True,
            "reasoning_owner": "deterministic_code",
            "one_question_per_case": True,
            "target_count": target_count,
            "target_by_category": targets,
        },
    }
    audit = {
        "status": selection["status"],
        "input_scene_count": len(scenes),
        "candidate_count": len(all_candidates),
        "candidate_counts": dict(Counter(
            str(candidate["_capability"]) for candidate in all_candidates
        )),
        "selection": selection,
        "scenes": scene_audits,
    }
    return data, audit
