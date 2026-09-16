"""Generic Task 4 generation from normalized annotations."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .multihuman import derive_task4_answer_semantics, pair_timeline
from .scale_quality import TASK4_ID
from .semantic_gt import LanguageRealizer, seal_deterministic_question

TASK4_NAME = "Task 4 · Multi-Human Relational Dynamics"


@dataclass(frozen=True)
class Task4GenerationPolicy:
    min_states: int = 8
    min_span_ratio: float = 0.85
    min_dominance_ratio: float = 0.65
    min_dominance_margin: float = 0.20


class AnnotationEvidenceError(ValueError):
    """The annotation cannot support release-grade deterministic QA."""


def _dominant(values: Sequence[str]) -> tuple[str, float, float]:
    ordered = Counter(values).most_common()
    winner, count = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0
    return winner, count / len(values), (count - runner_up) / len(values)


def _validate_frame(scene: Mapping[str, Any]) -> None:
    frame = scene.get("human_coordinate_frame")
    if not isinstance(frame, Mapping):
        raise AnnotationEvidenceError("missing human_coordinate_frame")
    calibration = frame.get("orientation_calibration")
    if frame.get("right_sign") not in (-1, 1):
        raise AnnotationEvidenceError("right_sign must be -1 or 1")
    if not isinstance(calibration, Mapping) or not calibration.get("source"):
        raise AnnotationEvidenceError("orientation calibration provenance is missing")
    if "forward" not in str(frame.get("forward_axis", "")).lower():
        raise AnnotationEvidenceError("forward axis is not explicit")
    if "scene-up cross forward" not in str(frame.get("right_axis", "")).lower():
        raise AnnotationEvidenceError("right axis is not explicit")


def _coverage(scene: Mapping[str, Any], person_id: str) -> float:
    frames = scene.get("frames") or []
    found = sum(any(str(p.get("id")) == person_id for p in f.get("people", [])) for f in frames)
    return found / len(frames) if frames else 0.0


def _identity_evidence(scene: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
    configured = scene.get("person_identities") or {}
    aliases = {
        "A": str(configured.get("A") or "the first annotated person"),
        "B": str(configured.get("B") or "the second annotated person"),
    }
    if aliases["A"].lower() == aliases["B"].lower():
        raise AnnotationEvidenceError("person identities are not distinct")
    visible = [{"id": pid, "coverage": _coverage(scene, pid)} for pid in ("A", "B")]
    audit = {
        "status": "complete_annotation_identity",
        "source": "annotation_track_ids",
        "persistent_visible_person_count": 2,
        "max_visible_person_count": max((len(f.get("people", [])) for f in scene.get("frames", [])), default=0),
        "metric_3d_track_count": 2,
        "metric_identity_alignment": {"mapping": {"A": "A", "B": "B"}, "margin": 1.0},
        "visible_2d_tracks": visible,
    }
    status = {
        "status": "complete",
        "schema_version": "limo4si.annotation_identity.v1",
        "source": "annotation_identity",
        "configured_descriptors": aliases,
        "annotation_fields": ["frames.people.id"],
    }
    return aliases, audit, status


def _facing_text(state: str, a: str, b: str) -> str:
    return {
        "facing_each_other": f"For most of the clip, {a} and {b} face each other.",
        "back_to_back_or_away": f"For most of the clip, {a} and {b} face away from each other.",
        "side_by_side_or_oblique": f"For most of the clip, {a} and {b} remain side-by-side or oblique.",
    }[state]


def _options(correct: str, alternatives: list[str], seed: str) -> tuple[list[dict[str, str]], str]:
    if len(alternatives) != 3 or len({correct, *alternatives}) != 4:
        raise ValueError("four unique semantic options are required")
    offset = sum(map(ord, seed)) % 4
    values = list(alternatives)
    values.insert(offset, correct)
    labels = list("ABCD")
    return [{"label": label, "text": text} for label, text in zip(labels, values)], labels[offset]


def generate_task4_group(
    scene: Mapping[str, Any], *, policy: Task4GenerationPolicy | None = None,
    realizer: LanguageRealizer | None = None,
) -> dict[str, Any]:
    """Generate one high-confidence question without dataset-specific case ids."""
    policy = policy or Task4GenerationPolicy()
    _validate_frame(scene)
    timeline = pair_timeline(scene)
    states = timeline.get("states") or []
    if timeline.get("status") != "ok" or len(states) < policy.min_states:
        raise AnnotationEvidenceError("insufficient aligned A/B temporal states")
    duration = float(scene.get("duration_sec") or 0)
    times = [float(state["t"]) for state in states]
    if duration <= 0 or (times[-1] - times[0]) / duration < policy.min_span_ratio:
        raise AnnotationEvidenceError("annotation time coverage is below threshold")
    facing = [str(state["facing_state"]) for state in states]
    winner, dominance, margin = _dominant(facing)
    if dominance < policy.min_dominance_ratio or margin < policy.min_dominance_margin:
        raise AnnotationEvidenceError("no unambiguous Task 4 facing relation")

    aliases, identity_audit, alias_status = _identity_evidence(scene)
    a, b = aliases["A"], aliases["B"]
    correct = _facing_text(winner, a, b)
    alternatives = [_facing_text(value, a, b) for value in (
        "facing_each_other", "back_to_back_or_away", "side_by_side_or_oblique"
    ) if value != winner]
    alternatives.append(f"For most of the clip, {a} and {b} have no dominant facing relation.")
    options, label = _options(correct, alternatives, str(scene["scene_id"]))
    result = {
        "scene_id": scene["scene_id"], "answer_type": "dominant_facing_relation_over_video",
        "T_Q": True, "H_Q": True, "S_Q": True, "pair_timeline": timeline,
        "facing_counts": dict(Counter(facing)),
    }
    result["answer_semantics"] = derive_task4_answer_semantics(
        "dominant_facing_relation_over_video", result,
    )
    question = seal_deterministic_question({
        "task_id": TASK4_ID, "task_name": TASK4_NAME,
        "question_type": "dominant_facing_relation_over_video",
        "question": f"What body-facing relation dominates between {a} and {b} over the annotated time window?",
        "options": options, "correct_option": label, "correct_answer": correct,
        "answer": correct,
        "explanation": "The deterministic timeline aggregates the annotation-provided body-forward vectors over the complete window.",
        "method": "Projects annotated body-forward vectors and the between-person direction onto the ground plane, then applies dominance gates.",
        "status": "ok", "release_eligible": True, "result_json": result,
    }, case_id=str(scene["scene_id"]), realizer=realizer, provenance={
        "generator": "limo4si.task4_annotation.generate_task4_group",
        "reasoning_owner": "deterministic_code",
    })
    return {
        "name": str(scene["scene_id"]), "title": str(scene.get("title") or scene["scene_id"]),
        "dataset": str(scene.get("dataset") or "annotation bundle"),
        "video_clip": scene.get("video_clip"),
        "video_window": {"start_sec": float(scene.get("start_sec") or 0), "duration_sec": duration,
                         "metric_sample_count": len(states), "source": scene.get("source_file") or scene.get("source_video")},
        "visual_person_audit": identity_audit, "person_display_aliases": aliases,
        "person_display_alias_status": alias_status,
        "case_policy": "one deterministic question per annotation window", "qa": [question],
    }


def generate_task4_release(
    scenes: Sequence[Mapping[str, Any]], *, policy: Task4GenerationPolicy | None = None,
    realizer: LanguageRealizer | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    groups, rejected = [], []
    for scene in scenes:
        try:
            groups.append(generate_task4_group(scene, policy=policy, realizer=realizer))
        except (AnnotationEvidenceError, KeyError, TypeError, ValueError) as exc:
            rejected.append({"scene_id": str(scene.get("scene_id") or "<missing>"), "reason": str(exc)})
    data = {
        "title": "Task 4 Annotation-Driven Spatial QA",
        "subtitle": "Deterministic QA generated directly from normalized annotations.",
        "tasks": [{"id": TASK4_ID, "name": TASK4_NAME, "description": "Multi-human temporal spatial reasoning."}],
        "groups": groups,
        "release_policy": {"task_scope": [TASK4_ID], "annotation_only": True,
                           "reasoning_owner": "deterministic_code", "one_question_per_case": True},
    }
    audit = {"status": "ok" if groups else "no_accepted_cases", "input_scene_count": len(scenes),
             "accepted_count": len(groups), "rejected_count": len(rejected), "rejected": rejected}
    return data, audit
