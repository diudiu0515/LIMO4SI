"""Canonical Task 5 capability contract.

The requested capabilities require metric object geometry in a validated
wearer body frame. Image-plane gaze/mask questions are auxiliary and must not
count as coverage of these capabilities.
"""
from __future__ import annotations

TASK5_CAPABILITIES = (
    "relation_change_between_gazes",
    "gaze_onset_side_change",
    "last_gaze_annotated_object_relation_change",
)

QUESTION_CAPABILITIES = {
    "relation_change_between_gazes": "relation_change_between_gazes",
    "gaze_onset_side_change": "gaze_onset_side_change",
    "last_gaze_annotated_object_relation_change": "last_gaze_annotated_object_relation_change",
}

AUXILIARY_QUESTION_TYPES = frozenset(
    {
        "gaze_point_inside_relation_mask_at_anchor",
        "gaze_target_sequence_across_clip_checkpoints",
        "gaze_target_at_evidence_anchor",
    }
)


def capability_for(question_type: str) -> str | None:
    """Return a requested capability, excluding auxiliary gaze questions."""
    return QUESTION_CAPABILITIES.get(question_type)


def is_auxiliary(question_type: str) -> bool:
    return question_type in AUXILIARY_QUESTION_TYPES
