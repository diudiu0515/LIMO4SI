"""Canonical Task 4 capability and compound-answer release contract."""
from __future__ import annotations
from collections import Counter
from typing import Any, Mapping, Sequence

TASK4_CAPABILITIES = (
    "distance_evolution",
    "passing_side_and_final_position",
    "dominant_interaction_relation",
    "reunion_relation_restoration",
    "relation_change_cause",
    "group_reorganization",
    "physical_visibility_occlusion_timeline",
)
QUESTION_CAPABILITIES = {
    "metric_distance_pattern_over_video": "distance_evolution",
    "metric_separation_over_video": "distance_evolution",
    "nonmonotonic_distance_pattern": "distance_evolution",
    "distance_out_and_back_over_video": "distance_evolution",
    "approach_while_facing": "dominant_interaction_relation",
    "dominant_facing_relation_over_video": "dominant_interaction_relation",
    "position_consistency_between_people": "dominant_interaction_relation",
    "dominant_body_centric_position": "dominant_interaction_relation",
    "visible_pair_topology_change_2d": "group_reorganization",
    "visible_pair_topology_consistency_2d": "group_reorganization",
    "passing_side_and_final_position": "passing_side_and_final_position",
    "reunion_relation_restoration": "reunion_relation_restoration",
    "relation_change_cause": "relation_change_cause",
    "physical_visibility_occlusion_timeline": "physical_visibility_occlusion_timeline",
}

def capability_for(question_type: str) -> str | None:
    return QUESTION_CAPABILITIES.get(question_type)

def validate_compound_option_parts(options: Sequence[Mapping[str, Any]], correct_label: str, parts_by_label: Mapping[str, Sequence[str]]) -> list[str]:
    """Require a complete 2x2 grid, so neither subanswer reveals the key."""
    labels = [str(option.get("label")) for option in options]
    if set(labels) != set(parts_by_label):
        return ["compound option semantics do not cover the published option labels"]
    rows = [tuple(map(str, parts_by_label[label])) for label in labels]
    errors: list[str] = []
    if any(len(row) != 2 for row in rows):
        return ["compound answers must declare exactly two semantic parts"]
    if len(set(rows)) != 4:
        errors.append("compound options must contain four distinct semantic combinations")
    for index in range(2):
        if sorted(Counter(row[index] for row in rows).values()) != [2, 2]:
            errors.append(f"compound part {index + 1} must form a balanced 2x2 contrast")
    if correct_label not in parts_by_label:
        errors.append("compound correct option lacks semantic parts")
    return errors
