"""Deterministic analyzers for the seven canonical Task 4 capabilities."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from .multihuman import horizontal_side

def _side(relation: str) -> str:
    return relation.split("_", 1)[0]

def passing_side_and_final_position(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recognize an approach/cross/recede event from signed body-frame evidence."""
    if len(states) < 8:
        raise ValueError("passing analysis needs at least eight states")
    distances = [float(row["distance_m"]) for row in states]
    closest = min(range(len(distances)), key=distances.__getitem__)
    relations = [str(row["a_relative_to_b"]) for row in states]
    start_side, end_side = _side(relations[0]), _side(relations[-1])
    if not (2 <= closest <= len(states) - 3):
        raise ValueError("closest approach is not interior")
    if start_side not in {"left", "right"} or end_side not in {"left", "right"} or start_side == end_side:
        raise ValueError("A does not cross from one side of B to the other")
    if distances[closest] + 0.30 >= min(distances[0], distances[-1]):
        raise ValueError("passing event lacks a salient interior approach")
    # The passage side is the last non-centre side before the stable final side.
    before = [_side(value) for value in relations[:closest + 1]]
    pass_side = next((value for value in reversed(before) if value in {"left", "right"}), start_side)
    return {
        "kind": "passing_side_and_final_position",
        "passing_side": pass_side,
        "final_relation": relations[-1],
        "start_relation": relations[0],
        "closest_index": closest,
        "closest_distance_m": distances[closest],
    }

def reunion_relation_restoration(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(states) < 8:
        raise ValueError("reunion analysis needs at least eight states")
    distances = [float(row["distance_m"]) for row in states]
    peak = max(range(len(distances)), key=distances.__getitem__)
    if not (2 <= peak <= len(states) - 3):
        raise ValueError("separation peak is not interior")
    if distances[peak] - max(distances[0], distances[-1]) < 0.50:
        raise ValueError("separation and reunion are not salient")
    start = str(states[0]["b_relative_to_a"]); end = str(states[-1]["b_relative_to_a"])
    return {
        "kind": "reunion_relation_restoration",
        "restored": start == end,
        "start_relation": start,
        "end_relation": end,
        "maximum_distance_index": peak,
        "maximum_distance_m": distances[peak],
    }

def relation_change_cause(states: Sequence[Mapping[str, Any]], *, right_sign: int) -> dict[str, Any]:
    """Separate translation and anchor-body rotation with two counterfactuals."""
    first, last = states[0], states[-1]
    def person(row: Mapping[str, Any], key: str) -> dict[str, Any]:
        evidence = row["evidence"][key]
        return {"pelvis": evidence["pelvis_xyz_m"], "forward": evidence["forward_unit"], "right": evidence.get("right_unit"), "right_sign": right_sign}
    a0, b0 = person(first, "person_a"), person(first, "person_b")
    a1, b1 = person(last, "person_a"), person(last, "person_b")
    start = horizontal_side(a0, b0); end = horizontal_side(a1, b1)
    if start == end:
        raise ValueError("relation does not change")
    translation_only = horizontal_side({**a0, "pelvis": a1["pelvis"]}, {**b0, "pelvis": b1["pelvis"]})
    rotation_only = horizontal_side({**a0, "forward": a1["forward"], "right": a1.get("right")}, b0)
    translation_matches = translation_only == end
    rotation_matches = rotation_only == end
    if translation_matches and rotation_matches:
        cause = "either_component_suffices"
    elif translation_matches:
        cause = "position_movement"
    elif rotation_matches:
        cause = "anchor_body_turn"
    else:
        cause = "combined_motion"
    return {
        "kind": "relation_change_cause", "cause": cause,
        "start_relation": start, "end_relation": end,
        "translation_only_relation": translation_only,
        "rotation_only_relation": rotation_only,
    }
