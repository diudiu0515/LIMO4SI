"""Deterministic analyzers for the seven canonical Task 4 capabilities."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
from .multihuman import horizontal_side

def _side(relation: str) -> str:
    return relation.split("_", 1)[0]


def stable_visibility_timeline(
    blocked: Sequence[bool], *, minimum_run_states: int = 2,
) -> dict[str, Any]:
    """Collapse evaluated line-of-sight samples into stable visibility phases.

    Short runs are treated as annotation/sampling noise.  A publishable change
    needs at least two stable phases, each supported by ``minimum_run_states``
    consecutive samples.  This intentionally recognizes transient occlusion
    (clear -> blocked -> clear), which an endpoint-only comparison misses.
    """
    if minimum_run_states < 1:
        raise ValueError("minimum visibility run length must be positive")
    if not blocked:
        raise ValueError("visibility timeline is empty")
    runs: list[dict[str, Any]] = []
    for index, value in enumerate(blocked):
        state = bool(value)
        if not runs or runs[-1]["blocked"] != state:
            runs.append({"blocked": state, "start_index": index, "end_index": index})
        else:
            runs[-1]["end_index"] = index
    for run in runs:
        run["state_count"] = run["end_index"] - run["start_index"] + 1
    stable_runs = [run for run in runs if run["state_count"] >= minimum_run_states]
    stable_states = ["blocked" if run["blocked"] else "clear" for run in stable_runs]
    if len(stable_states) < 2 or len(set(stable_states)) < 2:
        raise ValueError("physical visibility has no stable temporal change")
    return {
        "kind": "stable_visibility_timeline",
        "minimum_run_states": minimum_run_states,
        "runs": runs,
        "stable_runs": stable_runs,
        "stable_states": stable_states,
        "transition_count": len(stable_states) - 1,
    }

def passing_side_and_final_position(
    states: Sequence[Mapping[str, Any]], *, relation_key: str = "a_relative_to_b",
) -> dict[str, Any]:
    """Recognize an approach/cross/recede event from signed body-frame evidence."""
    pair_ids = {
        "a_relative_to_b": ("A", "B"),
        "b_relative_to_a": ("B", "A"),
    }
    if relation_key not in pair_ids:
        raise ValueError("passing relation must be A-relative-to-B or B-relative-to-A")
    if len(states) < 8:
        raise ValueError("passing analysis needs at least eight states")
    distances = [float(row["distance_m"]) for row in states]
    closest = min(range(len(distances)), key=distances.__getitem__)
    relations = [str(row[relation_key]) for row in states]
    start_side, end_side = _side(relations[0]), _side(relations[-1])
    if not (2 <= closest <= len(states) - 3):
        raise ValueError("closest approach is not interior")
    if start_side not in {"left", "right"} or end_side not in {"left", "right"} or start_side == end_side:
        raise ValueError("passing actor does not cross from one side of the anchor to the other")
    if distances[closest] + 0.30 >= min(distances[0], distances[-1]):
        raise ValueError("passing event lacks a salient interior approach")
    # The passage side is the last non-centre side before the stable final side.
    before = [_side(value) for value in relations[:closest + 1]]
    pass_side = next((value for value in reversed(before) if value in {"left", "right"}), start_side)
    actor_id, anchor_id = pair_ids[relation_key]
    return {
        "kind": "passing_side_and_final_position",
        "actor_id": actor_id,
        "anchor_id": anchor_id,
        "relation_key": relation_key,
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
