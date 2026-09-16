"""Deterministic Task 4 identity and coordinate-frame preparation.

This module contains release-critical transformations that must happen before
question sealing or language realization.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

from .identity.public_names import render_reviewed_descriptor
from .multihuman import pair_timeline

PERSON_ATTRIBUTE_SCHEMA = "limo4si.person_attributes.v1"
DEFAULT_HOIM3_FRAME_POLICY: dict[str, Any] = {
    "origin": "SMPL-X pelvis/root translation",
    "forward_axis": "SMPL-X face/body-forward projected onto the scene ground plane",
    "right_axis": "scene-up cross forward (right-handed human frame)",
    "right_sign": 1,
    "calibration_scope": "HOI-M3 fitted humans",
}


def load_person_attribute_config(path: Path) -> dict[str, Any]:
    """Load and validate reviewed person attributes."""

    config = json.loads(path.read_text(encoding="utf-8"))
    entries = config.get("entries")
    if (
        config.get("schema_version") != PERSON_ATTRIBUTE_SCHEMA
        or not isinstance(entries, Mapping)
    ):
        raise ValueError(
            f"person attribute config must use {PERSON_ATTRIBUTE_SCHEMA}"
        )
    return config


def load_orientation_overrides(path: Path) -> dict[str, Any]:
    """Load scene- or sequence-level orientation calibration."""

    config = json.loads(path.read_text(encoding="utf-8"))
    entries = config.get("entries", config)
    if not isinstance(entries, Mapping):
        raise ValueError("orientation override entries must be a mapping")
    return dict(entries)


def person_attribute_record(
    group_name: str,
    entries: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve an exact scene record, falling back to its sequence."""

    if group_name in entries:
        return copy.deepcopy(entries[group_name])
    sequence = group_name.removeprefix("hoi_m3_").split("_win", 1)[0]
    return copy.deepcopy(entries.get(sequence) or {})


def render_person_descriptor(attributes: Mapping[str, Any]) -> str:
    """Compatibility name for evidence-bound public descriptor rendering."""

    return render_reviewed_descriptor(attributes)


def resolve_person_aliases(
    *,
    group_name: str,
    visual_person_audit: Mapping[str, Any],
    attribute_config: Mapping[str, Any],
    config_path: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Bind reviewed attributes to metric and visible track identifiers."""

    entries = attribute_config.get("entries") or {}
    record = person_attribute_record(group_name, entries)
    descriptors = {
        person_id: render_person_descriptor(attributes)
        for person_id, attributes in (record.get("people") or {}).items()
    }

    alignment = visual_person_audit.get("metric_identity_alignment") or {}
    metric_to_visible = alignment.get("mapping") or {}
    metric_ids = set(metric_to_visible)
    visible_ids = set(metric_to_visible.values())
    metric_configured = bool(metric_ids) and metric_ids.issubset(descriptors)
    visible_configured = visible_ids.issubset(descriptors)

    if not descriptors or not (metric_configured or visible_configured):
        return {}, {
            "status": "missing",
            "required_metric_ids": sorted(metric_ids),
            "required_visible_ids": sorted(visible_ids),
            "configured_ids": sorted(descriptors),
        }

    aliases = dict(descriptors)
    for metric_id, visible_id in metric_to_visible.items():
        if metric_configured:
            aliases[visible_id] = descriptors[metric_id]
        else:
            aliases[metric_id] = descriptors[visible_id]

    status = {
        "status": "complete",
        "schema_version": attribute_config["schema_version"],
        "source": record.get("source"),
        "config": config_path,
        "reviewed_attributes": record.get("reviewed_attributes") or [],
        "audit_status": record.get("audit_status"),
        "reviewed_at": record.get("reviewed_at"),
        "evidence_refs": record.get("evidence_refs") or [],
        "configured_descriptors": descriptors,
        "metric_to_visible": metric_to_visible,
    }
    return aliases, status


def _scene_orientation_override(
    scene_name: str,
    overrides: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    sequence_name = scene_name.rsplit("_win", 1)[0]
    return overrides.get(scene_name) or overrides.get(sequence_name)


def _calibrated_frame_policy(
    scene_name: str,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    policy = copy.deepcopy(DEFAULT_HOIM3_FRAME_POLICY)
    override = _scene_orientation_override(scene_name, overrides)
    if not override:
        return policy

    if "forward_signs" in override:
        policy["forward_signs"] = copy.deepcopy(override["forward_signs"])
    if "right_sign" in override:
        right_sign = int(override["right_sign"])
        if right_sign not in (-1, 1):
            raise ValueError(
                f"right_sign for {scene_name} must be -1 or 1"
            )
        policy["right_sign"] = right_sign

    expected_relations = override.get("expected_relations") or {}
    policy["orientation_calibration"] = {
        "source": override["source"],
        "expected_dominant_facing": override.get(
            "expected_dominant_facing"
        ),
        "expected_relation": copy.deepcopy(
            expected_relations.get(scene_name)
        ),
    }
    return policy


def _validate_expected_relation(
    scene_name: str,
    timeline: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> None:
    calibration = policy.get("orientation_calibration") or {}
    expected = calibration.get("expected_relation")
    if not expected:
        return

    states = timeline["states"]
    actual = {
        "start": states[0]["b_relative_to_a"],
        "end": states[-1]["b_relative_to_a"],
    }
    if actual != expected:
        raise ValueError(
            f"audited lateral relation mismatch for {scene_name}: "
            f"{actual} != {expected}"
        )


def recompute_hoim3_pair_timelines(
    data: dict[str, Any],
    dense: Mapping[str, Any],
    overrides: Mapping[str, Any],
    *,
    compact_timeline: Callable[
        [dict[str, Any]], dict[str, Any]
    ],
) -> None:
    """Recompute every metric Task 4 timeline from calibrated human frames."""

    scenes = {
        scene["scene_id"]: scene
        for scene in dense["scenes"]
    }
    for group in data["groups"]:
        question = (group.get("qa") or [{}])[0]
        result = question.get("result_json") or {}
        scene = scenes.get(group.get("name"))
        if not result.get("pair_timeline") or scene is None:
            continue

        scene_name = str(group["name"])
        policy = _calibrated_frame_policy(scene_name, overrides)
        calibrated_scene = copy.deepcopy(scene)
        calibrated_scene["human_coordinate_frame"] = policy
        timeline = compact_timeline(pair_timeline(calibrated_scene))
        _validate_expected_relation(scene_name, timeline, policy)

        result["pair_timeline"] = timeline
        result["human_coordinate_frame"] = copy.deepcopy(policy)
        relations = [
            state["b_relative_to_a"]
            for state in timeline["states"]
        ]
        if "relation_sequence" in result:
            result["relation_sequence"] = relations
        if "relation_counts" in result:
            result["relation_counts"] = dict(Counter(relations))
        group["human_coordinate_frame"] = copy.deepcopy(policy)
