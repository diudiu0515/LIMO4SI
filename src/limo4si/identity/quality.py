"""Fail-closed identity evidence validation."""
from __future__ import annotations

from typing import Any, Mapping


def validate_identity_result(result: Mapping[str, Any]) -> None:
    if result.get("status") != "aligned":
        raise ValueError(f"identity assignment is not aligned: {result.get('reason')}")
    mapping = result.get("mapping") or {}
    if len(mapping) < 2 or len(mapping) != len(set(mapping.values())):
        raise ValueError("identity mapping must be one-to-one")
    margin = result.get("margin")
    policy = result.get("policy") or {}
    if not isinstance(margin, (int, float)) or margin < float(policy.get("min_margin", 0.20)):
        raise ValueError("identity assignment margin is below policy")
    if result.get("alternative_cost") is None or result.get("best_cost") is None:
        raise ValueError("identity result lacks best and second-best costs")
