"""Evidence-bound public person naming; never performs identity matching."""
from __future__ import annotations

from typing import Any, Mapping


def render_reviewed_descriptor(attributes: Mapping[str, Any]) -> str:
    gender = attributes.get("gender_term")
    upper = attributes.get("upper_body")
    lower = attributes.get("lower_body")
    if gender not in {"man", "woman", "person"}:
        raise ValueError(f"unsupported or missing gender_term: {gender!r}")
    if not isinstance(upper, str) or not upper.strip():
        raise ValueError("person upper_body attribute is required")
    if lower is not None and (not isinstance(lower, str) or not lower.strip()):
        raise ValueError("person lower_body must be a non-empty string or null")
    descriptor = f"the {gender} in a {upper.strip()}"
    return descriptor + (f" and {lower.strip()}" if lower else "")
