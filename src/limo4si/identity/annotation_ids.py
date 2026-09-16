"""Stable internal and public identifiers for annotated people."""
from __future__ import annotations

from typing import Iterable


def stable_person_ids(raw_ids: Iterable[str]) -> dict[str, str]:
    values = sorted({str(value) for value in raw_ids})
    return {value: f"P{index:04d}" for index, value in enumerate(values, 1)}


def neutral_public_names(person_ids: Iterable[str]) -> dict[str, str]:
    ordered = list(person_ids)
    return {person_id: f"annotated person {index}" for index, person_id in enumerate(ordered, 1)}
