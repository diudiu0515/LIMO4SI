"""Global metric-person to visible-track assignment over a full window."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import permutations
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class AssignmentPolicy:
    min_margin: float = 0.20
    min_coverage: float = 0.80
    maximum_people_for_exact_margin: int = 8


@dataclass(frozen=True)
class AssignmentResult:
    status: str
    mapping: dict[str, str] | None
    best_cost: float | None
    alternative_cost: float | None
    margin: float | None
    cost_matrix: list[list[float]]
    metric_ids: list[str]
    visible_ids: list[str]
    policy: dict[str, Any]
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _points(profile: Mapping[str, Any]) -> np.ndarray:
    raw = profile.get("points") or profile.get("centers")
    array = np.asarray(raw, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        raise ValueError("identity profile needs at least three 2D/3D points")
    if not np.isfinite(array).all():
        raise ValueError("identity profile contains non-finite coordinates")
    return array[:, :3]


def _shape_signature(profile: Mapping[str, Any]) -> np.ndarray:
    points = _points(profile)
    displacement = np.diff(points, axis=0)
    scale = float(np.linalg.norm(displacement, axis=1).sum())
    if scale < 1e-9:
        scale = max(1.0, float(np.linalg.norm(points - points.mean(axis=0))))
    return displacement / scale


def _resample(signature: np.ndarray, length: int) -> np.ndarray:
    if len(signature) == length:
        return signature
    source = np.linspace(0.0, 1.0, len(signature))
    target = np.linspace(0.0, 1.0, length)
    return np.stack([
        np.interp(target, source, signature[:, axis])
        for axis in range(signature.shape[1])
    ], axis=1)


def trajectory_cost(metric: Mapping[str, Any], visible: Mapping[str, Any]) -> float:
    """Compare full-window motion shape independently of image/world scale."""
    left = _shape_signature(metric)
    right = _shape_signature(visible)
    length = max(len(left), len(right))
    left, right = _resample(left, length), _resample(right, length)
    dimensions = min(left.shape[1], right.shape[1])
    left, right = left[:, :dimensions], right[:, :dimensions]
    forward = float(np.mean(np.linalg.norm(left - right, axis=1)))
    acceleration_left = np.diff(left, axis=0)
    acceleration_right = np.diff(right, axis=0)
    acceleration = (
        float(np.mean(np.linalg.norm(acceleration_left - acceleration_right, axis=1)))
        if len(acceleration_left) else 0.0
    )
    return 0.75 * forward + 0.25 * acceleration


def assign_global_identities(
    metric_profiles: Mapping[str, Mapping[str, Any]],
    visible_profiles: Mapping[str, Mapping[str, Any]],
    *,
    policy: AssignmentPolicy | None = None,
) -> AssignmentResult:
    """Solve one-to-one identity mapping and compute the exact second-best margin."""
    policy = policy or AssignmentPolicy()
    metric_ids = sorted(metric_profiles)
    visible_ids = sorted(visible_profiles)
    empty = AssignmentResult(
        status="unresolved", mapping=None, best_cost=None, alternative_cost=None,
        margin=None, cost_matrix=[], metric_ids=metric_ids, visible_ids=visible_ids,
        policy=asdict(policy),
    )
    if len(metric_ids) < 2 or len(metric_ids) != len(visible_ids):
        return AssignmentResult(**{**asdict(empty), "reason": "metric and visible identity counts must match and be at least two"})
    if len(metric_ids) > policy.maximum_people_for_exact_margin:
        return AssignmentResult(**{**asdict(empty), "reason": "person count exceeds exact assignment-margin policy"})
    low_coverage = [
        visible_id for visible_id in visible_ids
        if float(visible_profiles[visible_id].get("coverage", 0.0)) < policy.min_coverage
    ]
    if low_coverage:
        return AssignmentResult(**{**asdict(empty), "reason": f"visible coverage below threshold: {low_coverage}"})

    matrix = np.asarray([
        [trajectory_cost(metric_profiles[m], visible_profiles[v]) for v in visible_ids]
        for m in metric_ids
    ], dtype=np.float64)
    ranked: list[tuple[float, tuple[int, ...]]] = []
    for assignment in permutations(range(len(visible_ids))):
        total = sum(float(matrix[row, column]) for row, column in enumerate(assignment))
        ranked.append((total / len(metric_ids), assignment))
    ranked.sort(key=lambda item: (item[0], item[1]))
    best, second = ranked[0], ranked[1]
    margin = second[0] - best[0]
    reliable = margin >= policy.min_margin
    mapping = {
        metric_id: visible_ids[best[1][index]]
        for index, metric_id in enumerate(metric_ids)
    }
    return AssignmentResult(
        status="aligned" if reliable else "unresolved",
        mapping=mapping if reliable else None,
        best_cost=round(best[0], 6), alternative_cost=round(second[0], 6),
        margin=round(margin, 6), cost_matrix=np.round(matrix, 6).tolist(),
        metric_ids=metric_ids, visible_ids=visible_ids, policy=asdict(policy),
        reason=None if reliable else "assignment margin is below threshold",
    )
