"""Real-data helpers for the Ego-Exo4D spatial-relation pipeline.

EgoPose camera extrinsics are a 3x4 world-to-camera transform. The exo
annotation coordinates use the pinhole intrinsics directly.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


@dataclass(frozen=True)
class PointSelection:
    xyz_world: np.ndarray
    camera_depth_m: np.ndarray
    scanned_points: int
    quality_points: int
    projected_points: int


def iter_semidense_points(
    path: Path,
    *,
    chunk_size: int = 500_000,
    max_dist_std_m: float | None = 0.10,
) -> Iterator[tuple[np.ndarray, int, int]]:
    """Yield quality-filtered world points and chunk input statistics."""

    opener = gzip.open if path.suffix == ".gz" else open
    xyz: list[tuple[float, float, float]] = []
    scanned = quality = 0
    with opener(path, "rt", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"px_world", "py_world", "pz_world", "dist_std"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Unexpected point-cloud columns in {path}")
        for row in reader:
            scanned += 1
            good = max_dist_std_m is None or float(row["dist_std"]) <= max_dist_std_m
            if good:
                xyz.append(tuple(float(row[f"p{axis}_world"]) for axis in "xyz"))
                quality += 1
            if scanned >= chunk_size:
                yield np.asarray(xyz, dtype=np.float64).reshape(-1, 3), scanned, quality
                xyz, scanned, quality = [], 0, 0
    if scanned:
        yield np.asarray(xyz, dtype=np.float64).reshape(-1, 3), scanned, quality


def project_world_points(
    xyz_world: np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]],
    camera_extrinsics: Sequence[Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points, returning full-resolution pixels and camera Z."""

    points = np.asarray(xyz_world, dtype=np.float64).reshape(-1, 3)
    intrinsics = np.asarray(camera_intrinsics, dtype=np.float64)
    extrinsics = np.asarray(camera_extrinsics, dtype=np.float64)
    if intrinsics.shape != (3, 3) or extrinsics.shape != (3, 4):
        raise ValueError("Expected 3x3 intrinsics and 3x4 extrinsics")
    camera_xyz = points @ extrinsics[:, :3].T + extrinsics[:, 3]
    depth = camera_xyz[:, 2]
    pixels_h = camera_xyz @ intrinsics.T
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels = pixels_h[:, :2] / pixels_h[:, 2:3]
    return pixels, depth


def select_mask_points(
    point_cloud_path: Path,
    mask: np.ndarray,
    camera_intrinsics: Sequence[Sequence[float]],
    camera_extrinsics: Sequence[Sequence[float]],
    *,
    chunk_size: int = 500_000,
    max_dist_std_m: float | None = 0.10,
) -> PointSelection:
    """Select quality-filtered points that project inside a binary mask."""

    binary_mask = np.asarray(mask, dtype=bool)
    if binary_mask.ndim != 2 or not binary_mask.any():
        raise ValueError("Object mask must be a non-empty 2D array")
    height, width = binary_mask.shape
    selected_xyz: list[np.ndarray] = []
    selected_depth: list[np.ndarray] = []
    scanned_total = quality_total = projected_total = 0

    for xyz, scanned, quality in iter_semidense_points(
        point_cloud_path,
        chunk_size=chunk_size,
        max_dist_std_m=max_dist_std_m,
    ):
        scanned_total += scanned
        quality_total += quality
        if not len(xyz):
            continue
        pixels, depth = project_world_points(xyz, camera_intrinsics, camera_extrinsics)
        finite = np.isfinite(pixels).all(axis=1) & (depth > 0)
        px = np.rint(pixels[:, 0]).astype(np.int64, casting="unsafe")
        py = np.rint(pixels[:, 1]).astype(np.int64, casting="unsafe")
        visible = finite & (px >= 0) & (px < width) & (py >= 0) & (py < height)
        projected_total += int(visible.sum())
        indices = np.flatnonzero(visible)
        indices = indices[binary_mask[py[indices], px[indices]]]
        if len(indices):
            selected_xyz.append(xyz[indices])
            selected_depth.append(depth[indices])

    xyz_result = (
        np.concatenate(selected_xyz)
        if selected_xyz
        else np.empty((0, 3), dtype=np.float64)
    )
    depth_result = (
        np.concatenate(selected_depth)
        if selected_depth
        else np.empty((0,), dtype=np.float64)
    )
    return PointSelection(
        xyz_result, depth_result, scanned_total, quality_total, projected_total
    )


def robust_object_center(
    selection: PointSelection,
    *,
    min_points: int = 8,
    mad_scale: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the median world centroid and retained robust inlier points."""

    xyz, depth = selection.xyz_world, selection.camera_depth_m
    if len(xyz) < min_points:
        raise ValueError(f"Only {len(xyz)} mask points; need at least {min_points}")
    depth_median = np.median(depth)
    depth_mad = np.median(np.abs(depth - depth_median))
    depth_limit = max(0.03, mad_scale * 1.4826 * depth_mad)
    candidate = xyz[np.abs(depth - depth_median) <= depth_limit]
    center = np.median(candidate, axis=0)
    radius = np.linalg.norm(candidate - center, axis=1)
    radius_median = np.median(radius)
    radius_mad = np.median(np.abs(radius - radius_median))
    radius_limit = radius_median + max(0.03, mad_scale * 1.4826 * radius_mad)
    inliers = candidate[radius <= radius_limit]
    if len(inliers) < min_points:
        raise ValueError(f"Only {len(inliers)} robust object points remain")
    return np.median(inliers, axis=0), inliers
