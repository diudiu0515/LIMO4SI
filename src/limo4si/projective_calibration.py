"""Estimate a static exo pinhole camera from EgoPose 3D/2D correspondences."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np


def _normalize_2d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    scale = np.sqrt(2.0) / np.mean(np.linalg.norm(points - center, axis=1))
    transform = np.array(
        [[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]]
    )
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform.T)[:, :2], transform


def _normalize_3d(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    scale = np.sqrt(3.0) / np.mean(np.linalg.norm(points - center, axis=1))
    transform = np.eye(4)
    transform[:3, :3] *= scale
    transform[:3, 3] = -scale * center
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (homogeneous @ transform.T)[:, :3], transform


def estimate_projection_matrix(
    world_xyz: np.ndarray, image_xy: np.ndarray
) -> np.ndarray:
    """Normalized DLT estimate of a 3x4 world-to-image camera matrix."""

    world = np.asarray(world_xyz, dtype=np.float64).reshape(-1, 3)
    image = np.asarray(image_xy, dtype=np.float64).reshape(-1, 2)
    if len(world) < 12 or len(world) != len(image):
        raise ValueError("Need at least 12 paired 3D/2D observations")
    world_n, world_transform = _normalize_3d(world)
    image_n, image_transform = _normalize_2d(image)
    rows = []
    for (x, y, z), (u, v) in zip(world_n, image_n):
        point = np.array([x, y, z, 1.0])
        rows.append(np.r_[point, np.zeros(4), -u * point])
        rows.append(np.r_[np.zeros(4), point, -v * point])
    _, _, vh = np.linalg.svd(np.asarray(rows), full_matrices=False)
    normalized = vh[-1].reshape(3, 4)
    projection = np.linalg.inv(image_transform) @ normalized @ world_transform
    return projection / np.linalg.norm(projection[2, :3])


def decompose_projection_matrix(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a projective camera into K and a metric world-to-camera [R|t]."""

    matrix = np.asarray(projection, dtype=np.float64).reshape(3, 4)
    if np.linalg.det(matrix[:, :3]) < 0:
        matrix = -matrix
    intrinsics, rotation, camera_h, *_ = cv2.decomposeProjectionMatrix(matrix)
    intrinsics /= intrinsics[2, 2]
    if np.linalg.det(rotation) < 0:
        rotation *= -1
        intrinsics *= -1
        intrinsics /= intrinsics[2, 2]
    camera_center = (camera_h[:3] / camera_h[3]).reshape(3)
    translation = -rotation @ camera_center
    return intrinsics, np.column_stack((rotation, translation))


def collect_pose_correspondences(
    body: Mapping[str, list[dict]],
    camera: str,
    *,
    max_points: int = 6000,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect distributed valid EgoPose correspondences for one camera."""

    world, image = [], []
    frames = sorted(body, key=int)
    stride = max(1, len(frames) // 500)
    for frame in frames[::stride]:
        for person in body[frame]:
            joints_3d = person.get("annotation3D", {})
            joints_2d = person.get("annotation2D", {}).get(camera, {})
            for name, point_3d in joints_3d.items():
                point_2d = joints_2d.get(name)
                if not point_3d or not point_2d:
                    continue
                if not all(axis in point_3d for axis in "xyz"):
                    continue
                if not all(axis in point_2d for axis in "xy"):
                    continue
                world.append([point_3d[axis] for axis in "xyz"])
                image.append([point_2d[axis] for axis in "xy"])
                if len(world) >= max_points:
                    return np.asarray(world), np.asarray(image)
    return np.asarray(world), np.asarray(image)


def reprojection_statistics(
    world_xyz: np.ndarray,
    image_xy: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> dict:
    camera = world_xyz @ extrinsics[:, :3].T + extrinsics[:, 3]
    homogeneous = camera @ intrinsics.T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    errors = np.linalg.norm(projected - image_xy, axis=1)
    return {
        "point_count": int(len(errors)),
        "median_px": float(np.median(errors)),
        "mean_px": float(np.mean(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": float(np.max(errors)),
    }
