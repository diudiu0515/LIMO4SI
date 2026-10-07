"""EgoBody-to-normalized-Task4 conversion primitives."""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def rodrigues(axis_angle: Sequence[float]) -> np.ndarray:
    vector = np.asarray(axis_angle, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    cross = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def smplx_forward(global_orient: Sequence[float]) -> list[float]:
    """Transform SMPL-X canonical +Z into the master-Kinect frame."""
    value = rodrigues(global_orient) @ np.array([0.0, 0.0, 1.0])
    horizontal = np.array([value[0], 0.0, value[2]])
    length = float(np.linalg.norm(horizontal))
    if length < 1e-6:
        raise ValueError("SMPL-X forward is vertical and cannot define a human ground frame")
    return (horizontal / length).tolist()


def smplx_head_forward(
    global_orient: Sequence[float], body_pose: Sequence[float],
) -> list[float]:
    """Transform canonical face +Z through the SMPL-X head kinematic chain.

    ``body_pose`` uses the official 21-joint SMPL-X body order (root omitted).
    The head chain is root → spine1 → spine2 → spine3 → neck → head, whose
    zero-based body-pose indices are 2, 5, 8, 11, and 14.
    """
    local = np.asarray(body_pose, dtype=float).reshape(-1, 3)
    if local.shape != (21, 3):
        raise ValueError("SMPL-X body_pose must contain 21 axis-angle joints")
    rotation = rodrigues(global_orient)
    for index in (2, 5, 8, 11, 14):
        rotation = rotation @ rodrigues(local[index])
    value = rotation @ np.array([0.0, 0.0, 1.0])
    horizontal = np.array([value[0], 0.0, value[2]])
    length = float(np.linalg.norm(horizontal))
    if length < 1e-6:
        raise ValueError("SMPL-X head forward is vertical and cannot define a face frame")
    return (horizontal / length).tolist()


def pv_face_axes(
    pv2world: Sequence[float] | np.ndarray,
    holo_to_kinect: Sequence[Sequence[float]] | np.ndarray,
) -> tuple[list[float], list[float]]:
    """Return PV face-forward and human-right axes in master-Kinect space."""
    pose = np.asarray(pv2world, dtype=float).reshape(4, 4)
    calibration = np.asarray(holo_to_kinect, dtype=float).reshape(4, 4)
    rotation = (calibration @ pose)[:3, :3]

    def horizontal(axis: np.ndarray, name: str) -> np.ndarray:
        value = np.array([axis[0], 0.0, axis[2]])
        length = float(np.linalg.norm(value))
        if length < 1e-6:
            raise ValueError(f"PV {name} axis is vertical")
        return value / length

    forward = horizontal(-rotation[:, 2], "forward")
    right_projected = horizontal(rotation[:, 0], "right")
    right = right_projected - float(np.dot(right_projected, forward)) * forward
    right_length = float(np.linalg.norm(right))
    if right_length < 1e-6:
        raise ValueError("projected PV right axis is parallel to face-forward")
    right = right / right_length
    return forward.tolist(), right.tolist()
