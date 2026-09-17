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
