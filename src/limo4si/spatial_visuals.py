"""Drawing helpers for spatial-pipeline acceptance images."""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from .spatial_real import project_world_points


SKELETON = (
    ("nose", "left-eye"),
    ("nose", "right-eye"),
    ("left-eye", "left-ear"),
    ("right-eye", "right-ear"),
    ("left-shoulder", "right-shoulder"),
    ("left-shoulder", "left-elbow"),
    ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
    ("left-hip", "left-knee"),
    ("left-knee", "left-ankle"),
    ("right-hip", "right-knee"),
    ("right-knee", "right-ankle"),
)


def draw_pose_skeleton(
    image: np.ndarray,
    joints_2d: Mapping[str, Mapping[str, float] | None],
    *,
    annotation_width: int,
    annotation_height: int,
) -> int:
    """Overlay the original EgoPose skeleton in annotation coordinates."""

    height, width = image.shape[:2]
    sx, sy = width / annotation_width, height / annotation_height
    points = {
        name: (round(float(point["x"]) * sx), round(float(point["y"]) * sy))
        for name, point in joints_2d.items()
        if point and "x" in point and "y" in point
    }
    for start, end in SKELETON:
        if start in points and end in points:
            cv2.line(image, points[start], points[end], (255, 210, 20), 3, cv2.LINE_AA)
    for name, point in points.items():
        color = (30, 30, 240) if name == "nose" else (255, 255, 40)
        cv2.circle(image, point, 5, color, -1, cv2.LINE_AA)
    return len(points)


def draw_forward_axis(
    image: np.ndarray,
    human_frame: Mapping[str, list[float]],
    calibration: Mapping[str, list],
    *,
    annotation_width: int,
    annotation_height: int,
    length_m: float = 0.45,
) -> bool:
    """Project the semantic body-forward axis onto the exo image."""

    origin = np.asarray(human_frame["origin"], dtype=np.float64)
    forward = np.asarray(human_frame["forward"], dtype=np.float64)
    world = np.stack((origin, origin + length_m * forward))
    pixels, depth = project_world_points(
        world,
        calibration["camera_intrinsics"],
        calibration["camera_extrinsics"],
    )
    if not np.isfinite(pixels).all() or np.any(depth <= 0):
        return False
    height, width = image.shape[:2]
    scaled = np.column_stack(
        (pixels[:, 0] * width / annotation_width, pixels[:, 1] * height / annotation_height)
    )
    start, end = (tuple(np.rint(point).astype(int)) for point in scaled)
    cv2.arrowedLine(image, start, end, (220, 80, 220), 4, cv2.LINE_AA, tipLength=0.22)
    cv2.putText(
        image,
        "BODY FRONT",
        (end[0] + 6, end[1] - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 80, 220),
        2,
        cv2.LINE_AA,
    )
    return True
