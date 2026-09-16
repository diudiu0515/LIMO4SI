"""Deterministic visible-person association for Task 4 evidence."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

_MAX_ASSOCIATION_COST = 0.85
_MAX_TRACK_GAP_SEC = 1.6


def iou(left: np.ndarray, right: np.ndarray) -> float:
    """Return intersection-over-union for two xyxy boxes."""

    x1, y1 = np.maximum(left[:2], right[:2])
    x2, y2 = np.minimum(left[2:], right[2:])
    intersection = (
        max(0.0, float(x2 - x1))
        * max(0.0, float(y2 - y1))
    )
    left_area = (
        max(0.0, float(left[2] - left[0]))
        * max(0.0, float(left[3] - left[1]))
    )
    right_area = (
        max(0.0, float(right[2] - right[0]))
        * max(0.0, float(right[3] - right[1]))
    )
    return intersection / max(
        1e-6,
        left_area + right_area - intersection,
    )


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.55,
) -> list[int]:
    """Apply deterministic score-ordered non-maximum suppression."""

    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        index = order.pop(0)
        keep.append(index)
        order = [
            candidate
            for candidate in order
            if iou(boxes[index], boxes[candidate]) < threshold
        ]
    return keep


def appearance_hist(
    frame: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """Describe upper and lower clothing with separate HSV histograms."""

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1 = max(0, min(width - 1, x1))
    x2 = max(x1 + 1, min(width, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(y1 + 1, min(height, y2))

    crop = frame[y1:y2, x1:x2]
    midpoint = max(1, crop.shape[0] // 2)
    body_parts = [crop[:midpoint], crop[midpoint:]]
    signatures = []
    for body_part in body_parts:
        hsv = cv2.cvtColor(body_part, cv2.COLOR_BGR2HSV)
        histogram = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            None,
            [12, 8, 8],
            [0, 180, 0, 256, 0, 256],
        )
        signatures.append(
            cv2.normalize(histogram, histogram).flatten()
        )
    return np.concatenate(signatures)


def appearance_distance(
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    """Compare upper and lower appearance with equal weight."""

    midpoint = len(left) // 2
    pairs = (
        (left[:midpoint], right[:midpoint]),
        (left[midpoint:], right[midpoint:]),
    )
    distances = [
        float(
            cv2.compareHist(
                first.astype(np.float32),
                second.astype(np.float32),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        for first, second in pairs
    ]
    return sum(distances) / len(distances)


def center(box: np.ndarray) -> np.ndarray:
    """Return an xy box center."""

    return np.array(
        [
            (box[0] + box[2]) / 2.0,
            (box[1] + box[3]) / 2.0,
        ],
        dtype=np.float32,
    )






def _predicted_center(
    track: dict[str, Any],
    timestamp: float,
) -> np.ndarray:
    """Predict the next center with constant velocity."""

    last = track["obs"][-1]
    prediction = center(last["box"])
    if len(track["obs"]) < 2:
        return prediction

    previous = track["obs"][-2]
    delta_time = max(
        1e-3,
        last["t"] - previous["t"],
    )
    velocity = (
        center(last["box"])
        - center(previous["box"])
    ) / delta_time
    return prediction + velocity * (
        timestamp - last["t"]
    )


def _association_cost(
    track: dict[str, Any],
    detection: dict[str, Any],
    prediction: np.ndarray,
    frame_diagonal: float,
) -> float:
    """Combine motion, overlap, live appearance, and anchor appearance."""

    last_box = track["obs"][-1]["box"]
    motion = float(
        np.linalg.norm(
            prediction - center(detection["box"])
        )
        / frame_diagonal
    )
    overlap = iou(last_box, detection["box"])
    live_appearance = appearance_distance(
        track["hist"],
        detection["hist"],
    )
    anchor_appearance = appearance_distance(
        track["anchor_hist"],
        detection["hist"],
    )
    appearance = (
        0.35 * live_appearance
        + 0.65 * anchor_appearance
    )
    return (
        0.05 * motion
        + 0.05 * (1.0 - overlap)
        + 0.90 * appearance
    )


def _new_track(
    raw_id: int,
    timestamp: float,
    detection: dict[str, Any],
) -> dict[str, Any]:
    signature = detection["hist"].copy()
    return {
        "raw_id": raw_id,
        "obs": [
            {
                "t": timestamp,
                "box": detection["box"],
                "score": detection["score"],
                "hist": signature.copy(),
            }
        ],
        "hist": signature.copy(),
        "anchor_hist": signature,
    }


def associate(
    detected: list[dict[str, Any]],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    """Associate detections into persistent, appearance-stable tracks."""

    if not detected:
        return []

    tracks: list[dict[str, Any]] = []
    frame_diagonal = math.hypot(width, height)
    for row in detected:
        timestamp = float(row["t"])
        detections = row["detections"]
        active = [
            track
            for track in tracks
            if timestamp - track["obs"][-1]["t"]
            <= _MAX_TRACK_GAP_SEC
        ]
        assigned_detections: set[int] = set()

        if active and detections:
            costs = np.full(
                (len(active), len(detections)),
                10.0,
                dtype=np.float32,
            )
            for track_index, track in enumerate(active):
                prediction = _predicted_center(
                    track,
                    timestamp,
                )
                for detection_index, detection in enumerate(
                    detections
                ):
                    costs[
                        track_index,
                        detection_index,
                    ] = _association_cost(
                        track,
                        detection,
                        prediction,
                        frame_diagonal,
                    )

            rows, columns = linear_sum_assignment(costs)
            for track_index, detection_index in zip(
                rows.tolist(),
                columns.tolist(),
            ):
                if (
                    costs[track_index, detection_index]
                    > _MAX_ASSOCIATION_COST
                ):
                    continue
                track = active[track_index]
                detection = detections[detection_index]
                track["obs"].append(
                    {
                        "t": timestamp,
                        "box": detection["box"],
                        "score": detection["score"],
                        "hist": detection["hist"].copy(),
                    }
                )
                track["hist"] = (
                    0.92 * track["hist"]
                    + 0.08 * detection["hist"]
                )
                assigned_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in assigned_detections:
                continue
            tracks.append(
                _new_track(
                    len(tracks) + 1,
                    timestamp,
                    detection,
                )
            )

    minimum_hits = max(
        4,
        int(round(len(detected) * 0.25)),
    )
    minimum_span = float(detected[-1]["t"]) * 0.35
    reliable = [
        track
        for track in tracks
        if len(track["obs"]) >= minimum_hits
        and (
            track["obs"][-1]["t"]
            - track["obs"][0]["t"]
        )
        >= minimum_span
    ]
    reliable.sort(
        key=lambda track: (
            track["obs"][0]["t"],
            float(center(track["obs"][0]["box"])[0]),
        )
    )
    for index, track in enumerate(reliable, 1):
        track["id"] = f"V{index}"
    return reliable


def box_at(
    track: dict[str, Any],
    timestamp: float,
    max_gap: float = 1.25,
) -> np.ndarray | None:
    """Interpolate one track box near a requested timestamp."""

    observations = track["obs"]
    before = [
        row
        for row in observations
        if row["t"] <= timestamp
    ]
    after = [
        row
        for row in observations
        if row["t"] >= timestamp
    ]
    if before and after:
        left = before[-1]
        right = after[0]
        gap = right["t"] - left["t"]
        if gap <= max_gap:
            if abs(gap) < 1e-6:
                return left["box"]
            fraction = (
                timestamp - left["t"]
            ) / gap
            return (
                (1.0 - fraction) * left["box"]
                + fraction * right["box"]
            )

    nearest = min(
        observations,
        key=lambda row: abs(row["t"] - timestamp),
    )
    if abs(nearest["t"] - timestamp) <= max_gap / 2:
        return nearest["box"]
    return None


def nearest_box(
    track: dict[str, Any],
    timestamp: float,
) -> np.ndarray:
    """Return the observed box nearest a requested timestamp."""

    return min(
        track["obs"],
        key=lambda row: abs(row["t"] - timestamp),
    )["box"]
