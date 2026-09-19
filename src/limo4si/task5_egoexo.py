"""Deterministic helpers for Ego-Exo4D 2D gaze/mask Task 5 cases."""
from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import cv2
import numpy as np
from ego4d.research.util.lzstring import decompress_from_encoded_uri
from pycocotools import mask as mask_utils


QUESTION_TYPE = "gaze_point_inside_relation_mask_at_anchor"
CLIP_SEQUENCE_QUESTION_TYPE = "gaze_target_sequence_across_clip_checkpoints"
RELEASE_MIN_WINDOW_SEC = 14.5
RELEASE_MAX_WINDOW_SEC = 15.5
RELEASE_MIN_ANCHOR_SPAN_RATIO = 0.65
GAZE_GROUNDING_METHOD = "frame_aligned_2d_gaze_inside_relation_mask"


def display_object_name(object_id: str) -> str:
    """Turn a Relations instance id into a readable label without changing identity."""
    value = " ".join(re.sub(r"_\d+$", "", str(object_id)).replace("_", " ").split()).lower()
    corrections = {
        "stir fried spagetti": "stir-fried spaghetti",
        "mini tomato package": "mini-tomato package",
        "alluminuim skiilet": "aluminium skillet",
    }
    return corrections.get(value, value)


def gaze_row_for_video_frame(
    rows: Sequence[Mapping[str, Any]],
    video_frame: int,
    *,
    video_fps: float = 30.0,
    gaze_hz: float = 10.0,
    maximum_skew_ms: float = 1.0,
) -> tuple[Mapping[str, Any], float]:
    """Resolve the synchronized gaze row and fail closed on non-integral alignment."""
    if video_frame < 0 or video_fps <= 0 or gaze_hz <= 0:
        raise ValueError("video frame and sampling rates must be positive")
    expected = float(video_frame) * gaze_hz / video_fps
    gaze_index = int(round(expected))
    if not 0 <= gaze_index < len(rows):
        raise ValueError(f"video frame {video_frame} is outside the gaze stream")
    row = rows[gaze_index]
    stored_index = int(row.get("frame_num", -1))
    if stored_index != gaze_index:
        raise ValueError(f"gaze row index {gaze_index} stores frame_num={stored_index}")
    skew_ms = abs(float(gaze_index) / gaze_hz - float(video_frame) / video_fps) * 1000.0
    if skew_ms > maximum_skew_ms:
        raise ValueError(f"video/gaze alignment skew {skew_ms:.3f} ms exceeds policy")
    try:
        x, y = float(row["x"]), float(row["y"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"gaze row {gaze_index} lacks a finite 2D point") from exc
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError(f"gaze row {gaze_index} lacks a finite 2D point")
    return row, skew_ms


def mask_evidence(mask: np.ndarray, x: float, y: float) -> dict[str, Any]:
    """Compute exact point containment, boundary margin, area, and a mask digest."""
    array = np.asarray(mask, dtype=np.uint8)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Relations mask must be a non-empty 2D array")
    xi, yi = int(round(float(x))), int(round(float(y)))
    inside = bool(0 <= yi < array.shape[0] and 0 <= xi < array.shape[1] and array[yi, xi] != 0)
    margin = 0.0
    if inside:
        # Padding prevents OpenCV's all-foreground sentinel value and measures
        # distance to the image edge as a legitimate mask boundary.
        padded = np.pad(array, 1, constant_values=0)
        distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
        margin = float(distance[yi + 1, xi + 1])
    return {
        "inside": inside,
        "gaze_pixel_xy": [round(float(x), 6), round(float(y), 6)],
        "rounded_pixel_xy": [xi, yi],
        "mask_width": int(array.shape[1]),
        "mask_height": int(array.shape[0]),
        "mask_area_px": int(np.count_nonzero(array)),
        "boundary_margin_px": round(margin, 6),
        "decoded_mask_sha256": "sha256:" + hashlib.sha256(array.tobytes()).hexdigest(),
    }


def unique_mask_hit(
    object_masks: Mapping[str, np.ndarray],
    x: float,
    y: float,
    *,
    minimum_boundary_margin_px: float,
) -> dict[str, Any]:
    """Return one unambiguous Relations mask hit or reject the anchor."""
    evidence = {object_id: mask_evidence(mask, x, y) for object_id, mask in object_masks.items()}
    hits = [(object_id, row) for object_id, row in evidence.items() if row["inside"]]
    if len(hits) != 1:
        raise ValueError(f"gaze point intersects {len(hits)} annotated masks; exactly one is required")
    object_id, row = hits[0]
    if float(row["boundary_margin_px"]) < float(minimum_boundary_margin_px):
        raise ValueError(
            f"gaze point margin {row['boundary_margin_px']:.3f} px is below "
            f"{minimum_boundary_margin_px:.3f} px"
        )
    return {
        "object_id": object_id,
        "object_name": display_object_name(object_id),
        "annotated_mask_count": len(object_masks),
        "unique_annotated_mask_hit": True,
        **row,
    }


def _coco_rle(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one Ego-Exo4D Relations mask record to a pycocotools RLE."""
    try:
        width = int(record["width"])
        height = int(record["height"])
        encoded = str(record["encodedMask"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Relations mask record is incomplete") from exc
    if width <= 0 or height <= 0 or not encoded:
        raise ValueError("Relations mask dimensions/encoding are invalid")
    counts = decompress_from_encoded_uri(encoded)
    if not isinstance(counts, str) or not counts:
        raise ValueError("Relations mask RLE cannot be decompressed")
    return {"size": [height, width], "counts": counts.encode()}


def unique_encoded_mask_hit(
    object_records: Mapping[str, Mapping[str, Any]],
    x: float,
    y: float,
    *,
    minimum_boundary_margin_px: float,
) -> dict[str, Any]:
    """Find one exact hit while avoiding full decode of irrelevant masks."""
    if not object_records:
        raise ValueError("no Relations masks are annotated at this frame")
    xi, yi = int(round(float(x))), int(round(float(y)))
    candidates: dict[str, np.ndarray] = {}
    for object_id, record in object_records.items():
        rle = _coco_rle(record)
        x0, y0, width, height = (float(value) for value in mask_utils.toBbox(rle))
        # Inclusive upper bounds deliberately make the prefilter conservative.
        if x0 <= xi <= x0 + width and y0 <= yi <= y0 + height:
            candidates[str(object_id)] = mask_utils.decode(rle).astype(np.uint8)
    if not candidates:
        raise ValueError("gaze point intersects 0 annotated masks; exactly one is required")
    hit = unique_mask_hit(
        candidates,
        x,
        y,
        minimum_boundary_margin_px=minimum_boundary_margin_px,
    )
    hit["annotated_mask_count"] = len(object_records)
    hit["bbox_prefilter_candidate_count"] = len(candidates)
    return hit


def anchor_options(clip_times_s: Sequence[float] | None = None) -> list[dict[str, str]]:
    if clip_times_s is None:
        labels = ["the first marked moment", "the second marked moment", "the third marked moment"]
        return [
            {"id": f"anchor_{index + 1}", "statement": f"The gaze lands there at {label}."}
            for index, label in enumerate(labels)
        ] + [{"id": "no_anchor", "statement": "The gaze does not land there at any marked moment."}]
    if len(clip_times_s) != 3:
        raise ValueError("exactly three clip-relative anchor times are required")
    times = [f"{float(value):.1f}s" for value in clip_times_s]
    options = []
    for index, selected in enumerate(times):
        others = [value for position, value in enumerate(times) if position != index]
        options.append({
            "id": f"anchor_{index + 1}",
            "statement": f"At {selected} into the clip, but not at {others[0]} or {others[1]}."
        })
    options.append({
        "id": "no_anchor",
        "statement": f"At none of these clip times: {times[0]}, {times[1]}, or {times[2]}."
    })
    return options


def validate_review_result(result: Mapping[str, Any], *, minimum_boundary_margin_px: float = 10.0) -> None:
    """Audit the stored evidence shape independently of public wording.

    The historical function name is retained for callers, but this validator is
    also the deterministic evidence validator for release cases.
    """
    if result.get("annotation_direct") is not True:
        raise ValueError("review result is not marked annotation-direct")
    if result.get("gaze_grounding_method") != GAZE_GROUNDING_METHOD:
        raise ValueError("review result uses an unsupported gaze-grounding method")
    anchors = result.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 3:
        raise ValueError("review result must expose exactly three anchors")
    frames = [int(anchor.get("video_frame", -1)) for anchor in anchors]
    if frames != sorted(set(frames)):
        raise ValueError("review anchors must be unique and ordered")
    times = [float(anchor.get("time_s", math.nan)) for anchor in anchors]
    if not all(math.isfinite(value) for value in times) or times != sorted(set(times)):
        raise ValueError("review anchor times must be finite, unique, and ordered")
    for anchor in anchors:
        if anchor.get("unique_annotated_mask_hit") is not True:
            raise ValueError("review anchor is not a unique annotated-mask hit")
        if float(anchor.get("boundary_margin_px", -1)) < minimum_boundary_margin_px:
            raise ValueError("review anchor boundary margin is below policy")
        if float(anchor.get("alignment_skew_ms", math.inf)) > 1.0:
            raise ValueError("review anchor gaze/video alignment is too stale")
        if int(anchor.get("mask_width", 0)) <= 0 or int(anchor.get("mask_height", 0)) <= 0:
            raise ValueError("review anchor mask dimensions are invalid")
        if int(anchor.get("mask_area_px", 0)) <= 0:
            raise ValueError("review anchor mask is empty")
    target = str(result.get("target_object_id"))
    object_ids = [str(anchor.get("object_id")) for anchor in anchors]
    if len(set(object_ids)) != 3:
        raise ValueError("review anchors must contain three distinct annotated object hits")
    matches = [index for index, anchor in enumerate(anchors) if str(anchor.get("object_id")) == target]
    if len(matches) != 1 or int(result.get("target_anchor_index", -1)) != matches[0]:
        raise ValueError("review target must occur at exactly its stored anchor index")
    if str(result.get("correct_semantic_option_id")) != f"anchor_{matches[0] + 1}":
        raise ValueError("review correct semantic option is stale")

    source_window = result.get("source_window")
    if source_window is not None:
        try:
            start = float(source_window["start_sec"])
            end = float(source_window["end_sec"])
            duration = float(source_window["duration_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("review source window is incomplete") from exc
        if not all(math.isfinite(value) for value in (start, end, duration)) or duration <= 0:
            raise ValueError("review source window is invalid")
        if abs((end - start) - duration) > 1e-6:
            raise ValueError("review source window duration is stale")
        if any(time < start or time > end for time in times):
            raise ValueError("review anchor falls outside the source window")
