"""Open-vocabulary image grounding and structured spatial-reference parsing."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


_NOUNS = {
    "杯子": "cup",
    "水杯": "cup",
    "瓶子": "bottle",
    "油瓶": "cooking oil bottle",
    "蚝油瓶": "oyster sauce bottle",
    "碗": "bowl",
    "盘子": "plate",
    "罐子": "jar",
    "盒子": "box",
    "架子": "shelf",
    "货架": "shelf",
}
_TARGET_TERMS = {
    "cup": ("cup", "mug", "glass"),
    "bottle": ("bottle", "container"),
    "cooking oil bottle": ("cooking oil bottle", "oil bottle", "bottle"),
    "oyster sauce bottle": ("oyster sauce bottle", "sauce bottle", "bottle"),
    "plate": ("plate", "dish"),
    "bowl": ("bowl",),
    "jar": ("jar", "container"),
    "box": ("box", "carton"),
}

_CN_NUM = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}


@dataclass(frozen=True)
class ReferentQuery:
    raw: str
    target: str
    support: str | None = None
    level_from_top: int | None = None
    ordinal: int | None = None
    order: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _number(text: str) -> int | None:
    if text.isdigit():
        return int(text)
    if text in _CN_NUM:
        return _CN_NUM[text]
    if text.startswith("十") and len(text) == 2:
        return 10 + _CN_NUM.get(text[1], 0)
    if text.endswith("十") and len(text) == 2:
        return _CN_NUM.get(text[0], 0) * 10
    return None


def target_terms(target: str) -> tuple[str, ...]:
    """Return detector prompts for a normalized target noun."""

    return _TARGET_TERMS.get(target, (target,))


def parse_referent(text: str) -> ReferentQuery:
    """Parse common Chinese/English shelf-level and ordinal references.

    Shelf levels are numbered top-to-bottom. ``右数第五`` means candidates are
    ordered by image x from right to left; the output records that convention.
    """

    lower = text.lower().strip()
    nouns = [
        (pos + len(zh), len(zh), en)
        for zh, en in _NOUNS.items()
        if (pos := lower.rfind(zh)) >= 0
    ]
    target = max(nouns)[2] if nouns else lower
    support = "shelf" if any(word in lower for word in ("架子", "货架", "shelf", "rack")) else None
    level = None
    match = re.search(r"第([一二两三四五六七八九十\d]+)层", lower)
    if match:
        level = _number(match.group(1))
    if level is None:
        match = re.search(r"(?:level|shelf)\s*(\d+)", lower)
        if match:
            level = int(match.group(1))
    order = None
    ordinal = None
    match = re.search(r"(右数|左数)第?([一二两三四五六七八九十\d]+)", lower)
    if match:
        order = "right_to_left" if match.group(1) == "右数" else "left_to_right"
        ordinal = _number(match.group(2))
    else:
        match = re.search(r"(\d+)(?:st|nd|rd|th)\s+from\s+(right|left)", lower)
        if match:
            ordinal = int(match.group(1))
            order = "right_to_left" if match.group(2) == "right" else "left_to_right"
    return ReferentQuery(lower, target, support, level, ordinal, order)


@dataclass
class Candidate:
    index: int
    label: str
    score: float
    box_xyxy: list[float]
    center_xy: list[float]
    level_from_top: int | None = None
    ordinal_in_level: int | None = None
    mask: np.ndarray | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result.pop("mask")
        return result


class OpenVocabularyGrounder:
    """Grounding DINO boxes followed by SAM2 box-prompted masks."""

    def __init__(self, detector: Path, segmenter: Path, device: str = "cuda"):
        import torch
        from transformers import (
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            Sam2Model,
            Sam2Processor,
        )

        self.torch = torch
        self.device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
        self.det_processor = AutoProcessor.from_pretrained(detector, local_files_only=True)
        self.det_model = AutoModelForZeroShotObjectDetection.from_pretrained(
            detector, local_files_only=True
        ).to(self.device).eval()
        self.sam_processor = Sam2Processor.from_pretrained(segmenter, local_files_only=True)
        self.sam_model = Sam2Model.from_pretrained(
            segmenter, local_files_only=True
        ).to(self.device).eval()

    def detect(
        self,
        image: Image.Image,
        label: str,
        *,
        box_threshold: float = 0.25,
        text_threshold: float = 0.20,
        start_index: int = 0,
    ) -> list[Candidate]:
        inputs = self.det_processor(images=image, text=label, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            outputs = self.det_model(**inputs)
        result = self.det_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        return [
            Candidate(
                start_index + i,
                str(labels[i]),
                float(score),
                [float(v) for v in box],
                [float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2)],
            )
            for i, (score, box) in enumerate(zip(result["scores"], result["boxes"]))
        ]

    def segment(self, image: Image.Image, candidates: Sequence[Candidate]) -> None:
        if not candidates:
            return
        boxes = [row.box_xyxy for row in candidates]
        inputs = self.sam_processor(
            images=image, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)
        with self.torch.inference_mode():
            outputs = self.sam_model(**inputs, multimask_output=False)
        masks = self.sam_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        masks = masks[:, 0] if masks.ndim == 4 else masks
        for candidate, mask in zip(candidates, masks):
            candidate.mask = np.asarray(mask > 0, dtype=bool)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def filter_candidates(
    candidates: Sequence[Candidate],
    image_size: tuple[int, int],
    *,
    max_box_area_frac: float = 0.12,
    nms_iou: float = 0.55,
) -> list[Candidate]:
    """Drop whole-scene boxes and merge duplicate detections."""

    width, height = image_size
    image_area = float(width * height)
    kept: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        x1, y1, x2, y2 = candidate.box_xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area <= 0 or area / image_area > max_box_area_frac:
            continue
        if any(_iou(candidate.box_xyxy, other.box_xyxy) >= nms_iou for other in kept):
            continue
        candidate.index = len(kept)
        kept.append(candidate)
    return kept


def assign_shelf_levels(candidates: list[Candidate]) -> None:
    """Cluster detections into image rows, numbering rows top-to-bottom."""

    if not candidates:
        return
    rows: list[list[Candidate]] = []
    heights = [max(1.0, c.box_xyxy[3] - c.box_xyxy[1]) for c in candidates]
    tolerance = max(12.0, float(np.median(heights)) * 0.75)
    for candidate in sorted(candidates, key=lambda c: c.center_xy[1]):
        if not rows or abs(candidate.center_xy[1] - np.mean([c.center_xy[1] for c in rows[-1]])) > tolerance:
            rows.append([candidate])
        else:
            rows[-1].append(candidate)
    for level, row in enumerate(rows, 1):
        for candidate in row:
            candidate.level_from_top = level


def resolve_candidate(
    candidates: list[Candidate], query: ReferentQuery
) -> tuple[Candidate | None, dict]:
    """Resolve a structured reference and expose ambiguity instead of guessing."""

    assign_shelf_levels(candidates)
    pool = candidates
    if query.level_from_top is not None:
        pool = [c for c in pool if c.level_from_top == query.level_from_top]
    if query.order:
        reverse = query.order == "right_to_left"
        pool = sorted(pool, key=lambda c: c.center_xy[0], reverse=reverse)
        for i, candidate in enumerate(pool, 1):
            candidate.ordinal_in_level = i
    elif len(pool) == 1:
        pool = list(pool)
    if query.ordinal is not None:
        pool = [c for c in pool if c.ordinal_in_level == query.ordinal]
    selected = pool[0] if len(pool) == 1 else None
    reason = "resolved" if selected else ("no_matching_candidate" if not pool else "ambiguous_candidates")
    return selected, {
        "status": "resolved" if selected else "needs_confirmation",
        "reason": reason,
        "matching_candidate_indices": [c.index for c in pool],
        "level_convention": "top_to_bottom_in_image",
        "ordering_frame": "image",
    }
