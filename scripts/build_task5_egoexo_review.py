#!/usr/bin/env python3
"""Build signed EgoExo4D Task 5 gaze/mask review or release examples."""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ego4d.research.util.masks import decode_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.scale_quality import require_release_quality  # noqa: E402
from limo4si.semantic_gt import (  # noqa: E402
    LanguageRealizer,
    compute_result_evidence_signature,
    load_language_realizer,
    make_semantic_gt,
    realize_question,
    validate_sealed_question,
)
from limo4si.task5_egoexo import (  # noqa: E402
    GAZE_GROUNDING_METHOD,
    QUESTION_TYPE,
    RELEASE_MAX_WINDOW_SEC,
    RELEASE_MIN_WINDOW_SEC,
    anchor_options,
    display_object_name,
    gaze_row_for_video_frame,
    unique_mask_hit,
    validate_review_result,
)


TASK_ID = "task5_human_state_grounded_spatial_reasoning"
TASK_NAME = "Task 5 · Human-State–Grounded Spatial Reasoning"
ORDINALS = ("first", "second", "third")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def load_gaze_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty gaze file: {path}")
    expected = list(range(len(rows)))
    actual = [int(row["frame_num"]) for row in rows]
    if actual != expected:
        raise ValueError(f"gaze frame_num is not contiguous in {path}")
    return rows


def measured_gaze_hz(rows: list[dict[str, str]]) -> float:
    timestamps = [int(row["tracking_timestamp_us"]) for row in rows]
    deltas = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    if not deltas:
        raise ValueError("gaze timestamps do not increase")
    return 1_000_000.0 / statistics.median(deltas)


def available_ego_cameras(entry: dict[str, Any], frames: list[int]) -> list[str]:
    cameras = set()
    for tracks in entry["object_masks"].values():
        cameras.update(camera for camera in tracks if camera.startswith("aria"))
    usable = []
    for camera in sorted(cameras):
        if all(any(
            str(frame) in tracks.get(camera, {}).get("annotation", {})
            for tracks in entry["object_masks"].values()
        ) for frame in frames):
            usable.append(camera)
    return usable


def decoded_masks_at_frame(entry: dict[str, Any], camera: str, frame: int) -> dict[str, np.ndarray]:
    masks = {}
    for object_id, tracks in entry["object_masks"].items():
        record = tracks.get(camera, {}).get("annotation", {}).get(str(frame))
        if record is not None:
            masks[str(object_id)] = decode_mask(record).astype(np.uint8)
    if not masks:
        raise ValueError(f"no Relations masks for {camera} frame {frame}")
    return masks


def video_metadata(path: Path) -> tuple[float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open source video {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if not 29.9 <= fps <= 30.1 or min(frame_count, width, height) <= 0:
        raise ValueError(f"unexpected frame-aligned video metadata for {path}")
    return fps, frame_count, width, height


def read_frame(path: Path, frame: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = capture.read()
    capture.release()
    if not ok:
        raise ValueError(f"cannot decode {path} frame {frame}")
    return image


def render_evidence(path: Path, video: Path, anchors: list[dict[str, Any]], masks: list[np.ndarray]) -> None:
    panels = []
    for index, (anchor, mask) in enumerate(zip(anchors, masks), 1):
        image = read_frame(video, int(anchor["video_frame"]))
        height, width = image.shape[:2]
        scaled_mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay = image.copy()
        overlay[scaled_mask] = (
            overlay[scaled_mask].astype(np.float32) * 0.35 + np.array([45, 210, 80], np.float32) * 0.65
        ).astype(np.uint8)
        canvas = cv2.addWeighted(image, 0.55, overlay, 0.45, 0)
        source_width, source_height = int(anchor["mask_width"]), int(anchor["mask_height"])
        gaze_x = int(round(float(anchor["gaze_pixel_xy"][0]) * width / source_width))
        gaze_y = int(round(float(anchor["gaze_pixel_xy"][1]) * height / source_height))
        cv2.drawMarker(canvas, (gaze_x, gaze_y), (20, 20, 245), cv2.MARKER_CROSS, 28, 4)
        cv2.circle(canvas, (gaze_x, gaze_y), 11, (20, 20, 245), 3)
        cv2.rectangle(canvas, (0, 0), (width, 66), (18, 25, 38), -1)
        label = f"moment {index}  clip t={anchor['clip_time_s']:.1f}s  hit: {anchor['object_name']}"
        cv2.putText(canvas, label, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)
        detail = f"margin {anchor['boundary_margin_px']:.1f}px  unique among {anchor['annotated_mask_count']} masks"
        cv2.putText(canvas, detail, (12, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 230, 240), 1)
        panels.append(canvas)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.hconcat(panels), [cv2.IMWRITE_JPEG_QUALITY, 91])


def export_clip(source: Path, output: Path, start_sec: float, duration_sec: float) -> float:
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start_sec:.6f}", "-i", str(source), "-t", f"{duration_sec:.6f}",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-movflags", "+faststart", str(output),
    ], check=True)
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(output),
    ], check=True, capture_output=True, text=True)
    actual = float(probe.stdout.strip())
    if abs(actual - duration_sec) > 0.15:
        raise ValueError(
            f"exported clip {output} is {actual:.3f}s, expected {duration_sec:.3f}s"
        )
    return actual


def build_case(
    spec: dict[str, Any],
    *,
    case_index: int,
    config: dict[str, Any],
    dataset_root: Path,
    take_by_uid: dict[str, dict[str, Any]],
    relation_entry: dict[str, Any],
    media_dir: Path,
    media_url_prefix: str,
    export_media: bool,
    release_mode: bool,
    realizer: LanguageRealizer | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    case_id = str(spec["id"])
    uid = str(spec["take_uid"])
    frames = [int(value) for value in spec["anchor_frames"]]
    if len(frames) != 3 or frames != sorted(set(frames)):
        raise ValueError(f"{case_id} requires three unique ordered frames")
    take = take_by_uid[uid]
    take_name = str(take["take_name"])
    cameras = available_ego_cameras(relation_entry, frames)
    if len(cameras) != 1:
        raise ValueError(f"{case_id} resolves to {len(cameras)} eligible ego RGB cameras")
    camera = cameras[0]
    source_video = dataset_root / "takes" / take_name / "frame_aligned_videos" / "downscaled" / "448" / f"{camera}.mp4"
    gaze_path = dataset_root / "takes" / take_name / "eye_gaze" / "general_eye_gaze_2d.csv"
    rows = load_gaze_rows(gaze_path)
    gaze_hz = measured_gaze_hz(rows)
    fps, frame_count, width, height = video_metadata(source_video)
    if abs(gaze_hz - 10.0) > 0.05:
        raise ValueError(f"{case_id} gaze sampling rate {gaze_hz:.4f} Hz is outside policy")
    if frames[-1] >= frame_count:
        raise ValueError(f"{case_id} anchor exceeds source video")

    anchors = []
    hit_masks = []
    for frame in frames:
        gaze_row, skew_ms = gaze_row_for_video_frame(rows, frame, video_fps=fps)
        x, y = float(gaze_row["x"]), float(gaze_row["y"])
        masks = decoded_masks_at_frame(relation_entry, camera, frame)
        hit = unique_mask_hit(
            masks, x, y,
            minimum_boundary_margin_px=float(config["minimum_boundary_margin_px"]),
        )
        anchors.append({
            "video_frame": frame,
            "time_s": round(frame / fps, 6),
            "gaze_frame_num": int(gaze_row["frame_num"]),
            "gaze_tracking_timestamp_us": int(gaze_row["tracking_timestamp_us"]),
            "alignment_skew_ms": round(skew_ms, 6),
            **hit,
        })
        hit_masks.append(masks[str(hit["object_id"])])

    target_object_id = str(spec["target_object_id"])
    target_matches = [index for index, anchor in enumerate(anchors) if anchor["object_id"] == target_object_id]
    if len(target_matches) != 1:
        raise ValueError(f"{case_id} target must be the unique hit at exactly one anchor")
    target_index = target_matches[0]
    target_name = display_object_name(target_object_id)
    correct_semantic_option_id = f"anchor_{target_index + 1}"

    window_duration = float(config["window_duration_sec"])
    source_duration = frame_count / fps
    if source_duration < window_duration:
        raise ValueError(f"{case_id} source video is shorter than its requested window")
    midpoint = (frames[0] / fps + frames[-1] / fps) / 2.0
    start_sec = min(
        max(0.0, midpoint - window_duration / 2.0),
        source_duration - window_duration,
    )
    end_sec = start_sec + window_duration
    clip_times = [round(float(anchor["time_s"]) - start_sec, 6) for anchor in anchors]
    for anchor, clip_time in zip(anchors, clip_times):
        anchor["clip_time_s"] = clip_time
    if frames[0] / fps < start_sec or frames[-1] / fps > end_sec:
        raise ValueError(f"{case_id} anchors do not fit the review window")

    result = {
        "status": "ok",
        "answer_type": QUESTION_TYPE,
        "T_Q": True,
        "H_Q": True,
        "S_Q": True,
        "annotation_direct": True,
        "release_status": (
            "signed_semantic_gt_egoexo_primary" if release_mode else "review_only_not_release"
        ),
        "annotation_source": "Ego-Exo4D v2 take_eye_gaze + Relations masks",
        "gaze_grounding_method": GAZE_GROUNDING_METHOD,
        "coordinate_frame": "frame-aligned ego RGB annotation plane; containment only, no directional claim",
        "take_uid": uid,
        "take_name": take_name,
        "camera": camera,
        "target_object_id": target_object_id,
        "target_object_name": target_name,
        "target_anchor_index": target_index,
        "correct_semantic_option_id": correct_semantic_option_id,
        "anchors": anchors,
        "alignment_diagnostics": {
            "video_fps": round(fps, 6),
            "gaze_frame_rate_hz_declared": 10.0,
            "gaze_timestamp_rate_hz_measured": round(gaze_hz, 6),
            "frame_mapping": "video_frame / 3 == gaze frame_num",
            "maximum_anchor_skew_ms": max(anchor["alignment_skew_ms"] for anchor in anchors),
            "video_frame_count": frame_count,
            "source_video_resolution": [width, height],
        },
        "claim_limits": [
            "Claims only synchronized 2D gaze-point containment in a Relations annotation mask.",
            "Does not claim 3D fixation depth, body-relative direction, contact, or an unannotated gaze target.",
        ],
        "source_window": {
            "start_sec": round(start_sec, 6),
            "end_sec": round(end_sec, 6),
            "duration_sec": round(window_duration, 6),
        },
        "source_evidence": {
            "relations": "annotations/relations_val.json",
            "gaze": str(gaze_path.relative_to(dataset_root)),
            "video": str(source_video.relative_to(dataset_root)),
        },
    }
    validate_review_result(result, minimum_boundary_margin_px=float(config["minimum_boundary_margin_px"]))
    evidence_signature = compute_result_evidence_signature(result)
    semantic_gt = make_semantic_gt(
        case_id=case_id,
        task_id=TASK_ID,
        question_type=QUESTION_TYPE,
        question_focus=(
            f"At which stated clip time ({clip_times[0]:.1f}s, {clip_times[1]:.1f}s, or {clip_times[2]:.1f}s) "
            f"does the camera wearer's gaze land on the {target_name}?"
        ),
        options=anchor_options(clip_times),
        correct_option_id=correct_semantic_option_id,
        evidence_statement=(
            f"The synchronized gaze point lands inside the annotated {target_name} region "
            f"only at {clip_times[target_index]:.1f} seconds into the clip. "
            + f"At {clip_times[0]:.1f}s, {clip_times[1]:.1f}s, and {clip_times[2]:.1f}s, it lands on "
            + ", ".join(anchor["object_name"] for anchor in anchors)
            + ", respectively."
        ),
        semantic_facts=[
            {"id": "target_object_id", "value": target_object_id},
            {"id": "target_anchor_index", "value": target_index},
            {"id": "anchor_hits", "value": [
                {"frame": anchor["video_frame"], "object_id": anchor["object_id"]}
                for anchor in anchors
            ]},
            {"id": "result_evidence_signature", "value": evidence_signature},
        ],
        evidence_refs=[
            {
                "kind": "egoexo_relations_mask_and_gaze_anchor",
                "video_frame": anchor["video_frame"],
                "gaze_frame_num": anchor["gaze_frame_num"],
                "object_id": anchor["object_id"],
                "decoded_mask_sha256": anchor["decoded_mask_sha256"],
            }
            for anchor in anchors
        ] + [{"kind": "result_json_sha256", "sha256": evidence_signature}],
        provenance={
            "dataset": "Ego-Exo4D v2",
            "take_uid": uid,
            "take_name": take_name,
            "computation": "deterministic_frame_aligned_2d_point_in_mask",
            "selection_config": str(config["_config_source"]),
            "release_status": result["release_status"],
        },
    )
    language = realize_question(semantic_gt, realizer=realizer, correct_index=case_index % 4)
    result.update({
        "semantic_gt_id": semantic_gt["semantic_gt_id"],
        "answer_signature": semantic_gt["answer_signature"],
        "evidence_signature": evidence_signature,
        "reasoning_owner": "deterministic_code",
        "language_model_role": "wording_only",
    })
    question = {
        "task_id": TASK_ID,
        "task_name": TASK_NAME,
        "question_type": QUESTION_TYPE,
        "question_categories": ["evidence_closed_gaze_mask_anchor"],
        **language,
        "status": "ok",
        "release_eligible": release_mode,
        "method": "Synchronized gaze point plus exact Relations mask containment; no LLM semantic judgment.",
        "result_json": result,
    }
    if not release_mode:
        question["review_only_reason"] = (
            "2D gaze/mask candidate pending promotion through the shared release quality gate"
        )
    validate_sealed_question(question)
    validate_review_result(result, minimum_boundary_margin_px=float(config["minimum_boundary_margin_px"]))

    clip_path = media_dir / f"{case_id}.mp4"
    evidence_path = media_dir / f"{case_id}_evidence.jpg"
    published_clip_duration_sec = None
    if export_media:
        published_clip_duration_sec = export_clip(
            source_video, clip_path, start_sec, window_duration
        )
        render_evidence(evidence_path, source_video, anchors, hit_masks)
    group = {
        "name": case_id,
        "title": f"Task 5 · EgoExo4D · {take_name}",
        "video_clip": f"{media_url_prefix.rstrip('/')}/{clip_path.name}",
        "original_image": f"{media_url_prefix.rstrip('/')}/{evidence_path.name}",
        "original_caption": "Post-answer audit: green is the uniquely hit Relations mask; red is synchronized 2D gaze.",
        "video_window": {
            "source_sequence": take_name,
            "start_sec": round(start_sec, 6),
            "end_sec": round(end_sec, 6),
            "duration_sec": window_duration,
            "anchor_frames": frames,
            "published_clip_duration_sec": (
                round(published_clip_duration_sec, 6)
                if published_clip_duration_sec is not None
                else None
            ),
        },
        "qa": [question],
        "case_policy": (
            "one signed annotation-derived question per unique 15-second EgoExo4D window"
            if release_mode
            else "Review-only; derived clips and annotations remain outside Git."
        ),
    }
    audit = {
        "case_id": case_id,
        "status": "ok",
        "take_name": take_name,
        "target": target_name,
        "target_anchor": target_index + 1,
        "correct_option": question["correct_option"],
        "anchor_objects": [anchor["object_name"] for anchor in anchors],
        "boundary_margins_px": [anchor["boundary_margin_px"] for anchor in anchors],
        "answer_signature": question["answer_signature"],
        "evidence_signature": result["evidence_signature"],
        "published_clip_duration_sec": (
            round(published_clip_duration_sec, 6)
            if published_clip_duration_sec is not None
            else None
        ),
    }
    return group, audit


def render_html(path: Path, groups: list[dict[str, Any]]) -> None:
    cards = []
    for group in groups:
        question = group["qa"][0]
        options = "".join(
            f"<li><span>{html.escape(str(option['label']))}</span>{html.escape(str(option['text']))}</li>"
            for option in question["options"]
        )
        result = question["result_json"]
        anchors = "".join(
            f"<tr><td>{i}</td><td>{anchor['time_s']:.1f}s</td><td>{html.escape(anchor['object_name'])}</td>"
            f"<td>{anchor['boundary_margin_px']:.1f}px</td><td>{anchor['annotated_mask_count']}</td></tr>"
            for i, anchor in enumerate(result["anchors"], 1)
        )
        cards.append(f"""
<section class="card">
  <h2>{html.escape(group['title'])}</h2>
  <p class="meta">{html.escape(group['name'])}</p>
  <video controls preload="metadata" src="{html.escape(group['video_clip'])}"></video>
  <h3>{html.escape(question['question'])}</h3>
  <ol>{options}</ol>
  <details><summary>Show answer and deterministic evidence</summary>
    <p><strong>{html.escape(question['correct_option'])}.</strong> {html.escape(question['correct_answer'])}</p>
    <p>{html.escape(question['explanation'])}</p>
    <img src="{html.escape(group['original_image'])}" alt="Three annotated gaze anchors">
    <table><thead><tr><th>Anchor</th><th>Time</th><th>Unique mask hit</th><th>Margin</th><th>Masks checked</th></tr></thead>
    <tbody>{anchors}</tbody></table>
    <p class="sig">answer {html.escape(question['answer_signature'])}<br>evidence {html.escape(result['evidence_signature'])}</p>
  </details>
</section>""")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task 5 · Ego-Exo4D examples</title>
<style>
body{{margin:0;background:#f2f4f7;color:#172033;font:16px/1.45 system-ui,sans-serif}}
main{{max-width:1050px;margin:auto;padding:32px 18px 80px}}h1{{margin-bottom:4px}}.intro{{color:#52606d;margin-top:0}}
.card{{background:white;border:1px solid #d8dee8;border-radius:14px;padding:22px;margin:24px 0;box-shadow:0 5px 18px #17203310}}
.meta,.sig{{color:#687386;font:13px ui-monospace,monospace;overflow-wrap:anywhere}}video,img{{display:block;width:100%;max-height:570px;background:#111;border-radius:9px;object-fit:contain}}
ol{{list-style:none;padding:0;display:grid;grid-template-columns:1fr 1fr;gap:10px}}li{{border:1px solid #ccd5e2;border-radius:8px;padding:10px}}li span{{font-weight:700;margin-right:10px}}
details{{margin-top:18px;border-top:1px solid #e2e7ef;padding-top:14px}}summary{{cursor:pointer;font-weight:700;color:#16697a}}
table{{border-collapse:collapse;width:100%;margin-top:14px}}th,td{{border:1px solid #d9e0e9;padding:7px;text-align:left}}th{{background:#f5f7fa}}
@media(max-width:700px){{ol{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>Task 5 · Ego-Exo4D examples</h1>
<p class="intro">{len(groups)} signed 15-second cases. Answer evidence is collapsed; open each answer to inspect exact mask hits, margins, and signatures.</p>
{''.join(cards)}</main></body></html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def load_site_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    return json.loads(match.group(1) if match else text)


def save_site_data(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        "window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n",
        encoding="utf-8",
    )


def main(*, default_release: bool = False) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/task5_egoexo_release_cases.json"
            if default_release
            else "configs/task5_egoexo_review_cases.json"
        ),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path("data/egoexo4d"))
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "outputs/qa/task5_scaled_qa.jsonl"
            if default_release
            else "outputs/qa/task5_egoexo_review.jsonl"
        ),
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=Path(
            "outputs/qa/task5_scale_audit.json"
            if default_release
            else "outputs/qa/task5_egoexo_review_audit.json"
        ),
    )
    parser.add_argument(
        "--quality-output",
        type=Path,
        default=Path("outputs/qa/task5_scale_quality.json"),
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path("site/qa_benchmark" if default_release else "site/task5_review"),
    )
    parser.add_argument("--media-output-dir", type=Path)
    parser.add_argument("--media-url-prefix")
    parser.add_argument(
        "--site-data",
        type=Path,
        default=Path("site/qa_benchmark/data.js") if default_release else None,
    )
    parser.add_argument("--language-client-factory", help="Optional module:function language-only client factory.")
    parser.add_argument("--no-media", action="store_true")
    args = parser.parse_args()

    config = json.loads(resolve(args.config).read_text(encoding="utf-8"))
    config["_config_source"] = str(args.config)
    release_status = str(config.get("release_status"))
    allowed_statuses = {"review_only_not_release", "signed_semantic_gt_egoexo_primary"}
    if release_status not in allowed_statuses:
        raise SystemExit(f"unsupported EgoExo4D Task 5 release_status: {release_status}")
    release_mode = release_status == "signed_semantic_gt_egoexo_primary"
    if default_release and not release_mode:
        raise SystemExit("release builder requires signed_semantic_gt_egoexo_primary config")
    duration = float(config.get("window_duration_sec", 0))
    if release_mode and not RELEASE_MIN_WINDOW_SEC <= duration <= RELEASE_MAX_WINDOW_SEC:
        raise SystemExit(
            f"release window must be {RELEASE_MIN_WINDOW_SEC:g}–{RELEASE_MAX_WINDOW_SEC:g} seconds"
        )

    dataset_root = resolve(args.dataset_root)
    take_by_uid = {row["take_uid"]: row for row in json.loads((dataset_root / "takes.json").read_text(encoding="utf-8"))}
    relation_rows = json.loads((dataset_root / "annotations" / "relations_val.json").read_text(encoding="utf-8"))["annotations"]
    site_dir = resolve(args.site_dir)
    media_dir = (
        resolve(args.media_output_dir)
        if args.media_output_dir
        else site_dir / ("task5_media" if release_mode else "media")
    )
    media_url_prefix = args.media_url_prefix or ("./task5_media" if release_mode else "./media")
    realizer = load_language_realizer(args.language_client_factory)

    groups, reports = [], []
    for case_index, spec in enumerate(config["cases"]):
        uid = str(spec["take_uid"])
        if uid not in relation_rows or uid not in take_by_uid:
            raise ValueError(f"{spec['id']} take is absent from metadata or Relations")
        group, report = build_case(
            spec,
            case_index=case_index,
            config=config,
            dataset_root=dataset_root,
            take_by_uid=take_by_uid,
            relation_entry=relation_rows[uid],
            media_dir=media_dir,
            media_url_prefix=media_url_prefix,
            export_media=not args.no_media,
            release_mode=release_mode,
            realizer=realizer,
        )
        groups.append(group)
        reports.append(report)

    labels = [group["qa"][0]["correct_option"] for group in groups]
    expected_labels = ["ABCD"[index % 4] for index in range(len(groups))]
    if labels != expected_labels:
        raise ValueError(f"EgoExo4D Task 5 answer positions are not balanced: {labels}")
    windows = {
        (
            group["video_window"]["source_sequence"],
            group["video_window"]["start_sec"],
            group["video_window"]["duration_sec"],
        )
        for group in groups
    }
    if len(windows) != len(groups):
        raise ValueError("EgoExo4D Task 5 release contains duplicate video windows")

    quality = require_release_quality({"groups": groups}) if release_mode else None
    output = resolve(args.output_jsonl)
    output.parent.mkdir(parents=True, exist_ok=True)
    if release_mode:
        rows = [
            {
                "case_index": index,
                "case_id": group["name"],
                "video_clip": group["video_clip"],
                **group["qa"][0],
            }
            for index, group in enumerate(groups, 1)
        ]
    else:
        rows = [
            {"case_index": index, **group}
            for index, group in enumerate(groups, 1)
        ]
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )

    audit = {
        "status": "ok",
        "release_status": release_status,
        "dataset": "Ego-Exo4D v2",
        "case_count": len(groups),
        "unique_video_windows": len(windows),
        "window_duration_sec": duration,
        "correct_option_labels": labels,
        "minimum_boundary_margin_px": config["minimum_boundary_margin_px"],
        "semantic_gt_schema": "limo4si.semantic_gt.v1",
        "reasoning_owner": "deterministic_code",
        "language_model_role": "wording_only",
        "language_realizer": groups[0]["qa"][0]["language_realization"]["realizer"] if groups else None,
        "cases": reports,
    }
    audit_path = resolve(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if release_mode:
        quality_path = resolve(args.quality_output)
        quality_path.parent.mkdir(parents=True, exist_ok=True)
        quality_path.write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if args.site_data:
            data_path = resolve(args.site_data)
            data = load_site_data(data_path)
            data["groups"] = [
                group
                for group in data.get("groups", [])
                if not any(q.get("task_id") == TASK_ID for q in group.get("qa", []))
            ]
            data["groups"].extend(groups)
            task = {
                "id": TASK_ID,
                "name": TASK_NAME,
                "description": (
                    "Which synchronized EgoExo4D gaze anchor is contained in a directly annotated object mask."
                ),
            }
            data["tasks"] = [row for row in data.get("tasks", []) if row.get("id") != TASK_ID] + [task]
            data["title"] = "Task 4 + Task 5 Spatial QA"
            data["subtitle"] = "One evidence-grounded temporal question per unique video window."
            policy = data.setdefault("release_policy", {})
            policy["task_scope"] = [row["id"] for row in data["tasks"]]
            policy["task5_scope"] = (
                "EgoExo4D synchronized 2D gaze points + Relations masks; "
                "15-second windows; containment only; deterministic signed GT"
            )
            save_site_data(data_path, data)
    else:
        render_html(site_dir / "index.html", groups)

    print(json.dumps({
        "status": "ok",
        "release_status": release_status,
        "cases": len(groups),
        "jsonl": str(output),
        "audit": str(audit_path),
        "quality": str(resolve(args.quality_output)) if release_mode else None,
        "site_data": str(resolve(args.site_data)) if release_mode and args.site_data else None,
        "review_page": str(site_dir / "index.html") if not release_mode else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
