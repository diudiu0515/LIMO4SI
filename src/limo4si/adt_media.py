"""Strict ADT calibration and RGB-media alignment helpers.

The official ADT preview MP4 stores the DEVICE_TIME timestamp of every RGB
frame as a JSON array in the format ``description`` tag.  That makes the
preview a lossily encoded, but temporally exact, alternative to decoding the
large VRS recording.  Spatial reasoning never uses pixels from either media
container; it uses annotation geometry plus the versioned device calibration.
"""
from __future__ import annotations

import json
import math
import subprocess
from bisect import bisect_left
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_REGISTRY = ROOT / "configs" / "adt_device_calibrations.json"


def index_gaze_rows_by_device_timestamp(
    rows: Iterable[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    """Index raw ADT gaze rows by DEVICE_TIME without assuming analysis row offsets."""
    indexed: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        timestamp_ns = int(row["tracking_timestamp_us"]) * 1000
        if timestamp_ns in indexed:
            raise ValueError(f"duplicate ADT gaze timestamp {timestamp_ns}")
        indexed[timestamp_ns] = row
    return indexed


def pixel_inside_box(
    pixel: tuple[float, float] | None,
    box: tuple[float, float, float, float] | list[float],
    margin: float = 0.0,
) -> bool:
    """Return whether a projected pixel is auditable against a 2D target box."""
    if pixel is None or len(box) != 4 or margin < 0:
        return False
    x1, y1, x2, y2 = (float(value) for value in box)
    return (
        x1 - margin <= float(pixel[0]) <= x2 + margin
        and y1 - margin <= float(pixel[1]) <= y2 + margin
    )


def _rotation_is_valid(matrix: list[list[float]], tolerance: float = 1e-6) -> bool:
    rotation = [row[:3] for row in matrix[:3]]
    for left in range(3):
        for right in range(3):
            dot = sum(rotation[row][left] * rotation[row][right] for row in range(3))
            expected = 1.0 if left == right else 0.0
            if abs(dot - expected) > tolerance:
                return False
    determinant = (
        rotation[0][0] * (rotation[1][1] * rotation[2][2] - rotation[1][2] * rotation[2][1])
        - rotation[0][1] * (rotation[1][0] * rotation[2][2] - rotation[1][2] * rotation[2][0])
        + rotation[0][2] * (rotation[1][0] * rotation[2][1] - rotation[1][1] * rotation[2][0])
    )
    return abs(determinant - 1.0) <= tolerance


def load_sequence_calibration(
    sequence: Path, registry_path: Path = DEFAULT_CALIBRATION_REGISTRY,
) -> dict[str, Any]:
    """Resolve and validate immutable calibration by the ADT device serial."""
    metadata_path = sequence / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"{sequence} is missing metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    serial = str(metadata.get("serial") or "")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "limo4si.adt_device_calibrations.v1":
        raise ValueError("unsupported ADT device calibration registry schema")
    entry = (registry.get("devices") or {}).get(serial)
    if not isinstance(entry, dict):
        raise ValueError(f"no reviewed ADT calibration is registered for device {serial!r}")
    transform = entry.get("transform_device_cpf")
    if (
        not isinstance(transform, list) or len(transform) != 4
        or any(not isinstance(row, list) or len(row) != 4 for row in transform)
        or any(not math.isfinite(float(value)) for row in transform for value in row)
        or not _rotation_is_valid(transform)
        or [float(value) for value in transform[3]] != [0.0, 0.0, 0.0, 1.0]
    ):
        raise ValueError(f"invalid device-to-CPF calibration for {serial}")
    rgb = entry.get("rgb_camera")
    if not isinstance(rgb, dict) or len(rgb.get("projection_params") or []) != 15:
        raise ValueError(f"invalid RGB camera calibration for {serial}")
    return {**entry, "device_serial": serial, "registry": str(registry_path)}


@lru_cache(maxsize=16)
def _rgb_projection_model(sequence_path: str) -> tuple[Any, Any]:
    import numpy as np
    from projectaria_tools.core import calibration, sophus

    stored = load_sequence_calibration(Path(sequence_path))
    rgb = stored["rgb_camera"]
    model = calibration.CameraModelType.__members__[rgb["model"]]
    width, height = (int(value) for value in rgb["image_size"])
    camera = calibration.CameraCalibration(
        "camera-rgb", model, np.asarray(rgb["projection_params"], dtype=float),
        sophus.SE3.from_matrix(np.eye(4)), width, height,
        float(rgb["valid_radius"]), float(rgb["max_solid_angle"]),
        stored["device_serial"],
    )
    return camera, np.asarray(rgb["transform_camera_cpf"], dtype=float)


def project_gaze_to_raw_rgb(
    sequence: Path, yaw_rads: float, pitch_rads: float, depth_m: float,
) -> tuple[float, float] | None:
    """Project an ADT CPF gaze point with the official CAD camera geometry."""
    import numpy as np

    camera, transform_camera_cpf = _rgb_projection_model(str(sequence.resolve()))
    point_cpf = np.asarray([
        math.tan(float(yaw_rads)) * float(depth_m),
        math.tan(float(pitch_rads)) * float(depth_m),
        float(depth_m), 1.0,
    ])
    point_camera = transform_camera_cpf @ point_cpf
    pixel = camera.project(point_camera[:3] / point_camera[3])
    return None if pixel is None else (float(pixel[0]), float(pixel[1]))


def _preview_probe(preview: Path) -> dict[str, Any]:
    output = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(preview),
    ])
    return json.loads(output)


def preview_frame_timestamps(preview: Path) -> list[int]:
    """Read and validate the official per-frame DEVICE_TIME timestamp table."""
    probe = _preview_probe(preview)
    video_streams = [row for row in probe.get("streams", []) if row.get("codec_type") == "video"]
    if len(video_streams) != 1:
        raise ValueError(f"{preview} must contain exactly one video stream")
    description = ((probe.get("format") or {}).get("tags") or {}).get("description")
    try:
        timestamps = [int(value) for value in json.loads(description)]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{preview} lacks a valid DEVICE_TIME frame table") from exc
    frame_count = int(video_streams[0].get("nb_frames") or 0)
    if frame_count < 2 or len(timestamps) != frame_count:
        raise ValueError(
            f"{preview} timestamp/frame count mismatch: {len(timestamps)} vs {frame_count}"
        )
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{preview} DEVICE_TIME timestamps move backward")
    return timestamps


def resolve_rgb_media(sequence: Path) -> tuple[Path, str]:
    preview = sequence / "preview_rgb.mp4"
    if preview.is_file():
        return preview, "official_preview_mp4"
    vrs = sequence / "video.vrs"
    if vrs.is_file():
        return vrs, "vrs_rgb_stream"
    raise FileNotFoundError(f"{sequence} has neither preview_rgb.mp4 nor video.vrs")


def rgb_frame_timestamps(media: Path, media_kind: str) -> list[int]:
    if media_kind == "official_preview_mp4":
        return preview_frame_timestamps(media)
    if media_kind != "vrs_rgb_stream":
        raise ValueError(f"unsupported ADT RGB media kind: {media_kind}")
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain
    from projectaria_tools.core.stream_id import StreamId

    provider = data_provider.create_vrs_data_provider(str(media))
    return list(provider.get_timestamps_ns(StreamId("214-1"), TimeDomain.DEVICE_TIME))


def closest_timestamp_index(timestamps: list[int], requested_ns: int) -> int:
    if not timestamps:
        raise ValueError("cannot align against an empty RGB timestamp stream")
    insertion = bisect_left(timestamps, requested_ns)
    choices = [index for index in (insertion - 1, insertion) if 0 <= index < len(timestamps)]
    return min(choices, key=lambda index: abs(timestamps[index] - requested_ns))


def rgb_window_alignment(
    sequence: Path, start_device_time_ns: int, end_device_time_ns: int,
    maximum_boundary_skew_ms: float = 50.0,
) -> tuple[Path, dict[str, Any]]:
    if end_device_time_ns <= start_device_time_ns:
        raise ValueError("RGB window end must follow its start")
    media, media_kind = resolve_rgb_media(sequence)
    timestamps = rgb_frame_timestamps(media, media_kind)
    start_index = closest_timestamp_index(timestamps, start_device_time_ns)
    end_index = closest_timestamp_index(timestamps, end_device_time_ns)
    if end_index <= start_index:
        raise ValueError("aligned RGB window has fewer than two frames")
    start_skew_ms = abs(timestamps[start_index] - start_device_time_ns) / 1_000_000
    end_skew_ms = abs(timestamps[end_index] - end_device_time_ns) / 1_000_000
    if max(start_skew_ms, end_skew_ms) > maximum_boundary_skew_ms:
        raise ValueError(
            f"selected RGB boundary skew {max(start_skew_ms, end_skew_ms):.3f} ms "
            f"exceeds {maximum_boundary_skew_ms:.3f} ms"
        )
    return media, {
        "time_domain": "device_time_ns",
        "start_device_time_ns": int(start_device_time_ns),
        "end_device_time_ns": int(end_device_time_ns),
        "media_kind": media_kind,
        "timestamp_source": (
            "mp4_format_description_json" if media_kind == "official_preview_mp4"
            else "vrs_rgb_stream_device_time"
        ),
        "start_frame_index": start_index,
        "end_frame_index": end_index,
        "actual_start_device_time_ns": int(timestamps[start_index]),
        "actual_end_device_time_ns": int(timestamps[end_index]),
        "start_boundary_skew_ms": round(start_skew_ms, 6),
        "end_boundary_skew_ms": round(end_skew_ms, 6),
        "frame_count": end_index - start_index + 1,
        "duplicate_timestamp_count": sum(
            right == left for left, right in zip(timestamps[start_index:end_index], timestamps[start_index + 1:end_index + 1])
        ),
    }
