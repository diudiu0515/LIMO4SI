#!/usr/bin/env python3
"""Export an exactly DEVICE_TIME-aligned ADT RGB window."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.adt_media import closest_timestamp_index, rgb_frame_timestamps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("rgb_media", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=448)
    parser.add_argument("--start-device-time-ns", type=int)
    parser.add_argument("--end-device-time-ns", type=int)
    args = parser.parse_args()
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required") from exc
    media_kind = "official_preview_mp4" if args.rgb_media.suffix.lower() == ".mp4" else "vrs_rgb_stream"
    timestamps = rgb_frame_timestamps(args.rgb_media, media_kind)
    if len(timestamps) < 2:
        raise ValueError("ADT RGB stream has too few frames")
    if (args.start_device_time_ns is None) != (args.end_device_time_ns is None):
        raise ValueError("start and end device timestamps must be supplied together")
    if args.start_device_time_ns is None:
        start_index, end_index = 0, len(timestamps) - 1
    else:
        if args.end_device_time_ns <= args.start_device_time_ns:
            raise ValueError("end device timestamp must follow start device timestamp")
        start_index = closest_timestamp_index(timestamps, args.start_device_time_ns)
        end_index = closest_timestamp_index(timestamps, args.end_device_time_ns)
    selected_timestamps = timestamps[start_index:end_index + 1]
    if len(selected_timestamps) < 2:
        raise ValueError("selected ADT RGB window has too few frames")
    start_skew_ms = (
        abs(selected_timestamps[0] - args.start_device_time_ns) / 1_000_000
        if args.start_device_time_ns is not None else 0.0
    )
    end_skew_ms = (
        abs(selected_timestamps[-1] - args.end_device_time_ns) / 1_000_000
        if args.end_device_time_ns is not None else 0.0
    )
    if max(start_skew_ms, end_skew_ms) > 50.0:
        raise ValueError(
            f"selected RGB boundary skew {max(start_skew_ms, end_skew_ms):.3f} ms exceeds 50 ms"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if media_kind == "official_preview_mp4":
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(args.rgb_media),
            "-vf", (
                f"trim=start_frame={start_index}:end_frame={end_index + 1},"
                f"setpts=PTS-STARTPTS,scale={args.size}:{args.size}:flags=lanczos"
            ),
            "-c:v", "libx264", "-preset", "fast", "-crf", "27", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(args.output),
        ], check=True)
        print(
            f"{args.output} frames={len(selected_timestamps)} "
            f"start_skew_ms={start_skew_ms:.3f} end_skew_ms={end_skew_ms:.3f} "
            "source=official_preview_mp4"
        )
        return

    from projectaria_tools.core import data_provider
    from projectaria_tools.core.stream_id import StreamId

    provider = data_provider.create_vrs_data_provider(str(args.rgb_media))
    stream = StreamId("214-1")
    fps = 1e9 * (len(selected_timestamps) - 1) / (
        selected_timestamps[-1] - selected_timestamps[0]
    )
    with tempfile.TemporaryDirectory() as directory:
        intermediate = Path(directory) / "adt_rgb_mp4v.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.size, args.size),
        )
        if not writer.isOpened():
            raise RuntimeError("could not open intermediate video writer")
        for index in range(start_index, end_index + 1):
            image, _ = provider.get_image_data_by_index(stream, index)
            rgb = image.to_numpy_array()
            resized = cv2.resize(rgb, (args.size, args.size), interpolation=cv2.INTER_AREA)
            resized = cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)
            writer.write(cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
        writer.release()
        subprocess.run([
            "ffmpeg", "-loglevel", "error", "-y", "-i", str(intermediate),
            "-c:v", "libx264", "-preset", "fast", "-crf", "27", "-pix_fmt", "yuv420p",
            "-an", "-movflags", "+faststart", str(args.output),
        ], check=True)
    print(
        f"{args.output} frames={len(selected_timestamps)} "
        f"start_skew_ms={start_skew_ms:.3f} end_skew_ms={end_skew_ms:.3f} "
        "source=vrs_rgb_stream"
    )


if __name__ == "__main__":
    main()
