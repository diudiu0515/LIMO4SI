#!/usr/bin/env python3
"""Export a compact browser-compatible RGB MP4 from an ADT VRS sequence."""
from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vrs", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--size", type=int, default=448)
    args = parser.parse_args()
    try:
        import cv2
        from projectaria_tools.core import data_provider
        from projectaria_tools.core.sensor_data import TimeDomain
        from projectaria_tools.core.stream_id import StreamId
    except ImportError as exc:
        raise RuntimeError("projectaria-tools and OpenCV are required") from exc
    provider = data_provider.create_vrs_data_provider(str(args.vrs))
    stream = StreamId("214-1")
    timestamps = provider.get_timestamps_ns(stream, TimeDomain.DEVICE_TIME)
    if len(timestamps) < 2:
        raise ValueError("ADT RGB stream has too few frames")
    fps = 1e9 * (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        intermediate = Path(directory) / "adt_rgb_mp4v.mp4"
        writer = cv2.VideoWriter(
            str(intermediate), cv2.VideoWriter_fourcc(*"mp4v"), fps, (args.size, args.size),
        )
        if not writer.isOpened():
            raise RuntimeError("could not open intermediate video writer")
        for index in range(len(timestamps)):
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
    print(args.output)


if __name__ == "__main__":
    main()
