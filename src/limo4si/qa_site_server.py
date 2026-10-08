"""Local QA-site HTTP handler with browser-compatible single-byte ranges."""
from __future__ import annotations

import os
import re
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler
from typing import BinaryIO


_BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")


def parse_byte_range(value: str | None, size: int) -> tuple[int, int] | None:
    """Parse one inclusive HTTP byte range, rejecting malformed requests."""
    if value is None:
        return None
    match = _BYTE_RANGE.fullmatch(value.strip())
    if not match or size <= 0:
        raise ValueError("unsupported byte range")
    start_text, end_text = match.groups()
    if not start_text:
        suffix = int(end_text or 0)
        if suffix <= 0:
            raise ValueError("invalid suffix byte range")
        return max(0, size - suffix), size - 1
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start >= size or end < start:
        raise ValueError("unsatisfiable byte range")
    return start, min(end, size - 1)


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """Serve static files with the single-range behavior used by HTML video."""

    _range_end: int | None = None

    def send_head(self) -> BinaryIO | None:
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            if not self.path.endswith("/"):
                self.send_response(HTTPStatus.MOVED_PERMANENTLY)
                self.send_header("Location", self.path + "/")
                self.end_headers()
                return None
            for index in ("index.html", "index.htm"):
                candidate = os.path.join(path, index)
                if os.path.isfile(candidate):
                    path = candidate
                    break
            else:
                return self.list_directory(path)
        try:
            source = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        try:
            stat = os.fstat(source.fileno())
            try:
                selected = parse_byte_range(self.headers.get("Range"), stat.st_size)
            except ValueError:
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{stat.st_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                source.close()
                return None
            self._range_end = selected[1] if selected else None
            self.send_response(HTTPStatus.PARTIAL_CONTENT if selected else HTTPStatus.OK)
            self.send_header("Content-type", self.guess_type(path))
            self.send_header("Accept-Ranges", "bytes")
            if selected:
                start, end = selected
                self.send_header("Content-Range", f"bytes {start}-{end}/{stat.st_size}")
                self.send_header("Content-Length", str(end - start + 1))
                source.seek(start)
            else:
                self.send_header("Content-Length", str(stat.st_size))
            self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
            self.end_headers()
            return source
        except Exception:
            source.close()
            raise

    def copyfile(self, source: BinaryIO, outputfile: BinaryIO) -> None:
        if self._range_end is None:
            super().copyfile(source, outputfile)
            return
        remaining = self._range_end - source.tell() + 1
        while remaining > 0:
            chunk = source.read(min(64 * 1024, remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            remaining -= len(chunk)
