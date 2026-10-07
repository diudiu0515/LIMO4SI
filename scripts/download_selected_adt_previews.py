#!/usr/bin/env python3
"""Download verified ADT preview videos for already-selected Task 4 sources."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


def load_site(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    if not match:
        raise ValueError(f"cannot parse QA site data: {path}")
    return json.loads(match.group(1))


def sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_adt_sequences(site: dict[str, Any]) -> list[str]:
    sequences = set()
    for group in site.get("groups") or []:
        if group.get("dataset") != "Aria Digital Twin":
            continue
        source = str((group.get("video_window") or {}).get("source") or "")
        match = re.search(r"(?:^|/)adt/([^/]+)/(?:video\.vrs|preview_rgb\.mp4)$", source)
        if not match:
            raise ValueError(f"selected ADT group has an unrecognized source path: {source}")
        sequences.add(match.group(1))
    return sorted(sequences)


def download_one(sequence: str, row: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    media = row.get("video_main_rgb") or {}
    url = str(media.get("download_url") or "")
    expected_size = int(media.get("file_size_bytes") or 0)
    expected_sha1 = str(media.get("sha1sum") or "")
    if not url or expected_size <= 0 or len(expected_sha1) != 40:
        raise ValueError(f"{sequence}: incomplete video_main_rgb catalog entry")
    target = dataset_root / sequence / "preview_rgb.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == expected_size and sha1(target) == expected_sha1:
        return {"sequence": sequence, "status": "already_verified", "path": str(target)}
    partial = target.with_suffix(".mp4.part")
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected_size:
            partial.unlink()
            offset = 0
        if offset == expected_size:
            break
        headers = {"User-Agent": "LIMO4SI/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=120) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                if not append:
                    offset = 0
                with partial.open("ab" if append else "wb") as handle:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
        except OSError:
            if attempt == 5:
                raise
    if not partial.is_file() or partial.stat().st_size != expected_size:
        actual = partial.stat().st_size if partial.exists() else 0
        raise ValueError(f"{sequence}: size mismatch {actual} != {expected_size} after retries")
    actual_sha1 = sha1(partial)
    if actual_sha1 != expected_sha1:
        raise ValueError(f"{sequence}: SHA-1 mismatch {actual_sha1} != {expected_sha1}")
    os.replace(partial, target)
    return {"sequence": sequence, "status": "downloaded", "path": str(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site-data", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=Path("ADT_download_urls.json"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/adt"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    site = load_site(args.site_data)
    sequences = selected_adt_sequences(site)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8")).get("sequences") or {}
    missing = [sequence for sequence in sequences if sequence not in catalog]
    if missing:
        raise ValueError(f"selected sequences missing from ADT catalog: {missing}")
    total_bytes = sum(int(catalog[name]["video_main_rgb"]["file_size_bytes"]) for name in sequences)
    if args.dry_run:
        print(json.dumps({
            "sequence_count": len(sequences), "total_bytes": total_bytes,
            "sequences": sequences,
        }, indent=2))
        return
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(
            lambda name: download_one(name, catalog[name], args.dataset_root),
            sequences,
        ))
    print(json.dumps({
        "status": "ok", "sequence_count": len(sequences),
        "total_bytes": total_bytes, "results": results,
    }, indent=2))


if __name__ == "__main__":
    main()
