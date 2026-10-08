#!/usr/bin/env python3
"""Serve the local QA review site with MP4 byte-range support."""
from __future__ import annotations

import argparse
import sys
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from limo4si.qa_site_server import RangeRequestHandler  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--directory", type=Path, default=ROOT / "site" / "qa_benchmark")
    args = parser.parse_args()
    handler = partial(RangeRequestHandler, directory=str(args.directory.resolve()))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {args.directory.resolve()} on http://{args.bind}:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
