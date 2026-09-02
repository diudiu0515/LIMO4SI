#!/usr/bin/env bash
set -euo pipefail
# HOI-M3 official download helper.
# WARNING: full videos are ~5.85 TB. Prefer downloading selected sequences/views first.

HOIM3_ROOT="${HOIM3_ROOT:-/root/autodl-tmp/LIMO4SI/data/HOI-M3}"
mkdir -p "$HOIM3_ROOT"

echo "HOI-M3 official sources:"
echo "  Toolbox: https://github.com/Juzezhang/HOIM3_Toolbox"
echo "  Videos:  HuggingFace dataset JuzeZhang/HOI-M3 (~5.85 TB full)"
echo "  Annotations: linked Google Drive from toolbox README"
echo
echo "To download videos after installing huggingface_hub and logging in if needed:"
echo "  huggingface-cli download JuzeZhang/HOI-M3 --repo-type dataset --local-dir $HOIM3_ROOT"
echo
echo "Then download annotations from the official Google Drive link in the toolbox README into $HOIM3_ROOT and extract tar files:"
echo "  cd $HOIM3_ROOT"
echo "  for f in *.tar; do tar -xf \"$f\"; done"
echo
echo "Recommended for our benchmark: do NOT download all 5.85TB first. Start with annotations + 1-3 selected sequences/views, then run:"
echo "  python scripts/build_hoim3_multihuman_qa.py --hoi-m3-root data/HOI-M3"
