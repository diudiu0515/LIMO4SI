#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="$project_root/.lfs-chunks/manifest.tsv"

if [[ ! -f "$manifest" ]]; then
  echo "Missing chunk manifest: $manifest" >&2
  exit 1
fi

while IFS=$'\t' read -r expected_sha expected_size relative_path chunk_dir; do
  [[ -z "$expected_sha" || "$expected_sha" == "sha256" ]] && continue
  target="$project_root/$relative_path"
  source_dir="$project_root/.lfs-chunks/$chunk_dir"
  mkdir -p "$(dirname "$target")"
  cat "$source_dir"/part-* > "$target"
  actual_size="$(stat -c %s "$target")"
  actual_sha="$(sha256sum "$target" | cut -d' ' -f1)"
  if [[ "$actual_size" != "$expected_size" || "$actual_sha" != "$expected_sha" ]]; then
    echo "Verification failed: $relative_path" >&2
    exit 1
  fi
  echo "Restored: $relative_path"
done < "$manifest"

echo "All chunked files restored and SHA-256 verified."
