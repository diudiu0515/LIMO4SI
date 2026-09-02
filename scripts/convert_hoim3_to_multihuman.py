#!/usr/bin/env python3
"""Convert HOI-M3-style exports into the LIMO4SI multihuman schema.

Why this adapter exists
-----------------------
The official HOI-M3 release is very large and usually needs the HOIM3 toolbox to
extract/prepare videos, masks, MHR/SMPL-X fits, object poses and calibration.
This script consumes a lightweight export after that preprocessing step and
normalizes it into the schema used by ``limo4si.multihuman``.

Preferred input: a JSON file or directory containing JSON files with either the
canonical schema directly:

{
  "scene_id": "hoi_m3_sequence_x",
  "title": "...",
  "duration_sec": 15,
  "frames": [
    {"t": 0.0, "people": [
      {"id": "A", "pelvis": [x,y,z], "head": [x,y,z], "forward": [x,y,z]},
      {"id": "B", "pelvis": [x,y,z], "head": [x,y,z], "forward": [x,y,z]}
    ]}
  ],
  "blockers": [{"id": "partition", "center": [x,y,z], "radius": 0.4}],
  "objects": [{"id": "chair", "center": [x,y,z]}]
}

or a relaxed HOI-M3/toolbox export with fields like ``humans``/``persons`` and
per-frame ``joints3d``/``pelvis``/``head``/``forward``.

This script intentionally does not download the full HOI-M3 dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from typing import Any, Iterable


def vec3(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(k in value for k in ('x', 'y', 'z')):
            return [float(value['x']), float(value['y']), float(value['z'])]
        if 'translation' in value:
            return vec3(value['translation'])
        if 'center' in value:
            return vec3(value['center'])
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return [float(value[0]), float(value[1]), float(value[2])]
    return None


def sub(a: list[float], b: list[float]) -> list[float]:
    return [a[i] - b[i] for i in range(3)]


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x*x for x in v))
    return [0.0, 0.0, 1.0] if n < 1e-9 else [x/n for x in v]


def joint(joints: Any, names: Iterable[str]) -> list[float] | None:
    if not joints:
        return None
    if isinstance(joints, dict):
        for name in names:
            if name in joints:
                got = vec3(joints[name])
                if got is not None:
                    return got
    return None


def infer_forward(person: dict[str, Any]) -> list[float]:
    explicit = vec3(person.get('forward') or person.get('body_forward') or person.get('facing'))
    if explicit:
        return unit(explicit)
    joints = person.get('joints3d') or person.get('joints') or person.get('keypoints3d')
    nose = joint(joints, ['nose', 'head', 'head_top'])
    pelvis = vec3(person.get('pelvis')) or joint(joints, ['pelvis', 'root', 'body_root'])
    neck = joint(joints, ['neck', 'spine3', 'chest'])
    if nose and pelvis:
        return unit(sub(nose, pelvis))
    if neck and pelvis:
        return unit(sub(neck, pelvis))
    return [0.0, 0.0, 1.0]


def normalize_person(raw: dict[str, Any], fallback_id: str) -> dict[str, Any] | None:
    pid = str(raw.get('id') or raw.get('person_id') or raw.get('track_id') or fallback_id)
    joints = raw.get('joints3d') or raw.get('joints') or raw.get('keypoints3d') or {}
    pelvis = vec3(raw.get('pelvis') or raw.get('root') or raw.get('translation')) or joint(joints, ['pelvis', 'root', 'body_root'])
    head = vec3(raw.get('head')) or joint(joints, ['head', 'nose', 'head_top', 'neck'])
    if pelvis is None:
        return None
    if head is None:
        head = [pelvis[0], pelvis[1] + 1.6, pelvis[2]]
    return {'id': pid, 'pelvis': pelvis, 'head': head, 'forward': infer_forward({**raw, 'pelvis': pelvis})}


def normalize_frame(raw: dict[str, Any], index: int) -> dict[str, Any] | None:
    people_raw = raw.get('people') or raw.get('humans') or raw.get('persons') or raw.get('subjects') or []
    people = []
    for pi, p in enumerate(people_raw):
        norm = normalize_person(p, chr(ord('A') + pi))
        if norm:
            people.append(norm)
    if len(people) < 2:
        return None
    return {'t': float(raw.get('t', raw.get('time', raw.get('timestamp', index)))), 'frame_id': raw.get('frame_id', raw.get('frame', index)), 'people': people}


def normalize_object(raw: dict[str, Any], fallback_id: str) -> dict[str, Any] | None:
    center = vec3(raw.get('center') or raw.get('translation') or raw.get('position'))
    if center is None:
        return None
    return {'id': str(raw.get('id') or raw.get('name') or fallback_id), 'center': center, 'radius': float(raw.get('radius', raw.get('approx_radius', 0.35)))}


def normalize_scene(raw: dict[str, Any], source: Path) -> dict[str, Any] | None:
    if 'frames' not in raw:
        return None
    frames = []
    for i, fr in enumerate(raw.get('frames') or []):
        nf = normalize_frame(fr, i)
        if nf:
            frames.append(nf)
    if not frames:
        return None
    blockers = []
    for i, obj in enumerate(raw.get('blockers') or raw.get('occluders') or []):
        no = normalize_object(obj, f'blocker{i}')
        if no:
            blockers.append(no)
    objects = []
    for i, obj in enumerate(raw.get('objects') or raw.get('landmarks') or []):
        no = normalize_object(obj, f'object{i}')
        if no:
            objects.append(no)
    return {
        'scene_id': str(raw.get('scene_id') or raw.get('sequence') or raw.get('seq') or source.stem),
        'title': str(raw.get('title') or f'HOI-M3 · {source.stem}'),
        'dataset': 'HOI-M3',
        'source_file': str(source),
        'duration_sec': float(raw.get('duration_sec') or (frames[-1]['t'] - frames[0]['t'] if len(frames) > 1 else 0.0)),
        'frames': frames,
        'blockers': blockers,
        'objects': objects,
    }



def rodrigues(axis_angle: list[float]) -> list[list[float]]:
    theta = math.sqrt(sum(float(x)*float(x) for x in axis_angle))
    if theta < 1e-9:
        return [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    x,y,z = [float(v)/theta for v in axis_angle]
    c = math.cos(theta); s = math.sin(theta); C = 1.0 - c
    return [
        [c+x*x*C, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, c+z*z*C],
    ]


def mat_vec(R: list[list[float]], v: list[float]) -> list[float]:
    return [sum(R[i][j]*v[j] for j in range(3)) for i in range(3)]


def video_fps(video: Path, fallback: float = 30.0) -> float:
    """Read the actual source FPS instead of assuming HOI-M3 is 30 FPS."""
    if not video.exists():
        return fallback
    try:
        raw = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=avg_frame_rate",
                "-of", "default=noprint_wrappers=1:nokey=1", str(video),
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        if "/" in raw:
            a, b = raw.split("/", 1)
            return float(a) / float(b)
        return float(raw)
    except (OSError, ValueError, subprocess.SubprocessError):
        return fallback


def scenes_from_smplx_npz(
    root: Path,
    duration_sec: float = 15.0,
    fps: float | None = None,
    windows_per_sequence: int = 10,
    samples_per_window: int = 16,
    exclude_sequences: set[str] | None = None,
) -> list[dict[str, Any]]:
    smplx_roots = [root/'smplx_with_distortion', root/'smplx']
    files = []
    for sr in smplx_roots:
        if sr.exists():
            files.extend(sr.glob('*_person*.npz'))
    by_seq: dict[str, list[Path]] = {}
    for f in files:
        name = f.name
        if '_person' not in name:
            continue
        seq = name.split('_person')[0]
        by_seq.setdefault(seq, []).append(f)
    scenes = []
    exclude_sequences = exclude_sequences or set()
    for seq, seq_files in sorted(by_seq.items()):
        if seq in exclude_sequences:
            continue
        if len(seq_files) < 2:
            continue
        chosen = sorted(seq_files)
        arrays = [np.load(f) for f in chosen]
        n = min(len(a['transl']) for a in arrays)
        if n < 3:
            continue
        video = root/'videos'/seq/'0.mp4'
        source_fps = float(fps) if fps else video_fps(video)
        win = int(round(duration_sec * source_fps))
        if n <= win + 2:
            starts = [0]
        else:
            max_start = n - win - 1
            count = max(1, windows_per_sequence)
            starts = [int(round(i * max_start / max(1, count - 1))) for i in range(count)]
        for wi, start_idx in enumerate(starts, 1):
            end_idx = min(n - 1, start_idx + win)
            # A 15-second temporal question must not be inferred from only three
            # poses. Sample approximately once per second by default while
            # preserving exact start/end anchors.
            sample_count = max(3, int(samples_per_window))
            idxs = sorted(set(
                int(round(start_idx + i * (end_idx - start_idx) / (sample_count - 1)))
                for i in range(sample_count)
            ))
            frames = []
            for idx in idxs:
                people = []
                for pi, arr in enumerate(arrays):
                    transl = [float(x) for x in arr['transl'][idx].tolist()]
                    orient = [float(x) for x in arr['global_orient'][idx].tolist()] if 'global_orient' in arr.files else [0.0,0.0,0.0]
                    R = rodrigues(orient)
                    forward = unit(mat_vec(R, [0.0, 0.0, 1.0]))
                    people.append({'id': chr(ord('A') + pi), 'source_person_file': chosen[pi].name, 'pelvis': transl, 'head': [transl[0], transl[1] + 1.6, transl[2]], 'forward': forward})
                frames.append({'t': (idx - start_idx) / source_fps, 'frame_id': int(arrays[0]['frame_ids'][idx]) if 'frame_ids' in arrays[0].files else idx, 'source_index': idx, 'people': people})
            scene_id = f'hoi_m3_{seq}_win{wi:02d}'
            rel_clip = f'./outputs/hoim3/{seq}/win{wi:02d}_view0_15s.mp4' if video.exists() else None
            scenes.append({'scene_id': scene_id, 'title': f'HOI-M3 · {seq} · window {wi:02d} · {len(chosen)} metric 3D tracks', 'dataset': 'HOI-M3', 'duration_sec': duration_sec, 'source_fps': source_fps, 'start_sec': start_idx/source_fps, 'end_sec': end_idx/source_fps, 'start_frame': int(arrays[0]['frame_ids'][start_idx]) if 'frame_ids' in arrays[0].files else start_idx, 'source_start_index': start_idx, 'source_end_index': end_idx, 'tracked_person_count': len(chosen), 'source_smplx_files': [str(f.relative_to(root)) for f in chosen], 'video_clip': rel_clip, 'source_video': str(video.relative_to(Path.cwd())) if video.exists() else None, 'frames': frames, 'blockers': [], 'objects': []})
    return scenes


def discover_json_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    preferred = []
    for pat in ['**/humans_in_space*.json', '**/*multihuman*.json', '**/*hoi*.json', '**/*.json']:
        for p in path.glob(pat):
            if p.is_file() and p not in preferred and p.stat().st_size < 200_000_000:
                preferred.append(p)
        if preferred:
            break
    return preferred


def load_scenes(input_path: Path) -> list[dict[str, Any]]:
    scenes = []
    for p in discover_json_files(input_path):
        try:
            raw = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        candidates = raw if isinstance(raw, list) else raw.get('scenes') if isinstance(raw, dict) and isinstance(raw.get('scenes'), list) else [raw]
        for c in candidates:
            if isinstance(c, dict):
                scene = normalize_scene(c, p)
                if scene:
                    scenes.append(scene)
    return scenes


def write_template(path: Path) -> None:
    template = {
        'scene_id': 'hoi_m3_sequence_name',
        'title': 'HOI-M3 · sequence name · 15s window',
        'duration_sec': 15,
        'frames': [
            {'t': 0.0, 'frame_id': 0, 'people': [
                {'id': 'A', 'pelvis': [0.0, 0.0, 0.0], 'head': [0.0, 1.6, 0.0], 'forward': [1.0, 0.0, 0.0]},
                {'id': 'B', 'pelvis': [2.0, 0.0, 0.5], 'head': [2.0, 1.6, 0.5], 'forward': [-1.0, 0.0, 0.0]},
            ]},
            {'t': 7.5, 'frame_id': 225, 'people': [
                {'id': 'A', 'pelvis': [0.6, 0.0, 0.1], 'head': [0.6, 1.6, 0.1], 'forward': [1.0, 0.0, 0.0]},
                {'id': 'B', 'pelvis': [1.5, 0.0, 0.3], 'head': [1.5, 1.6, 0.3], 'forward': [-1.0, 0.0, 0.0]},
            ]},
            {'t': 15.0, 'frame_id': 450, 'people': [
                {'id': 'A', 'pelvis': [1.0, 0.0, 0.2], 'head': [1.0, 1.6, 0.2], 'forward': [0.0, 0.0, 1.0]},
                {'id': 'B', 'pelvis': [1.8, 0.0, 0.5], 'head': [1.8, 1.6, 0.5], 'forward': [-1.0, 0.0, 0.0]},
            ]},
        ],
        'blockers': [{'id': 'partition', 'center': [1.3, 1.6, 0.2], 'radius': 0.35}],
        'objects': [{'id': 'table', 'center': [1.2, 0.0, 1.2], 'radius': 0.6}],
    }
    path.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, default=Path('data/HOI-M3'))
    ap.add_argument('--output', type=Path, default=Path('outputs/qa/hoim3_multihuman_scenes.json'))
    ap.add_argument('--windows-per-sequence', type=int, default=10)
    ap.add_argument('--samples-per-window', type=int, default=16, help='Pose samples per 15-second window; default is about 1 Hz including both endpoints.')
    ap.add_argument('--exclude-sequences', default='', help='Comma-separated sequence ids to skip when visual people are not fully covered by local 3D tracks.')
    ap.add_argument('--write-template', type=Path)
    args = ap.parse_args()
    if args.write_template:
        out = args.write_template if args.write_template.is_absolute() else Path.cwd() / args.write_template
        out.parent.mkdir(parents=True, exist_ok=True)
        write_template(out)
        print(out)
        return
    inp = args.input if args.input.is_absolute() else Path.cwd() / args.input
    out = args.output if args.output.is_absolute() else Path.cwd() / args.output
    if not inp.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'dataset': 'HOI-M3', 'status': 'missing_dataset', 'input': str(inp), 'expected': ['HOI-M3 toolbox export JSON with frames/humans/persons', 'or canonical humans_in_space*.json produced from HOI-M3 SMPL-X/MHR trajectories']}, ensure_ascii=False, indent=2), encoding='utf-8')
        print('missing_dataset', inp)
        print(out)
        return
    scenes = load_scenes(inp)
    if not scenes:
        exclude = {x.strip() for x in args.exclude_sequences.split(',') if x.strip()}
        scenes = scenes_from_smplx_npz(
            inp,
            windows_per_sequence=args.windows_per_sequence,
            samples_per_window=args.samples_per_window,
            exclude_sequences=exclude,
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'dataset': 'HOI-M3', 'status': 'ok' if scenes else 'missing_usable_scenes', 'scenes': scenes}, ensure_ascii=False, indent=2), encoding='utf-8')
    print('scenes', len(scenes))
    print(out)


if __name__ == '__main__':
    main()
