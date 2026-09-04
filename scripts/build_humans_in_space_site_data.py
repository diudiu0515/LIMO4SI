#!/usr/bin/env python3
"""Build video-level Humans in Space QA site data.

This generator upgrades the old frame-centered display into ~15 second video QA.
Each question must satisfy T(Q) and H(Q) and S(Q): it uses multiple frames,
is grounded in a human reference frame/state, and asks about spatial relations,
visibility, topology, or human-induced spatial change.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from build_task1_task3_site_data import DEFAULT_SUMMARIES, ok_samples, label  # noqa: E402
from build_perspective_qa import find_camera_calibration  # noqa: E402
from limo4si.human_frame import apply_forward_sign, build_human_frame, describe_relation  # noqa: E402
from limo4si.perspective_qa import (  # noqa: E402
    _joint_xyz,
    human_centric_answer,
    relation_phrase,
    static_reachability_answer,
    visibility_answer,
)

TASKS = [
    {
        'id': 'task1_dynamic_human_referenced_relations',
        'name': 'Task 1 · Dynamic Human-Referenced Relations',
        'description': 'How human-referenced spatial relations change or remain stable across the video window.',
    },
    {
        'id': 'task2_human_induced_spatial_change',
        'name': 'Task 2 · Human-Induced Spatial Change',
        'description': 'Whether the human action appears to change object spatial state; current implementation uses mask-motion proxy when full 3D object tracking is unavailable.',
    },
    {
        'id': 'task3_human_scene_topological_reasoning',
        'name': 'Task 3 · Human–Scene Topological Reasoning',
        'description': 'How the human trajectory passes landmarks and which side of the path objects lie on.',
    },
    {
        'id': 'task4_multi_human_relational_dynamics',
        'name': 'Task 4 · Multi-Human Relational Dynamics',
        'description': 'Multi-human relation changes; falls back to an explicit evidence note when only one human pose is available.',
    },
]

REL_WORDS = {
    'left': 'left', 'right': 'right', 'front': 'front', 'behind': 'behind',
    'above': 'above', 'below': 'below', 'slightly_above': 'slightly above',
    'slightly_below': 'slightly below', 'same_lateral_position': 'centered laterally',
    'same_longitudinal_position': 'near the body origin', 'same_height': 'same height',
}


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def xyz_dict(annotation: Mapping) -> dict[str, list[float]]:
    return {k.replace('-', '_'): [float(v[a]) for a in 'xyz'] for k, v in annotation.items() if v and all(a in v for a in 'xyz')}


def fps_for_video(video: Path) -> float:
    out = subprocess.run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', str(video)
    ], capture_output=True, text=True, check=True).stdout.strip()
    parts = out.split('/')
    return float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(parts[0])


def build_clip(root: Path, group: dict, seconds: float) -> None:
    first = group['raw_summary']['samples'][0]
    video = root / 'data/egoexo4d/takes' / first['take_name'] / 'frame_aligned_videos/downscaled/448' / f"{first['camera']}.mp4"
    fps = fps_for_video(video)
    center_frame = int(first['frame'])
    start = max(0.0, center_frame / fps - seconds / 2.0)
    clip_filename = group.get('clip_filename') or f'showcase_clip_{int(round(seconds))}s.mp4'
    out = (root / group['summary_path']).parent / clip_filename
    group['video_clip'] = './' + str(out.relative_to(root))
    group['video_window'] = {
        'center_frame': center_frame, 'fps': fps, 'start_sec': round(start, 3),
        'duration_sec': seconds, 'source_video': str(video.relative_to(root)),
    }
    if out.exists() and out.stat().st_size > 1000:
        return
    subprocess.run([
        'ffmpeg', '-y', '-ss', f'{start:.3f}', '-i', str(video), '-t', str(seconds),
        '-vf', 'scale=720:-2', '-an', '-c:v', 'libx264', '-preset', 'veryfast',
        '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def body_timeline(root: Path, sample: dict, seconds: float, stride_frames: int = 15) -> list[dict]:
    body_path = Path(sample.get('inputs', {}).get('body_pose', ''))
    if not body_path.exists():
        body_path = root / 'data/egoexo4d/annotations/ego_pose/val/body/automatic' / f"{sample['take_uid']}.json"
    body = load_json(body_path)
    override_path = root / "configs" / "spatial_orientation_overrides.json"
    overrides = load_json(override_path) if override_path.exists() else {}
    orientation = overrides.get(str(sample.get("take_uid")), {})
    forward_sign = int(orientation.get("forward_sign", 1))
    if forward_sign not in (-1, 1):
        raise ValueError(f"Invalid temporal forward_sign for {sample.get('take_uid')}: {forward_sign}")
    take_video = root / 'data/egoexo4d/takes' / sample['take_name'] / 'frame_aligned_videos/downscaled/448' / f"{sample['camera']}.mp4"
    fps = fps_for_video(take_video)
    center = int(sample['frame'])
    half = int(round(seconds * fps / 2.0))
    frame_ids = list(range(max(0, center - half), center + half + 1, stride_frames))
    # Force start/mid/end anchors.
    for fr in [max(0, center - half), center, center + half]:
        if fr not in frame_ids:
            frame_ids.append(fr)
    rows = []
    for fr in sorted(set(frame_ids)):
        rec = body.get(str(fr))
        if rec and rec[0].get('annotation3D'):
            joints = xyz_dict(rec[0]['annotation3D'])
            try:
                frame = build_human_frame(joints)
                frame = apply_forward_sign(frame, forward_sign)
            except Exception:
                continue
            rows.append({'frame': fr, 't_sec_from_center': (fr-center)/fps, 'joints': rec[0]['annotation3D'], 'joints_xyz': joints, 'human_frame': frame, 'orientation': {'forward_sign': forward_sign, 'source': orientation.get('source', 'nose_or_body_cross_product')}})
    return rows


def relation_for(obj: Mapping, frame) -> dict:
    human_xyz = frame.world_to_human(obj['object_xyz_world_m'])
    return describe_relation(human_xyz)


def relation_label(rel: Mapping) -> str:
    parts = []
    for k in ['lateral_relation', 'longitudinal_relation', 'vertical_relation']:
        v = rel.get(k)
        if v and v not in {'same_height', 'same_lateral_position', 'same_longitudinal_position'}:
            parts.append(REL_WORDS.get(v, str(v)))
    return ' and '.join(parts) if parts else 'roughly aligned'


def object_relation_track(obj: Mapping, timeline: Sequence[Mapping]) -> dict:
    states = []
    for row in timeline:
        rel = relation_for(obj, row['human_frame'])
        states.append({
            'frame': row['frame'], 't_sec_from_center': row['t_sec_from_center'],
            'relation_label': relation_label(rel), 'relation': rel,
            'orientation': row.get('orientation'),
        })
    if not states:
        return {'object_id': obj.get('object_id'), 'states': [], 'status': 'missing_evidence'}
    first, mid, last = states[0], min(states, key=lambda r: abs(r['t_sec_from_center'])), states[-1]
    changes = []
    for axis in ['lateral_relation', 'longitudinal_relation', 'vertical_relation']:
        a = first['relation'].get(axis); b = last['relation'].get(axis)
        if a != b:
            changes.append({'axis': axis, 'from': a, 'to': b})
    return {
        'object_id': obj.get('object_id'), 'status': 'ok', 'states': states,
        'start': first, 'middle': mid, 'end': last,
        'changed': bool(changes), 'changes': changes,
        'dominant_relations': dict(Counter(s['relation_label'] for s in states).most_common()),
    }


def human_motion_summary(timeline: Sequence[Mapping]) -> dict:
    if len(timeline) < 2:
        return {'status': 'missing_evidence'}
    origins = [row['human_frame'].origin for row in timeline]
    disp = math.dist(origins[0], origins[-1])
    path = sum(math.dist(a, b) for a, b in zip(origins, origins[1:]))
    f0 = timeline[0]['human_frame'].forward
    f1 = timeline[-1]['human_frame'].forward
    cosang = max(-1.0, min(1.0, sum(a*b for a,b in zip(f0,f1))))
    turn = math.degrees(math.acos(cosang))
    return {'status': 'ok', 'start_frame': timeline[0]['frame'], 'end_frame': timeline[-1]['frame'], 'sample_count': len(timeline), 'displacement_m': disp, 'path_length_m': path, 'body_turn_deg': turn}


def visibility_track(obj: Mapping, timeline: Sequence[Mapping], samples: Sequence[Mapping]) -> dict:
    states = []
    for row in timeline:
        vis = visibility_answer(obj, row['joints'], candidates=samples)
        states.append({'frame': row['frame'], 't_sec_from_center': row['t_sec_from_center'], 'visible': vis.get('visible'), 'visibility_state': vis.get('visibility_state'), 'fov_zone': vis.get('fov_zone'), 'blocker': (vis.get('blocker') or {}).get('object_id'), 'angle_deg': vis.get('angle_to_view_direction_deg'), 'raw': vis})
    visible_values = [s['visible'] for s in states]
    changed = len(set(visible_values)) > 1
    cause = 'unchanged'
    if changed:
        first = states[0]; last = states[-1]
        if first['fov_zone'] != last['fov_zone'] and first['blocker'] == last['blocker']:
            cause = 'human_rotation_or_head/body_direction_change'
        elif first['blocker'] != last['blocker']:
            cause = 'listed_intervening_object_change'
        else:
            cause = 'mixed_or_uncertain_visibility_change'
    return {'object_id': obj.get('object_id'), 'states': states, 'changed': changed, 'change_cause_proxy': cause}


def hand_distance_track(objects: Sequence[Mapping], timeline: Sequence[Mapping]) -> dict:
    rows = []
    for obj in objects:
        xyz = obj.get('object_xyz_world_m')
        if not xyz: continue
        series = []
        for row in timeline:
            ds = []
            for name in ['left_wrist', 'right_wrist']:
                p = _joint_xyz(row['joints'], name)
                if p is not None:
                    ds.append(float(math.dist(p.tolist(), xyz)))
            if ds:
                series.append({'frame': row['frame'], 't_sec_from_center': row['t_sec_from_center'], 'distance_m': min(ds)})
        if series:
            rows.append({'object_id': obj.get('object_id'), 'start_distance_m': series[0]['distance_m'], 'end_distance_m': series[-1]['distance_m'], 'min_distance_m': min(x['distance_m'] for x in series), 'approach_m': series[0]['distance_m'] - series[-1]['distance_m'], 'series': series})
    rows.sort(key=lambda r: (-r['approach_m'], r['min_distance_m']))
    return {'chosen': rows[0] if rows else None, 'candidates': rows}




def nearest_object_change(objects: Sequence[Mapping], timeline: Sequence[Mapping]) -> dict:
    if not objects or len(timeline) < 2:
        return {'status': 'missing_evidence', 'missing_evidence': ['objects or human timeline']}
    rows = []
    for row in [timeline[0], timeline[len(timeline)//2], timeline[-1]]:
        origin = row['human_frame'].origin
        ranked = []
        for obj in objects:
            xyz = obj.get('object_xyz_world_m')
            if xyz:
                ranked.append({'object_id': obj.get('object_id'), 'distance_m': float(math.dist(origin, xyz))})
        ranked.sort(key=lambda x: x['distance_m'])
        rows.append({'frame': row['frame'], 't_sec_from_center': row['t_sec_from_center'], 'nearest': ranked[0] if ranked else None, 'ranked': ranked[:4]})
    return {'status': 'ok', 'states': rows, 'changed': (rows[0]['nearest'] or {}).get('object_id') != (rows[-1]['nearest'] or {}).get('object_id')}


def reachability_change_for_object(obj: Mapping, timeline: Sequence[Mapping], candidates: Sequence[Mapping]) -> dict:
    if len(timeline) < 2:
        return {'status': 'missing_evidence', 'missing_evidence': ['human timeline']}
    states = []
    for row in [timeline[0], timeline[len(timeline)//2], timeline[-1]]:
        ans = static_reachability_answer(obj, row['joints'], candidates=candidates)
        states.append({'frame': row['frame'], 't_sec_from_center': row['t_sec_from_center'], 'reachable': ans.get('reachable'), 'best_arm': ans.get('best_arm'), 'grasp_cue': ans.get('grasp_cue'), 'obstacle_free': ans.get('obstacle_free'), 'raw': ans})
    vals = [x.get('reachable') for x in states]
    return {'status': 'ok', 'object_id': obj.get('object_id'), 'states': states, 'changed': len(set(vals)) > 1}


def relation_change_cause_proxy(track: Mapping, motion: Mapping) -> dict:
    turn = float(motion.get('body_turn_deg') or 0.0)
    disp = float(motion.get('displacement_m') or 0.0)
    if not track.get('changed'):
        cause = 'relation_stays_stable'
    elif turn >= 45.0 and disp < 0.75:
        cause = 'mostly_human_body_rotation'
    elif disp >= 0.75 and turn < 45.0:
        cause = 'mostly_human_translation'
    else:
        cause = 'mixed_rotation_and_translation'
    return {'status': 'ok', 'object_id': track.get('object_id'), 'changed': track.get('changed'), 'body_turn_deg': turn, 'displacement_m': disp, 'cause_proxy': cause, 'T_Q': True, 'H_Q': True, 'S_Q': True}

def mask_bbox_center(record: Mapping) -> tuple[float, float] | None:
    clicks = record.get('intSegClicks', {}) or {}
    ul = clicks.get('upperLeft') or []
    br = clicks.get('bottomRight') or []
    if ul and br:
        return ((float(ul[0]['x']) + float(br[0]['x'])) / 2.0, (float(ul[0]['y']) + float(br[0]['y'])) / 2.0)
    return None


def object_mask_motion_proxy(root: Path, obj: Mapping, timeline: Sequence[Mapping]) -> dict:
    rel_path = root / 'data/egoexo4d/annotations/relations_val.json'
    rels = load_json(rel_path)['annotations']
    uid, cam, oid = obj['take_uid'], obj['camera'], obj['object_id']
    track = rels.get(uid, {}).get('object_masks', {}).get(oid, {}).get(cam, {}).get('annotation', {})
    centers = []
    for row in [timeline[0], timeline[len(timeline)//2], timeline[-1]] if timeline else []:
        rec = track.get(str(row['frame']))
        c = mask_bbox_center(rec) if rec else None
        centers.append({'frame': row['frame'], 'bbox_center_xy': c})
    valid = [c['bbox_center_xy'] for c in centers if c['bbox_center_xy']]
    if len(valid) < 2:
        return {'status': 'missing_evidence', 'object_id': oid, 'missing_evidence': ['object mask/bbox at start and end frames'], 'centers': centers}
    shift = math.dist(valid[0], valid[-1])
    return {'status': 'ok', 'object_id': oid, 'bbox_center_shift_px': shift, 'appears_moved_2d': shift > 35.0, 'centers': centers, 'approximations': ['2D mask/bbox motion proxy; not full 3D object tracking and not proof of human-caused motion']}


def path_side_analysis(objects: Sequence[Mapping], timeline: Sequence[Mapping]) -> dict:
    """Classify object centers against the human path on the horizontal plane.

    Ego-Exo4D world coordinates are not guaranteed to use x/z as the floor
    plane.  We therefore remove the start-frame human-up component and compute
    signed side in the 3D horizontal plane.  Positive means right of travel,
    negative means left of travel, with a 0.25 m lateral dead zone.
    """
    if len(timeline) < 2:
        return {'status': 'missing_evidence', 'missing_evidence': ['human trajectory']}
    start = list(timeline[0]['human_frame'].origin)
    end = list(timeline[-1]['human_frame'].origin)
    up = list(timeline[0]['human_frame'].up)
    d = [end[i] - start[i] for i in range(3)]
    dot_d_up = sum(d[i] * up[i] for i in range(3))
    d_h = [d[i] - dot_d_up * up[i] for i in range(3)]
    norm = math.sqrt(sum(v * v for v in d_h))
    if norm < 1e-6:
        return {'status': 'ok', 'motion_state': 'little_translation', 'objects_by_path_side': [], 'path_start_world_m': start, 'path_end_world_m': end}
    rows = []
    for obj in objects:
        xyz = obj.get('object_xyz_world_m')
        if not xyz:
            continue
        v = [float(xyz[i]) - start[i] for i in range(3)]
        dot_v_up = sum(v[i] * up[i] for i in range(3))
        v_h = [v[i] - dot_v_up * up[i] for i in range(3)]
        cross = [d_h[1] * v_h[2] - d_h[2] * v_h[1], d_h[2] * v_h[0] - d_h[0] * v_h[2], d_h[0] * v_h[1] - d_h[1] * v_h[0]]
        signed_lateral_m = sum(cross[i] * up[i] for i in range(3)) / norm
        along = sum(v_h[i] * d_h[i] for i in range(3)) / norm
        side = 'right_of_path' if signed_lateral_m > 0.25 else 'left_of_path' if signed_lateral_m < -0.25 else 'near_path_centerline'
        rows.append({'object_id': obj.get('object_id'), 'side_of_path': side, 'signed_side_value_m2': sum(cross[i] * up[i] for i in range(3)), 'signed_lateral_m': signed_lateral_m, 'along_path_m': along})
    return {'status': 'ok', 'objects_by_path_side': rows, 'path_start_world_m': start, 'path_end_world_m': end, 'horizontal_axis_source': 'start human-frame up projection'}



def make_mcq(question_type: str, answer: str, result: Mapping) -> dict:
    """Create deterministic four-option multiple-choice metadata for the site.

    The correct option is rotated deterministically so it is not always A.
    The full computed answer remains as the explanation/evidence-backed answer.
    """
    if question_type == 'relation_change_over_video':
        track = result.get('object_track', {})
        start = (track.get('start') or {}).get('relation_label', 'the start relation')
        end = (track.get('end') or {}).get('relation_label', 'the end relation')
        correct = f"It changes from {start} to {end}." if track.get('changed') else f"It remains {start} throughout the sampled window."
        distractors = [
            f"It stays {end} throughout the whole clip.",
            f"It changes from {end} to {start}.",
            "There is not enough temporal evidence to compare the beginning and end.",
        ]
    elif question_type == 'relation_consistency_over_video':
        left = result.get('always_left_objects') or []
        correct = ("; ".join(left) + " stay(s) on the person's left side.") if left else "No listed object stays on the person's left side for the entire sampled clip."
        distractors = [
            "All listed objects stay on the person's left side for the entire clip.",
            "Only the nearest object stays on the person's left side for the entire clip.",
            "The answer can be determined from the middle frame alone.",
        ]
    elif question_type == 'visibility_change_cause_over_video':
        vt = result.get('visibility_track', {})
        states = vt.get('states') or []
        first = states[0].get('visibility_state', 'unknown') if states else 'unknown'
        last = states[-1].get('visibility_state', 'unknown') if states else 'unknown'
        cause = vt.get('change_cause_proxy', 'unknown')
        correct = f"It goes from {first} to {last}; the proxy cause is {cause}."
        distractors = [
            "It is continuously visible with no visibility-state change.",
            "It is continuously outside the field of view throughout the clip.",
            "The change is caused only by the camera moving, not by the person or blockers.",
        ]
    elif question_type == 'hand_approach_over_video':
        chosen = (result.get('hand_distance_tracks') or {}).get('chosen')
        candidates = (result.get('hand_distance_tracks') or {}).get('candidates') or []
        if chosen:
            correct = f"{chosen['object_id']} is approached most by the nearest hand."
            other = [c['object_id'] for c in candidates if c.get('object_id') != chosen.get('object_id')]
            distractors = [
                f"{other[0]} is approached most by the nearest hand." if other else "Another listed object is approached most by the nearest hand.",
                "No listed object has any wrist-distance change over the clip.",
                "The object farthest from the hand is the strongest approach target.",
            ]
        else:
            correct = "No listed object has enough wrist trajectory evidence."
            distractors = ["The first listed object is definitely approached most.", "The nearest object by pelvis distance is definitely approached most.", "All listed objects are approached equally."]
    elif question_type == 'relation_change_cause_proxy_over_video':
        cause = result.get('cause_proxy', 'mixed_rotation_and_translation')
        correct = f"The best geometry proxy is {cause}."
        distractors = [
            "It is explained only by the camera moving, not by the person's state.",
            "It can be answered from the middle frame alone.",
            "There is no human-referenced spatial change to compare over time.",
        ]
    elif question_type == 'nearest_object_change_over_video':
        states = result.get('states') or []
        if states:
            a = (states[0].get('nearest') or {}).get('object_id')
            b = (states[-1].get('nearest') or {}).get('object_id')
            correct = f"Nearest object changes: {result.get('changed')}; start={a}, end={b}."
        else:
            correct = "There is not enough object/human trajectory evidence."
        distractors = [
            "The farthest object is used instead of the nearest object.",
            "Only image size is used, without 3D distance.",
            "This asks about object category, not human-object distance.",
        ]
    elif question_type == 'reachability_change_over_video':
        states = result.get('states') or []
        if states:
            correct = f"Reachability changes: {result.get('changed')}; start={states[0].get('reachable')}, end={states[-1].get('reachable')}."
        else:
            correct = "There is not enough arm/hand pose evidence."
        distractors = [
            "Reachability is judged only by object label.",
            "The camera's distance to the object is the reachability measure.",
            "The answer ignores the person's arm and hand pose over time.",
        ]
    elif question_type == 'object_motion_proxy_over_video':
        if result.get('status') == 'missing_evidence':
            correct = "There is not enough start/end mask evidence to judge object motion."
        else:
            correct = f"Apparent 2D object motion is {result.get('appears_moved_2d')} by the mask/bbox proxy."
        distractors = [
            "The object is proven to have been moved in 3D by the human.",
            "The object is proven completely static in world coordinates.",
            "This can be answered from a single frame without temporal evidence.",
        ]
    elif question_type == 'objects_along_human_path_sides':
        rows = result.get('objects_by_path_side') or []
        correct = '; '.join(f"{r['object_id']}: {r['side_of_path']}" for r in rows[:3]) if rows else "The person translates too little to define reliable path sides."
        distractors = [
            "All listed objects are on the same side of the path.",
            "Path side cannot be discussed because no human trajectory is used.",
            "The answer is based only on image left/right, not the person's path.",
        ]
    elif question_type == 'multi_human_evidence_gate':
        correct = "No: this case lacks a second temporally tracked human pose, so multi-human dynamics is unavailable."
        distractors = [
            "Yes: two tracked people are available and line of sight is computed.",
            "Yes: the camera alone is enough to infer multi-human relational dynamics.",
            "No: because spatial reasoning is not relevant to this task.",
        ]
    else:
        correct = str(answer)
        distractors = ["The opposite relation is correct.", "There is no temporal evidence.", "The answer is determined only by image coordinates."]

    # Deduplicate while preserving order; pad if needed.
    raw = [correct] + distractors
    unique = []
    for item in raw:
        if item not in unique:
            unique.append(item)
    while len(unique) < 4:
        unique.append(f"Distractor option {len(unique)}")
    unique = unique[:4]
    offset = sum(ord(c) for c in question_type + correct) % 4
    ordered = unique[1:]
    ordered.insert(offset, correct)
    labels = ['A', 'B', 'C', 'D']
    options = [{'label': label, 'text': text} for label, text in zip(labels, ordered)]
    correct_label = labels[offset]
    return {'options': options, 'correct_option': correct_label, 'correct_answer': correct, 'explanation': str(answer)}

def add(qas, task_id, qtype, question, answer, method, result, raw):
    task = next(t for t in TASKS if t['id'] == task_id)
    mcq = make_mcq(qtype, answer, result)
    qas.append({'task_id': task_id, 'task_name': task['name'], 'question_type': qtype, 'question': question, 'options': mcq['options'], 'correct_option': mcq['correct_option'], 'correct_answer': mcq['correct_answer'], 'answer': answer, 'explanation': mcq['explanation'], 'status': result.get('status', 'ok'), 'method': method, 'result_json': result, 'raw_json': raw})


def build_group(root: Path, summary_path: Path, clip_seconds: float) -> dict:
    summary = load_json(summary_path)
    samples = ok_samples(summary)
    if not samples:
        raise ValueError(f'No usable samples in {summary_path}')
    first = samples[0]
    second = samples[1] if len(samples) > 1 else first
    timeline = body_timeline(root, first, clip_seconds)
    if len(timeline) < 2:
        raise ValueError(f'No temporal pose timeline for {summary_path}')
    tracks = [object_relation_track(obj, timeline) for obj in samples]
    motion = human_motion_summary(timeline)
    vis_track = visibility_track(second, timeline, samples)
    hand_track = hand_distance_track(samples, timeline)
    topo = path_side_analysis(samples, timeline)
    mask_motion = object_mask_motion_proxy(root, first, timeline)

    first_track = tracks[0]
    always_left = [t['object_id'] for t in tracks if t['states'] and all(s['relation']['lateral_relation'] == 'left' for s in t['states'])]
    stable = [t['object_id'] for t in tracks if t['states'] and len(set(s['relation_label'] for s in t['states'])) == 1]

    qas = []
    start_label = first_track['start']['relation_label']; end_label = first_track['end']['relation_label']
    change_text = 'changed' if first_track['changed'] else 'stayed the same'
    add(qas, 'task1_dynamic_human_referenced_relations', 'relation_change_over_video',
        f"Across this {clip_seconds:.0f}-second clip, how does the {label(first)} change relative to the person from the beginning to the end?",
        f"It {change_text}: at the beginning it is {start_label}; by the end it is {end_label}. The person moves {motion.get('displacement_m',0):.2f} m and turns about {motion.get('body_turn_deg',0):.1f}°.",
        'Samples body-centric human frames across the full clip and compares object relation at start/middle/end.',
        {'status': 'ok', 'answer_type': 'dynamic_relation_change', 'object_track': first_track, 'human_motion': motion, 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'target': first, 'timeline_frames': [r['frame'] for r in timeline]})

    add(qas, 'task1_dynamic_human_referenced_relations', 'relation_consistency_over_video',
        f"During the whole {clip_seconds:.0f}-second clip, which listed objects remain consistently on the person's left side, if any?",
        (f"The objects consistently on the person's left are: {', '.join(always_left)}." if always_left else "No listed object stays on the person's left side for the entire sampled clip."),
        'Checks every sampled frame in the clip; a candidate must be left in all sampled body-centric frames.',
        {'status': 'ok', 'answer_type': 'relation_consistency', 'always_left_objects': always_left, 'stable_relation_objects': stable, 'object_tracks': tracks, 'sampled_frame_count': len(timeline), 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'candidates': samples})

    vcause = vis_track['change_cause_proxy']
    vfirst, vlast = vis_track['states'][0], vis_track['states'][-1]
    add(qas, 'task1_dynamic_human_referenced_relations', 'visibility_change_cause_over_video',
        f"Over the {clip_seconds:.0f}-second clip, does the {label(second)} become visible or invisible to the person, and is the change more consistent with body/head rotation or a listed blocker?",
        f"Visibility changes: {vis_track['changed']}. It starts as {vfirst['visibility_state']} and ends as {vlast['visibility_state']}; the proxy cause is {vcause}.",
        'Computes visibility at multiple frames using head/body direction and listed-object sightline blockers, then compares FOV/blocker state changes.',
        {'status': 'ok', 'answer_type': 'dynamic_visibility_cause', 'visibility_track': vis_track, 'human_motion': motion, 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'target': second, 'candidates': samples})

    chosen = hand_track.get('chosen')
    add(qas, 'task1_dynamic_human_referenced_relations', 'hand_approach_over_video',
        f"Across the whole {clip_seconds:.0f}-second clip, which listed object does the person's hand approach most?",
        (f"The strongest hand-approach target is {chosen['object_id']}: nearest-hand distance changes from {chosen['start_distance_m']:.2f} m to {chosen['end_distance_m']:.2f} m, with a maximum approach of {chosen['approach_m']:.2f} m." if chosen else "No listed object has enough wrist trajectory evidence."),
        'Uses wrist-to-object distance time series over the full clip, not a single frame.',
        {'status': 'ok' if chosen else 'missing_evidence', 'answer_type': 'video_hand_approach', 'hand_distance_tracks': hand_track, 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'candidates': samples})

    cause_proxy = relation_change_cause_proxy(first_track, motion)
    add(qas, 'task1_dynamic_human_referenced_relations', 'relation_change_cause_proxy_over_video',
        f"Across this {clip_seconds:.0f}-second clip, is the {label(first)}'s relation change better explained by the person's rotation, translation, or both?",
        f"The relation-change cause proxy is {cause_proxy['cause_proxy']}: body turn is {cause_proxy['body_turn_deg']:.1f}° and pelvis displacement is {cause_proxy['displacement_m']:.2f} m.",
        'Compares relation timeline with human body turn and pelvis displacement; this is a geometry proxy for cause, not a semantic action label.',
        cause_proxy,
        {'target': first, 'motion': motion})

    nearest_change = nearest_object_change(samples, timeline)
    ns0 = nearest_change.get('states', [{}])[0].get('nearest') if nearest_change.get('states') else None
    ns2 = nearest_change.get('states', [{}])[-1].get('nearest') if nearest_change.get('states') else None
    add(qas, 'task1_dynamic_human_referenced_relations', 'nearest_object_change_over_video',
        f"From the beginning to the end of this {clip_seconds:.0f}-second clip, does the nearest listed object to the person change?",
        (f"Nearest object changes: {nearest_change.get('changed')}. It starts as {(ns0 or {}).get('object_id')} at {(ns0 or {}).get('distance_m', 0):.2f} m and ends as {(ns2 or {}).get('object_id')} at {(ns2 or {}).get('distance_m', 0):.2f} m." if nearest_change.get('status') == 'ok' else 'Missing object or human trajectory evidence.'),
        'Ranks listed objects by pelvis-to-object 3D Euclidean distance at start/middle/end of the clip.',
        {**nearest_change, 'answer_type': 'nearest_object_change', 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'candidates': samples})

    reach_change = reachability_change_for_object(first, timeline, samples)
    rs0 = reach_change.get('states', [{}])[0] if reach_change.get('states') else {}
    rs2 = reach_change.get('states', [{}])[-1] if reach_change.get('states') else {}
    add(qas, 'task1_dynamic_human_referenced_relations', 'reachability_change_over_video',
        f"Does the person's current-pose reachability to the {label(first)} change across the {clip_seconds:.0f}-second clip?",
        f"Reachability changes: {reach_change.get('changed')}. It starts as reachable={rs0.get('reachable')} and ends as reachable={rs2.get('reachable')}.",
        'Computes static reachability at start/middle/end using arm/hand geometry and object center, then compares states over time.',
        {**reach_change, 'answer_type': 'reachability_change', 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'target': first})

    add(qas, 'task2_human_induced_spatial_change', 'object_motion_proxy_over_video',
        f"During this {clip_seconds:.0f}-second clip, does the {label(first)} appear to be spatially moved by the human?",
        (f"The 2D mask/bbox center shifts by {mask_motion.get('bbox_center_shift_px',0):.1f} px, so apparent object motion is {mask_motion.get('appears_moved_2d')}. This is only a motion proxy, not proof of human-caused 3D movement." if mask_motion['status']=='ok' else "There is not enough start/end mask evidence to judge object motion in this clip."),
        'Compares object mask/bbox centers at start/middle/end. This satisfies temporal evidence but is explicitly marked as 2D proxy without full 3D object tracking.',
        {**mask_motion, 'answer_type': 'human_induced_spatial_change_proxy', 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'target': first})

    side_rows = topo.get('objects_by_path_side', [])
    side_answer = '; '.join(f"{r['object_id']} is {r['side_of_path']}" for r in side_rows[:3]) if side_rows else 'The person translates too little to define a reliable path side.'
    add(qas, 'task3_human_scene_topological_reasoning', 'objects_along_human_path_sides',
        f"Considering the person's actual path through the {clip_seconds:.0f}-second clip, which listed objects lie to the left or right side of that path?",
        side_answer,
        'Uses the human pelvis trajectory from start to end and classifies listed object centers by signed side of the path line.',
        {**topo, 'answer_type': 'human_path_topology', 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'candidates': samples})

    add(qas, 'task4_multi_human_relational_dynamics', 'multi_human_evidence_gate',
        "Does this clip support a multi-human relational dynamics question such as whether two people regain line of sight?",
        "Not for this case: the current local evidence contains one tracked human pose, so multi-human dynamics is marked unavailable rather than guessed.",
        'Evidence gate for Task 4. Requires at least two temporally tracked human poses; current EgoPose sample supplies one primary person.',
        {'status': 'missing_evidence', 'answer_type': 'multi_human_evidence_gate', 'missing_evidence': ['second temporally tracked human pose'], 'T_Q': True, 'H_Q': True, 'S_Q': True},
        {'timeline_frames': [r['frame'] for r in timeline]})

    rel_dir = summary_path.parent.relative_to(root)
    group = {
        'name': summary_path.parent.name,
        'title': f"{first.get('take_name')} · {first.get('camera')} · {clip_seconds:.0f}s dynamic window around frame {first.get('frame')}",
        'original_image': './' + str(rel_dir / 'showcase_original_multi.jpg'),
        'topdown_image': './' + str(rel_dir / 'showcase_topdown_multi.jpg'),
        'summary_path': str(summary_path.relative_to(root)),
        'raw_summary': summary,
        'dynamic_timeline': {'frames': [r['frame'] for r in timeline], 'sample_count': len(timeline), 'human_motion': motion},
        'qa': qas,
    }
    build_clip(root, group, clip_seconds)
    return group


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs='*', default=DEFAULT_SUMMARIES)
    ap.add_argument('--output', type=Path, default=Path('site/qa_benchmark/data.js'))
    ap.add_argument('--clip-seconds', type=float, default=15.0)
    args = ap.parse_args()
    groups = [build_group(ROOT, ROOT / s, args.clip_seconds) for s in args.summaries]
    data = {
        'title': 'Humans in Space',
        'subtitle': 'Dynamic human-referenced spatial reasoning in exocentric videos',
        'source': '15-second video windows built from selected outputs/spatial summary.json files',
        'benchmark_rule': 'A valid QA should satisfy T(Q) ∧ H(Q) ∧ S(Q) = 1.',
        'tasks': TASKS,
        'groups': groups,
    }
    out = ROOT / args.output if not args.output.is_absolute() else args.output
    out.write_text('window.QA_DATA = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')
    print(out)
    print('groups', len(groups), 'qa', sum(len(g['qa']) for g in groups))


if __name__ == '__main__':
    main()
