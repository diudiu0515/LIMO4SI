"""Multi-human dynamic spatial reasoning utilities.

Input schema is intentionally simple so EgoHumans / Inter-X / CMU Panoptic /
HOI-M3 style data can be converted into it:

{
  "scene_id": "...",
  "duration_sec": 15,
  "frames": [
    {"t": 0, "people": [
      {"id": "A", "pelvis": [x,y,z], "forward": [x,y,z], "head": [x,y,z]}, ...
    ]}
  ],
  "landmarks": [{"id":"table", "center":[x,y,z]}],
  "blockers": [{"id":"partition", "center":[x,y,z], "radius":0.4}]
}
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

Vec = Sequence[float]


def dot(a: Vec, b: Vec) -> float:
    return float(sum(float(x)*float(y) for x, y in zip(a, b)))


def sub(a: Vec, b: Vec) -> list[float]:
    return [float(x)-float(y) for x, y in zip(a, b)]


def norm(a: Vec) -> float:
    return math.sqrt(dot(a, a))


def unit(a: Vec) -> list[float]:
    n = norm(a)
    return [0.0, 0.0, 1.0] if n < 1e-9 else [float(x)/n for x in a]


def horizontal_unit(a: Vec) -> list[float]:
    """Normalize a direction on the metric ground plane (world X/Z)."""
    return unit([float(a[0]), 0.0, float(a[2])])


def dist(a: Vec, b: Vec) -> float:
    return norm(sub(a, b))


def person(frame: Mapping[str, Any], pid: str) -> Mapping[str, Any] | None:
    for p in frame.get('people', []):
        if p.get('id') == pid:
            return p
    return None


def facing_score(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    af = horizontal_unit(a.get('forward', [0, 0, 1]))
    bf = horizontal_unit(b.get('forward', [0, 0, 1]))
    ab = horizontal_unit(sub(b['pelvis'], a['pelvis']))
    ba = horizontal_unit(sub(a['pelvis'], b['pelvis']))
    return (dot(af, ab) + dot(bf, ba)) / 2.0


def facing_label(score: float) -> str:
    if score > 0.55:
        return 'facing_each_other'
    if score < -0.35:
        return 'back_to_back_or_away'
    return 'side_by_side_or_oblique'


def horizontal_side(anchor: Mapping[str, Any], target: Mapping[str, Any]) -> str:
    f = horizontal_unit(anchor.get('forward', [0, 0, 1]))
    right = [f[2], 0.0, -f[0]]
    v = sub(target['pelvis'], anchor['pelvis'])
    lateral = dot(v, right)
    forward = dot(v, f)
    lr = 'right' if lateral > 0.15 else 'left' if lateral < -0.15 else 'center'
    fb = 'front' if forward > 0.15 else 'behind' if forward < -0.15 else 'same_depth'
    return f'{lr}_{fb}'


def line_blocked(a_xyz: Vec, b_xyz: Vec, blockers: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not blockers:
        return {
            'status': 'not_evaluated_no_blocker_geometry',
            'blocked': None,
            'blocker': None,
        }
    ab = sub(b_xyz, a_xyz)
    ab_len2 = dot(ab, ab)
    best = None
    for blk in blockers or []:
        c = blk.get('center')
        if not c or ab_len2 < 1e-9:
            continue
        t = max(0.0, min(1.0, dot(sub(c, a_xyz), ab) / ab_len2))
        proj = [float(a_xyz[i]) + t * ab[i] for i in range(3)]
        d = dist(c, proj)
        radius = float(blk.get('radius', 0.25))
        if 0.05 < t < 0.95 and d <= radius:
            cand = {'blocker_id': blk.get('id'), 'distance_to_line_m': d, 'radius_m': radius, 'depth_fraction': t}
            if best is None or d < best['distance_to_line_m']:
                best = cand
    return {'status': 'evaluated', 'blocked': best is not None, 'blocker': best}


def forward_field_state(a: Mapping[str, Any], b: Mapping[str, Any], half_angle_deg: float = 60.0) -> dict[str, Any]:
    """Check whether each person lies inside the other's body-forward field.

    This is intentionally not called gaze or unoccluded line-of-sight: HOI-M3
    root orientation supplies a body-forward proxy, while dense occluder
    geometry is a separate requirement.
    """
    af = horizontal_unit(a.get('forward', [0, 0, 1]))
    bf = horizontal_unit(b.get('forward', [0, 0, 1]))
    ab = horizontal_unit(sub(b['pelvis'], a['pelvis']))
    ba = horizontal_unit(sub(a['pelvis'], b['pelvis']))
    a_cos = max(-1.0, min(1.0, dot(af, ab)))
    b_cos = max(-1.0, min(1.0, dot(bf, ba)))
    a_angle = math.degrees(math.acos(a_cos))
    b_angle = math.degrees(math.acos(b_cos))
    a_sees = a_angle <= half_angle_deg
    b_sees = b_angle <= half_angle_deg
    if a_sees and b_sees:
        state = 'mutual_body_forward_field'
    elif a_sees:
        state = 'a_only_body_forward_field'
    elif b_sees:
        state = 'b_only_body_forward_field'
    else:
        state = 'neither_body_forward_field'
    return {
        'state': state,
        'a_contains_b': a_sees,
        'b_contains_a': b_sees,
        'a_to_b_angle_deg': a_angle,
        'b_to_a_angle_deg': b_angle,
        'half_angle_deg': half_angle_deg,
        'scope': 'body-forward field only; does not prove gaze or absence of occlusion',
    }


def pair_timeline(scene: Mapping[str, Any], a_id: str = 'A', b_id: str = 'B') -> dict[str, Any]:
    rows = []
    blockers = scene.get('blockers', [])
    for fr in scene.get('frames', []):
        a = person(fr, a_id); b = person(fr, b_id)
        if not a or not b:
            continue
        d = dist(a['pelvis'], b['pelvis'])
        score = facing_score(a, b)
        los = line_blocked(a.get('head', a['pelvis']), b.get('head', b['pelvis']), blockers)
        forward_field = forward_field_state(a, b)
        rows.append({
            't': fr.get('t'),
            'frame_id': fr.get('frame_id'),
            'distance_m': d,
            'facing_score': score,
            'facing_state': facing_label(score),
            'b_relative_to_a': horizontal_side(a, b),
            'a_relative_to_b': horizontal_side(b, a),
            'line_of_sight_blocked': los['blocked'],
            'line_of_sight_status': los['status'],
            'blocker': los['blocker'],
            'body_forward_field': forward_field,
            'evidence': {
                'person_a': {'id': a.get('id'), 'pelvis_xyz_m': a.get('pelvis'), 'head_xyz_m': a.get('head'), 'forward_unit': unit(a.get('forward', [0, 0, 1]))},
                'person_b': {'id': b.get('id'), 'pelvis_xyz_m': b.get('pelvis'), 'head_xyz_m': b.get('head'), 'forward_unit': unit(b.get('forward', [0, 0, 1]))},
                'computed_from': scene.get('evidence_source') or ['SMPL-X transl as pelvis/root proxy', 'SMPL-X global_orient-derived body forward', 'head = pelvis + 1.6m proxy when fitted head joints are not loaded'],
            },
        })
    if not rows:
        return {'status': 'missing_evidence', 'missing_evidence': ['two tracked people over time']}
    evaluated_los = [r['line_of_sight_blocked'] for r in rows if r['line_of_sight_status'] == 'evaluated']
    return {
        'status': 'ok',
        'pair': [a_id, b_id],
        'states': rows,
        'distance_change_m': rows[-1]['distance_m'] - rows[0]['distance_m'],
        'facing_changed': rows[0]['facing_state'] != rows[-1]['facing_state'],
        'los_changed': len(set(evaluated_los)) > 1 if evaluated_los else None,
        'line_of_sight_evidence_status': 'evaluated' if evaluated_los else 'missing_blocker_geometry',
    }


def multi_person_metric_timeline(scene: Mapping[str, Any], min_people: int = 3) -> dict[str, Any]:
    """Compute all pair distances for a stable, fully annotated multi-person set."""
    configured = [str(value) for value in scene.get("metric_person_ids", [])]
    if configured:
        person_ids = configured
    else:
        counts: dict[str, int] = {}
        frames = scene.get("frames", [])
        for frame in frames:
            for value in {str(person.get("id")) for person in frame.get("people", [])}:
                counts[value] = counts.get(value, 0) + 1
        person_ids = sorted(
            (person_id for person_id, count in counts.items() if frames and count / len(frames) >= 0.8),
            key=str,
        )
    if len(person_ids) < min_people:
        return {
            "status": "missing_evidence",
            "missing_evidence": [f"{min_people} stable metric person tracks"],
            "metric_person_ids": person_ids,
        }

    states = []
    for frame in scene.get("frames", []):
        by_id = {str(value.get("id")): value for value in frame.get("people", [])}
        if any(person_id not in by_id for person_id in person_ids):
            continue
        pair_rows = []
        for left, right in combinations(person_ids, 2):
            pair_rows.append({
                "pair": f"{left}–{right}",
                "distance_m": dist(by_id[left]["pelvis"], by_id[right]["pelvis"]),
            })
        pair_rows.sort(key=lambda row: row["distance_m"])
        states.append({
            "t": frame.get("t"),
            "frame_id": frame.get("frame_id"),
            "pair_distances_m": pair_rows,
            "closest_pair": pair_rows[0]["pair"],
            "closest_pair_margin_m": pair_rows[1]["distance_m"] - pair_rows[0]["distance_m"],
        })
    if not states:
        return {
            "status": "missing_evidence",
            "missing_evidence": ["frames containing every stable metric person"],
            "metric_person_ids": person_ids,
        }
    return {
        "status": "ok",
        "metric_person_ids": person_ids,
        "person_count": len(person_ids),
        "states": states,
        "scope": "all-pairs metric 3D pelvis geometry; no 2D detections substitute for missing people",
    }


def multihuman_qas(scene: Mapping[str, Any]) -> list[dict[str, Any]]:
    tl = pair_timeline(scene)
    if tl['status'] != 'ok':
        return []
    s0 = tl['states'][0]; sm = tl['states'][len(tl['states'])//2]; s1 = tl['states'][-1]
    def mcq(correct: str, distractors: list[str], seed: str) -> tuple[list[dict[str,str]], str]:
        vals=[]
        for x in [correct]+distractors:
            if x not in vals: vals.append(x)
        while len(vals)<4: vals.append(f'Not enough evidence option {len(vals)}')
        off=sum(ord(c) for c in seed+correct)%4
        ordered=vals[1:4]; ordered.insert(off, correct)
        labels=['A','B','C','D']
        return [{'label':l,'text':t} for l,t in zip(labels,ordered)], labels[off]
    rows=[]
    specs=[
        ('distance_change_between_people', 'Across the clip, do person A and person B move closer together or farther apart?', f"Their distance changes from {s0['distance_m']:.2f} m to {s1['distance_m']:.2f} m, so the signed change is {tl['distance_change_m']:.2f} m.", ['They are judged only by image left/right.', 'There is no temporal comparison.', 'Only one person is tracked.']),
        ('facing_relation_change', 'Across the clip, how does the facing relation between A and B change?', f"It goes from {s0['facing_state']} to {s1['facing_state']}.", ['They remain in exactly the same facing state.', 'The answer uses object masks only.', 'The camera distance alone decides this.']),
        ('line_of_sight_change', 'Does the line of sight between A and B become blocked or unblocked during the clip?', f"Line-of-sight blocked changes from {s0['line_of_sight_blocked']} to {s1['line_of_sight_blocked']}.", ['Visibility is not checked.', 'A and B are the same person.', 'Only the middle frame is used.']),
        ('relative_position_from_a', 'At the end of the clip, where is B relative to A in A\'s body-centric frame?', f"At the end, B is {s1['b_relative_to_a']} relative to A.", ['B is described relative to the camera.', 'The start-frame relation is used as the answer.', 'No body-centric frame is used.']),
        ('mid_clip_social_spacing', 'Around the middle of the clip, are A and B close enough for direct interaction?', f"At the middle, their distance is {sm['distance_m']:.2f} m, so direct interaction is {'plausible' if sm['distance_m'] < 1.8 else 'not close'} by distance proxy.", ['This is an object-object topology question.', 'No metric distance is used.', 'The final frame alone is used.']),
    ]
    # Never emit a clear/blocked line-of-sight QA when no blocker geometry
    # was provided.  None means unknown, not clear.
    if tl.get('line_of_sight_evidence_status') != 'evaluated':
        specs = [spec for spec in specs if spec[0] != 'line_of_sight_change']
    for qtype, question, correct, distractors in specs:
        opts, lab = mcq(correct, distractors, qtype)
        rows.append({'task_id':'task4_multi_human_relational_dynamics','task_name':'Task 4 · Multi-Human Relational Dynamics','question_type':qtype,'question':question,'options':opts,'correct_option':lab,'correct_answer':correct,'answer':correct,'explanation':correct,'status':'ok','method':'Uses two tracked human pelvis/head/forward trajectories over the video window; computes distance, body-centric relative position, facing score, and simple line-of-sight blocker geometry.','result_json':{'scene_id':scene.get('scene_id'),'answer_type':qtype,'pair_timeline':tl,'T_Q':True,'H_Q':True,'S_Q':True}})
    return rows
