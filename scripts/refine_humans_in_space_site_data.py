#!/usr/bin/env python3
"""Keep only high-signal video-level Humans in Space questions.

This is a second, semantic quality pass after the evidence precision gate. It
removes questions whose measured change is too small, whose answer is constant,
or whose geometry is only an unsupported proxy. It can merge freshly rebuilt
Ego-Exo groups with the already verified HOI-M3 groups.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_ALIAS_PATH = ROOT / 'configs/object_display_aliases.json'
DISPLAY_ALIASES = json.loads(_ALIAS_PATH.read_text(encoding='utf-8')) if _ALIAS_PATH.exists() else {}

def load(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    m = re.search(r'window\.QA_DATA\s*=\s*(.*);\s*$', text, re.S)
    if not m: raise ValueError(f'Cannot parse {path}')
    return json.loads(m.group(1))

def write(path: Path, data: dict[str, Any]) -> None:
    path.write_text('window.QA_DATA = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')

def merge(base: dict[str, Any], ego: dict[str, Any] | None) -> dict[str, Any]:
    if not ego: return base
    replacements = {g.get('name'): g for g in ego.get('groups', [])}
    groups=[]; seen=set()
    for g in base.get('groups', []):
        name=g.get('name')
        if name in replacements: groups.append(replacements[name]); seen.add(name)
        else: groups.append(g)
    for g in ego.get('groups', []):
        if g.get('name') not in seen and not any(x.get('name') == g.get('name') for x in groups): groups.append(g)
    out=dict(base); out['groups']=groups
    return out

def update_reachability_text(q: dict[str, Any]) -> None:
    r=q.get('result_json') or {}; states=r.get('states') or []
    if not states: return
    seq=' → '.join(str(x.get('reachable')) for x in states)
    correct=f'Reachability state across the sampled timeline (start → middle → end): {seq}.'
    q['correct_answer']=correct; q['answer']=correct; q['explanation']=correct
    label=q.get('correct_option')
    for opt in q.get('options', []):
        if opt.get('label') == label: opt['text']=correct

def pretty_obj(value: Any) -> str:
    text = '' if value is None else str(value)
    for aliases in DISPLAY_ALIASES.values():
        for source, shown in aliases.items():
            text = text.replace(source, shown)
            text = text.replace(source.replace('_0', ''), shown)
    text = text.replace('_0', '').replace('_1', '')
    text = re.sub(r'_([2-9])$', r' #\1', text)
    return text.replace('_', ' ')


def pretty_state(value: Any) -> str:
    return str(value or 'unknown').replace('_', ' ').replace('or', ' or ')


def assign_options(q: dict[str, Any], correct: str, distractors: list[str]) -> None:
    vals=[]
    for x in [correct] + distractors:
        if x not in vals: vals.append(x)
    while len(vals) < 4: vals.append('The available evidence does not support this alternative.')
    vals=vals[:4]
    offset=sum(ord(c) for c in q.get('question_type','') + correct) % 4
    ordered=vals[1:]; ordered.insert(offset, correct)
    labels=['A','B','C','D']; q['options']=[{'label':a,'text':b} for a,b in zip(labels,ordered)]; q['correct_option']=labels[offset]
    q['correct_answer']=correct; q['answer']=correct; q['explanation']=correct


def rewrite_options(q: dict[str, Any]) -> None:
    """Rewrite visible questions/options from measured states, not generic fillers."""
    r=q.get('result_json') or {}; t=q.get('question_type','')
    if t == 'relation_change_over_video':
        tr=r.get('object_track') or {}; obj=pretty_obj(tr.get('object_id')); start=pretty_state((tr.get('start') or {}).get('relation_label')); end=pretty_state((tr.get('end') or {}).get('relation_label'))
        q['question']=f"As the person moves or turns, how does the {obj}'s position relative to the person change from the first to the last sampled time?"
        correct=f"The {obj} is treated as fixed in the scene; relative to the person it changes from {start} to {end}."
        assign_options(q,correct,[f"It remains {start} relative to the person throughout.",f"It changes from {end} to {start} relative to the person.","Only its height relation changes; its left/right and front/behind relations stay the same."])
    elif t == 'relation_consistency_over_video':
        objs=[pretty_obj(x) for x in (r.get('always_left_objects') or [])]
        q['question']='Which listed object stays on the person\'s left side at every sampled time in the clip?'
        correct=', '.join(objs)+' stays on the person\'s left side at every sampled time.'
        assign_options(q,correct,['No listed object stays on the person\'s left side at every sampled time.','Every listed object stays on the person\'s left side.','Only the object nearest the person at the midpoint stays on the left.'])
    elif t == 'hand_approach_over_video':
        tracks=r.get('hand_distance_tracks') or {}; chosen=tracks.get('chosen') or {}; obj=pretty_obj(chosen.get('object_id')); candidates=[pretty_obj(x.get('object_id')) for x in tracks.get('candidates') or [] if x.get('object_id')]
        q['question']='Across the full 15-second clip, which listed object shows the largest decrease in nearest-wrist distance?'
        correct=f"{obj} shows the largest nearest-wrist approach ({float(chosen.get('approach_m') or 0):.2f} m)."
        others=[f"{x} shows the largest nearest-wrist approach." for x in candidates if x != obj][:2]
        fallbacks=['No listed object shows a measurable nearest-wrist decrease.','The nearest-wrist distance increases rather than decreases.','The object farthest from the wrist shows the largest approach.']
        assign_options(q,correct,others+fallbacks)
    elif t == 'reachability_change_over_video':
        states=r.get('states') or []; seq=' → '.join(str(x.get('reachable')) for x in states); obj=pretty_obj(r.get('object_id'))
        q['question']=f"Does the person\'s estimated reachability to the {obj} vary at any sampled time in the 15-second clip?"
        correct=f"Yes. The reachability sequence (start → middle → end) is {seq}."
        vals=[str(x.get('reachable')) for x in states]; rev=' → '.join(reversed(vals)); const=' → '.join([vals[0]]*len(vals)) if vals else 'unknown'
        assign_options(q,correct,[f"No; it stays {vals[0] if vals else 'the same'} at every sampled time.",f"The sequence is reversed: {rev}.",f"It stays constant at {const}."])
    elif t == 'objects_along_human_path_sides':
        rows=r.get('objects_by_path_side') or []; clear=[x for x in rows if x.get('side_of_path') in {'left_of_path','right_of_path'}]
        correct='; '.join(f"{pretty_obj(x.get('object_id'))}: {'left' if x.get('side_of_path')=='left_of_path' else 'right'} of the travel path" for x in clear)
        q['question']='Relative to the person\'s actual travel path through the clip, which listed objects are clearly on its left or right side?'
        swapped='; '.join(f"{pretty_obj(x.get('object_id'))}: {'right' if x.get('side_of_path')=='left_of_path' else 'left'} of the travel path" for x in clear)
        assign_options(q,correct,[swapped or 'All listed objects are near the path centerline.','All listed objects are on the same side of the travel path.','Image-left/image-right alone is enough; the person\'s path is not used.'])
    elif t == 'distance_change_between_people':
        states=(r.get('pair_timeline') or {}).get('states') or []; a=states[0].get('distance_m',0) if states else 0; b=states[-1].get('distance_m',0) if states else 0; delta=b-a
        q['question']='Over the full clip, how does the metric pelvis-to-pelvis distance between person A and person B change?'
        direction='decreases, so they move closer' if delta < 0 else 'increases, so they move farther apart'
        correct=f"It {direction}: {a:.2f} m at the start to {b:.2f} m at the end (Δ {delta:+.2f} m)."
        assign_options(q,correct,[f"It changes in the opposite direction: {b:.2f} m to {a:.2f} m.",f"It stays approximately constant at about {(a+b)/2:.2f} m.",'The answer is determined from image left/right, not metric distance.'])
    elif t == 'facing_relation_change':
        states=(r.get('pair_timeline') or {}).get('states') or []; a=pretty_state(states[0].get('facing_state')) if states else 'unknown'; b=pretty_state(states[-1].get('facing_state')) if states else 'unknown'
        q['question']='How does the body-facing relation between A and B change from the start to the end of the clip?'
        correct=f"It changes from {a} to {b}."
        assign_options(q,correct,[f"It stays {a} throughout.",f"It changes in the opposite direction, from {b} to {a}.",'Only the distance is measured; body orientation is not used.'])
    elif t == 'relative_position_from_a':
        states=(r.get('pair_timeline') or {}).get('states') or []; end=pretty_state(states[-1].get('b_relative_to_a')) if states else 'unknown'; start=pretty_state(states[0].get('b_relative_to_a')) if states else 'unknown'
        q['question']='At the end of the clip, where is B relative to A in A\'s body-centric frame?'
        correct=f"At the end, B is {end} relative to A."
        assign_options(q,correct,[f"B is {start} relative to A, using the start frame instead.",f"B is {end.replace('left','right',1) if 'left' in end else end.replace('right','left',1)} relative to A.",'B is described in the camera frame rather than A\'s body-centric frame.'])
    elif t == 'mid_clip_social_spacing':
        states=(r.get('pair_timeline') or {}).get('states') or []; mid=states[len(states)//2] if states else {}; d=float(mid.get('distance_m') or 0)
        q['question']='Around the middle of the clip, is the metric distance between A and B within the near-interaction range?'
        correct=f"Yes; their midpoint distance is {d:.2f} m, which is within the near-interaction range by the distance proxy."
        assign_options(q,correct,[f"No; their midpoint distance is {d:.2f} m and is outside the near-interaction range.",'This is an object-to-object topology question.','The midpoint is not used; only the final frame is considered.'])


def keep_qa(q: dict[str, Any]) -> tuple[bool, str]:
    t=q.get('question_type',''); r=q.get('result_json') or {}
    if t in {'relation_change_cause_proxy_over_video','nearest_object_change_over_video','object_motion_proxy_over_video','line_of_sight_change'}:
        return False, 'constant/causal proxy is not high-signal enough for showcase'
    if t == 'relation_change_over_video':
        tr=r.get('object_track') or {}; motion=r.get('human_motion') or {}
        if not tr.get('changed'): return False, 'relation does not change'
        axes={x.get('axis') for x in tr.get('changes') or []}
        a=((tr.get('start') or {}).get('relation') or {}).get('human_xyz_m') or {}
        b=((tr.get('end') or {}).get('relation') or {}).get('human_xyz_m') or {}
        component_delta=max((abs(float(a.get(k, 0))-float(b.get(k, 0))) for k in ('right','forward','up')), default=0.0)
        motion_strong=float(motion.get('body_turn_deg') or 0) >= 25 or float(motion.get('displacement_m') or 0) >= .15
        # A categorical boundary crossing by only a few centimetres is usually
        # threshold noise; require either clear motion or >=25 cm component change.
        if not motion_strong and component_delta < .25:
            return False, f'change too small/threshold noise ({component_delta:.3f}m)'
        if not ({'lateral_relation','longitudinal_relation'} & axes) and component_delta < .25:
            return False, 'only small vertical/threshold change'
        return True, 'robust relation change'
    if t == 'relation_consistency_over_video':
        return (bool(r.get('always_left_objects')), 'has at least one object consistently left' if r.get('always_left_objects') else 'empty consistency answer')
    if t == 'hand_approach_over_video':
        chosen=(r.get('hand_distance_tracks') or {}).get('chosen') or {}
        approach=float(chosen.get('approach_m') or 0)
        return (approach >= .15, f'approach={approach:.3f}m' if approach >= .15 else f'approach too small={approach:.3f}m')
    if t == 'reachability_change_over_video':
        vals=[x.get('reachable') for x in r.get('states') or []]
        ok=len(vals) >= 2 and len(set(vals)) > 1
        if ok: update_reachability_text(q)
        return ok, 'reachability state changes over timeline' if ok else 'reachability is constant'
    if t == 'objects_along_human_path_sides':
        rows=r.get('objects_by_path_side') or []
        clear=[x for x in rows if x.get('side_of_path') in {'left_of_path','right_of_path'}]
        return (bool(clear), 'clear path-side evidence' if clear else 'all objects are in path dead zone')
    if t == 'visibility_change_cause_over_video':
        tr=r.get('visibility_track') or {}; states=tr.get('states') or []
        return (len(states)>=2 and bool(tr.get('changed')), 'visibility state changes over timeline' if tr.get('changed') else 'visibility is constant')
    if t == 'distance_change_between_people':
        p=r.get('pair_timeline') or {}; delta=abs(float(p.get('distance_change_m') or 0))
        return (delta >= .15, f'distance change={delta:.3f}m' if delta >= .15 else f'distance change too small={delta:.3f}m')
    if t == 'facing_relation_change':
        states=(r.get('pair_timeline') or {}).get('states') or []
        vals=[x.get('facing_state') for x in states]
        return (len(set(vals)) > 1, 'facing state changes' if len(set(vals)) > 1 else 'facing state is constant')
    if t == 'relative_position_from_a':
        states=(r.get('pair_timeline') or {}).get('states') or []
        vals=[x.get('b_relative_to_a') for x in states]
        return (len(set(vals)) > 1, 'relative position changes' if len(set(vals)) > 1 else 'relative position is constant')
    if t == 'mid_clip_social_spacing':
        states=(r.get('pair_timeline') or {}).get('states') or []
        if not states: return False, 'missing pair timeline'
        mid=states[len(states)//2]
        # Keep only genuinely close/interpersonal mid-clip examples; this is a
        # distance proxy, so do not use it for distant pairs.
        d=float(mid.get('distance_m') or 999)
        return (d <= 1.5, f'mid distance={d:.3f}m' if d <= 1.5 else f'mid distance too far={d:.3f}m')
    return True, 'not rejected by semantic gate'

def refine(data: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audit=[]; kept=[]
    for g in data.get('groups', []):
        nq=[]
        for idx,q in enumerate(g.get('qa', []),1):
            ok,reason=keep_qa(q)
            if ok: rewrite_options(q)
            audit.append({'case_id':g.get('name'),'qa_index':idx,'question_type':q.get('question_type'),'kept':ok,'reason':reason})
            if ok: nq.append(q)
        if nq:
            ng=dict(g); ng['qa']=nq; kept.append(ng)
    out=dict(data); out['groups']=kept
    out['semantic_quality_gate']={'status':'applied','policy':'keep only robust temporal change, meaningful path side, or substantial multi-human relation change','kept_groups':len(kept),'kept_qas':sum(len(g.get('qa',[])) for g in kept)}
    return out,audit

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--input',type=Path,default=Path('site/qa_benchmark/data.js')); ap.add_argument('--ego-data',type=Path); ap.add_argument('--output',type=Path,default=Path('site/qa_benchmark/data.js')); ap.add_argument('--audit-output',type=Path,default=Path('outputs/qa/semantic_quality_audit.json'))
    a=ap.parse_args(); inp=ROOT/a.input if not a.input.is_absolute() else a.input; outp=ROOT/a.output if not a.output.is_absolute() else a.output
    data=load(inp); ego=load(ROOT/a.ego_data) if a.ego_data and not a.ego_data.is_absolute() else (load(a.ego_data) if a.ego_data else None)
    refined,audit=refine(merge(data,ego)); write(outp,refined)
    apath=ROOT/a.audit_output if not a.audit_output.is_absolute() else a.audit_output; apath.parent.mkdir(parents=True,exist_ok=True); apath.write_text(json.dumps({'status':'ok','kept_groups':len(refined['groups']),'kept_qas':sum(len(g.get('qa',[])) for g in refined['groups']),'decisions':audit},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'kept_groups':len(refined['groups']),'kept_qas':sum(len(g.get('qa',[])) for g in refined['groups']),'rejected_qas':sum(not x['kept'] for x in audit)},ensure_ascii=False))
if __name__=='__main__': main()
