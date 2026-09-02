#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_site_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    match = re.search(r'window\.QA_DATA\s*=\s*(.*);\s*$', text, re.S)
    if not match:
        raise ValueError(f'Cannot parse QA data from {path}')
    return json.loads(match.group(1))


def esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ''), quote=True)


def dump_json(value: Any) -> str:
    return esc(json.dumps(value, ensure_ascii=False, indent=2))


# Human-reviewed display aliases. The source track ID remains in audit JSON;
# this only fixes the natural-language name shown to a reviewer.
_ALIAS_PATH = ROOT / 'configs/object_display_aliases.json'
DISPLAY_ALIASES = json.loads(_ALIAS_PATH.read_text(encoding='utf-8')) if _ALIAS_PATH.exists() else {}


def display_text(group: dict[str, Any], value: Any) -> str:
    text = '' if value is None else str(value)
    for source, shown in DISPLAY_ALIASES.get(str(group.get('name', '')), {}).items():
        text = text.replace(source, shown)
        text = text.replace(source.replace('_0', ''), shown)
    return text


def evidence_payload(group: dict[str, Any], qa: dict[str, Any]) -> dict[str, Any]:
    """Return only evidence needed by the current video-level question.

    Raw source JSON is deliberately not rendered here. It stays in data.js and
    output files for audit, while the website shows temporal/human/spatial
    fields that justify this particular question.
    """
    r = qa.get('result_json') or {}
    qtype = qa.get('question_type', '')
    out: dict[str, Any] = {'T/H/S': {'T': r.get('T_Q'), 'H': r.get('H_Q'), 'S': r.get('S_Q')}}
    if qtype in {'relation_change_over_video', 'turn_induced_relation_change_over_video'}:
        track = r.get('object_track') or {}
        out['object'] = track.get('object_id')
        out['relation_timeline'] = [
            {'frame': x.get('frame'), 'time_s': x.get('t_sec_from_center'), 'relation': x.get('relation_label'), 'distance_m': round((x.get('relation') or {}).get('distance_m', 0), 3)}
            for x in [track.get('start'), track.get('middle'), track.get('end') if track.get('end') else None]
            if x
        ]
        out['human_motion'] = r.get('human_motion')
    elif qtype == 'relation_consistency_over_video':
        out['sampled_frame_count'] = r.get('sampled_frame_count')
        out['always_left_objects'] = r.get('always_left_objects')
        out['stable_relation_objects'] = r.get('stable_relation_objects')
    elif qtype in {'nearest_object_consistency_over_video', 'nearest_and_vertical_consistency_over_video'}:
        out['sampled_frame_count'] = r.get('sampled_frame_count')
        out['nearest_counts'] = dict(__import__('collections').Counter(r.get('nearest_sequence') or []))
        if 'below_count' in r:
            out['below_count'] = r.get('below_count')
    elif qtype == 'multi_object_front_consistency_over_video':
        out['sampled_frame_count'] = r.get('sampled_frame_count')
        out['front_counts'] = r.get('front_counts')
    elif qtype == 'human_object_distance_pattern_over_video':
        track = r.get('object_track') or {}
        out['object'] = track.get('object_id')
        out['sampled_frame_count'] = r.get('sampled_frame_count')
        out['distance_series_m'] = [round(float(x), 3) for x in (r.get('distance_series_m') or [])]
        out['human_motion'] = r.get('human_motion')
    elif qtype in {'visibility_change_cause_over_video', 'body_forward_visibility_change_cause_over_video'}:
        track = r.get('visibility_track') or {}
        out['object'] = track.get('object_id')
        states = track.get('states') or []
        selected = states if len(states) <= 2 else [states[0], states[-1]]
        out['visibility_timeline'] = [
            {'frame': x.get('frame'), 'time_s': x.get('t_sec_from_center'), 'state': x.get('visibility_state'), 'fov_zone': x.get('fov_zone'), 'blocker': x.get('blocker'), 'angle_deg': x.get('angle_deg')}
            for x in selected
        ]
        out['change_cause_proxy'] = track.get('change_cause_proxy')
    elif qtype == 'hand_approach_over_video':
        tracks = r.get('hand_distance_tracks') or {}
        chosen = tracks.get('chosen') or {}
        out['chosen_object'] = chosen.get('object_id')
        out['nearest_hand_distance_m'] = {k: chosen.get(k) for k in ('start_distance_m', 'end_distance_m', 'min_distance_m', 'approach_m') if k in chosen}
        out['candidate_count'] = len(tracks.get('candidates') or [])
    elif qtype == 'relation_change_cause_proxy_over_video':
        out.update({k: r.get(k) for k in ('object_id', 'changed', 'body_turn_deg', 'displacement_m', 'cause_proxy')})
    elif qtype == 'nearest_object_change_over_video':
        states = r.get('states') or []
        selected = states if len(states) <= 2 else [states[0], states[-1]]
        out['nearest_timeline'] = [
            {'frame': x.get('frame'), 'time_s': x.get('t_sec_from_center'), 'object': (x.get('nearest') or {}).get('object_id'), 'distance_m': round((x.get('nearest') or {}).get('distance_m', 0), 3)}
            for x in selected
        ]
        out['changed'] = r.get('changed')
    elif qtype == 'reachability_change_over_video':
        states = r.get('states') or []
        selected = states if len(states) <= 2 else [states[0], states[-1]]
        out['object'] = r.get('object_id')
        out['reachability_timeline'] = [
            {'frame': x.get('frame'), 'time_s': x.get('t_sec_from_center'), 'reachable': x.get('reachable'), 'grasp_cue': x.get('grasp_cue'), 'obstacle_free': x.get('obstacle_free')}
            for x in selected
        ]
        out['changed'] = r.get('changed')
    elif qtype == 'objects_along_human_path_sides':
        out['path_start_world_m'] = r.get('path_start_world_m')
        out['path_end_world_m'] = r.get('path_end_world_m')
        out['objects_by_path_side'] = r.get('objects_by_path_side')
    elif qtype == 'visible_human_motion_ranking_2d':
        out['visible_2d_tracks'] = r.get('visible_2d_tracks')
        out['visual_person_audit'] = r.get('visual_person_audit') or group.get('visual_person_audit')
        out['distance_unit'] = 'normalized image-plane path length (fraction of frame diagonal)'
    elif qtype in {'closest_visible_pair_change_2d', 'visible_pair_topology_change_2d'}:
        out['start_pair_distances_normalized'] = r.get('start_pair_distances_normalized')
        out['end_pair_distances_normalized'] = r.get('end_pair_distances_normalized')
        out['visual_person_audit'] = r.get('visual_person_audit') or group.get('visual_person_audit')
        out['distance_unit'] = 'image-plane separation normalized by frame diagonal'
    elif qtype in {
        'distance_change_between_people', 'facing_relation_change',
        'line_of_sight_change', 'relative_position_from_a',
        'mid_clip_social_spacing', 'position_consistency_between_people',
        'dominant_facing_relation_over_video', 'metric_distance_pattern_over_video',
        'body_forward_visibility_consistency', 'body_centric_relation_change_over_video',
        'metric_separation_over_video', 'dominant_body_centric_position',
        'nonmonotonic_distance_pattern', 'approach_while_facing',
        'coupled_distance_relation_change', 'distance_out_and_back_over_video',
    }:
        pair = r.get('pair_timeline') or {}
        states = pair.get('states') or []
        out['pair'] = pair.get('pair')
        out['timeline'] = [
            {
                k: x.get(k)
                for k in (
                    't', 'frame_id', 'distance_m', 'facing_state',
                    'b_relative_to_a', 'a_relative_to_b',
                    'line_of_sight_status', 'line_of_sight_blocked', 'blocker',
                    'body_forward_field',
                )
                if k in x
            }
            for x in states
        ]
        out['computed_from'] = ((states[0].get('evidence') or {}).get('computed_from') if states else None)
        out['visual_person_audit'] = r.get('visual_person_audit') or group.get('visual_person_audit')
        for key in ('distance_series_m', 'relation_sequence', 'relation_counts', 'facing_counts', 'body_forward_field_counts', 'peak_sample_index'):
            if key in r:
                out[key] = r[key]
    else:
        for key in ('answer_type', 'sampled_frame_count', 'human_motion', 'changed'):
            if key in r:
                out[key] = r[key]
    return out


def build_static_html(data: dict[str, Any]) -> str:
    parts: list[str] = []
    parts.append('''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Humans in Space QA</title>
<link rel="stylesheet" href="styles.css" />
<style>
body{background:#eef2f7}.staticShell{max-width:1180px;margin:0 auto;padding:24px}.caseNav{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 22px}.caseNav a{padding:8px 10px;border:1px solid #dbe3ef;border-radius:999px;background:white;color:#2456d6;text-decoration:none;font-weight:750;font-size:13px}.staticCase{margin:28px 0;padding:18px;border-radius:18px;background:white;box-shadow:0 12px 30px rgba(15,23,42,.08)}.mediaGrid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:14px 0}.mediaGrid img{width:100%;border-radius:14px;border:1px solid #dbe3ef;background:#f8fafc}.videoPanel{margin:10px 0 16px}.inlineVideo{display:block;width:100%;max-height:440px;border-radius:14px;border:1px solid #dbe3ef;background:#0f172a}.originalVideoBox{margin:12px 0 16px;border:1px solid #dbe3ef;border-radius:14px;background:#f8fafc;padding:10px}.originalVideoBox summary{cursor:pointer;font-weight:850;color:#2456d6}.originalVideoBox .inlineVideo{margin-top:10px}.qaList{display:grid;gap:14px}.answerBox{display:block}.answerBox.hidden{display:none}.option{cursor:pointer;text-align:left;width:100%;font:inherit}.option.selected{outline:3px solid #7c3aed;background:#f3e8ff}.option.correctChoice{border-color:#16a34a;background:#dcfce7}.option.wrongChoice{border-color:#dc2626;background:#fee2e2}.submitAnswer{margin-top:10px;padding:9px 14px;border:0;border-radius:9px;background:#2456d6;color:#fff;font-weight:800;cursor:pointer}.submitAnswer:disabled{opacity:.45;cursor:not-allowed}.feedback{margin-top:10px;padding:10px;border-radius:10px;background:#f8fafc;border:1px solid #dbe3ef}.jsonBlock{white-space:pre-wrap}.topNote{padding:14px;border-radius:14px;background:white;border:1px solid #dbe3ef;color:#475569}.metaGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:14px 0}.metaBox{background:#f8fafc;border:1px solid #dbe3ef;border-radius:12px;padding:10px}.metaBox strong{display:block;color:#0f172a}.metaBox span{color:#64748b;font-size:13px}.coverageNotice{margin:12px 0;padding:12px 14px;border-radius:12px;border:1px solid #f59e0b;background:#fffbeb;color:#92400e;line-height:1.5}.coverageNotice.ok{border-color:#86efac;background:#f0fdf4;color:#166534}@media(max-width:900px){.mediaGrid{grid-template-columns:1fr}.staticShell{padding:14px}}
</style>
</head>
<body>
<main class="staticShell">
<h1>Humans in Space QA Benchmark</h1>
<p class="topNote">当前只展示 Task 1 和 Task 4。每个 case 只有一道需要整段视频证据的问题；先看 15 秒视频、人体定位与俯视图，再作答。提交后显示答案和本题直接相关的计算证据。</p>
<nav class="caseNav">
''')
    for i, _ in enumerate(data.get('groups', []), 1):
        parts.append(f'<a href="#case-{i}">Case {i}</a>')
    parts.append('</nav>')
    for i, group in enumerate(data.get('groups', []), 1):
        qa = group.get('qa', [])
        ok = sum(1 for q in qa if q.get('status') == 'ok')
        parts.append(f'<section class="staticCase" id="case-{i}">')
        parts.append(f'<p class="eyebrow">Case {i}</p><h2>{esc(group.get("title") or group.get("name"))}</h2>')
        parts.append('<div class="metaGrid">')
        parts.append(f'<div class="metaBox"><strong>{len(qa)}</strong><span>questions</span></div>')
        parts.append(f'<div class="metaBox"><strong>{ok}</strong><span>directly answerable</span></div>')
        vw = group.get('video_window') or {}
        parts.append(f'<div class="metaBox"><strong>{esc(vw.get("duration_sec", "15"))}s</strong><span>video window</span></div>')
        audit = group.get('visual_person_audit') or {}
        if audit:
            parts.append(f'<div class="metaBox"><strong>{esc(audit.get("persistent_visible_person_count"))} / {esc(audit.get("metric_3d_track_count"))}</strong><span>visible 2D / metric 3D tracks</span></div>')
        parts.append('</div>')
        if audit:
            cls = 'coverageNotice ok' if str(audit.get('status','')).startswith('complete_') else 'coverageNotice'
            parts.append(f'<div class="{cls}"><strong>多人覆盖校准：{esc(audit.get("status"))}</strong><br>原视频定位层覆盖 {esc(audit.get("persistent_visible_person_count"))} 条持续可见人物轨迹；米制距离和身体朝向只使用 {esc(audit.get("metric_3d_track_count"))} 条 SMPL-X 3D 轨迹，不把未标注人物编入三维答案。</div>')
        if group.get('video_clip'):
            parts.append('<div class="videoPanel"><div class="taskName">15-second evidence video</div>')
            parts.append(f'<video class="inlineVideo" controls muted playsinline preload="metadata"><source src="{esc(group["video_clip"])}" type="video/mp4">Your browser cannot play this video.</video></div>')
        if group.get('localization_video'):
            parts.append('<div class="videoPanel"><div class="taskName">Original-video person localization evidence (2D)</div>')
            parts.append(f'<video class="inlineVideo" controls muted playsinline preload="metadata"><source src="{esc(group["localization_video"])}" type="video/mp4">Your browser cannot play this video.</video></div>')
        if group.get('original_video'):
            parts.append('<details class="originalVideoBox"><summary>Show original full video on this page</summary>')
            parts.append(f'<video class="inlineVideo" controls muted playsinline preload="none"><source src="{esc(group["original_video"])}" type="video/mp4">Your browser cannot play this video.</video></details>')
        parts.append('<div class="mediaGrid">')
        if group.get('original_image'):
            original_caption = 'Original-video localization: boxes + head/pelvis points + persistent 2D IDs' if group.get('localization_image') else 'Photo / skeleton evidence'
            parts.append(f'<figure><img src="{esc(group["original_image"])}" loading="lazy" alt="original"><figcaption>{esc(original_caption)}</figcaption></figure>')
        if group.get('topdown_image'):
            topdown_caption = 'Metric 3D top-down map (SMPL-X tracks only)' if group.get('visual_person_audit') else 'Top-down human-centered map'
            parts.append(f'<figure><img src="{esc(group["topdown_image"])}" loading="lazy" alt="topdown"><figcaption>{esc(topdown_caption)}</figcaption></figure>')
        parts.append('</div><div class="qaList">')
        for qi, q in enumerate(qa, 1):
            status_class = 'ok' if q.get('status') == 'ok' else 'skip'
            parts.append(f'<article class="card {esc(q.get("task_id"))}">')
            parts.append('<div class="cardHead"><div>')
            parts.append(f'<div class="taskName">{esc(q.get("task_name"))}</div>')
            parts.append(f'<div class="questionType">{esc(q.get("question_type"))}</div>')
            parts.append(f'<div class="question">Q{qi}. {display_text(group, q.get("question"))}</div>')
            parts.append(f'</div><span class="pill {status_class}">{esc(q.get("status"))}</span></div>')
            parts.append(f'<div class="optionsGrid answerOptions" data-correct="{esc(q.get("correct_option"))}">')
            for opt in q.get('options', []):
                parts.append(f'<button type="button" class="option" data-option="{esc(opt.get("label"))}"><span class="optionLabel">{esc(opt.get("label"))}</span><span class="optionText">{display_text(group, opt.get("text"))}</span></button>')
            parts.append('</div>')
            parts.append('<button type="button" class="submitAnswer" disabled>提交答案</button>')
            parts.append('<div class="feedback" hidden></div>')
            parts.append(f'<div class="answerBox hidden"><div class="answerLabel">Correct answer: {esc(q.get("correct_option"))}</div><div class="answer">{display_text(group, q.get("correct_answer"))}</div><div class="explanation">{display_text(group, q.get("explanation"))}</div></div>')
            parts.append(f'<div class="methodBox">{display_text(group, q.get("method"))}</div>')
            parts.append('<details class="evidenceDetails"><summary>查看本题新版证据</summary>')
            parts.append(f'<pre class="jsonBlock">{display_text(group, dump_json(evidence_payload(group, q)))}</pre></details>')
            parts.append('</article>')
        parts.append('</div></section>')
    parts.append('''<script>
(function(){
  document.querySelectorAll('.card').forEach(function(card){
    var grid=card.querySelector('.answerOptions'), submit=card.querySelector('.submitAnswer'), feedback=card.querySelector('.feedback'), answer=card.querySelector('.answerBox');
    if(!grid||!submit) return;
    var selected=null, correct=grid.dataset.correct;
    grid.querySelectorAll('.option').forEach(function(btn){btn.addEventListener('click',function(){
      if(submit.dataset.done) return;
      grid.querySelectorAll('.option').forEach(function(x){x.classList.remove('selected');x.setAttribute('aria-pressed','false');});
      btn.classList.add('selected'); btn.setAttribute('aria-pressed','true'); selected=btn.dataset.option; submit.disabled=false;
    });});
    submit.addEventListener('click',function(){
      if(!selected||submit.dataset.done) return;
      submit.dataset.done='1'; submit.disabled=true;
      grid.querySelectorAll('.option').forEach(function(btn){
        if(btn.dataset.option===correct) btn.classList.add('correctChoice');
        if(btn.dataset.option===selected && selected!==correct) btn.classList.add('wrongChoice');
      });
      var ok=selected===correct; feedback.hidden=false; feedback.textContent=ok?'回答正确。':'回答不正确，正确选项是 '+correct+'。'; feedback.style.color=ok?'#166534':'#991b1b';
      answer.classList.remove('hidden');
    });
  });
})();
</script>''')
    parts.append('</main></body></html>')
    return '\n'.join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-js', type=Path, default=Path('site/qa_benchmark/data.js'))
    ap.add_argument('--output', type=Path, default=Path('site/qa_benchmark/index.html'))
    args = ap.parse_args()
    data_path = args.data_js if args.data_js.is_absolute() else ROOT / args.data_js
    out = args.output if args.output.is_absolute() else ROOT / args.output
    data = load_site_data(data_path)
    out.write_text(build_static_html(data), encoding='utf-8')
    print(out)
    print('groups', len(data.get('groups', [])), 'qa', sum(len(g.get('qa', [])) for g in data.get('groups', [])))


if __name__ == '__main__':
    main()
