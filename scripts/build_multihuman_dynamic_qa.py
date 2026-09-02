#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from limo4si.multihuman import multihuman_qas  # noqa: E402


def demo_scenes() -> list[dict[str, Any]]:
    return [
        {
            'scene_id': 'mh_demo_approach_turn', 'title': 'Multi-human demo · approach and turn', 'duration_sec': 15,
            'frames': [
                {'t': 0, 'people': [{'id': 'A', 'pelvis': [0,0,0], 'head':[0,1.6,0], 'forward':[1,0,0]}, {'id': 'B', 'pelvis': [3,0,0.8], 'head':[3,1.6,0.8], 'forward':[-1,0,-0.2]}]},
                {'t': 7.5, 'people': [{'id': 'A', 'pelvis': [0.8,0,0.1], 'head':[0.8,1.6,0.1], 'forward':[1,0,0]}, {'id': 'B', 'pelvis': [2.0,0,0.4], 'head':[2.0,1.6,0.4], 'forward':[-1,0,-0.1]}]},
                {'t': 15, 'people': [{'id': 'A', 'pelvis': [1.2,0,0.2], 'head':[1.2,1.6,0.2], 'forward':[0,0,1]}, {'id': 'B', 'pelvis': [1.7,0,0.5], 'head':[1.7,1.6,0.5], 'forward':[-1,0,0]}]},
            ],
            'blockers': [], 'landmarks': [{'id':'table','center':[1.5,0,1.2]}]
        },
        {
            'scene_id': 'mh_demo_partition_los', 'title': 'Multi-human demo · partition line-of-sight change', 'duration_sec': 15,
            'frames': [
                {'t': 0, 'people': [{'id': 'A', 'pelvis': [0,0,0], 'head':[0,1.6,0], 'forward':[1,0,0]}, {'id': 'B', 'pelvis': [3,0,0], 'head':[3,1.6,0], 'forward':[-1,0,0]}]},
                {'t': 7.5, 'people': [{'id': 'A', 'pelvis': [0.6,0,0.0], 'head':[0.6,1.6,0], 'forward':[1,0,0]}, {'id': 'B', 'pelvis': [2.6,0,0.0], 'head':[2.6,1.6,0], 'forward':[-1,0,0]}]},
                {'t': 15, 'people': [{'id': 'A', 'pelvis': [1.0,0,1.0], 'head':[1.0,1.6,1.0], 'forward':[1,0,-0.2]}, {'id': 'B', 'pelvis': [2.4,0,1.1], 'head':[2.4,1.6,1.1], 'forward':[-1,0,-0.1]}]},
            ],
            'blockers': [{'id':'partition','center':[1.5,1.6,0.0],'radius':0.45}], 'landmarks': []
        },
        {
            'scene_id': 'mh_demo_side_by_side', 'title': 'Multi-human demo · side-by-side around table', 'duration_sec': 15,
            'frames': [
                {'t': 0, 'people': [{'id': 'A', 'pelvis': [0,0,0], 'head':[0,1.6,0], 'forward':[0,0,1]}, {'id': 'B', 'pelvis': [0.7,0,0.1], 'head':[0.7,1.6,0.1], 'forward':[0,0,1]}]},
                {'t': 7.5, 'people': [{'id': 'A', 'pelvis': [0.2,0,1.0], 'head':[0.2,1.6,1.0], 'forward':[0.4,0,1]}, {'id': 'B', 'pelvis': [0.9,0,1.2], 'head':[0.9,1.6,1.2], 'forward':[0.4,0,1]}]},
                {'t': 15, 'people': [{'id': 'A', 'pelvis': [0.9,0,2.0], 'head':[0.9,1.6,2.0], 'forward':[1,0,0]}, {'id': 'B', 'pelvis': [1.5,0,2.2], 'head':[1.5,1.6,2.2], 'forward':[1,0,0]}]},
            ],
            'blockers': [], 'landmarks': [{'id':'table','center':[1.2,0,1.2]}]
        },
    ]


def write_svg(scene: dict[str, Any], out: Path) -> str:
    pts=[]
    for fr in scene['frames']:
        for p in fr['people']:
            pts.append((p['pelvis'][0], p['pelvis'][2]))
    for b in scene.get('blockers',[]): pts.append((b['center'][0], b['center'][2]))
    xs=[p[0] for p in pts]; zs=[p[1] for p in pts]
    minx,maxx=min(xs)-0.8,max(xs)+0.8; minz,maxz=min(zs)-0.8,max(zs)+0.8
    def xy(x,z):
        return ((x-minx)/(maxx-minx)*760+20, (1-(z-minz)/(maxz-minz))*420+20)
    colors={'A':'#2563eb','B':'#dc2626'}
    lines=['<svg xmlns="http://www.w3.org/2000/svg" width="820" height="480" viewBox="0 0 820 480">','<rect width="820" height="480" fill="#f8fafc"/>']
    for b in scene.get('blockers',[]):
        x,y=xy(b['center'][0], b['center'][2]); r=b.get('radius',0.4)/(maxx-minx)*760
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="#94a3b8" opacity="0.45"/><text x="{x+8:.1f}" y="{y:.1f}" font-size="14">{b["id"]}</text>')
    for pid in ['A','B']:
        path=[]
        for fr in scene['frames']:
            p=next(x for x in fr['people'] if x['id']==pid)
            x,y=xy(p['pelvis'][0],p['pelvis'][2]); path.append((x,y,p))
        lines.append('<polyline points="'+' '.join(f'{x:.1f},{y:.1f}' for x,y,_ in path)+f'" fill="none" stroke="{colors[pid]}" stroke-width="4"/>')
        for idx,(x,y,p) in enumerate(path):
            lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" fill="{colors[pid]}"/><text x="{x+12:.1f}" y="{y-8:.1f}" font-size="16" font-weight="700">{pid} t{idx}</text>')
            f=p['forward']; x2,y2=xy(p['pelvis'][0]+0.35*f[0], p['pelvis'][2]+0.35*f[2])
            lines.append(f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{colors[pid]}" stroke-width="3" marker-end="url(#arrow)"/>')
    lines.insert(1,'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L0,6 L6,3 z" fill="#334155"/></marker></defs>')
    lines.append(f'<text x="20" y="465" font-size="16" fill="#334155">{scene["title"]}</text></svg>')
    out.write_text('\n'.join(lines))
    return './'+str(out.relative_to(ROOT/'site/qa_benchmark'))


def load_scene_file(path: Path) -> list[dict[str, Any]]:
    raw=json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get('scenes'), list):
        return raw['scenes']
    if isinstance(raw, dict) and raw.get('frames'):
        return [raw]
    return []

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-jsonl', type=Path, default=Path('outputs/qa/multihuman_dynamic_qa.jsonl'))
    ap.add_argument('--site-data', type=Path, default=Path('site/qa_benchmark/data.js'))
    ap.add_argument('--scenes-json', type=Path, help='Canonical/converted multihuman scenes JSON, e.g. outputs/qa/hoim3_multihuman_scenes.json')
    ap.add_argument('--source-label', default='multihuman adapter demo scene')
    ap.add_argument('--replace-prefix', default='mh_demo_')
    args=ap.parse_args()
    media_dir=ROOT/'site/qa_benchmark/multihuman_media'; media_dir.mkdir(parents=True, exist_ok=True)
    scenes = load_scene_file(ROOT/args.scenes_json if args.scenes_json and not args.scenes_json.is_absolute() else args.scenes_json) if args.scenes_json else demo_scenes()
    groups=[]; rows=[]
    for scene in scenes:
        qas=multihuman_qas(scene)
        img=write_svg(scene, media_dir/f"{scene['scene_id']}.svg")
        group={'name':scene['scene_id'],'title':scene.get('title', scene['scene_id']),'original_image':img,'topdown_image':img,'summary_path':str(args.scenes_json or 'examples/multihuman_demo_scenes.json'),'raw_summary_path':str(args.scenes_json or 'examples/multihuman_demo_scenes.json'),'dynamic_timeline':{'duration_sec':scene.get('duration_sec'),'frames':[f.get('t') for f in scene['frames']]},'qa':qas,'video_clip':scene.get('video_clip'),'original_video':('./hoim3_data/'+scene.get('source_video','').split('data/HOI-M3/',1)[1]) if scene.get('source_video','').startswith('data/HOI-M3/') else None,'video_window':{'duration_sec':scene.get('duration_sec'),'source':scene.get('source_video') or args.source_label}}
        groups.append(group)
        for q in qas:
            rows.append({'scene_id':scene['scene_id'],**q})
    out=ROOT/args.output_jsonl; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
    if not args.scenes_json:
        (ROOT/'examples/multihuman_demo_scenes.json').write_text(json.dumps(demo_scenes(),ensure_ascii=False,indent=2))
    # append or replace multihuman demo groups in site data
    import re
    p=ROOT/args.site_data
    text=p.read_text()
    data=json.loads(re.search(r'window\.QA_DATA\s*=\s*(.*);\s*$', text, re.S).group(1))
    data['groups']=[g for g in data['groups'] if not str(g.get('name','')).startswith(args.replace_prefix)]
    data['groups'].extend(groups)
    p.write_text('window.QA_DATA='+json.dumps(data,ensure_ascii=False,separators=(',',':'))+';')
    print('multihuman groups',len(groups),'qa',len(rows))

if __name__ == '__main__':
    main()
