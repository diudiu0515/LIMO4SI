#!/usr/bin/env python3
"""Build reusable Task 1 / Task 3 QA site data from spatial summary files.

This script is the batch-generation entry point for the current website format.
It consumes existing spatial `summary.json` files, computes structured results
with `limo4si.perspective_qa`, writes natural-language QA, and preserves optional
computed/raw JSON for audit.
"""
from __future__ import annotations

import argparse, json, re, sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))

from build_perspective_qa import find_body_pose, find_camera_calibration  # noqa: E402
from limo4si.perspective_qa import (  # noqa: E402
    human_centric_answer, spatial_relation_quality_answer, visibility_answer, nearest_reachable_object, nearest_object_analysis, static_reachability_answer, reach_for_intent, unified_reach_analysis,
    level2_occlusion_answer, reference_frame_switching_answer,
)

DEFAULT_SUMMARIES = [
 'outputs/spatial/showcase_multi/sfu0083_cam04_3450/summary.json',
 'outputs/spatial/showcase_queries/query_01_iiith32_frame5280/summary.json',
 'outputs/spatial/showcase_queries/query_02_iiith30_frame4440/summary.json',
 'outputs/spatial/showcase_queries/query_03_replacement_iiith30_frame4440_all/summary.json',
 'outputs/spatial/showcase_queries/query_04_iiith145_frame11250/summary.json',
 'outputs/spatial/showcase_queries/query_05_iiith31_frame4170/summary.json',
 'outputs/spatial/showcase_diverse_new/diverse_sfu_010_3_frame6750/summary.json',
 'outputs/spatial/showcase_diverse_new/diverse_sfu0103_extra_frame2730/summary.json',
 'outputs/spatial/showcase_diverse_new/diverse_sfu0103_extra_frame3390/summary.json',
 'outputs/spatial/showcase_diverse_new/diverse_sfu0103_extra_frame4410/summary.json',
 'outputs/spatial/showcase_diverse_new/diverse_sfu0103_extra_frame6630/summary.json',
 'outputs/spatial/showcase_diverse_new/diverse_sfu_008_3_frame6960/summary.json',
]
def label(sample: dict) -> str:
    return str(sample.get('object_id','object')).replace('_0','').replace('_1','')

def rel_words(rel: dict) -> str:
    vals=[]
    for k in ['lateral_relation','longitudinal_relation','vertical_relation']:
        v=rel.get(k)
        if v and v not in ['same_lateral_position','same_longitudinal_position','same_height']:
            vals.append(str(v).replace('_',' '))
    return ' and '.join(vals) or 'roughly aligned with the person'

def ok_samples(summary: dict) -> list[dict]:
    samples=summary.get('samples',[])
    return [s for s in samples if s.get('object_xyz_world_m') and s.get('status')=='ok'] or [s for s in samples if s.get('object_xyz_world_m')]

def nearest_by_pelvis(samples: list[dict]) -> dict:
    return min(samples, key=lambda s: float(s.get('distance_m', 1e9)))

def add(qas, task_id, task_name, qtype, question, answer, method, result, raw):
    qas.append({
        'task_id': task_id,
        'task_name': task_name,
        'question_type': qtype,
        'question': question,
        'answer': answer,
        'status': result.get('status','ok') if isinstance(result, dict) else 'ok',
        'method': method,
        'result_json': result,
        'raw_json': raw,
    })



def hand_pose_at_frame(root: Path, sample: dict) -> dict | None:
    """Load 3D hand/finger pose at the key frame when EgoPose hand data exists."""
    hand_path = root/'data/egoexo4d/annotations/ego_pose/val/hand/automatic'/f"{sample['take_uid']}.json"
    if not hand_path.exists():
        return None
    hand = json.loads(hand_path.read_text(encoding='utf-8'))
    row = hand.get(str(sample['frame']))
    if row and row[0].get('annotation3D'):
        return row[0]['annotation3D']
    return None

def temporal_pose_sequence(root: Path, sample: dict, *, half_window_frames: int = 45, stride_frames: int = 5) -> list[dict]:
    """Load 3D body poses around the key frame for reach-for intent."""
    body_path = Path(sample.get('inputs', {}).get('body_pose', ''))
    if not body_path.exists():
        body_path = root/'data/egoexo4d/annotations/ego_pose/val/body/automatic'/f"{sample['take_uid']}.json"
    body = json.loads(body_path.read_text(encoding='utf-8'))
    center = int(sample['frame'])
    poses = []
    for frame in range(center - half_window_frames, center + half_window_frames + 1, stride_frames):
        row = body.get(str(frame))
        if row and row[0].get('annotation3D'):
            poses.append(row[0]['annotation3D'])
    if not poses and body.get(str(center)):
        poses.append(body[str(center)][0]['annotation3D'])
    return poses

def build_clip(root: Path, group: dict, seconds: float) -> None:
    first=group['raw_summary']['samples'][0]
    video=root/'data/egoexo4d/takes'/first['take_name']/'frame_aligned_videos/downscaled/448'/f"{first['camera']}.mp4"
    out=(root/group['summary_path']).parent/'showcase_clip_3s.mp4'
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','default=noprint_wrappers=1:nokey=1',str(video)],capture_output=True,text=True,check=True).stdout.strip()
    fps=(lambda x: float(x[0])/float(x[1]) if len(x)==2 else float(x[0]))(probe.split('/'))
    start=max(0.0, float(first['frame'])/fps - seconds/2)
    group['video_clip']='./'+str(out.relative_to(root))
    group['video_window']={'center_frame':int(first['frame']),'fps':fps,'start_sec':round(start,3),'duration_sec':seconds,'source_video':str(video.relative_to(root))}
    if out.exists() and out.stat().st_size > 1000:
        return
    subprocess.run(['ffmpeg','-y','-ss',f'{start:.3f}','-i',str(video),'-t',str(seconds),'-vf','scale=720:-2','-an','-c:v','libx264','-preset','veryfast','-crf','24','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)

def build_group(root: Path, summary_path: Path, clip_seconds: float) -> dict:
    summary=json.loads(summary_path.read_text(encoding='utf-8'))
    samples=ok_samples(summary)
    if not samples: raise ValueError(f'No usable samples in {summary_path}')
    first=samples[0]; second=samples[1] if len(samples)>1 else first
    body=find_body_pose(root, first); joints=body['annotation3D']; cal=find_camera_calibration(root, first)
    nearest=nearest_by_pelvis(samples)
    reach=nearest_reachable_object(samples, joints)
    pose_seq=temporal_pose_sequence(root, first)
    hand_joints=hand_pose_at_frame(root, first)
    target_reach=static_reachability_answer(second, joints, hand_joints=hand_joints, candidates=samples)
    nearest_analysis=nearest_object_analysis(samples, joints, hand_joints=hand_joints)
    reach_intent=reach_for_intent(samples, pose_seq, current_joints=joints, hand_joints=hand_joints)
    vis=visibility_answer(second, joints, candidates=samples)
    lvl=level2_occlusion_answer(joints, first, samples)
    ref=reference_frame_switching_answer(first['object_xyz_world_m'], first['human_frame'], camera_intrinsics=cal.get('camera_intrinsics') if cal else None, camera_extrinsics=cal.get('camera_extrinsics') if cal else None)
    hc_first=human_centric_answer(first['object_xyz_world_m'], first['human_frame']); rel_first=hc_first['relation']; spatial_quality=spatial_relation_quality_answer(first)
    hc_second=human_centric_answer(second['object_xyz_world_m'], second['human_frame'])
    hc_near=human_centric_answer(nearest['object_xyz_world_m'], nearest['human_frame'])
    qas=[]
    add(qas,'task1','Task 1 · Human-Object Spatial Relation','quantitative_distance_and_direction',f"How far is the {label(first)} from the person, and where is it relative to the person?",f"The {label(first)} is {rel_words(rel_first)} relative to the person, about {rel_first['distance_m']:.2f} m from the pelvis center. Direction confidence: {spatial_quality['direction_confidence']}.",'3D object center is expressed in the person body frame; distance is pelvis-to-object Euclidean distance; relation quality audits near-distance and point-cloud validation.',spatial_quality,first)
    add(qas,'task1','Task 1 · Human-Object Spatial Relation','reachability',f"Can the person reach the {label(second)} from the current pose?",target_reach['answer'],'Static reachability uses the current 3D shoulder-elbow-wrist arm span and wrist-to-object distance. It answers can reach, not is reaching.',target_reach,{'target':second,'candidates':samples})
    add(qas,'task1','Task 1 · Human-Object Spatial Relation','visibility',f"Can the person see the {label(second)}?",vis['answer'],'Visibility uses head/face direction when available and checks listed-object sightline blockers.',vis,{'target':second,'candidates':samples})
    add(qas,'task1','Task 1 · Human-Object Spatial Relation','nearest_referring_object','Among the listed objects, which one is nearest to the person? Is it also easiest to reach?',nearest_analysis['answer'],'Nearest object is ranked by pelvis-to-object 3D Euclidean distance; easiest-to-reach is separately ranked by current arm/hand geometry.',nearest_analysis,{'candidates':samples})
    add(qas,'task1','Task 1 · Human-Object Spatial Relation','current_interaction_object','Which listed object is the person most likely reaching for right now?',reach_intent['answer'],'Reach-for intent uses short wrist trajectory plus current static reachability, hand/fingertip cue when available, and visibility gating.',reach_intent,{'candidates':samples})
    add(qas,'task3','Task 3 · Perspective-Grounded QA','person_perspective_left_right_front_back',f"From the person's own perspective, where is the {label(first)}?",hc_first['answer'],'Answer is in the human-centric frame, not image left/right.',hc_first,first)
    add(qas,'task3','Task 3 · Perspective-Grounded QA','perspective_visibility_occlusion',f"From the person's viewpoint, is the {label(second)} visible or blocked?",vis['answer'],'Uses observer head/body direction plus sightline blocker detection.',vis,{'target':second,'candidates':samples})
    add(qas,'task3','Task 3 · Perspective-Grounded QA','perspective_reachable_nearest',"From the person's recent hand motion, which listed object are they most likely reaching for?",reach_intent['answer'],'Uses temporal reach-for intent plus static reachability, hand/fingertip cue when available, and visibility gating.',reach_intent,{'candidates':samples})
    add(qas,'task3','Task 3 · Perspective-Grounded QA','level2_perspective_taking',f"From the observer's perspective, which listed object blocks the {label(first)}?",lvl['answer'],'Checks which candidate lies between observer and target along the sightline.',lvl,{'target':first,'candidates':samples})
    ans=ref['answer']; cam=ans.get('egocentric'); cam_txt='camera coordinates are available'
    if isinstance(cam,dict): cam_txt=f"in the camera frame its depth is {cam.get('forward_depth_m',0):.2f} m"
    add(qas,'task3','Task 3 · Perspective-Grounded QA','reference_frame_switching',f"Describe the {label(first)} using human-centric, camera-centric, and world frames.",f"Human-centric: {ans.get('human_centric')} Also, {cam_txt}; world coordinates are available in the optional JSON.",'Switches between human body frame, camera frame, and Ego-Exo4D world xyz.',ref,first)
    rel_dir=summary_path.parent.relative_to(root)
    group={'name':summary_path.parent.name,'title':f"{first.get('take_name')} · {first.get('camera')} · frame {first.get('frame')}",'original_image':'./'+str(rel_dir/'showcase_original_multi.jpg'),'topdown_image':'./'+str(rel_dir/'showcase_topdown_multi.jpg'),'summary_path':str(summary_path.relative_to(root)),'raw_summary':summary,'qa':qas}
    build_clip(root, group, clip_seconds)
    return group

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--summaries', nargs='*', default=DEFAULT_SUMMARIES)
    ap.add_argument('--output', type=Path, default=Path('site/qa_benchmark/data.js'))
    ap.add_argument('--clip-seconds', type=float, default=3.0)
    args=ap.parse_args()
    groups=[build_group(ROOT, ROOT/s, args.clip_seconds) for s in args.summaries]
    data={'title':'LIMO4SI Task 1 & Task 3 QA Examples','subtitle':'Human-object spatial relation and perspective-grounded QA from third-person video','source':'selected outputs/spatial summary.json files','tasks':[{'id':'task1','name':'Task 1 · Human-Object Spatial Relation','description':'Covers distance/direction, reachability, visibility, nearest referring object, and current interaction-object inference.'},{'id':'task3','name':'Task 3 · Perspective-Grounded QA','description':'Covers person-perspective direction, perspective visibility/occlusion, reachable-nearest, Level-2 perspective taking, and reference-frame switching.'}], 'groups':groups}
    out=ROOT/args.output if not args.output.is_absolute() else args.output
    out.write_text('window.QA_DATA = '+json.dumps(data,ensure_ascii=False,indent=2)+';\n',encoding='utf-8')
    print(out)
    print('groups', len(groups), 'qa', sum(len(g['qa']) for g in groups))
if __name__=='__main__': main()
