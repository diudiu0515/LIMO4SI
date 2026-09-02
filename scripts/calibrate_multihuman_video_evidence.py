#!/usr/bin/env python3
"""Audit visible people in HOI-M3 clips and build video-grounded evidence.

The metric QA layer may contain fewer SMPL-X tracks than people visible in a
camera view.  This script keeps those two facts separate:

* Grounding DINO + lightweight temporal association audits every visible person.
* SMPL-X remains the only source for metric 3D distance/orientation answers.
* If visible-person coverage exceeds 3D coverage, ambiguous A/B metric QA is
  replaced by conservative image-plane multi-human dynamics QA.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'site/qa_benchmark'
COLORS = [(37, 99, 235), (220, 38, 38), (22, 163, 74), (217, 119, 6), (147, 51, 234), (8, 145, 178)]


def load_site(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    match = re.search(r'window\.QA_DATA\s*=\s*(.*);\s*$', text, re.S)
    if not match:
        raise ValueError(f'Cannot parse {path}')
    return json.loads(match.group(1))


def save_site(path: Path, data: dict[str, Any]) -> None:
    path.write_text('window.QA_DATA = ' + json.dumps(data, ensure_ascii=False, indent=2) + ';\n', encoding='utf-8')


def site_path(value: str) -> Path:
    return SITE / value[2:] if value.startswith('./') else ROOT / value


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = np.maximum(a[:2], b[:2]); x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(0.0, float(x2-x1)) * max(0.0, float(y2-y1))
    aa = max(0.0, float(a[2]-a[0])) * max(0.0, float(a[3]-a[1]))
    bb = max(0.0, float(b[2]-b[0])) * max(0.0, float(b[3]-b[1]))
    return inter / max(1e-6, aa + bb - inter)


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float = 0.55) -> list[int]:
    order = list(np.argsort(-scores))
    keep: list[int] = []
    while order:
        idx = order.pop(0); keep.append(idx)
        order = [j for j in order if iou(boxes[idx], boxes[j]) < threshold]
    return keep


def appearance_hist(frame: np.ndarray, box: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(x)) for x in box]
    x1=max(0,min(w-1,x1)); x2=max(x1+1,min(w,x2)); y1=max(0,min(h-1,y1)); y2=max(y1+1,min(h,y2))
    crop = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [18, 16], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0]+box[2])/2.0, (box[1]+box[3])/2.0], dtype=np.float32)


def detect_clip(video: Path, processor: Any, model: Any, device: str, sample_fps: float, box_threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS)); frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps
    times = list(np.arange(0.0, max(0.01, duration - 0.05), 1.0/sample_fps))
    if not times or duration - 0.08 - times[-1] > 0.2:
        times.append(max(0.0, duration - 0.08))
    frames: list[np.ndarray] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t*1000.0)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        else:
            frames.append(np.zeros((height, width, 3), np.uint8))
    cap.release()

    detections: list[dict[str, Any]] = []
    for start in range(0, len(frames), 8):
        batch = frames[start:start+8]
        images = [Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)) for x in batch]
        inputs = processor(images=images, text=['person.']*len(images), return_tensors='pt').to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=box_threshold, text_threshold=0.20,
            target_sizes=[(x.shape[0], x.shape[1]) for x in batch],
        )
        for local, (frame, result) in enumerate(zip(batch, results)):
            boxes = result['boxes'].detach().cpu().numpy().astype(np.float32)
            scores = result['scores'].detach().cpu().numpy().astype(np.float32)
            good=[]
            for j in nms(boxes, scores):
                b=boxes[j]
                area=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
                if area < width*height*0.003:
                    continue
                good.append({'box': b, 'score': float(scores[j]), 'hist': appearance_hist(frame,b)})
            detections.append({'t': float(times[start+local]), 'frame': frame, 'detections': good})
    return detections, {'fps':fps,'frame_count':frame_count,'width':width,'height':height,'duration_sec':duration}


def associate(detected: list[dict[str, Any]], width: int, height: int) -> list[dict[str, Any]]:
    tracks: list[dict[str, Any]]=[]; diag=math.hypot(width,height)
    for row in detected:
        t=float(row['t']); dets=row['detections']
        active=[tr for tr in tracks if t-tr['obs'][-1]['t'] <= 1.6]
        assigned_t=set(); assigned_d=set()
        if active and dets:
            costs=np.full((len(active),len(dets)),10.0,dtype=np.float32)
            for i,tr in enumerate(active):
                last=tr['obs'][-1]; pred=center(last['box'])
                if len(tr['obs'])>=2:
                    prev=tr['obs'][-2]; dt=max(1e-3,last['t']-prev['t'])
                    pred=pred+(center(last['box'])-center(prev['box']))*((t-last['t'])/dt)
                for j,d in enumerate(dets):
                    cd=float(np.linalg.norm(pred-center(d['box']))/diag)
                    ov=iou(last['box'],d['box'])
                    hd=float(cv2.compareHist(tr['hist'].astype(np.float32),d['hist'].astype(np.float32),cv2.HISTCMP_BHATTACHARYYA))
                    costs[i,j]=0.52*cd+0.28*(1.0-ov)+0.20*hd
            rr,cc=linear_sum_assignment(costs)
            for i,j in zip(rr.tolist(),cc.tolist()):
                tr=active[i]; d=dets[j]; cd=float(np.linalg.norm(center(tr['obs'][-1]['box'])-center(d['box']))/diag)
                if costs[i,j] > 0.62 or (cd>0.28 and iou(tr['obs'][-1]['box'],d['box'])<0.02):
                    continue
                tr['obs'].append({'t':t,'box':d['box'],'score':d['score']})
                tr['hist']=0.82*tr['hist']+0.18*d['hist']; assigned_t.add(id(tr)); assigned_d.add(j)
        for j,d in enumerate(dets):
            if j not in assigned_d:
                tracks.append({'raw_id':len(tracks)+1,'obs':[{'t':t,'box':d['box'],'score':d['score']}],'hist':d['hist']})
    min_hits=max(4,int(round(len(detected)*0.25)))
    reliable=[tr for tr in tracks if len(tr['obs'])>=min_hits and tr['obs'][-1]['t']-tr['obs'][0]['t']>=detected[-1]['t']*0.35]
    reliable.sort(key=lambda tr:(tr['obs'][0]['t'],float(center(tr['obs'][0]['box'])[0])))
    for i,tr in enumerate(reliable,1): tr['id']=f'V{i}'
    return reliable


def box_at(track: dict[str, Any], t: float, max_gap: float = 1.25) -> np.ndarray | None:
    obs=track['obs']
    before=[x for x in obs if x['t']<=t]; after=[x for x in obs if x['t']>=t]
    if before and after:
        a=before[-1]; b=after[0]
        if b['t']-a['t']<=max_gap:
            if abs(b['t']-a['t'])<1e-6: return a['box']
            u=(t-a['t'])/(b['t']-a['t']); return (1-u)*a['box']+u*b['box']
    nearest=min(obs,key=lambda x:abs(x['t']-t))
    return nearest['box'] if abs(nearest['t']-t)<=max_gap/2 else None


def draw_person(frame: np.ndarray, box: np.ndarray, label: str, color_rgb: tuple[int,int,int]) -> None:
    color=(color_rgb[2],color_rgb[1],color_rgb[0]); x1,y1,x2,y2=[int(round(x)) for x in box]
    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3,cv2.LINE_AA)
    head=(int((x1+x2)/2),int(y1+0.14*(y2-y1))); pelvis=(int((x1+x2)/2),int(y1+0.72*(y2-y1)))
    cv2.circle(frame,head,6,(255,255,255),-1,cv2.LINE_AA); cv2.circle(frame,head,4,color,-1,cv2.LINE_AA)
    cv2.circle(frame,pelvis,7,(255,255,255),-1,cv2.LINE_AA); cv2.circle(frame,pelvis,5,color,-1,cv2.LINE_AA)
    cv2.line(frame,head,pelvis,color,2,cv2.LINE_AA)
    txt=f'{label} · visible 2D track'; (tw,th),_=cv2.getTextSize(txt,cv2.FONT_HERSHEY_SIMPLEX,.62,2)
    ty=max(th+8,y1); cv2.rectangle(frame,(x1,ty-th-8),(x1+tw+8,ty+3),color,-1)
    cv2.putText(frame,txt,(x1+4,ty-3),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2,cv2.LINE_AA)


def render_evidence(video: Path, tracks: list[dict[str, Any]], meta: dict[str, Any], out_video: Path, out_image: Path) -> None:
    out_video.parent.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video)); fps=float(meta['fps']); w=int(meta['width']); h=int(meta['height'])
    with tempfile.TemporaryDirectory(prefix='limo_mh_') as td:
        raw=Path(td)/'localized.mp4'; writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
        idx=0; selected=[]; sheet_times=np.linspace(0,max(0.0,float(meta['duration_sec'])-.1),6)
        while True:
            ok,frame=cap.read()
            if not ok: break
            t=idx/fps
            for k,tr in enumerate(tracks):
                b=box_at(tr,t)
                if b is not None: draw_person(frame,b,tr['id'],COLORS[k%len(COLORS)])
            cv2.putText(frame,f'Visible-person audit: {len(tracks)} persistent tracks | metric 3D coverage checked separately',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.67,(20,20,20),4,cv2.LINE_AA)
            cv2.putText(frame,f'Visible-person audit: {len(tracks)} persistent tracks | metric 3D coverage checked separately',(18,30),cv2.FONT_HERSHEY_SIMPLEX,.67,(255,255,255),2,cv2.LINE_AA)
            writer.write(frame)
            if any(abs(t-x)<=0.5/fps for x in sheet_times): selected.append((t,frame.copy()))
            idx+=1
        cap.release(); writer.release()
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(raw),'-an','-c:v','libx264','-preset','veryfast','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(out_video)],check=True)
    if selected:
        thumbs=[]
        for t,frame in selected[:6]:
            thumb=cv2.resize(frame,(640,int(round(h*640/w))))
            cv2.putText(thumb,f't={t:.1f}s',(14,thumb.shape[0]-14),cv2.FONT_HERSHEY_SIMPLEX,.72,(0,0,0),4,cv2.LINE_AA)
            cv2.putText(thumb,f't={t:.1f}s',(14,thumb.shape[0]-14),cv2.FONT_HERSHEY_SIMPLEX,.72,(255,255,255),2,cv2.LINE_AA)
            thumbs.append(thumb)
        while len(thumbs)<6: thumbs.append(thumbs[-1].copy())
        out_image.parent.mkdir(parents=True,exist_ok=True); cv2.imwrite(str(out_image),np.vstack([np.hstack(thumbs[:3]),np.hstack(thumbs[3:6])]),[cv2.IMWRITE_JPEG_QUALITY,91])


def compact_track(track: dict[str, Any], duration: float, diag: float) -> dict[str, Any]:
    obs=track['obs']; centers=[center(x['box']) for x in obs]
    path=sum(float(np.linalg.norm(b-a)) for a,b in zip(centers,centers[1:]))/diag
    displacement=float(np.linalg.norm(centers[-1]-centers[0]))/diag
    raw_coverage = len(obs) / max(1, round(duration / (obs[1]['t'] - obs[0]['t'])) if len(obs) > 1 else len(obs))
    return {'id':track['id'],'first_seen_sec':round(obs[0]['t'],3),'last_seen_sec':round(obs[-1]['t'],3),'coverage':round(min(1.0,raw_coverage),3),'observations':len(obs),'mean_score':round(float(np.mean([x['score'] for x in obs])),3),'normalized_path_length':round(path,4),'normalized_displacement':round(displacement,4)}


def nearest_box(track: dict[str, Any], t: float) -> np.ndarray:
    return min(track['obs'],key=lambda x:abs(x['t']-t))['box']


def mcq(correct: str, distractors: list[str], seed: str) -> tuple[list[dict[str,str]],str]:
    vals=[]
    for x in [correct]+distractors:
        if x not in vals: vals.append(x)
    while len(vals)<4: vals.append('The tracked evidence does not support this alternative.')
    vals=vals[:4]; offset=sum(map(ord,seed+correct))%4; ordered=vals[1:]; ordered.insert(offset,correct); labels=list('ABCD')
    return [{'label':a,'text':b} for a,b in zip(labels,ordered)],labels[offset]


def metric_identity_alignment(group: dict[str, Any], tracks: list[dict[str, Any]], diag: float) -> dict[str, Any]:
    """Associate two visible tracks with metric A/B by temporal motion profile.

    This does not pretend to be camera calibration.  It is only accepted when
    the correct assignment has a clearly lower error than the swapped one.
    """
    states=[]
    for q in group.get('qa',[]):
        states=((q.get('result_json') or {}).get('pair_timeline') or {}).get('states') or []
        if states: break
    if len(states)<3 or len(tracks)!=2:
        return {'status':'unavailable','reason':'requires exactly two visible and two metric tracks'}
    times=[float(x.get('t') or 0) for x in states]
    def profile(points: list[np.ndarray]) -> list[float]:
        seg=[float(np.linalg.norm(b-a)) for a,b in zip(points,points[1:])]
        total=sum(seg)
        return [x/max(total,1e-6) for x in seg]+[total]
    metric={}
    for pid,key in [('A','person_a'),('B','person_b')]:
        pts=[np.asarray((x.get('evidence') or {}).get(key,{}).get('pelvis_xyz_m'),np.float32) for x in states]
        metric[pid]=profile(pts)
    visual={}
    for tr in tracks:
        pts=[center(nearest_box(tr,t)) for t in times]
        row=profile(pts); row[-1]/=diag; visual[tr['id']]=row
    def shape_cost(a: list[float],b: list[float]) -> float:
        return sum(abs(x-y) for x,y in zip(a[:-1],b[:-1]))
    ids=[tracks[0]['id'],tracks[1]['id']]
    direct=shape_cost(metric['A'],visual[ids[0]])+shape_cost(metric['B'],visual[ids[1]])
    swapped=shape_cost(metric['A'],visual[ids[1]])+shape_cost(metric['B'],visual[ids[0]])
    best=min(direct,swapped); alternative=max(direct,swapped); margin=alternative-best
    mapping={'A':ids[0],'B':ids[1]} if direct<=swapped else {'A':ids[1],'B':ids[0]}
    reliable=margin>=0.18
    return {'status':'aligned' if reliable else 'unresolved','mapping':mapping if reliable else None,'best_cost':round(best,4),'alternative_cost':round(alternative,4),'margin':round(margin,4),'metric_motion_profiles':metric,'visual_motion_profiles':visual,'criterion':'all available temporal segment-motion profiles; accept only when assignment margin >= 0.18'}


def mismatch_qas(scene_id: str, tracks: list[dict[str, Any]], audit: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    diag=math.hypot(width,height); compact=audit['visible_2d_tracks']; far=max(compact,key=lambda x:x['normalized_path_length'])
    correct=f"{far['id']} has the largest image-plane path length ({far['normalized_path_length']:.3f} of the frame diagonal)."
    opts,lab=mcq(correct,[f"{x['id']} has the largest image-plane path length." for x in compact if x['id']!=far['id']]+['All visible tracks move by the same amount.','None of the labeled people is tracked for enough of the clip to compare path length.'],'motion'+scene_id)
    common={'task_id':'task4_multi_human_relational_dynamics','task_name':'Task 4 · Multi-Human Relational Dynamics','status':'ok','method':'Grounding DINO detects every visible person at 2 Hz; temporal association uses motion, overlap, and appearance. This answer is camera-plane topology, not a metric 3D claim.'}
    rows=[{**common,'question_type':'visible_human_motion_ranking_2d','question':'Across the entire clip, which labeled visible person follows the longest path in the camera image?','options':opts,'correct_option':lab,'correct_answer':correct,'answer':correct,'explanation':correct,'result_json':{'scene_id':scene_id,'answer_type':'visible_human_motion_ranking_2d','visible_2d_tracks':compact,'visual_person_audit':audit,'T_Q':True,'H_Q':True,'S_Q':True,'approximations':['2D image-plane person tracking; not metric 3D identity']}}]
    if len(tracks)>=3:
        pairs=[]
        for i in range(len(tracks)):
            for j in range(i+1,len(tracks)):
                a,b=tracks[i],tracks[j]
                ds=float(np.linalg.norm(center(nearest_box(a,0))-center(nearest_box(b,0)))/diag)
                de=float(np.linalg.norm(center(nearest_box(a,audit['duration_sec']))-center(nearest_box(b,audit['duration_sec'])))/diag)
                pairs.append((a['id']+'–'+b['id'],ds,de))
        sp=min(pairs,key=lambda x:x[1]); ep=min(pairs,key=lambda x:x[2])
        correct=f'The closest pair changes from {sp[0]} at the start to {ep[0]} at the end.' if sp[0]!=ep[0] else f'{sp[0]} is the closest visible pair at both the start and the end.'
        distract=[f'{x[0]} is the closest visible pair at both endpoints.' for x in pairs if x[0] not in {sp[0],ep[0]}]
        distract += [f'The order is reversed: {ep[0]} at the start and {sp[0]} at the end.','No visible pair can be compared over time.']
        opts,lab=mcq(correct,distract,'closest'+scene_id)
        rows.append({**common,'question_type':'closest_visible_pair_change_2d','question':'Using the labeled localization tracks, how does the closest visible pair in the camera plane change from the start to the end?','options':opts,'correct_option':lab,'correct_answer':correct,'answer':correct,'explanation':correct,'result_json':{'scene_id':scene_id,'answer_type':'closest_visible_pair_change_2d','start_pair_distances_normalized':[{'pair':p,'distance':round(a,4)} for p,a,_ in pairs],'end_pair_distances_normalized':[{'pair':p,'distance':round(b,4)} for p,_,b in pairs],'visual_person_audit':audit,'T_Q':True,'H_Q':True,'S_Q':True,'approximations':['2D image-plane person tracking; not metric 3D identity']}})
    return rows


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--site-data',type=Path,default=Path('site/qa_benchmark/data.js'))
    ap.add_argument('--model',type=Path,default=Path('models/grounding-dino-tiny'))
    ap.add_argument('--audit-output',type=Path,default=Path('outputs/qa/multihuman_visual_calibration.json'))
    ap.add_argument('--sample-fps',type=float,default=2.0)
    ap.add_argument('--box-threshold',type=float,default=0.30)
    args=ap.parse_args()
    site_data=args.site_data if args.site_data.is_absolute() else ROOT/args.site_data
    model_path=args.model if args.model.is_absolute() else ROOT/args.model
    device='cuda' if torch.cuda.is_available() else 'cpu'
    processor=AutoProcessor.from_pretrained(str(model_path),local_files_only=True)
    model=AutoModelForZeroShotObjectDetection.from_pretrained(str(model_path),local_files_only=True).to(device).eval()
    data=load_site(site_data); audits=[]
    for group in data.get('groups',[]):
        if not str(group.get('name','')).startswith('hoi_m3_') or not group.get('video_clip'): continue
        video=site_path(group['video_clip'])
        if not video.exists(): continue
        detected,meta=detect_clip(video,processor,model,device,args.sample_fps,args.box_threshold)
        tracks=associate(detected,meta['width'],meta['height']); counts=[len(x['detections']) for x in detected]
        tracked_3d=0
        for q in group.get('qa',[]):
            states=((q.get('result_json') or {}).get('pair_timeline') or {}).get('states') or []
            if states: tracked_3d=max(tracked_3d,len((states[0].get('evidence') or {}).get('person_a',{}))>0 and 2)
        if not tracked_3d:
            title=str(group.get('title','')); m=re.search(r'(\d+) metric 3D tracks',title); tracked_3d=int(m.group(1)) if m else 2
        persistent=len(tracks); mismatch=persistent>tracked_3d; diag=math.hypot(meta['width'],meta['height'])
        identity=metric_identity_alignment(group,tracks,diag) if not mismatch else {'status':'unavailable','reason':'visible count exceeds metric track count'}
        if identity.get('status')=='aligned':
            for metric_id,visible_id in identity['mapping'].items():
                next(x for x in tracks if x['id']==visible_id)['id']=metric_id
        status='coverage_mismatch' if mismatch else ('complete_and_identity_aligned' if identity.get('status')=='aligned' else 'identity_unresolved')
        audit={'scene_id':group['name'],'status':status,'detector':'IDEA Research Grounding DINO tiny (local weights)','tracking':'motion + box overlap + HSV appearance Hungarian association','sample_fps':args.sample_fps,'box_threshold':args.box_threshold,'duration_sec':round(meta['duration_sec'],3),'sample_count':len(detected),'sampled_visible_counts':counts,'mode_visible_person_count':Counter(counts).most_common(1)[0][0] if counts else 0,'max_visible_person_count':max(counts) if counts else 0,'persistent_visible_person_count':persistent,'metric_3d_track_count':tracked_3d,'metric_identity_alignment':identity,'geometry_scope':'all visible people have 2D localization; metric distance/orientation applies only to visually aligned SMPL-X tracks','visible_2d_tracks':[compact_track(x,meta['duration_sec'],diag) for x in tracks]}
        out_dir=SITE/'multihuman_media'; out_video=out_dir/f"{group['name']}_localized.mp4"; out_image=out_dir/f"{group['name']}_localized.jpg"
        render_evidence(video,tracks,meta,out_video,out_image)
        group['localization_video']='./'+str(out_video.relative_to(SITE)); group['localization_image']='./'+str(out_image.relative_to(SITE)); group['original_image']=group['localization_image']; group['visual_person_audit']=audit
        seq=re.sub(r'^hoi_m3_','',group['name']).rsplit('_win',1)[0]; win=group['name'].rsplit('_win',1)[-1]
        group['title']=f"HOI-M3 · {seq} · window {win} · {persistent} visible 2D tracks / {tracked_3d} metric 3D tracks"
        for q in group.get('qa',[]):
            (q.setdefault('result_json',{}))['visual_person_audit']=audit
        if mismatch or identity.get('status')!='aligned':
            group['metric_topdown_image']=group.get('topdown_image')
            group['topdown_image']=None
            group['qa']=mismatch_qas(group['name'],tracks,audit,meta['width'],meta['height'])
        audits.append(audit); print(group['name'],audit['status'],persistent,tracked_3d,flush=True)
    data['multihuman_visual_calibration']={'status':'applied','audited_groups':len(audits),'coverage_mismatches':sum(x['status']=='coverage_mismatch' for x in audits),'identity_unresolved':sum(x['status']=='identity_unresolved' for x in audits),'policy':'2D localization covers every persistent visible person; metric 3D QA is never extended to unannotated people.'}
    save_site(site_data,data)
    out=args.audit_output if args.audit_output.is_absolute() else ROOT/args.audit_output; out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps({'status':'ok','groups':audits},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(data['multihuman_visual_calibration'],ensure_ascii=False))


if __name__=='__main__': main()
