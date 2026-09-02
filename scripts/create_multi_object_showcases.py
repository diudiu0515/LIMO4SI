#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from ego4d.research.util.masks import decode_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from limo4si.distance_validation import validate_metric_distance
from limo4si.human_frame import build_human_frame, describe_relation
from limo4si.spatial_real import iter_semidense_points, project_world_points, robust_object_center, PointSelection
from limo4si.spatial_visuals import draw_pose_skeleton, draw_forward_axis

COLORS=[(30,110,240),(40,180,40),(210,90,210),(0,170,230)]
_ALIAS_PATH=ROOT/'configs/object_display_aliases.json'
DISPLAY_ALIASES=json.loads(_ALIAS_PATH.read_text(encoding='utf-8')) if _ALIAS_PATH.exists() else {}

def display_label(case_name, take, object_id):
    return DISPLAY_ALIASES.get(case_name, {}).get(object_id, object_id).replace('_0','').replace('_1','')

REL={'left':'left','right':'right','front':'front','behind':'behind','above':'above','below':'below','slightly_above':'slightly above','slightly_below':'slightly below','same_lateral_position':'center','same_longitudinal_position':'near origin','same_height':'same height'}


def relation_mask(record):
    """Decode an Ego-Exo4D Relations mask, with bbox fallback for pycocotools/numpy issues."""
    try:
        return decode_mask(record).astype(bool), "mask"
    except Exception:
        h, w = int(record.get('height', 0)), int(record.get('width', 0))
        mask = np.zeros((h, w), dtype=bool)
        clicks = record.get('intSegClicks', {}) or {}
        ul = clicks.get('upperLeft') or []
        br = clicks.get('bottomRight') or []
        if ul and br and h > 0 and w > 0:
            x1 = max(0, min(w - 1, int(round(float(ul[0].get('x', 0))))))
            y1 = max(0, min(h - 1, int(round(float(ul[0].get('y', 0))))))
            x2 = max(0, min(w - 1, int(round(float(br[0].get('x', 0))))))
            y2 = max(0, min(h - 1, int(round(float(br[0].get('y', 0))))))
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            mask[y1:y2 + 1, x1:x2 + 1] = True
        return mask, "bbox_fallback"


def xyz_dict(annotation):
    return {k.replace('-','_'):[float(v[a]) for a in 'xyz'] for k,v in annotation.items() if v and all(a in v for a in 'xyz')}


def read_frame(video, frame):
    cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame)); ok,img=cap.read(); cap.release()
    if not ok: raise RuntimeError(f'Cannot read {video} frame {frame}')
    return img


def select_from_cached(mask, xyz, depth, px, py, visible):
    h,w=mask.shape
    inside=visible & (px>=0) & (px<w) & (py>=0) & (py<h)
    idx=np.flatnonzero(inside)
    idx=idx[mask[py[idx], px[idx]]]
    return PointSelection(xyz[idx], depth[idx], int(len(xyz)), int(len(xyz)), int(visible.sum()))


def load_cloud_cached(cache: dict, path: Path) -> np.ndarray:
    key = str(path)
    if key not in cache:
        chunks = []
        for xyz, _, _ in iter_semidense_points(path, max_dist_std_m=0.10):
            if len(xyz):
                chunks.append(xyz)
        cache[key] = np.concatenate(chunks) if chunks else np.empty((0, 3), float)
    return cache[key]


def make_group(group, root: Path, output_root: Path, min_distance_m: float, dead_zone_m: float, cloud_cache: dict | None = None):
    uid=group['take_uid']; cam=group['camera']; frame=int(group['frame']); objects=group['objects']
    takes={r['take_uid']:r for r in json.loads((root/'takes.json').read_text())}
    take=takes[uid]['take_name']
    relations=json.loads((root/'annotations'/'relations_val.json').read_text())['annotations']
    camera_value = group.get('camera_pose_path')
    camera_path = Path(camera_value) if camera_value else root/'annotations'/'ego_pose'/'val'/'camera_pose'/f'{uid}.json'
    if not camera_path.exists():
        for c in [ROOT/'outputs'/'calibration'/'val_12'/f'{uid}.json', ROOT/'outputs'/'calibration'/'val_3'/f'{uid}.json']:
            if c.exists(): camera_path=c; break
    cal=json.loads(camera_path.read_text())[cam]
    body_path=root/'annotations'/'ego_pose'/'val'/'body'/'automatic'/f'{uid}.json'
    pose=json.loads(body_path.read_text())[str(frame)][0]
    frame3d=build_human_frame(xyz_dict(pose['annotation3D']))
    cloud=root/'takes'/take/'trajectory'/'semidense_points.csv.gz'
    xyz = load_cloud_cached(cloud_cache if cloud_cache is not None else {}, cloud)
    pixels,depth=project_world_points(xyz, cal['camera_intrinsics'], cal['camera_extrinsics'])
    finite=np.isfinite(pixels).all(axis=1) & (depth>0)
    px=np.rint(pixels[:,0]).astype(np.int64, casting='unsafe')
    py=np.rint(pixels[:,1]).astype(np.int64, casting='unsafe')
    out=output_root/group['name']; out.mkdir(parents=True, exist_ok=True)
    results=[]
    queries = group.get('queries', [])
    for obj_index, obj in enumerate(objects):
        mask_record=relations[uid]['object_masks'][obj][cam]['annotation'][str(frame)]
        mask, mask_source = relation_mask(mask_record)
        sel=select_from_cached(mask, xyz, depth, px, py, finite)
        center,inliers=robust_object_center(sel)
        rel=describe_relation(frame3d.world_to_human(center), dead_zone_m=dead_zone_m)
        check=validate_metric_distance(center, frame3d.to_dict(), rel['human_xyz_m'], pose['annotation3D'])
        eligible=rel['distance_m']>=min_distance_m and check['validated']
        raw=dict(rel)
        if not eligible:
            rel['lateral_relation']=rel['longitudinal_relation']=rel['vertical_relation']=rel['text_zh']=None
        row={'status':'ok' if eligible else 'filtered_near_or_invalid','recognition_status':'eligible' if eligible else 'filtered_near_or_invalid','take_uid':uid,'take_name':take,'camera':cam,'frame':frame,'object_id':obj,'query': queries[obj_index] if obj_index < len(queries) else f'Where is {obj} relative to the person?','object_xyz_world_m':center.tolist(),'human_frame':frame3d.to_dict(),'human_xyz_m':rel['human_xyz_m'],'distance_m':rel['distance_m'],'horizontal_distance_m':rel['horizontal_distance_m'],'lateral_relation':rel['lateral_relation'],'longitudinal_relation':rel['longitudinal_relation'],'vertical_relation':rel['vertical_relation'],'text_zh':rel['text_zh'],'raw_relation_before_filter':raw,'distance_validation':check,'quality':{'mask_pixels':int(mask.sum()),'mask_source':mask_source,'points_in_mask':int(len(sel.xyz_world)),'robust_inliers':int(len(inliers))},'inputs':{'point_cloud':str(cloud),'camera_pose':str(camera_path),'body_pose':str(body_path)}}
        (out/f"{take}_frame{frame}_{obj}.json").write_text(json.dumps(row,ensure_ascii=False,indent=2)+'\n')
        results.append(row)
    summary={'sample_count':len(results),'success_count':sum(r['status']=='ok' for r in results),'filtered_count':sum(r['status']!='ok' for r in results),'min_distance_m':min_distance_m,'distance_definition':'Euclidean distance from pelvis midpoint to robust 3D object centroid in Ego-Exo4D world meters.','samples':results}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    draw_images(root,out,relations,cal,pose,summary)
    print(out)


def draw_images(root,out,relations,cal,pose,summary):
    results=summary['samples']; r0=results[0]; uid=r0['take_uid']; take=r0['take_name']; cam=r0['camera']; frame=r0['frame']
    video=root/'takes'/take/'frame_aligned_videos'/'downscaled'/'448'/f'{cam}.mp4'
    img=read_frame(video,frame); h,w=img.shape[:2]
    ann_w=int(round(2*cal['camera_intrinsics'][0][2])); ann_h=int(round(2*cal['camera_intrinsics'][1][2]))
    canvas=img.copy(); legend=[]
    for i,r in enumerate(results,1):
        color=COLORS[(i-1)%len(COLORS)]; obj=r['object_id']; short=display_label(out.name, take, obj)
        mask_record=relations[uid]['object_masks'][obj][cam]['annotation'][str(frame)]
        mask, _mask_source = relation_mask(mask_record)
        sm=cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)
        overlay=canvas.copy(); overlay[sm]=(np.array(color)*0.75+overlay[sm].astype(np.float32)*0.25).astype(np.uint8)
        canvas=cv2.addWeighted(canvas,0.72,overlay,0.28,0)
        ys,xs=np.where(sm)
        if len(xs):
            x1,y1,x2,y2=xs.min(),ys.min(),xs.max(),ys.max(); cx,cy=int(xs.mean()),int(ys.mean())
            cv2.rectangle(canvas,(x1,y1),(x2,y2),color,3)
        else: cx=cy=30
        if r['recognition_status']=='eligible':
            note=f"{i}. {short}: {REL.get(r['lateral_relation'],r['lateral_relation'])} + {REL.get(r['longitudinal_relation'],r['longitudinal_relation'])}"
        else:
            note=f"{i}. {short}: near, skipped"
        cv2.circle(canvas,(cx,cy),18,color,-1,cv2.LINE_AA); cv2.putText(canvas,str(i),(cx-7,cy+7),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
        legend.append((note,color,r['distance_m']))
    draw_pose_skeleton(canvas, pose['annotation2D'][cam], annotation_width=ann_w, annotation_height=ann_h)
    draw_forward_axis(canvas, results[0]['human_frame'], cal, annotation_width=ann_w, annotation_height=ann_h, length_m=.55)
    final=cv2.copyMakeBorder(canvas,132,0,0,0,cv2.BORDER_CONSTANT,value=(24,24,24))
    cv2.putText(final,f'{take}  {cam}  frame {frame}   multi-object spatial relation',(14,30),cv2.FONT_HERSHEY_SIMPLEX,.72,(245,245,245),2,cv2.LINE_AA)
    cv2.putText(final,'yellow=skeleton  magenta=person front  colored masks=objects',(14,60),cv2.FONT_HERSHEY_SIMPLEX,.55,(220,220,220),2,cv2.LINE_AA)
    for j,(text,color,dist) in enumerate(legend):
        y=94+j*27; cv2.circle(final,(24,y-6),7,color,-1,cv2.LINE_AA); cv2.putText(final,text+f'  ({dist:.2f}m)',(42,y),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),2,cv2.LINE_AA)
    cv2.imwrite(str(out/'showcase_original_multi.jpg'), final)
    size=760; margin=90; top=np.full((size,size,3),248,dtype=np.uint8); origin=np.array([size//2,size//2])
    vals=[(abs(r['human_xyz_m']['right']),abs(r['human_xyz_m']['forward'])) for r in results]
    extent=max(.8,max(max(a,b) for a,b in vals)*1.25); scale=(size/2-margin)/extent
    def pix(x,z):
        p=origin+np.array([x*scale,-z*scale]); return int(round(p[0])),int(round(p[1]))
    cv2.circle(top,tuple(origin),int(round(summary['min_distance_m']*scale)),(210,210,210),2)
    cv2.line(top,(margin,origin[1]),(size-margin,origin[1]),(170,170,170),2); cv2.line(top,(origin[0],margin),(origin[0],size-margin),(170,170,170),2)
    cv2.arrowedLine(top,tuple(origin),pix(0,extent*.42),(40,120,220),5); cv2.arrowedLine(top,tuple(origin),pix(extent*.42,0),(220,120,40),5); cv2.circle(top,tuple(origin),15,(30,30,30),-1)
    for i,r in enumerate(results,1):
        color=COLORS[(i-1)%len(COLORS)] if r['recognition_status']=='eligible' else (150,150,150); p=pix(r['human_xyz_m']['right'],r['human_xyz_m']['forward'])
        cv2.line(top,tuple(origin),p,color,2,cv2.LINE_AA); cv2.circle(top,p,18,color,-1,cv2.LINE_AA); cv2.putText(top,str(i),(p[0]-7,p[1]+7),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2,cv2.LINE_AA)
    for j,(text,color,dist) in enumerate(legend):
        cv2.circle(top,(24,30+j*28),7,color,-1,cv2.LINE_AA); cv2.putText(top,text,(42,36+j*28),cv2.FONT_HERSHEY_SIMPLEX,.58,(30,30,30),2,cv2.LINE_AA)
    cv2.putText(top,'FRONT',(origin[0]+12,margin+10),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,120,220),2); cv2.putText(top,'RIGHT',(size-margin-75,origin[1]-12),cv2.FONT_HERSHEY_SIMPLEX,.65,(220,120,40),2)
    cv2.imwrite(str(out/'showcase_topdown_multi.jpg'), top)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--groups',type=Path,required=True); ap.add_argument('--root',type=Path,default=Path('data/egoexo4d')); ap.add_argument('--output-root',type=Path,default=Path('outputs/spatial/showcase_multi_extra')); ap.add_argument('--min-distance-m',type=float,default=.60); ap.add_argument('--dead-zone-m',type=float,default=.15)
    a=ap.parse_args(); groups=json.loads(a.groups.read_text()); a.output_root.mkdir(parents=True,exist_ok=True)
    cloud_cache = {}
    for g in groups:
        make_group(g,a.root,a.output_root,a.min_distance_m,a.dead_zone_m,cloud_cache)

if __name__=='__main__': main()
