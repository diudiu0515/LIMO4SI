#!/usr/bin/env python3
from __future__ import annotations
import json, subprocess, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT/'logs/hoim3_background.log'
STATUS = ROOT/'outputs/qa/hoim3_background_status.json'
TARGETS = ['videos/office_data19/12.mp4', 'videos/office_data19/9.mp4']

def write_status(**kw):
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(kw, ensure_ascii=False, indent=2))

def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a') as f:
        f.write(msg+'\n')
    print(msg, flush=True)

def main():
    try:
        from huggingface_hub import hf_hub_download
        write_status(status='downloading_videos', targets=TARGETS)
        log('HOI-M3 Python background job started')
        out_root = ROOT/'data/HOI-M3'
        out_root.mkdir(parents=True, exist_ok=True)
        downloaded=[]
        for rel in TARGETS:
            log(f'downloading {rel}')
            got = hf_hub_download(repo_id='JuzeZhang/HOI-M3', repo_type='dataset', filename=rel, local_dir=str(out_root), local_dir_use_symlinks=False, resume_download=True)
            p=Path(got)
            downloaded.append({'path':str(p), 'bytes':p.stat().st_size})
            log(f'downloaded {p} {p.stat().st_size} bytes')
        write_status(status='videos_downloaded', downloaded=downloaded)
        log('checking annotations / converted scenes')
        subprocess.run([sys.executable, 'scripts/convert_hoim3_to_multihuman.py', '--input', 'data/HOI-M3', '--output', 'outputs/qa/hoim3_multihuman_scenes.json'], cwd=ROOT, check=True)
        meta=json.loads((ROOT/'outputs/qa/hoim3_multihuman_scenes.json').read_text())
        if meta.get('status') != 'ok' or not meta.get('scenes'):
            write_status(status='waiting_for_annotations', message='Videos downloaded, but no usable HOI-M3 multihuman annotation export was found. Need Google Drive annotations/toolbox export.', downloaded=downloaded, converted_status=meta)
            log('waiting_for_annotations')
            return
        subprocess.run([sys.executable, 'scripts/build_hoim3_multihuman_qa.py', '--hoi-m3-root', 'data/HOI-M3'], cwd=ROOT, check=True)
        subprocess.run([sys.executable, 'scripts/build_static_qa_site.py'], cwd=ROOT, check=True)
        write_status(status='complete', message='HOI-M3 QA and visualization generated.', downloaded=downloaded)
        log('complete')
    except Exception as e:
        write_status(status='failed', error=repr(e), traceback=traceback.format_exc())
        log('failed: '+repr(e))
        raise

if __name__ == '__main__':
    main()
