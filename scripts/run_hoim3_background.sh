#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs data/HOI-M3 outputs/qa
LOG="logs/hoim3_background.log"
PIDFILE="logs/hoim3_background.pid"
STATUS="outputs/qa/hoim3_background_status.json"

cat > "$STATUS" <<JSON
{"status":"starting","message":"HOI-M3 subset background job is starting"}
JSON

nohup bash -lc '
set -euo pipefail
cd /root/autodl-tmp/LIMO4SI
LOG="logs/hoim3_background.log"
STATUS="outputs/qa/hoim3_background_status.json"
{
  echo "[$(date -Is)] HOI-M3 subset job started"
  echo "[$(date -Is)] Step 1: download small video subset from HuggingFace"
  python -m pip show huggingface_hub >/dev/null 2>&1 || python -m pip install -q huggingface_hub
  hf download JuzeZhang/HOI-M3 \
    --repo-type dataset \
    --local-dir data/HOI-M3 \
    --include "videos/office_data19/12.mp4" \
    --include "videos/office_data19/9.mp4"
  echo "[$(date -Is)] Video subset download finished"
  python - <<PY2
import json
from pathlib import Path
root=Path("data/HOI-M3")
videos=[p for p in [root/"videos/office_data19/12.mp4", root/"videos/office_data19/9.mp4"] if p.exists()]
status={"status":"videos_downloaded","videos":[str(p) for p in videos],"video_bytes":sum(p.stat().st_size for p in videos)}
Path("outputs/qa/hoim3_background_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
print(status)
PY2
  echo "[$(date -Is)] Step 2: check annotations / converted multihuman scenes"
  python scripts/convert_hoim3_to_multihuman.py --input data/HOI-M3 --output outputs/qa/hoim3_multihuman_scenes.json
  python - <<PY3
import json
from pathlib import Path
meta=json.loads(Path("outputs/qa/hoim3_multihuman_scenes.json").read_text())
if meta.get("status") != "ok" or not meta.get("scenes"):
    status={"status":"waiting_for_annotations","message":"Videos downloaded, but no usable HOI-M3 multihuman annotation export was found. Download/extract Google Drive annotations or provide humans_in_space*.json, then rerun scripts/build_hoim3_multihuman_qa.py.","converted_status":meta}
    Path("outputs/qa/hoim3_background_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print(json.dumps(status,ensure_ascii=False,indent=2))
    raise SystemExit(0)
PY3
  echo "[$(date -Is)] Step 3: generate real HOI-M3 Task4 QA and rebuild website"
  python scripts/build_hoim3_multihuman_qa.py --hoi-m3-root data/HOI-M3
  python scripts/build_static_qa_site.py
  python - <<PY4
import json
from pathlib import Path
status={"status":"complete","message":"HOI-M3 annotations converted; Task4 QA and website rebuilt."}
Path("outputs/qa/hoim3_background_status.json").write_text(json.dumps(status,ensure_ascii=False,indent=2))
print(status)
PY4
  echo "[$(date -Is)] HOI-M3 subset job complete"
} >> "$LOG" 2>&1
' >/dev/null 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"
echo "Started HOI-M3 background job PID=$PID"
echo "Log: $LOG"
echo "Status: $STATUS"
