#!/usr/bin/env python3
"""Remove legacy/raw evidence from the website data while keeping audit files intact."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    if not m:
        raise ValueError(f"Cannot parse {path}")
    return json.loads(m.group(1))

def slim(data: dict) -> dict:
    groups=[]
    for group in data.get("groups", []):
        keep_group={k:v for k,v in group.items() if k not in {"raw_summary", "dynamic_timeline", "summary_path"}}
        qas=[]
        for qa in group.get("qa", []):
            qas.append({k:v for k,v in qa.items() if k not in {"raw_json", "raw_json_path"}})
        keep_group["qa"]=qas
        groups.append(keep_group)
    data=dict(data)
    data["groups"]=groups
    data["site_data_policy"]={
        "status":"slim",
        "note":"Website retains only upgraded video-task fields; raw JSON remains in outputs/qa and outputs/spatial for audit."
    }
    return data

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("site/qa_benchmark/data.js"))
    ap.add_argument("--output", type=Path, default=Path("site/qa_benchmark/data.js"))
    args=ap.parse_args()
    inp=ROOT/args.input if not args.input.is_absolute() else args.input
    out=ROOT/args.output if not args.output.is_absolute() else args.output
    data=slim(load(inp))
    out.write_text("window.QA_DATA = "+json.dumps(data,ensure_ascii=False,indent=2)+";\n",encoding="utf-8")
    print(json.dumps({"groups":len(data.get("groups",[])),"qa":sum(len(g.get("qa",[])) for g in data.get("groups",[])),"bytes":out.stat().st_size},ensure_ascii=False))

if __name__=="__main__":
    main()
