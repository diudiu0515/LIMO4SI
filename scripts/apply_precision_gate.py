#!/usr/bin/env python3
"""Precision gate for the LIMO4SI QA website/export.

This is intentionally conservative: the showcase should contain fewer examples
rather than examples whose visible evidence and computed evidence disagree.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE_ROOT = ROOT / "site/qa_benchmark"

# Known visual/evidence mismatches in the currently downloaded lightweight subset.
# bedroom_data04 visibly contains an extra person in view0, but the local subset
# only has person0/person1 SMPL-X trajectories. It is therefore not a precise
# multi-human QA example until person2/person3 tracks are added.
DEFAULT_EXCLUDED_CASE_SUBSTRINGS = {"bedroom_data04", "mh_demo_"}

NON_FATAL_MISSING = {
    "left_fingertips",
    "right_fingertips",
    "semantic world_axes for room-level allocentric labels",
}

# Proxies that are acceptable if explicitly reported; they are still computed
# geometry, not guessed answers. Stronger claims should not be made from them.
ACCEPTABLE_APPROX_PREFIXES = (
    "visibility uses geometric FOV",
    "occlusion only checks listed candidate object centers",
    "eye origin approximated by nose joint",
    "head origin approximated from shoulder center",
    "view direction approximated by body forward axis",
    "easiest-to-reach is a geometric proxy",
    "object centers are held fixed at the key frame",
    "2D image-plane person tracking; not metric 3D identity",
)

REJECT_QUESTION_TYPES = {
    # This asks about human-caused object motion but current implementation is only
    # 2D mask/bbox motion, so it is not precise enough for the main showcase.
    "object_motion_proxy_over_video",
    # Evidence-gate rows are useful internally but should not be shown as QA.
    "multi_human_evidence_gate",
}


def load_site_data(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"window\.QA_DATA\s*=\s*(.*);\s*$", text, re.S)
    if not m:
        raise ValueError(f"Cannot parse {path}")
    return json.loads(m.group(1))


def write_site_data(path: Path, data: dict[str, Any]) -> None:
    path.write_text("window.QA_DATA = " + json.dumps(data, ensure_ascii=False, indent=2) + ";\n", encoding="utf-8")


def walk_values(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_values(child)


def collect_list(result: Any, key: str) -> list[str]:
    found: list[str] = []
    for node in walk_values(result):
        v = node.get(key)
        if isinstance(v, list):
            found.extend(str(x) for x in v if x)
    return sorted(set(found))


def media_exists(path_value: str | None) -> bool:
    if not path_value:
        return True
    p = SITE_ROOT / path_value[2:] if path_value.startswith("./") else ROOT / path_value
    return p.exists() and p.stat().st_size > 0


def acceptable_approx(item: str) -> bool:
    return any(item.startswith(prefix) for prefix in ACCEPTABLE_APPROX_PREFIXES)


def qa_decision(group: dict[str, Any], qa: dict[str, Any]) -> tuple[bool, list[str], str]:
    reasons: list[str] = []
    qtype = qa.get("question_type", "")
    result = qa.get("result_json") or {}
    status = qa.get("status") or result.get("status")
    if qtype in REJECT_QUESTION_TYPES:
        reasons.append(f"question_type rejected for precision gate: {qtype}")
    if status == "missing_evidence" or result.get("status") == "missing_evidence":
        reasons.append("missing_evidence status")
    missing = collect_list(result, "missing_evidence")
    fatal_missing = [x for x in missing if x not in NON_FATAL_MISSING]
    if fatal_missing:
        reasons.append("fatal missing evidence: " + "; ".join(fatal_missing))
    approximations = collect_list(result, "approximations")
    bad_approx = [x for x in approximations if not acceptable_approx(x)]
    if bad_approx:
        reasons.append("unaccepted approximation: " + "; ".join(bad_approx))
    # Dynamic benchmark: every displayed QA should explicitly satisfy T/H/S.
    if isinstance(result, dict):
        if result.get("T_Q") is not True or result.get("H_Q") is not True or result.get("S_Q") is not True:
            # Older static task1/task3 rows may not carry T/H/S; do not show them in strict dynamic showcase.
            reasons.append("does not explicitly carry T_Q/H_Q/S_Q evidence flags")
    return (not reasons), reasons, "high" if not approximations and not missing else "audited_proxy"


def group_decision(group: dict[str, Any], exclude_substrings: set[str]) -> tuple[bool, list[str]]:
    name = str(group.get("name", ""))
    title = str(group.get("title", ""))
    hay = name + " " + title
    reasons = []
    for bad in exclude_substrings:
        if bad and bad in hay:
            reasons.append(f"excluded case substring: {bad}")
    for media_key in ("video_clip", "original_image", "topdown_image"):
        if not media_exists(group.get(media_key)):
            reasons.append(f"missing media: {media_key}={group.get(media_key)}")
    # If a HOI-M3 group says 2 tracked people, keep it only as two-tracked-person QA;
    # the wording must not claim all visible people are covered.
    if name.startswith("hoi_m3_"):
        for q in group.get("qa", []):
            tl = ((q.get("result_json") or {}).get("pair_timeline") or {})
            if tl and tl.get("status") != "ok":
                reasons.append("HOI-M3 pair timeline is not ok")
    return (not reasons), reasons


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("site/qa_benchmark/data.js"))
    ap.add_argument("--output", type=Path, default=Path("site/qa_benchmark/data.js"))
    ap.add_argument("--audit-output", type=Path, default=Path("outputs/qa/precision_gate_audit.json"))
    ap.add_argument("--exclude-case-substrings", default=",".join(sorted(DEFAULT_EXCLUDED_CASE_SUBSTRINGS)))
    args = ap.parse_args()

    inp = ROOT / args.input if not args.input.is_absolute() else args.input
    out = ROOT / args.output if not args.output.is_absolute() else args.output
    audit_out = ROOT / args.audit_output if not args.audit_output.is_absolute() else args.audit_output
    exclude = {x.strip() for x in args.exclude_case_substrings.split(",") if x.strip()}

    data = load_site_data(inp)
    kept_groups = []
    rejected_groups = []
    rejected_qas = []
    quality_counter = Counter()
    type_counter = Counter()

    for idx, group in enumerate(data.get("groups", []), 1):
        gok, greasons = group_decision(group, exclude)
        if not gok:
            rejected_groups.append({"case_index": idx, "case_id": group.get("name"), "title": group.get("title"), "reasons": greasons})
            continue
        new_group = dict(group)
        new_qas = []
        for qidx, qa in enumerate(group.get("qa", []), 1):
            ok, reasons, quality = qa_decision(group, qa)
            if ok:
                qq = dict(qa)
                qq["precision_quality"] = quality
                qq["precision_gate"] = {"status": "pass", "note": "Kept by strict evidence/QA gate."}
                new_qas.append(qq)
                quality_counter[quality] += 1
                type_counter[(qq.get("task_id"), qq.get("question_type"))] += 1
            else:
                rejected_qas.append({"case_index": idx, "case_id": group.get("name"), "qa_index": qidx, "question_type": qa.get("question_type"), "question": qa.get("question"), "reasons": reasons})
        if new_qas:
            new_group["qa"] = new_qas
            new_group["precision_gate"] = {"status": "pass", "kept_qa_count": len(new_qas), "original_qa_count": len(group.get("qa", []))}
            kept_groups.append(new_group)
        else:
            rejected_groups.append({"case_index": idx, "case_id": group.get("name"), "title": group.get("title"), "reasons": ["no QA remained after precision gate"]})

    data["groups"] = kept_groups
    data["precision_gate"] = {
        "status": "applied",
        "policy": "fewer but precise: remove demos, known visual/evidence mismatches, missing-evidence QA, and unsupported high-level proxy claims",
        "kept_group_count": len(kept_groups),
        "kept_qa_count": sum(len(g.get("qa", [])) for g in kept_groups),
        "quality_counts": dict(quality_counter),
    }
    write_site_data(out, data)
    audit = {
        "status": "ok",
        "input": str(inp.relative_to(ROOT) if inp.is_relative_to(ROOT) else inp),
        "output": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "kept_groups": len(kept_groups),
        "kept_qas": sum(len(g.get("qa", [])) for g in kept_groups),
        "quality_counts": dict(quality_counter),
        "question_type_counts": {f"{k[0]}::{k[1]}": v for k, v in sorted(type_counter.items())},
        "rejected_groups": rejected_groups,
        "rejected_qas": rejected_qas,
        "nonfatal_missing": sorted(NON_FATAL_MISSING),
        "acceptable_approx_prefixes": list(ACCEPTABLE_APPROX_PREFIXES),
    }
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"kept_groups": audit["kept_groups"], "kept_qas": audit["kept_qas"], "rejected_groups": len(rejected_groups), "rejected_qas": len(rejected_qas)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
