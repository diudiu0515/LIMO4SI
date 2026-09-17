"""Fail-closed evidence audit for EgoExo4D Task 4 candidates.

This module only decides whether the supplied annotations can support a
deterministic multi-human claim.  It does not infer spatial relations from
video and it never calls a language model.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


def audit_task4_evidence(
    *,
    capture: Mapping[str, Any],
    trajectory_session_uids: Sequence[str],
    body_subject_counts: Mapping[str, int],
    relations_take_names: Sequence[str],
    available_files: Sequence[str],
) -> dict[str, Any]:
    """Report which Task 4 capabilities are provable from source annotations."""
    cameras = list(capture.get("cameras") or [])
    tracked_ego_cameras = [
        str(camera.get("cam_id"))
        for camera in cameras
        if camera.get("is_ego") and camera.get("device_type") == "aria"
    ]
    session_uids = sorted({str(value) for value in trajectory_session_uids if value})
    counts = {str(key): int(value) for key, value in body_subject_counts.items()}
    multi_subject_takes = sorted(key for key, value in counts.items() if value >= 2)
    capture_name = str(capture.get("capture_name") or "")
    related_takes = sorted(
        name for name in {str(value) for value in relations_take_names if value}
        if name == capture_name or name.startswith(capture_name + "_")
    )
    files = {str(value) for value in available_files}

    passing_reasons: list[str] = []
    if len(session_uids) < 2:
        passing_reasons.append("fewer than two independently tracked human trajectories")
    if len(tracked_ego_cameras) < 2:
        passing_reasons.append("fewer than two trajectory-bearing Aria ego cameras")
    if not multi_subject_takes:
        passing_reasons.append("EgoPose has fewer than two annotated subjects per frame")
    passing_reasons.append("no annotation-backed body-forward vector for two distinct people")

    occlusion_reasons: list[str] = []
    if not related_takes:
        occlusion_reasons.append("no Relations annotation for this capture")
    if not multi_subject_takes:
        occlusion_reasons.append("no two-person target geometry")
    if "semidense_points.csv.gz" not in files:
        occlusion_reasons.append("no semidense scene point cloud in the downloaded capture package")
    occlusion_reasons.append("Relations object masks do not provide time-aligned 3D blocker surfaces")

    capabilities = {
        "passing_side_and_final_position": {
            "status": "accepted" if not passing_reasons else "rejected",
            "reasons": passing_reasons,
        },
        "physical_visibility_occlusion_timeline": {
            "status": "accepted" if not occlusion_reasons else "rejected",
            "reasons": occlusion_reasons,
        },
    }
    accepted = [name for name, row in capabilities.items() if row["status"] == "accepted"]
    return {
        "schema_version": "limo4si.egoexo_task4_evidence_audit.v1",
        "capture_uid": str(capture.get("capture_uid") or ""),
        "capture_name": capture_name,
        "reasoning_owner": "deterministic_code",
        "language_model_used": False,
        "trajectory_session_uids": session_uids,
        "tracked_ego_cameras": tracked_ego_cameras,
        "body_subject_counts": counts,
        "multi_subject_takes": multi_subject_takes,
        "relations_takes": related_takes,
        "capabilities": capabilities,
        "release_eligible": bool(accepted),
        "accepted_capabilities": accepted,
    }
