from __future__ import annotations

from limo4si.human_frame import build_human_frame
from limo4si.perspective_qa import (
    level2_occlusion_answer,
    nearest_reachable_object,
    nearest_object_analysis,
    spatial_relation_quality_answer,
    reference_frame_switching_answer,
    reach_for_intent,
    static_reachability_answer,
    unified_reach_analysis,
    visibility_answer,
)


def joints():
    return {
        "left_hip": [-0.2, 0.0, 0.0],
        "right_hip": [0.2, 0.0, 0.0],
        "left_shoulder": [-0.25, 0.7, 0.0],
        "right_shoulder": [0.25, 0.7, 0.0],
        "nose": [0.0, 0.85, 0.35],
        "left_eye": [-0.04, 0.86, 0.28],
        "right_eye": [0.04, 0.86, 0.28],
        "left_wrist": [-0.35, 0.35, 0.55],
        "right_wrist": [0.35, 0.35, 0.55],
    }


def test_visibility_detects_in_fov_target():
    target = {"object_id": "cup", "object_xyz_world_m": [0.0, 0.8, 1.3]}
    result = visibility_answer(target, joints(), candidates=[target], fov_degrees=120)
    assert result["status"] == "ok"
    assert result["inside_fov"] is True
    assert result["visible"] is True




def test_visibility_reports_radius_scored_blocker():
    target = {"object_id": "tv", "object_xyz_world_m": [0.0, 0.8, 2.0]}
    blocker = {
        "object_id": "box",
        "object_xyz_world_m": [0.08, 0.8, 1.0],
        "quality": {"object_radius_m": 0.12},
    }
    result = visibility_answer(target, joints(), candidates=[target, blocker], fov_degrees=120, tube_radius_m=0.05)
    assert result["status"] == "ok"
    assert result["inside_fov"] is True
    assert result["visible"] is False
    assert result["visibility_state"] == "occluded_by_listed_object"
    assert result["blocker"]["object_id"] == "box"
    assert result["blocker"]["occlusion_radius_m"] == 0.12
    assert "occlusion_score" in result["blocker"]
    assert "angular_margin_deg" in result["blocker"]


def test_visibility_reports_outside_fov():
    target = {"object_id": "cup", "object_xyz_world_m": [2.0, 0.8, 0.2]}
    result = visibility_answer(target, joints(), candidates=[target], fov_degrees=90)
    assert result["status"] == "ok"
    assert result["inside_fov"] is False
    assert result["visible"] is False
    assert result["visibility_state"] == "outside_field_of_view"
    assert result["fov_zone"] == "outside"

def test_level2_detects_centroid_blocker():
    target = {"object_id": "tv", "object_xyz_world_m": [0.0, 0.8, 2.0]}
    blocker = {"object_id": "box", "object_xyz_world_m": [0.02, 0.8, 1.0]}
    result = level2_occlusion_answer(joints(), target, [target, blocker], tube_radius_m=0.12)
    assert result["status"] == "ok"
    assert result["blocker"]["object_id"] == "box"
    assert result["occlusion_state"] == "blocked_by_listed_object"
    assert result["blocker_confidence"] in {"medium", "medium_high"}
    assert "occlusion_score" in result["blocker"]


def test_nearest_reachable_uses_wrist_distance():
    near = {"object_id": "spoon", "object_xyz_world_m": [0.35, 0.35, 0.62]}
    far = {"object_id": "pan", "object_xyz_world_m": [1.8, 0.2, 1.8]}
    result = nearest_reachable_object([far, near], joints(), reach_radius_m=0.25)
    assert result["status"] == "ok"
    assert result["chosen"]["object_id"] == "spoon"
    assert result["chosen"]["reachable"] is True


def test_nearest_object_analysis_separates_distance_and_reachability():
    pose = joints()
    pose["right_elbow"] = [0.35, 0.55, 0.25]
    near_far_hand = {"object_id": "plate", "object_xyz_world_m": [-0.2, 0.2, 0.1], "distance_m": 0.25, "human_frame": build_human_frame(pose).to_dict()}
    easy = {"object_id": "cup", "object_xyz_world_m": [0.36, 0.35, 0.62], "distance_m": 0.80, "human_frame": build_human_frame(pose).to_dict()}
    result = nearest_object_analysis([near_far_hand, easy], pose)
    assert result["status"] == "ok"
    assert result["nearest_by_distance"]["object_id"] == "plate"
    assert result["easiest_to_reach"]["object_id"] == "cup"
    assert result["same_object"] is False


def test_spatial_relation_quality_flags_near_object():
    frame = build_human_frame(joints()).to_dict()
    sample = {"object_id": "cup", "object_xyz_world_m": [0.05, 0.1, 0.1], "human_frame": frame, "quality": {"points_in_mask": 20, "robust_inliers": 12}, "distance_validation": {"validated": True}}
    result = spatial_relation_quality_answer(sample, min_distance_m=0.6)
    assert result["status"] == "ok"
    assert result["near_threshold"] is True
    assert result["direction_confidence"] == "low"



def test_reference_frame_switching_camera_coordinates():
    frame = build_human_frame(joints())
    result = reference_frame_switching_answer(
        [1.0, 2.0, 3.0],
        frame,
        camera_intrinsics=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        camera_extrinsics=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    )
    assert result["status"] == "ok"
    assert result["egocentric_camera"]["camera_xyz_m"] == [1.0, 2.0, 3.0]
    assert "pixel_xy" in result["egocentric_camera"]
    assert result["semantic_room_frame"]["status"] == "unavailable"
    assert "semantic world_axes for room-level allocentric labels" in result["missing_evidence"]



def test_reach_for_intent_uses_temporal_approach():
    obj = {"object_id": "cup", "object_xyz_world_m": [0.3, 0.35, 0.55]}
    other = {"object_id": "plate", "object_xyz_world_m": [1.2, 0.35, 1.2]}
    seq = []
    for z in [0.05, 0.18, 0.32, 0.46, 0.54]:
        pose = joints()
        pose["right_wrist"] = [0.3, 0.35, z]
        seq.append(pose)
    result = reach_for_intent([other, obj], seq, reach_radius_m=0.25, min_approach_m=0.05)
    assert result["status"] == "ok"
    assert result["chosen"]["object_id"] == "cup"
    assert result["chosen"]["approaching"] is True
    assert result["chosen"]["min_distance_m"] < 0.05



def test_reach_for_intent_uses_static_hand_and_visibility_evidence():
    obj = {"object_id": "cup", "object_xyz_world_m": [0.36, 0.35, 0.62]}
    other = {"object_id": "plate", "object_xyz_world_m": [1.5, 0.35, 1.2]}
    pose = joints()
    pose["right_elbow"] = [0.35, 0.55, 0.25]
    pose["right_wrist"] = [0.35, 0.35, 0.55]
    hand = {
        "right_index_4": [0.36, 0.35, 0.62],
        "right_middle_4": [0.37, 0.35, 0.62],
        "right_thumb_4": [0.35, 0.34, 0.62],
    }
    seq = [pose, pose]
    result = reach_for_intent([other, obj], seq, current_joints=pose, hand_joints=hand, min_approach_m=0.05)
    assert result["status"] == "ok"
    assert result["chosen"]["object_id"] == "cup"
    assert result["chosen"]["grasp_cue"] is True
    assert result["chosen"]["intent_state"] == "possible_contact_or_grasp"
    assert "static_reachability" in result["chosen"]
    assert "visibility" in result["chosen"]


def test_static_reachability_uses_arm_span():
    pose = joints()
    pose["right_elbow"] = [0.35, 0.55, 0.25]
    pose["right_wrist"] = [0.35, 0.35, 0.55]
    reachable = {"object_id": "cup", "object_xyz_world_m": [0.36, 0.35, 0.62]}
    far = {"object_id": "box", "object_xyz_world_m": [2.5, 0.35, 2.5]}
    yes = static_reachability_answer(reachable, pose)
    no = static_reachability_answer(far, pose)
    assert yes["status"] == "ok"
    assert yes["reachable"] is True
    assert yes["hand_already_close"] is True
    assert no["status"] == "ok"
    assert no["reachable"] is False



def test_unified_reach_analysis_combines_static_and_motion():
    target = {"object_id": "cup", "object_xyz_world_m": [0.36, 0.35, 0.62]}
    other = {"object_id": "plate", "object_xyz_world_m": [1.2, 0.35, 1.2]}
    seq = []
    for z in [0.05, 0.22, 0.38, 0.52, 0.61]:
        pose = joints()
        pose["right_elbow"] = [0.35, 0.55, 0.25]
        pose["right_wrist"] = [0.35, 0.35, z]
        seq.append(pose)
    result = unified_reach_analysis(target, [other, target], seq[-1], seq, min_approach_m=0.05)
    assert result["status"] == "ok"
    assert result["answer_type"] == "unified_reach_analysis"
    assert result["static_reachable"] is True
    assert result["reaching_motion"] is True
    assert result["reach_state"] == "reachable_and_reaching"



def test_static_reachability_uses_fingertips_and_obstacles():
    pose = joints()
    pose["right_elbow"] = [0.35, 0.55, 0.25]
    pose["right_wrist"] = [0.35, 0.35, 0.55]
    hand = {
        "right_index_4": [0.36, 0.35, 0.62],
        "right_middle_4": [0.37, 0.35, 0.62],
        "right_thumb_4": [0.35, 0.34, 0.62],
    }
    target = {"object_id": "cup", "object_xyz_world_m": [0.36, 0.35, 0.62]}
    obstacle = {"object_id": "spoon", "object_xyz_world_m": [0.355, 0.35, 0.585]}
    result = static_reachability_answer(target, pose, hand_joints=hand, candidates=[target, obstacle], obstacle_tube_radius_m=0.05)
    assert result["reachable"] is True
    assert result["grasp_cue"] is True
    right_arm = next(arm for arm in result["arms"] if arm["side"] == "right")
    assert right_arm["fingertip_count"] == 3
    assert right_arm["obstacle_count"] == 1
