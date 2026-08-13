from __future__ import annotations

from limo4si.human_frame import build_human_frame
from limo4si.perspective_qa import (
    level2_occlusion_answer,
    nearest_reachable_object,
    reference_frame_switching_answer,
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


def test_level2_detects_centroid_blocker():
    target = {"object_id": "tv", "object_xyz_world_m": [0.0, 0.8, 2.0]}
    blocker = {"object_id": "box", "object_xyz_world_m": [0.02, 0.8, 1.0]}
    result = level2_occlusion_answer(joints(), target, [target, blocker], tube_radius_m=0.12)
    assert result["status"] == "ok"
    assert result["blocker"]["object_id"] == "box"


def test_nearest_reachable_uses_wrist_distance():
    near = {"object_id": "spoon", "object_xyz_world_m": [0.35, 0.35, 0.62]}
    far = {"object_id": "pan", "object_xyz_world_m": [1.8, 0.2, 1.8]}
    result = nearest_reachable_object([far, near], joints(), reach_radius_m=0.25)
    assert result["status"] == "ok"
    assert result["chosen"]["object_id"] == "spoon"
    assert result["chosen"]["reachable"] is True


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
