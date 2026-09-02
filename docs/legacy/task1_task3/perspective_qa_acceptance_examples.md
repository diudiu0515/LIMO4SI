# Legacy Perspective-Grounded QA Acceptance Examples

每个 task 抽 3 条真实由 `scripts/build_perspective_qa.py` 生成的结果，用于人工验收。

## Human-centric left/right/front/back

### Example 1: query_01_iiith32_frame5280

Q: From the person's view, where is the stainless salt container?

A: Object is left and above relative to the person; distance 1.04 m.

Status: `ok`

Method: Computed from human_frame.world_to_human(object_centroid) and independent lateral/longitudinal/vertical thresholds.

- lateral: `left`
- longitudinal: `same_longitudinal_position`
- vertical: `above`
- distance_m: `1.0372619756925032`

```json
{
  "status": "ok",
  "answer_type": "human_centric_spatial",
  "relation": {
    "human_xyz_m": {
      "right": -0.8479744005470141,
      "up": 0.5934671982273414,
      "forward": 0.06817995939157447
    },
    "distance_m": 1.0372619756925032,
    "horizontal_distance_m": 0.8507109326003192,
    "lateral_relation": "left",
    "longitudinal_relation": "same_longitudinal_position",
    "vertical_relation": "above",
    "text_zh": "物体在人的左侧、前后原点附近、上方，前向分量 0.07 m，右向分量 -0.85 m，高度差 0.59 m，直线距离 1.04 m"
  },
  "answer": "Object is left and above relative to the person; distance 1.04 m."
}
```

### Example 2: query_02_iiith30_frame4440

Q: From the person's view, where is the stainless bowl?

A: Object is right and front relative to the person; distance 0.41 m.

Status: `ok`

Method: Computed from human_frame.world_to_human(object_centroid) and independent lateral/longitudinal/vertical thresholds.

- lateral: `right`
- longitudinal: `front`
- vertical: `same_height`
- distance_m: `0.4115773047850878`

```json
{
  "status": "ok",
  "answer_type": "human_centric_spatial",
  "relation": {
    "human_xyz_m": {
      "right": 0.24690396695242572,
      "up": 0.0177867497144261,
      "forward": 0.3288129262238772
    },
    "distance_m": 0.4115773047850878,
    "horizontal_distance_m": 0.41119278854176594,
    "lateral_relation": "right",
    "longitudinal_relation": "front",
    "vertical_relation": "same_height",
    "text_zh": "物体在人的右侧、前方、近似同高，前向分量 0.33 m，右向分量 0.25 m，高度差 0.02 m，直线距离 0.41 m"
  },
  "answer": "Object is right and front relative to the person; distance 0.41 m."
}
```

### Example 3: query_03_iiith30_frame4440_better

Q: From the person's view, where is the stainless paprika container?

A: Object is right and behind and above relative to the person; distance 1.39 m.

Status: `ok`

Method: Computed from human_frame.world_to_human(object_centroid) and independent lateral/longitudinal/vertical thresholds.

- lateral: `right`
- longitudinal: `behind`
- vertical: `above`
- distance_m: `1.3913757833302112`

```json
{
  "status": "ok",
  "answer_type": "human_centric_spatial",
  "relation": {
    "human_xyz_m": {
      "right": 0.1759104605490857,
      "up": 0.5057430045212667,
      "forward": -1.284214193070988
    },
    "distance_m": 1.3913757833302112,
    "horizontal_distance_m": 1.2962062273479327,
    "lateral_relation": "right",
    "longitudinal_relation": "behind",
    "vertical_relation": "above",
    "text_zh": "物体在人的右侧、后方、上方，前向分量 -1.28 m，右向分量 0.18 m，高度差 0.51 m，直线距离 1.39 m"
  },
  "answer": "Object is right and behind and above relative to the person; distance 1.39 m."
}
```

## Visibility and occlusion reasoning

### Example 1: query_01_iiith32_frame5280

Q: Can the person see the stainless bowl? Consider head/body direction and occlusion.

A: Likely not visible: target is 56.2° from the viewing direction, outside the 110° field of view.

Status: `ok`

Method: Head/face direction if available; otherwise body-forward. Occlusion is approximated by listed object centers inside an observer-to-target tube.

- visible: `False`
- angle_to_view_direction_deg: `56.22924754843461`
- inside_fov: `False`
- occluders: `0`

```json
{
  "status": "ok",
  "answer_type": "visibility_occlusion",
  "target_object_id": "stainless bowl_0",
  "observer": {
    "origin": [
      -0.6623212586193541,
      -0.7319972639088088,
      0.09503001668101665
    ],
    "forward": [
      -0.40926716262817275,
      -0.9116844055989748,
      0.036495673468632944
    ],
    "right": [
      -0.8969006913293953,
      0.44073701246928526,
      0.03633229599821321
    ],
    "up": [
      0.016844176075673727,
      -0.04805045777003149,
      0.9987028723501411
    ],
    "source": "eyes_to_nose/eye_midpoint",
    "approximations": []
  },
  "angle_to_view_direction_deg": 56.22924754843461,
  "fov_degrees": 110.0,
  "inside_fov": false,
  "occluders": [],
  "visible": false,
  "approximations": [
    "occlusion uses object centroids/tube unless dense masks are supplied"
  ],
  "answer": "Likely not visible: target is 56.2° from the viewing direction, outside the 110° field of view."
}
```

### Example 2: query_02_iiith30_frame4440

Q: Can the person see the stainless spoon? Consider head/body direction and occlusion.

A: Likely visible: target is 31.4° from the viewing direction and no listed object blocks the sightline.

Status: `ok`

Method: Head/face direction if available; otherwise body-forward. Occlusion is approximated by listed object centers inside an observer-to-target tube.

- visible: `True`
- angle_to_view_direction_deg: `31.358032400428005`
- inside_fov: `True`
- occluders: `0`

```json
{
  "status": "ok",
  "answer_type": "visibility_occlusion",
  "target_object_id": "stainless spoon_0",
  "observer": {
    "origin": [
      -0.7486490570346019,
      0.06172462027836706,
      0.09998526527981783
    ],
    "forward": [
      0.5770774271880182,
      0.09612617728940943,
      -0.811012577627365
    ],
    "right": [
      0.6415703610074331,
      0.766960783976596,
      0.012594749652021862
    ],
    "up": [
      -0.019214368772831118,
      -0.00034559033573153253,
      0.9998153272479781
    ],
    "source": "eyes_to_nose/eye_midpoint",
    "approximations": []
  },
  "angle_to_view_direction_deg": 31.358032400428005,
  "fov_degrees": 110.0,
  "inside_fov": true,
  "occluders": [],
  "visible": true,
  "approximations": [
    "occlusion uses object centroids/tube unless dense masks are supplied"
  ],
  "answer": "Likely visible: target is 31.4° from the viewing direction and no listed object blocks the sightline."
}
```

### Example 3: query_03_iiith30_frame4440_better

Q: Can the person see the mixed vegetables? Consider head/body direction and occlusion.

A: Likely visible: target is 38.3° from the viewing direction and no listed object blocks the sightline.

Status: `ok`

Method: Head/face direction if available; otherwise body-forward. Occlusion is approximated by listed object centers inside an observer-to-target tube.

- visible: `True`
- angle_to_view_direction_deg: `38.34647043158891`
- inside_fov: `True`
- occluders: `0`

```json
{
  "status": "ok",
  "answer_type": "visibility_occlusion",
  "target_object_id": "mixed vegetables_0",
  "observer": {
    "origin": [
      -0.7486490570346019,
      0.06172462027836706,
      0.09998526527981783
    ],
    "forward": [
      0.5770774271880182,
      0.09612617728940943,
      -0.811012577627365
    ],
    "right": [
      0.6415703610074331,
      0.766960783976596,
      0.012594749652021862
    ],
    "up": [
      -0.019214368772831118,
      -0.00034559033573153253,
      0.9998153272479781
    ],
    "source": "eyes_to_nose/eye_midpoint",
    "approximations": []
  },
  "angle_to_view_direction_deg": 38.34647043158891,
  "fov_degrees": 110.0,
  "inside_fov": true,
  "occluders": [],
  "visible": true,
  "approximations": [
    "occlusion uses object centroids/tube unless dense masks are supplied"
  ],
  "answer": "Likely visible: target is 38.3° from the viewing direction and no listed object blocks the sightline."
}
```

## Reachable nearest object

### Example 1: query_01_iiith32_frame5280

Q: Which listed object is closest and reachable by the person?

A: tawa pan_0 is the nearest reachable candidate (0.17 m from nearest hand; reach radius 1.04 m).

Status: `ok`

Method: Uses nearest wrist-to-object distance with reach radius estimated from shoulder width; falls back to pelvis distance if wrists are absent.

- chosen: `tawa pan_0`
- reachable: `True`
- distance_to_nearest_hand_m: `0.17236891009136385`
- reach_radius_m: `1.0358492666035521`

```json
{
  "status": "ok",
  "answer_type": "reachability",
  "reach_radius_m": 1.0358492666035521,
  "chosen": {
    "object_id": "tawa pan_0",
    "distance_to_nearest_hand_m": 0.17236891009136385,
    "distance_to_pelvis_m": 0.6103541905645341,
    "reachable": true,
    "distance_source": "nearest_wrist",
    "raw_object": {
      "status": "ok",
      "recognition_status": "eligible",
      "take_uid": "35bfade9-8ead-46a4-b2f0-cdcfb86df1d6",
      "take_name": "iiith_cooking_32_1",
      "camera": "cam02",
      "frame": 5280,
      "object_id": "tawa pan_0",
      "query": "Where is the pan relative to the person?",
      "object_xyz_world_m": [
        -1.0028125,
        -0.8834,
        -0.388036
      ],
      "human_frame": {
        "origin": [
          -0.5418148109533744,
          -0.5221078677063776,
          -0.5597373796569087
        ],
        "right": [
          -0.8969006913293953,
          0.44073701246928526,
          0.03633229599821321
        ],
        "up": [
          0.016844176075673727,
          -0.04805045777003149,
          0.9987028723501411
        ],
        "forward": [
          -0.4419111037586456,
          -0.8963492842345219,
          -0.03567263695153611
        ]
      },
      "human_xyz_m": {
        "right": 0.2604726363404197,
        "up": 0.18107378715047612,
        "forward": 0.521438900797153
      },
      "distance_m": 0.6103541905645342,
      "horizontal_distance_m": 0.5828760773497843,
      "lateral_relation": "right",
      "longitudinal_relation": "front",
      "vertical_relation": "above",
      "text_zh": "物体在人的右侧、前方、上方，前向分量 0.52 m，右向分量 0.26 m，高度差 0.18 m，直线距离 0.61 m",
      "raw_relation_before_filter": {
        "human_xyz_m": {
          "right": 0.2604726363404197,
          "up": 0.18107378715047612,
          "forward": 0.521438900797153
        },
        "distance_m": 0.6103541905645342,
        "horizontal_distance_m": 0.5828760773497843,
        "lateral_relation": "right",
        "longitudinal_relation": "front",
        "vertical_relation": "above",
        "text_zh": "物体在人的右侧、前方、上方，前向分量 0.52 m，右向分量 0.26 m，高度差 0.18 m，直线距离 0.61 m"
      },
      "distance_validation": {
        "world_direct_m": 0.6103541905645341,
        "human_components_m": 0.6103541905645342,
        "agreement_residual_m": 1.1102230246251565e-16,
        "agreement_pass": true,
        "skeleton_scale": {
          "shoulder_width_m": 0.34827052775148,
          "hip_width_m": 0.2023961252677169,
          "torso_length_m": 0.5502762678348555,
          "plausible": true
        },
        "validated": true
      },
      "quality": {
        "mask_pixels": 176754,
        "points_in_mask": 40618,
        "robust_inliers": 38222
      },
      "inputs": {
        "point_cloud": "data/egoexo4d/takes/iiith_cooking_32_1/trajectory/semidense_points.csv.gz",
        "camera_pose": "data/egoexo4d/annotations/ego_pose/val/camera_pose/35bfade9-8ead-46a4-b2f0-cdcfb86df1d6.json",
        "body_pose": "data/egoexo4d/annotations/ego_pose/val/body/automatic/35bfade9-8ead-46a4-b2f0-cdcfb86df1d6.json"
      }
    }
  },
  "candidates": [
    {
      "object_id": "tawa pan_0",
      "distance_to_nearest_hand_m": 0.17236891009136385,
      "distance_to_pelvis_m": 0.6103541905645341,
      "reachable": true,
      "distance_source": "nearest_wrist",
      "raw_object": {
        "status": "ok",
        "recognition_status": "eligible",
        "take_uid": "35bfade9-8ead-46a4-b2f0-cdcfb86df1d6",
        "t
```

### Example 2: query_02_iiith30_frame4440

Q: Which listed object is closest and reachable by the person?

A: stainless spoon_0 is the nearest reachable candidate (0.20 m from nearest hand; reach radius 1.04 m).

Status: `ok`

Method: Uses nearest wrist-to-object distance with reach radius estimated from shoulder width; falls back to pelvis distance if wrists are absent.

- chosen: `stainless spoon_0`
- reachable: `True`
- distance_to_nearest_hand_m: `0.20398133960583753`
- reach_radius_m: `1.0357746836344146`

```json
{
  "status": "ok",
  "answer_type": "reachability",
  "reach_radius_m": 1.0357746836344146,
  "chosen": {
    "object_id": "stainless spoon_0",
    "distance_to_nearest_hand_m": 0.20398133960583753,
    "distance_to_pelvis_m": 0.34562519654818474,
    "reachable": true,
    "distance_source": "nearest_wrist",
    "raw_object": {
      "status": "filtered_near_or_invalid",
      "recognition_status": "filtered_near_or_invalid",
      "take_uid": "360be2ce-99dd-424e-a7c6-a5908c7aa2f3",
      "take_name": "iiith_cooking_30_1",
      "camera": "cam03",
      "frame": 4440,
      "object_id": "stainless spoon_0",
      "query": "Where is the spoon relative to the person?",
      "object_xyz_world_m": [
        -0.686437,
        0.256212,
        -0.506263
      ],
      "human_frame": {
        "origin": [
          -0.6258500271633722,
          -0.08353375575452052,
          -0.5252054565000108
        ],
        "right": [
          0.6415703610074331,
          0.766960783976596,
          0.012594749652021862
        ],
        "up": [
          -0.019214368772831118,
          -0.00034559033573153253,
          0.9998153272479781
        ],
        "forward": [
          -0.7668234998416871,
          0.6416938806076657,
          -0.014514946821170022
        ]
      },
      "human_xyz_m": {
        "right": 0.22193944064847726,
        "up": 0.01998568593355679,
        "forward": 0.2641973382367386
      },
      "distance_m": 0.3456251965481848,
      "horizontal_distance_m": 0.34504687920156096,
      "lateral_relation": null,
      "longitudinal_relation": null,
      "vertical_relation": null,
      "text_zh": null,
      "raw_relation_before_filter": {
        "human_xyz_m": {
          "right": 0.22193944064847726,
          "up": 0.01998568593355679,
          "forward": 0.2641973382367386
        },
        "distance_m": 0.3456251965481848,
        "horizontal_distance_m": 0.34504687920156096,
        "lateral_relation": "right",
        "longitudinal_relation": "front",
        "vertical_relation": "same_height",
        "text_zh": "物体在人的右侧、前方、近似同高，前向分量 0.26 m，右向分量 0.22 m，高度差 0.02 m，直线距离 0.35 m"
      },
      "distance_validation": {
        "world_direct_m": 0.3456251965481848,
        "human_components_m": 0.3456251965481848,
        "agreement_residual_m": 0.0,
        "agreement_pass": true,
        "skeleton_scale": {
          "shoulder_width_m": 0.3482394515143394,
          "hip_width_m": 0.17814265728705164,
          "torso_length_m": 0.5191240026452791,
          "plausible": true
        },
        "validated": true
      },
      "quality": {
        "mask_pixels": 10048,
        "points_in_mask": 4762,
        "robust_inliers": 3231
      },
      "inputs": {
        "point_cloud": "data/egoexo4d/takes/iiith_cooking_30_1/trajectory/semidense_points.csv.gz",
        "camera_pose": "outputs/calibration/val_12/360be2ce-99dd-424e-a7c6-a5908c7aa2f3.json",
        "body_pose": "data/egoexo4d/annotations/ego_pose/val/body/automatic/360be2ce-99dd-424e-a7c6-a5908c7aa2f3.json"
      }
    }
  },
  "candidates": [
    {
      "object_id": "stainless spoon_0",
      "distance_to_nearest_hand_m": 0.20398133960583753,
      "distance_to_pelvis_m": 0.34562519654818474,
      "reachable": true,
      "distance_source": "nearest_wrist",
      "raw_object": {
        "status": "filtered_near_or_invalid",
        "recognition_status": "filtered_near_or_invalid",
        "take_uid": "360be2ce-99dd-424e-a7c6-a5908c7aa2f3",
  
```

### Example 3: query_03_iiith30_frame4440_better

Q: Which listed object is closest and reachable by the person?

A: sliced onion_0 is the nearest reachable candidate (0.35 m from nearest hand; reach radius 1.04 m).

Status: `ok`

Method: Uses nearest wrist-to-object distance with reach radius estimated from shoulder width; falls back to pelvis distance if wrists are absent.

- chosen: `sliced onion_0`
- reachable: `True`
- distance_to_nearest_hand_m: `0.34565522134794296`
- reach_radius_m: `1.0357746836344146`

```json
{
  "status": "ok",
  "answer_type": "reachability",
  "reach_radius_m": 1.0357746836344146,
  "chosen": {
    "object_id": "sliced onion_0",
    "distance_to_nearest_hand_m": 0.34565522134794296,
    "distance_to_pelvis_m": 0.547935810060509,
    "reachable": true,
    "distance_source": "nearest_wrist",
    "raw_object": {
      "status": "filtered_near_or_invalid",
      "recognition_status": "filtered_near_or_invalid",
      "take_uid": "360be2ce-99dd-424e-a7c6-a5908c7aa2f3",
      "take_name": "iiith_cooking_30_1",
      "camera": "cam03",
      "frame": 4440,
      "object_id": "sliced onion_0",
      "query": "Where is the sliced onion relative to the person?",
      "object_xyz_world_m": [
        -0.8628205,
        0.407964,
        -0.47511950000000003
      ],
      "human_frame": {
        "origin": [
          -0.6258500271633722,
          -0.08353375575452052,
          -0.5252054565000108
        ],
        "right": [
          0.6415703610074331,
          0.766960783976596,
          0.012594749652021862
        ],
        "up": [
          -0.019214368772831118,
          -0.00034559033573153253,
          0.9998153272479781
        ],
        "forward": [
          -0.7668234998416871,
          0.6416938806076657,
          -0.014514946821170022
        ]
      },
      "human_xyz_m": {
        "right": 0.22555709235352683,
        "up": 0.05446008816751891,
        "forward": 0.49637863454471415
      },
      "distance_m": 0.5479358100605091,
      "horizontal_distance_m": 0.5452226616195005,
      "lateral_relation": null,
      "longitudinal_relation": null,
      "vertical_relation": null,
      "text_zh": null,
      "raw_relation_before_filter": {
        "human_xyz_m": {
          "right": 0.22555709235352683,
          "up": 0.05446008816751891,
          "forward": 0.49637863454471415
        },
        "distance_m": 0.5479358100605091,
        "horizontal_distance_m": 0.5452226616195005,
        "lateral_relation": "right",
        "longitudinal_relation": "front",
        "vertical_relation": "slightly_above",
        "text_zh": "物体在人的右侧、前方、略偏上，前向分量 0.50 m，右向分量 0.23 m，高度差 0.05 m，直线距离 0.55 m"
      },
      "distance_validation": {
        "world_direct_m": 0.547935810060509,
        "human_components_m": 0.5479358100605091,
        "agreement_residual_m": 1.1102230246251565e-16,
        "agreement_pass": true,
        "skeleton_scale": {
          "shoulder_width_m": 0.3482394515143394,
          "hip_width_m": 0.17814265728705164,
          "torso_length_m": 0.5191240026452791,
          "plausible": true
        },
        "validated": true
      },
      "quality": {
        "mask_pixels": 18438,
        "points_in_mask": 1812,
        "robust_inliers": 1170
      },
      "inputs": {
        "point_cloud": "data/egoexo4d/takes/iiith_cooking_30_1/trajectory/semidense_points.csv.gz",
        "camera_pose": "outputs/calibration/val_12/360be2ce-99dd-424e-a7c6-a5908c7aa2f3.json",
        "body_pose": "data/egoexo4d/annotations/ego_pose/val/body/automatic/360be2ce-99dd-424e-a7c6-a5908c7aa2f3.json"
      }
    }
  },
  "candidates": [
    {
      "object_id": "sliced onion_0",
      "distance_to_nearest_hand_m": 0.34565522134794296,
      "distance_to_pelvis_m": 0.547935810060509,
      "reachable": true,
      "distance_source": "nearest_wrist",
      "raw_object": {
        "status": "filtered_near_or_invalid",
        "recognition_status": "filtered_near_or_invalid",
        "take_uid": "360be2ce-99dd-4
```

## Level-2 perspective taking

### Example 1: query_01_iiith32_frame5280

Q: From the observer's perspective, which listed object blocks the stainless salt container?

A: No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates.

Status: `ok`

Method: Casts a line segment from observer head to target and selects candidate objects between them with smallest depth and tube distance.

- blocker: `None`
- occluder_count: `0`

```json
{
  "status": "ok",
  "answer_type": "level2_perspective_occlusion",
  "observer": {
    "origin": [
      -0.6623212586193541,
      -0.7319972639088088,
      0.09503001668101665
    ],
    "forward": [
      -0.40926716262817275,
      -0.9116844055989748,
      0.036495673468632944
    ],
    "right": [
      -0.8969006913293953,
      0.44073701246928526,
      0.03633229599821321
    ],
    "up": [
      0.016844176075673727,
      -0.04805045777003149,
      0.9987028723501411
    ],
    "source": "eyes_to_nose/eye_midpoint",
    "approximations": []
  },
  "target_object_id": "stainless salt container_0",
  "blocker": null,
  "occluders": [],
  "approximations": [
    "single-observer demo when only one person pose is available"
  ],
  "answer": "No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates."
}
```

### Example 2: query_02_iiith30_frame4440

Q: From the observer's perspective, which listed object blocks the stainless bowl?

A: No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates.

Status: `ok`

Method: Casts a line segment from observer head to target and selects candidate objects between them with smallest depth and tube distance.

- blocker: `None`
- occluder_count: `0`

```json
{
  "status": "ok",
  "answer_type": "level2_perspective_occlusion",
  "observer": {
    "origin": [
      -0.7486490570346019,
      0.06172462027836706,
      0.09998526527981783
    ],
    "forward": [
      0.5770774271880182,
      0.09612617728940943,
      -0.811012577627365
    ],
    "right": [
      0.6415703610074331,
      0.766960783976596,
      0.012594749652021862
    ],
    "up": [
      -0.019214368772831118,
      -0.00034559033573153253,
      0.9998153272479781
    ],
    "source": "eyes_to_nose/eye_midpoint",
    "approximations": []
  },
  "target_object_id": "stainless bowl_0",
  "blocker": null,
  "occluders": [],
  "approximations": [
    "single-observer demo when only one person pose is available"
  ],
  "answer": "No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates."
}
```

### Example 3: query_03_iiith30_frame4440_better

Q: From the observer's perspective, which listed object blocks the stainless paprika container?

A: No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates.

Status: `ok`

Method: Casts a line segment from observer head to target and selects candidate objects between them with smallest depth and tube distance.

- blocker: `None`
- occluder_count: `0`

```json
{
  "status": "ok",
  "answer_type": "level2_perspective_occlusion",
  "observer": {
    "origin": [
      -0.7486490570346019,
      0.06172462027836706,
      0.09998526527981783
    ],
    "forward": [
      0.5770774271880182,
      0.09612617728940943,
      -0.811012577627365
    ],
    "right": [
      0.6415703610074331,
      0.766960783976596,
      0.012594749652021862
    ],
    "up": [
      -0.019214368772831118,
      -0.00034559033573153253,
      0.9998153272479781
    ],
    "source": "eyes_to_nose/eye_midpoint",
    "approximations": []
  },
  "target_object_id": "stainless paprika container_0",
  "blocker": null,
  "occluders": [],
  "approximations": [
    "single-observer demo when only one person pose is available"
  ],
  "answer": "No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates."
}
```

## Reference-frame switching

### Example 1: query_01_iiith32_frame5280

Q: Describe the stainless salt container in human-centric, camera/egocentric, and allocentric/world frames.

A: {"human_centric": "Object is left and above relative to the person; distance 1.04 m.", "egocentric": {"camera_xyz_m": [0.2627456066799033, -0.17067591161893514, 1.71789951769448], "right_m": 0.2627456066799033, "down_or_up_depends_on_camera_y_m": -0.17067591161893514, "forward_depth_m": 1.71789951769448, "pixel_xy": [2106.7559291435196, 958.6859317665256]}, "allocentric": [0.198601, -0.985471, -0.000281]}

Status: `ok`

Method: Human-centric uses body axes; egocentric uses camera extrinsics; allocentric reports world xyz unless semantic room axes are supplied.

- human relation keys: `['human_xyz_m', 'distance_m', 'horizontal_distance_m', 'lateral_relation', 'longitudinal_relation', 'vertical_relation', 'text_zh']`
- camera xyz exists: `True`
- missing_evidence: `['semantic world_axes for room-level allocentric labels']`

```json
{
  "status": "ok",
  "answer_type": "reference_frame_switching",
  "human_centric": {
    "human_xyz_m": {
      "right": -0.8479744005470141,
      "up": 0.5934671982273414,
      "forward": 0.06817995939157447
    },
    "distance_m": 1.0372619756925032,
    "horizontal_distance_m": 0.8507109326003192,
    "lateral_relation": "left",
    "longitudinal_relation": "same_longitudinal_position",
    "vertical_relation": "above",
    "text_zh": "物体在人的左侧、前后原点附近、上方，前向分量 0.07 m，右向分量 -0.85 m，高度差 0.59 m，直线距离 1.04 m"
  },
  "allocentric_world_xyz_m": [
    0.198601,
    -0.985471,
    -0.000281
  ],
  "allocentric_note": "Raw Ego-Exo4D world coordinates. Semantic room axes require a scene/world-axis declaration.",
  "egocentric_camera": {
    "camera_xyz_m": [
      0.2627456066799033,
      -0.17067591161893514,
      1.71789951769448
    ],
    "right_m": 0.2627456066799033,
    "down_or_up_depends_on_camera_y_m": -0.17067591161893514,
    "forward_depth_m": 1.71789951769448,
    "pixel_xy": [
      2106.7559291435196,
      958.6859317665256
    ]
  },
  "missing_evidence": [
    "semantic world_axes for room-level allocentric labels"
  ],
  "answer": {
    "human_centric": "Object is left and above relative to the person; distance 1.04 m.",
    "egocentric": {
      "camera_xyz_m": [
        0.2627456066799033,
        -0.17067591161893514,
        1.71789951769448
      ],
      "right_m": 0.2627456066799033,
      "down_or_up_depends_on_camera_y_m": -0.17067591161893514,
      "forward_depth_m": 1.71789951769448,
      "pixel_xy": [
        2106.7559291435196,
        958.6859317665256
      ]
    },
    "allocentric": [
      0.198601,
      -0.985471,
      -0.000281
    ]
  }
}
```

### Example 2: query_02_iiith30_frame4440

Q: Describe the stainless bowl in human-centric, camera/egocentric, and allocentric/world frames.

A: {"human_centric": "Object is right and front relative to the person; distance 0.41 m.", "egocentric": {"camera_xyz_m": [-0.021520399978460336, 0.44126610002154076, 0.6506144888317851], "right_m": -0.021520399978460336, "down_or_up_depends_on_camera_y_m": 0.44126610002154076, "forward_depth_m": 0.6506144888317851, "pixel_xy": [1877.8755943193344, 1911.9970287827048]}, "allocentric": [-0.719927, 0.31682299999999997, -0.509085]}

Status: `ok`

Method: Human-centric uses body axes; egocentric uses camera extrinsics; allocentric reports world xyz unless semantic room axes are supplied.

- human relation keys: `['human_xyz_m', 'distance_m', 'horizontal_distance_m', 'lateral_relation', 'longitudinal_relation', 'vertical_relation', 'text_zh']`
- camera xyz exists: `True`
- missing_evidence: `['semantic world_axes for room-level allocentric labels']`

```json
{
  "status": "ok",
  "answer_type": "reference_frame_switching",
  "human_centric": {
    "human_xyz_m": {
      "right": 0.24690396695242572,
      "up": 0.0177867497144261,
      "forward": 0.3288129262238772
    },
    "distance_m": 0.4115773047850878,
    "horizontal_distance_m": 0.41119278854176594,
    "lateral_relation": "right",
    "longitudinal_relation": "front",
    "vertical_relation": "same_height",
    "text_zh": "物体在人的右侧、前方、近似同高，前向分量 0.33 m，右向分量 0.25 m，高度差 0.02 m，直线距离 0.41 m"
  },
  "allocentric_world_xyz_m": [
    -0.719927,
    0.31682299999999997,
    -0.509085
  ],
  "allocentric_note": "Raw Ego-Exo4D world coordinates. Semantic room axes require a scene/world-axis declaration.",
  "egocentric_camera": {
    "camera_xyz_m": [
      -0.021520399978460336,
      0.44126610002154076,
      0.6506144888317851
    ],
    "right_m": -0.021520399978460336,
    "down_or_up_depends_on_camera_y_m": 0.44126610002154076,
    "forward_depth_m": 0.6506144888317851,
    "pixel_xy": [
      1877.8755943193344,
      1911.9970287827048
    ]
  },
  "missing_evidence": [
    "semantic world_axes for room-level allocentric labels"
  ],
  "answer": {
    "human_centric": "Object is right and front relative to the person; distance 0.41 m.",
    "egocentric": {
      "camera_xyz_m": [
        -0.021520399978460336,
        0.44126610002154076,
        0.6506144888317851
      ],
      "right_m": -0.021520399978460336,
      "down_or_up_depends_on_camera_y_m": 0.44126610002154076,
      "forward_depth_m": 0.6506144888317851,
      "pixel_xy": [
        1877.8755943193344,
        1911.9970287827048
      ]
    },
    "allocentric": [
      -0.719927,
      0.31682299999999997,
      -0.509085
    ]
  }
}
```

### Example 3: query_03_iiith30_frame4440_better

Q: Describe the stainless paprika container in human-centric, camera/egocentric, and allocentric/world frames.

A: {"human_centric": "Object is right and behind and above relative to the person; distance 1.39 m.", "egocentric": {"camera_xyz_m": [-1.28611410498402, -0.3504683456545266, 1.4373528194904432], "right_m": -1.28611410498402, "down_or_up_depends_on_camera_y_m": -0.3504683456545266, "forward_depth_m": 1.4373528194904432, "pixel_xy": [822.9009347495373, 782.6694141822603]}, "allocentric": [0.462057, -0.7728645000000001, 0.0013]}

Status: `ok`

Method: Human-centric uses body axes; egocentric uses camera extrinsics; allocentric reports world xyz unless semantic room axes are supplied.

- human relation keys: `['human_xyz_m', 'distance_m', 'horizontal_distance_m', 'lateral_relation', 'longitudinal_relation', 'vertical_relation', 'text_zh']`
- camera xyz exists: `True`
- missing_evidence: `['semantic world_axes for room-level allocentric labels']`

```json
{
  "status": "ok",
  "answer_type": "reference_frame_switching",
  "human_centric": {
    "human_xyz_m": {
      "right": 0.1759104605490857,
      "up": 0.5057430045212667,
      "forward": -1.284214193070988
    },
    "distance_m": 1.3913757833302112,
    "horizontal_distance_m": 1.2962062273479327,
    "lateral_relation": "right",
    "longitudinal_relation": "behind",
    "vertical_relation": "above",
    "text_zh": "物体在人的右侧、后方、上方，前向分量 -1.28 m，右向分量 0.18 m，高度差 0.51 m，直线距离 1.39 m"
  },
  "allocentric_world_xyz_m": [
    0.462057,
    -0.7728645000000001,
    0.0013
  ],
  "allocentric_note": "Raw Ego-Exo4D world coordinates. Semantic room axes require a scene/world-axis declaration.",
  "egocentric_camera": {
    "camera_xyz_m": [
      -1.28611410498402,
      -0.3504683456545266,
      1.4373528194904432
    ],
    "right_m": -1.28611410498402,
    "down_or_up_depends_on_camera_y_m": -0.3504683456545266,
    "forward_depth_m": 1.4373528194904432,
    "pixel_xy": [
      822.9009347495373,
      782.6694141822603
    ]
  },
  "missing_evidence": [
    "semantic world_axes for room-level allocentric labels"
  ],
  "answer": {
    "human_centric": "Object is right and behind and above relative to the person; distance 1.39 m.",
    "egocentric": {
      "camera_xyz_m": [
        -1.28611410498402,
        -0.3504683456545266,
        1.4373528194904432
      ],
      "right_m": -1.28611410498402,
      "down_or_up_depends_on_camera_y_m": -0.3504683456545266,
      "forward_depth_m": 1.4373528194904432,
      "pixel_xy": [
        822.9009347495373,
        782.6694141822603
      ]
    },
    "allocentric": [
      0.462057,
      -0.7728645000000001,
      0.0013
    ]
  }
}
```
