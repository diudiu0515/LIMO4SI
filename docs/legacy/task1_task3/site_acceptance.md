# Legacy Task 1 & Task 3 Website Acceptance

当前网站展示已改为 Task + Question / Answer 形式，不再使用 richer / expanded task 标注。

## Counts

- Cases: 12
- Total QA: 90
- Task 1 · Human-Object Spatial Relation: 54 QA
- Task 3 · Perspective-Grounded QA: 36 QA

## How to view

```bash
cd /root/autodl-tmp/LIMO4SI/site/qa_benchmark
python -m http.server 8010 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8010
```

## Sample cases

### Task 1 · Human-Object Spatial Relation

1. `sfu0083_cam04_3450`

   Q: Where is the stainless bowl relative to the person, and how far is it?

   A: The stainless bowl is left and front and slightly above relative to the person. Its pelvis-to-object distance is 0.71 m.

2. `sfu0083_cam04_3450`

   Q: Where is the knife relative to the person, and how far is it?

   A: The knife is right and front and above relative to the person. Its pelvis-to-object distance is 0.87 m.

3. `sfu0083_cam04_3450`

   Q: Among the listed objects, which one is closest and reachable by the person?

   A: stainless bowl_0 is the nearest reachable candidate (0.27 m from nearest hand; reach radius 1.12 m).

4. `sfu0083_cam04_3450`

   Q: Can the person see the knife?

   A: Likely not visible: target is 100.8° from the viewing direction, outside the 110° field of view.

5. `sfu0101_cam05_5460`

   Q: Where is the white plate relative to the person, and how far is it?

   A: The white plate is left and behind and above relative to the person. Its pelvis-to-object distance is 2.89 m.

6. `sfu0101_cam05_5460`

   Q: Where is the cooking oil bottle relative to the person, and how far is it?

   A: The cooking oil bottle is left and front and above relative to the person. Its pelvis-to-object distance is 2.87 m.


### Task 3 · Perspective-Grounded QA

1. `sfu0083_cam04_3450`

   Q: From the person's own perspective, is the stainless bowl on the left/right and front/back side?

   A: Object is left and front and slightly_above relative to the person; distance 0.71 m.

2. `sfu0083_cam04_3450`

   Q: From the observer's perspective, which listed object blocks the stainless bowl?

   A: No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates.

3. `sfu0083_cam04_3450`

   Q: Describe the stainless bowl in human-centric, camera-centric, and world coordinates.

   A: {"human_centric": "Object is left and front and slightly_above relative to the person; distance 0.71 m.", "egocentric": {"camera_xyz_m": [-0.5057601645688642, 0.1398263256789743, 1.004722042163515], "right_m": -0.5057601645688642, "down_or_up_depends_on_camera_y_m": 0.1398263256789743, "forward_depth_m": 1.004722042163515, "pixel_xy": [1284.778279440259, 1257.0794384331261]}, "allocentric": [0.716446, -1.292799, -0.516879]}

4. `sfu0101_cam05_5460`

   Q: From the person's own perspective, is the white plate on the left/right and front/back side?

   A: Object is left and behind and above relative to the person; distance 2.89 m.

5. `sfu0101_cam05_5460`

   Q: From the observer's perspective, which listed object blocks the white plate?

   A: No listed object center lies on the observer-to-target sightline, so no blocker is detected among candidates.

6. `sfu0101_cam05_5460`

   Q: Describe the white plate in human-centric, camera-centric, and world coordinates.

   A: {"human_centric": "Object is left and behind and above relative to the person; distance 2.89 m.", "egocentric": {"camera_xyz_m": [0.675178909442854, -0.351167679832832, 1.345533942818335], "right_m": 0.675178909442854, "down_or_up_depends_on_camera_y_m": -0.351167679832832, "forward_depth_m": 1.345533942818335, "pixel_xy": [2549.1564204102756, 752.7691026520337]}, "allocentric": [-1.019257, -1.322883, -0.382841]}
