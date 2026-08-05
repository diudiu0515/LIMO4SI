# 未标注物体的空间关系

入口是 `scripts/ground_open_vocab_object.py`。它不依赖 Relations 物体 mask：

1. Grounding DINO 根据物体词找候选框；
2. SAM2 把候选框细化成像素 mask；
3. 中文解析器处理“第四层、右数第五个”等约束；
4. 半稠密点云投进选中 mask，以相机深度中位数和三维离群过滤得到物体中心；
5. 物体中心转换到以骨盆中点为原点的身体坐标，输出右、上、前三个独立分量。

层号按图像从上到下编号。“右数”暂按当前相机图像的 x 坐标排序，JSON
中的 `ordering_frame` 会明确写成 `image`。候选不足或不唯一时状态为
`needs_confirmation`，脚本只保存带编号的候选图，不猜测三维关系。

直线距离定义为“人体骨盆中点到物体稳健三维中心的欧氏距离”，不是二维
像素距离，也不是相机到物体的深度。水平距离是身体右向与前向分量的欧氏
范数。小于默认 0.60 m 的物体不输出方位标签。

示例：

```bash
.venv/bin/python scripts/ground_open_vocab_object.py \
  --take-uid e2b190bb-f8b2-43a7-b2da-b80f3708dcf3 \
  --camera cam05 --frame 9540 \
  --query '第四层架子上的右数第五个杯子'
```

如果语义检测漏检，可在候选图上确认像素框后加入
`--box X1 Y1 X2 Y2`。后续 SAM2 分割、三维定位、人体坐标与距离验收仍完全
相同。


## 当前验收结果

已下载本地模型：

- `models/grounding-dino-tiny`: text-to-box grounding
- `models/sam2.1-hiera-tiny`: box-prompted mask segmentation

开放词汇验收输出在 `outputs/spatial/open_vocab_validated/`：

- `sfu_cooking_007_3_cam01_frame10890.json`: 蚝油瓶，距离 1.52 m，eligible，右后方，略偏上；
- `sfu_cooking_008_3_cam04_frame6450.json`: salad cream bottle，距离 0.97 m，eligible，左后方，下方；
- `sfu_cooking_010_1_cam05_frame9540.json`: 杯子，距离 0.40 m，小于 0.60 m，正确过滤，不输出方位；
- `iiith_cooking_32_1_cam02_frame750.json`: salt container 检测未命中官方投影位置，保留为 needs-confirmation 反例。

每条成功三维定位的结果会同时生成：

- `*_candidates.jpg`: DINO 候选编号图；
- `*_reprojection.jpg`: 原图叠加选中 mask、人体骨架、身体前向和 3D 中心重投影；
- `*_topdown.jpg`: 人体坐标俯视图，灰圈是 0.60 m 过滤半径。

批量问答时，如果结构化描述能唯一解析，就直接输出空间关系；如果候选不足或不唯一，状态是 `needs_confirmation`，由上层 AI 或人工确认 `candidate_index` 后再计算三维关系，不在视觉证据不足时猜测。
