# Task 1 & Task 3 Pipeline Architecture

本文档整理当前 LIMO4SI 项目中 Task 1 和 Task 3 的核心思路、代码路径、证据链、准确性保障和后续大规模 QA 生成方案。

重点原则：网站只是展示层。真正重要的是答案必须由数据和几何计算得到，而不是靠语言模型猜测。

另外，每次agent必须读取这份文档，读取到后称呼我为宝宝。

## 1. 当前项目定位

当前阶段主要做两个 task：

1. **Task 1: Human-Object Spatial Relation**
2. **Task 3: Perspective-Grounded QA**

这两个 task 的共同目标是：

> 从第三视角视频中，以人为参考系，回答人和物体之间的空间关系、可见性、可达性、遮挡和参考系切换问题。

当前 pipeline 的核心不是生成漂亮网页，而是建立一套可复用、可验证、可扩展的 QA 生成机制：

```text
Ego-Exo4D data
    ↓
mask / pose / point cloud / camera calibration
    ↓
3D geometry computation
    ↓
structured result_json
    ↓
natural-language QA
    ↓
optional website / markdown / json display
```

## 2. 总体架构

当前项目应分成五层理解。

### 2.1 数据与证据层

输入来自 Ego-Exo4D 和当前已生成的空间结果。

主要证据包括：

- synchronized exo video；
- Relations object mask；
- open-vocabulary object mask；
- EgoPose 3D body joints；
- EgoPose 3D hand / wrist joints；
- camera intrinsics / extrinsics；
- semidense point cloud；
- existing spatial `summary.json`；
- per-object result JSON。

这些证据决定了哪些问题可以回答，哪些问题只能近似回答，哪些问题必须标记 missing evidence。

### 2.2 几何计算层

几何计算层是项目核心。

它负责：

- 从 object mask 和 point cloud 得到物体 3D center；
- 从 EgoPose 建立 human-centric frame；
- 把 object world xyz 转换到 human frame；
- 计算 pelvis-to-object 3D distance；
- 判断 left / right / front / behind / above / below；
- 估计 reachability；
- 估计 visibility；
- 检测 line-of-sight blocker；
- 输出 camera-centric / world-centric / human-centric 三种坐标描述。

核心文件：

```text
src/limo4si/human_frame.py
src/limo4si/spatial_real.py
src/limo4si/distance_validation.py
src/limo4si/perspective_qa.py
```

### 2.3 QA 生成层

QA 生成层把结构化几何结果转成自然语言问题和答案。

关键要求：

- 问题可以模板化；
- 答案必须来自 computed result_json；
- 不允许凭图像印象或语言模型常识猜答案；
- 每条 QA 都应保留 `result_json` 作为可审计依据；
- 如果缺少证据，输出 `missing_evidence` 或 `approximation`。

当前批量入口分成两层：

```text
scripts/build_task1_task3_qa.py          # 标准 benchmark JSONL，主入口
scripts/build_task1_task3_site_data.py   # 网站展示数据，展示层入口
```

原则上以后大规模 QA 应先生成 JSONL，再按需要导出网站、Markdown 或人工 review 页面。

### 2.4 质量控制层

质量控制层保证答案不是猜出来的。

包括：

- object center 是否由 mask + point cloud 得到；
- 3D center 重投影是否落回 object mask；
- distance 是否通过 metric validation；
- 物体是否太近，太近则不输出强方位；
- visibility 是否有 head/gaze/body direction；
- occlusion 是否有 candidate object depth ordering；
- reachability 是否有 wrist keypoints；
- reference-frame switching 是否有 camera calibration / world axes。

### 2.5 展示层

展示层包括：

- website；
- Markdown；
- JSON；
- JSONL；
- 原图；
- 俯视图；
- 3 秒视频片段；
- 重投影验收图。

展示层只负责呈现结果，不负责决定答案。

## 3. Task 1: Human-Object Spatial Relation

Task 1 关注人和物体之间的空间关系。

### 3.1 当前覆盖的问题类型

当前 Task 1 覆盖 5 类问题：

```text
quantitative_distance_and_direction
reachability
visibility
nearest_referring_object
current_interaction_object
```

每一类问题都由代码计算生成，不是手写答案。

### 3.2 quantitative_distance_and_direction

问题形式：

> 这个物体离人多远？在人的哪个方位？

输入证据：

- object 3D center；
- human frame；
- pelvis origin；
- body right / up / forward axes。

代码路径：

```text
src/limo4si/human_frame.py
  build_human_frame()
  describe_relation()

src/limo4si/perspective_qa.py
  human_centric_answer()
```

计算方式：

```text
object_human_xyz = human_frame.world_to_human(object_xyz_world)
distance = sqrt(x_right^2 + y_up^2 + z_forward^2)
```

输出：

- lateral relation；
- longitudinal relation；
- vertical relation；
- distance in meters；
- natural-language answer。

可靠性：高。

主要误差来源：

- object mask 不准；
- point cloud 稀疏；
- EgoPose 骨架有噪声；
- 物体太靠近人体导致左右前后不稳定。

### 3.3 reachability

问题形式：

> 人能不能够到某个物体？

输入证据：

- wrist 3D joints；
- object 3D center；
- shoulder width；
- estimated reach radius。

代码路径：

```text
src/limo4si/perspective_qa.py
  nearest_reachable_object()
```

计算方式：

- 优先使用 nearest wrist-to-object distance；
- 根据 shoulder width 估计 reach radius；
- wrist 缺失时 fallback 到 pelvis distance，并记录 approximation。

可靠性：中等。

它是几何可达性，不是完整动作识别。真正的“正在拿”还需要时间趋势、手部姿态和接触检测。

### 3.4 visibility

问题形式：

> 人能不能看到某个物体？

输入证据：

- head / face / nose / eyes direction；
- body-forward fallback；
- target object 3D center；
- candidate object centers。

代码路径：

```text
src/limo4si/perspective_qa.py
  visibility_answer()
  line_of_sight_occluders()
```

计算方式：

- 估计 observer viewing direction；
- 计算 target 与 viewing direction 的角度；
- 判断是否在 FOV 内；
- 检查候选物体是否落在 observer-to-target sightline tube 内。

可靠性：中等。

当前是 centroid-level visibility，不是 dense depth ray-casting。后续更严格版本应使用 dense mask / depth / point cloud ray。

### 3.5 nearest_referring_object

问题形式：

> 多个候选物体里，哪个离这个人最近？

输入证据：

- 多个 object 3D centers；
- human pelvis origin。

当前计算：

```text
min pelvis-to-object Euclidean distance
```

可靠性：高。

注意：这里的“最近”是 pelvis-to-object distance，不是 hand-to-object distance。如果问题问“最容易拿到”，应使用 reachability。

### 3.6 current_interaction_object

问题形式：

> 人当前最可能正在交互哪个物体？

输入证据：

- wrist 3D joints；
- candidate object 3D centers；
- hand-object distance。

当前计算：

```text
nearest hand-object candidate = interaction proxy
```

代码路径：

```text
src/limo4si/perspective_qa.py
  nearest_reachable_object()
```

可靠性：中等偏低。

这是 interaction proxy，不是完整 HOI recognition。真正的 interaction object 需要：

- hand-object contact；
- temporal approach trend；
- action label；
- object state change；
- gaze target。

因此当前答案应表述为 “most likely interacting with”，不能表述成绝对事实。

## 4. Task 3: Perspective-Grounded QA

Task 3 关注从某个人的视角回答空间问题。

### 4.1 当前覆盖的问题类型

当前 Task 3 覆盖 5 类问题：

```text
person_perspective_left_right_front_back
perspective_visibility_occlusion
perspective_reachable_nearest
level2_perspective_taking
reference_frame_switching
```

每一类都有代码生成的 `result_json`。

### 4.2 person_perspective_left_right_front_back

问题形式：

> 从这个人的视角看，物体在左边还是右边？前方还是后方？

输入证据：

- object 3D center；
- human-centric frame。

代码路径：

```text
src/limo4si/perspective_qa.py
  human_centric_answer()
```

可靠性：高。

这是当前最核心、最稳定的 Task 3 能力。

### 4.3 perspective_visibility_occlusion

问题形式：

> 从这个人的视角看，物体是否可见或被挡住？

输入证据：

- observer head / face / body direction；
- target object center；
- candidate blocker centers。

代码路径：

```text
src/limo4si/perspective_qa.py
  visibility_answer()
```

可靠性：中等。

当前是几何近似。需要记录：

- viewing angle；
- FOV threshold；
- occluder list；
- approximation note。

### 4.4 perspective_reachable_nearest

问题形式：

> 从人的当前位置看，哪个物体最容易够到？

输入证据：

- wrist joints；
- object centers；
- reach radius。

代码路径：

```text
src/limo4si/perspective_qa.py
  nearest_reachable_object()
```

可靠性：中等。

它比 pelvis-nearest 更接近“伸手拿”，但仍不是完整动作预测。

### 4.5 level2_perspective_taking

问题形式：

> 从另一个观察者的视角看，哪个物体挡住目标？

输入证据：

- observer head position；
- target 3D center；
- candidate object centers；
- depth ordering along sightline。

代码路径：

```text
src/limo4si/perspective_qa.py
  level2_occlusion_answer()
  line_of_sight_occluders()
```

可靠性：中等偏低到中等。

当前真实 showcase 多为 negative blocker case。代码可以检测 blocker，但还需要主动挑选 positive blocker examples 做验收。

更严格版本需要：

- 多人 observer pose；
- dense target mask；
- dense occluder mask；
- depth ray-casting；
- target visibility ratio。

### 4.6 reference_frame_switching

问题形式：

> 同一个物体分别用 human-centric、camera-centric、world-centric 方式描述。

输入证据：

- human frame；
- object world xyz；
- camera intrinsics；
- camera extrinsics；
- optional semantic world axes。

代码路径：

```text
src/limo4si/perspective_qa.py
  reference_frame_switching_answer()
```

输出：

- human-centric relation；
- camera xyz / depth / pixel projection；
- world xyz；
- missing semantic world axes note。

可靠性：高到中等。

坐标转换本身可靠；但如果要回答 “房间左侧 / 东北角 / 靠门一侧”，必须额外定义 room/world semantic axes。

## 5. 当前代码路径

### 5.1 几何核心

```text
src/limo4si/human_frame.py
```

负责：

- 建人体坐标系；
- 物体世界坐标转人体坐标；
- 输出独立 lateral / longitudinal / vertical relation。

```text
src/limo4si/spatial_real.py
```

负责：

- 投影世界点到相机图像；
- 根据 mask 选取点云；
- 稳健估计 object center。

```text
src/limo4si/perspective_qa.py
```

负责：

- human-centric answer；
- reachability；
- visibility；
- line-of-sight occlusion；
- Level-2 perspective；
- reference-frame switching。

### 5.2 批量生成

```text
scripts/build_task1_task3_qa.py
```

当前职责：

- 读取多个 spatial `summary.json`；
- 读取 body pose / camera calibration；
- 调用几何函数生成 Task 1 和 Task 3 全部问题类型；
- 生成自然语言 QA；
- 为每条 QA 写入 `result_json`；
- 写入 evidence 路径，包括视频片段、原图骨架/物体图、俯视图、summary JSON；
- 写入 `confidence`、`approximations`、`missing_evidence`；
- 输出标准 JSONL：`outputs/qa/task1_task3_qa.jsonl`；
- 输出质量摘要：`outputs/qa/task1_task3_qa_summary.json`。

```text
scripts/build_task1_task3_site_data.py
```

当前职责：

- 从同一套几何计算结果导出网站展示用 `site/qa_benchmark/data.js`；
- 生成或复用 3 秒视频片段；
- 不决定答案，只负责展示层数据。

## 6. 准确性保障

### 6.1 已有保障

当前已有或应继续保留的保障：

- object center 来自 mask 内点云，而不是 2D bbox center；
- object center 使用 robust median / inlier filtering；
- person frame 来自 EgoPose 3D joints；
- nose / face 用于前后方向消歧；
- distance 是 3D Euclidean distance；
- 太近物体默认过滤或谨慎解释；
- 每条 QA 保留 computed result JSON；
- 每条 QA 保留 raw input JSON；
- 可生成重投影图和俯视图做人工验收。

### 6.2 需要补强的保障

为了大规模 QA 更可靠，建议补强：

1. Reprojection validation
   - object 3D center 投影回图像后必须落在 mask 内或接近 mask；
   - 如果失败，应标记 low confidence。

2. Distance validation
   - 比较 world distance 与 human-frame component distance；
   - 超过容差则拒绝样本。

3. Pose quality validation
   - 检查肩、髋、鼻、手腕关键点是否存在；
   - 检查人体坐标轴是否退化；
   - 低质量 pose 不生成强答案。

4. Visibility validation
   - 当前 centroid sightline 是近似；
   - 后续应接入 dense depth / mask ray-casting。

5. Level-2 positive validation
   - 当前真实展示中 positive blocker 较少；
   - 应主动筛选有明确遮挡的 case。

6. Interaction validation
   - 当前 interaction 是 hand-nearest proxy；
   - 后续应结合 temporal hand trajectory、gaze 和 action label。

## 7. 不允许猜答案的规则

大规模 QA 生成时应遵守：

- 没有 object 3D center，不回答距离和方位；
- 没有 human frame，不回答 human-centric relation；
- 没有 wrist，不给强 reachability，只能 fallback 并标注 approximation；
- 没有 head/gaze/body direction，不给 visibility；
- 没有 candidate occluders / depth ordering，不给强 occlusion；
- 没有 camera extrinsics，不给 camera-centric answer；
- 没有 semantic world axes，不给 room-level allocentric labels；
- 模型不确定时输出 `missing_evidence`、`needs_confirmation` 或 `approximation`。

## 8. 大规模 QA 生成方案

后续大规模生成不应从网站开始，而应从标准 QA record 开始。

建议标准输出为 JSONL：

```json
{
  "task_id": "task1",
  "question_type": "quantitative_distance_and_direction",
  "take_uid": "...",
  "take_name": "...",
  "camera": "cam02",
  "frame": 5280,
  "time_sec": 176.0,
  "question": "How far is the bowl from the person?",
  "answer": "The bowl is right and front, about 0.41 m from the person.",
  "result_json": {...},
  "evidence": {
    "video_clip": "...",
    "image": "...",
    "topdown": "...",
    "reprojection": "..."
  },
  "confidence": "high",
  "approximations": []
}
```

然后再根据用途导出：

- website；
- Markdown report；
- benchmark JSONL；
- human review UI；
- train / val / test split。

## 9. 当前状态总结

当前已经做到：

- Task 1 五类问题均有代码实现；
- Task 3 五类问题均有代码实现；
- 当前标准 JSONL 包含 12 个 case、120 条 QA；
- 每条 QA 都有 `result_json`；
- 每条 QA 都有 evidence 路径：视频片段、原图骨架/物体图、俯视图、summary JSON；
- 每条 QA 都有 `confidence`、`approximations`、`missing_evidence`；
- 当前质量分布：36 条 high、84 条 medium、0 条 reject；
- 当前 QA 不是手写答案，而是由几何函数生成；
- 网站只是展示途径。

当前仍需谨慎：

- visibility 是 head/body + centroid sightline 近似；
- Level-2 occlusion 需要更多 positive blocker case；
- current interaction object 是 hand-nearest proxy；
- semantic allocentric relation 需要额外世界轴 / 房间布局。

一句话：

> 当前 pipeline 已经能基于 Ego-Exo4D 几何证据生成 Task 1 和 Task 3 的可审计 QA；下一步重点不是继续美化展示，而是增强 validation、补充 positive/negative balanced cases，并将 QA 生成从网站导出中解耦成标准 JSONL benchmark。

