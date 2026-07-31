# LIMO4SI\-GYJ\-WORKFLOW

# Human Spatial Intent QA 项目交接与下一步执行方案 

## 项目目标

我们现在在做一个 **Human Spatial Intent / Ego\-Exo Spatial\-Temporal QA Benchmark** 的早期探索。

这个项目不是普通图像 QA，而是希望构造更接近下面几类能力的 benchmark：

1. **Ego\-Exo Correspondence**
同一个物体在第一视角和第三视角中如何对应。

2. **Visibility / Occlusion Reasoning**
某个物体在 ego wearer 的视角中是否可见；第三视角能看到但第一视角不一定能看到的物体是什么。

3. **Temporal Grounding**
给定一个动作、gaze event、object interaction 或自然语言描述，在视频中定位对应的时间点或时间窗口。

4. **Gaze\-Grounded Attention QA**
利用 Ego\-Exo4D 官方 gaze 数据，判断 ego wearer 在某个时刻可能正在看哪里、看哪个物体。

5. **Perspective\-Grounded QA**
从人的视角出发，判断物体在他的左边、右边、前方、后方，或者是否被遮挡。

6. **Intention / Action\-Goal Spatial Reasoning**
根据动作、gaze、手部运动、物体位置和 temporal context，判断人可能正在操作什么、即将操作什么、注意力集中在哪里。

当前我们已经完成了 Ego\-Exo4D Relations 数据上的一个 pilot pipeline，但这只是第一阶段。下一步最重要的是：**接入 Ego\-Exo4D gaze 数据和 temporal grounding 信号**。

---

## 当前已经完成的工作

项目目录：

```Bash
/home/gaoyajing/view/gaoyajing/human_spatial_intent_qa
```

Ego\-Exo4D 数据目录：

```Bash
/home/gaoyajing/view/gaoyajing/egoexo4d
```

当前确认存在的数据：

```Bash
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/relations_train.json
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/relations_val.json
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/relations_test.json
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/atomic_descriptions_train.json
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/atomic_descriptions_val.json
/home/gaoyajing/view/gaoyajing/egoexo4d/takes.json
/home/gaoyajing/view/gaoyajing/egoexo4d/takes/
```

已经下载的视频：

```Plain Text
20 个 pilot take
174 个 mp4
```

视频路径格式：

```Bash
/home/gaoyajing/view/gaoyajing/egoexo4d/takes/<take_name>/frame_aligned_videos/downscaled/448/*.mp4
```

注意：视频目录是按 `take_name` 命名，不是按 `take_uid` 命名。

视频命名约定：

```Plain Text
cam01.mp4, cam02.mp4, cam03.mp4, cam04.mp4, cam05.mp4
```

是 exo / 第三视角视频。

```Plain Text
aria*.mp4
```

是 ego / Aria 第一视角视频。

```Plain Text
ego_preview.mp4
```

只是预览视频，一般不要当正式相机流使用。

---

## 当前已有脚本

目前 Ego\-Exo4D 相关脚本是：

```Bash
scripts/19_inspect_egoexo_download.py
scripts/20_list_egoexo_relation_takes.py
scripts/21_check_pilot_videos.py
scripts/22_inspect_one_relation_take.py
scripts/23_extract_demo_frames.py
scripts/24_build_relation_qa_candidates.py
scripts/25_audit_qa_candidates.py
scripts/26_extract_qa_evidence_frames.py
scripts/27_build_qa_review_html.py
```

它们的作用是：

### `19_inspect_egoexo_download.py`

检查 Ego\-Exo4D 数据根目录里已经下载了哪些 annotation 和 metadata。

### `20_list_egoexo_relation_takes.py`

从 relations annotation 里筛选有 relation/mask 标注的 take，生成 pilot take 候选。

### `21_check_pilot_videos.py`

检查 20 个 pilot take 是否都有 ego/exo 视频。

目前结果：

```Plain Text
Total pilot takes: 20
Existing video directories: 20
With ego video: 20
With exo video: 20
With both ego and exo: 20
```

### `22_inspect_one_relation_take.py`

解析一个 pilot take 的 relation annotation schema。

当前默认 take 是：

```Plain Text
take_name = sfu_cooking_005_2
take_uid  = 39feb026-bcf8-4a61-89ad-9f5566c2c7bf
```

生成：

```Bash
data_intermediate/egoexo_relation_records_sfu_cooking_005_2.jsonl
```

数量：

```Plain Text
6876 records
```

### `23_extract_demo_frames.py`

从 `sfu_cooking_005_2` 里抽取 demo frames，并生成 contact sheet。

### `24_build_relation_qa_candidates.py`

从 flattened relation records 生成 QA candidates。

当前结果：

```Plain Text
Input relation rows: 6876
QA candidate rows: 9126
Complete QA candidates: 9105
Incomplete QA candidates: 21
```

题型分布：

```Plain Text
visibility: 6876
correspondence: 2170
attention/action proxy: 60
relation: 20
```

注意：`relation` 题全部 incomplete，因为当前 relation records 里缺少 `object_b` 和 `relation_type`。

### `25_audit_qa_candidates.py`

审查 QA candidate 的分布、重复、完整性。

当前 audit 结果：

```Plain Text
Total candidates: 9126
Complete candidates: 9105
Incomplete candidates: 21
Unique objects: 31
Unique views/cameras: 3
Unique frames/timestamps: 293
Duplicate groups: 0
Confidence distribution: 9045 high, 81 low
```

### `26_extract_qa_evidence_frames.py`

为 30 个 visual\-audit candidates 抽取证据帧。

当前结果：

```Plain Text
Selected visual-audit candidates: 30
Evidence frame folders generated: 30
Extracted images: 50
Contact sheets: 20
Fallback timestamp extractions: 0
Exact frame extraction was used for all selected evidence items.
```

### `27_build_qa_review_html.py`

生成静态人工审查网页：

```Bash
debug/qa_review_sfu_cooking_005_2/index.html
```

当前结果：

```Plain Text
Candidates included: 120
Evidence images linked: 50
Contact sheets linked: 20
Candidates with evidence images/contact sheets: 30
Candidates with missing evidence folders/images: 90
```

90 个 missing 是正常的，因为目前只给 30 个样本抽了 evidence frames，但 review HTML 包含了 120 个 sampled candidates。

---

## Temporal Grounding 和 Gaze 是下一阶段主线

之前的下一步主要是：

```Plain Text
人工审查当前 QA
接入 atomic descriptions
从 relations 派生 2D spatial QA
检查 perspective-grounding signals
```

现在要改成：

```Plain Text
接入 Ego-Exo4D gaze 数据
解析 atomic descriptions
把 gaze / atomic / relation / video frame 对齐
构造 temporal-grounded QA
构造 gaze-grounded QA
再做 2D spatial QA
最后检查 perspective-grounded QA 可行性
```

## Task：如果 Gaze 未下载，只补下载 Gaze / MPS 小文件

## 目标

只下载 20 个 pilot take 的 gaze 相关文件，不要重新下载全量视频。

## 操作原则

```Plain Text
不要下载 full-resolution video
不要重新下载已有 448p 视频
不要下载全量 Ego-Exo4D
只补 gaze / MPS / eye_gaze 相关文件
```

## 建议命令

先生成 UIDs：

```Bash
cd /home/gaoyajing/view/gaoyajing/human_spatial_intent_qa

UIDS=$(paste -sd' ' data_intermediate/egoexo_pilot_take_uids.txt)
```

然后让 Codex 或终端先查看 downloader 支持哪些 parts：

```Bash
egoexo --help
```

再尝试 gaze 相关 part。候选命令包括：

```Bash
egoexo -o /home/gaoyajing/view/gaoyajing/egoexo4d \
  --release v2 \
  --parts take_eye_gaze \
  --uids $UIDS
```

如果 `take_eye_gaze` 不存在，不要乱猜下载大文件。记录报错，再检查 downloader manifest 或官方 docs 中 gaze 对应的 part 名称。

也可以尝试查看当前 package 的 part metadata，例如只运行 help / dry\-run 级别命令，不要直接下载全量。

## 输出

写日志：

```Bash
debug/egoexo_gaze_download_attempt.md
```

记录：

```Plain Text
运行了什么命令
命令是否成功
下载了多少文件
下载大小
哪些 take 成功
哪些 take 失败
如果 part name 不支持，报错是什么
```

---

## Task：解析 Gaze Records，并对齐到视频 timestamp/frame

## 目标

把 Ego\-Exo4D gaze CSV 转成统一 JSONL，方便后续和 relation mask / atomic descriptions 对齐。

## 新增脚本

创建：

```Bash
scripts/29_parse_gaze_records.py
```

## 输入

```Bash
data_intermediate/egoexo_pilot_gaze_inventory.csv
/home/gaoyajing/view/gaoyajing/egoexo4d/takes/<take_name>/eye_gaze/*.csv
```

## 输出

```Bash
data_intermediate/egoexo_gaze_records.jsonl
data_intermediate/egoexo_gaze_records_sfu_cooking_005_2.jsonl
debug/egoexo_gaze_records_summary.md
```

## 统一字段

每条 gaze record 至少包含：

```Plain Text
take_uid
take_name
gaze_source: general / personalized
gaze_space: 2d_ego / 3d_ray
timestamp_ns
timestamp_sec
frame_number_estimated
x_2d
y_2d
yaw
pitch
depth
valid
source_file
notes
```

如果字段不存在，就写 null，不要编造。

## 对齐逻辑

优先级：

```Plain Text
1. 如果 gaze CSV 里有 frame_number，就直接用。
2. 如果有 timestamp_ns 或 timestamp_sec，就用视频 fps 估计 frame number。
3. 如果采样频率低于视频 fps，记录 nearest-frame mapping，不要假设每帧都有 gaze。
4. 如果 timestamp 单位不明确，在 report 里标记为 unclear，不要硬猜。
```

## 验收标准

报告中回答：

```Plain Text
1. 总共解析多少 gaze records？
2. 每个 take 多少 gaze records？
3. general / personalized 各多少？
4. 2D gaze / 3D gaze 各多少？
5. timestamp 是否连续？
6. gaze sampling fps 估计是多少？
7. 是否覆盖完整视频？
8. 是否能和当前 relation frame records 对齐？
```

---

## Task：接入 Atomic Descriptions，做 Temporal Grounding 基础表

## 目标

Atomic Descriptions 是 temporal grounding 的关键，因为它们是带 timestamp 的动作文本。

这一步不是直接生成 QA，而是先建立：

```Plain Text
atomic action timestamp
gaze timestamp
relation object/mask timestamp
video frames
```

之间的对齐基础。

## 新增脚本

创建：

```Bash
scripts/30_inspect_atomic_descriptions.py
scripts/31_align_atomic_gaze_relation.py
```

## 输入

```Bash
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/atomic_descriptions_train.json
/home/gaoyajing/view/gaoyajing/egoexo4d/annotations/atomic_descriptions_val.json
data_intermediate/egoexo_gaze_records_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_relation_records_sfu_cooking_005_2.jsonl
```

## 输出

```Bash
data_intermediate/atomic_records_sfu_cooking_005_2.jsonl
data_intermediate/atomic_records_sfu_cooking_005_2.csv
data_intermediate/atomic_gaze_relation_aligned_sfu_cooking_005_2.jsonl
debug/atomic_schema_sfu_cooking_005_2.md
debug/atomic_gaze_relation_alignment_sfu_cooking_005_2.md
```

## `30_inspect_atomic_descriptions.py` 要做什么

1. 读取 atomic descriptions train/val。

2. 不要假设 schema，先打印 top\-level keys 和 sample records。

3. 找 `sfu_cooking_005_2` 或 `take_uid=39feb026-bcf8-4a61-89ad-9f5566c2c7bf` 对应 records。

4. 输出字段统计：

    - timestamp

    - text / description

    - subject

    - narration\_subject

    - ego\_visible

    - best\_exo

    - camera id

    - unsure

5. 保存 atomic records JSONL/CSV 和 schema report。

## `31_align_atomic_gaze_relation.py` 要做什么

对每条 atomic description：

1. 取 atomic timestamp。

2. 找 timestamp 附近窗口内的 gaze records。

3. 找同一窗口内的 relation object/mask records。

4. 找同一窗口内的 ego/exo video frames。

5. 输出一个 temporal alignment record。

时间窗口先跑三个：

```Plain Text
±0.5s
±1.0s
±2.0s
```

## Temporal alignment 输出字段

```Plain Text
take_uid
take_name
atomic_timestamp
atomic_text
window_size
window_start
window_end
matched_gaze_records
matched_relation_objects
matched_ego_views
matched_exo_views
ego_visible
best_exo_camera
confidence
missing_fields
notes
```

## 验收标准

报告中回答：

```Plain Text
1. 这个 take 有多少 atomic descriptions？
2. 每条 description 有没有 timestamp？
3. 有没有 ego_visible？
4. 有没有 best_exo camera？
5. 文本里是否出现具体 object/action？
6. 多少 atomic descriptions 能匹配到 gaze？
7. 多少能匹配到 relation object/mask？
8. 哪个 time window 效果最好？
9. 能不能用于 temporal grounding QA？
```

---

## Task：Gaze 点匹配 Ego Object Mask / BBox

## 目标

利用 gaze 2D 坐标判断 camera wearer 在某个时刻可能看向哪个 object。

这是 human spatial intent 的关键任务之一。

## 新增脚本

创建：

```Bash
scripts/32_match_gaze_to_ego_objects.py
```

## 输入

```Bash
data_intermediate/egoexo_gaze_records_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_relation_records_sfu_cooking_005_2.jsonl
```

## 输出

```Bash
data_intermediate/egoexo_gaze_object_matches_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_gaze_object_matches_sfu_cooking_005_2.csv
debug/egoexo_gaze_object_matches_sfu_cooking_005_2.md
```

## 核心逻辑

对每个 gaze 点：

1. 找同一 take、同一 ego view、时间最近的 relation mask/bbox record。

2. 如果 gaze 点落在 object mask 内：

    - `match_type = inside_mask`

    - `confidence = high`

3. 如果暂时没有 mask decode，或者先不 decode mask，则用 bbox：

    - gaze 点落在 bbox 内：`inside_bbox`

    - `confidence = medium`

4. 如果 gaze 点不在任何 object 内，但离某个 object center 很近：

    - `match_type = nearest_object`

    - `confidence = low`

5. 如果没有候选 object：

    - `match_type = unmatched`

## 输出字段

```Plain Text
take_uid
take_name
timestamp_sec
frame_number
gaze_x
gaze_y
ego_view
matched_object_id
matched_object_name
match_type
distance_to_object_center
inside_mask
inside_bbox
confidence
candidate_objects
notes
```

## 注意

不要把 gaze 附近的 object 直接当作强 gaze label。

强标签只给：

```Plain Text
inside_mask
```

中等置信给：

```Plain Text
inside_bbox
```

低置信给：

```Plain Text
nearest_object
```

## 验收标准

报告中统计：

```Plain Text
gaze records
matched records
inside_mask matches
inside_bbox matches
nearest_object matches
unmatched records
unique matched objects
examples
failure cases
```

---

# Task：把 Ego Gaze Target 映射到 Exo View

## 目标

利用 Relations 的 same\-object ego\-exo correspondence，把 ego gaze 看到的 object 映射到 exo view 里。

这一步可以构造真正的 ego\-exo gaze/attention correspondence QA。

## 新增脚本

创建：

```Bash
scripts/33_map_gaze_target_to_exo.py
```

## 输入

```Bash
data_intermediate/egoexo_gaze_object_matches_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_relation_records_sfu_cooking_005_2.jsonl
```

## 输出

```Bash
data_intermediate/egoexo_gaze_exo_correspondence_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_gaze_exo_correspondence_sfu_cooking_005_2.csv
debug/egoexo_gaze_exo_correspondence_sfu_cooking_005_2.md
```

## 核心逻辑

对每个 matched gaze object：

1. 获取：

    - object\_id / object\_name

    - timestamp / frame

    - ego view

2. 在同一 timestamp/frame 附近找 exo views 中同一 object 的 mask/bbox。

3. 如果找到：

    - 生成 exo correspondence record

    - 包含 exo camera、exo bbox/mask、frame

4. 如果找不到：

    - 标记为 ego\-only gaze target

    - 这可能是 visibility gap candidate，但不能直接当作 exo 不可见。

## QA 潜力

可以生成：

```Plain Text
Question: Which object is the ego wearer looking at?
Answer: <object_name>

Question: In the exocentric view, which object corresponds to the ego wearer's gaze target?
Answer: <object_name>

Question: Is the object being looked at by the ego wearer visible in cam04?
Answer: yes / unknown
```

注意：如果 exo 没有对应 mask，不能直接写 no，除非确认 annotation 是 exhaustive。

---

## Task：构造 Temporal\-Grounded QA 和 Gaze\-Grounded QA

## 目标

把 atomic / gaze / relation alignment 变成 QA candidates。

## 新增脚本

创建：

```Bash
scripts/34_build_temporal_gaze_qa_candidates.py
```

## 输入

```Bash
data_intermediate/atomic_gaze_relation_aligned_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_gaze_object_matches_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_gaze_exo_correspondence_sfu_cooking_005_2.jsonl
```

## 输出

```Bash
data_intermediate/temporal_gaze_qa_sfu_cooking_005_2.jsonl
data_intermediate/temporal_gaze_qa_sfu_cooking_005_2.csv
debug/temporal_gaze_qa_sfu_cooking_005_2.md
```

## QA 类型

### temporal\_action\_grounding

```Plain Text
Question: When does the person perform the action "<atomic_text>"?
Answer: timestamp or temporal window
```

### action\_description

```Plain Text
Question: What is the person doing at this moment?
Answer: <atomic description text>
```

### gaze\_target

```Plain Text
Question: Which object is the ego wearer looking at?
Answer: <matched_object_name>
```

只对 `inside_mask` / `inside_bbox` 生成。

### gaze\_temporal\_grounding

```Plain Text
Question: At what time does the ego wearer look at <object>?
Answer: timestamp/window
```

### action\_object\_attention

```Plain Text
Question: While doing "<atomic_text>", which object is the ego wearer looking at?
Answer: <matched gaze object>
```

只有 atomic\-gaze alignment 成功时生成。

### gaze\_exo\_correspondence

```Plain Text
Question: In the exocentric view, which object corresponds to the ego wearer's gaze target?
Answer: <object_name>
```

### best\_exo\_evidence\_view

```Plain Text
Question: Which exocentric camera best shows the action "<atomic_text>"?
Answer: <camera id>
```

只有 atomic record 有 `best_exo` 字段时生成。

### ego\_exo\_visibility\_during\_action

```Plain Text
Question: During "<atomic_text>", is the gazed object visible in both ego and exo views?
Answer: yes / unknown
```

没有 exo mask 时不要直接写 no。

## 每条 QA 必须包含

```Plain Text
take_uid
take_name
question
answer
question_type
temporal_window
timestamp
source_signals: atomic / gaze / relation / video
confidence
evidence_fields
missing_fields
notes
```

## 置信度

```Plain Text
high:
  atomic timestamp exists
  gaze point inside object mask
  object has exo correspondence

medium:
  atomic timestamp exists
  gaze point inside bbox
  object has exo correspondence

low:
  nearest-object gaze match
  weak action-object association
```

## 验收标准

报告中统计：

```Plain Text
input atomic records
input gaze matches
input exo correspondences
generated QA candidates
complete candidates
incomplete candidates
question_type distribution
confidence distribution
example QA candidates
limitations
```

---

## Task：抽取 Temporal Evidence Frames / Clips

## 目标

Temporal grounding 不能只抽一帧，需要抽取动作或 gaze event 附近的一小段 evidence。

## 新增脚本

创建：

```Bash
scripts/35_extract_temporal_gaze_evidence.py
```

## 输入

```Bash
data_intermediate/temporal_gaze_qa_sfu_cooking_005_2.jsonl
```

## 输出

```Bash
debug/temporal_gaze_evidence/sfu_cooking_005_2/
debug/temporal_gaze_evidence_sfu_cooking_005_2.md
```

## 抽取内容

对每条 selected QA：

1. 抽取 timestamp 前后窗口：

```Plain Text
t - 1s
t
t + 1s
```

2. 对 ego video 抽帧。

3. 对 best exo camera 或 cam04 抽帧。

4. 如果 ffmpeg 支持，额外生成短 clip：

```Plain Text
[t - 2s, t + 2s]
```

5. 生成 contact sheet。

## 每个样本文件夹

```Bash
debug/temporal_gaze_evidence/sfu_cooking_005_2/candidate_0001/
```

包含：

```Plain Text
metadata.json
ego_t_minus_1.jpg
ego_t.jpg
ego_t_plus_1.jpg
exo_t_minus_1.jpg
exo_t.jpg
exo_t_plus_1.jpg
contact_sheet.jpg
optional_clip_ego.mp4
optional_clip_exo.mp4
```

## 验收标准

报告中统计：

```Plain Text
selected QA count
generated evidence folders
extracted frames
generated clips
contact sheets
missing videos
missing timestamps
fallback cases
```

---

## Task：升级 Review HTML

## 目标

把原来的 review HTML 升级成 temporal \+ gaze evidence review。

## 新增脚本

创建：

```Bash
scripts/36_build_temporal_gaze_review_html.py
```

## 输出

```Bash
debug/temporal_gaze_review_sfu_cooking_005_2/index.html
debug/temporal_gaze_review_sfu_cooking_005_2/review_manifest.jsonl
debug/temporal_gaze_review_sfu_cooking_005_2/review_instructions.md
```

## 页面展示

每个 QA 显示：

```Plain Text
question
answer
question_type
temporal_window
gaze timestamp
gaze point x/y
matched object
match_type
confidence
ego evidence frames
exo evidence frames
contact sheet
source signals
```

人工审查字段：

```Plain Text
temporal_grounding_correct: yes / no / unclear
gaze_target_correct: yes / no / unclear
answerable_from_video: yes / no / unclear
object_match_quality: good / medium / bad
keep_for_benchmark: yes / no / maybe
reviewer_notes
```

## 审查重点

优先审查：

```Plain Text
gaze_target
gaze_exo_correspondence
action_object_attention
temporal_action_grounding
```

其次审查：

```Plain Text
visibility
correspondence
2D spatial
```

暂时不重点审查：

```Plain Text
incomplete relation
person-centric perspective QA
```

---

## Task K：从 Relations 派生 2D Spatial QA

## 目标

Relations 里没有直接的 `relation_type`，但有 bbox/mask 坐标。我们可以自己派生 image\-plane spatial relations。

先只做：

```Plain Text
left / right
above / below
closer / farther in image plane
overlap
```

不要做：

```Plain Text
physically left
person's left
front / behind
on / holding / using
```

因为这些需要 3D/person\-centric/action 信息。

## 新增脚本

创建：

```Bash
scripts/37_build_2d_spatial_relation_qa.py
```

## 输入

```Bash
data_intermediate/egoexo_relation_records_sfu_cooking_005_2.jsonl
```

## 输出

```Bash
data_intermediate/egoexo_2d_spatial_qa_sfu_cooking_005_2.jsonl
data_intermediate/egoexo_2d_spatial_qa_sfu_cooking_005_2.csv
debug/egoexo_2d_spatial_qa_sfu_cooking_005_2.md
```

## 核心逻辑

对同一个：

```Plain Text
take_name
view/camera
frame/timestamp
```

下的多个 object，计算每个 object 的 2D center。

如果有 bbox：

```Plain Text
cx = x + width / 2
cy = y + height / 2
```

如果有 mask，可以 decode mask 后算 mask centroid。第一版可以先用 bbox 或已有 mask bbox，不要强行复杂 decode。

对每两个 object A/B：

### left/right

```Plain Text
if cx_A < cx_B:
    A is left of B in this image
else:
    A is right of B in this image
```

### above/below

```Plain Text
if cy_A < cy_B:
    A is above B in this image
else:
    A is below B in this image
```

### distance

```Plain Text
dist = sqrt((cx_A - cx_B)^2 + (cy_A - cy_B)^2)
normalized_dist = dist / image_diagonal
```

### overlap

如果有 bbox，可以算 IoU：

```Plain Text
IoU = area(intersection) / area(union)
```

如果 IoU \> threshold，生成 overlap QA。

## QA 写法必须保守

正确写法：

```Plain Text
In the cam04 image at frame 1234, is the bowl to the left or right of the knife?
```

错误写法：

```Plain Text
Is the bowl physically to the left of the knife?
```

因为我们现在只能保证 2D image\-plane relation，不保证真实 3D 世界关系。

---

## Task：Perspective\-Grounded QA 可行性检查

## 目标

这是更高级的方向：从人的视角判断 left/right/front/behind。

当前不能直接做，需要先查数据里有没有 human/person pose、head orientation、Aria wearer 位姿、camera calibration。

## 新增脚本

创建：

```Bash
scripts/38_check_perspective_grounding_signals.py
```

## 检查内容

在 Ego\-Exo4D 根目录里搜索以下信息：

```Plain Text
camera intrinsics
camera extrinsics
calibration
pose
body
hand
ego pose
aria trajectory
head
person bbox
```

不要加载巨大文件。只做文件名级别和小 sample 检查。

## 输出

```Bash
debug/perspective_grounding_signal_check.md
```

## 报告回答

```Plain Text
1. 当前本地有没有 camera calibration？
2. 当前本地有没有 human/person bbox？
3. 当前本地有没有 body pose / head pose？
4. 当前本地有没有 ego wearer trajectory？
5. 当前本地有没有 object 2D/3D position？
6. 当前能不能构造 human-centric left/right/front/behind？
7. 还需要下载哪些 Ego-Exo4D annotation parts？
```

## 当前预期结论

大概率是：

```Plain Text
当前本地 relations + atomic descriptions + gaze 可以支持 gaze-grounded / temporal-grounded QA。
但 person-centric left/right/front/behind 仍然需要额外 pose / orientation / calibration / trajectory。
可以先做 image-plane left/right，不要声称 person-centric perspective。
```

---

## 每一步完成后需要汇报什么

每完成一个脚本，请汇报：

```Plain Text
1. 新增了哪些文件
2. 运行了哪些命令
3. 生成了哪些输出
4. 数量统计是什么
5. 发现了什么 limitation
6. 下一步建议是什么
```



---

## 最终希望她帮忙产出的结论

这一轮结束后，我们希望得到一份清晰结论：

```Plain Text
20 个 pilot take 里有多少个有 gaze 数据？
Gaze 能不能和 448p ego video frame 对齐？
Gaze 点能不能匹配到 ego object mask/bbox？
Ego gaze target 能不能映射到 exo view？
Atomic Descriptions 能不能支持 temporal grounding query？
Temporal evidence clip 是否能支持 QA？
2D spatial relation QA 能不能从 bbox/mask 派生？
目前是否有足够信号做 human-centric perspective QA？
如果要做 gaze/attention，还需要什么额外数据？
```



# 2026\.7\.9

### 用了哪些 Ego\-Exo4D 数据



数据根目录：



/home/gaoyajing/view/gaoyajing/egoexo4d



已有数据规模：



- takes\.json: 5035 个 take，约 41\.2 MB

- relation annotation files: 3 个

    - relations\_train\.json: 1\.1 GB

    - relations\_val\.json: 363\.5 MB

    - relations\_test\.json: 461\.6 MB

        

- atomic description files: 2 个

    - atomic\_descriptions\_train\.json: 125\.9 MB

    - atomic\_descriptions\_val\.json: 32\.4 MB

        

- Ego\-Exo4D root 总大小：约 16 GB

- 已下载视频：174 个视频文件

- 已选 pilot takes：20

- 20 个 pilot take 都有 ego 和 exo 视频。

    

    视频结构是：

    

    egoexo4d/takes/\<take\_name\>/frame\_aligned\_videos/downscaled/448/\*\.mp4

    

    其中：

    

- aria\*\.mp4 是 ego / Aria 第一视角

- cam\*\.mp4 是 exo / 第三视角

- ego\_preview\.mp4 只是 preview，不当作普通 camera stream

    

    ### pipeline 是怎么建的

    

1. 数据下载检查k 

scripts/19\_inspect\_egoexo\_download\.py



检查 takes\.json、relation annotations、atomic descriptions 是否存在，统计文件大小和 take 数。



2. relation take 筛选

scripts/20\_list\_egoexo\_relation\_takes\.py



从 relation annotations 里找有 object masks、ego view、exo view、并且同一物体同时出现在 ego/exo 的 take。

找到 relation take candidates：1165 个。

最后选了 20 个 pilot take。



3. pilot 视频检查

scripts/21\_check\_pilot\_videos\.py



检查这 20 个 pilot take 的视频目录。结果：

- 20/20 有视频目录

- 20/20 有 ego 视频

- 20/20 有 exo 视频

- 20/20 同时有 ego 和 exo

    

4. 单个 take schema inspection

scripts/22\_inspect\_one\_relation\_take\.py



目标 take：

- take\_uid: 39feb026\-bcf8\-4a61\-89ad\-9f5566c2c7bf

- take\_name: sfu\_cooking\_005\_2

- task: Cooking

    

    对这个 take 展开 relation object masks，得到：

- flattened relation records: 6876 行

- objects: 31

- views: aria01\_214\-1, cam02, cam04

    

5. demo 抽帧

scripts/23\_extract\_demo\_frames\.py



从 sfu\_cooking\_005\_2 抽了 ego/exo 示例帧，并生成 contact sheet。



6. QA candidate 生成

scripts/24\_build\_relation\_qa\_candidates\.py



从 flattened relation records 生成 QA candidates：

- total QA candidates: 9126

- complete: 9105

- incomplete: 21

    

    分布：

- visibility: 6876

- correspondence: 2170

- attention/action proxy: 60

- relation: 20

    

7. QA audit 和 sample

scripts/25\_audit\_qa\_candidates\.py



统计 QA 候选质量、重复、confidence、question type，并采样：

- sampled candidates: 120

- duplicate groups: 0

- unique objects: 31

- unique views/cameras: 3

- unique frames/timestamps: 293

    

8. QA evidence frame extraction

scripts/26\_extract\_qa\_evidence\_frames\.py







- visibility QA 可以作为 positive\-only candidate。

- ego/exo correspondence 可以作为 “同一个 object\_name 在 ego/exo 同帧出现” 的候选任务。

- relation QA 暂时不能作为 benchmark valid item，因为缺：

    - object\_b

    - relation\_type

        

- attention/action proxy 只能低置信度使用，因为 mask 不能证明 gaze、hand contact、action 或 intent。

- 当前没有可靠 negative examples，不能把“没有 mask”直接当作“不可见”。

    ### todo

    3. Perspective\-Grounded QA

        

        这个需要的是 human\-centric reference frame，不只是 object mask。

        

        需要补的信号：

        

    - 人的位置

    - 人的朝向 / head pose / body orientation

    - object 的 2D/3D 位置

    - camera calibration 或 multi\-view geometry

    - ego view visibility

    - occlusion / depth / mask overlap

        

        当前 relation correspondence 可以帮你找到“同一个 object 在 ego/exo 都可见”的样本，但不能直接回答“在他的左边还是右边”。

        

        下一步应该做：

        

    - 从 exo view 定位 person \+ object

    - 用 Aria/ego wearer 或 pose 推断 person facing direction

    - 把 object 坐标投到 person\-centric frame

    - 构造 left/right/front/behind/visible/occluded QA

        

    4. Intention \& Action\-Goal Spatial Reasoning

        

        这个不是 relation mask 能解决的。需要：

        

    - hand trajectory

    - object proximity

    - atomic descriptions

    - action labels

    - future trajectory / forecasting labels

    - maybe contact / manipulation annotations

        

        当前 mask correspondence 只能给候选 object set，比如“哪些 object 在 ego/exo 里存在”。

        但“他即将拿哪个杯子”需要 temporal signal。

        

        下一步应该结合：

        

    - atomic\_descriptions\_train/val\.json

    - ego video temporal window

    - hand/object proximity if available

    - maybe action narration or forecasting annotations

        

    5. Gaze, Attention \& Visibility Reasoning

        

        这个方向最接近我们现在做的东西，但也不能直接等价。

        

        当前可做：

        

    - positive visibility: object 有 ego mask，所以可见

    - ego\-exo visibility gap candidate: exo 里有 object，ego 里同帧没有 object，可能是 ego 看不到

        

        但要小心：没有 mask 不一定等于不可见，除非确认 annotation 是 exhaustive。

        

        真正 gaze/attention 需要：

        

    - gaze target labels

    - eye gaze / head gaze

    - hand\-object interaction

    - joint attention labels

    - VideoAttentionTarget / GazeFollow / GIMO / Gaze360 这类数据

        

        所以当前 relation correspondence 可以作为 visibility/correspondence seed，但不能直接当 gaze label。

# 2026\.6\.25

1. 先做

1. Interaction Object

Q: Which object is the person interacting with?

A: bicycle



2. Left / Right Relation

Q: Is the bicycle to the left or right of the person?

A: right



3. Nearest Object

Q: Which object is closest to the person?

A: bicycle



4. Reachability Proxy

Q: Can the person likely reach the cup?

A: yes / no

Part 0\. HICO\-100

基础人\-物空间关系 sanity check



Part 1\. VAT v0

第三视角 gaze geometry QA



Part 2\. VAT / GazeFollow v1

object\-grounded gaze QA



Part 3\. Action Genome / Something\-Something

temporal intention / future interaction QA



Part 4\. Ego\-Exo4D

真实 perspective\-taking / actor\-view visibility QA



Part 5\. GIMO

3D controlled gaze \+ intention QA

# 数据集

## A\. HICO\-DET

```Plain Text
zhimeng/hico_det
```

HICO\-DET 是一个 Human\-Object Interaction 数据集。它里面有：

```Plain Text
image
human bbox
object bbox
object label
interaction verb / positive captions
positive HOI object pair
```

一开始我们用它做了一个大的 v0：

```Plain Text
qa/hico_interaction_object_qa.jsonl
```

后来发现这个文件虽然干净，但问题太泛，图片里可能多个人，“the person” 有歧义，而且它不够贴合你真正想做的第三视角 gaze/intention。所以最后把 HICO 精简成了 **100 题高质量 sanity split**。

最终 HICO 输出：

```Plain Text
qa/hico_100_core_qa.jsonl
```

配套可视化图片：

```Plain Text
data_intermediate/hico_100_visuals/
debug/hico_100_core/index.html
debug/hico_100_core/selection_report.md
```



---

## B\. VideoAttentionTarget / GazeFollow

```Plain Text
scripts/15_inspect_vat_or_gazefollow.py
scripts/16_parse_vat_gaze_annotations.py
scripts/17_vat_gaze_to_geometry_qa.py
scripts/18_visualize_vat_gaze_qa.py
```

但是目前本地还没有原始数据：

```Plain Text
data_raw/video_attention_target/ 不存在
data_raw/gazefollow/ 不存在
```

所以这部分代码已经有了，但数据还没下载，实际生成：

```Plain Text
qa/vat_gaze_geometry_qa.jsonl
```

里面现在是 0 条。

---

# 

## 最终保留文件

### HICO\-100 core

```Plain Text
qa/hico_100_core_qa.jsonl
```

数量：

```Plain Text
100 条
```

分布：

```Plain Text
interaction_object: 50
left_right_of_person: 50
reachability_proxy: 0
nearest_object_to_person: 0
```

验证结果：

```Plain Text
invalid examples: 0
annotated images: 100
```

这是今天真正应该保留下来的成果。

---

# 最终 HICO\-100 的问题格式

## A\. interaction\_object

视觉输入：

```Plain Text
原图上用红框标出 target person
不画 object box
```

问题：

```Plain Text
Which object is the red-box person interacting with?
```

选项：

```Plain Text
["motorcycle", "chair", "bottle", "dog"]
```

答案：

```Plain Text
motorcycle
```

这样做的原因是：如果把 object 也框出来，答案就太简单了；但红框必须有，否则图里多个人时不知道问的是谁。

---

## B\. left\_right\_of\_person

视觉输入：

```Plain Text
红框：target person
蓝框：target object
```

问题：

```Plain Text
In the image, is the blue-box object to the left or right of the red-box person?
```

选项：

```Plain Text
["left", "right"]
```

答案：

```Plain Text
left / right
```

注意：这个任务是 **图像坐标系里的 2D left/right**，不是“从这个人的视角看左还是右”。

---

```Plain Text
red box = target person
blue box = target object, only for left/right
```

问题也改成：

```Plain Text
red-box person
blue-box object
```

---

---

# 代码框架

```Plain Text
human_spatial_intent_qa/
├── data_raw/
│   ├── video_attention_target/        # 还没下载数据
│   └── gazefollow/                    # 还没下载数据
│
├── data_intermediate/
│   ├── hico_100_visuals/              # HICO-100 标框图片
│   └── vat_gaze_annotations_normalized.jsonl
│
├── qa/
│   ├── hico_interaction_object_qa.jsonl
│   ├── hico_spatial_bbox_qa.jsonl
│   ├── hico_spatial_bbox_qa_balanced.jsonl
│   ├── hico_100_core_qa.jsonl         # 最终保留的 HICO-100
│   └── vat_gaze_geometry_qa.jsonl     # 目前 0 条，因为还没数据
│
├── debug/
│   ├── hico_100_core/
│   │   ├── index.html
│   │   └── selection_report.md
│   ├── visual_hico_spatial/
│   ├── vat_gaze_geometry/
│   │   ├── index.html
│   │   └── schema_report.md
│   ├── reachability_threshold_report.md
│   └── nearest_object_report.md
│
└── scripts/
    ├── 00_check_hico.py
    ├── 01_hico_caption_to_qa.py
    ├── 02_stats_qa.py
    ├── 03_validate_qa.py
    ├── 04_inspect_hico_bbox.py
    ├── 05_hico_bbox_to_spatial_qa.py
    ├── 06_visualize_hico_spatial_qa.py
    ├── 07_audit_reachability_thresholds.py
    ├── 08_audit_nearest_object.py
    ├── 09_balance_hico_spatial_qa.py
    ├── 14_build_hico_100_core.py
    ├── 15_inspect_vat_or_gazefollow.py
    ├── 16_parse_vat_gaze_annotations.py
    ├── 17_vat_gaze_to_geometry_qa.py
    └── 18_visualize_vat_gaze_qa.py
```

---

# 现在 benchmark 的阶段性定义

今天之后，你的 benchmark 可以这样定义：

## Part 0: HICO\-100 auxiliary sanity split

用途：

```Plain Text
测试模型能不能理解最基础的人-物交互和图像坐标关系。
```

包含：

```Plain Text
interaction_object
left_right_of_person
```

不包含：

```Plain Text
gaze
intention
perspective-taking
true reachability
3D reasoning
```

---

## Part 1: VAT / GazeFollow gaze geometry split

代码已搭好，但数据还没下载。

计划做：

```Plain Text
gaze_target_location_9way
gaze_direction_from_head
in_frame_gaze
temporal_attention_shift
gaze_target_motion_direction
```



---

> 完成了 Human Spatial Intent QA 的 HICO 辅助模块：从原始 HICO\-DET 解析、QA 生成、bbox 空间题、质量审计，到最终压缩成 100 条带红/蓝框可视化输入的高质量 sanity split；搭好了 VAT/GazeFollow gaze pipeline，但还没下载原始 gaze 数据，所以 gaze 部分还没有实际样本。
> 
> 

# 26\.5\.13

What the “Annotation” Is

The final improved run is located at:

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4_v2/
```

The generated annotation consists of three files:

---

## 1\.1 Object Annotation Example

From `geometry.jsonl`:

```JSON
{
  "sample_id": "Apartment_release_clean_seq131_M1292",
  "objects": [
    {
      "instance_id": "Apartment_release_clean_seq131_M1292::bottle_1",
      "category": "bottle",
      "centroid": [-0.417569, -2.034021, 1.226626],
      "axes_lengths": [0.354296, 0.354296, 0.573605],
      "member_count": 2
    },
    {
      "instance_id": "Apartment_release_clean_seq131_M1292::wardrobe_1",
      "category": "wardrobe",
      "centroid": [1.28455, -0.090265, -0.435677],
      "axes_lengths": [0.548804, 0.486094, 0.474402],
      "member_count": 1
    },
    {
      "instance_id": "Apartment_release_clean_seq131_M1292::sofa_1",
      "category": "sofa",
      "centroid": [0.41436, -2.083013, 1.423689],
      "axes_lengths": [0.671391, 0.354296, 0.354296],
      "member_count": 1
    }
  ]
}
```

### Field Meaning

---

## 1\.2 Relation Annotation Example

From `relations.jsonl`:

```JSON
{
  "sample_id": "Apartment_release_clean_seq131_M1292",
  "relations": [
    {
      "subject": "Apartment_release_clean_seq131_M1292::bottle_1",
      "subject_category": "bottle",
      "object": "Apartment_release_clean_seq131_M1292::wardrobe_1",
      "object_category": "wardrobe",
      "predicate": "left_of"
    },
    {
      "subject": "Apartment_release_clean_seq131_M1292::bottle_1",
      "subject_category": "bottle",
      "object": "Apartment_release_clean_seq131_M1292::plate_1",
      "object_category": "plate",
      "predicate": "larger_than"
    },
    {
      "subject": "Apartment_release_clean_seq131_M1292::bottle_1",
      "subject_category": "bottle",
      "object": "Apartment_release_clean_seq131_M1292::sofa_1",
      "object_category": "sofa",
      "predicate": "near_to"
    }
  ]
}
```

This file is the structured spatial annotation produced on top of the object\-level annotations\.

### Supported Predicate Families

---

## 1\.3 Generated QA Example

From `qa_proxy.jsonl`:

```JSON
{
  "qa_id": "proxy_qa_00000001",
  "sample_id": "Apartment_release_clean_seq131_M1292",
  "dataset": "adt",
  "task_family": "proxy_spatial",
  "question_type": "existence",
  "media_path": ".../Apartment_release_clean_seq131_M1292__f000.jpg",
  "structured_evidence": {
    "category": "bottle",
    "count": 2
  },
  "template_used": "exists_from_pseudo",
  "question": "Is there any bottle in the scene?",
  "answer": "yes"
}
```

```JSON
{
  "qa_id": "proxy_qa_00000002",
  "sample_id": "Apartment_release_clean_seq131_M1292",
  "dataset": "adt",
  "task_family": "proxy_spatial",
  "question_type": "count",
  "media_path": ".../Apartment_release_clean_seq131_M1292__f000.jpg",
  "structured_evidence": {
    "category": "bottle",
    "count": 2
  },
  "template_used": "count_from_pseudo",
  "question": "How many bottle are there in the scene?",
  "answer": "2"
}
```

Because Pipeline 1 and Pipeline 2 are now unified, these QA examples are generated by the same rule engine\.

---



## 2\.1 Final Improved Pipeline2 Outputs

Root directory:

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4_v2/
```

### Important Files

---

## 2\.2 Pipeline1 Unified Comparison Files

Directory:

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4_v2/p1_unified/
```

---

The pipeline is not only generating QA\. It produces three layers of annotation:

Therefore, if the goal is a data engine for unannotated videos, the real product is not just QA\. It is:

1. **Objects**
`instance_id`, `category`, `centroid`, `axes_lengths`

2. **Relations**
`subject`, `predicate`, `object`

3. **QA**
Natural\-language supervision derived from objects and relations

---

## Object Examples

A sample scene may contain annotations such as:

---

## Relation Examples

```Plain Text
bottle_1 left_of wardrobe_1
bottle_1 near_to sofa_1
bottle_1 larger_than plate_1
tv_1 below bowl_1
tv_1 near_to heater_1
```

---

## QA Examples

---



下面我把这段内容**翻译成中文 \+ 顺便解释每一条到底改了什么、为什么有用**。这份文本是在总结你现在 **Pipeline2 real adapter / data engine** 相比最初版本做了哪些改进，以及这些改进带来的指标变化。

---

## Sample\-level frame aggregation：把帧级结果合成场景级结果

## 原来

没有一个步骤把多帧结果合并成一个 scene\-level annotation。

## 现在

新增了聚合脚本：

```Plain Text
/home/gaoyajing/view/gaoyajing/LIMO4SI/data_engine/Cambrian-S/script/08b_aggregate_frames_to_samples.py
```

它做的事：

```Plain Text
按 category + centroid proximity 合并 frame-level detections
生成 canonical sample-level instance ID
重新从合并后的 scene 生成 relations + QA
```

比如把不同帧里的椅子合并成：

```Plain Text
sample_id::chair_1
```

## 解释

这一步让 QA 从“描述单帧”变成“描述整个场景”。

原来：

```Plain Text
frame_0 里有什么？
```

现在：

```Plain Text
这个 scene 里有什么？
这些 object 在整个 scene 里的空间关系是什么？
```

这对于 data engine 很关键。你最终要生成的是 video / scene\-level spatial QA，而不是 single\-image QA。

---

## 更稳健的 relation evaluation

## 原来

relation precision / recall 经常显示为 0。

但这不一定是模型真的完全错，而是 evaluation 有问题：

```Plain Text
category-id mismatch
predicate 没有 normalize
GT relation 里 object_category 是空的
prediction relation 里缺 subject_category / object_category
```

## 现在

改了三个地方。

### 4\.1 准备输入时补充 GT category

脚本：

```Plain Text
07_prepare_proxy_inputs.py
```

改动：

```Plain Text
GT near_to / far_from 现在携带 object_category，不再是空字符串
```

### 4\.2 预测关系也带 category

脚本：

```Plain Text
pipeline2_agent_interface.py
```

改动：

```Plain Text
predicted relations 现在携带 subject_category 和 object_category
```

### 4\.3 评估时做 predicate normalization

脚本：

```Plain Text
09_eval_pipeline2_proxy.py
```

改动：

```Plain Text
relation matching 路径里加入 predicate normalization
丢弃 invalid triples，而不是静默通过
报告 per-predicate breakdown
增加 relation recall
```

## 解释

这一步解决的是 **评估假 0**。

比如真实关系是：

```Plain Text
bottle near_to sofa
```

预测也是：

```Plain Text
bottle near_to sofa
```

但如果 GT 里 object\_category 缺失，或者 predicate 名字一个叫 `nearest_to`、一个叫 `near_to`，评估就会认为不匹配。

所以 relation 指标原来可能是“代码评估不对导致为 0”，不是 pipeline 完全没能力。

---

## Coarse\-vocabulary matching：粗粒度词表匹配

## 原来

GT 可能是：

```Plain Text
grey sofa
black coffee table
```

Detector 输出：

```Plain Text
sofa
coffee table
```

评估时完全匹配不上。

## 现在

在评估脚本里加入 head\-noun map：

```Plain Text
grey sofa → sofa
black coffee table → coffee table
```

相关脚本：

```Plain Text
09_eval_pipeline2_proxy.py
```

并且同时报告：

```Plain Text
raw precision / recall
coarse precision / recall
```

## 解释

这一步特别合理。

因为 open\-vocabulary detector 通常不会输出完整细粒度描述词，而是输出核心物体名：

```Plain Text
grey sofa → sofa
wooden chair → chair
black coffee table → coffee table
```

如果严格字符串匹配，category recall 会被严重低估。

所以 coarse matching 让评估更接近真实能力。

---

## 每个视频自动生成 prompt

## 原来

Grounding\-DINO 总是使用固定的 23 个词的 prompt list。

## 现在

新增脚本：

```Plain Text
07e_auto_prompts_from_frames.py
```

作用：

```Plain Text
给每个 input row 附上 per-sample prompts list
默认使用 80+ generic indoor nouns
预留 VLM backend hook
```

同时，`08_run_pipeline2_proxy.py` 现在会：

```Plain Text
优先读取每行自己的 prompts
如果没有，则 fallback 到 CLI default
```

## 解释

这一步是为了提升 open\-vocabulary detection 的召回。

原来固定 23 个词太少，例如只有：

```Plain Text
chair, table, bed, sofa...
```

现在扩展成 80\+ 室内通用名词。更进一步，还可以让 Qwen2\-VL / Qwen3\-VL 先看视频帧，提取可能出现的名词，再喂给 Grounding\-DINO。

重要的是：这一步是 **dataset\-agnostic** 的。

文本强调：

```Plain Text
no ADT-specific words baked in
```

也就是说没有把 ADT 的 GT 类别词硬塞进去，不算 annotation leakage。

---

## 基于先验的 metric scale calibration

## 原来

VGGT 输出是 up\-to\-scale 的，预测几何大约比真实世界小 4 倍。

## 现在

在聚合脚本中加入：

```Plain Text
calibrate_scale()
```

相关脚本：

```Plain Text
08b_aggregate_frames_to_samples.py
```

它的逻辑是：

```Plain Text
把预测出来的 object size 和常见物体真实尺寸先验匹配
例如：
door ≈ 2 m
fridge ≈ 1.7 m
chair ≈ 0.9 m
```

然后：

```Plain Text
取 median scale factor
对 centroids + axes 做 isotropic rescale
```

## 解释

这一步非常关键，而且比直接用 ADT `room_size` 更适合你的 **unlabeled data engine**。

因为它没有查 GT 字段，而是用了通用现实世界先验：

```Plain Text
chair 大概 0.9m
door 大概 2m
fridge 大概 1.7m
```

这使得它理论上可以用于任意未标注视频。

结果也很明显：

```Plain Text
预测 scene centroid extent ratio:
从 4.97 变成 1.02
```

1\.0 是最理想，所以 1\.02 基本接近修好了。

---

## 自适应 relation thresholds

## 原来

空间关系阈值是写死的：

```Plain Text
left/right margin = 0.2 m
near = 1.0 m
far = 2.5 m
```

这只适合小房间。

## 现在

`extract_relations(adaptive=True)` 会根据预测场景自己的 diagonal 动态设阈值：

```Plain Text
margin = 5% of diag
near = 25% of diag
far = 50% of diag
```

相关脚本：

```Plain Text
pipeline2_agent_interface.py
08b_aggregate_frames_to_samples.py
```

后者通过：

```Bash
--adaptive_relations
```

启用。

## 解释

这一步解决真实房间尺度变化问题。

小房间和大房间不能用同一套 near/far 阈值。

比如：

```Plain Text
小房间 diag = 4m
near = 1m
far = 2m

大房间 diag = 10m
near = 2.5m
far = 5m
```

这会比固定阈值合理很多。

文本说这一步让 real\-scale rooms 上的 predicate precision / recall 不再是 0。

---

## P1 和 P2 使用统一 QA generator

## 原来

Pipeline1 和 Pipeline2 各自有不同的 QA 生成代码。

差异包括：

```Plain Text
P1 max 30 pairwise
P2 max 12 pairwise

P1 用 nearest_to / farthest_from
P2 用 near_to / far_from
```

这会导致 P1 / P2 不可公平比较。

## 现在

新增共享 QA generator：

```Plain Text
qa_templates.py
generate_qa_unified()
```

然后：

```Plain Text
04_generate_template_qa.py 调用 shared generator
pipeline2_agent_interface.py 的 generate_qa() 也 delegate 到 shared generator
```

## 解释

现在 P1 和 P2 的区别只剩下 evidence source：

```Plain Text
P1: GT evidence
P2: predicted evidence
```

而不是 QA generator 本身不同。

这很重要，因为你可以公平比较：

```Plain Text
GT object graph → QA
predicted object graph → QA
```

也就是比较 evidence 质量，而不是比较两个不同的 QA 生成器。

---



## 1\. `category_recall_mean_coarse = 0.207` 还是偏低

这说明物体类别召回仍然是瓶颈。

优先改：

```Plain Text
per-video VLM noun extraction
better open-vocabulary detector
larger but clean indoor vocabulary
```

---

## 当前 frame aggregation 只用 category \+ centroid proximity

这对多物体场景还是不够稳。

下一步应该做：

```Plain Text
SAM2 mask embedding
CLIP crop embedding
DINOv2 visual feature
```

来做 object tracking。

---

## scale calibration 依赖 category\-size prior

这比用 GT 好，但也会受类别识别错误影响。

例如把：

```Plain Text
cabinet 识别成 fridge
```

会导致 scale factor 出错。

所以 scale calibration 最好加 confidence filter 或 median robust estimator，现在已经用 median 是一个好方向。

---





# 26\.5\.9

## Pipeline2 v2 Diagnostics: Object Inflation and QA Rule Comparability

## Why Does Pipeline2 Produce More Objects Than GT?

Pipeline2 v2 predicts more objects than GT:

```Plain Text
Predicted objects / sample: 33.2
GT objects / sample:        28.0
```

This does **not** mean Pipeline2 is simply finding extra real objects\. Looking at one concrete sample:

```Plain Text
Apartment_release_clean_seq131_M1292
```

we see the following:

- **GT:** 28 instances
Categories include `grey sofa`, `black coffee table`, `white cabinet`, `black ceramic bowl`, `kitch island`\.
Each instance corresponds to a unique physical object with its own ID\.

- **Pipeline2 v2 prediction:** 35 objects
Top categories include `2 sofa`, `2 cabinet`, `2 chair`, `2 wardrobe`, `2 ladder`, `2 picture`, `2 bottle`, `1 sink`, `1 cupboard`, etc\.

The inflation comes from three different sources\.

---

## 1\.1 Sources of Object Count Inflation

---

## 1\.2 Main Diagnosis

The fact that Pipeline2 v2 predicts `33.2` objects per sample while GT has `28.0` is mostly caused by:

1. **category aliasing across frames**, and

2. **over\-segmentation of the same physical object**\.

In other words, the aggregation step is too strict because it uses the category string as a hard merge key\.

The fix is the roadmap item previously labeled as **A6: visual instance tracking**\.

Instead of merging objects by exact category string, we should merge using:

```Plain Text
SAM2 mask similarity
CLIP crop similarity
visual instance tracking
cross-frame object identity matching
```

With this fixed, the predicted object count should decrease, while category recall should increase because more frame\-level evidence will collapse onto the correct object\.

---

## 1\.3 Diagnostic Confirmation

The calibration log provides supporting evidence:

```Plain Text
A typical sample has 12–18 known-size anchor matches,
but around 35 predicted objects.
```

This suggests that roughly half of the predictions are not genuinely new objects\. They are more likely caused by:

```Plain Text
over-segmentation
category aliasing
cross-frame duplicate clusters
```

---

## Are Pipeline1 and Pipeline2 Using the Same QA\-Generation Rules?

No\.

Pipeline1 and Pipeline2 are similar in spirit, but they do **not** currently share the same QA\-generation code path\.

Both generate template QA over:

```Plain Text
existence
count
pairwise position
pairwise size
distance
```

However, the implementation details differ in several important places\.

---

## 2\.1 Pipeline1 vs Pipeline2 QA Generator Comparison

---

## 2\.2 What Is Equivalent?

### Existence and Count QA

These are rule\-equivalent\.

The templates are basically the same:

```Plain Text
Is there any {category} in the scene?
How many {category} are there in the scene?
```

The main difference is the evidence source:

```Plain Text
Pipeline1: GT objects
Pipeline2: predicted objects
```

So existence/count QA is a relatively fair comparison\.

---

## 2\.3 What Is Not Equivalent?

### Pairwise Position QA

The question template is the same, but the volume cap is different:

```Plain Text
Pipeline1 cap: 30 per sample
Pipeline2 cap: 12 per sample
```

This alone explains most of the pairwise\-position QA volume gap:

```Plain Text
Pipeline1 pairwise_position: 1020
Pipeline2 pairwise_position: 417
```

So the lower number in Pipeline2 is partly an implementation artifact, not purely a model/pipeline weakness\.

---

### Pairwise Size QA

Pipeline1 supports both:

```Plain Text
larger_than
smaller_than
```

Pipeline2 only emits:

```Plain Text
larger_than
```

Therefore, Pipeline2 has roughly half of Pipeline1’s potential pairwise\-size QA capacity\.

This helps explain the large gap:

```Plain Text
Pipeline1 pairwise_size: 480
Pipeline2 pairwise_size: 72
```

---

### Distance QA

Pipeline1 uses:

```Plain Text
nearest_to
farthest_from
```

Pipeline2 uses:

```Plain Text
near_to
far_from
```

In the current scale50 run, Pipeline1 emits zero distance QA because the relation extractor in this codebase does not populate `nearest_to` / `farthest_from`\.

Pipeline2 emits:

```Plain Text
distance QA: 111
```

So the `0 vs 111` comparison is caused by a predicate vocabulary mismatch, not necessarily by Pipeline2 being inherently better at distance QA\.

---

# Main Conclusion

The previous Pipeline1 vs Pipeline2 comparison is **not fully apples\-to\-apples**\.

The evidence source is different, which is expected:

```Plain Text
Pipeline1: GT-derived objects and relations
Pipeline2: predicted objects and relations
```

But the QA\-generation rules are also different, which makes the comparison less clean\.

The main confounding factors are:

```Plain Text
different pairwise caps
different predicate vocabulary
different size-relation templates
different qa_id and task_family policies
different relation availability
```

So the reported QA differences mix together two effects:

1. the quality gap between GT evidence and Pipeline2 evidence;

2. implementation differences in the QA generator\.

For a fair comparison, we need to unify the QA\-generation code\.

---

# Recommended Fix: Unify QA Generation

The right fix is to create a single shared QA generator used by both Pipeline1 and Pipeline2\.

The shared function should:

1. Take per\-sample objects, relations, and sample metadata as inputs\.

2. Work for both GT\-derived objects and predicted objects\.

3. Use one canonical predicate vocabulary\.

4. Use one configurable `max_pairwise_per_sample`\.

5. Use the same `task_family` and `qa_id` policy\.

6. Normalize predicates on both sides before QA generation\.

---

## 4\.1 Canonical Predicate Vocabulary

Use a unified vocabulary such as:

```Plain Text
left_of
right_of
above
below
larger_than
smaller_than
near_to
far_from
```

Then map old or alternative names into the canonical form:

---

## 4\.2 Proposed Code Refactor

### Step 1: Create shared QA template module

Move the current Pipeline2 `generate_qa` logic from:

```Plain Text
script/pipeline2_agent_interface.py
```

into a shared file:

```Plain Text
script/qa_templates.py
```

---

### Step 2: Rewrite Pipeline1 QA generation

Modify:

```Plain Text
script/04_generate_template_qa.py
```

so that it calls the shared QA generator instead of maintaining a separate template implementation\.

---

### Step 3: Normalize relations before QA generation

Both Pipeline1 and Pipeline2 should normalize relation predicates before calling the shared generator\.

For example:

```Plain Text
nearest_to     → near_to
farthest_from  → far_from
```

---

### Step 4: Use the same pairwise cap

Both pipelines should use the same configurable parameter:

```Plain Text
max_pairwise_per_sample
```

For example:

```Plain Text
max_pairwise_per_sample = 30
```

or:

```Plain Text
max_pairwise_per_sample = 12
```

The exact number matters less than ensuring both pipelines use the same cap\.

---

### Step 5: Re\-run the comparison

After unifying QA generation, re\-run:

```Plain Text
Pipeline1 04
Pipeline2 08b
P1 vs P2 comparison
```

on the same 50 samples\.

Then the comparison will mean:

```Plain Text
same QA rules, different evidence source
```

That is the fair version of the experiment\.

---

# Final Summary

Pipeline2 v2 predicts more objects than GT mainly because of cross\-frame duplicate merging failures and category aliasing, not because it is simply finding many extra real objects\.

The main object\-level fix is:

```Plain Text
visual instance tracking
```

using SAM2 mask similarity or CLIP crop similarity instead of exact category\-string matching\.

For QA comparison, Pipeline1 and Pipeline2 are currently not using identical QA\-generation rules\. Existence/count QA is mostly comparable, but pairwise position, pairwise size, and distance QA are confounded by different caps, predicate vocabularies, and templates\.

The correct next step is to unify QA generation into a shared module so that the final comparison isolates only one variable:

```Plain Text
GT evidence vs Pipeline2 predicted evidence
```



# 26\.5\.8

---

## Pipeline2 v2 Results

**Experiment:** Improved Pipeline2 v2
**Dataset:** ADT, 50 samples, 4 frames per sample
**Compared against:**

- Pipeline2 v1 baseline

- ADT Ground Truth

- Pipeline1, GT\-driven QA pipeline

---

## A\. Improved Pipeline2 v2 vs Ground Truth: Object Information

**Report file:**

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4_v2/object_vs_gt_v1_v2.json
```

---

## A\.1 Counting \& Coverage

**Question:** Does Pipeline2 find the right objects?

### Reading

Pipeline2 v2 now produces a slightly over\-complete object set:

```Plain Text
23.94 objects/sample → 33.20 objects/sample
GT = 28.00 objects/sample
```

However, the overlap with GT is much higher:

```Plain Text
category_recall_raw:    0.038 → 0.083
category_recall_coarse: 0.128 → 0.207
instance_coverage:      0.837 → 0.977
```

The new auto\-prompts allow GDINO to recognize categories that the old fixed 23\-prompt list missed entirely\.

---

## A\.2 3D Geometry Agreement

**Question:** Are the objects the right size and in the right places?

### Reading

The size\-prior calibration fixes the previous global 4× scale problem almost completely:

```Plain Text
centroid extent GT/pred ratio: 4.97 → 1.02
```

The object axes are still imperfect:

```Plain Text
axes_length GT/pred mean ratio = 0.66
```

This means v2 overshoots object sizes by about:

```Plain Text
1 / 0.66 ≈ 1.5×
```

Object localization also improves:

```Plain Text
mean NN distance:   1.66 m → 1.30 m
median NN distance: 1.46 m → 1.19 m
```

For objects that are roughly 0\.5–2 m wide, this is “approximately right,” but not yet metrically locked in\. Without ADT camera intrinsics and true multi\-view fusion, this is a realistic ceiling\.

---

## A\.3 Remaining Differences from GT

### Vocabulary Leakage

ADT GT uses fine\-grained object labels such as:

```Plain Text
grey sofa
black coffee table
```

Current recall is still limited:

```Plain Text
raw category recall:    0.083
coarse category recall: 0.207
```

A per\-video VLM\-derived prompt source should close much of this gap\.

---

### Residual Scale Error

v2 still has about:

```Plain Text
~1.5× residual scale error
```

The current size priors are coarse, and real per\-class object size variance is large\.

Possible improvements:

```Plain Text
scale-prior network
ICP refinement
axis-length-based scale correction
```

---

### Local Position Error

The average nearest\-neighbor position error remains:

```Plain Text
1.30 m
```

This is mainly bounded by:

- VGGT single\-image 3D estimation

- only 4 input frames

- no true multi\-view fusion

- no ADT camera intrinsics / extrinsics

---

## B\. Pipeline1 GT\-driven vs Improved Pipeline2 v2: QA

**Report file:**

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4_v2/p1_vs_p2_v2_scale50.json
```

---

## B\.1 QA Quality Metrics

**Question:** How close is Pipeline2 v2 to the GT\-driven Pipeline1?

### Reading

Pipeline1 enumerates QA from GT templates, so it produces more QA but also more near\-duplicates\.

Pipeline2 v2 produces fewer QA items:

```Plain Text
4,000 → 3,092
```

But it is much cleaner:

```Plain Text
duplicate_rate: 0.0822 → 0.0026
```

This makes Pipeline2 v2 more suitable for clean training data without heavy deduplication\.

---

## B\.2 Task\-Type Distribution

### Reading

Pipeline2 v2 almost matches Pipeline1 on factual QA:

```Plain Text
existence: 1,250 vs 1,246
count:     1,250 vs 1,246
```

This means the data engine now produces nearly the same volume of factual QA as the GT\-driven pipeline\.

However, Pipeline2 v2 still produces fewer spatial relation QA:

```Plain Text
pairwise_position: 1,020 → 417
pairwise_size:       480 → 72
```

This is because adaptive relation thresholds and scale calibration emit fewer but more reliable spatial relations\.

Pipeline2 v2 also produces a new QA type:

```Plain Text
distance: 111
```

Pipeline1 does not produce this because its GT relations do not include explicit `near/far` predicates\.

---

## B\.3 Pipeline2 v2 Against Its Own GT

### Reading

Answerable rate improves clearly:

```Plain Text
0.730 → 0.807
```

Relation recall roughly doubles:

```Plain Text
0.012 → 0.023
```

Answer consistency is essentially flat:

```Plain Text
0.159 → 0.150
```

So v2 mainly improves coverage, answerability, and relation recall, while consistency remains similar\.

---

# C\. Summary

The improved Pipeline2 v2 finds more GT objects, improves category recall, fixes the previous global scale failure, and produces cleaner QA\.

Key improvements:

```Plain Text
instance_coverage:              0.837 → 0.977
category_recall_coarse:         0.128 → 0.207
centroid extent GT/pred ratio:  4.97  → 1.02
mean NN distance:               1.66 m → 1.30 m
answerable_rate:                0.730 → 0.807
```

On QA, Pipeline2 v2 now generates nearly the same number of existence/count questions as the GT\-driven Pipeline1, while having a much lower duplicate rate:

```Plain Text
duplicate_rate: 0.0822 → 0.0026
```

The remaining gap to GT is mainly caused by:

1. detector vocabulary still missing rare ADT classes;

2. about 1\.3 m residual local position error;

3. limited pairwise spatial QA coverage\.

---

# D\. Recommended Next Steps

## Step 1: Add per\-video VLM auto\-prompts

Goal:

```Plain Text
improve category coverage, especially long-tail ADT object categories
```

Expected gains:

```Plain Text
higher raw category recall
higher coarse category recall
more stable existence/count QA
```

---

## Step 2: Improve 3D localization

Possible directions:

```Plain Text
more frames
true multi-view fusion
ADT camera intrinsics/extrinsics
ICP refinement
```

Goal:

```Plain Text
reduce the current ~1.30 m local position error
```

---

## Step 3: Improve pairwise spatial QA

Current gap:

```Plain Text
pairwise_position: 417 vs 1,020
pairwise_size:      72 vs 480
```

Possible directions:

```Plain Text
relation threshold tuning
scale calibration
object pair sampling
predicate balancing
```

This is the most important part if the final target is spatial reasoning training data\.

# 26\.5\.3

## 这次完成了什么

---

## 真实适配器 Pipeline2 在 50 个样本上的成绩

**输出目录：**

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4
```

### 2\.1 流水线运行情况

### 2\.2 核心指标

---

## Pipeline1（GT 驱动）vs 真实 Pipeline2

**对比对象：同样的 50 个样本**

**报告文件：**

```Plain Text
/mnt/ssd4t/data/gaoyajing/LIMO4SI/data_engine/Cambrian-S/outputs/pipeline2_real/adt/scale50_qwen3vl_multiframe4/p1_vs_p2_real_scale50.json
```

### 读法

Pipeline1 因为是按 GT 模板枚举所有实例对，所以 QA 数量更大，但重复率也更高。

Pipeline2 使用真实模型推断，生成量更少，但重复率显著降低：

```Plain Text
0.082 → 0.011
```

---

## 真实 Pipeline2 自身的演进

### 要点

- 从单帧扩展到 4 帧后，实例覆盖率从约 `0.49` 提升到 `0.84+`，提升约 70%。

- `answerable_rate` 从 `0.58` 提升到 `0.73`。

- `answer_consistency_rate` 从 `0.05` 提升到 `0.16`。

- 5 个样本和 50 个样本上的指标基本一致，说明多帧带来的提升不是小数据噪声，扩展到 50 个样本后依然成立。

---

## 关系层各 predicate 拆分，粗粒度词表

完整数字在：

```Plain Text
proxy_eval_real_coarse.json
```

字段位置：

```Plain Text
relation_eval.predicate_breakdown_coarse
```

### 说明

修复前，关系层指标全部为 0。

修复后，所有空间 predicate 都已经有非零命中，说明关系评测链路已经打通，但整体 precision / recall 仍然偏低。

---

## 还没做的事和当前瓶颈

---

## 关键产物文件

---

## 总结版结论

这次主要完成了 Pipeline2 的多帧化、样本级聚合、关系评测修复和粗粒度词表对齐，并把真实适配器从 5 个样本扩展到了 50 个样本。

实验结果说明：

1. **多帧输入显著提升实例覆盖率。**
单帧的 instance coverage 约为 `0.486`，4 帧后提升到 `0.837–0.886`。

2. **QA 可回答率明显提升。**
`answerable_rate` 从 `0.583` 提升到 `0.730`。

3. **QA 一致性也有提升。**
`answer_consistency_rate` 从 `0.048` 提升到约 `0.159`。

4. **真实 Pipeline2 相比 GT 驱动的 Pipeline1，生成 QA 数量更少，但重复率更低。**
P1 的重复率为 `0.082`，P2 降到 `0.011`。

5. **当前最大瓶颈仍然是检测词表。**
原始类别匹配只有 `0.038`，粗词表对齐后提升到 `0.128`，但仍然偏低。后续主要应该优化 GDINO prompt 词表，或者更换更强的检测器。



# 26\.4\.15 

## pipeline2

- `script/07_prepare_proxy_inputs.py`

    - 输入：metainfo JSON（top\-level dict，含 GT）

    - 输出：

        - `proxy_input.jsonl`（给 pipeline2 运行，隐藏 GT）

        - `gt_holdout.jsonl`（只用于离线评测）

        - `proxy_prepare_summary.json`

    - `proxy_input.jsonl` 每行字段：

        - `sample_id, dataset`

        - `image_path, video_path`

        - `room_size, room_center`

    - `gt_holdout.jsonl` 每行字段：

        - `sample_id, dataset`

        - `object_counts`

        - `instances`（GT 实例，含 `instance_id/category/centroid/axes_lengths/bbox_min/bbox_max`）

        - `gt_relations`（由 GT 几何推导的关系集合）

    - 目的： runtime 不读 GT；GT 仅用于后续 `09` 评测。

---

- `script/pipeline2_agent_interface.py`

    - 作用：定义 pipeline2 各阶段统一接口（当前是 mock/占位实现，可替换真实工具）

    - 主要接口：

        - `detect_objects(samples, prompts) -> detections`

        - `segment_instances(samples, detections) -> masks`

        - `estimate_3d(samples, masks) -> geometry`

        - `postprocess_geometry(geometry) -> cleaned_geometry`

        - `extract_relations(geometry) -> relations`

        - `generate_qa(samples, geometry, relations) -> qa_rows`

    - 产物语义：

        - `detections`：类别 \+ `bbox_xywh_norm` \+ score

        - `masks`：实例 mask 面积比例 \+ bbox

        - `geometry`：实例 3D `centroid/axes_lengths`

        - `relations`：`subject/predicate/object`

        - `qa_rows`：canonical QA（existence/count/pairwise/distance）

---

- `script/08_run_pipeline2_proxy.py`

    - 输入：`proxy_input.jsonl`

    - 处理顺序（每个样本）：

        - detect \-\> segment \-\> estimate\_3d \-\> postprocess \-\> relations \-\> qa

    - 输出目录文件：

        - `detections.jsonl`

        - `masks.jsonl`

        - `geometry.jsonl`

        - `relations.jsonl`

        - `qa_proxy.jsonl`

        - `run_log.jsonl`

        - `run_summary.json`

    - `run_log.jsonl` 关键字段：

        - `sample_id, status`

        - `error_stage, error_message`

        - `latency_ms`

        - `keep, drop_reason`

        - `qa_count`

    - 目的： 把 pipeline2 全链路跑通，并保留可调试中间产物。

---

- `script/09_eval_pipeline2_proxy.py`

    - 输入：

        - `gt_holdout.jsonl`（来自 07）

        - `qa_proxy.jsonl / geometry.jsonl / relations.jsonl`（来自 08）

    - 输出：`proxy_eval.json`

    - 评测层：

        - 检测层：

            - `category_recall_mean`

            - `instance_coverage_mean`

        - 关系层：

            - `predicate_precision_mean`

        - QA 层：

            - `answerable_rate`

            - `answer_consistency_rate`

        - QA 质量层：

            - 复用 `script/06_eval_quality.py` 的统计（`duplicate_rate/ambiguity_rate/task_distribution`）

    - 目的： 判断 pipeline2 不只是“能跑”，还能看清“哪一层在掉点”。

- agent\_interface

    - `detect_objects(...)`

        - 输入样本和 prompts

        - 输出伪检测框（类别、bbox、score）

        - 现在是 deterministic mock，不是真检测模型

    - `segment_instances(...)`

        - 基于 detection 生成 mask 信息（占位）

        - 输出每个实例的 mask area ratio 等

    - `estimate_3d(...)`

        - 把 2D 框/掩码映射到简化 3D 几何

        - 输出 `centroid` 和 `axes_lengths`

    - `postprocess_geometry(...)`

        - 做几何后处理（比如 erosion 风格的缩边）

        - 清理/收缩尺寸，得到更稳定几何

    - `extract_relations(...)`

        - 从 3D 几何里算关系

        - 例如 left/right、above/below、larger\_than、near/far

    - `generate_qa(...)`

        - 从伪标注几何\+关系生成标准 QA 记录

        - 包括 existence/count/pairwise/distance 等



# 26\.4\.13

## Task：

1. 打包所有的几何关系 \+ 图像/视频 \+ 问题模板 \-\> MLLM \-\> QA

2. 视频 \-\> MLLM \-\> tool calling \(grounding dino, sam3, vggt, erosion\.\.\) \-\> QA

## Task1：

### 处理解析metainfo

#### 目录中剩余文件

- `20250416-batch/arkitscenes_train_20250305.txt`

- `20250416-batch/scannet_train.txt`

- `20250416-batch/scannetpp_v2_train_20250305.txt`

- `appr_order_raw/qa_obj_appearance_order_scannetpp_v2_converted.json`

- `appr_order_raw/qa_obj_appearance_order_test_0418.json`

- `fianlver-vsibench/arkitscenes_train_coreset_anno_filtered_20250305.json`

- `fianlver-vsibench/scannet_train_meta_info-20250130.json`

- `fianlver-vsibench/scannetpp_v2_coreset_anno_filtered_20250304.json`

- `finalver-adt-video/adt_meta_info_v5(1).json`

- `finalver-hypersim/hypersim_meta_info_v6.json`

- `finalver-hypersim/hypersim_meta_info_v7.json`

- `finalver-s3dis/s3dis_meta_info_v2.json`

- `structured3d_meta_info_v5.json`

- `scannet_train_frame_category_info_20250304.npy`

- `scannetpp_v2_train_frame_category_info_20250420.npy`

- `gitattributes`

---

#### 逐文件 Profile

A\. split 列表文件（TXT）

- `20250416-batch/arkitscenes_train_20250305.txt`

    - 格式: 纯文本，每行一个场景 ID（如 `40777060`）

    - 用途: ARKitScenes 训练 split

- `20250416-batch/scannet_train.txt`

    - 格式: 纯文本，每行一个场景名（如 `scene0191_00`）

    - 用途: ScanNet 训练 split

- `20250416-batch/scannetpp_v2_train_20250305.txt`

    - 格式: 纯文本，每行一个场景 ID（如 `39f36da05b`）

    - 用途: ScanNet\+\+ v2 训练 split

---

B\. QA 任务文件（JSON 数组）

- `appr_order_raw/qa_obj_appearance_order_scannetpp_v2_converted.json`（68\.3MB）

- `appr_order_raw/qa_obj_appearance_order_test_0418.json`（14\.4MB）

    - 顶层格式: `[]`（list of dict）

    - 样本字段: `dataset_name`, `video_path`, `scene_name`, `question_type`, `question`, `options`, `ground_truth`, `selected_option` 等

    - 用途: “物体首次出现顺序”问答评测/指令数据，不是检测元信息

---

C\. 主元信息文件（JSON 对象，key=scene/frame）

这些文件共同特征：

- 顶层格式: `{}`（dict）

- 典型字段: `dataset`, `video_path`（部分数据有）, `object_counts`, `object_bbox`

- 用途: 训练/评测的数据索引与标注汇总（按场景或按帧）

具体如下：

- `fianlver-vsibench/arkitscenes_train_coreset_anno_filtered_20250305.json`（22\.0MB）

    - key 示例: `40777060`

    - 用于 ARKitScenes coreset \+ 过滤标注

- `fianlver-vsibench/scannet_train_meta_info-20250130.json`（19\.3MB）

    - key 示例: `scene0191_00`

    - 用于 ScanNet 训练元信息

- `fianlver-vsibench/scannetpp_v2_coreset_anno_filtered_20250304.json`（23\.5MB）

    - key 示例: `39f36da05b`

    - 用于 ScanNet\+\+ coreset 元信息

- `finalver-adt-video/adt_meta_info_v5(1).json`（4\.9MB）

    - key 示例: `Lite_release_recognition_Flask_seq030_61283`

    - ADT 视频定版元信息

- `finalver-hypersim/hypersim_meta_info_v6.json`（322\.9MB）

- `finalver-hypersim/hypersim_meta_info_v7.json`（181\.4MB）

    - key 示例: `hypersim/.../frame.0004.tonemap.jpg`（帧级 key）

    - 以帧为索引的 Hypersim 元信息；v7 更精简/更新（体积更小）

- `finalver-s3dis/s3dis_meta_info_v2.json`（1\.2MB）

    - key 示例: `Area_6_lounge_1`

    - S3DIS 场景元信息，含 `video_path/object_counts/object_bbox`

- `structured3d_meta_info_v5.json`（169\.2MB）

    - key 示例: `scene_00000`

    - Structured3D 元信息，结构同上

---

D\. 帧类别索引（NPY）

- `scannet_train_frame_category_info_20250304.npy`（628\.3MB）

- `scannetpp_v2_train_frame_category_info_20250420.npy`（1\.6GB）

    - 格式: NumPy `.npy`，header 显示 `descr='|O'`, `shape=()`，即object 标量（通常是 pickled Python dict）

    - 用途: 快速做 frame \-\> category 信息查询

    - 读取方式: 必须 `allow_pickle=True`

---

E\. 其他

- `gitattributes`

    - 格式: Git 配置文本

    - 用途: 指定 LFS 跟踪规则（不参与训练）

**具体**

- `appr_order_raw/qa_obj_appearance_order_scannetpp_v2_converted.json`

    - 顶层：`list`（68036 条）

    - 每条字段：`dataset_name, video_path, scene_name, question_type, question_parameters, question, ground_truth, options, template_used, selected_option, related_category`

    - 类型：QA 样本格式

- `appr_order_raw/qa_obj_appearance_order_test_0418.json`

    - 顶层：`list`（13184 条）

    - 每条字段同上

    - 类型：QA 样本格式（测试集）

- `fianlver-vsibench/arkitscenes_train_coreset_anno_filtered_20250305.json`

    - 顶层：`dict`（2899 条，key 是 scene id，如 `40777060`）

    - value 字段：`video_path, room_size, room_center, object_counts, object_bbox, dataset`

    - 类型：场景级 metainfo

- `fianlver-vsibench/scannet_train_meta_info-20250130.json`

    - 顶层：`dict`（1201 条，key 如 `scene0191_00`）

    - value 字段：`room_size, object_counts, object_bbox, room_center, video_path, dataset`

    - 类型：场景级 metainfo

- `fianlver-vsibench/scannetpp_v2_coreset_anno_filtered_20250304.json`

    - 顶层：`dict`（856 条，key 如 `39f36da05b`）

    - value 字段：`dataset, video_path, room_size, room_center, object_counts, object_bbox`

    - 类型：场景级 metainfo

- `finalver-adt-video/adt_meta_info_v5(1).json`

    - 顶层：`dict`（236 条，key 为序列名）

    - value 字段：`dataset, video_path, room_size, room_center, object_counts, object_bbox`

    - 类型：视频/序列级 metainfo

- `finalver-hypersim/hypersim_meta_info_v6.json`

    - 顶层：`dict`（10492 条，key 为 frame 路径）

    - value 字段：`object_counts, object_bbox, dataset, image_path`

    - 类型：帧级 metainfo

- `finalver-hypersim/hypersim_meta_info_v7.json`

    - 顶层：`dict`（10193 条，key 为 frame 路径）

    - value 字段：`object_counts, object_bbox, dataset, image_path`

    - 类型：帧级 metainfo（v7）

- `finalver-s3dis/s3dis_meta_info_v2.json`

    - 顶层：`dict`（199 条，key 如 `Area_6_lounge_1`）

    - value 字段：`video_path, object_counts, object_bbox, room_size, dataset`

    - 类型：场景级 metainfo

- `structured3d_meta_info_v5.json`

    - 顶层：`dict`（3297 条，key 如 `scene_00000`）

    - value 字段：`dataset, video_path, object_counts, object_bbox`

    - 类型：场景级 metainfo

---

一句话总结你的 JSON 家族：

- 两类主格式：

    1. `dict[scene_or_frame_id] -> meta`（metainfo 主体）

    2. `list[qa_sample]`（QA 任务样本）

- 其中 `hypersim_*` 是帧级，其余大多是场景/序列级。

#### 1\.1\.3 scripts

- `script/01_convert_metainfo.py`

    - 输出：`*.jsonl`（如 `outputs/s3dis_unified.jsonl`）

    - 每行一个统一样本，字段：

        - `sample_id`, `dataset`, `granularity`

        - `image_path`, `video_path`

        - `room_size`, `room_center`

        - `instances`（扁平实例列表，每个实例有 `instance_id/category/raw_category/centroid/axes_lengths/normalized_axes/bbox_min/bbox_max`）

        - `object_counts`

    - granularity

        `frame: has image_path`

        If there is an `image_path`, that entry is probably describing one image\.

        `sequence: has scene_name`

        If there is `scene_name` and it looks like a video clip/sequence identifier, treat it as a sequence\-level sample\.

        `scene: fallback`

        If it is neither clearly a frame nor a sequence, treat it as a whole scene\.

- `script/02_profile_axes_and_bbox.py`

    - 输出：`*.json`（如 `outputs/s3dis_axis_profile.json`）

    - 记录统计报告：

        - `record_count`, `total_instances`, `invalid_bbox_count`

        - `axis_lengths_stats`（3维均值/最小/最大）

        - `bbox_span_stats`

        - `centroid_stats`

        - `note`（轴语义判定提示）

- `script/03_extract_relations.py`

    - 输出：`*.jsonl`（如 `outputs/s3dis_relations.jsonl`）

    - 每行一个样本关系集：

        - `sample_id`, `dataset`

        - `relations`: 列表，元素形如
        `{"subject","predicate","object"(可选),"score"}`

        - 关系类型含：`exists/count/left_of/right_of/in_front_of/behind/above/below/larger_than/smaller_than/nearest_to/farthest_from`

- `script/04_generate_template_qa.py`

    - 输出：`*.jsonl`（如 `outputs/qa_template_baseline_s3dis.jsonl`）

    - 每行一个 QA，字段：

        - `qa_id`, `sample_id`, `dataset`

        - `task_family`, `question_type`

        - `media_path`

        - `structured_evidence`

        - `template_used`

        - `question`, `answer`, `options`（当前为 `null`）

        - `metadata`（例如 `rule_name`）

- `script/05_mllm_rewrite_or_filter.py`

    - 输出：`*.jsonl`（模式相关）

    - `--mode rewrite`：在原 QA 基础上新增

        - `rewritten_question`, `rewrite_method`

    - `--mode filter`：在原 QA 基础上新增

        - `filter_keep`, `filter_reason`

- `script/06_eval_quality.py`

    - 输出：`*.json`（如 `outputs/qa_template_baseline_s3dis_eval.json`）

    - 质量评估指标：

        - `total_items`

        - `duplicate_items`, `duplicate_rate`

        - `ambiguous_question_pairs`, `ambiguity_rate`

        - `task_distribution`（各 `question_type` 数量）

# 26\.4\.9

## Learning Notes

**Grounding DINO** 是一个**open\-vocabulary detection（开放词表目标检测）模型**，核心能力是：

> 用“文本 prompt”来检测图像中的任意目标，而不是固定类别
> 
> 

传统检测器（如 Faster R\-CNN / YOLO）的问题：

- 类别固定（closed\-set） 

- 需要标注数据（每个类别都要bbox） 

### Grounding\-DINO作用：

- 支持任意文本类别（open\-set） 

- 可以 zero\-shot（不用该类训练数据） 

- 可以 phrase\-level grounding（比如 “a man holding a red cup”） 

本质：**Detection = Localization \+ Language grounding**

Grounding\-DINO = **DINO \+ Text Encoder \+ Cross\-modal Alignment**

结构拆解：

1. **Image Encoder**

    - backbone：通常是 Swin Transformer / ViT 

    - 输出：image feature map 

2. **Text Encoder**

    - 类似 BERT 或 CLIP text encoder 

    - 输入：prompt（如 "dog", "a red car"） 

    - 输出：text embeddings 

3. **Fusion（关键）**

    - cross\-attention between image \& text 

    - 把“文本语义”注入视觉 token 

4. **Detection Head（来自 DINO）**

    - object queries（类似 DETR） 

    - 每个 query 输出： 

        - bounding box 

        - 与文本的 matching score

### Grounding\-Dino问题：

1. 依赖 text prompt

- prompt quality 很重要 

- 类别集合要设计好 

👉 可以做：

- prompt engineering 

- 自动类别生成（LLM） 

---

2. box 质量不稳定

- 特别是 small objects 

- 或复杂 occlusion 

👉 所以后面必须接：

- SAM / SAM2 refine 

---

3. 不利用 temporal 信息

- 每帧独立 

- 没有 tracking 

👉 这正是你 pipeline 的关键问题点



### SAM2: Prompt → Mask

本质是：

mask=f\(image,prompt\)\\text\{mask\} = f\(\\text\{image\}, \\text\{prompt\}\)mask=f\(image,prompt\)

不是：

image → class → mask

而是：

\(image \+ prompt\) → mask

N 个 mask（通常 N=3）

每个 mask 是一个 H × W 的矩阵，像素级的二机函数

### SAM2 的关键改进

1. Memory mechanism

记住之前的 mask 

在后续帧传播 

👉 类似：mask\_t → memory → mask\_\{t\+1\}

---

2. 视频一致性

避免： mask 抖动  ID 不一致 

---

3. prompt propagation

只需要：第一帧标注，后面自动跟踪



### SAM2 误差/偏差

mask 的常见问题：

- 边界粘连 

- 细长部件漏掉 

- 被遮挡区域断裂 

- 相邻同类物体 merge、

\-\> SAM3

### **VGGT**

> VGGT = 一个把 **2D图像 → 3D几何结构（点云/深度/关系）** 的模型
> 
> 

可以理解为：

image → geometry（depth / point cloud / structure）



VGGT 的输出通常是：

---

1. Depth（深度） \(Map Anything\)

depth: \[H, W\]

含义：

D\(x,y\)=distance to cameraD\(x,y\) = \\text\{distance to camera\}D\(x,y\)=distance to camera

---

2. Point Cloud（点云）

points: \[H, W, 3\]

含义：

\(x,y\)→\(X,Y,Z\)\(x,y\) → \(X,Y,Z\)\(x,y\)→\(X,Y,Z\)

👉 每个像素 → 一个 3D 点

---

3. Geometry tokens / features（有些实现）

geometry embedding

用于：

- reasoning 

- QA 

- spatial understanding 



Camera Extrinsics:

- Camera pose\. \-\> motion ground truth



Z=D\(x,y\)

X=\(x−cx\)⋅Z/fx

Y=\(y−cy\)⋅Z/fy

👉 这就是：

> depth → point cloud
> 
> 

---

Transformer 做几何建模

VGGT 用 Transformer 来建模：

- 空间关系 

- 深度一致性 

- 多视角结构 

### VGGT误差/偏差

这是整条链里噪声最大的地方之一：

- 反光、透明体、镜子会让 3D 很不稳 

- 纹理贫乏区域深度模糊 

- 遮挡区无法可靠恢复 

- 单帧方法对绝对尺度尤其脆弱





![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTFmM2NlMjdhNzY3MDg2NWU4YTA4N2JjN2U3OGEzYzlfYWYxMjA0NmNjODYwNDNmNTdlYjYzMzkwMjVmOGMxMWJfSUQ6NzYyNjU4NDAxNzI1NjkyNjQwNV8xNzg1MzM4MjkyOjE3ODU0MjQ2OTJfVjM)

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=OTRlZjU3ZDhmNzIwMjBmN2FmMDhhNWYzN2Y4MzcyZmJfZTUxNzhlNGY5MjIxM2ExZmM3OGZkODFhMmYzOTFhNzBfSUQ6NzYyNjU4NDE2NDQyNjQ2ODU1Ml8xNzg1MzM4MjkyOjE3ODU0MjQ2OTJfVjM)

QAs are grouped by: question types \(left\) and task groups \(right\)\.

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=YTFiMmRmN2IyOTE0NDEzMTgzNTk3MmU5NGI1NWM4NTNfNjQ4N2ZlNTRhNzI0ZmQzNWI1YTRmZWIyYmI5MTI4MTBfSUQ6NzYyNjU4NDIwNjI2ODczMDMxN18xNzg1MzM4MjkyOjE3ODU0MjQ2OTJfVjM)

Cambrian\-s:https://github\.com/cambrian\-mllm/cambrian\-s

MetaInfohttps://huggingface\.co/datasets/nyu\-visionx/VSI\-590K\-MetaInfo/tree/main



## Meta Info

很好，这个截图信息量已经够了，我直接帮你“读结构 \+ 给你决策”。

---

## **这个 MetaInfo =  70% 的 pipeline**

你现在：

- ❌ 不需要自己跑 Grounding\-DINO / SAM2 / VGGT（至少第一阶段不用）

- ❌ 不需要自己做 pseudo\-annotation

- ✅ 可以直接做：

    - VSI\-590K 分析

    - QA generation / taxonomy 分析

    - data mixture 研究

### 1️⃣ `*_meta_info.json`（核心）

### 包括：

- `adt_video_meta_info_*.json`

- `hypersim_meta_info.json`

- `s3dis_meta_info_v2.json`

- `structured3d_meta_info_v5.json`

👉 这些就是：

> **已经整理好的 meta information file（你刚问的那个）**
> 
> 

---

里面一定包含（推断 \+ 经验）：

```JSON
{
  "scene_id": "...",
  "image/frame_id": "...",
  "objects": [
    {
      "category": "...",
      "bbox": ...,
      "center": ...,
      "size": ...
    }
  ],
  "room_size": ...,
  "counts": ...
}
```

👉 也就是：

- 已经做完 geometry normalization

- 已经统一 schema

- 已经可以直接算 spatial relation

---

这意味着：

你可以直接做：

```Python
# example
chairs = [o for o in objects if o["category"] == "chair"]
count = len(chairs)
```

**直接生成 QA，不需要任何视觉模型**

---

### 2️⃣ `scannet_*.npy` / `scannetpp_*.npy`

文件名：

- `scannet_train_frame_category_info_*.npy`

- `scannetpp_v2_train_frame_category_info_*.npy`

---

**frame\-level category statistics / mapping**

很可能是：

```Python
frame_id -> [categories present]
```

或者：

```Python
category -> frames list
```

---

它的作用：

👉 **快速筛选 candidate frame**

比如：

- 找包含 ≥3 个 object 的帧

- 找包含 chair \+ table 的帧

- 找适合 count / relation 的帧

---

这是 Cambrian pipeline 里的：

> “object count threshold \+ eligibility filtering”
> 
> 

---

### 3️⃣ `meta info mega.zip`

这个很可能是：

> **统一打包后的全部 meta info（大集合）**
> 
> 

也可能包含：

- 所有数据源 merge 后的统一格式

- 或者 QA\-ready metadata

---

## 4️⃣ `.json vs .npy 的区别`



---

---

```Plain Text
[MetaInfo] → （你现在从这里开始） → QA generation → analysis
```

- 多数据源统一 schema

- geometry\-ready 数据

- frame\-level 筛选信息



![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZThkODA2Y2U0NjNiYWU4Mjc2MTIzODM1ZDkzMzExYWRfYTIxMmQ2Zjg5MGIzNmVkNWE1ZWM4ZjAzZTg1OTRlNzRfSUQ6NzYyNjYzNzY1Nzg0MjAxMTA5NV8xNzg1MzM4MjkyOjE3ODU0MjQ2OTJfVjM)



很好，这个 sample 非常典型，而且信息是**非常干净的“geometry\-ready meta”**。我直接帮你把：

> ✅ 你应该分析什么
> ✅ 怎么从这个结构里算出来
> ✅ 最后写成什么“论文级结论”
> 
> 

一步一步给你拆清楚。

---

```Python
scene = {
    "dataset": "adt_video",
    "object_counts": {...},   # 已经统计好
    "object_bbox": {...},     # 每个实例的3D bbox
}
```

`object_counts` = **语义分布**

- `object_bbox` = **几何分布（3D）**

- 每个 object 有多个 instance（list）

---

### 现象 1：类别非常细粒度

比如：

- ❌ chair

- ✅ black kitchen chair / red arm chair / black arm chair

👉 说明：

> **不是标准 taxonomy，而是带属性的 fine\-grained label**
> 
> 

---

### 现象 2：长尾极强

- 大部分 object count = 1

- 只有极少数（比如 chair）是 2\+

👉 说明：

> **真实场景分布（long\-tail），不是合成数据**
> 
> 



```Python
len(object_counts) ≈ 30 类
总 object ≈ 30+
```



> ❗不是“每类多实例”，而是“多类 \+ 单实例”
> 
> 

---

![Image](https://internal-api-drive-stream.feishu.cn/space/api/box/stream/download/authcode/?code=ZDdiN2Y5MzA4MmU0MTNmMWU1YzkyNjhlYWRkNGU1ZjVfMTUxNzA5MGRhMjc5NzdjOTQ5MzU1MTIxN2NjYWJkMjNfSUQ6NzYyNjY0MDY4MDI3NjUyODA1OF8xNzg1MzM4MjkyOjE3ODU0MjQ2OTJfVjM)



# 26\.4\.7

回答三个问题：

1. **什么叫真正的 spatial intelligence / supersensing？**

2. **只靠更好的数据和 SFT，最多能把现有范式推到哪里？**

3. **VSI\-590K 这种数据到底是怎么造出来的？**

论文的核心结论很明确：

- 他们把“spatial supersensing”分成 4 个层级：semantic perception、streaming event cognition、implicit 3D spatial cognition、predictive world modeling。

- 为了测试这个 gap，他们提出了 **VSI\-SUPER**；为了测试“是不是只差数据”，他们又构造了 **VSI\-590K** 并训练 **Cambrian\-S**。结果是：在当前 MLLM 范式下，精心做 spatial data 确实能显著推高 VSI\-Bench，但在 VSI\-SUPER 上依然不够，这说明**数据扩张 \+ 上下文扩张并不能直接等于 supersensing**。\([arXiv](https://arxiv.org/abs/2511.04670)\)

---

## 论文主要内容

这篇论文不是单纯“提一个更强的视频模型”。

它更像是在说：

> 现在的视频 MLLM，大多还是把视频当成“很多帧 \+ 很长上下文”的 QA 问题；
> 但真正的 spatial intelligence，不该只是把帧喂进去做回答，
> 而是要在连续视觉流里形成、更新、筛选、预测一个隐式世界模型。
> 
> 

所以它分三部分：

- **Part 1：benchmark diagnosis**
先审视现有 benchmark 到底在测什么，指出很多 benchmark 视觉依赖并没那么强，更多在吃语言先验；VSI\-Bench 才更像是在测真正的视觉空间能力。

- **Part 2：data scaling within current paradigm**
构建 **VSI\-590K**，训练 **Cambrian\-S**，看看“高质量 spatial data \+ 更强 base model \+ 更强 video SFT”能把现有范式推多远。

- **Part 3：new paradigm**
发现即使这样做了，VSI\-SUPER 还是不行，于是提出 **predictive sensing**：用下一 latent frame prediction 的 error，也就是“surprise”，去驱动 memory management 和 event segmentation。

所以这篇论文真正的逻辑主线是：

**benchmark 诊断 → 数据扩张验证上限 → 提出新范式方向**

---

## Cambrian\-S 本身是什么模型

模型上它没有走特别花哨的新架构，而是把 **Cambrian\-1** 做了升级：

- vision encoder 升级为 **SigLIP2\-SO400m**

- language model 换成 **Qwen2\.5\-Instruct**

- vision\-language connector 用简单的两层 MLP

- 整体仍然延续 Cambrian\-1 的多阶段训练流程。

训练后期的关键是两阶段 video tuning：

- **Stage 3：general video instruction tuning**
用通用视频数据训练一般视频理解能力。

- **Stage 4：spatial video instruction tuning**
用 **VSI\-590K** 专门强化 spatial reasoning。这里他们把每个视频的帧数从 64 提到 128，sequence length 扩到 16,384，以支持更强 temporal modeling。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

所以你可以把 Cambrian\-S 理解成：

**“更强的 image MLLM base \+ 通用 video SFT \+ spatial\-specific video SFT”**

而不是一个从底层彻底重做的视频架构。

---

## VSI\-590K 是怎么构建的

这是你这两天任务的核心。

论文里 **VSI\-590K** 的设计思想很清楚：
单一来源的数据不够，必须混合三类来源，来同时兼顾：

- 几何真值质量

- 场景/外观覆盖面

- 真实世界分布

他们把数据分成三类：

### A\. Annotated real videos

来自带 3D instance\-level annotation 的真实室内/第一人称数据：

- S3DIS

- Aria Digital Twin

- ScanNet

- ScanNet\+\+ V2

- ARKitScenes \([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

处理方式是：

1. 把每个数据集原有的 3D 标注、bbox、room size、object counts 等统一整理到 meta\-information file

2. 再基于这些 scene\-level / object\-level 几何信息，用模板自动生成 QA。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

这类数据的优点是：

- **几何真值最可靠**

- 适合 size / distance / relative direction / room size 这种对几何质量敏感的题型

所以这部分是整个数据引擎里**质量最高的锚点**。

---

### B\. Simulated data

他们用模拟器扩 coverage：

- ProcTHOR：生成 625 条 spatially grounded video traversals

- Hypersim：从 461 个室内场景采样 5,113 张图像，再用 instance bbox 生成 QA。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

这里的目的不是替代真实数据，而是补：

- 更多 layout

- 更多 object configuration

- 更多 appearance diversity

- 更容易大规模程序化生成。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

这和你写的总结“模拟数据扩充覆盖面”是完全一致的。

---

### C\. Unannotated real videos

这最像你说的 **Data Engine Agent**。

来源包括：

- 约 19K YouTube room tour 视频

- Open\-X\-Embodiment

- AgiBot\-World。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

因为这些视频没有 3D annotation，所以他们做了一个 **pseudo\-annotation pipeline**：

1. **subsample frames**

2. **filter blurry images**

3. 用 **Grounding\-DINO** 在预定义类别上做 open\-vocabulary detection

4. 如果 frame 里对象足够多，就用 **SAM2** 做 instance masks

5. 再用 **VGGT** 把 2D 图像内容转成 3D point sets

6. 将 point sets 和 instance masks 对齐

7. 用 erosion 精修 mask，减轻边界处点云估计误差

8. 最后再从几何关系中自动生成 QA。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)





而且论文特别强调了一点：

> 他们选择在 **image level** 做 pseudo\-annotation，而不是对整段视频做 full\-video pseudo\-annotation，
> 因为基于 recognition \+ reconstruction 的整视频伪标注噪声太大。 \([ar5iv](https://ar5iv.org/pdf/2511.04670)\)
> 
> 

这一点非常关键。

它其实说明作者在 data engine 上的立场是：

**宁可牺牲时序完整性，也要保住几何伪标签质量。**

这对你后面 build pipeline 非常重要。

---

## pipeline

- 真实 3D 标注数据提供高质量几何真值

- 模拟数据扩充覆盖面

- 无标注真实视频通过 Grounding\-DINO \+ SAM2 \+ VGGT 做图像级伪 3D 标注

- 最后围绕 size / direction / count / distance / appearance order 模板生成

这和论文几乎完全对齐。对应关系是：

- **真实 3D 标注数据** → annotated real videos，提供 object counts、bbox、room dimensions 等可靠几何信息。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

- **模拟数据扩覆盖** → ProcTHOR \+ Hypersim，提升 layout / config / appearance diversity。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

- **无标注真实视频伪标注** → Grounding\-DINO \+ SAM2 \+ VGGT \+ mask erosion 的 pseudo\-annotation。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

- **模板生成** → 论文定义 12 类 spatiotemporal question types，主轴就是 size / direction / count / distance / appearance order。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

你现在完全可以把这个 pipeline 抽象成：

**Geometry sources → normalized scene graph / object set / point set → QA template engine**

这会比“数据集堆砌”更适合你后续自己实现。

---

## 题型设计

论文明确说，他们定义了 **12 个 question types**，并沿几条轴组织：

- 属性类型：size / direction / count / distance / appearance order

- relative vs absolute

- perspective：camera perspective / object perspective

- modality：image only / video required

- group：configuration / measurement / spatiotemporal。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

一些模板例子：

- size relative：两个物体谁更大

- size absolute：物体高度是多少

- room size absolute：房间大小是多少

- direction camera：从相机视角看物体在左还是右

- direction object：从某物体面向另一物体时，第三个物体在左/右/后

- count relative：A 是否比 B 少

- count absolute：一共有多少个 object

- distance camera：谁离 camera 更近

- distance object：谁离参考物更近

- appearance order：多个物体在视频中首次出现的顺序。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

这里有两个特别有价值的设计点：

### 第一，relative / absolute 都做

这很重要。

因为 relative 更像判别式空间关系，absolute 更像 metric grounding。两者一起做，模型学到的 spatial representation 会更完整。\(ar5iv\)

### 第二，camera perspective 和 object perspective 都做

这等于把“视觉坐标系”和“对象中心参考系”都纳入训练。

对 direction / distance 这种题，这会明显增加难度和几何泛化性。\(ar5iv\)

如果你后面自己 build pipeline，我建议你不要只抄 5 大类，而是把这几个“轴”也实现出来。
真正决定数据质量的，往往不是大类名，而是这些**factorized taxonomy**。

---

## 数据混合结果说明了什么

作者做了 data source ablation，结论很明确：

- **Full Mix** 最好

- 单独来源里，效果排序大致是
**annotated real videos \> simulated data \> pseudo\-annotated images**。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

论文还特别指出：

- 视频数据比静态图像对 spatial reasoning 更有信息量

- temporal continuity 和 multi\-view diversity 对 robust spatial representations 很关键。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

这对你现在的任务有两个直接启发：

### 启发 1

**真实高质量几何数据** 是最值钱的，不只是“有就行”，而是整个 mixture 的性能支点。

### 启发 2

伪标注 web video 当然有用，但更像是 **coverage booster**，不是性能主引擎。
所以你做 pipeline 时，优先级应该是：

**先把高质量 source 统一好，再考虑大规模 pseudo\-label 扩容。**

---

## 为什么 Cambrian\-S 对你有参考价值，但又不能直接照搬

论文结论是：

- VSI\-590K \+ Cambrian\-S 确实能把 **VSI\-Bench** 推到 SOTA 级别

- 但 **VSI\-SUPER** 依然很难，说明“多数据 \+ 长上下文 \+ 更强 SFT”还不够。\([ar5iv](https://ar5iv.org/pdf/2511.04670)\)

这意味着它给你的价值主要在两层：

### 第一层：工程层

它给了一个非常清晰的 **spatial data engine 配方**：

- source mixture

- pseudo\-3D annotation

- geometry\-driven QA generation

- factorized template taxonomy

这部分很值得你借鉴。

### 第二层：研究层

它也告诉你这个方向的上限：

> 只做数据、只做 instruction tuning、只做更多帧，
> 最后还是会碰到范式天花板。
> 
> 

这其实和你在 LIMO4SI 里的观察是相通的：

单纯堆更多视觉 token、更多帧，并不自动等于更强空间智能。

---

## 如果把这篇论文翻译成你这两天的具体任务

你现在的两个任务是：

- Analyze VSI\-590K

- Build the data curation pipeline

那你读 Cambrian\-S 时，最该产出的不是“论文总结”，而是下面这两个东西。

### A\. 对 VSI\-590K 的分析框架

你可以按这 5 个维度分析：

1. **source composition**
annotated real / simulated / pseudo\-real 各占多少、各自负责什么能力

2. **label fidelity**
真 3D 标注 vs 伪 3D 标注的噪声结构是什么

3. **question taxonomy**
12 类题型在 attribute / rel\-abs / perspective / modality / group 上如何分布

4. **geometry dependence**
哪些题型强依赖高质量几何，哪些更适合伪标注扩容

5. **video vs image balance**
哪些能力必须靠视频，哪些图像也能覆盖

### B\. 对 data curation pipeline 的工程拆分

你可以把 pipeline 拆成 6 个模块：

1. **source ingest**
各数据集统一成 scene/object/meta 格式

2. **geometry normalization**
bbox / masks / point sets / room stats 统一表示

3. **pseudo\-annotation**
frame sampling → blur filtering → Grounding\-DINO → SAM2 → VGGT → mask refinement

4. **QA eligibility check**
场景里对象是否足够、几何是否稳定、是否满足某模板约束

5. **template engine**
根据 taxonomy 生成问题、答案、选项

6. **mixture balancing**
控制 source、question type、difficulty、video/image 的配比

这其实就是你说的 **Data Engine Agent** 的最自然落地方式。

---

做 VSI\-590K 分析和数据管线是在**夯实 spatial cognition 的数据基础**，不是一步到位解决 supersensing。



# 26\.4\.1

## **配置**

- `configs/experiments.py`

    - model: `Qwen3.5-9B`

    - bench: `vsi_bench` / `openeqa` / `omnispatial`

    - 视角筛选维度：`level`（L0/L1/L2/L3）、`num_candidate_frames`、`num_selected_frames`、`diversity_lambda`、`coarse_keep_ratio`

    - 当前共有 18 组实验配置（`ALL_EXPERIMENTS`）

## 全部 User Prompt（当前代码）

### VSI\-Bench

\*\*NA（数值题）\*\*三行：

1. `These are frames selected from a video.`

2. `<question原文>`

3. `Please answer the question using a single word or phrase.`

\*\*MCA（多选题）\*\*四行：

1. `These are frames selected from a video.`

2. `<question原文>`

3. `Options:\n<options按行拼接>`

4. `Answer with the option's letter from the given choices directly.`

---

### OpenEQA

四行：

1. `These are frames from an indoor environment video.`

2. `Based on what you can observe in these frames, answer the following question.`

3. `<question原文>`

4. `Answer concisely in a few words.`

---



## 实验目的

本实验旨在验证：在视觉问答任务中，先对候选视图（视频帧或图像裁剪）做筛选，再输入 VLM，是否能在可控成本下提升问答性能。

核心关注三点：

- 有效性：视图筛选能否提升准确率（VSI 的 Wavg/MCA/NA\-MRA，OmniSpatial 的 Acc，OpenEQA 的 F1/contain）。

- 效率：在相同或更低输入 token 成本下，能否达到更好效果。

- 机制：比较从“无筛选”到“相关性\+多样性\+两阶段精排”的逐级策略，分析性能变化来源。

> *方法本质是 问题驱动（question\-based）的输入级视图筛选，不是模型内部 token pruning。*
> 
> 

---

## Level 0–3 实验结构

#### Level 0：无智能筛选（Baseline）

- 做法：直接使用候选视图，不做 CLIP 排序。

    - 视频任务：均匀抽帧后直接用。

- 作用：作为对照组，衡量后续筛选策略净收益。

#### Level 1：相关性筛选（CLIP Top\-k）

- 做法：对每个候选视图计算 `sim(image, question)`，按分数取 Top\-k。

- 直觉：优先保留和问题语义最相关的视图，去掉无关视图。

- 特点：只考虑“与问题的相关性”，不显式抑制重复视图。

#### Level 2：相关性 \+ 多样性（MMR）

- 做法：在 Level 1 的相关性基础上，加入冗余惩罚： \[ score\_i = relevance\_i \- \\lambda \\cdot redundancy\_i \] 其中 `redundancy_i` 是与已选视图的最大相似度。

- 直觉：既要“看对地方”，也要“别都看同一处”。

- 可控参数：`λ`（多样性权重），越大越倾向去重。

#### Level 3：两阶段粗到细（Coarse\-to\-Fine）

- 做法：

    1. 粗筛：先用全局 CLIP 分数从 N 个候选里保留一部分（`coarse_keep_ratio`）。

    2. 精排：在保留子集上，用多尺度特征 \+ MMR 再选最终 k 张。

- 直觉：先快速缩小搜索空间，再做更细致的质量筛选。

- 目标：在较大候选集下提升筛选质量与稳定性。

## 已完成实验表格（当前成功 7 组，1 组在跑）

### 2\.1 VSI\-Bench（已出分 3 组）

### 2\.2 失败/未出分/进行中

- 进行中：`L3_vsi_c64_k8`（当前 `results_gpu0.jsonl` 已 2056 条，尚未生成 `scores.json`）

- 未开始或未完成（共 10 组）：

    - `L0_openeqa_k8`

    - `L1_openeqa_c32_k8`

    - `L1_openeqa_c64_k8`

    - `L1_vsi_c32_k16`

    - `L1_vsi_c32_k4`

    - `L1_vsi_c64_k8`

    - `L2_openeqa_c32_k8_lam3`

    - `L2_vsi_c32_k8_lam1`

    - `L2_vsi_c32_k8_lam5`

    - `L3_openeqa_c64_k8`

---

## 结果分析（当前阶段）

1. VSI：L1/L2 比 L0 的 Wavg 有小幅提升

    - `0.2539 → 0.2646`（\+0\.0107）

    - 但结构上是 MCA 下降、NA\(MRA\)上升，说明收益主要来自数值题。

2. VSI：L1 与 L2（λ=0\.3）Wavg 持平

    - L2 进一步提高了 NA（0\.4104 vs 0\.4032），但 MCA 又降了一点，最终抵消。

3. Token 成本

    - VSI 当前这 3 组 `avg_input_tokens` 都是 2501（因为最终都喂 8 张图）。

筛选策略更偏向“保留与问题相关的局部证据”，这对数值/测量类问题更有帮助；但对 MCA 这类可能依赖更广上下文或细粒度辨别的问题，过强筛选可能损失背景信息。

## **后续：**

把 OpenEQA 跑出来，更像“给定问题，从场景中找证据”



先补齐最关键对照：

- uniform 8 

- random 8 

- CLIP top\-8 

- MMR 8 

- coarse\-to\-fine 8 

再和：

- uniform 16 

- uniform 32 

比较。



# 26\.3\.30

## 配置

- `configs/experiments.py`

    - model: `Qwen3.5-9B`,bench: `vsibench`

    - 定义实验参数维度：`input_type`（video\_native / frames\_as\_images）、`num_frames`（4/8/16/32/64）、`sampling_method`（uniform/fps/keyframe）

### User prompt

**NA（数值题）**用户文本是三行：

1. `These are frames of a video.`

2. `<question原文>`

3. `Please answer the question using a single word or phrase.`

**MCA（多选题）**用户文本是四行：

1. `These are frames of a video.`

2. `<question原文>`

3. `Options:\n<把 options 按行拼起来>`

4. `Answer with the option's letter from the given choices directly.`



## 已完成实验表格（当前成功 9 组；失败 1 组仍 pending）

> *表内指标来自各自的 **`results/<exp>/scores.json`**，其中 **`Wavg`** 为加权平均（代码按 MCA 与 NA 数量加权），**`Tokens`** 为该实验的平均 **`input_tokens`**。*
> 
> 

失败/未出分：

- `input_video_native_8f`（video\_native）— 无 `scores.json`

---

## 结果分析

1. 总体趋势（帧数越多，Wavg 越高，但成本线性上升）

    - baseline\(8帧\) `Wavg=0.2479`

    - 32帧 `Wavg=0.3121`（提升明显）

    - 64帧 `Wavg=0.3468`（继续提升，但每增加一档帧数，成本会更快增长）

    - 成本方面，Tokens 从 2492 → 9740 → 19404（约 1x、3\.9x、7\.8x）

2. 16 帧固定时：FPS 抽样略优，keyframe 最差

    - `sample_fps2_16f`（Wavg 0\.2915） \> `sample_uniform_16f`（0\.2875） \> `sample_fps1_16f`（0\.2807）

    - `sample_keyframe_16f`（0\.2633）显著低，尤其 NA 的 MRA 掉得比较多（0\.3788）

3. 按题型看：关键帧策略对数值题更不友好

    - `sample_keyframe_16f` 的 MCA=0\.1361 并不算最差，但 NA MRA=0\.3788 明显更低

    - 关键帧启发式抓到“变化大”的时刻，但数值题（计数/距离/面积等）可能需要更稳定的跨时序观察与更均匀的时间覆盖。

4. 4 帧在 MCA 端还行，但数值题很吃亏

    - `nframes_4_uniform`：MCA 0\.1246 并不差，但 NA MRA 0\.3221，导致整体 Wavg 低（0\.2281）

    - 说明任务里数值题对时间/视角覆盖更敏感。

---

## 后续计划：

1. 没成功的实验：`input_video_native_8f` 已经修改环境依赖，等卡中

2. keyframe代码我将再修改一下，找点更适配的任务，现在的还不够合理

3. 正在仔细看qwen3\.5代码、visual input、TimeSformer、VideoMAE、Multiscale Vision Transformer、Token pruning相关论文，我想再想想实验到底怎么做更好。



