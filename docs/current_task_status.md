# Current Task Status

最近更新：2026-09-09。

本文档是当前 benchmark 的唯一集中状态记录。当前发布新版 **Task 1、Task 3、Task 4**，并先发布 2 个约 9 秒的高质量 Task 5 pilot case；网站只是验收入口，答案来自代码与保存的几何/视觉证据。

## 当前交付

| 项目 | 当前真实状态 |
|---|---:|
| 展示 case | 32 |
| QA | 32 |
| 每个 case 的问题数 | 1 |
| 不重复的视频窗口 | 32 / 32 |
| Task 1 | 11 case；四类能力均至少 2 例 |
| Task 3 | 6 case；三类能力均 2 例 |
| Task 4 | 13 case；六类能力均至少 2 例 |
| Task 5 | 2 个约 9 秒高质量 pilot case；gaze onset、最后 gaze 物体各 1 例；重复 gaze 暂缺 |
| 每题选项 | 4 个互不重复选项 |
| 视频长度 | Task 1/4 约 15 秒；Task 3 为 30 秒；当前 Task 5 pilot 为 8.5–10 秒 |
| 非 Task 1 / Task 3 / Task 4 / Task 5 题 | 0 |

最终数据：

- `outputs/qa/task1_task4_curated_qa.jsonl`
- `outputs/qa/task1_task4_curated_audit.json`
- `outputs/qa/task3_scaled_qa.jsonl`
- `outputs/qa/task3_static_landmarks.json`
- `outputs/qa/task3_scale_audit.json`
- `outputs/qa/task5_scaled_qa.jsonl`
- `outputs/qa/task5_adt_analysis.json`
- `outputs/qa/task5_scale_audit.json`
- `outputs/qa/task1_task3_task4_task5_scale_quality.json`
- `site/qa_benchmark/data.js`
- `site/qa_benchmark/index.html`

生成入口：

```bash
# 日常重建使用已审计的静态地标缓存
python scripts/build_all_scaled_qa.py

# 只有修改 Task 3 地标框时才重跑原始点云落点
python scripts/ground_task3_landmarks.py
python scripts/build_all_scaled_qa.py
```

## Task 1：Dynamic Human-Referenced Relations

| 问题类型 | case | 证据 | 当前状态 |
|---|---|---|---|
| 人参照关系总体变化 | Ego-Exo4D `iiith_cooking_145_2` | 29 个有效人体 3D pose；物体固定 3D 中心；逐时刻 body-centric 变换 | 已发布，高置信 |
| 转身导致的相对方位变化 | Ego-Exo4D `iiith_cooking_30_1` | 10 个有效 pose；约 56.7° body turn；净位移约 0.20 m | 已发布，高置信 |
| 全程关系保持 | Ego-Exo4D `iiith_cooking_32_1` | 21 个有效 pose；要求目标在每个采样时刻都位于人的左侧 | 已发布，高置信 |
| body/head-forward 可见性变化原因 | Ego-Exo4D `sfu_cooking_008_3` | 31 个有效 pose；目标方向与人体 forward 的夹角；候选 blocker 检查 | 已发布，但标为 `audited_proxy` |

Task 1 的可见性题回答的是“目标是否进入人的 body/head-forward 视场代理”，不是眼动真值。当前没有把缺少 dense blocker geometry 的样例写成“无遮挡”。

新增 6 个 Task 1 temporal case：

| 动态模式 | 数据窗口 | 整段证据 |
|---|---|---|
| 最近物体全程保持 | `iiith_cooking_145_2` frame 7380 | white chopping board 在 31/31 个采样中最近 |
| 多物体全程在人前方 | `iiith_cooking_29_3` frame 4350 | 3 个对象在 26/26 个采样中均为 front |
| 最近且始终低于人体原点 | `iiith_cooking_31_3` frame 1620 | chopped tomato 在 19/19 个采样中同时满足 |
| 左右关系翻转 | `sfu_cooking_010_1` frame 5460 | white plate 从人的 left 变为 right |
| 前后关系翻转 | `sfu_cooking_007_3` frame 10890 | oyster-sauce bottle 从 front 变为 behind |
| 人物体距离显著增加 | `uniandes_cooking_001_5` frame 6690 | egg whisk 距离约 0.55 m 增至 1.86 m |

## Task 3：Human–Scene Topological Reasoning

Task 3 回答人的完整三维轨迹与静态场景地标之间的关系，不使用图像左右，也不把可移动厨具当作场景地标。

| 类别 | 例数 | 计算与发布门槛 |
|---|---:|---|
| `local_path_side` | 2 | 在最近点处用平滑轨迹的局部切向量定义左右；横向 margin ≥ 0.25 m，局部移动 ≥ 0.35 m |
| `temporal_landmark_order` | 2 | 分别计算两个固定地标对完整轨迹的最近点；两次访问均须为片段内部的显著靠近事件，时间间隔 ≥ 2 s |
| `full_route_proximity` | 2 | 比较至少三个固定地标到完整轨迹的最小水平距离；第一、第二名 margin ≥ 0.15 m |

静态地标采用人工核验的固定设施框（操作台、餐桌、固定设备等），再由 Ego-Exo4D 半稠密点云计算世界坐标。每个发布地标必须通过：`manual_static_review = true`、稳健三维内点数 ≥ 12、三维中心回投仍在人工框内。答案完全由 annotation/点云和人体轨迹计算，LLM 不参与判题。当前没有 room region、doorway footprint 或 obstacle mesh，因此不发布“进入哪个房间”“从障碍哪侧绕过”等超出证据的问题。

## Task 4：Multi-Human Relational Dynamics

| 类别 | 问题类型 | case | 证据与门槛 | 当前状态 |
|---|---|---|---|---|
| position | `position_consistency_between_people` | HOI-M3 `bedroom_data02_win08` | 16 个 metric pose；B 在全部采样中都位于 A 的右前方 | 已发布 |
| orientation | `dominant_facing_relation_over_video` | HOI-M3 `bedroom_data01_win06` | 16 个 metric pose；14/16 为 facing each other | 已发布 |
| distance | `metric_separation_over_video` | HOI-M3 `bedroom_data01_win02` | 16 个 pelvis/root 状态；约 1.49 m 增至 3.55 m | 已发布 |
| topology | `visible_pair_topology_change_2d` | HOI-M3 `bedroom_data05_win02` | Grounding DINO + 时序关联覆盖 3 人；比较归一化图像平面 pair distance | 已发布，明确为 2D topology |
| visibility | `body_forward_visibility_consistency` | HOI-M3 `bedroom_data02_win05` | 16 个 body-forward 样本；双方均落在彼此 ±60° forward field | 已发布，但标为 `audited_proxy` |
| relation change | `body_centric_relation_change_over_video` | HOI-M3 `bedroom_data03_win03` | 前 4 个样本 left-front，之后 12 个样本持续 right-front | 已发布 |

新增 6 个 Task 4 temporal case：

| 动态模式 | HOI-M3 窗口 | 主要证据 |
|---|---|---|
| 持续分离 | `bedroom_data01_win02` | 1.49 m → 3.55 m，后半段持续大于 3 m |
| 主导方位 | `bedroom_data01_win04` | B 在 13/16 个采样中位于 A 的 right-behind |
| 先分离再部分靠近 | `bedroom_data02_win04` | 1.24 m → 2.68 m 峰值 → 2.08 m |
| 面对时靠近 | `bedroom_data02_win06` | facing 14/16；最近约 0.96 m |
| 靠近并从右侧跨到左侧 | `bedroom_data03_win09` | 2.72 m → 1.14 m，同时 right-front → left-front |
| 近—远—再近 | `bedroom_data03_win08` | 中段约 2.41 m，结尾约 0.68 m |

所有米制/朝向/A-centered Task 4 case 必须同时满足：

1. 原视频中持续可见人物数与本地 SMPL-X 数量一致；
2. 2D 可见轨迹与 metric A/B 的运动轮廓通过身份对齐门槛；
3. 每个 15 秒窗口使用 16 个 metric pose，而不是旧版仅 3 个时刻；
4. 三人但只有两条 SMPL-X 的窗口不能回答三人米制问题，只能回答覆盖全部三人的 2D topology。

## Task 5：Human-State–Grounded Spatial Reasoning

**当前 pilot 发布要求（2026-09-09）**：每个公开视频必须具有真实 annotation 覆盖的 8.5–10 秒窗口，目标为 9 秒；`between_repeated_gaze_events` 中第一次 gaze 结束到第二次 gaze 开始必须至少间隔 2 秒。不得通过循环、补帧、静帧或黑帧凑时长。完整 ADT 下载权限到位后再扩展数量和场景。

Task 5 当前使用 Aria Digital Twin v2 的官方完整样本，不把 Task 1/4 的 body-forward proxy 当作 gaze。逐帧答案链路为：真实 `eyegaze.csv` 射线与 fixation depth → 同时刻物体 6DoF/3D OBB 包含检验 → 世界坐标中的物体中心 → 重力对齐的 wearer CPF 左右/前后坐标。动态物体使用逐帧位姿，不假设物体固定；LLM 不参与答案标签判断。

| 类别 | 例数 | 计算与发布门槛 |
|---|---:|---|
| `between_repeated_gaze_events` | 0 | 同一物体两个稳定 gaze 段、每段至少 0.3 s / 10 个直接命中，间隔至少 2 s；当前数据没有合格候选，故不强行发布 |
| `after_gaze_turns_to_object` | 1 | gaze onset 前不在注视目标；onset 后至少持续 0.3 s / 10 个直接命中；wearer 水平转向至少 8°，目标横向变化至少 0.20 m |
| `last_gaze_annotated_object` | 1 | 目标必须是约 9 秒窗口内最后一个持续 gaze 事件，并从完整窗口重算其关系序列 |

当前 2 个 pilot 窗口来自同一官方 10 秒端到端验证序列，分别覆盖 gaze onset 和最后 gaze 物体。旧的短促 gaze、小位移和大面积支撑物样例已撤下；当前数据没有同时满足两段稳定 gaze 与至少 2 秒间隔的 repeated-gaze 候选，因此该类保持为空。批量扩展时，`scripts/scale_task5_adt.py` 可直接遍历更多 ADT 序列；独立门禁会拒绝 head-forward 伪 gaze、非连续单帧命中、大型支撑物目标、图像左右、陈旧关系标签和信息量不平衡选项。

Task 5 的官方原始样本下载和 VRS 解析只在重新挖掘/导出证据时需要；日常网站重建使用已保存的 `task5_adt_analysis.json`、短视频和定位图。当前 Ego-Exo4D 账号对 gaze manifest 返回 403，因此未把其 gaze 写入答案，也没有用推测结果补齐。

### Task 5 批量 scale pipeline

Task 5 已从手工单序列配置升级为可直接遍历多个 ADT 序列的 fail-closed pipeline：

```bash
# 先只挖掘、自动选例和检查配额，不修改网站
.venv/bin/python scripts/scale_task5_adt.py /path/to/unpacked_adt \
  --target-per-category 100 --plan-only --reuse-analysis

# 确认 pipeline_report.json 后正式导出媒体、生成 QA、更新网站并执行合并门禁
.venv/bin/python scripts/scale_task5_adt.py /path/to/unpacked_adt \
  --target-per-category 100 --reuse-analysis --reuse-media
```

一条命令会执行：发现完整序列 → 时间对齐 gaze/wearer/object annotation → gaze ray + fixation depth/OBB 包含检验 → 持续 gaze 事件 → 三类候选自动挖掘 → 跨序列和物体平衡选择 → 视频/定位证据 → QA → 独立发布门禁。关键输出位于 `outputs/qa/task5_scale/`。单个坏序列会记录为 `rejected_sequence` 并继续处理其他序列；任一类别数量不足、证据导出失败或质量门禁失败时，不会静默发布。

此前的三个通用生成要求在 Task 5 中均已编码：

1. **选项信息量一致**：所有 transition 选项严格使用两个关系槽，sequence 选项严格保持相同状态数；共享门禁再次检查词数、数字槽、时间槽和关系槽。
2. **距离不伪精确**：当前 Task 5 发布答案不使用米制距离；以后若加入，公共门禁最多允许 1 位小数。完整精度只保留在隐藏证据 JSON 中。
3. **样例规则进入算法**：同一物体两次 gaze、gaze onset 前后变化、最后 gaze 物体三类均由 annotation 自动提候选，不再依赖手写答案；当前只发布通过强化门槛的 onset 与 last-gaze case，完整数据到位后由 `--target-per-category` 扩展到每类至少 2 例。

scale 门槛还包括：公开视频 8.5–10 秒、重复 gaze 间隔 ≥ 2 秒、wearer 对齐误差 ≤ 10 ms、动态物体位姿误差 ≤ 50 ms、fixation depth 到目标 OBB 的残差 ≤ 0.05 m、repeated/onset gaze event 至少 0.3 秒且至少 10 个直接命中、last-gaze event 至少 0.1 秒且至少 4 个直接命中、event 命中支持率 ≥ 80%、内部缺失最多 1 个状态、关系横向变化 ≥ 0.20 m，gaze onset 题还要求 wearer 转向 ≥ 8°。关系阶段必须至少稳定 6 帧；桌面、地板、墙面、天花板和大型柜体等支撑/场景目标不得作为答案对象。

## 本轮关键修正

- 当前合并发布为 32 case / 32 题；Task 1/4 为 24 个约 15 秒窗口，Task 3 为 6 个 30 秒窗口，Task 5 为 2 个约 9 秒 gaze pilot 窗口，每个 case 只保留一道高信号题。
- Task 3 已以新的静态场景地标拓扑定义恢复；Task 2、reachability、hand-approach 仍不在当前发布范围。
- 删除同一窗口重复 case；32 个 case 使用 32 个不同视频窗口；同一源序列内的窗口也以起止时间共同定义并使用独立 clip 文件名。
- HOI-M3 每段 metric timeline 从 3 个时刻提高到 16 个时刻，约 1 Hz 覆盖完整 15 秒。
- 人体朝向、A-centered 左右前后均投影到真实 X/Z 地面，不让竖直分量干扰。
- 没有 blocker geometry 时，`line_of_sight_blocked` 现在为 `null`，状态为 `missing_blocker_geometry`，不再错误输出 `False = clear`。
- 新增 body-forward field，名称和解释都明确说明它不是 gaze 或物理无遮挡真值。
- 网站只显示与当前题相关的时间线、定位和俯视证据；答题后才显示正确答案。

## 尚未完成 / 不得虚报

| 能力 | 当前缺口 | 当前处理 |
|---|---|---|
| 人绕过隔板后 A/B 是否真正互相可见 | 本地 HOI-M3 子集没有隔板/场景 mesh 的时变遮挡几何与相机对齐标注 | 不发布此类题；结果为 `missing_blocker_geometry` |
| Task 4 的“看着对方”或 mutual gaze | HOI-M3 当前仍只有 SMPL-X root/body forward；ADT 的 wearer gaze 不能补成两人 mutual gaze | Task 4 只称 body-forward field；Task 5 才使用真实 wearer gaze |
| 三人米制距离/身体朝向 | `bedroom_data05` 画面有 3 人，但本地只有 2 条 SMPL-X | 只发布覆盖三人的 2D topology |
| 完整 SMPL-X 关节头部位置 | 当前本地只有参数文件，未装载受许可的人体模型文件 | pelvis 使用 transl/root；head 明确标为 `pelvis + 1.6 m proxy` |

因此，当前可以真实地说：**合并发布包含 32 个合格唯一窗口：Task 1 为 11 例、Task 3 为 6 例、Task 4 为 13 例、Task 5 为 2 个约 9 秒 pilot。Task 5 使用真实 wearer gaze；现有数据仅支持 1 个高质量 gaze-onset case 和 1 个 last-gaze case，repeated-gaze 不合格样例没有继续发布。完整 ADT 权限到位后仍需扩展到每类至少 2 例。Task 4 mutual gaze、物理遮挡和语义房间拓扑仍未完成。**

## 验收

启动网站：

```bash
cd /root/autodl-tmp/LIMO4SI
./serve_qa_site.sh
```

浏览器打开：

```text
http://<服务器IP>:8000/
```

自动检查：

```bash
python scripts/build_all_scaled_qa.py
```

合并审计文件 `outputs/qa/task1_task3_task4_task5_scale_quality.json` 应满足：

- `status = ok`
- `case_count = accepted_count = 32`
- `rejected_count = 0`

Task 3 子审计 `outputs/qa/task3_scale_audit.json` 还应满足：6 个唯一窗口，`local_path_side`、`temporal_landmark_order`、`full_route_proximity` 各 2 例。

Task 5 子审计 `outputs/qa/task5_scale_audit.json` 当前应满足：2 个唯一窗口，`after_gaze_turns_to_object`、`last_gaze_annotated_object` 各 1 例；`between_repeated_gaze_events` 为 0，并在 `known_deficit` 中明确记录现有数据不足。

## 工作原则

- 答案必须来自代码和证据，不靠猜。
- 缺证据就拒绝发布，不用 proxy 冒充真值。
- proxy 必须在问题、方法和审计中明确命名。
- 网站只是展示层；JSONL、审计和可复用生成脚本才是主交付。
- 后续只维护本文档，不恢复或新增零散状态 Markdown。
