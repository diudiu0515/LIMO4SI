# Current Task Status

最近更新：2026-09-02。

本文档是当前 benchmark 的唯一集中状态记录。当前只做新版 **Task 1** 和 **Task 4**；网站只是验收入口，答案来自代码与保存的几何/视觉证据。

## 当前交付

| 项目 | 当前真实状态 |
|---|---:|
| 展示 case | 25 |
| QA | 25 |
| 每个 case 的问题数 | 1 |
| 不重复的视频窗口 | 25 / 25 |
| Task 1 | 11 case；四类能力均至少 2 例 |
| Task 4 | 14 case；六类能力均至少 2 例 |
| 每题选项 | 4 个互不重复选项 |
| 视频长度 | 每段约 15 秒 |
| 非 Task 1 / Task 4 题 | 0 |

最终数据：

- `outputs/qa/task1_task4_curated_qa.jsonl`
- `outputs/qa/task1_task4_curated_audit.json`
- `site/qa_benchmark/data.js`
- `site/qa_benchmark/index.html`

生成入口：

```bash
python scripts/build_task1_task4_curated.py
python scripts/build_task1_task4_scaled.py
python scripts/build_static_qa_site.py
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

## Task 4：Multi-Human Relational Dynamics

| 类别 | 问题类型 | case | 证据与门槛 | 当前状态 |
|---|---|---|---|---|
| position | `position_consistency_between_people` | HOI-M3 `bedroom_data02_win08` | 16 个 metric pose；B 在全部采样中都位于 A 的右前方 | 已发布 |
| orientation | `dominant_facing_relation_over_video` | HOI-M3 `bedroom_data01_win06` | 16 个 metric pose；14/16 为 facing each other | 已发布 |
| distance | `metric_distance_pattern_over_video` | HOI-M3 `bedroom_data03_win04` | 16 个 pelvis/root 样本；约 1.08 m 增至 3.09 m | 已发布 |
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

## 本轮关键修正

- 在首版 10 case / 10 题的基础上新增 12 个通过门控的 temporal case；当前为 22 case / 22 题，每个 case 仍只保留一道高信号题。
- 删除 Task 2、Task 3、reachability、hand-approach 等不在当前范围内的问题。
- 删除同一窗口重复 case；22 个 case 使用 22 个不同的 15 秒窗口。
- HOI-M3 每段 metric timeline 从 3 个时刻提高到 16 个时刻，约 1 Hz 覆盖完整 15 秒。
- 人体朝向、A-centered 左右前后均投影到真实 X/Z 地面，不让竖直分量干扰。
- 没有 blocker geometry 时，`line_of_sight_blocked` 现在为 `null`，状态为 `missing_blocker_geometry`，不再错误输出 `False = clear`。
- 新增 body-forward field，名称和解释都明确说明它不是 gaze 或物理无遮挡真值。
- 网站只显示与当前题相关的时间线、定位和俯视证据；答题后才显示正确答案。

## 尚未完成 / 不得虚报

| 能力 | 当前缺口 | 当前处理 |
|---|---|---|
| 人绕过隔板后 A/B 是否真正互相可见 | 本地 HOI-M3 子集没有隔板/场景 mesh 的时变遮挡几何与相机对齐标注 | 不发布此类题；结果为 `missing_blocker_geometry` |
| “看着对方”或 mutual gaze | 当前使用 SMPL-X root/body forward，没有真实眼动或可靠 head gaze | 只称 body-forward field，不称 gaze |
| 三人米制距离/身体朝向 | `bedroom_data05` 画面有 3 人，但本地只有 2 条 SMPL-X | 只发布覆盖三人的 2D topology |
| 完整 SMPL-X 关节头部位置 | 当前本地只有参数文件，未装载受许可的人体模型文件 | pelvis 使用 transl/root；head 明确标为 `pelvis + 1.6 m proxy` |

因此，当前可以真实地说：**Task 1 已有 10 个可验收 temporal case，Task 4 已有 12 个可验收 temporal case；六类 Task 4 能力均有覆盖。物理遮挡与 gaze 真值仍未完成，不算进“已完成”。**

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
python scripts/build_task1_task4_curated.py
python scripts/build_task1_task4_scaled.py
python scripts/build_static_qa_site.py
```

审计文件 `outputs/qa/task1_task4_curated_audit.json` 应满足：

- `status = ok`
- `case_count = 25`
- `qa_count = 25`
- `one_question_per_case = true`
- `unique_case_windows = 25`
- Task 1 = 11，Task 4 = 14；每个能力类别至少 2 例

## 工作原则

- 答案必须来自代码和证据，不靠猜。
- 缺证据就拒绝发布，不用 proxy 冒充真值。
- proxy 必须在问题、方法和审计中明确命名。
- 网站只是展示层；JSONL、审计和可复用生成脚本才是主交付。
- 后续只维护本文档，不恢复或新增零散状态 Markdown。
