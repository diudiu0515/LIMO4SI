# Task 1 / Task 4 更新后工作总结

最近更新：2026-09-02。

> 历史阶段说明：本文记录 Task 1 / Task 4 切换阶段的工作；当前发布已加入新版 Task 3。最新数量、门槛与命令以 `current_task_status.md` 为准。

## 目标调整

Benchmark 已从旧版 Task 1 / Task 3 调整为新版 Task 1 / Task 4：

- Task 1：Dynamic Human-Referenced Relations，关注人体运动或转身过程中，人—物空间关系的变化与保持。
- Task 4：Multi-Human Relational Dynamics，关注多人之间的位置、朝向、距离、拓扑、可见方向代理和关系变化。
- 删除不属于当前范围的 Task 2、Task 3、reachability、hand-approach 等问题。
- 每个独立 15 秒视频窗口只保留一道高信号、可审计的问题。

## 数据与证据链

- Task 1 使用 Ego-Exo4D 的多帧人体 3D pose、物体 3D 中心和 body-centric 坐标变换。
- Task 4 引入 HOI-M3 多人数据，并建立 SMPL-X root/body-forward 的时序关系计算。
- Task 4 米制题统一使用覆盖约 15 秒的 16 个 pose 样本，不再只比较少量离散时刻。
- 加入可见人物数量、2D 轨迹与 metric A/B 身份对齐门控。
- 三人但只有两条 SMPL-X 的场景只生成覆盖三人的 2D topology 问题，不生成三人米制问题。
- 缺少 blocker geometry 时，遮挡状态保持 unknown，不把缺失证据解释成无遮挡。

## 高质量问题模板与覆盖

当前发布 25 个 case、25 道题，使用 25 个不同视频窗口；每题有 4 个互不重复的选项。

Task 1 共 11 个 case：

- 4 个核心类型：总体关系变化、转身引起的方位变化、全程关系保持、body-forward 可见方向变化原因。
- 6 个扩展 temporal 模式：最近物体保持、多物体前方保持、最近且垂直关系保持、左右翻转、前后翻转、人物距离显著变化。

Task 4 共 14 个 case：

- 六类核心能力均已覆盖：position、orientation、distance、topology、visibility、relation change。
- 另有 6 个动态模式：持续分离、主导方位、先分离再靠近、面对时靠近、靠近并横跨人体左右、近—远—再近。

最终发布与审计产物：

- `outputs/qa/task1_task4_curated_qa.jsonl`
- `outputs/qa/task1_task4_curated_audit.json`
- `site/qa_benchmark/data.js`
- `site/qa_benchmark/index.html`

审计要求为：`status = ok`、25 case、25 QA、每个 case 一题、25 个唯一窗口，并且 Task 1 四类、Task 4 六类能力均至少有 2 个独立样例。

## 网站可视化

- 网站当前只展示 Task 1 和 Task 4。
- 每个 case 提供约 15 秒视频、人体/对象定位图、俯视空间图和与当前问题直接相关的时序证据。
- 用户先选择答案并提交，之后才显示正确答案、解释和计算证据。
- 静态 HTML 与交互版共用同一份 curated 数据，媒体路径已完成检查。
- 根目录 `index.html` 提供入口，`serve_qa_site.sh` 可启动本地验收服务。

## 关键实现

- `scripts/build_task1_task4_curated.py`：生成新版 22-case curated benchmark 与硬门控审计。
- `scripts/build_task1_task4_scaled.py`：补齐能力类别、修正文案并执行每类至少两例的发布硬门控。
- `scripts/build_static_qa_site.py`：生成可直接验收的静态网站。
- `scripts/build_multihuman_dynamic_qa.py`、`scripts/calibrate_multihuman_video_evidence.py`：构建和校准多人时序证据。
- `src/limo4si/multihuman.py`：多人距离、朝向和 body-centric 关系计算。
- `configs/object_display_aliases.json`：人工核验后的对象展示名称映射。

## 尚未完成的能力

- 没有场景 mesh、隔板几何和相机对齐证据，因此没有发布物理遮挡/无遮挡问题。
- body-forward visibility 是方向视场代理，不是真实 gaze 或 eye tracking。
- 当前本地三人场景缺少第三人的 SMPL-X，三人米制距离与身体朝向尚未发布。
- 当前 head 位置仍是明确标注的 proxy，没有冒充完整 SMPL-X head joint 真值。

因此，当前可验收范围已经完成；真实物理遮挡、gaze 和三人 metric 关系仍属于后续工作。
