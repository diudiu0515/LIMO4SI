# LIMO4SI

# Summary

### 第一层：数据解析（基本完成）

已经完成了 Relation 数据的解析、QA 候选生成、证据抽取和人工 Review 页面，可以稳定从 Relation 标注自动构建第一版 QA。

### 第二层：多源信号融合（正在规划）

- 引入 Gaze（视线） 

- 引入 Atomic Descriptions（带时间戳动作） 

- 将 Gaze、Relation、Atomic 和视频帧统一到同一时间轴上 

- 建立 Temporal Alignment 

### 第三层：Benchmark 构建（目标阶段）

等多源信号融合完成后，就可以自动生成大量高质量 QA，并配套证据帧、Review 系统和人工审核，最终形成一个真正能够评测**时空推理、跨视角理解和人类意图理解**能力的 Benchmark。





# 一、项目目标

目标不是训练一个模型，而是**构建一个新的 Video QA Benchmark**。

相比传统 Video QA（识别物体、描述动作），本项目希望模型具备：

- Ego\-Exo Correspondence（第一视角与第三视角对应） 

- Visibility / Occlusion Reasoning（可见性与遮挡推理） 

- Temporal Grounding（时间定位） 

- Gaze\-Grounded QA（视线推理） 

- Perspective Grounding（视角推理） 

- Human Spatial Intent（人类空间意图理解） 

最终希望回答的问题包括：

- 人什么时候做了某个动作？ 

- 人现在正在看什么？ 

- 人准备操作什么物体？ 

- 第一视角看到的物体，对应第三视角中的哪个物体？ 

- 某个动作发生时，人关注的是哪个目标？ 

---

# 二、项目发展阶段

## Phase 1（已完成）：Relation QA Pipeline

利用 Ego\-Exo4D 的 Relation Annotation 构建第一版 QA Pipeline。

完成内容：

- Relation 数据解析 

- Relation Records 生成 

- QA Candidate 自动生成 

- Evidence Frame 抽取 

- Review HTML 页面生成 

这一阶段已经能够自动生成约 9000\+ QA。

但这些 QA 主要还是：

- Visibility 

- Correspondence 

- 简单 Spatial Relation 

无法体现人的注意力和意图。

---

## Phase 2（当前阶段）：引入 Gaze 与 Temporal 信息

项目方向正式从 Relation QA 转向：

Relation

↓

Gaze

↓

Atomic Description

↓

Temporal Grounding

↓

Human Spatial Intent

这是目前整个项目最重要的工作。

---

# 三、当前需要完成的工作

## Gaze 数据

目的：

获取每一时刻人的注视位置。

需要完成：

- 下载 gaze annotation 

- 解析 gaze CSV 

- 转换统一 JSONL 格式 

- 与视频 Frame 对齐 

输出：

每一条 gaze 包含：

- timestamp 

- frame 

- gaze point 

- source 

- validity 

---

## Atomic Description

目的：

获取带时间戳的动作描述。

例如：

15\.2s：

Cut tomato

18\.3s：

Open drawer

需要完成：

- 解析 Atomic Annotation 

- 建立统一格式 

- 与 Gaze、Relation、Video 对齐 

最终得到：

Atomic

↓

Timestamp

↓

Relation

↓

Gaze

↓

Video Frame

统一时间轴。

---

## Gaze → Object Matching

整个项目最重要的一步。

流程：

Gaze

↓

Object Mask

↓

Object

↓

Attention

如果 gaze 落在 Object Mask：

→ High Confidence

如果落在 BBox：

→ Medium Confidence

否则：

→ Nearest Object（Low Confidence）

最终得到：

人在这一时刻真正关注哪个物体。

---

## Ego → Exo Correspondence

利用 Relation Annotation：

建立：

Ego Object

↓

Same Object

↓

Exo Object

这样就能知道：

第一视角看的物体，

在第三视角对应的是哪个物体。

---

## Temporal Alignment

将四类信息统一：

Video

- 

Relation

- 

Atomic

- 

Gaze

↓

Temporal Alignment

每个时间点都知道：

- 正在做什么 

- 正在看什么 

- 周围有哪些物体 

- 各视角对应关系 

这是整个 Benchmark 的核心。

---

## QA Generation（后续重点）

基于上述信息生成新的 QA。

主要 Question Type：

### Temporal QA

When does the person perform the action?

---

### Action QA

What is the person doing?

---

### Gaze QA

Which object is the person looking at?

---

### Action \+ Attention QA

While cutting tomato,

which object is the person looking at?

---

### Ego\-Exo Correspondence QA

Which object in the exocentric view corresponds to the ego wearer's gaze target?

---

### Visibility QA

Is the gazed object visible in both ego and exo views?

---

# 四、后续工作

## Evidence Pipeline

升级证据抽取。

从：

单帧

升级为：

时间窗口

↓

Clip

↓

Contact Sheet

方便人工审核。

---

## Review HTML

升级 Review 页面。

展示：

- Question 

- Answer 

- Evidence 

- Gaze 

- Confidence 

- Temporal Window 

方便人工筛选 Benchmark。

---

## 2D Spatial QA

利用 BBox 自动生成：

- left / right 

- above / below 

- overlap 

注意：

这里只讨论 **Image Plane**，

不讨论真实三维空间关系。

---

## Perspective Grounding（Future Work）

最终目标：

实现真正的人类视角空间推理：

- person's left 

- person's right 

- front 

- behind 

目前由于缺少：

- Pose 

- Calibration 

- Trajectory 

暂时无法实现。

---

# 五、目前项目进度总结

Prompt

我现在需要搭一个pipeline从第三视角重建人和空间的关系，也就是我作为旁观者（第三视角），重建观察者（第一视角）人和空间的关系，比如我看到这个人前面有一杯水，然后以这个人为主体建立坐标系，说这杯水在他的前方什么的。这个task需要在egoexo4D这个数据集上完成，代码咱们先放一边，我需要用什么卡去训练比较好呢，前期可能用十几个试一下就可以，后面可能会大规模

ssh \-p 41299 root@connect\.westb\.seetacloud\.com 西北B区611

