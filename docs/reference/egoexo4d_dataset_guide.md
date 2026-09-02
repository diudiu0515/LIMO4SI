# Ego-Exo4D 数据集全面指南

> 面向数据下载、结构理解、多视角研究和 LIMO4SI 空间关系 pipeline。
> 本文以 Ego-Exo4D V2 官方文档为准，整理日期：2026-07-30。

## 1. 数据集是什么

Ego-Exo4D 是一个大规模、多模态、多视角的人类技能活动数据集。它的核心特点
是同步采集：

- 第一视角（egocentric）：参与者佩戴 Project Aria 眼镜；
- 第三视角（exocentric）：环境中布置多台 GoPro 等外部相机；
- 三维空间信号：相机轨迹、半稠密点云、眼动、相机标定；
- 人体信号：3D body pose、3D hand pose；
- 语言与任务标注：atomic descriptions、keysteps、专家评论、熟练度等。

V2 包含约 5,035 个 takes、1,286.30 小时视频（其中约 221.26 ego-hours）。
整个项目覆盖 800 多名参与者、13 个城市和 131 个自然场景。活动包括烹饪、
自行车维修、医疗、篮球、足球、舞蹈、攀岩和音乐等。

官方入口：

- 数据集主页：https://docs.ego-exo4d-data.org/
- 入门与许可：https://docs.ego-exo4d-data.org/getting-started/
- 下载器：https://docs.ego-exo4d-data.org/download/
- 数据说明：https://docs.ego-exo4d-data.org/data/
- 标注说明：https://docs.ego-exo4d-data.org/annotations/
- Benchmark：https://docs.ego-exo4d-data.org/benchmarks/

## 2. Ego、Exo、Capture 和 Take

### 2.1 Ego 与 Exo

Ego 是摄像机佩戴者的第一视角。主要 RGB 流常以类似
`aria01_214-1` 的名字出现。Aria 还提供 SLAM 相机、IMU、音频、眼动等信号。

Exo 是环境中的第三视角相机，通常命名为 `cam01`、`cam02` 等。一个 take
一般包含一个 ego 相机和一到多台 exo 相机。

### 2.2 Capture

Capture 是一次连续录制 session。它包含：

- 多台相机的原始时间线；
- capture 级 timesync；
- capture 级 trajectory、eye gaze 和 point cloud；
- 多个活动片段。

### 2.3 Take

Take 是从 capture 中按活动切出的时间片段，也是多数训练和标注使用的基本单位。
同一 capture 可以包含多个 takes。take 级数据已按照开始/结束时间裁剪。

重要标识：

- `capture_uid`：录制 session 的唯一 ID；
- `take_uid`：take 的唯一 ID，跨文件关联时应优先使用；
- `take_name`：可读目录名，例如 `sfu_cooking_007_3`；
- `participant_uid`：参与者 ID；
- `best_exo`：对该 take 最有信息量的第三视角相机。

## 3. 同步机制

Ego-Exo4D 提供 frame-aligned videos。同一个 take 中，不同相机视频的第 `i`
帧对应同一时间点，因此跨视角取帧时应尽量使用统一 frame index。

Capture 级的 `timesync.csv` 提供各相机：

- presentation timestamp；
- frame number；
- capture timestamp。

`takes.json` 中的 `timesync_start_idx` 和 `timesync_end_idx` 指定 take 在
capture 时间同步表中的范围。

注意：

- 视频常按约 30 FPS 工作；
- 2D gaze CSV 常为 10 FPS；
- trajectory 与 IMU 时间戳频率更高，官方建议按纳秒时间戳寻找最近记录；
- annotation 自己可能包含 `annotation_fps`，不可默认所有标注都是 30 FPS。

## 4. 主要传感器和模态

### 4.1 同步 RGB 视频

`takes` 包含 frame-aligned videos。全分辨率体积非常大。原型阶段推荐使用
`downscaled_takes/448`，即短边缩放到 448 像素的同步视频。

典型目录：

```text
takes/<take_name>/frame_aligned_videos/
├── aria01_214-1.mp4
├── cam01.mp4
├── cam02.mp4
└── downscaled/448/
    ├── aria01_214-1.mp4
    ├── cam01.mp4
    └── cam02.mp4
```

### 4.2 VRS

Aria 原始数据采用 VRS 格式。VRS 可以包含：

- RGB camera；
- 左右 SLAM camera；
- IMU；
- 7 通道音频；
- 设备与相机标定。

`take_vrs_noimagestream` 不含图像流，体积比完整 `take_vrs` 小。若已经使用
frame-aligned MP4，通常不需要再下载含 RGB 的完整 VRS。

### 4.3 Trajectory

Trajectory 来自 Project Aria Machine Perception Services（MPS），描述相机
佩戴者随时间变化的 6DoF 位姿。它可用于：

- 获取相机佩戴者在世界坐标系中的位置；
- 将 ego 观测变换到全局坐标；
- 重建运动轨迹；
- 做 gaze、pose、point cloud 的空间融合。

### 4.4 Point Cloud

点云主要包括半稠密三维点及其 observations。点云与 trajectory 处于同一
三维坐标系，可用于：

- 场景几何重建；
- 深度或表面近似；
- 将 2D object mask 关联到 3D 点；
- 估计物体位置和尺寸。

点云不是现成的语义物体点云。要得到“杯子的 3D 点”，通常需要使用相机标定把
点云投影到图像，再用杯子的 segmentation mask 筛选。

### 4.5 Eye Gaze

眼动由 Aria MPS 估计。常见文件包括：

```text
eye_gaze/
├── general_eye_gaze.csv
├── personalized_eye_gaze.csv
├── general_eye_gaze_2d.csv
├── personalized_eye_gaze_2d.csv
└── summary.json
```

- 3D gaze 表示从双眼之间出发的视线射线；
- 2D gaze 是投影到 ego RGB 图像的坐标；
- personalized gaze 使用参与者校准，并非所有 take 都有；
- general gaze 覆盖更广；
- `takes.json` 的 `has_trimmed_eye_gaze` 可用于筛选。

2D gaze 特别适合做 `gaze point -> object mask -> gaze target`。

### 4.6 音频、转写和专家评论

Aria 提供多通道音频。数据集另提供 take audio、transcription 和 expert
commentary，可用于动作理解、技能评估和多模态 Video-QA。

## 5. Metadata

核心 metadata 文件通常包括：

```text
metadata.json
takes.json
captures.json
participants.json
physical_setting.json
visual_objects.json
annotations/splits.json
```

### takes.json

每个元素描述一个 take，常见字段包括：

- `take_uid`、`take_name`；
- `capture_uid`、`participant_uid`；
- `task_name`、`parent_task_name`；
- `duration_sec`；
- `best_exo`；
- `frame_aligned_videos`；
- `has_trimmed_trajectory`；
- `has_trimmed_eye_gaze`；
- 各种相对路径。

### splits.json

包含：

- `take_uid_to_split`；
- `split_to_take_uids`；
- `take_uid_to_benchmark`。

## 6. 主要标注

### 6.1 Relations

Relations 标注关注 ego-exo 之间的物体实例对应关系。主要字段：

- `scenario`；
- `take_name`；
- `object_names`：ego 图像上的对象名称和 bbox；
- `object_masks`：按对象、相机和帧组织的 segmentation masks；
- `annotation_fps`；
- `annotated_frames`。

相机名以 `aria` 开头时通常是 ego，否则通常是 exo。

Relations 可用于：

- ego-to-exo object correspondence；
- exo-to-ego correspondence；
- visibility / occlusion；
- 物体 track；
- 2D image-plane spatial QA。

限制：Relations mask 本身是二维标注，不能单独证明物体在人体三维坐标系中的
前、后、左、右或米制距离。

### 6.2 EgoPose

EgoPose 提供：

- 世界坐标系 3D body joints；
- 世界坐标系 3D hand joints；
- ego/exo 对应的 2D keypoints；
- 相机内参与外参 metadata。

Body benchmark 常用 17 个身体关节。EgoPose 是建立人体坐标系的关键数据：

```text
origin = pelvis center
+X = left shoulder -> right shoulder
+Y = pelvis center -> shoulder center
+Z = X cross Y
```

还需要 nose/head 或 Aria 朝向来消除前后符号歧义。

### 6.3 Atomic Action Descriptions

Atomic descriptions 是带单一时间点的短动作描述，常见字段：

- `timestamp`；
- `text`；
- `narration_subject`：`C` 表示 camera wearer，`O` 表示其他人；
- `ego_visible`；
- `best_exo`；
- `unsure`。

它适合把“空间状态”与“此刻正在做什么”连接起来。

### 6.4 Keystep

Keystep 是带开始/结束时间的程序步骤标注，包含：

- `start_time`、`end_time`；
- `step_name`、`step_description`；
- `step_id`、`step_unique_id`；
- `is_essential`；
- 层级 taxonomy 和全局 vocabulary。

### 6.5 Proficiency

熟练度标注支持：

- demonstrator proficiency：参与者整体水平；
- demonstration proficiency：具体时间段表现好坏与改进建议。

### 6.6 Expert Commentary

专家评论包含专家对表演过程的语音或文本评价，可带时间点、错误描述、原因和空间
drawing 信息。

## 7. 官方 Benchmark

主要 benchmark 包括：

### Keystep

- Fine-grained Keystep Recognition；
- Task Graph；
- Energy-efficient online activity detection。

### Relations

- Correspondence：一个视角中的物体 mask 与另一视角中同一实例对应；
- Translation：从已观察视角的 mask 推断未观察视角中的 mask。

### EgoPose

- EgoBodyPose；
- EgoHandPose。

### Proficiency Estimation

- 参与者熟练度分类；
- 时间定位的表现评价。

## 8. 下载 Part 和官方体积

以下为 V2 官方下载页给出的近似体积：

| Part | 体积 |
|---|---:|
| metadata | 0.046 GB |
| annotations | 10.533 GB |
| takes | 10,553.486 GB |
| captures | 43.618 GB |
| take_trajectory | 509.503 GB |
| take_eye_gaze | 3.265 GB |
| take_point_cloud | 6,164.615 GB |
| take_vrs | 12,301.458 GB |
| take_vrs_noimagestream | 995.592 GB |
| capture_trajectory | 851.691 GB |
| capture_eye_gaze | 5.619 GB |
| capture_point_cloud | 4,750.039 GB |
| downscaled_takes/448 | 438.556 GB |
| features/omnivore_video | 49.986 GB |
| features/maws_clip_2b | 533.826 GB |
| ego_pose_pseudo_gt | 138.629 GB |
| expert_commentary | 42.292 GB |
| take_transcription | 0.094 GB |
| take_audio | 1,056.907 GB |
| default | 12,112.778 GB |
| all | 38,449.753 GB |

这些是全数据集体积。使用 `--uids` 筛选少量 take 后会小很多。

## 9. 许可和凭证

下载前必须：

1. 申请并接受 Ego-Exo4D License；
2. 等待审核并获取 AWS credentials；
3. 配置 AWS profile；
4. 安装 `ego4d>=1.7.1`。

官方说明凭证通常在 14 天后过期，不能把凭证写进代码、Git 或公开文档。

```bash
aws configure --profile egoexo
```

## 10. 下载命令

安装：

```bash
python -m pip install "ego4d>=1.7.1" awscli
egoexo --help
```

只下载 metadata 和 Relations：

```bash
egoexo \
  -o /path/to/egoexo4d \
  --release v2 \
  --parts metadata annotations \
  --benchmarks relations \
  --s3_profile egoexo \
  -y
```

下载指定 take 的空间数据：

```bash
egoexo \
  -o /path/to/egoexo4d \
  --release v2 \
  --parts downscaled_takes/448 take_trajectory take_point_cloud ego_pose_pseudo_gt \
  --uids <take_uid_1> <take_uid_2> \
  --s3_profile egoexo \
  -y
```

下载指定 take 的 gaze：

```bash
egoexo \
  -o /path/to/egoexo4d \
  --release v2 \
  --parts take_eye_gaze \
  --uids <take_uid_1> <take_uid_2> \
  --s3_profile egoexo \
  -y
```

不要不带 `--parts` 直接运行默认下载，除非已经准备约 12 TB 空间。

## 11. 针对 LIMO4SI 的推荐数据流

目标：从第三视角观察 camera wearer 和物体，并在人体局部坐标系中描述物体。

```text
Relations object mask
        |
        v
选择 ego/exo 中同一物体实例
        |
        v
相机内外参 + trajectory + point cloud
        |
        v
恢复 object world-space 3D points
        |
        +----------------------+
                               |
EgoPose world-space joints ----+
                               v
                    建立 human-centric frame
                               |
                               v
              object world -> human coordinates
                               |
                               v
     八方向 + 上下 + 绝对距离 + 相对距离 + 尺寸
```

建议空间标签：

- 水平八方向：front、front-right、right、back-right、back、back-left、left、
  front-left；
- 垂直：above、same-height、below；
- 距离：absolute distance、nearest/farthest、relative order；
- 几何：object size、contact、overlap、inside、on-support；
- 时间：appearance/disappearance、direction change、distance change。

## 12. 常见陷阱

### 2D 左右不等于人的左右

第三视角图像左侧只是相机像平面的左侧。必须把物体转换到人体坐标系后，才能说
它在人的左边或右边。

### Body、Head 和 Gaze 坐标系不同

- body-centric：以肩髋定义身体朝向；
- head-centric：以头部或 Aria 朝向定义；
- gaze-centric：以实时视线方向定义。

Benchmark 必须明确使用哪一种。

### 前后存在符号歧义

只用左右肩和髋部可以得到身体平面法向，但法向可能朝前也可能朝后。需要 nose、
head orientation 或 Aria forward axis 来确定符号。

### Mask 不是 3D 物体

从 mask 得到 3D 点时应：

1. 投影点云到对应相机；
2. 筛选落在 mask 内且深度合理的点；
3. 去除背景和离群点；
4. 用中位数或聚类中心估计物体中心；
5. 用多相机一致性估计置信度。

### 时间戳不可混用

必须记录每个数据源的：

- timestamp unit；
- FPS；
- frame index 基准；
- take-relative 或 capture-relative 时间；
- 最近邻匹配误差。

## 13. 本服务器当前数据

项目根目录：

```text
/root/autodl-tmp/LIMO4SI
```

数据根目录：

```text
/root/autodl-tmp/LIMO4SI/data/egoexo4d
```

已下载 Relations V2 train/val/test 和 metadata。当前选择三个 validation takes：

```text
iiith_cooking_32_1
sfu_cooking_007_3
sfu_cooking_010_1
```

这三个 take 的 448p 视频、trajectory、point cloud 和 EgoPose 合计约
8.730 GiB。

## 14. 推荐阅读顺序

1. Overview：https://docs.ego-exo4d-data.org/overview/
2. Getting Started：https://docs.ego-exo4d-data.org/getting-started/
3. CLI Downloader：https://docs.ego-exo4d-data.org/download/
4. Takes：https://docs.ego-exo4d-data.org/data/takes/
5. MPS：https://docs.ego-exo4d-data.org/data/mps/
6. Relations：https://docs.ego-exo4d-data.org/annotations/relations/
7. EgoPose：https://docs.ego-exo4d-data.org/annotations/ego_pose/
8. Gaze Tutorial：https://docs.ego-exo4d-data.org/tutorials/gaze/
9. Atomic Descriptions：https://docs.ego-exo4d-data.org/annotations/atomic_descriptions/

