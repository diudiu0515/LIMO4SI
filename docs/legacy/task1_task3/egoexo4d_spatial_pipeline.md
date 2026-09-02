# Ego-Exo4D 人体中心空间关系：最小 Pipeline

## 当前数据状态

`metadata + relations` 只提供对象名称、2D bbox/mask、ego/exo 相机对应关系。
它可以回答“哪个 ego 物体对应哪个 exo 物体”，但不能独立产生可靠的米制
3D 位置。要回答“杯子在人的前方 0.8 m”，还需要选定 take 的：

- `downscaled_takes/448`：同步的 ego/exo 视频；
- `take_trajectory`：Aria 相机在世界坐标系的位姿；
- `take_point_cloud`：场景几何；
- `ego_pose_pseudo_gt`：人的世界坐标系 3D 关键点。

## 1. 从 validation 标注选择 15 个 take

```bash
cd /root/autodl-tmp/LIMO4SI
source .venv/bin/activate

python scripts/select_relation_takes.py \
  --split val \
  --count 15
```

评分偏向：非人体物体多、ego/exo 两边都有 mask、exo 相机多、轨迹有效。
结果位于 `outputs/selection/val_15_uids.txt` 和对应 JSON 报告。

## 2. 查看预计下载内容

```bash
python scripts/download_selected_takes.py \
  outputs/selection/val_15_uids.txt \
  --dry-run
```

确认容量后，去掉 `--dry-run` 才会实际下载：

```bash
python scripts/download_selected_takes.py \
  outputs/selection/val_15_uids.txt
```

如果先验证轻量版本，可不下载点云：

```bash
python scripts/download_selected_takes.py \
  outputs/selection/val_15_uids.txt \
  --parts downscaled_takes/448 take_trajectory ego_pose_pseudo_gt
```

## 3. 人体局部坐标系

所有关节与物体点必须先变换到同一个世界坐标系。代码使用：

- 原点：左右髋关节中点；
- `+X`：左肩指向右肩，即人的右侧；
- `+Y`：髋中心指向肩中心，并与 X 正交化；
- `+Z`：`X × Y`，即人体正面法向；
- nose：用于消除正面/背面的符号歧义。

世界坐标中的物体点 `p` 转到人体坐标：

```text
d = p - pelvis
x = dot(d, right)
y = dot(d, up)
z = dot(d, forward)
```

因此 `z > 0` 是前方、`x > 0` 是右侧、`y > 0` 是上方。

运行自带几何示例：

```bash
python scripts/compute_spatial_relation.py examples/spatial_input.json
```

## 4. 从数据到最终关系

完整数据流为：

```text
exo 视频中的人/物 mask
        │  相机内外参 + 深度/点云
        ▼
物体的世界坐标 3D centroid
        │
        ├── EgoPose 世界坐标 3D joints
        ▼
人体局部坐标变换
        ▼
前/后/左/右/上/下 + 米制距离
```

Relations mask 用来确定“杯子”的像素集合。像素通过相机内参形成射线，再和
点云/深度相交，取去异常值后的 3D 中位数作为物体中心。EgoPose 给出同一世界
坐标系中的肩、髋和鼻子。最后调用 `build_human_frame()` 与
`describe_relation()` 得到结构化结果。

## 重要限制

- 只有 2D bbox/mask 时不能声称得到真实米制距离。
- 肩部轴只能确定人体平面的法向，必须用 nose、头部朝向或 Aria 朝向消除
  前后符号歧义。
- “人的前方”应采用身体朝向还是头部/视线朝向，需要在 benchmark 定义中固定；
  当前实现采用身体朝向。
- 物体中心最好使用 mask 内点云的稳健中位数，而不是 2D bbox 中心。
