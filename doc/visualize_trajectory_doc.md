# visualize_trajectory.py — 轨迹可视化工具文档

## 用途

对 LeRobot 格式的机器人演示数据集，加载指定 episode，生成 state（蓝色）和 action（红色）的多维度分析图，用于**数据质量检查和分布分析**。

## 运行方式

```bash
python tests/visualize_trajectory.py \
    --dataset-path example_data/pick_banana_100_newTable_1_offset_state \
    --episode 0 1 2
```

输出保存到 `tests/results/visualize_trajectory/<dataset_name>/`。

---

## 两种归一化方式

| 方式 | 公式 | 用途 |
|------|------|------|
| **q01 归一化** | clip 到 [q01, q99] 后线性映射到 [-1, 1] | 鲁棒归一化，不受极端值影响 |
| **z-score** | (x - mean) / std | 评估数据是否接近标准正态分布 |

---

## 运动学计算

使用**中心差分法**（`_central_diff`），窗口 = 0.5s，平滑计算：

| 量 | 含义 |
|----|------|
| **Velocity** | 位置的一阶导数（速度） |
| **Acceleration** | 速度的一阶导数（加速度） |
| **Jerk** | 加速度的一阶导数（急动度） |

---

## 输出的图及含义

### 1. 每段 episode 的图 (per-episode)

#### 1.1 `episode_{n}_trajectory.png`

| 列 | 内容 | 含义 |
|----|------|------|
| **第1列** Raw trajectory | 各维度 state/action 原始轨迹 | 观察数据范围、趋势、异常跳变 |
| **第2列** Normalized | q01 归一化后的轨迹（y轴固定 [-1.5, 1.5]） | 检查归一化效果是否合理 |
| **第3列** Per-dim histogram | 每个维度的原始/归一化分布叠加 | 检查各维度的分布形态 |
| **底部第1行** | 所有维度合并的原始值直方图 | 全局数据范围概览 |
| **底部第2行** | 所有维度合并的归一化值直方图 | 全局归一化分布 |
| **底部第3行** | raw + normalized 叠加对比 | 直观对比归一化前后的分布变化 |

#### 1.2 `episode_{n}_per_dim_histogram.png`

| 列 | 内容 |
|----|------|
| **左列** | 每个维度的原始值直方图 |
| **右列** | 每个维度的 q01 归一化后直方图 |

> 用于检查哪些维度分布异常（如夹爪的离散分布 vs 关节的连续分布）。

#### 1.3 `episode_{n}_kinematics.png` / `episode_{n}_kinematics_normalized.png`

| 列 | 内容 |
|----|------|
| **第1列** | 速度 (Velocity) 轨迹 + 直方图 |
| **第2列** | 加速度 (Acceleration) 轨迹 + 直方图 |
| **第3列** | 急动度 (Jerk) 轨迹 + 直方图 |

> 用于检查运动平滑度：速度/加速度/急动度是否存在异常尖峰（可能是传感器噪声或跟踪丢失）。`_normalized` 版本是基于 q01 归一化后数据计算的运动学。

#### 1.4 `episode_{n}_velocity.png`

| 子图 | 内容 |
|------|------|
| **第1列** | 原始速度轨迹 |
| **第2列** | 归一化后的速度轨迹（先 normalize 再 diff） |
| **第3列** | 原始速度直方图 |
| **第4列** | 归一化速度直方图 |

> 对比两种顺序的效果：
> - raw: `diff(raw_state)` → 直接看原始速度
> - norm: `diff(normalize(state))` → 看归一化后的速度

#### 1.5 `episode_{n}_velocity_selfnorm.png`

| 子图 | 内容 |
|------|------|
| **左列** | 速度自归一化轨迹（先 diff 再 normalize） |
| **右列** | 速度自归一化直方图 |

> 与前一张图的区别：先用原始数据算速度（diff），再用速度自身的 q01/q99 做归一化。用于对比 `diff → normalize` vs `normalize → diff` 两种顺序的效果差异。

#### 1.6 `episode_{n}_gaussian_histogram.png`

| 子图 | 内容 |
|------|------|
| **前 dim 行** | 每个维度的 z-score 分布 + N(0,1) 标准正态曲线 |
| **最后一行** | 所有维度合并的 z-score 分布 + N(0,1) 曲线 |

> 用于判断数据是否符合高斯分布假设——如果数据分布与黑色虚线 (N(0,1)) 偏差大，说明 z-score 归一化后数据并不高斯，可能需要其他归一化策略。

---

### 2. 多 episode 叠加图 (multi-episode)

用于对比不同 episode 之间的一致性。

#### 2.1 `multi_episode_raw_trajectory.png`

多个 episode 的原始 state/action 轨迹叠加在同一张图上。

#### 2.2 `multi_episode_normalized_trajectory.png`

多个 episode 的 q01 归一化轨迹叠加（y轴固定 [-1.5, 1.5]）。

#### 2.3 `multi_episode_raw_histogram.png`

多个 episode 的原始值直方图叠加（state / action 分开）。

#### 2.4 `multi_episode_normalized_histogram.png`

多个 episode 归一化值直方图叠加。

#### 2.5 `multi_episode_{vel|acc|jerk}_trajectory.png`

多个 episode 的运动学（速度/加速度/急动度）轨迹叠加。

#### 2.6 `multi_episode_{vel|acc|jerk}_normalized_trajectory.png`

基于归一化数据计算的运动学轨迹叠加。

---

## 输出文件汇总

| 文件 | 每 episode / 多 episode | 图数 |
|------|------------------------|------|
| `episode_{n}_trajectory.png` | 每 episode 1 张 | N |
| `episode_{n}_per_dim_histogram.png` | 每 episode 1 张 | N |
| `episode_{n}_kinematics.png` | 每 episode 1 张 | N |
| `episode_{n}_kinematics_normalized.png` | 每 episode 1 张 | N |
| `episode_{n}_velocity.png` | 每 episode 1 张 | N |
| `episode_{n}_velocity_selfnorm.png` | 每 episode 1 张 | N |
| `episode_{n}_gaussian_histogram.png` | 每 episode 1 张 | N |
| `multi_episode_raw_trajectory.png` | 1 张 | 1 |
| `multi_episode_normalized_trajectory.png` | 1 张 | 1 |
| `multi_episode_raw_histogram.png` | 1 张 | 1 |
| `multi_episode_normalized_histogram.png` | 1 张 | 1 |
| `multi_episode_vel_trajectory.png` | 1 张 | 1 |
| `multi_episode_acc_trajectory.png` | 1 张 | 1 |
| `multi_episode_jerk_trajectory.png` | 1 张 | 1 |
| `multi_episode_vel_normalized_trajectory.png` | 1 张 | 1 |
| `multi_episode_acc_normalized_trajectory.png` | 1 张 | 1 |
| `multi_episode_jerk_normalized_trajectory.png` | 1 张 | 1 |

**总计**：`7 × N + 10` 张图（N = episode 数量）
