# test_trend_alignment.py — 趋势对齐测试与可视化文档

## 用途

对 LeRobot 数据集执行 **Stage 2: 状态-动作趋势对齐** 分析，检测 action 与 state 之间的时间延迟和对齐质量，并生成可视化图 + JSON 结果。

## 运行方式

```bash
python tests/test_trend_alignment.py
```

输出保存到 `tests/results/test_trend_alignment/run_<timestamp>/`。

---

## 算法背景（来自 `utils/trend_alignment.py`）

### 1. 互相关延迟检测

在 `[-max_lag, +max_lag]` 范围内逐帧平移 action 信号，对每个偏移量计算各维度的 Pearson 相关系数，然后聚合（mean/median），取相关系数最大的 lag 为最优延迟。

**lag 正负的含义：**

| Lag | 含义 | 判定 |
|-----|------|------|
| `lag < 0` | Action 领先 state \|lag\| 帧 | ✅ 正常（action 先发生，state 后到达） |
| `lag = 0` | Action 与 state 同步 | ✅ 正常 |
| `lag > 0` | Action 落后 state | ❌ 可疑（违反因果关系） |

### 2. 方向一致性 (Directional Agreement, DA)

对齐后，对比 action 与 state 的一阶差分（即变化方向）是否一致：

```
agree[t,d] = sign(action_diff[t,d]) == sign(state_diff[t,d])
```

如果两者都无变化（差分接近 0），也视为一致。DA 分数 = 一致的帧数 / 总帧数。

DA < 0.6 判定为可疑。

---

## 输出的图及含义

### 1. `ep{n}_cross_correlation.png` — 互相关曲线

| 行 | 内容 |
|----|------|
| **前 D 行**（如 Joint 1~7） | 每个维度的 Pearson 相关系数随 lag 的变化曲线，红色虚线标出该维度的最优 lag |
| **最后一行** | 所有维度聚合后的平均相关系数曲线，红色虚线标出全局最优 lag |

> 用于判断 action 与 state 之间的**时间偏移量**。峰值越尖锐、越靠近 0，对齐越好。

### 2. `ep{n}_lag_bars.png` — 各维度最佳延迟柱状图

| 元素 | 含义 |
|------|------|
| **绿色柱子** | 该维度的最优 lag = 全局最优 lag |
| **蓝色柱子** | 该维度的最优 lag 与全局不同 |
| **红色虚线** | 全局最优 lag |

> 用于检查**各维度之间的一致性**——如果大部分维度是绿色，说明所有关节具有相同的延迟模式。如果蓝色很多，说明不同维度的时间关系不一致，数据可能有问题。

### 3. `ep{n}_alignment_overlay.png` — 对齐前后对比

| 线条 | 含义 |
|------|------|
| **蓝色细线** | 原始 state（对齐前） |
| **红色细线** | 原始 action（对齐前） |
| **蓝色粗线** | 对齐后的 state |
| **红色粗线** | 对齐后的 action |

> 直接目视检查对齐效果——对齐后红蓝曲线是否基本重合。

### 4. `ep{n}_da_heatmap.png` — 方向一致性热力图

| 颜色 | 含义 |
|------|------|
| **绿色** | 该帧该维度的 action 与 state 方向一致 |
| **黄色** | 方向不完全一致 |
| **红色** | 方向不一致 |

> 热力图中红色竖条集中的区域，就是 action 和 state"打架"的异常片段。

### 5. `summary.png` — 所有 episode 汇总

| 行 | 内容 |
|----|------|
| **第1行** | 每段 episode 的最优 lag（柱状图） |
| **第2行** | 每段 episode 的最大相关系数（橙色虚线 = 0.5） |
| **第3行** | 每段 episode 的 DA 分数（橙色虚线 = 0.6 阈值，绿色=全部达标，红色=有可疑） |

> 快速总览：哪段 episode 的延迟异常、相关性低、或方向一致性差。

---

## 输出 JSON 字段

| 字段 | 含义 |
|------|------|
| `optimal_lag` | 全局最优延迟（帧数） |
| `optimal_correlation` | 最优延迟下的平均相关系数 |
| `per_dim_optimal_lags` | 每个维度各自的最优延迟 |
| `per_dim_optimal_corrs` | 每个维度在各自最优延迟下的相关系数 |
| `da_overall` | 整体方向一致性分数 |
| `da_per_dim` | 每个维度的方向一致性分数 |
| `num_disagreement_frames` | 方向不一致的帧数 |
| `disagreement_frames` | 方向不一致的帧索引列表 |
| `is_suspicious` | 是否可疑（lag>0 或 DA<0.6） |
| `is_suspicious_lag` | 是否因延迟可疑 |
| `is_suspicious_da` | 是否因 DA 过低可疑 |

---

## 输出文件汇总

| 文件 | 数量 | 含义 |
|------|------|------|
| `ep{n}_cross_correlation.png` | N | 互相关曲线 |
| `ep{n}_lag_bars.png` | N | 各维度最优延迟柱状图 |
| `ep{n}_alignment_overlay.png` | N | 对齐前后轨迹对比 |
| `ep{n}_da_heatmap.png` | N | 方向一致性热力图 |
| `summary.png` | 1 | 所有 episode 汇总 |
| `results.json` | 1 | 完整的结果 JSON |

N = episode 数量（默认 5 段：0~4）
