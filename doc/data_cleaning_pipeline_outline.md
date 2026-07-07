# 数据清洗管线 — 算子实现大纲

基于论文 Section 2.4,清洗管线分为 **5 阶段状态-动作信号滤波** + **3 阶段跨模态质量检查**。

---

## 第一阶段:突变检测 (Sudden Change Detection)

**功能**: 检测并过滤信号中的物理碰撞、传感器异常引起的瞬态突变。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `SignalSmoother` | 原始信号序列 | 平滑趋势 + 残差信号 | 级联中值滤波 → Savitzky-Golay 平滑 |
| `ResidualCalculator` | 原始 + 平滑序列 | 绝对残差(1阶差分) | `|raw - smoothed|` |
| `DerivativeCalculator` | 平滑序列 | 加速度(2阶)、急动度(3阶) | 有限差分 |
| `JointThresholdJudge` | 残差、加速度、急动度 | 异常帧标记 | 残差超阈值 **且** (加速度超阈值 **或** 急动度超阈值) |
| `EpisodeDiscarder` | 异常帧标记 | 最终动作 | 按数据集类型:帧级移除 / 整段丢弃(如 InternData-A1 物理碰撞导致整段丢弃) |

**输出**: 清洗后的轨迹(移除了异常帧或整段的版本)。

---

## 第二阶段:状态-动作趋势对齐 (State-Action Trend Alignment)

**功能**: 校正异步时钟、丢包引起的状态-动作时间错位。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `TrajectorySmoother` | 状态序列 + 动作序列 | 平滑后的状态/动作轨迹 | Savitzky-Golay 平滑(同 S1) |
| `CrossCorrelationLagger` | 平滑后的状态/动作(同关节维度) | 最优时间延迟 `τ` | 互相关估计 |
| `DirectionalAgreementCalculator` | 对齐后的1阶差分序列 | DA 指标(标量) | 计算同符号方向比例 |
| `EpisodeFilterByDA` | DA 指标 + 阈值 | 保留/丢弃标记 | DA < 0.6~0.7 时丢弃整段 |
| `DeltaActionIntegrator` | delta 动作序列 | 绝对动作序列 | 累积积分(仅对 delta 动作数据集生效) |

**输出**: 剔除时间不同步的 episodes;RoboMIND UR 数据 81% 在此阶段被淘汰。

---

## 第三阶段:极值过滤 (Extreme Value Filtering)

**功能**: 去除超出合理范围的帧,防止归一化时被异常值扭曲。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `PercentileCalculator` | 整数据集的每维信号 | 每维 Q1、Q99 | 按本体类型分组统计 |
| `ExtremeValueDetector` | 单帧信号 + 百分位阈值 | 极值帧标记 | `frame < Q1 - α*(Q99-Q1) || frame > Q99 + α*(Q99-Q1)` |
| `GripperMaskApplier` | 极值帧标记 | 过滤后的标记 | 夹爪维度被豁免(因其双峰分布) |
| `FrameFilter` | 极值帧标记 + 原始序列 | 删除极值帧后的序列 | 仅删除异常帧,保留其余帧 |

**输出**: 移除极值帧后的轨迹(对后续的 quantile normalization 友好)。

---

## 第四阶段:关节-末端正运动学一致性 (FK Consistency)

**功能**: 通过正向运动学检查,自动修正关节角符号约定、TCP偏移、旋转表示不一致等问题。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `FKComputer` | URDF + 关节角度 | 期望末端位姿 | Pinocchio 库计算 FK |
| `PoseComparator` | 记录末端位姿 vs FK 结果 | 位姿差异(平移+旋转) | 计算每个维度的偏差 |
| `TCPOffsetResolver` | 恒定平移偏移 | 修正后的 TCP 定义 | 自动调整 tool-center-point |
| `JointOffsetResolver` | 恒定关节偏移 | 修正后的关节零点 | 检查并修正关节偏移 |
| `RotationRepConverter` | 旋转表示不一致 | 统一旋转表示 | 检测并转换为标准表示(6D 连续表示) |
| `BaseFrameTransformer` | 局部/肩部参考系位姿 | 世界坐标系位姿 | 肩部相对双臂姿态→世界坐标系变换 |

**输出**: 尽可能修正而非激过滤;仅无法修正的才丢弃。

---

## 第五阶段:基座/末端方向对齐 (Base Frame & Orientation Alignment)

**功能**: 统一不同数据集的世界坐标系约定。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `WorldFrameAlignment` | 各数据集的位姿数据 | 对齐后的位姿 | 按数据集施加旋转校正 |
| `ForwardDirectionChecker` | 对齐后的位姿 | 一致性标记 | 确保 x 轴正向 = 机器人前方 |
| `PerDatasetRotationCalibrator` | 数据集级别的旋转偏差 | 旋转校正矩阵 | 为每个数据集计算并存储固定的旋转变换 |

**输出**: 所有数据在世界坐标系下几何一致的位姿表示。

---

## 跨模态质量检查 C1:指令一致性 (Instruction Consistency)

**功能**: 验证每个演示与语言标注之间的语义一致性。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `EpisodeSegmenter` | 长片段 + 语言标注 | 子任务级片段 | 基于 VLM 的时域分割(Lei et al., 2026) |
| `StructuredReasoningChecker` | 图像片段 + 指令 | 结构化推理文本 + 一致性标签 | VLM 三步推理:关注物体→动作语义→时序→最终判断 |
| `MultiExpertAdjudicator` | 多个 VLM 的判断结果 | 最终一致性标签 | 交叉模型投票 |
| `InconsistencyFilter` | 一致性标签 | 保留/丢弃 episodes | 不一致样本排除 |

**输出**: 排除语言-视觉语义不一致的 episodes。

---

## 跨模态质量检查 C2:视频-状态一致性 (Video-State Consistency)

**功能**: 验证机器人在视频中的视觉外观与记录的状态数据一致。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `RobotRenderer` | URDF + 关节状态 + 相机参数 | 渲染的机器人 mask + 深度图 | 3D 渲染(离线) |
| `RobotSegmenter` | 原始视频帧 | 真实机器人 mask | 微调后的 SAM3 分割 |
| `OverlapMeasurer` | 渲染 mask + 分割 mask | IoU 得分 | `|mask1 ∩ mask2| / |mask1 ∪ mask2|` |
| `LowOverlapFilter` | IoU 得分 + 阈值 | 保留/丢弃 episodes | IoU 低于阈值的丢弃或优化相机参数 |

**输出**: 剔除视频-状态不匹配的样本。

---

## 跨模态质量检查 C3:视频质量过滤 (Video Quality Filtering)

**功能**: 剔除视觉上无效的视频帧。

| 算子 | 输入 | 输出 | 方法 |
|------|------|------|------|
| `BlackFrameDetector` | 视频帧 | 黑帧标记 | 像素值统计(均值/方差接近0) |
| `CorruptedFrameDetector` | 视频帧 | 损坏帧标记 | 解码错误检测 / 像素异常值 |
| `BlurDetector` | 视频帧 | 模糊帧标记 | Laplacian 方差 / 梯度幅值 |
| `StaticSegmentDetector` | 连续帧 + 状态/动作信号 | 静态段标记 | 联合视觉+状态变化量判定 |
| `KeyFramePreserver` | 帧过滤标记 | 关键帧保留标记 | 夹爪闭合事件等关键帧强制保留 |
| `FrameQualityFilter` | 各检测器标记 | 过滤后视频 | 综合各检测结果,移除无效帧 |

**输出**: 剔除视觉无效帧,同时保留语义关键帧。

---

## 附:辅助工具算子

| 算子 | 功能 |
|------|------|
| `QuantileNormalizer` | 基于 Q1-Q99 的 `[−1,1]` 归一化(训练时使用) |
| `PerDimBinaryMaskGenerator` | 按本体类型生成规范的 80 维二进制掩码 |
| `DatasetMetadataReader` | 读取数据集元信息(本体系类型、数据来源、基座移动性等) |
| `VisualizationLogger` | 可视化每阶段的清洗日志,输出清洗前后对比图 |

---

## 流水线编排建议

```
Raw Dataset
  │
  ├─→ S1. SuddenChangeDetector ──→ [异常帧/段移除]
  │
  ├─→ S2. TrendAligner ──→ [时间不对齐段丢弃]
  │
  ├─→ S3. ExtremeValueFilter ──→ [极值帧移除]
  │
  ├─→ S4. FKConsistency ──→ [自动修正 / 丢弃不可修正段]
  │
  ├─→ S5. OrientationAlignment ──→ [统一世界坐标系]
  │
  ├─→ C1. InstructionConsistency ──→ [语义不一致丢弃]
  │
  ├─→ C2. VideoStateConsistency ──→ [视频-状态不匹配丢弃]
  │
  └─→ C3. VideoQualityFilter ──→ [视觉无效帧移除]
       │
       └─→ Clean Dataset (统一规范表示)
```

每个阶段应支持**可配置的阈值**(per-dataset),并输出**清洗日志**(丢弃了多少帧/段、原因)。第四阶段应优先做**自动修正**而非直接丢弃。
