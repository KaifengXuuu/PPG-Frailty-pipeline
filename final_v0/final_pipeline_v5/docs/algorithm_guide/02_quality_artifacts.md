# 3. 质量与运动处理

本章对应 V5 的 `signal/sqi.py`、`quality/`、`artifacts/`，并追踪 `experiment.py` 中的实际调用和 `models/motion.py` 中的运动模型。所有源码位置相对于 V5 的 `src/ppg_frailty/`；行号是本说明对应的源代码定位，函数名用于代码移动后的检索。这里解释现有运算，不把“应当如何实现”写成“已经实现”。

每节先给不依赖专业术语的直觉，再给可对照源码的公式、步骤和边界。“默认”指构造器的默认值；“finalcase”指 `configs/presets/finalcase.yaml` 的具体选择，两者不混用。原始光学数值没有在本章被换算成血液体积或医学量，不能把这些数值的高低直接解释为健康程度。

SQI 的内部 dataclass 字段名与公开 YAML 键并非全部相同。以下映射来自 `signal/sqi.py:210–317`，是调参时应使用的形式：

| 内部字段 | YAML 键与默认值 |
|---|---|
| `q_rate_threshold` / `q_morph_threshold` | `quality.rate_threshold: 0.50` / `quality.morph_threshold: 0.65` |
| `cardiac_low_hz` / `cardiac_high_hz` | `quality.cardiac_band_hz: [0.5, 3.0]` |
| `spectral_analysis_low_hz` / `spectral_analysis_high_hz` | `quality.spectral_analysis_band_hz: [0.2, 8.0]` |
| `peak_density_min_bpm` / `peak_density_max_bpm` | `quality.peak_density_bpm_range: [30.0, 200.0]` |
| `ppi_min_s` / `ppi_max_s` | `quality.ppi_range_s: [0.3, 2.0]` |
| `calibrator_lower_quantile` / `calibrator_upper_quantile` | `quality.calibrator_quantiles: [0.10, 0.90]` |
| 各种 reference、scale、shape center、单项阈值及 template 半宽 | 放在 `quality.component_normalization` 子字典，具体值见各节 |
| `welch_max_nperseg`、template 三个数量字段、`ppi_stability_min_intervals` | 直接放在 `quality` 下 |

校准策略还有一个容易误读的默认差异：直接 `SqiConfig()` 的默认是固定公式；从 quality 映射读取时若省略/留空 `calibrator`，则选择经验分位数。因此正式 `quality.mode=route` 应显式写出需要的 `quality.calibrator`，不能依据构造器默认值猜 YAML 的行为。配置解析器在非 route 的去噪恢复路径才主动补固定公式；依据是 `config.py:197–275`。

## 3.1 先看真正执行的顺序和几个互相独立的开关

直觉：先看一段曲线能不能数清起伏，再看每次起伏的形状是否可信；另外可以请一位“动作观察员”看同步的身体动作。如果某段太乱，可以选一种整理方法试一次。整理后能数清起伏，不等于原来的细节形状也恢复了。

实际主入口是 `experiment.py:696–743` 的 `_apply_quality_motion_routing`，不是文件中仍保留的整条 recording 路由函数 `_route_records`。前者在需要路由时调用 `experiment.py:888–1133` 的 `_route_records_window_level`。

| 设置 | 含义 | finalcase |
|---|---|---|
| `quality.mode` | `off` 不评估；`diagnostics_only` 留诊断；`route` 让质量结果参与路由 | `off` |
| `quality.calibrator` | 固定分数公式或只从训练参与者拟合的分数映射 | 未启用；构造器默认固定公式 |
| `artifact.motion_detector_enabled` | 使用已训练的动作检测器逐窗打分 | `false` |
| `artifact.denoiser_enabled` | 允许不合格窗口请求一次整理波形 | `false` |
| `artifact.reducer` | 实际整理算法；参数放 `artifact.parameters` | `identity` |
| `artifact.degraded_policy` | 整理后的速率特征用途 | `drop` |
| `quality.window_selection.policy` | 另一路 raw 训练窗口筛选，不等于上面的 SQI | `none` |

`route_module_switches_from_config`（`quality/routing.py:158–175`）分别读取这三个路由开关；未提供 `denoiser_enabled` 时才以“reducer 是否非 identity”推导其值。finalcase 明确提供 `false`，不依赖这个推导。

finalcase 的三个路由开关全关，主入口直接保留已经通过物理读入/预处理的记录，不计算 SQI、不载入运动网络、不执行去噪算法，也不因为角色是 R/S/W 自动剔除。`roles` 的输入选择仍然独立生效。这里“保留”不取消前一章的缺失、非有限值、长度、分窗等处理。

开启路由时的顺序是：

1. 如选择经验映射，用外层训练参与者的直接波形窗口拟合 SQI 映射。
2. 若需要运动检测，载入对应已有模型、输入变换和阈值；此处不再训练。
3. 在 400 Hz 网格上建立完整 8 s、步长 2 s 的判断窗；在整条直接波形上找峰，再把峰证据切到各窗。
4. 每窗合并 SQI 和动作证据，得到 Excellent、Acceptable 或 Unfit。
5. 只要某窗请求去噪，选定 reducer 对整条记录执行一次；不是每个重叠窗各重跑一次。
6. 从整理后的完整波形重新找峰，逐个请求恢复的窗重算 `Q_rate`；成功只能进入 Acceptable，而不能重新宣称形态完全可信。
7. 把重叠判断窗变成不重叠的时间归属区间，交给 raw/feature/matrix/fusion 各自处理。

## 3.2 SQI 共同输入、原始观察量与分数

### 3.2.1 输入准备：选取真正有效的连续曲线

源码：`signal/sqi.py:742–881`，`_raw_quality_observations`。

输入是一个 `CanonicalSignalViews`，或显式的 `[N, 1]` / `[N, 2]` 浮点数组；列为一个或两个波长，采样率必须是 400 Hz。另有峰的位置、峰间隔、有效性标记、同步 IMU、原始质量证据。输出是供后续小模块共用的观察量，而不是分类预测。

直觉：一段有洞的纸带，不能把洞两边直接接在一起假装时间连续。这里找出一段最长、确实可用的连续纸带做曲线观察，同时另外记下整条纸带有多少比例可用。

代码逐步做的是：

1. `canonical.analysis_signal` 选择直接过滤波形或已经对齐的整理后波形；`canonical.route` 决定哪些性质仍可解释。
2. 从 `rate_valid_mask` 的假/真跳变找连续真区间，选 `stop-start` 最大的区间。没有任何有效区间时抛错，不生成零分伪装正常观察。
3. 若存在无效位置，波形和同步 IMU 截取到该最长区间；覆盖率仍取原始有效标记的均值。诊断模式在这种截取后重新取得峰，不把旧位置直接套到新数组。
4. 对数组输入检查有限性、列数；没有显式 `route` 则拒绝，因为不能猜测曲线是否已经被改变。
5. 未提供峰结果时调用配置指定的 `detect_pulses`，不是暗中另选一个找峰算法。峰算法本身见下一章。
6. 有效间隔同时要求 `pulse.valid_interval_mask` 为真、数值有限，并处于 `ppi_min_s=0.3` 到 `ppi_max_s=2.0` 秒。

此处会重复使用下面的 Welch、自相关、模板等小算法；不会用 frailty 标签。固定公式不需要跨参与者拟合。

### 3.2.2 心跳常见快慢范围内的起伏占比

源码：`signal/sqi.py:507–532`，`_welch_metrics`；分数变换见 `evaluate_quality:1063–1067`。

直觉：把曲线想象成许多不同快慢的波浪叠起来。若大部分起伏都集中在每秒约半次到三次，比较像可以数的脉搏；若大部分集中在别处，这个提示就弱一些。它不是“落在这个范围就一定是心跳”。

数学步骤：

1. 每个波长分别调用 `signal.welch(x, fs=400, nperseg=min(2048,N))`，得到频率 `f_k` 和功率 `P_k`。
2. 取总体分析频带 `U={k:0.2≤f_k≤8.0}`，心跳频带 `H={k:0.5≤f_k≤3.0}`。
3. 原始观察 `C=Σ_H P_k / Σ_U P_k`。这里是频率格点功率求和，不是 `trapezoid` 积分。
4. 分别计算 RED、IR 后取均值；某通道分母不大于零则跳过。没有可用通道则返回 NaN，后面标成 unavailable。
5. 质量分数 `s_C=clip(C/0.65,0,1)`，其中 `0.65` 对应 `quality.component_normalization.cardiac_concentration_reference`。

输入光学幅值平方除以幅值平方，输出无单位。公开可调键为 `quality.cardiac_band_hz`、`quality.spectral_analysis_band_hz`、`quality.welch_max_nperseg` 和上述参考值；默认值如步骤所示。finalcase 不执行。

### 3.2.3 波浪的能量是否散得到处都是

源码：同上 `_welch_metrics`；`evaluate_quality` 的 `normalized_spectral_entropy`。

直觉：同样一盆沙子，集中在少数几个小格子，还是平均撒在每个格子？集中说明主要只有几种起伏速度，分散说明很多快慢混在一起。这里衡量“分散程度”，不是直接数心跳。

对上一节分析带内 `M` 个频点：

`p_k=P_k/Σ_U P_k`

`H=−Σ p_k log(p_k+10⁻¹⁵) / max(log M,10⁻¹²)`

双通道 `H` 取均值；最终分数 `s_H=clip(1−H,0,1)`。加上的小数仅避免 `log(0)` 或除零。它没有另一个可训练参数，和心跳集中度共享同一次 Welch。finalcase 不执行。

### 3.2.4 曲线平移一下，能否和自己对齐

源码：`signal/sqi.py:534–551`，`_autocorrelation_periodicity`。

直觉：复印一份曲线，把复印件向右缓慢移动。如果移动到一个心跳左右的距离后，大部分山峰和山谷又能对上，说明起伏比较有规律。

每个波长先 `x←x−mean(x)`，再计算：

`r[l]=Σ_n x[n]x[n+l] / Σ_n x[n]²`

代码用 `signal.correlate(..., mode="full", method="fft")` 后截取非负延迟。搜索 `l=round(0.3fs)…round(2.0fs)`，取最大值，再对可用通道取平均。这里不按每个延迟的重叠样本数再次调整；长延迟通常有更少重叠。

分数是 `clip(max_l r[l]/0.70,0,1)`。`quality.component_normalization.autocorrelation_reference=0.70`；延迟上下限使用 `quality.ppi_range_s`。能量不超过 `10⁻¹²` 或数组长度不超过最大延迟时，该通道不可用。finalcase 不执行。

### 3.2.5 每分钟找到多少个峰

源码：`signal/sqi.py:846–857, 1033–1055`。

直觉：在已经圈出的山峰上做计数。如果一分钟的山峰数明显少于 30 或多于 200，给它扣分；轻微越界不会立即跳成零。

设峰数为 `K`，观察长度为 `N/fs` 秒：

`d=60K/(N/fs)`，单位为次/分钟。

默认可接受区间 `[30,200]`；区间内分数为 1，区间外：

`s_d=exp(−|d−clip(d,30,200)|/30)`。

公开可调键为 `quality.peak_density_bpm_range`。这里 `K` 是 `pulse.peaks` 的长度，不是把所有失败峰另行剔除后再数。finalcase 不执行。

### 3.2.6 相邻峰的距离是否在生理范围内

源码：`signal/sqi.py:824–840, 1073–1077`。

直觉：一串山峰有些挤得几乎重叠，有些又隔得太远。数一数相邻距离中有多少个既被找峰流程认可、又不离谱。

对原始间隔 `Δ_i` 和已有有效标记 `v_i`：

`u_i=v_i ∧ finite(Δ_i) ∧ (0.3≤Δ_i≤2.0)`。

原始量和质量分数都为 `mean(u_i)`，无单位。若一个间隔也没有，返回不可用，而不是把“没有证据”写成“全部通过”。finalcase 不执行。

### 3.2.7 相邻距离是不是忽长忽短

源码：`signal/sqi.py:833–840, 1030–1032, 1078`。

直觉：把刚才认可的距离排成一列。长度很接近，看起来更整齐；长短跳动很大则分数更低。人的心跳本来会自然变化，所以这只是一个带权提示，不是真实心跳必须完全等距的要求。

至少有 `ppi_stability_min_intervals=3` 个有效间隔时：

`CV=std(Δ_valid,ddof=0)/mean(Δ_valid)`。

`s=exp(−max(CV,0)/0.20)`，`0.20` 为 `quality.component_normalization.ppi_cv_scale`。

`evaluate_quality` 中名为 `ppi_stability` 的 component，其 `raw_value` 也是已经指数映射的稳定分数；独立原始诊断保留的则是 `rate.ppi_cv`。阅读表格时不要将二者当作同一个量。finalcase 不执行。

### 3.2.8 红光与红外光是否一起升降

源码：`signal/sqi.py:803–809`，以及 `red_ir_agreement` component。

直觉：两条曲线同时记录同一处。即使一条画得大、一条画得小，只要起伏的时间很一致，说明它们互相支持。当前实现也接受“一个向上、另一个向下”的一致起伏。

`s=|corr(RED,IR)|`，其中相关是去均值后向量夹角的余弦形式。绝对值意味着相关为 −1 和 +1 都得 1 分。只有一个波长或任一通道没有变化时，该项不可用。无额外默认参数。finalcase 不执行。

### 3.2.9 身体动作越大，光学可信度提示越弱

源码：`signal/sqi.py:840–846, 1080–1085`。

直觉：手腕晃得越厉害，光学曲线越可能被牵动，因此给大动作较低分。但这项不声称已经区分出“动作”和“光学错误”。

同步 `dynamic_magnitude` 的有限值为 `a_i`，单位 m/s²：

`R=sqrt(mean(a_i²))`

`s=exp(−R/4.0)`，分母是 `quality.component_normalization.motion_rms_scale=4.0` m/s²。

没有这一派生量或长度不对齐则不可用。该固定公式不是运动 CNN，也不会训练一个动作分类器。若采用不减重力的 IMU 分支，`dynamic_magnitude` 名称仍在，但其物理内容包括未去除的重力；解释分数时必须结合实际 IMU 分支。finalcase 关闭 SQI，故不使用这项。

### 3.2.10 曲线是否完全平直

源码：`signal/sqi.py:855–857, 1086–1090`。

直觉：若两条曲线中有一条几乎是水平直线，就不能把它当成有起伏的测量。

`u=min_c std(x_c,ddof=0)`；若 `u>10⁻¹⁰` 得 1，否则得 0。阈值为 `quality.component_normalization.nonflat_std_threshold`，单位沿用光学原始幅值。它是最低通道变化量，不是双通道平均。finalcase 不执行。

### 3.2.11 真正有数据的位置占多少

源码：`signal/sqi.py:765–801, 1091`。

直觉：一张画布只有一小角画了曲线，不能因这一小角很漂亮就当整张画布都可靠。

覆盖率通常是有效采样位置比例：二维来源 mask 用每行所有通道都有效的比例；整理路线优先采用 `rate_valid_mask` 的比例。component 分数就是这个比例。终点判定还单独要求覆盖率至少 `quality.minimum_coverage=0.80`，即覆盖率不仅通过加权分数起作用，还直接约束终点是否通过。finalcase 不执行 SQI 的覆盖率判定。

### 3.2.12 连续水平段、极值堆积、饱和与长缺口：四个独立小模块

源码：`signal/sqi.py:647–711`，`_qc_components`。这些读取前置 PPG 检查的证据，不对已经过滤好的波形重新猜测原始 ADC 状态。

**连续水平段。** 直觉：不是只问整体有没有起伏，还要看是否有一长段完全不动。取所有波长最长常数段长度 `L`；原始量 `L/fs` 秒，分数 `1−min(L/(1.0fs),1)`。`quality.flatline_duration_s=1.0`。

**极值堆积。** 直觉：很多点像撞到同一堵墙一样停在最大值或最小值附近，可能是量程压住了。代码取各通道 `min_occupancy/max_occupancy` 中最大的比例 `c`，分数 `1−min(c/0.02,1)`；`quality.component_normalization.clipping_fraction_reference=0.02`。这是数据极值占用的启发式提示，不能等同于已证实达到硬件量程。

**明确的 ADC 饱和。** 直觉：只有知道仪器真正的上下限，才能说“撞到仪器的墙”。仅当证据包含有限 `adc_saturation_fraction=a`，计算 `1−min(a/0.02,1)`；`quality.component_normalization.saturation_fraction_reference=0.02`。不知道量程时标 unavailable，不把极值堆积冒充饱和。

**长缺口。** 直觉：缺一小角和缺一大块不能同样处理。取通道最大 `longest_nonfinite_gap_samples=G`，`G≤100` 得 1，否则 0；`quality.long_gap_max_samples=100`。这是质量评分，不负责填补缺失；补洞已在预处理阶段进行。

没有 `channels` 证据时四项全部 unavailable。finalcase 不执行这些 SQI component；前置的物理质量检查仍存在。

### 3.2.13 每次脉搏的形状像不像共同的样板

源码：`signal/sqi.py:553–586`，`_template_correlation`。

直觉：围绕每个山峰剪一小张纸条，把纸条都拉到同样宽度，再把它们叠在一起。若山脊轮廓很接近，典型样板就能代表大多数纸条；若每次都长得不同，相似度低。

计算顺序：

1. 至少需要 `template_min_peaks=5` 个峰；选择 `pulse.wavelength` 对应的 RED/IR 列。
2. 对已接受的峰 `p` 截取 `[p−w,p+w]`，`w=round(0.30fs)`；越界峰跳过。
3. 用线性插值把每段变成 `template_resample_points=101` 点。
4. 每段减均值、除自身标准差；标准差不超过 `10⁻¹²` 的段跳过。
5. 至少 `template_min_beats=3` 段才能继续；逐位置取中位数得到样板 `t_j=median_b z_bj`。
6. 计算每段与样板的相关，取相关的中位数 `r`；分数 `max(r,0)`，随后裁剪至 `[0,1]`。

`template_half_width_s` 位于 `quality.component_normalization`；`template_min_peaks/template_min_beats/template_resample_points` 直接位于 `quality`。非 identity 去噪输出不评估这项，不宣称峰形被保留。finalcase 不执行。

### 3.2.14 波形朝一边偏得多不多

源码：`signal/sqi.py:861–862, 1130–1134`。

直觉：把所有点按离平均高度的距离看一遍，是否一侧总有更长的尾巴？这项衡量不对称程度，偏得过大就扣分；并不是将某个方向的尖峰认定为特定疾病。

代码用 `stats.skew(matrix, axis=0, bias=False)`，先按通道算经有限样本校正的三阶形状量，再取通道均值 `g`。分数 `exp(−|g|/3.0)`，参数 `quality.component_normalization.morph_skewness_scale=3.0`。基础几何量是 `mean((x−mean x)³)/std(x)³`，但执行值采用 SciPy 的 `bias=False` 校正，不应以未校正手算值替换。finalcase 不执行。

对 N>2 且方差非零的一列，记 `m_k=mean((x−mean x)^k)`，实际有限样本校正为 `G1=sqrt(N(N−1))/(N−2)·m3/m2^(3/2)`。随后才对波长取平均和指数映射；不能先把两个波长混成一列再计算。

### 3.2.15 很尖很高的点是否过多

源码：`signal/sqi.py:862–863, 1135–1139`。

直觉：多数点挤在中间、偶尔几个点特别远，和点均匀铺开，会给人不同的“尖”和“厚尾”观感。代码希望这类形状量别离参考中心太远。

`k=mean_channels(stats.kurtosis(..., fisher=False, bias=False))`，这是 Pearson 定义，参考中心 3，而不是减去 3 后的 excess 定义。

分数 `exp(−|k−3.0|/5.0)`；参数为 `quality.component_normalization` 下的 `morph_kurtosis_center=3.0`、`morph_kurtosis_scale=5.0`。基础量来自四次方距离 `mean((x−mean x)⁴)/std(x)⁴`，实际仍使用 SciPy 有限样本校正。finalcase 不执行。

对 N>3 且方差非零的一列，以上一节 m2/m4 记号，设 `g2=m4/m2²−3`，校正后的 Pearson 值为 `K=(N−1)[(N+1)g2+6]/[(N−2)(N−3)]+3`。末尾加 3 对应 `fisher=False`；若省掉这一项，后面以 3 为中心的评分含义就变了。

## 3.3 从小分数得到两个用途不同的终点

### 3.3.1 小分数的共同裁剪与状态

源码：`signal/sqi.py:485–505`，`_component/_components`。

直觉：把不同尺子量出的东西都放到 0 到 1 的尺子上，但缺测不能随便补成零或满分。

`raw` 或 `normalized` 不存在/非有限时，结果是 `UNAVAILABLE`，值为 `None`。否则 `score=clip(normalized,0,1)`，`score≥quality.component_normalization.component_pass_threshold` 才标 PASS，默认阈值 0.50。这个单项 PASS 标记不意味着整段最终通过。

### 3.3.2 Q_rate：还能不能可靠地数脉搏

源码：`signal/sqi.py:14–19, 588–621, 1062–1109`。

直觉：几个观察员一起投票，但观察不到某一方面的人不参与；不是把缺席算成反对。少数提示再好，也不能弥补大部分时间根本没有数据。

各项权重按当前可用项重新归一化：

`Q=Σ_(i可用) w_i s_i / Σ_(i可用) w_i`。

实现先将权重除以最大权重再求和，以避免大权重尺度带来数值问题，数学上仍是上式。不存在可用正权重项时状态 UNAVAILABLE。通过要求 `Q≥0.50` 且覆盖率 `≥0.80`。

| Q_rate component | 默认权重 |
|---|---:|
| cardiac_concentration | 0.20 |
| autocorrelation_periodicity | 0.15 |
| normalized_spectral_entropy | 0.10 |
| peak_density_bpm | 0.08 |
| ppi_physiological_fraction | 0.15 |
| ppi_stability | 0.12 |
| red_ir_agreement | 0.08 |
| motion_energy_rms | 0.05 |
| nonflat_scale | 0.02 |
| source_coverage | 0.04 |
| flatline | 0.02 |
| clipping | 0.015 |
| saturation | 0.015 |
| long_gap | 0.01 |

权重键为 `quality.rate_component_weights`；公开终点阈值键为 `quality.rate_threshold`。权重不要求原始总和等于 1，因为函数会归一化。finalcase 不执行。

### 3.3.3 Q_shape 与 Q_morph：能否解释细节轮廓

源码：`signal/sqi.py:20–24, 1111–1189`。

直觉：能看出每座山在哪里，不代表能可靠比较山顶尖不尖、山脚宽不宽。形状可靠比“能数清山峰”要求更多。

`Q_shape` 是形状 component 的加权分数，按上节同样方式计算；公开默认阈值为 `quality.morph_threshold=0.65`。真正 `Q_morph` 的通过条件是：

`Q_morph_pass = Q_rate_pass AND Q_shape_pass`。

所以形状分数很高但数峰端失败时，`Q_morph` 仍失败，分数则保留 `Q_shape.score`。非 identity reducer 路线直接给 `Q_shape/Q_morph=NOT_APPLICABLE`，分数和阈值均为空，不是得到零分。

| Q_shape component | 默认权重 |
|---|---:|
| template_correlation | 0.30 |
| skewness | 0.08 |
| pearson_kurtosis | 0.08 |
| red_ir_agreement | 0.18 |
| cardiac_concentration | 0.16 |
| nonflat_scale | 0.04 |
| source_coverage | 0.12 |
| flatline | 0.03 |
| clipping | 0.02 |
| saturation | 0.02 |
| long_gap | 0.02 |

配置键为 `quality.morph_component_weights`。finalcase 不执行。

### 3.3.4 固定公式路线与训练数据分位数路线

源码：`signal/sqi.py:413–483, 623–645`；调用位置 `experiment.py:567–620`。

**固定公式 `fixed_formula_thresholds_v1`。** 直觉：每次都用同一把刻好刻度的尺子；不因本批受试者变化而重画刻度。使用前面明确的除数、指数公式和终点阈值，没有跨记录拟合。它是 `SqiConfig.calibrator` 默认值。

**经验分位数 `outer_train_empirical_quantiles_v1`。** 直觉：只看训练组，找较低和较高的常见位置作为尺子的两端。一个人提供很多纸条，也不能因此比另一个人拥有更大的总发言权。

逐行步骤：

1. `_fit_quality_calibrator` 仅选 `train_ids` 中参与者的直接波形；逐 8 s/2 s 判断窗用固定公式计算基础分数，不能先用测试窗画尺子。
2. 对 component 的每个有限观察值 `x_j`，权重设为 `1/n_p`，其中 `n_p` 是该参与者对此 component 的有限行数。每个参与者对此项的总权重为 1。
3. 按数值稳定排序；以 `t_j=(累计权重−0.5w_j)/总权重` 作为每个观察值在整条尺子中的位置。
4. 用线性插值取 `q=0.10` 和 `q=0.90` 的值，得到 `L,H`。公开参数为 `quality.calibrator_quantiles: [0.10,0.90]`。
5. 使用时变换 `s'=clip((s−L)/(H−L),0,1)`。若 `H≤L` 则只裁剪原基础分数；某项没有拟合边界则保持原值。
6. 保存 `fitted_on_participant_ids`；测试参与者不能出现在这份列表。既不使用 frailty 标签，也不在测试/推理时重新拟合。

一个需要按现代码理解的细节：rate component 先经过 `rate.*` 映射；morph 中共享的 RED/IR 一致性等直接复用这些 rate component，然后再应用 `morph.*` 映射。也就是说，共享项可能依次经过两次映射，不能把当前实现解释成所有 morph 项都直接从未映射基础分数进入一次映射。

### 3.3.5 diagnostics_only 与主流程记录的区别

源码：`signal/sqi.py:883–975`，`quality/routing.py:234–288`，`experiment.py:888–1014`。

`evaluate_quality_diagnostics` 只产出原始观察及 available 状态，不输出分类决策，也不进行训练组分位数拟合。`run_quality_mode(..., diagnostics_only)` 的默认 evaluator 就是这个函数。

但当前主流程窗口级编排的 `diagnostics_only` 分支还会调用一次固定公式 `evaluate_quality`，把 Q 值放进时间线，同时保存原始诊断。时间线的分级函数在此模式下忽略这些 Q 值；若运动开关开启，运动结果仍可独立影响路线。不能把这一路说成“运动和去噪也随之关闭”。

另外，全部路由开关关闭走直接保留快路径；`diagnostics_only` 会进入完整判断窗时间线。小于判断窗长度或尾端没有判断证据的区间，其结构性处理要以时间线结果为准。因此“诊断分数不作筛选”不等于可以不检查判断窗覆盖边界就断言两条完整执行路径逐样本完全相同。

## 3.4 质量与动作共同决定路线

### 3.4.1 Excellent / Acceptable / Unfit 的真值表

源码：`quality/routing.py:54–156`，`route_quality_tier`。

直觉：先问“有没有足够条件相信这段”，再问“只能数山峰，还是连山的轮廓也可信”。动作证据缺失不能假装动作很小。

| SQI 参与判定 | Q_rate | Q_morph | 动作检测 | 初步等级 |
|---|---|---|---|---|
| 关 | 不使用 | 不使用 | 关或 low | Excellent |
| 关 | 不使用 | 不使用 | high | Unfit |
| 开 | 非 PASS | 任意 | 任意 | Unfit |
| 开 | PASS | PASS | 关或 low | Excellent |
| 开 | PASS | 非 PASS | 关或 low | Acceptable |
| 开 | PASS | 任意 | high | Unfit |
| 任意 | 任意 | 任意 | 已启用但无有效证据 | Unfit |

这个函数本身不训练、不去噪，也不把 Unfit 自动解释为永久删除；是否试一次恢复由下一小节决定。等级不是 frailty 类别，也不是医学诊断。

### 3.4.2 仅尝试一次恢复及 rate-only 的含义

源码：`quality/routing_timeline.py:86–181`，`resolve_routing_evidence`；实际调用 `experiment.py:1026–1087`。

结构失败首先直接 Excluded。其他情况下，`denoiser_enabled` 为真、初步为 Unfit，并且 SQI 路由或运动检测确实启用，才请求 reducer。

直觉：试着整理一张皱纸一次；不能整理失败就不断换方法直到找到看起来顺眼的结果，也不能因为纸面变平就声称所有细小折痕位置都被保真恢复。

恢复成功必须同时满足：算法返回 success、输出时间轴对齐、整理后的 `Q_rate` 为 PASS。满足后最多成为 Acceptable，使用 `x_ar_400`；不能重新使用整理后的振幅/形态学作保真解释。

在 `quality.mode=off` 时，只有显式允许的 `feature_vector + artifact.degraded_policy=denoise_then_extract_rate_features` 能通过“动作高→整理→重评 Q_rate”恢复速率特征。raw/matrix/fusion 不以这个特殊开关偷偷打开 SQI 恢复；`diagnostics_only` 不用整理结果升级等级。

失败、不支持、没有通过重评则 Excluded。主流程保留 direct 和 processed 两套视图及区间归属，不把它们拼接成一条宣称统一保真的混合波形。

### 3.4.3 重叠判断窗如何变成唯一时间归属

源码：`quality/routing_timeline.py:51–83, 234–356`。

直觉：每位观察员看 8 秒，相邻观察员每隔 2 秒站一个位置，所以看见的范围重叠。最后用相邻观察员站位的中点分界，让每个时刻只有一位负责，不重复计算拥有权。

`W=round(8fs)=3200`、`H=round(2fs)=800`。完整窗起点为 `0,H,2H,…≤N−W`，不补零，也不自动补一个右对齐尾窗。窗中心 `c_i=(start_i+stop_i)/2`；相邻归属边界为 `round((c_i+c_(i+1))/2)`，代码用 `np.rint`。

第一归属区间从第一窗起点开始，最后一个到末窗终点。未覆盖的前后边缘单独标 Excluded；没有完整窗则整条记录没有判断证据。归属区间带有原窗 Q 值、运动概率、阈值、去噪状态和来源，不重新计算平均概率。

### 3.4.4 各表征消费路线的方式

源码：`experiment.py:1099–1118`；`quality/routing_timeline.py:359–379`。

raw 与 fusion 在 recording 初筛层只接受存在 Excellent direct 区间的记录。feature-vector 和 feature-matrix 可存在 Acceptable 区间，但具体哪些特征有效仍由下一章的特征与 mask 规则决定；Acceptable 不是“所有特征都允许”。

对 matrix 的一个目标行区间，`matrix_row_route` 收集所有与之有正长度重叠的归属区间：任何一个 Excluded 就使该行不可用；全部 Excellent 且 direct 则 Excellent；否则只要有 Acceptable 则 Acceptable。这里不做多数投票，也不按覆盖时长平均等级。

## 3.5 另一路可选 raw 窗口筛选

### 3.5.1 `none`：不计算，不筛选

源码：`quality/window_selection.py:23–67, 190–221`。

直觉：每张已经切好的纸条都保留。没有假造的“质量满分”，只是没有执行这一选择模块。

默认 `policy=none`、`keep_fraction=1.0`、`application_scope=outer_train_only`。返回原 `RawWindows` 和计数，不计算历史分数，也不拟合训练统计量。finalcase 使用此模式。

### 3.5.2 历史窗口分数：先缩到共同尺度

源码：`quality/window_selection.py:70–166`，`legacy_window_sqi_scores`。

输入 `[K,C,T]`，时间有效 mask `[K,T]`；数组先转 float32。这个评分器会另外建立自己的全部通道标准化视图，不修改配置指定的模型输入张量。

直觉：先把每张纸条平移到中间、缩到差不多高，再比较它的节奏和身体动作。不能让某个传感器单位大就凭空占优势。

1. 有效长度取 mask 最后一个真位置加 1；不是把所有真位置紧密拼接。长度不足 16 点给 0 分。
2. 每通道中心 `m=median(x)`，候选尺度 `r=(Q75−Q25)/1.349`；`r>10⁻⁶` 用 r，否则用总体标准差。
3. `z=clip((x−m)/(scale+10⁻⁶),−8,8)`。这里的常数与 raw 模块的可配置归一化参数不是同一个对象。
4. 在 RED/IR 中选择标准差较大者，平手选 IR。
5. 若有对应通道，`a=norm(z[2:5])`、`g=norm(z[5:8])`；通道不足时对应量取零。

finalcase 不执行此评分。

### 3.5.3 历史窗口分数中的四个小评分

源码：`quality/window_selection.py:119–152`。

**心跳频段占比。** 对选中的光学列 Welch，`nperseg=min(512,T)`。`S=trapz(P_0.5…3.0,f)/[trapz(P_all,f)+10⁻¹²]`。这里用梯形积分，与 3.2.2 的频点求和不是同一实现。

**距离整齐度。** `find_peaks` 在 `(ppg−median)/[std+10⁻⁸]` 上运行；最小距离 `round(0.28fs)`、最小突出程度 0.3。至少 3 个峰时 `D=1/[1+std(diff(peaks))/(mean(diff(peaks))+10⁻⁸)]`，否则 0。直觉是相邻山峰间距越接近，越整齐。

**峰数量提示。** 至少 3 个峰时 `P=min(1,K_peaks/max(2,3T/fs))`，否则 0。它按观察长度奖励足够多的峰，但不是 3.2.5 的 30–200 次/分钟区间函数。

**动作扣分。** `M=RMS(a)+0.25RMS(g)`，`A=1/[1+max(0,M−1)]`。这里 a/g 来自本窗标准化值，不是物理单位的动作 RMS。

最终 `score=0.40S+0.35D+0.15P+0.10A`；非有限值置零。每条文件的窗口分数再按第 5 与第 95 百分位缩放：`clip((score−q5)/(q95−q5+10⁻⁸),0,1)`；全部分数相同时不缩放。缩放和裁剪可能产生并列分数，因此应保持代码的排序方式，不能声称与任意稳定排序完全等价。

这些常数属于具名历史算法 `legacy_cardiac_motion_window_sqi_v1`；公开窗口选择参数只配置策略、保留比例和作用范围，不逐项暴露这四个历史系数。

### 3.5.4 每个文件内取最好的一部分

源码：`quality/window_selection.py:167–188, 244–291`。

直觉：每个人每份记录单独选纸条，不拿某个人的好纸条去挤掉另一个人的全部纸条。

对本文件 K 个分数，保留 `max(1,ceil(K·keep_fraction))` 个，K 为零则保持空。代码 `np.argsort(scores)[::-1]` 找排名；布尔 mask 再同时筛选值、有效性、起点和分数，因而输出仍保持原时间顺序，不是质量排名顺序。

`keep_fraction` 范围 `(0,1]`，默认 1.0。仅文件内统计、不读标签、不跨文件/跨折拟合。

### 3.5.5 三种作用范围不能互换

源码：`experiment.py:1596–1644`；`quality/window_selection.py:293–342`。

| `application_scope` | 训练窗口 | OOF 窗口预测行 | OOF 聚合 |
|---|---|---|---|
| `outer_train_only` | 取 top fraction | 全部保留 | 不因本策略删窗口；可由独立质量加权设置读取分数 |
| `all_partitions` | 取 top fraction | 同样取 top fraction，未选窗不进入预测 | 用保留下来的窗 |
| `legacy_train_and_aggregation` | 取 top fraction | 全部预测、全部保存 | 仅 top fraction 的 `window_aggregation_mask` 为真 |

第三种用 `mark_raw_windows_for_aggregation`，它只标记、不删除预测行。连续质量权重仍由独立的 `aggregation.quality_weighting` 与 `quality_weight_source=legacy_window_sqi` 控制。文件内 top fraction 不等于旧脚本直接把同一参与者所有记录的窗口混在一起排名。

## 3.6 动作检测：它学的是动作协议，不是光学伪影真值

### 3.6.1 训练标签和输入路线

源码：`quality/motion.py:253–255` 的 `motion_activity_label`；`quality/motion_adapters.py:41–101`；`quality/motion_reference.py:357–464, 659–773`。

直觉：给观察员看“坐着/按静态要求做动作”和“行走/活动”的示例，让它区分活动状态。标签来自任务安排，不是人工逐点标出光学曲线哪里坏了，所以它不能被叫作光学伪影真值检测器。

内部标签 `B/R→0`、`S/W→1`；PTT `sit→0`、`walk/run→1`。frailty 三分类标签不用于这个二分类目标。

正式运动模型输入为 `[batch,8,3200]`：RED、IR、三轴去重力加速度、三轴角速度，400 Hz，8 s 窗，2 s 步长。运动输入构造与训练折内 IMU 缩放见表征章的 `representations/motion.py`。其 RED/IR 来源是原始光学列，不是 frailty raw 模型使用的已过滤光学列；两条输入路线不得混称。

内部每个人用自己的 B 静态记录校准；PTT 用自己的 sit 记录校准。正式运动物化使用校准后的 roll/pitch EKF，不是 finalcase 的 `sensor_filter_only_no_gravity_removal`。finalcase 不启用运动检测，因此无需在这次分类运行中执行这些额外网络运算。

### 3.6.2 Light CNN 第一层：用小形状沿时间滑动

源码：`models/motion.py:110–127`，`LightCnnMotionDetector.__init__`。

直觉：拿许多个小尺子沿曲线滑动；每把尺子专门对某种局部起伏敏感。它可以同时看几条传感器曲线，而不是只看单个点的高度。

第一层 `Conv1d(C,12,kernel_size=9,padding=4)`：

`y[o,t]=b[o]+Σ_(c=1…C)Σ_(j=0…8) W[o,c,j]x[c,t+j−4]`。

两端按 Conv1d 的零填充处理，时间长度保持 3200。然后 `GroupNorm(1,12)`：对每个样本的全部 12 通道和时间位置计算均值/总体方差，按 `z=γ(y−μ)/sqrt(σ²+ε)+β` 调整；这里 `ε` 未显式传入，使用所安装 PyTorch 的该层默认值。不是 BatchNorm，测试样本之间不互相组成统计量，也没有训练期滑动均值。

`GELU` 把较小的负响应软压下去；数学形式 `zΦ(z)`，代码采用 `nn.GELU()` 默认实现。随后 `AvgPool1d(2)` 对邻接两点求均值，长度变为 1600。

### 3.6.3 第二、三层与输出：合成更长的动作图样

源码：`models/motion.py:120–140`。

直觉：第一层识别小弯折，下一层把小弯折组合成更长的动作片段，最后看整窗总体上更像哪一种活动。

第二层 `Conv1d(12,24,7,padding=3)`→`GroupNorm(1,24)`→`GELU`→`AvgPool1d(2)`，得到 `[batch,24,800]`。

第三层 `Conv1d(24,24,5,padding=2)`→`GroupNorm(1,24)`→`GELU`，仍为 800 个时间点。`AdaptiveAvgPool1d(1)` 对每条特征曲线全窗平均，得到 24 个数；`Flatten` 后 `Linear(24,1)` 输出一个实数 `l`。

预测器 `quality/motion_adapters.py:372–388` 使用 `p_active=1/(1+exp(−l))`，输出 `[batch]` 无单位概率。代码没有在 sigmoid 后另拟合 Platt/isotonic 概率校准器；元数据中的 `calibrated_p_active_probability` 字样不能作为存在额外校准层的证据。

### 3.6.4 三个命名架构与一个开发构造器

源码：`models/motion.py:148–212`；参数计数式在 `48–89`。

**正式 8 通道。** `build_formal_motion_cnn()`，固定双光学+六 IMU 轴、基础宽度 12，5965 个可训练参数。它绑定正式通道顺序和 3200 点窗的说明。

**11 通道派生量增强。** `build_motion_derived_augmentation_cnn()` 在正式 8 通道后增加动作大小、转动大小、动作变化速度三个量，6289 参数。直觉是把已有三轴合成更容易看的总体曲线，再交给同一个网络；不是增加三个新传感器。该具名 motion-only ablation 的构造器可用，但默认正式训练器仍只接收 8 通道。不能把“构造器存在”写成“当前 formal runner 会训练 11 通道”。

**历史 10 通道备份。** `build_historical_light_cnn_backup()` 使用一个历史光学列、六物理轴、三个派生量，6181 参数，基础宽度也是 12。档案对应 256 Hz、8 s/2 s 和阈值 0.05；构造器只建网络、不加载历史权重，也不把历史外部 SIM 得分升级成当前 PTT 或 V5 验证结果。

**开发构造器。** `build_parameterized_light_cnn(channel_names,base_channels=12)` 可指定有序通道名和宽度，但当前实现仍只接受 9/7/5 这组三层卷积长度。以 C 为输入通道数、b 为基础宽度，参数量为：

`(9Cb+3b)+(7b·2b+6b)+(5·2b·2b+6b)+(2b+1)`。

这个式子包括卷积偏置、GroupNorm 的缩放/平移和输出头。以上均非 finalcase 主分类网络；finalcase 主分类网络是另外的 Inception small，不能混为一谈。

### 3.6.5 训练抽样：类别少、参与者窗口多时怎样平衡

源码：`quality/motion_adapters.py:170–183, 253–266`。

直觉：某类纸条特别多，或者某个人特别能贡献纸条，不能每轮都让这些纸条完全占据课堂。当前方法降低它们被抽到的相对机会，但不是保证每个人每一轮恰好一样多。

每个窗口 i 的权重：

`w_i=1/[n_(dataset_i,class_i)·sqrt(n_participant_i)]`。

再除以全部窗口权重的均值，令平均权重为 1。`WeightedRandomSampler` 按这些权重有放回地抽 `len(examples)` 个窗口；每轮仍有同样多次抽样，但会重复某些窗、漏掉某些窗。seed 默认 42，batch size 默认 16。

不在损失函数中再叠加类别权重；`class_weighting=none_balancing_is_sampler_only`。这个 sampler 只用本次 fit 的训练窗口计数。

### 3.6.6 网络学习：十轮固定训练，没有 OOF 早停

源码：`quality/motion_adapters.py:117–145, 232–369`。

直觉：模型每看一批示例，就微调小尺子的形状，让自己的判断更接近已知活动标签；固定学十遍，不能偷看考试分数来挑哪一遍的模型最好。

训练输入先用本折训练参与者拟合的 IMU 中心/尺度变换；RED/IR 的逐窗处理在表征构造时完成。BCE-with-logits 对 logit l 和标签 y 的等价逐样本公式是：

`L=max(l,0)−l·y+log(1+exp(−|l|))`。

代码对批次取均值，反向传播；将梯度向量的总长度限制为 `gradient_clip_norm=1.0`，再用 Adam 更新。默认 `learning_rate=0.001`、`weight_decay=0.0`、`fixed_epochs=10`、`dropout=0.0`、`label_smoothing=0.0`、无数据增强、`num_workers=0`。这些正式 trainer 参数由当前实现固定检查；只有 device 可选 CPU 或 CUDA 设备，它不是任意调参模型训练入口。

每轮记录训练 loss 和以 0.5 为门槛的训练 balanced accuracy，后者只作历史记录；不是后文冻结动作阈值，也不用于选择 epoch。模型仅保存最终第十轮权重，不用 OOF/外部 PTT 选 epoch。

### 3.6.7 折内阈值：两类“典型得分”的中间

源码：`quality/motion.py:307–355`，`fit_train_only_midpoint_threshold`；`quality/motion_runner.py:555–582`。

直觉：每个人先各自说“我静态时通常多少分、活动时通常多少分”，再把各人的答案放在一起取典型值。两类典型值的中间作为分界，不到测试组中挑一个最好看的门槛。

训练好本折网络后，对本折训练窗口推理。对每位训练参与者 p、类别 c：

`m_(p,c)=median_i p_active(i)`，只包含该参与者该类别窗口。

`m_c=median_p m_(p,c)`；最终 `τ=(m_0+m_1)/2`。

要求每个训练参与者同时有两种活动类，且 `0≤m_0<m_1≤1`；中心颠倒或相同则失败，不把阈值改成 0.5 掩盖失败。测试窗口 `p≥τ` 判 motion=1，等号归活动；否则 static=0。

### 3.6.8 部署阈值：来自完整 OOF，而不是 all-data 自我预测

源码：`quality/motion_runner.py:584–606, 609–724`。

直觉：最终准备实际使用时，仍希望分界线参考“没见过这个人的模型给出的分数”，而不是全体人员训练过的模型对自己的答卷。

内部正式方案为 29 人、单次 participant-grouped 五折、seed 42。每折网络及该折阈值只读训练参与者；每个人一次作为外层 OOF。完成后另在 all29 上训练部署模型，但部署阈值按全部严格 OOF 窗口重新计算“每参与者类别中位数→跨参与者中位数→两中心中点”。不读外部 PTT 去调整阈值。

所以“折内阈值来自训练拟合预测”和“部署阈值来自严格 OOF”是两个不同环节，不能把两者合并说成全都 train-fit 或全都 OOF。

### 3.6.9 外部 PTT 和反向训练比较

源码：`quality/motion_reference.py:914–984`；`quality/motion_runner.py:928–1007, 1100–1204`。

PTT 输入为 22 人的 sit/walk/run 共 66 条记录，先按适配器转换到 400 Hz 并保持光学/IMU 同步。当前採用的 PTT 加速度单位证据是 m/s²，不能再乘 9.81；角速度 deg/s 转 rad/s。每人 sit 提供自己的静态校准。

内部训练→PTT 评估时固定模型和阈值，不在 PTT 拟合中心/尺度/阈值。反向 ablation 则在 PTT 的既定 repeat-0 grouped 五折上训练、生成 OOF、形成部署阈值，再全 PTT 重训，并在 Frailty29 只做推理。两方向的输入变换、标签和训练集归属分别保留。

`MotionOptionId` 中的 `sqi_only`、`sqi_plus_motion_override` 是历史协议描述选项，不是又两种网络算法；当前 classifier 是否实际使用动作检测，以 `artifact.motion_detector_enabled` 及 bundle adapter 的可执行路径为准。

### 3.6.10 在分类 pipeline 中复用动作 bundle

源码：`quality/motion_bundle_adapter.py:49–104, 244–385, 393–521`。

直觉：拿一位已经学好的观察员来工作，同时带上它上课时用的尺子和分界线；不能只拿权重、却换一种输入刻度。

默认 `enabled=false`、`device=cuda`、`batch_size=64`、`threshold_source=bundle_frozen`、`reuse_scope=all29_smoke_or_final_only`；证据路径和摘要默认空。启用需给已有 bundle，不会自动训练一个缺失的模型。

三种复用范围：

- `matching_outer_fold_or_all29_final`：有外层 OOF 时，按训练参与者名单精确匹配 Stage5 折模型及训练阈值；无外层 OOF 时才用 all29 模型。
- `all29_smoke_or_final_only`：仅最终/试运行用途；不能在含 OOF 的分类折中使用 all29 模型。
- `all29_frozen_in_sample_auxiliary`：允许作为明确标注的 in-sample 辅助比较，但 `valid_outer_oof_claim=false`，不能把它写成无泄漏 OOF。

逐窗概率使用原生 8 s/2 s 网格；`p<τ` 为 low，`p≥τ` 为 high。记录级中位数 `median(p_windows)` 只作诊断，当前窗口路由不用它代替各窗概率。缺失信号、无完整 8 s 窗或无效概率返回 unavailable/unfit 证据。

该 adapter 要求训练相同的校准 EKF 输入，且不接收修补过缺口的 native PPG。仅在 finalcase YAML 中将 motion 开关翻成 true、却继续保留 no-gravity 输入，并不自动构成一个兼容的动作 bundle 使用方案。

### 3.6.11 动作评价：每个人先各自算分，再等权平均

源码：`quality/motion_runner.py:282–369`。

直觉：考试按每个人分别算分，再把人的分数平均；不是谁录得久、纸条多，谁就有更大的评价权重。下面都是评估计算，不修改已训练模型。

**两类识别表现。** 用冻结阈值把概率分成 0/1。分别把 static 和 motion 当作目标类，计算 `recall_c=TP_c/(TP_c+FN_c)`、`precision_c=TP_c/(TP_c+FP_c)`（没有预测为该类时 precision=0）、`F1_c=2precision_c·recall_c/(precision_c+recall_c)`（零分母得 0）。`balanced_accuracy=mean(recall_0,recall_1)`，`macro_f1=mean(F1_0,F1_1)`，`sensitivity=recall_1`，`specificity=recall_0`。必须同时有两种真实活动类，不能对只有静态记录者伪造 AUC。

**ROC AUC 的排序实现。** 直觉：随机取一张活动纸条和一张静态纸条，活动纸条是否通常得到更高分？`_rank_average` 对升序概率排秩，相同概率共享平均秩。设正类数 `n1`、负类数 `n0`、正类秩和 `R1`，则 `AUC=[R1−n1(n1+1)/2]/(n1n0)`；平手等价计半次胜出。

**PR AUC 实际是 average precision。** 直觉：从高分纸条开始往低分逐步放宽标准，每多找回一批真正活动纸条，看看目前选中的纸条有多纯。代码按分数降序稳定排序，在每组同分末尾计算 recall 与 precision，`AP=Σ_k(recall_k−recall_(k−1))precision_k`。这不是对 precision–recall 曲线作梯形积分，不能拿另一种“PR AUC”实现替换后期待逐位一致。

**概率误差 ECE。** 将 `[0,1]` 等分 10 格，各格比较平均 `p_active` 与真实活动比例，求 `ECE=Σ_b(n_b/N)|mean(p_b)−mean(y_b)|`。前九格左闭右开，最后一格包含 1。这里看的是活动概率与活动频率，不是取 `max(p,1−p)` 之后的分类置信度版本。

**参与者汇总和最差折。** 同一参与者的评价行必须对应单个冻结阈值。先分别算上述全部指标，再对参与者等权平均；`worst_fold_balanced_accuracy` 是已有折汇总分数的最小值。`parameter_count` 数的是所有 `requires_grad` 参数元素。推理耗时单独测量：默认先热身 10 次、再测 50 次、batch=1，报告每窗毫秒 P50/P95 和 `1000/平均毫秒` 窗/秒；CUDA 计时前后同步，避免只测到异步提交时间。

## 3.7 所有 artifact reducer 的共同接口

源码：`artifacts/base.py:19–211`，`artifacts/router.py:79–107, 135–169`。

输入 `ppg[N,2]`：RED/IR 两列有限 float64，400 Hz；需要时另给同步 IMU 字典。成功输出 `x_ar[N,2]`、confidence、有效性与方法诊断，必须保持原采样网格。失败/unsupported 返回 `x_ar=None`；路由不会偷偷换回 direct，也不连续串接两个非 identity reducer。

直觉：所有整理工具都接收同样宽、同样长度的两条纸带，返回对齐的两条纸带或明确失败；不能少一截后仍说位置完全相同。

除 identity 外，成功输出均标为 rate-only。输出数据结构仍有两列，并不代表两列的原始幅值、独立生理含义或脉搏形状被保留。`confidence` 是各算法自定义的诊断量，不是经过统一校准的“去噪正确概率”，不能跨算法直接比较数值高低。

reducer 参数均从 `artifact.parameters` 传给该算法专属 dataclass。算法对象在当前记录/片段内做的分解、缩放、自适应权重更新，不是跨参与者 classifier 训练；但它们仍可能使用整段记录的未来采样，因此属于离线静态数据处理，不应宣传成逐采样因果在线算法。

下列“别名”指 `artifacts.router.get_reducer` 底层函数接受的名称；正式配置的 `artifact.reducer` 使用 `module_registry.py:671–703` 中的十个规范 ID，不接受全部底层别名。learned/hybrid/ONNX 也只是底层显式 unsupported 分支，不是正式 registry 的可运行模块。

### 3.7.1 共同 IMU 参考：六轴与九通道增强两种

源码：`artifacts/base.py:55–123`，`imu_reference_matrix`。

直觉：把身体向三个方向的移动和绕三个方向的转动放成六把尺子。每把尺子都用这段真正有效的数据重新居中和缩放，不让单位大小决定影响力。

正式参考 `imu_axes6_reference_v2` 使用 `dynamic_acc_mps2[N,3]` 与 `gyro_rads[N,3]`；增强 `imu_axes6_plus_derived3_augmentation_ablation_v2` 再加 `dynamic_magnitude`、`gyro_magnitude`、`jerk_magnitude`。

对有效行 j，逐列 `μ=mean(r_valid)`、`σ=std(r_valid,ddof=0)`；删除 `σ≤10⁻¹²` 的常数列；有效行标准化为 `(r−μ)/σ`，无效行暂置零但仍返回独立 mask。至少 32 个有效行，且至少一列非常数，否则失败。

无效行的零只是数值占位，不是“观测到零动作”；后续算法须传播 mask 或失败。此参考构造供 NLMS、谱掩蔽、PCA/FastICA 选源使用；SSA、EMD、CEEMD-lite、DWT 和 NMF 不靠此 IMU 矩阵。

## 3.8 Identity：不整理的直接对照

源码：`artifacts/identity.py:13–41`；注册名 `identity`，别名 `none/direct`。

直觉：原样复印两条纸带，不修也不挑。

`output=source.copy()`，逐点 `output[n,c]=source[n,c]`，confidence=1，参数为空，`max_absolute_change=0`。复制是为了隔离数组修改，不是数值转换。无效输入仍失败。

identity 保持 direct/identity 路线，不变成 rate-only。finalcase 配置的是它，但在路由模块全关的快路径中并不需要每条记录再调用一次 reducer 来得到相同结果。

## 3.9 NLMS：跟着身体动作估计该减掉的曲线

源码：`artifacts/nlms.py:29–136`，`NlmsReducer.reduce`；注册名 `nlms_imu_anc`，别名 `nlms`。

直觉：身体动作往往会在光学曲线上留下影子。把当前和稍早的动作曲线按不同份量混合，试着画出这个影子，再从光学曲线减掉。每看一个新点，就小幅调整混合份量。

默认参数：`taps_per_delay=8`、`delay_taps=[0,4,8,16]`（样本）、`step_size=0.15`、`epsilon=1e-6`、`leakage=1e-5`、`update_gate_reference_rms=0.10`、六轴 IMU 参考。

逐行数学步骤：

1. 用上一节得到已标准化参考 `r[n]`。延迟集合 `D=sorted({d+t:d∈delay_taps,t=0…7})`；默认并集为 `0…23`，不是 32 个互不重复的抽头。六轴全都非常数时向量维度为 `24×6=144`，而不是 `4×8×6=192`；若常数轴被删除，维度随之减少。
2. 拼出 `v_n=[r[n−d] for d∈D]`；每个波长一套权重，初始化 `W[2,dim]=0`。
3. 若任一延迟位置 IMU 无效，此样本不预测、不更新，输出数值暂保留原值但输出 mask 为假。最初 23 点同样没有完整延迟上下文。
4. 估计动作影子 `ŷ_n=Wv_n`；输出残差 `e_n=x_n−ŷ_n`，将该位置标有效。
5. 若 `sqrt(mean(v_n²))≥0.10`，更新 `W←(1−leakage)W+step_size·e_n v_nᵀ/[epsilon+v_nᵀv_n]`；否则保留当前权重。
6. 诊断 explained variance 为每通道 `clip(1−var(output)/max(var(source),1e-12),0,1)`，confidence 为两通道均值；这不是实际人工标注的干净程度。

`step_size` 必须在 `(0,2)`；缺 IMU、全常数参考等返回失败。运动与真实心跳可能一起变化，减掉动作影子也可能误减真实生理内容，所以结果不承诺形态保真。finalcase 不执行。

## 3.10 SSA：把反复出现的图样拆开后挑选

源码：`artifacts/decomposition.py:22–139`；注册名 `ssa_decomposition`，别名 `ssa/decomposition`。

直觉：从同一曲线上剪很多互相错开一点的小纸条，排成一个矩形。反复出现的起伏会在这个矩形里形成共同图案；把共同图案拆开，再只加回快慢合适的几种。

默认 `embedding_samples=160`、`max_components=12`、`minimum_cardiac_concentration=0.45`、`cardiac_low_hz=0.5`、`cardiac_high_hz=3.5`。

1. 每波长独立执行；实际纸条长度 `L=min(160,N//3)`，构造轨迹矩阵 `H[i,j]=x[i+j]`，形状 `[L,N−L+1]`。
2. `np.linalg.svd(H,full_matrices=False)` 得 `H=U diag(s) Vᵀ`。执行的是完整薄 SVD，不是只求前 12 项的截断求解器；`max_components` 只限制后续检查多少分量。
3. 第 i 个矩阵 `E_i=s_i·outer(U[:,i],Vᵀ[i,:])`。把其中代表同一原始时间 `i+j=n` 的元素求平均，得到一个长度 N 的分量。这是 `_diagonal_average` 的逐行加总与计数相除。
4. 每分量 Welch，`nperseg=min(1024,N)`；计算 `0.5–3.5 Hz` 功率和除以 `0.2–8 Hz` 功率和。
5. 保留占比 `≥0.45` 的分量并相加；一个也没有则失败，不改成“勉强挑最好的一个”。
6. 两波长分别重建再并列。confidence 是各波长候选分量的最高集中度取平均，不是保留分量数量比例。

不需要 IMU，不拟合跨记录参数。短于可用矩阵条件、分解失败或没有合格分量均失败。finalcase 不执行。

## 3.11 Spectral mask：看动作在哪种快慢上强，再压低对应光学起伏

源码：`artifacts/spectral.py:25–222`；注册名 `spectral_mask`，别名 `spectral/stft/stft_imu_mask`。

直觉：把曲线画成一张“横轴时间、纵轴起伏快慢、颜色表示强弱”的图。身体动作在某时某种快慢上特别强，就把光学图对应位置调暗，但在保留范围内不完全擦掉；范围外全部清空。

默认 `stft_window_s=4.0`、`stft_hop_s=1.0`、`imu_mask_quantile=0.75`、`mask_strength=0.80`、`preserve_band_hz=[0.5,3.0]`、六轴 IMU 参考。

1. 4 s 转成 1600 点窗、1 s 转成 400 点请求步长；短记录取 `nperseg=min(1600,N)`，实际步长 `min(400,nperseg)`，重叠量为二者之差。至少 32 点。`Hann` 窗，边缘补零，末尾补至完整 STFT 帧。
2. 对每轴 IMU 作 STFT，幅度图为 `|R_c(f,t)|`；合成 `M=sqrt(mean_c |R_c|²)`。
3. 对每时间帧按全部频点取第 75 百分位 `q_M(t)`；`M'=M/max(q_M,1e-12)`。
4. 每个光学通道同样作 STFT，`P'=|X|/max(percentile_95_f(|X|),1e-12)`。
5. 相对污染 `C=M'/max(M'+P',1e-12)`。
6. 带内增益 `G=clip(1−0.8C,0.2,1)`；0.2 来自 `1−mask_strength`，不是另一个隐藏参数。`0.5–3.0 Hz` 外 `G=0`。
7. 对复数 STFT 直接乘 G，保留剩余分量的相位；ISTFT 重建，截取回 N 点。重建比 N 短则失败。
8. IMU 无效行周围按整窗长度保守扩张为无效区间，输出仍对齐但这些位置不能用；有效点不足 32 则失败。
9. confidence 是有效位置上输入/输出相关的非负裁剪后双通道均值；`1−mean_gain` 另记为 suppression fraction，不能将压得越多解释为越可靠。

带宽须包含至少两个 STFT 频点且不越 Nyquist。输出窄带且 rate-only。finalcase 不执行。

## 3.12 PCA：把双光学曲线的共同变化转到两条新方向

源码：`artifacts/bss.py:85–93, 177–224, 241–330`；注册名 `pca_bss`，别名 `pca`。

直觉：把每个时刻的 RED/IR 当成平面上的一个点。点云可能沿斜方向拉长；旋转坐标纸，就能用两条新方向描述变化。再挑更像心跳、又不太像动作的一条。

1. 要求同步两波长且去均值后矩阵秩为 2；单通道或两通道完全线性重合则失败。
2. `PCA(n_components=2,svd_solver="full").fit_transform(X)`；对去均值矩阵 `X−μ` 做 SVD，得到分量 `S=(X−μ)V` 和 mixing `V`。
3. 每分量算 `C_i`：Welch 的 `0.5–3.5 Hz / 0.2–8 Hz` 功率比。
4. 对有效 IMU 行，计算该分量与各非常数参考轴的绝对相关，取最大值 `R_i`。
5. 选 `argmax_i(C_i−0.25R_i)`。没有另设最低集中度阈值；这与 SSA 的“必须达 0.45”不同。
6. 回投双通道 `X_hat=outer(S[:,i],V[:,i])+μ`，仅保留一个方向；输出有效 mask 沿用 IMU mask。

唯一专属配置是 `imu_reference_profile`，默认六轴。PCA 不使用 random_state、迭代次数或 NMF rank；不能给它设置这些参数后误以为生效。confidence=所选分量集中度裁剪到 `[0,1]`。finalcase 不执行。

## 3.13 FastICA：寻找彼此更少牵连的两条来源

源码：`artifacts/bss.py:95–107, 241–320, 332–339`；注册名 `fastica_bss`，别名 `ica/fastica`。

直觉：两台话筒都同时录到两个人说话。除了旋转到“变化最大”的方向，还试着找两条彼此更少一起涨落的来源。这里两路光学像两台话筒，但实际是否能被这样拆开仍取决于数据。

输入条件、选源评分 `C_i−0.25R_i` 和单来源回投与 PCA 相同；不同的是 `_fit` 使用 `FastICA(n_components=2,whiten="unit-variance",random_state=42,max_iter=1000,tol=1e-5)`。

标准 FastICA 内核先居中并把各方向尺度调整到统一，再用非线性对比函数寻找独立方向。当前调用未覆盖 sklearn 的算法和非线性默认值；执行细节属于安装的 sklearn 实现，不是仓库手写一个等价替代算法。数学上可写成 `S=(X−μ)Wᵀ`；代码读取 `model.mixing_` 得到回投矩阵 A，保留第 i 列后 `X_hat=outer(S[:,i],A[:,i])+μ`。

为便于审阅依赖调用，以下展开项目依赖清单对应 sklearn 1.8.0 的 `_fastica.py`（本地安装源码）的执行步骤，符号 R 专指迭代方向矩阵，避免与最终完整 unmixing 矩阵混淆：

1. `Z=(X−μ)ᵀ`，形状 `[2,N]`；对 Z 做 SVD `Z=U diag(d)Vᵀ`，固定 U 的符号；`K=(U/d)ᵀ` 是预处理映射。
2. `Z_w=KZ·sqrt(N)`；这是供迭代使用的统一尺度输入。
3. seed=42 生成初始 2×2 正态随机矩阵 R，并做 `R←(RRᵀ)^(-1/2)R`，使两方向互不重复。实现用特征分解并把过小特征值裁剪到 dtype 的 tiny，避免除零。
4. 默认 `algorithm="parallel"`、`fun="logcosh"`、`alpha=1`；对 `Y=RZ_w`，`g(Y)=tanh(Y)`、`g'_i=mean_n(1−tanh(Y_i,n)²)`。
5. `R_new=mean_n[g(Y)Z_wᵀ]−diag(g')R`，再对 R_new 做同样的对称去相关。
6. `max_i ||dot(R_new_i,R_i)|−1| < tolerance` 时停止；绝对值使方向正负翻转不被误当成不同解。否则最多 1000 次，未收敛产生 warning 并由本 reducer 视为失败。
7. 原尺度来源 `S=(RKZ)ᵀ`；`whiten="unit-variance"` 再按每列 S 的标准差同时缩放 S 和 R。最终 `components_=RK`、`mixing_=pinv(components_)`。

此处没有把库源码复制进 pipeline，也没有改变库调用；依赖版本更新时应以对应安装源码复核这些默认内核细节。

真实可调参数：`random_state=42`、`max_iter=1000`、`tolerance=1e-5`、`imu_reference_profile`。捕获 `ConvergenceWarning` 并返回失败；不能把未收敛结果当作 success，也不自动回退 PCA。finalcase 不执行。

## 3.14 NMF：用只能相加的频谱积木拼出两路光学

源码：`artifacts/bss.py:109–128, 342–444`；注册名 `nmf_bss`，别名 `nmf`。

直觉：把两张时间—快慢图用几个非负的颜色模板来拼，所有模板只能增加亮度，不能靠正负抵消。挑最集中在心跳快慢范围的模板，配回原来的起伏时序。

默认 `random_state=42`、`max_iter=1000`、`tolerance=1e-5`、`nmf_rank=2`、`nperseg=512`、`overlap_fraction=0.75`。

1. 两波长各作 Hann STFT；`nperseg=min(512,N)`，overlap 为 `min(nperseg−1,round(0.75nperseg))`，边缘/末尾补零。
2. 将两路幅度图按帧轴并列：`M=[|X_RED| |X_IR|]`，形状 `[F,2T_frames]`。
3. 令有效 rank 为 `min(requested_rank,F,2T_frames)`。`NMF(init="nndsvda",solver="cd")` 拟合 `M≈WH`，W/H 均非负；调用的默认目标是平方重建误差。代码 `basis=model.fit_transform(M)` 得 W，`model.components_` 得 H。
4. 每个 basis 评分 `C_i=Σ_(0.5…3.5) W[f,i]/max(Σ_(0.2…8) W[f,i],1e-12)`。这里求的是非负谱基幅度的和，不是 Welch 功率，也没有先平方。
5. 选择最大 `C_i`，用 `outer(W[:,i],H[i,:])` 重建单 basis 幅度。
6. 将帧轴重新分回 RED/IR；分别乘原 STFT 相位 `exp(j·angle(X_c))`，再 ISTFT 并裁回 N 点。

它不读取 IMU，也不使用 PCA/ICA 的动作相关惩罚。只有一个波长、窗太短、NMF 未收敛或重建长度不足则失败。confidence 为所选 basis 比例裁剪值；结果仍 rate-only。finalcase 不执行。

进一步展开所调用 sklearn 1.8.0 `_nmf.py` / `_cdnmf_fast.pyx` 的两项内核，避免仅用“拟合一个 NMF”代替数学步骤：

**NNDSVDA 初始积木。** 对 M 做固定随机种子的截断随机 SVD。第一对向量取绝对值并乘 `sqrt(s0)`。后续每对向量拆成正部和负部，比较两对各自的范数乘积，选择较大者归一化，再乘 `sqrt(s_j·所选范数乘积)`。小于 `1e-6` 的元素置零；`nndsvda` 最后用 `mean(M)` 填充 W/H 中的零。这个 a 后缀不是给零位置加随机噪声。

**交替坐标更新。** 无显式正则时，目标是 `0.5||M−WH||²` 且 W/H 非负。固定 H，设 `A=HHᵀ`、`B=MHᵀ`；逐个 W[i,k] 计算 `gradient=(WA−B)[i,k]`、`hessian=A[k,k]`，若 hessian 非零，执行 `W[i,k]←max(W[i,k]−gradient/hessian,0)`。更新 H 时交换 M 的转置及 W/H 角色，做相同操作。代码默认 `shuffle=false`，按固定列顺序循环。

每次完整更新累计投影梯度绝对值：当前元素为零时取 `min(0,gradient)`，非零时取 gradient；两边 W/H 的绝对值和为 violation。第一轮 violation 为零则停止，否则 `violation/初始violation≤tolerance` 时停止。因而 `tolerance=1e-5` 不是“光学输出最大误差不超过1e-5”的保证。

## 3.15 EMD 筛分：逐层剥去起伏的中线

源码：`artifacts/legacy.py:29–34, 67–120, 229–281`；注册名 `emd_sifting_rate_only`。

直觉：给曲线的山顶拉一条平滑线，给山谷也拉一条；把两条线之间的中线减掉。反复做，剥出一层围绕零上下摆动的薄纹理；再从剩下的曲线继续剥。最后把剥出的层加回，但不加剩余的慢背景。

默认 `max_imfs=6`、`max_sift=10`、`sd_threshold=0.2`。

1. `_local_extrema` 用一阶差分从正变负找极大、负变正找极小；剔除紧邻两端的位置。平顶不按另一套峰算法补点。
2. `_sift_mean` 要求至少两个极大和两个极小，将首尾点也加为结点，用自然边界三次样条分别拟合上、下包络 `u(t),l(t)`；均值 `m(t)=[u(t)+l(t)]/2`。
3. 对当前候选 `h` 执行 `h_new=h_old−m`。
4. 变化量 `SD=Σ(h_old−h_new)²/[Σh_old²+1e-18]`。小于 0.2 或完成十次筛分时结束这一层。
5. 保存候选所需的是“极大数量＋极小数量至少2”（112–115行），不等于包络所需的“极大、极小各至少2”。因此包络不足提前退出内循环时，未经完整筛分的候选仍可能被保存为一层 IMF。随后 `r←r−h`；残余极大少于1或极小少于1即结束，最多六层。
6. 各波长输出所有 IMF 的和，明确排除 residual；不是挑选特定心跳频段 IMF，也不是重加 residual 的完整逆分解。

无 IMF 则失败；不需要 IMU。confidence 固定 0.5，只是历史诊断默认值，不是模型估计的成功率。输出含 `residual_norm_fraction`，并标记形态不保证。finalcase 不执行。

## 3.16 CEEMD-lite + NLMS：先多次轻扰动找坏纹理，再自适应减去

源码：`artifacts/legacy.py:37–52, 122–227, 282–331`；注册名 `ceemd_lite_nlms_legacy`。

直觉：对同一张图撒一点随机细沙，再撒相反的细沙，多做几次并取平均，试着让“拆图”的结果稳定。把不像心跳的层拼成一个干扰参考，再逐点学习应减掉多少。这里参考来自光学曲线自己，不来自 IMU。

### 3.16.1 正负成对扰动与平均分解

默认 `pairs=6`、`noise_ratio=0.2`、`random_seed=2025`；EMD 参数仍 6 层、10 次筛分、0.2 停止阈值。

每波长重新建立固定种子的 `default_rng`，生成 `η~N(0,[0.2(std(x)+1e-12)]²)`，分别对 `x+η`、`x−η` 执行上一节 EMD。每个序号的 IMF 累加，缺少该层的 realization 不提供该层值。

重要的逐行细节：最后分母固定是 `2*pairs=12`，不是成功 realization 数；代码虽记录 `successful_realizations`，不拿它替换分母。平均后 residual 为 `x−Σ平均IMF`。没有任何 IMF 时失败。

### 3.16.2 保护像心跳的层，汇总疑似动作的层

`_welch_peak` 使用最多 8 s 的 Hann Welch，取指定频带功率最大频点。原曲线主频 `f_H` 在 `0.6–3.5 Hz` 找；每 IMF 主频在 `0–8 Hz` 找。

默认 `protect_bandwidth_hz=0.25`、`protect_harmonics=2`。若 IMF 主频位于 `h·f_H±0.25`（h=1,2），归保护层。否则，主频 `≤low_motion_hz=0.4` 或 `≥high_motion_hz=6.0` 归动作层。

其余层与原曲线经过二阶 `0.6–3.5 Hz` Butterworth 双向过滤后的结果算相关；`|corr|≥0.2` 归保护，否则归动作。参考等于所有动作 IMF 之和；residual 在 `0–2 Hz` 的主频若小于 0.4，再加入 residual。

这里相关界值 0.2、residual 主频界值 0.4、保护用滤波阶数/频段是当前历史实现内部常数；后一个 0.4 并不读取 `low_motion_hz`。参考非常数且有限才继续。

### 3.16.3 历史 leaky NLMS 与现代 IMU-NLMS 不同

默认 `nlms_length=32`、`nlms_mu=0.1`、`nlms_leak=1e-4`。每个采样把参考新值塞到 32 点缓存首位、旧值右移；初始化缓存和权重均为零。

`estimate[n]=wᵀv_n`

`clean[n]=x[n]−estimate[n]`

`w←(1−1e-4)w+0.1·clean[n]·v_n/[1e-6+v_nᵀv_n]`。

每点更新，没有现代 IMU-NLMS 的 RMS 更新门槛，也没有多个 delay groups。双波长分别执行，confidence 固定 0.5；没有可用参考或输出非有限则失败，不改用零参考假装成功。finalcase 不执行。

## 3.17 DWT A2 历史分支：取粗略轮廓系数再拉回原长度

源码：`artifacts/legacy.py:55–59, 333–400`；注册名 `dwt_a2_legacy`。

直觉：先用固定的小尺子把细节和较缓的轮廓分开两次，只拿剩下的粗轮廓那一串数，再把它在横轴上拉伸到原图那么长。

1. 每波长调用 `pywt.wavedec(x,"db4",level=2)`；返回粗略系数和两层细节系数。
2. 仅取 `coefficients[0]`，即 A2；至少两个且全部有限。
3. 为 A2 和原 N 点分别建立均匀 `[0,1]` 横坐标。
4. `np.interp(query,knots,approximation)` 线性插值到 N 点。

这不是把细节系数清零后 `waverec` 的标准小波逆重建；也没有补偿 A2 系数的幅值尺度。解释其幅值和形态必须保守，代码因此标记 rate-only。`wavelet=db4`、`level=2` 是此具名历史分支仅支持的设置，不接受其他值。

缺 PyWavelets 时返回 unsupported；不依赖 IMU。confidence 固定 0.5。finalcase 不执行。

## 3.18 Learned / hybrid / ONNX denoiser：登记了名称，但未实现可运行权重路线

源码：`artifacts/router.py:37–59, 105–106`，`UnsupportedReducer`。

`learned`、`learned_denoiser`、`hybrid_denoiser`、`onnx_denoiser` 都返回 unsupported，`x_ar=None`，原因是没有对应可审计模型 artifact。这些名称没有网络层定义、训练步骤或实际推理，不能替它们补写一个想象中的算法。

直觉：工具柜有这些标签，但柜子里没有可用工具；选择标签不会假装已经清理曲线。`parameters` 被记录，不意味着某个学习算法使用了它们。finalcase 不执行。

## 3.19 Stage5 研究模块：如何比较动作检测与去噪效果

这部分是独立 study 工作流，不是每次 frailty finalcase 的暗中前置步骤。源码主入口 `quality/stage5_pre.py:843–1039`。默认 `include_denoiser=True`，但只在显式运行对应研究计划时生效。

### 3.19.1 六个顺序步骤

1. 内部 Frailty29 的五折动作 OOF 和 all29 最终模型。
2. 固定内部模型/阈值对 PTT 作跨数据集动作评估。
3. PTT 的 repeat-0 五折动作训练比较和 all-PTT 最终模型。
4. 固定 PTT 模型/阈值对 Frailty29 反向评估。
5. 汇集对应模型参数包，形成 motion model comparison。
6. 可选 PTT denoiser benchmark；`--no-denoiser` 跳过这一项。

这些步骤复用前述算法。source evidence、schema、hash 的读写负责保存“用哪份数据/哪组参数做的”，不是另一个信号变换算法。断点恢复复用已完成阶段，不在报告脚本中重新训练。

### 3.19.2 PTT 去噪 comparison 的片段生成

源码：`quality/stage5_pre.py:195–204, 610–678`，`run_ptt_denoiser_benchmark`。

直觉：给每种整理工具同一批相同的纸条，然后用同一个数峰方法评估，不能让某种工具拿到更容易的纸条或另用更有利的尺子。

每人用 sit 校准 IMU；每条 sit/walk/run 生成已过滤 PPG 和对应 IMU。`segment_s` 由 study YAML 提供，不在函数签名设一个隐式默认。起点按片段长度前进；若最后一个完整网格起点没有对齐右端，再补右对齐片段，所以尾片段可能和上一片段重叠。

不足 8 s 或参考心搏不足 3 个的片段跳过。对每种 reducer 调用一次双通道处理；失败分别为 RED/IR 保留失败行。成功后两个波长分别用同一配置的峰检测器和下面的时差校正/逐搏匹配评分。这里不按每个片段的真实峰标签挑最优 reducer。

### 3.19.3 峰时间对齐：允许固定或缓慢变化的时差

源码：`quality/stage5_pre.py:206–324`，`_matched_pairs`、`_best_lag_grid`、`_piecewise_shift_reference`、`align_and_score_beats`。

直觉：两条记录的钟可能错开一点，心电和光学变化到达的时间也不同。先整体挪动参考的小旗子，让它们和预测旗子尽可能对应，再数漏掉多少、多插多少。较长记录还可分段允许时差慢慢变。

函数级默认 `max_lag_s=10.0`、`lag_step_s=0.02`、`tolerance_s=0.2`、`lag_window_s=None`；实际 study 可提供不同参数，不把函数默认冒称为所有试验的设置。

候选位移从 `−max_lag_s` 到 `+max_lag_s` 等步长遍历。先用区间累计得到潜在匹配数上界，再用 `match_events` 精确计数；比较键为 `(TP,−|lag|,lag)`，优先 TP 多，再优先位移绝对值小，最后优先较正的位移。剪枝只避免不可能赢的位移，不改评价目标。

若 `lag_window_s` 非空，将参考按固定时长分块，各块独立找位移；预测候选范围向两侧扩展最大位移和容差。最终所有位移后的参考统一逐搏匹配，不因窗口不同重复认领同一预测峰。

`_matched_pairs` 按参考时间顺序，对容差内尚未使用的预测峰取最近者，每个预测最多用一次。为了计算峰间隔误差，额外要求前后匹配在参考序号和预测序号中都连续，不能跨漏峰拼成一个貌似正常间隔。

### 3.19.4 峰值对照指标和参与者等权汇总

源码：`quality/stage5_pre.py:296–324, 416–495`。

`Recall=TP/(TP+FN)`，`PPV=TP/(TP+FP)`，`F1=2TP/(2TP+FP+FN)`；各分母为零按被调用匹配/汇总函数的实现处理。`timing_mae_s` 反映匹配峰的时间差。

对连续匹配峰，使用原始参考间隔 IBI 和预测间隔 PPI，而不是位移后可能跨块跳变的间隔：`e_i=PPI_i−IBI_i`；`RMSE_ms=1000sqrt(mean(e_i²))`，`MAE_ms=1000mean(|e_i|)`。没有可匹配间隔则为空。

汇总先在每位参与者内合计 TP/FP/FN；间隔误差先按 `RMSE_segment²×matched_count` 累积平方和，再开根号。最后对参与者指标取算术均值，避免长记录凭窗口多占更大权重；标准差采用 `ddof=1`，至少两人。失败片段不伪造 F1=0 插进已通过指标平均，另报通过/失败人数、片段数与覆盖率，因此阅读结果必须同时看 coverage。

### 3.19.5 静态峰算法 ablation 与检验

源码：`quality/stage5_pre.py:497–609, 680–727`。

静态峰对照在完整 PTT sit recording 上比较配置指定的检测器，默认与 `aboy_project` 比较的实际名单由计划决定。该路线不跑动态段，也不选择 denoiser。检测器数学在峰值章节展开。

原始报告可给各记录指标的中位数、第 25/75 百分位、IQR 和第 10/90 百分位。`_static_peak_rank_sum_comparisons` 先取两算法共同拥有的参与者/记录名单，再执行 `scipy.stats.ranksums(...,alternative="two-sided")`。虽然名单相同，检验本身是非配对秩和检验，不是 paired signed-rank；代码明确记录 `pairing_used_by_test=false`。

多重比较 `_holm_sidak_step_down`：按 p 从小到大排序，第 k 步剩 m−k+1 项，候选调整值 `1−(1−p_k)^(m−k+1)`，再取截至当前的最大值以保持单调；显著性门槛 `1−(1−α)^[1/(m−k+1)]`。一项未拒绝后，后续项不再拒绝。分别保留全部比较族及计划内预先指定比较族结果。

以上为现有实现的解释，不把“相同记录名单”解释成统计检验使用了配对关系，也不因此修改已有算法。

## 3.20 人工审阅覆盖清单

| 文件/算法单元 | 本章位置 | finalcase 实际执行 |
|---|---|---|
| `signal/sqi.py` 的原始观察、全部 rate/morph/QC component | 3.2 | 否 |
| SQI 固定公式、训练组经验映射、Q_rate/Q_shape/Q_morph、原始诊断 | 3.3 | 否 |
| `quality/routing.py` 的三个模式与真值表 | 3.1、3.4 | 仅全关快路径语义 |
| `quality/routing_timeline.py` 的判断窗、恢复、归属区间、matrix 行路由 | 3.4 | 否 |
| `quality/window_selection.py` 的 none、历史分数、top fraction、三作用范围 | 3.5 | none |
| `quality/motion.py` 的标签、五折协议、训练阈值、历史选项 | 3.6 | 否 |
| `models/motion.py` 的 8/11/10 通道与开发构造器 | 3.6.2–3.6.4 | 否 |
| `quality/motion_adapters.py` 的物化、抽样、训练、推理、权重加载 | 3.6 | 否 |
| `quality/motion_runner.py` 的 OOF、部署阈值、内部/PTT 双向比较 | 3.6.7–3.6.9 | 否 |
| `quality/motion_reference.py` 的同人静态校准与数据源适配 | 3.6.1、3.6.9 | 否 |
| `quality/motion_bundle_adapter.py` 的匹配折/全体模型复用、逐窗判定 | 3.6.10 | 否 |
| `artifacts/base.py` 的公共输入输出、六轴/九通道参考 | 3.7 | 无额外 reducer 计算 |
| identity、NLMS、SSA、spectral mask | 3.8–3.11 | identity 为配置对照；其余否 |
| PCA、FastICA、NMF | 3.12–3.14 | 否 |
| EMD、CEEMD-lite + NLMS、DWT A2 | 3.15–3.17 | 否 |
| learned/hybrid/ONNX unsupported 注册 | 3.18 | 否，且无可运行算法 |
| `quality/stage5_pre.py` 的研究顺序、去噪 benchmark、对齐、汇总与检验 | 3.19 | 否 |

无运算的 `__init__.py`、dataclass 字段、序列化、哈希绑定和路径/形状检查不另冒称一个数值算法；与算法输入、失败行为或拟合范围有关的检查已在对应小节说明。源码调用到的 SciPy、NumPy、sklearn、PyTorch 内部实现没有复制成仓库算法，本章明确给出调用、配置和数学含义，实际数值以安装依赖及源码为准。
