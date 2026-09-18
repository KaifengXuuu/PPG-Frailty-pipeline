# 4. 特征与表征

本章是 `final_pipeline_v5` 的源码伴读文档。下列源码路径除另行注明外均相对于 `src/ppg_frailty/`；行号以本章编写时的源码为准，函数名是代码更新后更稳定的定位入口。`N` 表示采样点数，`P` 表示检测到的峰数，`W` 表示窗口数，`D` 表示特征数，`K_i` 表示第 `i` 条记录的完整窗口数。`fs` 的单位为 Hz，采样点位置除以 `fs` 才变成秒。文中的 `median` 是中间位置的数，`IQR=Q75−Q25`，`MAD=median(|x−median(x)|)`；后两者不是同一个量。

阅读顺序：先定位一次脉搏，再得到脉搏间距，接着描述间距变化、每次起伏的外形和两条光路的关系；另一条并行支路直接描述各个时间窗内的波形；最后将这些结果排成模型需要的数组。每节的“直觉”只解释图形与动作，“代码与数学”才引入公式和术语。

## 4.1 本章模块在 finalcase 中的位置

源码入口：`experiment.py:2648–2666` 的折内准备流程；配置：`configs/presets/finalcase.yaml`。

| 模块 | finalcase 的选择 | 是否形成当前分类器输入 |
|---|---|---|
| 表征 | `representation_mode: raw` | 是，八通道短窗 |
| 窗口 | `windows.raw_dl.length_s: 5`、`hop_s: 2.5`、最多 128 个/recording | 是 |
| 窗内变换 | `signal.normalization.raw_ppg: per_window_robust` | 是，名称虽含 PPG，实际作用于八通道 |
| IMU 后变换 | `signal.normalization.raw_imu: none` | 不再额外拟合折内六轴缩放 |
| 模型输入采样率 | `signal.dl_resampling.target_fs_hz: 64` | 是，400 Hz 分窗和归一化后才重采样 |
| 峰检测配置 | `msptdfast_v2_3_python_port` | 保留配置；并不因此把 PPI/PRV 加入 raw 分类输入 |
| 文件特征组 | 全部七组 | 保留配置；raw 分支不执行 `_extract_vector` |
| 质量/运动/去噪 | `quality.mode: off`，`artifact.reducer: identity`，运动/denoiser 关闭 | 不启用相应分支 |

`experiment.py:2656–2666` 明确区分分支：`feature_vector` 和 `fusion` 才调用 `_extract_vector`，`feature_matrix` 调用 `_extract_matrix_features`，`raw` 和 `fusion` 构造原始窗口。不能因为 YAML 中列着某个模块，就认为每个 run 都执行了它。以下备选模块均保留在代码中，但不是 finalcase 默认全部同时启用。

## 4.2 峰检测共同输入、输出和波长选择

直觉：把两条曲线各自的山顶标在同一张时间尺上。标错的一段可以画叉，但不能把纸剪短，让两段原本相隔很远的时间看起来紧挨着。两条曲线都能给出一套标记，先选标记最完整、最可信的一条作为主线。

源码：`peaks/resolver.py:21–31,84–150,152–207` 的 `detect_pulses_per_wavelength`、`detect_pulses`；`peaks/pairing.py:121–133` 的 `select_reference_wavelength`。

输入为 `[N,1]` 或 `[N,2]` 的有限值波形，或携带合法信号视图的 `CanonicalSignalViews`。管线适配入口固定在 400 Hz 原始时间网格，默认至少 8 秒、5 个峰。输出 `PulseResult` 含 `[P]` 峰位置和秒时间，以及 `[P−1]` 间期、间期两端峰索引、有效标记、相邻关系和检测批次标记。无效间期不是数值零。

代码与数学：

1. `t_j = peak_j / fs`，`PPI_j=t_{j+1}−t_j`，单位为秒。
2. `valid_interval_mask` 表示该间期能否使用；`adjacency_mask` 表示它是否仍然连接原来的相邻事件。两者不可互换。
3. `detector_coverage=sum(valid PPI)/(last_peak_time−first_peak_time)`，分母不是整条记录时长。
4. `AUTO` 的一般比较顺序为 `(detector_score, detector_coverage, is_RED)`；完全相同时选 RED。历史 prominence 单结果入口另有自己的候选比较，见 4.6。
5. 对 artifact-rate-only 波形，检测器先选最长连续有效段并保留起点 offset，不拼接无效缺口两侧。少于最短时长/峰数时返回失败，不凭空补峰。

配置为 `signal.peak_detector.detector_id`、`min_observation_sec`、`min_peaks`、`parameters`。目前只有 MSPTD 分支接受 detector-specific `parameters`，另外三个分支的数值常量位于其源码中，并不是已经暴露的独立 CLI 参数。

## 4.3 MSPTDfast v2：从粗略山顶返回精确时间位置

直觉：先把曲线缩小，看一个点是否比左右不同距离的点都高。小毛刺通常只在很近处像山顶，大一些的起伏则在好几种观察距离下都站得住。找到大致位置后，回到原图附近挑最高点。

源码：`peaks/msptdfast_v2.py:29–34` 默认值；`_msptd_window_peaks:70–95`；`detect_msptdfast_v2:97–142`；`_prepare_input:144–191`；`detect_pulses_per_wavelength_msptdfast_v2:193–282`。

### 4.3.1 输入与窗口

direct/identity 路径取修复后的 `x_native`，不是已经 0.2–8 Hz 滤波的 `x_filter`，因为该检测器自己在窗口内去直线趋势。artifact-rate-only 路径取 `analysis_signal` 及其有效样本标记。输出峰位置仍是 400 Hz 原坐标。

默认参数是 `target_downsample_hz=20`、`minimum_heart_rate_bpm=30`、`window_s=6`、`overlap_fraction=0.2`。

`nominal=round(window_s*fs)`；窗口长度实际为 `nominal+1`，来自参考实现包含右端点的约定。步长为 `round(nominal*(1−overlap))`。末尾再放一个右对齐窗口以覆盖尾部。内部抽取因子 `q=max(1,floor(fs/target))`，实际粗网格采样率为 `fs/q`，不保证任意目标值都精确实现。

### 4.3.2 每个粗网格窗口内的比较

1. 第 78 行 `signal.detrend(..., type="linear")` 去掉最小二乘直线：`x'_i=x_i−(a i+b)`。
2. 第 82–88 行令 `L=ceil(n/2)−1`、窗口秒长 `T=n/fs_coarse`；只保留满足 `(L/k)/T >= min_bpm/60` 的距离 `k`。
3. 第 90–93 行构造布尔表 `M[k,i]=(x'_i>x'_{i−k}) AND (x'_i>x'_{i+k})`。没有左右邻居的位置为 false，等高平台也不满足严格大于。
4. 第 94 行选命中数量最多的那一行 `λ=argmax_k sum_i M[k,i]`，并非简单取最大允许距离。
5. 第 95 行要求 `i` 在第 1 至 `λ` 行全部为 true，才成为粗峰。

### 4.3.3 回到原始网格

第 124–141 行取 `segment[::q]`，这是检测器内部的直接抽点，不是整个模型输入的抗混叠重采样。粗峰 `p` 的初始位置为 `(p+1)q−1`，不能随意改成 `p*q`。以该位置为中心搜原始波形局部最大值：粗采样率 `<10` 时半径 0.2 秒，`<20` 时 0.1 秒，否则 0.05 秒；样本半径取 `ceil(fs*秒数)`。各窗口结果经 `np.unique` 排序去重。

最终只按 `60/210 <= PPI <= 60/35` 标记有效；峰仍保留；置信数组全部为 1，不能解释成经过校准的真实概率。检测分数为 `有效间期数量 + 0.5*coverage`。默认只检测向上的峰，`selected_polarity=1`，没有另外跑反向曲线。

短窗口少于 5 个粗点则没有粗峰；完整输入非法、时长不足或最终峰数不足会失败。源码自称参考 MATLAB 方程的 Python port，不承诺与 MATLAB 逐位相同。

## 4.4 Aboy project v1：整条记录选方向，十秒一段调整尺子

直觉：把曲线正着看一次、倒过来看一次。每十秒先粗略找山顶，再根据这一段山顶间隔是否稳定，调整“两个山顶至少应隔多远”和“山顶要高出旁边多少”。上一段得到的尺子交给下一段。最后整条记录只保留一个方向的结果。

源码：`peaks/aboy_project.py:27–32,72–196,198–369,420–517,519–560`，重点函数 `_block_parameters`、`_block_candidate`、`_clean_intervals`、`_polarity_candidate`。

输入使用合法 `analysis_signal` 的私有副本。默认十秒完整块，尾部不足十秒丢弃；即使公共最低时长为 8 秒，本分支仍至少需要一个完整十秒块。每个波长独立运行。

### 4.4.1 自适应滤波与初步峰

对方向 `s∈{+1,−1}`，先令 `z=s*x`。初始 `HRI=0`，它是代码内用于调节窗口的数，不是每分钟心率。

1. `d210=max(1,round(fs*60/210))`，限制初步峰间最短距离。
2. 上截止频率 `f_hi=min(8,max(1.5,3*(1+HRI_in)))`。
3. `_bandpass_block` 对当前块做二阶 Butterworth 0.5–`f_hi` Hz SOS 前后向滤波；不会把结果写回用于形态特征的原始分析视图。
4. `find_peaks(filtered,distance=d210)` 得到初步峰，差分得到初步间期。

### 4.4.2 更新尺子并再次找峰

第 224–247 行对应：

1. 仅保留严格大于初步间期第 30 百分位的间期 `R`。
2. 若 `|R|>=2` 且 `0.5*median(all preliminary PPI) <= mean(R) <= 1.5*median(all preliminary PPI)`，更新 `HRI_out=10*std_population(R)/mean(R)`；否则沿用旧值。
3. `HRwin=fs/(3*(1+HRI_out))`，最终距离为 `max(round(2*HRwin),d210)`。
4. 将初步峰处的波形高度排序，从 `floor(0.70*n)` 位置取到末尾求平均；当峰数小于 3 时取全部。
5. 显著程度阈值 `max(0.25*max(上述平均值, std_population(filtered)),1e−12)`。
6. 再调用 `find_peaks`，同时使用该距离和显著程度阈值。

`find_peaks` 的具体峰底搜索、等高峰处理由 SciPy 实现，本仓库这里只设置参数和使用返回的 prominence；不能把它简化为“高于零的峰”。

### 4.4.3 间期清理与最终方向

`_clean_intervals:139–196` 分三步：

1. 生理范围 `[60/210,60/35]` 秒，端点包含。
2. 若范围内至少 5 个间期，计算其中位数 `m` 和 `σ_r=1.4826*median(|PPI−m|)`。当 `σ_r>0`，保留 `|PPI−m|<=4σ_r`；`σ_r=0` 时不追加这项剔除。
3. 在前两步剩余间期上取中位数 `r`，要求 `0.5r<=PPI<=1.8r`。不合格间期的后一个峰标记为不接受，该峰两侧间期也无效；不删除峰，不把两侧重新连成新间期。

每种方向拼接各十秒块，保留原时间顺序并只去掉完全相同位置的峰。整体分数 `S=n_clean+0.5*coverage−min(CV,2)`，`CV` 使用有效 PPI 的总体标准差/均值；不足两个有效间期时 `CV=2`。比较顺序是分数、有效间期数、总峰数、方向值（最后正方向胜出）。置信为 `clip(prominence/(2*median(prominence)),0,1)`，不是概率校准。

选择 ID 为 `aboy_project_v1`；这些十秒、频率、阈值是此算法版本本身的源码常量。它是项目的 Aboy-inspired 实现，不声称原作者实现的逐位复刻。

## 4.5 Aboy project v2：每段选方向，先删峰再清理间期

直觉：不再强迫整段记录始终向同一面看，而是每十秒各自选择更清楚的朝向。先把明显不合拍的山顶擦掉，再测剩下山顶之间的新距离。这个“擦掉后重测”与上一节“保留标记但画叉”会产生不同的距离。

源码：`peaks/aboy_project_v2.py:58–155,170–260,263–373,376–410`，主要函数 `_prepare_input`、`_candidate`、`_remove_ratio_outlier_peaks`、`_result_for_channel`。

1. direct/identity 读取修复后的 native PPG；artifact-rate-only 读取 reducer 输出。第 110–124 行先对整个选定连续段做二阶 0.2 Hz 高通，再分完整十秒块。
2. 块内 0.5–自适应上截止滤波、初峰和 `HRI_out` 公式与 v1 类似，但更新检查用的是 `median(retained PPI)`，不是 v1 的全部初步间期中位数。
3. 上 30% 高度取 `ceil(0.3*n)` 个最高峰；v1 在小于 3 个峰时取全部，因此两者在小样本下也不同。
4. 每块用 `(score,n_clean_ppi,peak_count,polarity)` 选一次方向；下一块只继承胜出方向的 `HRI_out`。v1 则让两种方向各自跑完整条记录后再选。
5. 所有块的胜出峰拼起来，至少 3 个峰时计算**未先做生理/MAD清理**的 PPI 中位数 `r`；将 `<0.5r` 或 `>1.8r` 的间期后端峰从数组实际删除。这是一轮同时删除，不是循环直到稳定。
6. 用删峰后的时间点重新差分，再做 `[60/210,60/35]` 范围和 `4*1.4826*MAD` 检查；被保留的峰其 `accepted_peak_mask` 全为 true，间期有效性另存。
7. `selected_polarity` 是各块胜出方向之和的符号，平局为正。它只是一条记录的汇总方向，不等于每个块都采用这个方向。后续形态函数当前读取该单一方向，这是阅读代码时必须看见的实际行为。

最后 `peak_ordinals` 在删峰后的序列上重新编号，`adjacency_mask` 全 true；不能把 v1 的“原候选峰绝不删除”描述套到 v2 上。其分数和置信公式见 v1，但有效 PPI 的定义按本节执行。默认至少完整十秒和最终 5 峰，ID 为 `aboy_project_v2`，不接受额外 detector 参数。

## 4.6 历史双方向 prominence 消融

直觉：整条曲线正着、倒着都找一次较突出的山顶。不随着时间调整尺子，而是让山顶之间至少隔固定距离，再比较哪种结果更整齐、更像连续的起伏。

源码：`signal/peaks.py:35–72,75–225` 的 `_robust_scale`、`_candidate`、`_detect_pulses_dual_polarity_ablation`。ID 为 `dual_polarity_prominence_v1_ablation`。

`scale=(Q75−Q25)/1.349`，退化时用 `max(std_population,1e−12)`；对每个候选方向/波长用 `distance=round(0.30*fs)`、`prominence=max(0.15*scale,1e−12)` 找峰。

评分逐项为：

- `plausible`：落在 `[0.30,2.00]` 秒的间期比例。
- `density=exp(−((peak_count/duration−1.4)/1.4)^2)`。
- `prominence_score=tanh(median(prominence)/scale)`，无峰为 0。
- `CV=std_population(PPI)/mean(PPI)`；不足两峰时设 `plausible=0,CV=1`。
- `S=0.55*plausible+0.20*density+0.20*prominence_score+0.05*exp(−CV)`。

整条记录的候选比较为 `(score,peak_count,−channel_index,polarity)`；RED 索引较小因而优先。`detect_pulses_per_wavelength` 逐波长调用时只比较该波长的两个方向。最终仅按 `[0.30,2.00]` 秒标记间期，没有 Aboy 的 MAD 和参考比清理；至少有一侧有效间期的峰为接受峰。置信与 v1 一样按峰突出程度归一化。

## 4.7 PPI 清理选项的真实边界

直觉：四套找山顶方法自带四套“哪些距离算可信”的规则。换找山顶的方法，也换了距离清理方式；当前代码并没有另外一个可以任意混搭的“清理开关”。

| 检测器 | 范围 | 额外清理 | 是否实际删峰 |
|---|---|---|---|
| MSPTD | 35–210 bpm 对应秒范围 | 无 | 否 |
| Aboy v1 | 35–210 bpm | 先 MAD，后清理后中位数参考比 | 否，只标记后端峰和相邻间期无效 |
| Aboy v2 | 35–210 bpm | 先原间期中位数参考比删峰，再 MAD | 是，随后重新求 PPI |
| prominence 消融 | 0.30–2.00 秒 | 无 | 否 |

`features.prv_primary_backend` 和 `prv_library_comparison_scope` 也不提供新的 cleaner。相关库比较只接收完全相同的固定 PPI，见 4.16。

## 4.8 路由后拼接合法 PPI：保留真正的间断

直觉：不同颜色标记的时间段可以放在同一条长尺子上，但换颜色的位置要留一道断线；不能因为把结果放进同一个表格，就把两段之间当成正常的一次心跳。

源码：`features/window_matrix.py:114–263` 的 `_eligible_event_segment`、`build_route_eligible_rate_pulse`。

对 direct 和可选 processed 的每个 PPI，用其左右端时间乘 400 回到样本坐标。两端必须落在同一个 routing cell，该 cell 的来源与 PPI 来源一致，且最终质量为 excellent 或 acceptable。代码用右端 `stop−1e−9` 处理右开区间。

将满足条件的间期按原序号连续、同 cell、共用端点、同 detection run 划成小段，按起始时间排序。段内保留原始 PPI/有效性/邻接标记；段与段之间插入 `PPI=NaN`、`valid=False`、`adjacency=False`、`run_id="routing_boundary"` 的连接行。所有段必须使用同一波长，不把 RED 的半段接到 IR 的半段。没有合法段则失败。

该组装不拟合全数据的统计量；它消费已经由上游生成的路由决定。不能把这些路由结果当作跨折通用的确定性 cache。

## 4.9 基础间距和每分钟次数

直觉：先看相邻山顶之间的距离，报出通常有多远、变化有多大；再把“每次要等多久”换算成“一分钟大约有多少次”。先平均等待时间再取倒数，与先逐次换算再平均并不是一回事。

源码：`signal/prv.py:332–398` 的 `compute_prv`；默认配置 `PrvConfig:18–41`。

先设 `valid = valid_interval_mask & adjacency_mask & isfinite(PPI) & (PPI>0)`，有效间期为 `r_1…r_m`。记录有效个数 `m`、有效时间 `sum(r)` 和覆盖 `sum(r)/(last_peak−first_peak)`。三项诊断即使其它 PRV 不合格仍可有值。

默认基础资格为观测时长至少 8 秒、总峰数至少 5、有效间期至少 4；对应 `features.rate_prv_min_duration_s`、`rate_prv_min_peaks`。

| 输出 | 代码公式 | 单位 |
|---|---|---|
| `ppi_mean_s` / `ppi_median_s` | `mean(r)` / `median(r)` | 秒 |
| `ppi_sd_s` | 样本标准差 `sqrt(sum((r−mean(r))²)/(m−1))` | 秒 |
| `ppi_iqr_s` / `ppi_mad_s` | `Q75−Q25` / `median(abs(r−median(r)))` | 秒 |
| `ppi_cv` | `std_sample(r)/mean(r)` | 无单位 |
| `hr_mean_bpm` / `hr_median_bpm` | 先逐项 `h_i=60/r_i`，再求均值/中位数 | 次/分钟 |
| `hr_sd_bpm` | `std_sample(h)` | 次/分钟 |

不合格字段保留 NaN/false；不是用 0 表示没有波动。

## 4.10 时间变化：SDNN、RMSSD、SDSD、NN50 与 pNN50

直觉：一组距离可能整体分散，也可能只是慢慢变长。要分辨“相邻两次忽长忽短”，需要只看真正紧邻的两段距离的差；断线两边绝不能相减。

源码：`signal/prv.py:400–432`。默认 `features.time_prv_min_duration_s=60`、`time_prv_min_coverage=0.8`、`time_prv_min_intervals=30`。

1. 三项资格同时满足才计算 `sdnn_s=std_sample(r)`。
2. 配对条件为两个间期都有效、两个 adjacency 都 true、`stop_index[i]==start_index[i+1]`。
3. 只对该配对集合计算 `d_i=PPI_{i+1}−PPI_i`。
4. `RMSSD=sqrt(mean(d²))`；`SDSD=std_sample(d)`，仅一个差值时后者为 0。
5. `NN50=count(abs(d)>0.050)`，严格大于 50 ms；`pNN50=NN50/len(d)`，是 0–1 比例而非百分数。

若没有合法相邻差值，SDNN 仍可用，其它差值指标缺失，并记录原因。这里的标准差用 `ddof=1`，不同于后文短窗矩阵的总体标准差。

## 4.11 Poincaré 两个方向的宽度

直觉：把“这一段距离”和“下一段距离”画成平面上的点。沿左下到右上的方向分散，表示两段一起变长变短；垂直这个方向分散，表示一长一短的交替更明显。

源码：`signal/prv.py:417–427`；属于 `features.enabled_groups: hrv_nonlinear`。

使用上一节同一组合法差值：`SD1=sqrt(0.5)*SDSD`；`SD2=sqrt(max(2*var_sample(r)−0.5*SDSD²,0))`；最后 `SD1/SD2`。两轴单位秒，比例无单位。分母为 0 时比例缺失。这一实现直接由间期与差值的方差计算，并没有另行拟合椭圆。

## 4.12 样本熵：相似短片段延长后是否还相似

直觉：在一串距离中，找两小段，它们前两步很像；再各多看一步，看它们是否仍然像。如果延长后经常分开，说明局部走向较难猜。中间断掉的距离不能跨过去凑小段。

源码：`signal/prv.py:272–308` 的 `_sample_entropy`，以及 `434–458` 的资格选择。默认 `features.sample_entropy: {m: 2, r_sd_fraction: 0.2, min_intervals: 200}`。

1. 在 `valid & adjacency` 的连续段中，选累计 PPI 时间最长的一段，不是简单选元素最多的一段；只用这一段 `x`。
2. 默认至少 200 个连续间期。容差 `r=0.2*std_sample(x)`；容差非正或不有限时缺失。
3. 长度 `m` 的片段 `u_i=(x_i,…,x_{i+m−1})`，两片段满足 `max_j|u_i[j]−u_k[j]|<=r` 记一次匹配；不计算自己与自己，不重复计算 `(i,k)` 与 `(k,i)`。
4. 分别计算长度 `m`、`m+1` 的匹配次数 `C_m,C_{m+1}`，以及对应可比较对数 `A_m=n_m(n_m−1)/2`。
5. 输出 `−log((C_{m+1}/A_{m+1})/(C_m/A_m))`。必须是**匹配概率之比**，不是两种长度的原始匹配次数之比。

任一匹配概率为零返回 NaN；不是无限大。该实现为两层模板比较循环，长序列成本可接近平方级；不是分类器训练过程。

## 4.13 频域 PRV：在真实时间上看快慢摆动

直觉：把每次测得的距离放回它真正结束的时刻，再在一张等间距时间纸上连成曲线。观察这条曲线是缓慢起伏较多，还是较快起伏较多。中途很长的空白不能剪掉，否则快慢都会被改写。

源码：`signal/prv.py:311–317,460–518`。

默认资格：角色 family 为 B 或 R；route 为 direct、identity 或 artifact-rate-only；`q_rate_qualified=True`；总观察至少 300 秒；最长连续有效 PPI 的累计时间至少 300 秒；至少 200 个连续间期；总覆盖至少 0.8。

对应配置：`features.spectral_prv_min_duration_s=300`、`spectral_prv_min_intervals=200`、`spectral_prv_min_coverage=0.8`、`tachogram_fs_hz=4`，三个频带在 `spectral_bands_hz` 中。

1. 间期时间戳使用该间期的右端峰时间，不是累计“筛选后的有效 PPI”。
2. 等距网格从首个有效右端时间到最后一个右端之前，间距 `1/4` 秒。
3. 线性插值 PPI，再去最小二乘直线趋势。
4. 若网格至少 256 点且时间严格递增，调用 Welch，`nperseg=min(1024,N_grid)`、`detrend=False`；未显式传入的窗函数/重叠规则由安装的 SciPy 默认实现决定。
5. VLF `[0.003,0.04]`、LF `[0.04,0.15]`、HF `[0.15,0.40]` Hz，分别对区间内频率点做梯形积分。两端均包含，共用边界点按代码保留；某频带少于两点则 NaN。
6. 输出功率单位秒²；`LF/HF`、`LF/(LF+HF)`、`HF/(LF+HF)` 为无单位比例，分母不正时缺失。

资格足够不保证所有频带积分都足够点数。角色 W/S 即使很长，也不会进入本实现的正式频域 PRV。

## 4.14 脉搏形态：山脚之间的一次起伏

直觉：对每个山顶，在它左右各找一个谷底，在两个谷底之间拉一根直绳。山顶高出绳子的高度、爬上去和落下来的时间、半山腰的宽度、绳子上方围起来的面积，就是这一小段曲线的外形。

源码：`signal/morphology.py:13–21,33–61,63–166` 的 `extract_morphology`、`_crossing_time`。

输入为 `[N]` 或 `[N,2]` 的 `x_filter`、对应波长 `PulseResult`，固定 400 Hz。仅 direct/identity 可用；经过非恒等 artifact reducer 的曲线没有保真外形含义，不能拿来算这些指标。

代码与数学：

1. 第 93 行用检测器记录方向 `s` 乘波形，保证按向上的起伏测量。
2. 只考察第 2 个到倒数第 2 个峰，且中心峰须被接受。左右搜索边界是相邻峰与当前峰的整数中点。
3. 在左边界到当前峰、当前峰到右边界之间各找最小值位置 `l,r`，要求 `l<p<r`。
4. 谷底连线 `b(n)=x_l+(x_r−x_l)*(n−l)/(r−l)`；`y(n)=s*x(n)−b(n)`，其中谷底值也取已翻转的波形。
5. `A=y(p)>0` 为 amplitude。半高交点从邻近样本线性插值得到，左边取最靠近峰的一次交叉、右边取峰后的第一次交叉。

| 独立形态量 | 公式 | 单位 |
|---|---|---|
| 高度 `amplitude` | `A` | PPG 原计数单位 |
| 半高宽 `width_half_s` | `(right_cross−left_cross)/fs` | 秒 |
| 上升时间 `rise_s` | `(p−l)/fs` | 秒 |
| 下降时间 `decay_s` | `(r−p)/fs` | 秒 |
| 上升斜率 `rise_slope_per_s` | `A/rise` | PPG 单位/秒 |
| 下降斜率 `decay_slope_per_s` | `−A/decay` | PPG 单位/秒；保留负号 |
| 正面积 `positive_area` | 梯形积分 `max(y,0)`，步长 `1/fs` | PPG 单位·秒 |

每项都有独立逐搏 validity；全记录再各取 median 和 MAD，共 14 个字段，至少 3 个有效搏动才标记汇总有效。没有半高交点只影响半高宽，不自动抹掉已经可计算的上升/下降时间等指标。

## 4.15 双光路：配对、各自高度/底座和曲线相似程度

### 4.15.1 峰配对

直觉：先选择一条标记更可靠的时间线，在每个山顶左右邻居的中点画两条竖线，形成互不重叠的小格。另一条曲线在同一格中的最近山顶与它配成一对；一个山顶只能用一次。

源码：`peaks/pairing.py:148–315` 的 `pair_dual_wavelength_beats`。

两波长必须来自同一 detector 和 route。参考波长按 4.2 的 score/coverage/RED 规则选择。对被接受的参考峰 `p_j`，使用**被接受参考峰序列**的左右邻居，格子是 `[(p_{j−1}+p_j)/2,(p_j+p_{j+1})/2)`。首尾参考峰缺完整格子不配对。

在格子内选被接受且未用的另一波长峰，按绝对时间距离、较早位置、较小原编号依次打破平局；没有额外固定毫秒容差。落单和被拒绝峰也写审计行。配对时差恒为 `IR位置−RED位置`，除以 `fs` 得秒；正数表示 IR 较晚。

另一个辅助比较函数 `match_events:318–342` 用于有参考事件的诊断：按参考时间顺序逐个找容差内最近且未使用的预测事件，得到 TP/FP/FN、precision、recall、F1 与匹配绝对时间误差均值。这不是上述正式双波长配对算法。

### 4.15.2 AC、DC 与 PI

直觉：两条曲线虽然来自同一次起伏，也要各自找自己的山顶和谷底。高度描述“这次跳动抬起多少”，底座描述“整条曲线所在的亮度层”。把高度与底座大小相除，得到不那么依赖整体亮度的相对起伏。

源码：`signal/optical.py:128–189,191–331` 的 `_wavelength_local_ac_dc`、`extract_dual_optical`。

输入为 native 与 filter 两个 `[N,2]` 数组、两波长独立峰结果；只允许 direct/identity、400 Hz。每对峰在各自波长上重复谷底搜索：AC 取翻正的 filter 波形峰值减去谷底连线；DC 取同两个谷底位置在 native 波形中的连线在峰处的高度，**DC 不随峰方向翻转**。

1. 只有两波长都得到正 AC 和有限 DC 的配对才进入共同集合。
2. 每个配对可计算 `PI_R=AC_R/(abs(DC_R)+ε)`、`PI_I=AC_I/(abs(DC_I)+ε)`，`ε=1e−12`。
3. AC 比值 `AC_R/(AC_I+ε)`；DC 比值 `abs(DC_R)/(abs(DC_I)+ε)`；比值的比值 `PI_R/PI_I`。分母绝对值不大于 ε 时缺失。
4. 正式文件预测字段先在共同集合中分别取 `median(AC_R)`、`median(AC_I)`、`median(DC_R)`、`median(DC_I)`，**再由这四个数算比值**。字段名虽带 `_median`，并非“逐搏比值的中位数”。
5. 四个基础中位数和五个派生比值共 9 字段，至少 3 个共同有效配对才标有效。逐搏比值仅作诊断，不进入正式预测。

这不是血氧饱和度公式，代码没有把 ratio-of-ratios 映射到 SpO₂。

### 4.15.3 零时差与平移相似程度

直觉：先把两条曲线放到相同的高度中心与大小，再看它们是否一起上升下降。也允许把其中一条轻轻往左或往右挪，找最像的位置；移动方向说明谁比较晚。

源码：`signal/optical.py:66–117,333–345`。

每条完整 filter 波形仅标准化一次：`z=(x−mean(x))/std_population(x)`。零时差值为 `dot(z_R,z_I)/N`。在 `[-0.5,+0.5]` 秒所有整数样本位移中比较重叠部分：`ρ(lag)=dot(a,b)/(norm(a)*norm(b))`。这里每个 lag 重新算重叠部分向量长度，但**不再重新减重叠部分均值**。

取最大的有符号相关值，不是绝对值最大；相同时优先绝对 lag 最小，再优先有符号 lag 较小。输出零 lag 相关、最大相关、对应 lag 秒，共 3 字段；正 lag 表示 IR 晚。常数曲线无法缩放则三项缺失。代码没有计算 coherence。

## 4.16 PRV 外部后端对照：不是分类流程替换模块

直觉：把完全一样的一串距离交给不同计算器，比较它们报出的数字。不能一边换计算器，一边先改输入数字，否则不知道差异来自哪里。

源码：`features/prv_backend_compare.py:47–107,115–204,206–249`。可选 `local`、`aura_hrv_analysis`、`rhenan_hrv`，仅固定 PPI 向量诊断，不清理、不接分类器。

输入单位是毫秒，至少 4 个有限正值。local 直接计算均值、中位数、最大减最小、样本 SDNN/SDSD、RMSSD、`abs(diff)>50` 次数、百分比 `100*mean(abs(diff)>50)`、`CVNN=SDNN/mean(PPI)`、`CVSD=RMSSD/mean(PPI)`，以及 `60000/PPI_ms` 的均值、最小、最大和样本标准差。这里 `pnni_50` 是百分数，而 4.10 的 `pnn50` 是比例。

Aura 依次调用 `hrvanalysis` 的 time-domain、geometrical、frequency-domain、CSI/CVI、Poincaré、SampEn 函数；rhenan 先构造 `hrv.rri.RRi`，再调用 `hrv.classical.time_domain/frequency_domain/non_linear`。库内部方程未在本仓库适配器中定义，本章不假装这些内部代码已逐行核对。依赖缺失返回 `unavailable_optional_dependency`，其它异常返回 `backend_failed`，不自动改用另一库。

固定测试向量包括恒定 800 ms、760/840 ms 交替、两种周期叠加、740→880 ms 缓变、单个 1400 ms 离群点；每条 512 个间期，输入哈希用于确认各后端实际接收同一串数。

## 4.17 工程特征输入：每个完整时间窗的一行描述

直觉：不必先找到山顶，也可以给一段曲线写一张小卡片：通常在多高、上下摆多宽、是否经常冒出尖角、慢摆和快摆各有多少。两条光线和身体运动的几条曲线都用同样的尺子写卡片。

源码：`features/engineering.py:20–41,199–275,278–356` 的 `engineering_feature_names`、`_imu_columns`、`extract_engineering_features`。

输入来自 400 Hz amplitude-preserving PPG 与 SI 单位 IMU，窗口计划来自 `windows.engineering`，默认 10 秒/2 秒。输出 `[W,115]`，同时保存同形状逐值 validity、`[W]` 起点和行标记。只接受完整、无 padding 窗口；可配置长度/步长，不因为 schema 名含 `10s_hop2s` 就认为时长不能变化。

| 通道组 | 每通道字段 | 通道数 | 小计 |
|---|---|---:|---:|
| `ppg_red`, `ppg_ir` | 7 个时间统计 + 4 个频谱汇总 + 3 个频带功率 | 2 | 28 |
| 动态加速度长度、角速度长度、jerk 长度 | 7 + 4 + 4 个频带功率 | 3 | 45 |
| 动态加速度 x/y/z、角速度 x/y/z | 7 个时间统计 | 6 | 42 |
| 合计 | 固定顺序 | 11 | 115 |

加速度单位 m/s²、角速度 rad/s、jerk m/s³；相应长度为三轴平方和开根号。工程支路不使用八通道 DL 窗口的归一化副本。非恒等 rate-only route 下 28 个 PPG 字段是 NaN/false，IMU 字段仍可计算；但进入 146 矩阵时还有更严格的行级规则，见 4.23。

## 4.18 工程时间统计：七把不同的尺子

直觉：一把尺看平均高度，一把看相对平均位置的散开程度，一把把负的摆动也计入大小；还有两把少受极大尖峰干扰的尺；最后两把描述两侧是否对称、是否容易出现特别远的尖点。

源码：`features/engineering.py:124–181` 的 `_one_channel_features`。

先检查有限样本占比，少于 80% 时该通道全部字段缺失。时间统计对所有有限值 `x_1…x_n` 计算，不要求它们连续；频谱另用最长连续段。

| 小算法 | 对应行 | 公式及含义 |
|---|---|---|
| mean | 175 | `μ=sum(x)/n`，保留波形基线 |
| population_sd | 176 | `sqrt(sum((x−μ)²)/n)`，`ddof=0` |
| rms | 177 | `sqrt(sum(x²)/n)`；不先减均值 |
| iqr | 178 | `Q75−Q25`，中间一半数据的宽度 |
| mad | 179 | `median(abs(x−median(x)))`；未乘 1.4826 |
| skew_bias_corrected | 172,180 | `scipy.stats.skew(x,bias=False)`；标准化三阶中心矩的样本量修正 |
| pearson_kurtosis | 173,181 | `scipy.stats.kurtosis(x,fisher=False,bias=False)`；Pearson 定义，常见正态参照为 3 而非 0 |

前五项单位与源通道相同；后两项无单位。令 `m_k=mean((x−μ)^k)`，非退化足够样本时偏度为 `sqrt(n(n−1))/(n−2)*m_3/m_2^(3/2)`；Pearson 峰度为 `3+((n²−1)*m_4/m_2²−3(n−1)²)/((n−2)(n−3))`。精确的小样本/近常数处理由上述 SciPy 调用实现；返回不有限的值标记缺失，代码只抑制预期的精度警告，不替换返回值。

## 4.19 工程频谱：Welch 公共底座

直觉：把同一小段曲线分成几片，在每片两头稍微收窄，再问“用多少种快慢不同的波相加，能拼出这一片”。把各片的答案平均，减少一小处偶然尖点的影响。

源码：`features/engineering.py:86–103,141–161` 的 `engineering_welch_parameters` 和 `_one_channel_features`。

1. 有缺口时只取最长连续有限 run；不把缺口两侧拼起来后做频谱。
2. 每小片长度 `L=min(N_run,max(64,min(2048,round(4*fs))))`；默认 400 Hz 时完整窗口一般取 1600 点，即 4 秒。
3. 相邻小片重叠 `L//2`，窗函数显式为 `hann`，只返回单边谱。`detrend` 未显式传入，使用 SciPy Welch 默认。
4. 抽象表达：对第 `b` 小片 `y_b[n]` 加权后做离散变换 `Y_b[k]=sum_n w[n]*y_b[n]*exp(−j2πkn/L)`，按采样率和窗能量归一化其平方幅度并平均；单边频谱中正负对称频率的合并规则由 SciPy 执行。

输出频率 `f` 单位 Hz、功率密度 `P(f)` 单位为“源单位²/Hz”。下面五类频谱算法都复用这个结果；不是每个字段重新过滤/变换一遍。

## 4.20 工程频谱汇总与频带功率

### 4.20.1 总功率

直觉：把所有快慢的摆动贡献加起来，得到这一段总体有多“用力”。

源码：`features/engineering.py:183–190`。使用 `trapezoid(power,frequencies)`，即按频率间距对谱曲线下面积求和。单位为源单位²。它不是 PSD 数组直接 `sum`。

### 4.20.2 归一化谱熵

直觉：如果贡献几乎都堆在一个快慢上，答案很集中；如果很多快慢都差不多，答案很分散。

源码：`features/engineering.py:106–114,191`。先 `p_k=P_k/sum(P)`，只对正 `p_k` 计算 `H=−sum(p_k log(p_k))/log(number_of_bins)`。总能量不正或频点少于 2 时 NaN。这一公式按功率数组归一化，不是先梯形积分每个频点的面积。

### 4.20.3 主频

直觉：找哪一种快慢的摆动贡献最高。

源码：`features/engineering.py:185–186,192`。输出 `f[argmax(P)]`；要求谱有限且 `sum(P)>0`，单位 Hz；没有额外插值提高峰值频率分辨率。

### 4.20.4 频谱重心

直觉：把不同快慢的位置当作尺子，贡献大小当作砝码，求这把尺子的平衡点。

源码：`features/engineering.py:187–188,193`。`centroid=sum(f*P)/(sum(P)+float64_machine_epsilon)`，单位 Hz。与峰最高在哪里不同，它会受所有频点共同影响。

### 4.20.5 各频带功率

直觉：把“很慢、较慢、较快、很快”的贡献分袋统计。

源码：`features/engineering.py:35–36,117–121,195`。PPG 频带 `[0.2,0.5]`、`[0.5,3]`、`[3,8]`；IMU 三种长度通道频带 `[0.1,0.5]`、`[0.5,3]`、`[3,8]`、`[8,20]` Hz。每袋都含两侧边界，在袋内至少两个频点才做梯形积分；单位源单位²。六个轴通道不计算这些频带或频谱汇总。

## 4.21 文件特征向量：282 字段如何组成

直觉：一条记录可能有几十张窗口卡片，也有按山顶算出的卡片。先为每种卡片预留固定位置，再填有把握的数字；没有的地方保留空格，不能冒充“测得恰好为零”。

源码：`features/registry.py:43–78,128–151,278–405,407–434,436–504`；`representations/feature_vector.py:8–34`。

| 可独立选择的完整组 | 字段数 | 内容 |
|---|---:|---|
| `ppi_basic_rate` | 11 | 有效间期数/时长、6 个 PPI 描述、3 个 HR 描述 |
| `hrv_time_domain` | 5 | SDNN、RMSSD、SDSD、NN50、pNN50 |
| `hrv_spectral` | 6 | VLF/LF/HF、LF/HF、两个归一化功率 |
| `hrv_nonlinear` | 4 | SD1、SD2、两者比例、样本熵 |
| `morphology` | 14 | 7 形态量各 median/MAD |
| `dual_optical` | 12 | 9 AC/DC/比值字段 + 3 波形一致性 |
| `engineering_summary` | 230 | 115 窗口字段各跨窗 mean/population_sd |
| 全部 | 282 | 文件级 `[282]` 值和 `[282]` validity |

`summarize_engineering` 对每一列所有有效有限窗口值 `v_w` 取 `mean(v)` 和 `std_population(v)`。一个窗口也可形成有效摘要，其 SD 为 0；没有值则 NaN/false。

`features.enabled_groups` 默认七组全开。输入可以乱序和有别名，但最终会按固定注册顺序排列；代码选择的是完整组，不是任意乱序的单个字段。旧 `PPI`、`HRV`、`morphology_ppi_hrv` 等只是转换成这些组的兼容映射。

注册表排除 `prv.coverage`、SQI、运动概率、路由等技术/质量元信息。`build_feature_vector` 还按 route 和 validity 决定能否填入某位置。`experiment._extract_vector:1329–1480` 中 acceptable 情况只提供 pulse/PRV，非脉搏字段缺失；mixed routing 下 optical 只有全记录 cell 均 excellent/direct 才计算。缺失信息必须与真实测量值分开。

## 4.22 特征缩放与填补：相似名称，不同公式

直觉：把不同卡片上的数字放到容易比较的尺子上，首先用训练参与者确定“常见中心”和“一格多宽”，以后遇到新参与者直接沿用这把尺。不能先偷看新参与者，再决定尺子的大小。

### 4.22.1 115 列工程序列的变换

源码：`features/engineering.py:359–414` 的 `fit_fold_feature_transform`、`transform_engineering`。

仅将声明的 outer-train 提取结果逐行堆叠；每列选择有效有限值，`center=median`、`scale=max(IQR,1e−8)`。该列训练数据全缺失则保持 center=0、scale=1。输出 `(value−center)/scale`；无效值仍然 NaN。此函数依赖调用方确实传入训练提取结果，不从 `extractions` 对象本身再根据 participant ID 过滤每行。

### 4.22.2 供 fusion 使用的文件向量变换

源码：`features/vector_transform.py:133–241` 的 `fit_fold_feature_vector_transform`、`transform_feature_vector`。

输入记录向量及 participant ID，先实际筛出拟合 roster 的记录。每列 `center=median`、`scale=IQR`；若 `IQR<=1e−12`，改用 `1.4826*MAD`；仍不大于阈值则 scale=1。全缺失列 center=0、scale=1，另记录有效样本数。有效项输出 `(x−center)/scale`，无效项仍 NaN/false。

### 4.22.3 向量供 fusion 使用时的数值化

源码：`features/vector_transform.py:243–272` 的 `transform_feature_vector_batch`。

先完成上一节的训练折变换，再把无效值置为中性零，并把每列的 validity 转为 0/1 接在后半部：`[z_1,…,z_D,m_1,…,m_D]`，全部七组时宽度为 564。这里明确把缺失标记作为 fusion 的附加输入；后文 146 矩阵却不把 validity 拼成预测通道。两种表征不能混为一谈。

### 4.22.4 传统 feature-vector 分类器的填补与缩放

直觉：有的总卡没有测到某项，就先用训练卡片里该项的常见值填上。需要统一尺子的分类方式再移中心、调大小；不需要这一步的分类方式直接使用填好的卡片。测试卡片不能参与选择填充值。

源码：`models/feature_baselines.py:134–178,188–209` 的 `FeatureVectorBaseline._make_pipeline`、`fit`。这是表征到传统分类器的实际输入边界，**不是统一套用 4.22.2 的 IQR 变换**。

`logistic_regression` 和 `rbf_svm` 使用 `SimpleImputer(strategy="median",keep_empty_features=True)` 后接 `StandardScaler()`；训练列中位数填补 NaN，再用填补后训练列的均值和标准差做 `(x−μ_train)/σ_train`。全缺失列仍保留宽度，具体填补退化行为由 scikit-learn 的该实现处理。`extra_trees` 只有同样的中位数填补，不接 `StandardScaler`。所有步骤在传入的训练行上一起拟合，OOF 只 transform/predict；本函数只接收调用方提供的训练数据，训练/测试划分在上游完成。

## 4.23 146 通道窗口矩阵：按时间排列的细粒度描述

直觉：不再把一条记录压成一张总卡，而是把每个时间窗的小卡按发生顺序排成一条长带。这样既保留各窗内容，也保留先后变化；长记录自然有更多格子。

源码：`features/window_matrix.py:33–76,282–383,386–486`，核心 `extract_window_features`。

每行包含 115 个工程字段、14 个局部形态字段、17 个局部间期字段，总计 146。工程窗来自同一完整窗口计划。不是把 282 个文件向量反复复制到每个窗口；也没有把 file context 拼进去。

### 4.23.1 事件归入窗口

PPI 按两端峰的**时间中点**落入 `[window_start,window_stop)` 分配；两端还必须位于同一合法 routing cell。相邻差值除两个原间期都在当前选择集，还要求两者同 detection run、共用端点且 adjacency 真，并检查整个跨三峰范围不越过路由边界。形态则按中心峰时间归窗，并要求完整左右中点边界都在同一 excellent/direct cell。

### 4.23.2 九个局部间期字段

至少 4 个有效间期后计算 PPI 的 mean、median、population SD、IQR、MAD、CV；以及 `60/PPI` 的 mean、median、population SD。单位和 4.9 一致，但 SD 使用 `ddof=0`。这些是短窗描述，不冒称满足长时 PRV 资格。

### 4.23.3 六个局部相邻变化字段

至少 3 个合法差值 `d` 后计算 mean(d)、median(d)、population SD(d)、MAD(d)、mean(abs(d))、median(abs(d))；全部单位秒。没有把 rejected PPI 删除后重新相邻配对。

### 4.23.4 两个局部 PRV 描述

同一组至少 3 个差值上计算 `mean(d²)` 和 `mean(abs(d)>0.050)`。前者单位秒²，**没有开根号，因而不是 RMSSD**；后者是比例。这两个名字分别为 `local_prv.mean_squared_delta_ppi_s2` 和 `local_prv.pnn50_fraction`。

### 4.23.5 十四个局部形态字段和行级路由

每种形态量在该窗口满足条件的至少 3 个搏动上取 median/MAD。`extract_window_features:452–471` 的行级逻辑：

- excellent：可填 115 工程 + 14 形态，并从 direct pulse 提取 17 rate 字段。
- acceptable：仅填 17 rate 字段，可以使用符合其 cell 的 direct/processed pulse；前 129 项维持缺失，**不是只屏蔽 PPG 还保留 IMU 工程项**。
- 不合格行：行标记 false，所有值不可用，但原时间位置仍保留。

## 4.24 146 矩阵训练折缩放和存储

直觉：每种卡片内容用训练数据选一把尺，空格放到“普通中心”的位置；整条不合格的小卡则用另一个标记告诉模型不要把它算进去。卡片数量随记录长短变化，不在存储时强行裁成同样长。

源码：`features/window_matrix.py:489–606` 的 `fit_fold_window_feature_transform`、`transform_window_features`、`build_ordered_window_matrix`；`representations/feature_matrix.py:9–43`。

仅在训练提取结果的有效行且有效字段上拟合 `center=median`、`scale=IQR/1.349`。若 scale 非有限或 `<=1e−8`，回退 `std_population`；仍不合格则 1。与 4.22 的工程序列及文件向量公式均不同。

变换 `(x−center)/scale` 后无效字段与无效行设 0；变换中的 `WindowFeatureExtraction` 仍携带逐值 validity（556行），不添加为预测通道。构建最终矩阵时，把 `[K_i,146]` 转置为 `[146,K_i]`，另带 `[K_i]` row mask；588–591行只将逐值 validity 的压位摘要保存到 provenance，最终矩阵对象不携带完整逐字段 mask。要求至少一个可用行。recording 存储不裁剪、不 padding；批次组合时才可能补齐。文件长度 `K_i` 可变，旧 `matrix_k` 固定上限不再是当前参数。

## 4.25 Raw 八通道表征：波形副本而非手工特征

直觉：不先写摘要卡片，直接把每一小段的八条曲线整齐叠放给模型。每个小段把自己的中心移到零附近，并按自己的大小缩放；原始记录仍原样留着，方便其它模块测实际高度和身体运动大小。

源码：`representations/raw.py:59–112,115–244` 的 `_normalize_dl_window`、`build_raw_windows`；`normalization.py:15–25,74–89`。

输入是 amplitude-preserving PPG `views.x_analysis` 与 physical `dynamic_acc_mps2`/`gyro_rads`，先列拼成 `[N,8]`，顺序为 RED、IR、A_dyn_x/y/z、GX/GY/GZ。非恒等 `x_ar` 只供 rate 使用，不自动替换 raw 输入。每个计划窗口计算完成后转置并转换 float32，输出 `[W,8,T_400]` 与 `[W,T_400]` mask。

### 4.25.1 不缩放：`raw_ppg: none`

直接复制八通道，跳过居中、缩放和 clip；这不是只让 PPG 不缩放。数据仍经过上游缺失修复、滤波及 IMU 处理。

### 4.25.2 普通逐窗缩放：`per_window_standard_zscore`

各通道 `center=mean(x)`、`scale=std(x,ddof=standard_ddof)`，默认 ddof=0；尺度非有限或不大于 `scale_epsilon=1e−8` 时改为 1；输出 `(x−center)/scale`。

### 4.25.3 稳健逐窗缩放：`per_window_robust`

各通道 `center=median(x)`、`scale=(Q75−Q25)/robust_iqr_divisor`，默认除数 1.349。尺度太小/非有限时由 `iqr_fallback` 选择：

- `standard_deviation_then_finite_one`：普通标准差，默认 ddof=0；仍不可用则 1。
- `median_absolute_deviation_then_finite_one`：`MAD/mad_consistency_divisor`，默认除数 0.6744897501960817；仍不可用则 1。
- `finite_one`：直接 1。

上述两个缩放策略最终将不有限输出置 0，并可用 `clip_after_scale` 截断，默认 `[-8,8]`，`null` 关闭截断。原窗口含非法值或 IMU 初始化无估计样本时在进入这里前已整窗丢弃，不是依赖这个兜底填零继续训练。

### 4.25.4 完整窗口、padding 与采样率

默认只完整窗口；若显式允许短记录/尾部 padding，只在真实前缀上统计中心和尺度，补齐后缀为 0、mask=false，不把填充值参与统计。无合法窗口则失败。

`signal.normalization.raw_ppg` 是历史兼容名称，当前实现对**全部八通道**逐窗缩放；`raw_imu:none` 只禁止后续六轴折变换，不能解读成 IMU 从未缩放。

`experiment.py:1950–1994` 在 raw/fusion 数据集已构造后调用模型输入重采样。finalcase 的 5 秒窗口先有 2000 点，归一化后转换到 64 Hz 模型网格约 320 点。MSPTD 的内部 20 Hz、特征的 400 Hz 与模型的 64 Hz 是三件不同的事。

## 4.26 Raw/Fusion 的可选折内六轴后变换

直觉：除了每张小图自己缩放，也可以再用训练参与者共同选出的尺子统一六条身体运动曲线。它是第二次变换，不会把先前已经缩掉的真实大小找回来。

源码：`representations/imu_transform.py:151–240,243–290` 的 `fit_fold_imu_channel_transform`、`apply_fold_imu_channel_transform`。

输入 `[W,8,T]`、逐窗 participant ID 与有效采样点 mask；只选拟合 roster 的窗口真实样本，针对通道 2–7 分别计算：

- `raw_imu:none`：center=0、scale=1；最终不改变数值。
- `raw_imu:outer_train_mean_std`：训练样本均值与配置 ddof 标准差。
- `raw_imu:outer_train_robust`：训练样本 median 与 `IQR/1.349`；退化按上一节三个 fallback 选择，再兜底 1。

应用 `(x−center)/scale`，RED/IR 两列不动，padding 点仍 0，转回 float32。若 raw_ppg 已开逐窗缩放，这里拟合的是**已经逐窗缩放的 IMU 副本**，不是直接对上游 SI 数组拟合；源码的一些历史说明不如实际调用顺序精确。上表的 none 是底层变换函数能力；普通 pipeline 在 `experiment.py:1726–1732` 对 none 直接跳过拟合和应用。finalcase 默认 none，因此不增加第二次缩放。

## 4.27 Motion 检测器表征：八通道参考方案

直觉：为了看出动作有多明显，光线的起伏可以每小段自己调到相近大小，但身体运动曲线先保留大小关系，等训练参与者共同选好尺子后再统一换算。这与上一节分类器的八条曲线全部逐窗缩放不是同一做法。

源码：`representations/motion.py:19–46,50–91,135–196,259–368`。

`motion_8ch_axes_reference_v2` 使用 RED、IR 加 6 个 IMU 轴。窗口固定 8 秒/2 秒、400 Hz，即 `[W,8,3200]`，不补短记录，尾部不完整窗口不进入。窗口起点为 `0,800,1600,…`。

1. RED/IR 在每窗用 `(x−median)/IQR`，IQR 退化时用 `1.4826*MAD`，再退化用 1；阈值为 `1e−12`。注意这里**不除 1.349，也没有 [-8,8] clip**。
2. 六个 IMU 轴直接以 SI 单位拷入 float32 窗口，不做逐窗幅度缩放。
3. 在 outer-training participant 的所有窗口样本上逐轴拟合 median 和 `IQR/1.349`；尺度不大于 `1e−12` 时回退总体标准差，再回退 1。
4. 应用时每次只拿一个轴到 float64 工作数组，做减中心/除尺度，再转回 float32，PPG 保持已缩放值。逐轴处理减少峰值内存，不改变这一数值运算顺序。

这些 tensor 供运动模型，不直接取代 frailty raw 八通道。外部模型训练/推理和阈值使用见质量与运动章节。finalcase 关闭 motion detector，不执行该分支。

## 4.28 Motion 检测器表征：十一通道衍生量消融

直觉：在六条方向曲线之外，再加三条“总共动了多大、转了多大、变化有多急”的曲线。它们是额外信息，不是假装原本只有六个轴却悄悄增加了输入。

源码与上一节相同，配置 profile 为 `motion_11ch_derived_augmentation_ablation_v2`。输入 `[W,11,3200]`，追加 `|dynamic_acc|`、`|gyro|`、`|jerk|` 三通道；分别为三轴平方和开根号，单位 m/s²、rad/s、m/s³，具体物理信号形成见预处理章节。

PPG 仍仅逐窗 median/IQR→MAD→1；后九条 IMU/衍生曲线全部使用训练参与者的折内 median/(IQR/1.349)→population SD→1。它与八通道版使用不同 schema 和 scaler；不能把六轴 scaler 直接拿来变换九条曲线。新增三条曲线只属于该 motion 消融输入，不因此成为 frailty raw 模型的第 9–11 通道。

## 4.29 Fusion：每个文件只结合一次两条信息

直觉：一条记录有许多小图，也有一张总卡。先把所有小图读成一份总体印象，再与总卡结合一次；不能把同一张总卡复制给每幅小图，让长记录无端重复提交很多次相同信息。

源码：`representations/fusion.py:5–14` 的 `masked_file_mean`；`models/fusion.py:21–117` 的 `FileBagFusionClassifier`；`experiment.py:1938–1947` 的数据集组装。

输入是 `[batch_files,W_i,8,T]` 窗口袋、窗口 mask、可选采样点 mask，以及每文件一次的 `[batch_files,2D]` 特征（4.22.3）。只有真实窗口进入波形编码器，padding 袋位置不参与编码。令各有效窗口的表示为 `e_w`。

### 4.29.1 平均池化

`masked_file_mean` 的 NumPy 版本直接 `mean(embeddings[window_mask],axis=0)`；模型内 `pooling: mean` 用 `sum(m_w*e_w)/sum(m_w)`。每文件至少一个有效窗口。

### 4.29.2 注意力池化

`pooling: attention` 先计算每窗 `a_w=uᵀe_w+b`，无效窗设 `−∞`，权重 `α_w=exp(a_w)/sum_valid exp(a)`；总体表示为 `sum(α_w*e_w)`。`u,b` 是训练学习的参数，不是 SQI 分数，也不是额外的确定性预处理。

### 4.29.3 文件向量编码与拼接

第 47 行 `h=ReLU(W_f*v+b_f)`，默认 `feature_hidden_dim=32`；第 105–106 行把 `pooled_signal` 与 `h` 沿特征轴拼接，再 `ReLU(W_c*[pooled_signal,h]+b_c)`、训练时 dropout（默认 0.2），融合宽度默认 64。最后线性分类头得到每文件类别分数。

信号编码器可通过 `FileBagFusionCompact`、`FileBagFusionInception` 或可组合的 `FileBagFusion` 选择；网络架构本身不在本章展开。数学上四种表征不是四个数据格式别名：raw 学窗口波形，feature_vector 用文件统计，feature_matrix 学按时序排列的窗口统计，fusion 合并文件级波形表示与文件统计。

## 4.30 输入输出、拟合范围与缺失语义对照

| 模块 | 主要输出 | 是否从训练群体拟合 | 缺失/短数据处理 |
|---|---|---|---|
| 四种峰检测 | 原网格峰 + PPI + validity/adjacency | 否，记录内算法 | 最短时长/峰不足失败；有效段不跨缺口 |
| PRV | 字段字典 + validity | 否，记录内统计 | 每类资格独立，不够则 NaN |
| 形态/光路 | 逐搏数组 + 文件统计 | 否 | route 限定、至少 3 有效搏动；逐字段缺失 |
| 115 工程序列 | `[W,115]` | 提取否，单独 scaler 是 | 完整窗；有限值不足 80% 则通道字段缺失 |
| 文件向量 | `[D]` + validity | 提取否，折变换是 | 未测量为 NaN/false |
| Fusion 向量 | `[2D]` | 使用训练折向量变换 | 中性 0 + 单独 0/1 通道 |
| 窗口矩阵 | `[146,K_i]` + row mask | 146 列变换是 | 不合格项为变换后 0，validity 不入预测通道 |
| Frailty raw | `[W,8,T]` + sample mask | 逐窗否；可选 IMU 后变换是 | 非法真实点整窗丢弃；显式 padding 为 0/false |
| Motion 参考/消融 | `[W,8/11,3200]` | PPG 逐窗否；IMU scaler 是 | 必須完整八秒，无短窗 padding |
| Fusion 池化/编码 | 每文件一条表示 | attention/网络是 | 只汇聚有效窗口 |

所有训练群体拟合产物都必须随模型保存，并在 OOF/新参与者上只应用、不重新拟合。记录内运算“不用标签”并不意味着所有下游输出都可跨 fold 缓存：一旦依赖 fold scaler、质量校准或路由，就属于该 fold 的状态。

## 4.31 覆盖清单和验证边界

本章逐步对应以下源码的数值主体：

- `signal/peaks.py`：历史 prominence，方向评分、间期有效性。
- `peaks/`：resolver、MSPTDfast、Aboy v1/v2、双波长配对和事件匹配；`__init__.py` 只做导出，不增加算法。
- `signal/prv.py`：基础 rate、时间 PRV、Poincaré、样本熵、频谱 PRV 与资格。
- `signal/morphology.py`、`signal/optical.py`：逐搏外形、AC/DC/PI、双光路一致性。
- `features/engineering.py`：115 特征、Welch、折内工程变换。
- `features/registry.py`：七组 282 文件向量、摘要、route/missing 编码。
- `features/vector_transform.py`：文件向量折变换与 fusion 2D 编码。
- `features/window_matrix.py`：合法时间段组装、146 窗口特征、折变换和可变长度矩阵。
- `features/prv_backend_compare.py`：三个固定-PPI 比较适配器及本地方程。
- `representations/`：raw、六轴后变换、motion 两种输入、feature-vector/matrix 结构、fusion 池化。
- 必要调用边界：`normalization.py`、`experiment.py` 表征分支与 DL 重采样顺序、`models/feature_baselines.py` 输入填补/缩放、`models/fusion.py` 文件池化/融合。

以下边界不应误读成已经完成的验证：

1. 本章是当前仓库源码伴读，不是对所有可选配置重新训练后的数值回归报告；没有通过写文档重新运行训练。
2. SciPy 的滤波、Welch、峰 prominence、统计量等调用明确到函数与参数，但未将整个第三方库实现逐行复制进文档；精确库版本会影响其数值细节。
3. Aura/rhenan 内部公式不在适配器源码中，本章只核对输入、调用、输出单位与错误处理，未验证其所有内部算法与本地 PRV 等价。
4. shapeformer/inception/传统分类器训练、损失与优化器属于模型章节范围，不把它们冒充本章已展开的特征算法。
5. 当前 detector 参数面没有独立 PPI cleaner；矩阵缺失标记不作预测通道，而 fusion 缺失标记会作输入；Aboy v2 会实际删峰；这些都是实现语义，不应被教程中的泛化描述抹平。
