# Motion Detector、Denoising 与动态 HR 候选脚本总表

## 1. 文档目的与判定规则

本文件是用户指定的新候选路线文档。它登记三个问题中所有仍有工程或 benchmark 价值的脚本，并明确区分：

- **主候选**：值得在统一协议下优先实现/复测；
- **对照候选**：方法透明或历史结果有解释价值，但不应直接作为最终方案；
- **工程组件**：loader、split、scorecard、ONNX/runtime 等可复用，不等于算法有效；
- **失败历史**：必须保留以防重复试验，但不列为成功候选；
- **新路线**：数据与接口可实现，但仓库当前没有完整脚本或结果。

所有路径均相对 workspace 根。没有实际产物时明确写 `not_produced` 或 `proposed`，不按代码预期补写结果。

## 2. 总优先级

| 问题 | 首选路线 | 次选/消融 | 不恢复为主线 |
|---|---|---|---|
| Motion detector | `ppg_peak_hr_gating_train.py` 的 10-channel PPG+IMU Light CNN A/B 框架 | v7.4 fused detector；修复后的 fix9/Stage1；透明 IMU rules | 把 activity 标签直接称 artifact truth；未修复 legacy 近满分结果 |
| Denoising | 先补 hybrid 的严格生理 benchmark；同时优先新建“谱证据抑制但不强制重建波形”路线 | DWT-A2、IMU-NLMS、v7.4 MaskNet 仅作基线；BSS 作前端增强 | v7/v7.2/v8/Stage-2 full-waveform success 叙事；CEEMD self-reference clean route |
| 动态 HR | 新建 STFT/IMU 污染概率→候选峰→Viterbi/Kalman→SQI/拒绝 | 重构后的 PPG peak/IBI 多任务框架；Aboy++ 强基线 | 当前未校正 ECG timing target；强制每窗输出 HR |

---

# A. Motion detector 候选

## A1. 主候选：10-channel PPG+IMU detector A/B

### 脚本与应用位置

- 脚本：`ppg_peak_hr_gating_train.py`
- 入口族：`load_motion_detector_records`、`make_detector_feature_matrix`、`run_motion_detector_benchmark`
- 模型：`DenoiserEncoderMotionDetector` 与 `LightCnnMotionDetector`
- 注意：这是同脚本内独立的 motion benchmark，不是 PPG-only peak/gate 主网络。

### 具体算法

1. PPG 加载、带通/标准化。
2. 加速度单位推断并转为 `m/s²`，陀螺仪转为 `rad/s`；估计/扣除重力。
3. 构造 10 通道：PPG、dynamic ACC xyz、GYRO xyz、ACC magnitude、GYRO magnitude、jerk magnitude。
4. 按 subject 划分 train/validation/holdout；阈值只在 validation 按 balanced accuracy 选择。
5. 模型 A 使用 denoiser-style encoder classifier；模型 B 是直接训练的轻量 1D CNN。
6. 输出 pooled window 概率、阈值后标签、ROC/PR、混淆矩阵、PT/ONNX 和比较摘要。

### 输入数据名称与路径

| 数据 | 路径/解析 | 主要字段 |
|---|---|---|
| PTT-PPG | `physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv` | `pleth_*`, `a_x/y/z`, `g_x/y/z`; sit/walk/run 来自文件名 |
| SIM external | `physionet.org/files/simultaneous-measurements/1.0.0/generated_data/` | `SOT/Pleth`、FAROS accelerometer；缺失 gyro 补 0 |

### 输出路径与结构

- 根：`.CNN_results/<timestamped_run>/`
- 关键：`detector_benchmark_summary.json`、两模型 PT/ONNX/meta/export status、训练曲线、ROC/PR/CM、A/B comparison 图。
- M0 输出清单：`.CNN_results` 共 687 个多 run 文件；引用结果时必须固定具体 run，不跨 run 平均。

### 已有结果与状态

- PTT validation/holdout：两模型均为 `1.0`，高度提示 activity/domain 可分而非 artifact truth 完成。
- SIM external：
  - 模型 A：F1 `.7542`、BA `.7699`、AUC `.8269`；
  - Light CNN B：F1 `.7634`、BA `.7802`、AUC `.8642`。
- 状态：`primary_candidate_but_not_strictly_validated`。

### 下一步

1. 把窗口随机/pooled 结果改为 subject/record 聚合与 bootstrap 95% CI。
2. 新增人工标注或 ECG/PPG 可解释的窗口 artifact 标签，不能只用 sit/walk/run。
3. 固定重叠窗口去重、阈值拟合、校准和外部设备协议。
4. 与 IMU-only、PPG-only、规则 detector、SQI-only 做同 split 消融。
5. 输出 `motion_probability`、`artifact_probability`、`uncertain` 分离语义；不把活动强度当作光学质量。

## A2. 候选：v7.4 rule/AE/fused detector

### 脚本与算法

- `pttppg_pipeline_v7_4_noleak_viz_ae.py`
- 输入 10 个 PPG 特征 + 27 个 IMU 特征；单特征阈值、OR/AND、PPG CNN-BiLSTM AE anomaly、fused rule。
- lag 在 train subjects 中以同序号 PPG/IMU 特征绝对 Spearman 最大值选择；阈值和 AE threshold 均来自 train subjects。

### 输入/输出

- 输入：`physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv`；主要 `pleth_1/pleth_2` + `a_x/y/z,g_x/y/z`。
- 实际输出：`results_v7_4/detector/` 下 `cv_summary.json`、`detector_artifact.json`、rule NPZ、AE PT、fold/holdout CM、lag 图。
- 代码默认仍是 `results_v7_3`，所以运行时必须显式记录 `--out_dir`。

### 结果/判定

- holdout OR BA `.500`；AND `.573`；AE `.649`；fused AND `.670`。
- `use_dwt=true` 的 AE 没有 DWT-off 配对实验。
- 状态：`secondary_activity_proxy_candidate`。
- 价值：subject holdout 边界和可解释 feature/AE 融合可复用。
- 风险：跨记录 shift、同序号特征语义错配、activity 替代 artifact 真值。

## A3. 对照候选：Stage1 双 Logistic + OR/AND

- 脚本：`Arc/pttppg_stage1_detector.py`。
- 算法：PPG/IMU 手工特征各自 Logistic，做 OR/AND 与 lag 融合。
- 输入：PTT sit/walk/run CSV。
- 输出：`results_stage1/` 的 summary/模型/图。
- 历史结果：AND holdout BA≈`.8623`，OR≈`.6920`。
- 状态：`interpretable_historical_baseline`。
- 复用条件：按记录独立 lag、train-only transform、严格 subject holdout、人工 artifact 标签。

## A4. 对照候选：legacy v8 handcrafted score detector

- 脚本：`pttppg_detector_v8_scores_audit_fix9.py`。
- 算法：10 PPG + 27 IMU 特征、sit-clean AE anchor、Mahalanobis score、global lag、Logistic fusion。
- 输入：PTT `s*_sit/walk/run.csv`。
- 输出：`results_v8_audit/{1_0.5,2_0.5,6_1}/` 的 summary/audit/ROC/PR/CM，以及根部按窗口命名的 bundle NPZ。
- 2 s/0.5 s：fused BA `.9880`、F1 `.9881`、AUC `.9995`；IMU-only BA `.9992`；PPG-only BA `.7203`。
- 状态：`biased_baseline_only`。
- 判定：近满分主要是活动/IMU区分；CV transform、lag 和跨记录处理需重写后才可作为 benchmark。

## A5. 透明候选：IMU EKF/gravity/rule detector

- 脚本：`funcs.py`、`ppg.py`。
- 算法：静止 bias、姿态 EKF、重力扣除、ACC/GYRO/jerk 阈值、窗口标签与 padding。
- 输入：Dash 选择 CSV 的 `AX,AY,AZ,GX,GY,GZ` 或调用方映射列；采样率常见 400 Hz。
- 输出：内存 `a_dyn, acc_mag, gyro_mag, jerk_mag, labels, ids, frac` 和 Dash 图；无稳定结果目录。
- 状态：`transparent_baseline_needs_contract_fix`。
- 价值：低成本、易解释、可作为 SQI motion 分量；不能单独声称 artifact detector。

---

# B. Denoising 候选

## B1. 当前工程主候选：Hybrid pseudo-supervised suite

### 脚本

- `pttppg_denoiser_hybrid_core.py`
- `pttppg_denoiser_hybrid_train.py`
- `pttppg_denoiser_hybrid_preview.py`
- `pttppg_denoiser_hybrid_ab_compare.py`
- `pttppg_denoiser_hybrid_export_onnx.py`
- `pttppg_denoiser_onnx_runtime.py`
- `ppg_denoiser_dash_utils.py`

### 具体算法

1. PPG 0.5–8 Hz；动态 ACC/GYRO/magnitude/jerk。
2. 81 维 lag bank、ridge `alpha=8` 形成 linear artifact/clean baseline。
3. sit 片段按 IBI bin 构造 beat template；ECG/PPG peak 与跨域 delay 形成代理约束。
4. 1D U-Net 预测 residual artifact；`clean_hat = raw_norm - artifact_hat`。
5. loss 含 artifact、clean、sit template、peak、decorrelation、slope、anchor。
6. Python 完成 preprocessing/ridge/normalization/OLA，ONNX 仅导出 `model_input → artifact_hat`。

### 输入

- `physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv`
- `pleth_1/pleth_2`、IMU、ECG/peaks（代理/评价）
- 500 Hz、6 s window、1 s hop。

### 输出

| 变体 | 路径 | 结构 |
|---|---|---|
| raw+IMU | `results_hybrid_denoiser_raw_imu/` | PT、meta、history、splits、delay、ONNX contract |
| raw+IMU+baseline | `results_hybrid_denoiser_raw_imu_baseline/` | 同结构；15 通道 |
| 历史/默认 | `results_hybrid_denoiser/` | 旧 schema/变体，不与上两者混合 |
| preview | `denoiser_preview_output/` | 8 PNG，无数值 scorecard |

### 结果/状态

- split：15 train / 3 val / 4 holdout；holdout 只写入 split，未正式评分。
- raw+IMU best val `.54578`；baseline variant `.45273`。
- artifact L1 与 clean L1 数学等价；只是同一代理目标重复权重。
- 状态：`engineering_primary_candidate_proxy_only`。

### 下一步

- 先不改模型，补 raw/bandpass/linear/hybrid 的 subject-heldout 与 external scorecard。
- 指标：peak F1、PPI/HR MAE、coverage、sit identity distortion、PPG–IMU residual coherence、运行时间。
- 修复 whole-record ridge transductive、OLA 边界 0、IMU 单位、ONNX 端到端 parity。
- 若无法在生理终点上稳定优于 raw/high-quality-only，则退出主线，只保留工程资产。

## B2. 工程比较线：v7.4 STFT magnitude MaskNet

- 脚本：`pttppg_pipeline_v7_4_noleak_viz_ae.py`。
- 算法：`pleth_2/pleth_1` STFT magnitude + 37 个广播标量特征组成 39 通道，2D network 输出 mask，保留 noisy `pleth_2` phase 做 iSTFT。
- 输入：PTT 双 PPG + IMU；ECG/peaks仅代理约束。
- 输出：`results_v7_4/denoiser/{walk,run}` 的 PT/ONNX/`.onnx.data`/meta/subject-a。
- 结果：walk/run inner-val loss `.987795/.702436`；`a` 全为 `1.0`；无 holdout HR/PPI。
- 状态：`stft_engineering_baseline`。
- 复用：STFT/ISTFT、mask contract、ONNX 外置 preprocessing；不复用现有有效性结论。

## B3. 风险基线：多参考 IMU-NLMS ANC

- 脚本：`funcs.py:959-1066`、`ppg.py:1468-1575`；Dash 调用 `ppg.py:2871-2876`。
- 算法：6 维 `a_dyn xyz, |a_dyn|, |gyro|, |jerk|` 全记录 z-score；运动门内执行 memoryless NLMS，默认 `mu=.02`；输出 residual 后带通、找峰。
- 输入：`.env`/Dash 选取的 PPG/IMU CSV，要求 `AX..GZ`；历史 Notebook 实例为外部 D 盘路径，不作为可移植默认。
- 输出：内存波形、权重、峰、HR/HRV和 Dash 图；无稳定磁盘 scorecard。
- 结果：Notebook 只留下假峰、延迟、删心搏、不稳定和放弃时域 ANC 的记录；无数值 holdout。
- 状态：`negative_control_known_signal_loss_risk`。
- 后续仅实现 Wiener/LMS/NLMS/RLS 统一对照和“真实运动 HR 保留”安全测试，不作为首选最终路线。

## B4. 非平稳分解对照：DWT-A2 与 EMD/CEEMD-lite

### DWT-A2

- 脚本：`pttppg_pipeline_v7.py`、`cnnppg_v7.py`、v7.4 AE。
- 算法：`pywt.wavedec(db4, level=2)`，仅保留近似系数并线性插值回原长；不是 threshold denoising。
- 输出随 v7/v7.2/v7.4 detector 目录；没有 DWT-on/off 配对。
- 状态：`cheap_decomposition_baseline_only`。

### EMD/CEEMD-lite

- 脚本：`funcs.py:1162-1387`、`ppg.py:1671-1891`。
- 算法：互补噪声对手写 EMD；污染 PPG 自身估计 HR/IMF，选 motion IMF 后用 32-tap leaky NLMS。
- 输出：内存 IMF/reference/residual/权重和 Plotly 图；无固定 scorecard。
- 状态：`exploratory_deprecated_for_clean_waveform`。
- 原因：参考来自同一 PPG，不满足 ANC 独立性；污染主峰可能导致保护运动、删除心搏。

## B5. 新主线：谱域抑制只服务 HR 证据

- 当前完整脚本：`not_implemented`。
- 可复用：v7.4 STFT helpers、v7/v7.2 spectral loss/SoftHRFromFFT、现有 IMU preprocessing。
- 输入：PTT `pleth_1/pleth_2 + a/g`；以后可接 `IrFinger/RedFinger`。
- 建议算法：联合 STFT → IMU 谱污染概率/soft mask → top-K HR 候选 → 跨波长/谐波/SQI评分 → Viterbi/Kalman/Particle。
- 建议输出：`final_v0/benchmarks/m0_signal_routes/<run>/spectral_hr/{hr_track.csv,summary.json,diagnostics.npz,plots/}`。
- 状态：`promising_but_not_implemented`。
- 关键原则：mask 只改变候选证据，不宣称恢复 clean waveform；冲突时降低置信度或拒绝。

## B6. 新增强线：双波长 BSS

- 当前完整脚本：`not_implemented`。
- 数据：`PPG_Testing_05_01_2026/tradeali.csv` 等 `IrFinger/RedFinger`；PTT `pleth_1/2` 待语义确认。
- 算法候选：PCA、FastICA、双通道 STFT-NMF。
- 建议输出：`final_v0/benchmarks/m0_signal_routes/<run>/bss/{components.npz,component_scores.csv,summary.json}`。
- 状态：`data_available_implementation_missing`。
- 定位：作为最佳单通道/均值与 spectral tracker 的前端消融，不预设一定改善。

## B7. 失败历史：v8 MaskNet 与 Stage-2

- `pttppg_denoiser_v8_masknet.py` → `results_denoiser_v8/`：0 文件；局部变量覆盖 `F` 后调用 `F.interpolate`，默认 collate 也无法处理变长 peaks。
- `pttppg_stage2_denoiser.py` → `results_stage2/`：0 文件；time mask 沿频率复制、频率平滑恒 0、phase/subject 参数/holdout 选择有问题。
- 状态：`failed_history_not_candidate`；只用于阻止重复走同一路径。

---

# C. 动态 HR 候选

## C1. 首选新路线：spectral candidates + temporal tracking + SQI

- 当前仓库状态：STFT 工具存在，完整路线不存在。
- 输入：双 PPG `[T,2]`、IMU `[T,6+]`、timestamp；ECG peaks 只作训练/评价 reference。
- 推荐处理：100 Hz 工作采样率、8 s Hann window、1 s hop、较高 `n_fft` 或峰插值；40–210 bpm top-K 候选。
- 候选特征：主峰能量、谐波/次谐波、IR/RED 一致性、IMU overlap/coherence、SQI、前帧可达性。
- 轨迹：Viterbi 离线基准；Kalman 因果在线；Particle 处理多峰/快速变化。
- 输出：`time_center,hr_bpm,confidence,sqi,motion_score,selected_rank,is_missing,reason_code`。
- 状态：`priority_new_route`。

## C2. 工程成熟但当前失败：PPG-only peak/IBI/gate 多任务框架

- 脚本：`ppg_peak_hr_gating_train.py`。
- 模型：PPG-only 1D U-Net/encoder-decoder，输出 peak heatmap、beat、dense IBI、window gate；含多数据 loader、CV、external/extra holdout、PT/ONNX 和 scorecard。
- 输入路径：PTT、SIM、iAMwell、MIMIC、可选 VitalDB，均在 `physionet.org/files` 或 API resolver 下。
- 输出：`.CNN_results/<run>/peak_hr_gate_model.pt`、`cv_summary.json`、`holdout_summary.json`、`extra_holdout_summary.json`、`group_scorecards.json`、`deploy_export.json` 等。
- 结果：event F1@20 ms OOF `.3870`、holdout `.3780`、extra `.1540`、SIM `.0948`；40 ms holdout F1 `.5901`。匹配成功事件的 RRI MAE较低不能掩盖低召回。
- 核心错误：未把数据集 ECG→PPG delay 反馈到 target；重叠窗口重复 beat；HR bpm 没有正式计算；dense IBI aggregate 失真。
- 状态：`framework_reuse_model_deprecated`。
- 后续：保留 loader/split/scorecard/export，重做 pulse timing target、record-level event merge、HR track 和拒绝机制。

## C3. 透明强基线：Aboy++/PPI/HRV

- 脚本：`funcs.py`、`ppg.py` 及分析 UI。
- 算法：带通、双极性/多段自适应峰、峰间距、artifact rejection、HR/PPI/HRV。
- 输入：由 `.env` 和 Dash 选择的 RED/IR/IMU CSV；常用 400 Hz。
- 输出：peak indices、PPI/HR/HRV、交互图及部分 HRV CSV；无统一 benchmark 目录。
- 已知问题：`reject_artifacts` 参数错位、窗口首尾漏提交、`HRi` 键错误、RR 数值去重会删稳定心搏。
- 状态：`high_value_baseline_requires_parity_fix`。
- 后续：抽成纯函数、固定输入输出、与 ECG reference 做 20/50/100 ms peak F1、PPI/HR MAE 和 coverage。

## C4. 轻量复用基线：frailty classifier 内 Aboy++

- 脚本：`frailty_3class_classifier.py:399-462`。
- 算法：双极性、高通/自适应带通、分窗找峰、35–210 bpm 与 MAD 清理 PPI。
- 输入：`PPG_Testing_05_01_2026` 下 RED/IR 波形；当前用于特征提取。
- 输出：文件级 PPI/HRV/形态特征，最终写 `results_frailty3/...`；没有动态 HR track scorecard。
- 状态：`lightweight_baseline_candidate`。

## C5. 窗口谱基线：SoftHRFromFFT

- 脚本：`cnnppg_v7.py:416-443`。
- 算法：单个 6 s window 的 rFFT magnitude，在 `0.6–3.5 Hz` 内 soft-argmax，`tau=.1`。
- 用途：v7.2 训练/评价代理，不做跨窗路径。
- 结果：v7.2 holdout ECG-HR MAE `33.44–37.80 bpm`。
- 状态：`negative_window_level_spectral_baseline`。
- 后续：只保留为“raw STFT argmax/softargmax”对照，不作为轨迹方法。

---

# D. 三条路线共享的 SQI/拒绝层

## D1. 当前实现

- 脚本：`frailty_3class_classifier.py:1671-1766`。
- 公式：`0.40*pulse_band_ratio + 0.35*PPI_stability + 0.15*peak_density + 0.10*motion_penalty`。
- 选择：每文件/subject top70 或 top50；聚合可用 mean 或 SQI-weighted mean。
- 输出：窗口 `[0,1]` 质量分、keep mask、聚合使用窗口数。

## D2. 已有结果

- generalization sweep 路径：`results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2/analysis_report_20260703_1202/`。
- top50 mean subject BA `.51079`，none `.49790`；71/112 配对提升，41/112 下降；worst-class F1 `.38967` vs `.38571`。
- 状态：`weak_positive_signal_not_independent_validation`。

## D3. 必须补充

1. skew/kurtosis、自相关、模板相关、归一化谱熵、IBI有效率/异常率、RED/IR一致性、PPG–IMU coherence。
2. motion 分量用原始/校准物理单位，不在逐窗 z-score 后再估运动强度。
3. P5/P95、分量权重与阈值只在 train fold 拟合并序列化。
4. 输出 component table、flags、failure reason 和 version。
5. 任何 detector/denoiser/HR 方法都必须报告 quality–coverage 曲线，允许 `no_estimate`。

# E. 路线选择与未来方向建议

1. 先冻结统一数据/split/metrics，不立即训练新大模型。
2. 首批实现 raw/bandpass/Aboy++/current SQI、IMU-NLMS 风险基线、raw STFT argmax。
3. 第二批实现新的 SQI 与 spectral+Viterbi；这是最高优先级可检验路线。
4. 第三批实现 PCA/FastICA/STFT-NMF 和 DWT/WPT/EMD/VMD/SSA 对照。
5. Hybrid 先补现有 bundle 的 holdout scorecard；只有生理终点通过才继续改模型。
6. 每批完成后按 TODO 步骤向用户报告并等待确认；本文件本身不授权进入实现或 M1。

# F. 用户确认后的串行候选路线

## F1. SQI 与 29-subject Motion

- 29-subject 数据来自 PPG_Testing_05_01_2026/StudyData/*.csv 及其 Youngers 子队列；实测为 29 subjects × 9 roles = 261 个原始文件。当前静态 frailty B/R1–R4 特征缓存为 145 行/29 subjects。
- 原始通道为 RED,IR,AX,AY,AZ,GX,GY,GZ，足以计算用户点名的 SQI 与 motion 分量；但数据没有现成的窗口级 optical-artifact truth。
- B/R/S/W 可构造 activity/motion proxy，却不能无条件改名为 PPG artifact 标签。
- 现有 PTT Motion A/B 是 22 个 PTT subjects 的活动监督实验，不是 29 人 frailty CV；当前阈值由重叠 validation windows 池化搜索，也不是真正的 fold-nested threshold CV。
- SQI 必须在 classifier 逐窗 z-score 之前分支：raw/校准单位给 Motion detector 与 SQI，归一化副本给 classifier。现有 peak_density 还把 400.0 写死，实施时必须改为参数 fs。

## F2. 四条可实现路线

| Route ID | 前端核心 | 共用后端 | 当前状态 |
|---|---|---|---|
| spectral_track_sqi | STFT、IMU 谱抑制/掩蔽、候选谱峰 | Viterbi/Kalman、SQI、HR/PPI 聚合 | priority_new_route_not_implemented |
| dual_ppg_bss_sqi | 双波长 PCA/FastICA/STFT-NMF 与分量选择 | 同一候选、轨迹、SQI 后端 | data_available_not_implemented |
| nonstationary_sqi | DWT/WPT/EMD/EEMD/VMD/SSA 受控对照 | 同一候选、轨迹、SQI 后端 | partial_baselines_only |
| adaptive_sqi | Wiener/LMS/NLMS/RLS、IMU ANC 与安全门 | 同一候选、轨迹、SQI 后端 | negative_control_partial_only |

路线顺序表示实施与消融优先级，不预设后一路必然优于前一路。自适应滤波必须记录真实 motion-HR 被削弱的风险，不得只用残差能量判断成功。

## F3. PTT 监督测试

- 输入：physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv 的双 PPG、IMU、ECG 与 peaks。
- peaks 是人工核验 ECG R-peaks，不是 PPG pulse peaks；可直接生成 ECG HR/RRI reference。
- PPI 对 RRI 是代理评价；绝对 PPG 事件 F1 必须在 train subjects 内拟合 ECG→PPG delay，并在 val/test 冻结。
- 首轮指标固定为 HR MAE/RMSE/±5/±10 bpm、PPI MAE/RMSE、coverage、gross-error、按 subject/activity 分层和 quality–coverage 曲线。

## F4. Frailty 路线特征选择

- 四条路线分别输出显式特征块：motion HR、motion PPI、coverage/SQI 与缺失原因；不得由宽表后缀规则一次选中所有路线。
- 使用同一份 subject-level 5-fold split manifest 和同一 seeds：42,10042,20042,30042,40042。
- 初始 13 个候选为 baseline + 4 routes × {HR-only, PPI-only, HR+PPI}。
- 最高 BA 的选择必须嵌套在 outer fold 内；最终报告同时给 subject-level BA、macro-F1、worst-class recall/F1、coverage 和跨 seed 稳定性。
