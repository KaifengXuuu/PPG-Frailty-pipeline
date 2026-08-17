# 五类路线统一实现与 Benchmark 合同

## 1. 目的

本合同把“可以实现”转成可执行的模块边界、测试先决条件、数据切分、指标、输出 schema 和路线淘汰门。它不是新算法结果，也不授权立即编码；实际实施必须等待用户确认当前 M0 扩展。

核心原则：先统一数据与测试，再比较路线；测试不能晚于模型写完才临时补。所有路线允许拒绝输出，拒绝率/coverage 是主指标之一。

## 2. 建议未来目录（尚未创建）

所有未来新增仍只允许写入 `final_v0/`：

```text
final_v0/
├── m0_signal_routes/
│   ├── contracts.py                 # 公共数据类与失败码 / shared contracts and reason codes
│   ├── adaptive_anc.py              # Wiener/LMS/NLMS/RLS
│   ├── decomposition.py             # DWT/WPT/CWT/EMD/EEMD/VMD/SSA
│   ├── spectral_hr.py               # TFR/mask/candidates/Viterbi/Kalman/Particle
│   ├── bss.py                       # PCA/FastICA/STFT-NMF
│   ├── sqi.py                       # 可解释SQI / interpretable SQI
│   ├── baselines.py                 # raw/bandpass/Aboy++/argmax
│   ├── benchmark.py                 # subject级统一评价 / subject-level benchmark
│   └── tests/
│       ├── fixtures/
│       ├── test_adaptive_anc.py
│       ├── test_decomposition.py
│       ├── test_spectral_hr.py
│       ├── test_bss.py
│       ├── test_sqi.py
│       └── test_no_leakage_and_runtime.py
└── benchmarks/
    └── m0_signal_routes/<run_id>/
```

若实施，每个 Python 模块必须有详细中英文模块说明、函数 docstring、参数/单位/shape 注释和非显然分支行内注释。

## 3. 公共数据合同

### 3.1 记录是最小边界

```text
SignalRecord:
  record_id
  subject_id
  dataset_id
  activity_label             # sit/walk/run/unknown；不等于artifact真值
  timestamps[T]
  ppg[T,C_ppg]
  ppg_channel_names[C_ppg]
  imu[T,C_imu] | None
  imu_channel_names[C_imu]
  ecg[T] | None
  ecg_peak_indices[K] | None
  artifact_labels[...] | None
  sampling_rate
  units
  provenance
```

禁止跨 `record_id` 做卷积、lag、窗口拼接、标准化或轨迹连续性。每个输出必须能回溯到 `dataset_id/subject_id/record_id`。

### 3.2 对齐合同

每记录保存：原始采样率、工作采样率、重采样方法、PPG/IMU timestamp 偏差、估计 lag、缺样/插值比例、ECG→PPG delay 估计及其用途。用于 peak timing 的 delay 与用于 HR interval 的 reference 必须分开。

### 3.3 标签语义

| 标签 | 可以证明 | 不可以证明 |
|---|---|---|
| sit/walk/run | activity/motion state | PPG 每窗是否污染 |
| ECG R peaks | ECG timing/interval/HR reference | 未校正的 PPG pulse timing |
| PPG manual peaks | pulse event accuracy | waveform clean truth |
| 人工质量标签 | 可见质量/可用性 | clean waveform 数值真值 |
| synthetic clean | 算法在生成模型下的 waveform recovery | 真实设备泛化 |

## 4. 数据划分合同

1. 先按 subject 固定 external holdout；同一 subject 不得跨 train/validation/holdout。
2. train subjects 内使用 GroupKFold 或 nested subject CV。
3. 所有标准化、P5/P95、threshold、mask 权重、tracker transition、BSS 分量规则和超参数只在 train/validation 拟合。
4. holdout 一次性运行；不能用 holdout 选 epoch、phase、subject parameter 或 route。
5. 重叠窗口可以用于训练，但评价必须先合并成唯一 record-level event/trajectory；统计单位是 subject，不是窗口。
6. sit/walk/run 与数据集分别报告，再作预先定义的 subject-weighted 汇总。
7. seed、split manifest、代码 hash、输入 manifest、配置和失败记录必须随 run 保存。

## 5. 公共基线矩阵

| 类别 | 必须比较的方法 |
|---|---|
| 原始 | raw IR；raw RED；逐窗 best-SQI channel |
| 透明预处理 | 0.5–8 Hz bandpass；IR/RED平均；归一化加权平均 |
| 质量策略 | none；当前 SQI top50/top70；新 SQI 固定 coverage 50/70/90% |
| 峰/HR | 修复后的 Aboy++；raw Welch/STFT argmax；SoftHRFromFFT |
| 自适应 | ridge-Wiener；LMS；NLMS；RLS；IMU打乱负对照 |
| 分解 | DWT-A2；DWT-soft；WPT；CWT；EMD/EEMD；VMD；SSA |
| 谱追踪 | mask+argmax；candidate-only；Viterbi；Kalman；Particle |
| BSS | PCA；FastICA；STFT-NMF；每种+BSS前后Viterbi |
| 历史 | v7.2、v7.4 MaskNet、hybrid raw/raw+baseline；只按其证据边界比较 |

## 6. 五类模块接口

### 6.1 Adaptive ANC

```text
run_adaptive_anc(record, config, fitted_calibration)
  -> artifact_hat, residual, weights,
     convergence, physiological_retention,
     valid, reason_code
```

配置必须保存 method、FIR taps、mu 或 RLS λ、regularization、lag、gate、标准化统计来源。任何 `physiological_retention` 安全测试不通过时，不能输出名为 `clean_ppg` 的结果，只能标记 `suppressed_representation_invalid`。

### 6.2 Decomposition

```text
decompose_record(record, config)
  -> components, residual, metadata, reconstruction_error

select_and_reconstruct(result, imu_context, sqi_context, fitted_rules)
  -> selected_components, output, valid, reason_code
```

必须支持 identity reconstruction、原长、边界误差、seed 和 convergence。

### 6.3 Spectral HR

```text
compute_joint_tfr(ppg, imu, fs, config) -> spectra
estimate_motion_spectral_mask(spectra, config) -> mask_result
generate_hr_candidates(spectra, mask_result, sqi, config) -> candidate_lattice
track_hr_candidates(candidate_lattice, method, config) -> hr_track
```

每帧必须允许 `missing` 状态；confidence 要与误差/coverage 校准，不得只是模型概率原值。

### 6.4 BSS

```text
separate_dual_ppg(ppg[T,2], fs, config, imu=None)
  -> components, mixing, unmixing,
     component_scores, selected, stability,
     fallback, valid, reason_code
```

两通道近共线、常数、饱和或 seed 不稳定时，必须回退到 best-SQI single channel。

### 6.5 SQI

```text
compute_ppg_sqi(red_raw, ir_raw, acc_raw, gyro_raw, fs, calibration)
  -> score, components, flags, valid, version, units

apply_sqi_policy(scores, keys, policy, fitted_thresholds)
  -> keep_mask, coverage, rejection_reasons
```

训练、CV、最终模型、holdout 与 runtime 必须调用完全相同的版本和保存参数。

## 7. 测试先决条件

### 7.1 Unit gate

- shape、dtype、采样率、长度、空/短/常数/NaN/Inf。
- 参数合法区间、确定 seed、序列化/反序列化 parity。
- 记录边界和 timestamp 映射。
- 不可用输入必须返回明确 `reason_code`，不能静默制造数字。

### 7.2 Synthetic identifiability gate

统一生成器应包含：

- pulse morphology、HR constant/ramp/chirp、呼吸调制；
- baseline drift、白/彩噪、impulse、clipping/dropout；
- IMU 经已知 FIR/非线性混合生成 artifact；
- artifact 与 HR 不重叠、谐波重叠、频率穿越；
- 运动导致真实 HR 上升的假设破坏场景；
- 双通道可控 mixing matrix。

有 clean truth 时报告 ΔSNR、SI-SDR、相关、频谱失真、pulse amplitude/morphology retention；但 synthetic success 不能替代 real holdout。

### 7.3 Leakage gate

1. 改变 test 分布不得改变 train calibration、SQI、mask、threshold 或 split。
2. 打乱 holdout 标签不得改变训练权重/epoch。
3. 把同 subject 的 record 重命名后仍不得跨 split。
4. 每记录 reset lag/filter/tracker；不跨边界延续状态。
5. ECG/peaks 在推理输入中的任何出现必须由契约显式允许；最终可部署 HR 路线不得依赖 ECG。

### 7.4 Real subject-holdout gate

PTT sit/walk/run 使用 ECG interval/HR reference；pulse timing 另做 delay-aware event评价。每 subject 先生成指标，再计算均值、中位数、95% bootstrap CI 和成对差。

### 7.5 External gate

- SIM：motion detector 与 PPG/ECG HR 泛化。
- iAMwell/MIMIC/VitalDB：在可用 target 边界内评价 peak/IBI/HR。
- `PPG_Testing_05_01_2026` 无 ECG 的双波长记录：只报可用性、SQI、一致性和重复性。

### 7.6 Runtime/deployment gate

- batch 与 streaming parity。
- 最大算法延迟、每秒 CPU、峰值内存、掉帧/缺IMU恢复。
- PT/ONNX 仅在完整 preprocessing→inference→postprocess parity 通过后登记可部署。

## 8. 指标合同

### 8.1 Motion detector

- subject-level BA、F1、AUROC、AUPRC、sensitivity/specificity。
- PPG-only、IMU-only、fusion 消融。
- activity 标签与 artifact 标签分别报告。
- calibration error、coverage 与 uncertain rate。

### 8.2 HR/trajectory

- MAE、RMSE、median AE、P90/P95 AE。
- ±5/±10 bpm accuracy、gross error rate。
- 每分钟不合理 jump、motion onset/offset recovery latency。
- coverage、错误高置信率、confidence calibration。
- record/subject/activity/dataset 分层。

### 8.3 Peak/PPI

- 唯一事件 F1@20/50/100 ms；必须先合并重叠窗口事件。
- matched-event timing error、PPI MAE/RMSE、漏峰/双峰率。
- delay-aware PPG pulse 和 ECG R timing 不混称。

### 8.4 SQI

- 人工质量标签 AUROC/AUPRC。
- retention–peak F1/HR MAE 曲线。
- 固定 coverage 50/70/90% 下性能。
- false-reject clean、false-accept bad、错误高置信输出。
- frailty class/年龄/subject/dataset 保留率公平性。

### 8.5 Waveform（仅有合法 clean reference 时）

- ΔSNR、SI-SDR、相关、频谱距离、pulse morphology、identity distortion。
- 无真实 clean reference 时禁止使用“恢复准确率”；只报代理项并降级证据。

## 9. 标准输出结构

```text
<run_id>/
├── run_manifest.json
├── split_manifest.json
├── config.json
├── input_provenance.json
├── environment_and_hashes.json
├── per_record_metrics.csv
├── per_subject_metrics.csv
├── aggregate_summary.json
├── failures.csv
├── hr_track.csv
├── candidates_or_components.npz
├── calibration/
├── models/
└── plots/
```

`run_manifest.json` 必须包含 route ID、相对 M0 registry 的新增点、目标语义、是否使用 ECG/IMU、seed、代码 SHA、依赖版本、开始/结束状态和失败原因。

`failures.csv` 至少包含：`dataset_id,subject_id,record_id,stage,reason_code,message,recoverable`。

## 10. 路线验收门

| Gate | 通过条件 | 不通过处理 |
|---|---|---|
| G0 Contract | shape/units/record boundary 全明确 | 不进入训练 |
| G1 Unit | 全边界/确定性测试通过 | 修复实现 |
| G2 Synthetic safety | 可辨识且真实HR保留门通过 | 标记风险基线/淘汰 |
| G3 Leakage | train/holdout 隔离测试通过 | 结果作废 |
| G4 Subject holdout | 核心指标相对 raw/SQI baseline 有稳定成对改善 | 不进入 external claim |
| G5 External | 至少一个外部数据集且 subject-level CI | 只称内部探索 |
| G6 Runtime | 完整 pipeline parity、延迟和失败处理通过 | 不称可部署 |

对 waveform denoiser，若不能稳定改善 peak/PPI/HR 且不降低 coverage，就算视觉更平滑也淘汰。对 spectral HR，若相对 raw STFT+SQI 只降低 MAE 但显著降低 coverage，必须在预先规定的 utility/coverage 规则下判断，不能只挑一个指标。

## 11. 推荐实施顺序

1. 固化 `SignalRecord`、split、ECG reference HR、event merge、metrics 与失败码。
2. 修复并冻结 raw/bandpass/Aboy++/current SQI/raw STFT argmax 基线。
3. 实现新版 SQI；它是 detector、denoiser 和 HR 的共同前置层。
4. 实现 spectral candidate + Viterbi；先离线透明，不先上深度模型。
5. 实现 Kalman online；Particle 作为复杂场景消融。
6. 实现 PCA/FastICA/STFT-NMF 前端并与 best single channel 比较。
7. 实现 adaptive/decomposition 全套风险/对照矩阵。
8. 对现有 hybrid bundle 做同一 holdout benchmark，决定继续或淘汰。
9. 只有通过 G4/G5 的路线才进入最终模块整合。

## 12. 需要用户决定但本轮不阻塞归档的问题

这些决定在开始实际实现前必须暂停询问：

1. 是否授权外部联网精读并引用 TROIKA/JOSS 原始论文。
2. 是否允许新增依赖，还是所有 EEMD/VMD/SSA 都必须用当前依赖自行实现。
3. PTT `pleth_1/2` 的波长/位置语义以哪份数据字典为准。
4. motion 主要目标是 activity、可见 artifact、峰不可用性，还是三者分层多任务。
5. 最终效用函数对 HR error 与 coverage/rejection 的优先级。

在这些问题明确前，可以完成无争议的公共合同和 unit fixtures，但不能擅自锁定最终路线含义。

## 13. 用户确认路线的执行合同

### 13.1 固定顺序

SQI-v2 → Motion-29 threshold/CV → Motion→SQI → spectral_track_sqi → dual_ppg_bss_sqi → nonstationary_sqi → adaptive_sqi → PTT HR/PPI → frailty route-feature selection。

每一步必须完成流程报告、算法报告、结果/验证报告并等待用户确认，才能进入下一步。当前文档更新不等同于开始其中任一步实现。

### 13.2 Motion→SQI 数据流

~~~text
raw RED/IR/IMU
├── calibrated physical-unit branch → Motion detector → fold-local probability/state
├── morphology/periodicity branch → SQI components
└── classifier-normalized branch → downstream classifier

Motion probability/state + SQI components
→ train-fold calibration
→ score + valid + flags + no_estimate reason
~~~

禁止从逐窗 z-score 后的 IMU 反推物理运动强度。Motion 阈值、SQI percentile 和权重只能在训练折拟合，随后冻结用于 validation/test/runtime。

### 13.3 PTT reference 合同

- ECG peaks 生成 reference HR/RRI；不得直接称为 PPG pulse peaks。
- 重叠窗口预测先在 record level 合并为唯一事件，再计算 PPI/HR。
- 绝对 PPG pulse timing 必须使用 train-fit、test-frozen 的 ECG→PPG delay。
- 每条路线输出完全相同的 hr_track.csv、event/PPI table、coverage、quality 与 failure schema。

### 13.4 Frailty 特征合同

每条路线输出互斥列前缀：

~~~text
spectral_track_sqi__*
dual_ppg_bss_sqi__*
nonstationary_sqi__*
adaptive_sqi__*
~~~

三层聚合固定为 event/track → window → file/subject；缺失估计保留为 NaN，并附 coverage 与 reason code，不得删除整行或以 0 冒充 HR/PPI。

### 13.5 公平选择合同

1. 29 subjects 以 subject 为 group，所有路线共享同一 5-fold manifest。
2. 所有比较共享 seeds 42,10042,20042,30042,40042。
3. 初始候选固定为 baseline + 4 × {HR-only, PPI-only, HR+PPI}。
4. inner CV 选择路线/特征/阈值，outer test 只评价；outer test 不兼作 early stopping。
5. 主指标为 subject-level BA；同时报告 macro-F1、worst-class recall/F1、coverage、稳定性和失败数。
6. 若暂不做 nested selection，则最高 5-fold BA 只作为开发选择证据，并必须保留后续独立验证集。
