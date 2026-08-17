# Codex Implementation Specification: Thesis–Repository Alignment

- **Repository:** `KaifengXuuu/PPG-Frailty-pipeline`
- **Branch to inspect:** `dev0`
- **Audited commit:** `2eca0ecf0e17a4deaa1d3cc8e821098e5848e421`
- **Specification date:** 2026-08-15
- **Task type:** staged refactor and implementation; do not treat this as a request to run the existing broad historical TODO wholesale.

## 0. Mandatory operating rules

Before editing code:

1. Read `AGENTS.md` and `_agent/WRITE_RULES.md`.
2. Inspect the current git status and work on a dedicated branch; do not overwrite user data, result directories, model artifacts, notebooks, or historical scripts.
3. Do **not** edit `AGENTS.md` or any `_agent/*` file unless the user has explicitly approved the exact proposed text.
4. Treat the revised thesis Chapters 1–3 as the target contract. Current code is evidence of what can be reused, not the final specification.
5. Make small, reviewable commits by phase. Do not start expensive sweeps before the phase’s unit and smoke tests pass.
6. Preserve historical implementations as adapters or explicitly marked legacy code. Do not silently change old result semantics.

## 1. Source-of-truth precedence

When sources disagree, use this order:

1. This implementation specification.
2. `Chapter_3_Revised_Workflow_Redline.docx` (accepted/non-struck workflow text).
3. `Chapter_1_2_Revised_Redline.docx` (project objectives and theoretical scope).
4. Current `dev0` code at the audited commit.
5. `_agent/TODO.md`, `_agent/MODULES.md`, notebooks, archived scripts, and historical result folders as background only.

The broad `_agent/TODO.md` contains additional ideas (hierarchical classification, mobile optimization, extensive historical comparisons). They are **out of scope** unless they are explicitly required below.

## 2. Non-negotiable final workflow

Implement one canonical pipeline with the following hand-offs:

```text
raw files + labels
  -> versioned participant/file manifest and file QC
  -> canonical signal views:
       x_native  = baseline-preserving RED/IR
       x_filt    = detrended + 0.2–8 Hz analysis PPG
       imu_proc  = filtered axes + dynamic acceleration + A/Omega/J
       z_window  = per-window median/IQR-normalized 8-channel deep input
  -> endpoint quality routing (Q_rate, Q_morph)
       direct: x_analysis = x_filt
       processed: x_analysis = ArtifactReducer(x_filt, imu_proc), then quality is recomputed
       rejected: no physiological feature is fabricated
  -> common pulse and feature extractor on x_analysis
  -> three optional representation modes:
       raw             : robust-normalized 8-channel windows
       feature_vector  : one complete fixed vector per recording
       feature_matrix  : one ordered complete matrix [D, K=32] per recording
       fusion          : pooled raw file embedding + encoded feature vector, concatenated once
  -> configured classifier
  -> window -> file -> role-aware participant probability aggregation
  -> participant-grouped evaluation
```

### Required classifier routes

- Raw: 1-D CNN, full/small InceptionTime, ShapeFormer.
- Feature vector: L2 logistic regression, RBF SVM, Extra Trees.
- Feature matrix: InceptionTime, plus ROCKET (10,000 kernels) + ridge.
- Optional five-member InceptionTime ensemble: independently initialized members, equal probability average; usable for raw and feature-matrix inputs.
- Optional fusion: raw embeddings pooled to one file embedding, feature vector encoded once, then concatenated once.

## 3. Explicit non-goals and prohibited shortcuts

- Do not implement the Young-vs-Old hierarchical classifier in this task.
- Do not use an uncalibrated `SpO2 = 110 - 25R` formula or call optical ratios SpO2.
- Do not copy a full-file feature vector to every raw window in the final fusion route.
- Do not aggregate raw windows directly to a participant while bypassing file-level aggregation.
- Do not fit SQI quantiles, feature scalers, imputation, ROCKET kernels, shapelets, calibration, or model selection on held-out participants.
- Do not use the outer held-out fold for early stopping or epoch selection.
- Do not encode invalid or unavailable physiological features as a valid numerical zero without a validity flag.
- Do not include `n_rows`, file duration, number of windows, filenames, subject IDs, file order, or administrative missingness as default predictors.
- Do not claim clean-waveform recovery from the PTT-PPG dataset; artifact modules are judged by downstream peak/HR/PPI performance and coverage.
- Do not run all historical denoisers merely because they exist. Use adapters only where they satisfy the canonical interface.

## 4. Target package structure

Create a canonical package while keeping thin compatibility wrappers in the current scripts:

```text
ppg_frailty/
  __init__.py
  config.py
  contracts.py
  data_manifest.py
  validation.py
  preprocessing.py
  imu.py
  quality.py
  pulse.py
  pipeline.py
  representations.py
  evaluation.py
  bundle.py
  cli.py
  features/
    schema.py
    prv.py
    morphology.py
    dual_wavelength.py
    engineering.py
  artifact/
    base.py
    nlms.py
    decomposition.py
    spectral.py
    bss.py
  models/
    cnn.py
    inception.py
    shapeformer_adapter.py
    rocket.py
    tabular.py
    fusion.py
tests/
  fixtures/
  test_manifest.py
  test_preprocessing.py
  test_quality.py
  test_artifact.py
  test_pulse.py
  test_prv.py
  test_morphology.py
  test_dual_wavelength.py
  test_engineering_features.py
  test_representations.py
  test_models.py
  test_aggregation.py
  test_leakage.py
  test_serialization.py
  test_pipeline_smoke.py
```

Use dataclasses or typed containers for at least:

- `ManifestRow`
- `SignalViews(x_native, x_filt, x_analysis, imu_processed, metadata)`
- `QualityResult(q_rate, q_morph, state, components, reasons, coverage)`
- `PulseResult(peaks, accepted_mask, ppi, adjacency_mask, wavelength, detector_version)`
- `FeatureBundle(vector, vector_validity, engineering_sequence, matrix, matrix_mask, schema_version, provenance)`
- `PredictionBundle(file_probabilities, participant_probabilities, coverage, route, model_version)`

## 5. Staged implementation plan and gates

### P0A — 冻结论文目标、数据契约与测试骨架

- **Scope:** 建立 canonical package/config、SignalViews/FeatureBundle、manifest schema、tests fixtures；不改变模型表现。
- **Primary files:** 新建 ppg_frailty/、tests/；更新 pyproject/README
- **Planning effort:** 3–5 人日
- **Gate:** pytest 能在无私有数据的 synthetic fixtures 上通过；一个 internal smoke fixture 能完整走到 FeatureBundle。

### P0B — 数据 QC、预处理与 fold-safe SQI

- **Scope:** 拒绝全缺失通道；统一 profile；实现 Q_rate/Q_morph、training-only quality calibration、状态路由。
- **Primary files:** data_manifest.py, preprocessing.py, quality.py；替换 main script 相关 helper
- **Planning effort:** 5–8 人日
- **Gate:** QC/滤波/SQI 单元测试 + train/held-out leakage test 通过。

### P0C — 校正 HR/PPI/PRV、morphology、dual-wavelength 和 spectral schema

- **Scope:** 修复 53-gap 中 feature 公式/validity/aggregation；建立显式 vector schema 和 cache v3。
- **Primary files:** features/prv.py, morphology.py, dual_wavelength.py, engineering.py, schema.py
- **Planning effort:** 7–10 人日
- **Gate:** synthetic pulse/interval/PSD gold tests；vector schema 无 process metadata，invalid 可区分。

### P0D — 表示接口与分类器

- **Scope:** 实现 K=32 complete feature matrix、ROCKET+ridge、5-member Inception ensemble、统一 representation_mode。
- **Primary files:** representations.py, models/inception.py, models/rocket.py, tabular.py
- **Planning effort:** 7–11 人日
- **Gate:** raw/vector/matrix 三路在相同 fold registry 上完成 smoke run；序列化可重载。

### P0E — 修复 fusion、聚合和评估协议

- **Scope:** file-level pooling + concatenate once；window→file→role-aware subject；禁用 outer-fold early stopping；完整 config identity。
- **Primary files:** models/fusion.py, evaluation.py, holdout/sweep/analyze
- **Planning effort:** 6–9 人日
- **Gate:** no-leak tests；相同 folds/seeds 下 paired raw/vector/matrix/fusion smoke report。

### P1A — ArtifactReducer 接口与 motion PPG 生理特征回流

- **Scope:** 接入 none/direct、NLMS、至少一个 spectral/wavelet baseline；processed signal 重算 quality 并进入 common HR/PPI/PRV extractor。
- **Primary files:** artifact/ 包、pipeline.py；为 ppg_peak_hr_gating_train.py / hybrid core 写 adapter
- **Planning effort:** 8–14 人日
- **Gate:** external ECG 数据上 raw vs gated vs reducer 输出统一 peak/HR/PPI scorecard；internal W/S 生成带 provenance 的 features。

### P1B — 论文列出的其余可选 artifact modules

- **Scope:** 补齐 decomposition variants 和 RED/IR PCA/ICA/NMF；每个模块遵守同一接口和失败语义。
- **Primary files:** artifact/decomposition.py, spectral.py, bss.py
- **Planning effort:** 8–15 人日
- **Gate:** 每个 reducer 有 unit/smoke test、长度/对齐/finite/identity regression；不能恢复时返回 failure 而非伪信号。

### P1C — 统一 benchmark、回归、打包与复现

- **Scope:** 统一 benchmark/ablation、完整 run manifest、CI、bundle、CLI、README、CPU parity。
- **Primary files:** benchmark.py, bundle.py, cli.py, tests/, .github/
- **Planning effort:** 6–10 人日
- **Gate:** 一条命令完成 frozen folds 的全路线 smoke；bundle reload parity；CI 通过。

Do not advance to the next P0 phase while the current gate fails. P1B may be completed after the P0 static workflow is fully reproducible, but it is required if the thesis continues to state that all listed artifact-reduction families are available.

## 6. Detailed discrepancy backlog

### 数据准备与清单

#### D01 [P0]

- **Thesis contract:** 必需通道缺失或整列无有效数据时，文件应判为 invalid 并停止后续处理。
- **Current code:** read_numeric_csv 只做数值化；interp_nan 在整列无有限值时返回全零数组，可能生成看似合法的信号。
- **Required change:** 新增文件级 schema/QC 校验；整列缺失、通道不存在或有效样本不足时拒绝文件并记录 exclusion_reason。
- **Primary locations:** `frailty_3class_classifier.py: read_numeric_csv, interp_nan, build_manifest`

#### D02 [P1]

- **Thesis contract:** 统一 manifest 记录 subject、class、role、source version、fs、duration、units、channels、QC、reference availability。
- **Current code:** build_manifest 主要记录 path/dataset/subject/role/class/label；采样率、设备单位、版本、checksum 和 QC 状态不在表中。
- **Required change:** 实现版本化 manifest schema 与冻结文件；为内部和外部数据分别保存 source/version/hash/fs/unit/synchrony/reference 字段。
- **Primary locations:** `frailty_3class_classifier.py: build_manifest; 新建 ppg_frailty/data_manifest.py`

#### D03 [P0]

- **Thesis contract:** B/R1-R4 走低运动参考分支；W/S 走 motion-aware routing，并可在降噪后进入同一生理特征提取器。
- **Current code:** role_mode=all_roles 只把 S/W 文件加入同一分类输入，没有 artifact route、quality state 或处理来源字段。
- **Required change:** 把 role 选择与 signal route 分开；为每个片段/文件产生 direct、artifact_reduced、rejected 三类分析来源及处理 provenance。
- **Primary locations:** `frailty_3class_classifier.py: role_suffixes_for_config, build_manifest; 新建 pipeline.py`

#### D04 [P1]

- **Thesis contract:** 在使用固定 400 Hz 或外部 500 Hz 时间轴前验证 duration、sample count、timestamp/synchrony。
- **Current code:** 内部数据直接按配置 fs 重建时间；没有样本数/协议时长/同步一致性检查。
- **Required change:** 加入 duration contract、expected sample count 容差、timestamp 单调性和通道长度一致性测试。
- **Primary locations:** `frailty_3class_classifier.py; 新建 validation.py`

#### D05 [P1]

- **Thesis contract:** 外部 PTT-PPG 数据需冻结版本、通道映射、波长/placement、ECG reference 与 activity 元数据。
- **Current code:** heartbeat/denoiser 脚本各自识别列名和版本；没有统一 external manifest，也未与 Frailty pipeline 共用。
- **Required change:** 实现 external_dataset manifest 和 column mapping registry；固定 PTT-PPG 版本及校验和。
- **Primary locations:** `ppg_peak_hr_gating_train.py; pttppg_denoiser_hybrid_core.py; 新建 external_manifest.py`

#### D06 [P2]

- **Thesis contract:** 文件长度、行数、窗口数用于 QC/provenance，而非生理预测变量。
- **Current code:** extract_file_features 把 n_rows、duration_sec、n_windows 放进所有非 META_COLS 的特征矩阵，传统模型会直接使用。
- **Required change:** 把 process/QC metadata 与 predictor schema 分离；默认不进入模型，仅用于诊断和分层报告。
- **Primary locations:** `frailty_3class_classifier.py: extract_file_features, evaluate, save_final_model`

### 信号视图与预处理

#### P01 [P0]

- **Thesis contract:** 明确并传递 x_native、x_filt、x_AR、deep z 和 processed IMU 五类信号视图。
- **Current code:** 代码局部保留 raw/filtered arrays，但没有统一数据对象；不同脚本对同名信号含义不同。
- **Required change:** 定义 SignalViews/PreparedRecording 数据契约，所有模块只接收显式字段和 profile/version。
- **Primary locations:** `新建 ppg_frailty/contracts.py, preprocessing.py`

#### P02 [P0]

- **Thesis contract:** artifact reducer 输出 x_AR，并在原始时间网格上替代 x_filt 进入共同 pulse/feature workflow。
- **Current code:** Frailty 主分类器没有 x_AR 参数或 adapter；denoiser/heartbeat 脚本与 feature extractor 完全分离。
- **Required change:** 实现 ArtifactReducer protocol 与 AnalysisSignalSelector；none/direct/processed 都返回统一 x_analysis。
- **Primary locations:** `新建 artifact/base.py, pipeline.py; 修改 extract_file_features`

#### P03 [P1]

- **Thesis contract:** 同一 run 使用固定、可追溯的 preprocessing profile。
- **Current code:** frailty、funcs/ppg、hybrid denoiser、peak/gating 脚本分别使用 0.2-8、0.5-8、不同阶数、不同标准化和重力处理。
- **Required change:** 把滤波、重力分离、单位和标准化集中到 profile registry；历史脚本改为调用 adapter 或明确 deprecated。
- **Primary locations:** `frailty_3class_classifier.py; funcs.py; ppg.py; pttppg_*; 新建 config.py/preprocessing.py`

#### P04 [P1]

- **Thesis contract:** 内部主数据保持记录数值尺度；校准/EKF profile 在 profile 内显式转换单位。
- **Current code:** 主分类器保留 native units；动态脚本通过幅值启发式推断 g/mg/m/s² 和 deg/s/rad/s，语义不统一。
- **Required change:** manifest 中记录 unit source；禁止无记录的隐式推断进入 confirmatory pipeline；启发式仅作为明确 experimental adapter。
- **Primary locations:** `ppg_peak_hr_gating_train.py; pttppg_denoiser_hybrid_core.py; preprocessing.py`

#### P05 [P1]

- **Thesis contract:** 低通重力分离与 calibrated roll-pitch EKF 是可配置、run 内固定的 IMU profile。
- **Current code:** 低通重力分离在主线可用；EKF 主要在 funcs/notebook 历史路径，未接入 RunConfig 和统一 feature/model input。
- **Required change:** 实现 imu_profile={lowpass_gravity, calibrated_ekf}；统一输出 axes/magnitudes/jerk 和 provenance。
- **Primary locations:** `funcs.py; frailty_3class_classifier.py; 新建 imu.py`

#### P06 [P2]

- **Thesis contract:** 每个 preprocessing profile 有版本、序列化参数和固定 reference fixtures。
- **Current code:** 缓存版本仅为 FEATURE_CACHE_VERSION，未覆盖完整滤波/单位/QC/窗口配置；无数值回归 fixtures。
- **Required change:** 增加 preprocessing_version、config hash、fixture outputs 和 cache invalidation。
- **Primary locations:** `frailty_3class_classifier.py: FEATURE_CACHE_VERSION/cache names; tests/fixtures`

### 质量路由与运动伪影

#### Q01 [P0]

- **Thesis contract:** 每个分析段输出 invalid、direct/high-quality、motion-but-usable、unrecoverable 等明确状态。
- **Current code:** 只有 none/top70/top50 的相对窗口保留策略，没有状态机，也不表示 unusable/recovery failure。
- **Required change:** 实现 QualityState enum、route decision 和 coverage/reason codes。
- **Primary locations:** `frailty_3class_classifier.py: compute_window_sqi_scores/sqi_keep_mask; 新建 quality.py`

#### Q02 [P0]

- **Thesis contract:** 分别计算 Q_rate 与更严格的 Q_morph。
- **Current code:** 只有一个 composite SQI；pulse timing 与 morphology 共用同一窗口选择。
- **Required change:** 实现 endpoint-specific components、阈值、beat acceptance mask，并将 Q_rate/Q_morph 保存到 FeatureBundle。
- **Primary locations:** `新建 quality.py; 修改 pulse.py/morphology.py`

#### Q03 [P0]

- **Thesis contract:** 质量参考分位数和阈值只由 training participants 拟合。
- **Current code:** compute_window_sqi_scores 在整个已构建窗口池上计算 5th/95th percentiles，CV 之前已包含 held-out windows。
- **Required change:** 拆分 raw SQI components 与 fit/apply QualityCalibrator；每 fold 仅 fit train。
- **Primary locations:** `frailty_3class_classifier.py: compute_window_sqi_scores/evaluate_cnn; evaluation.py`

#### Q04 [P1]

- **Thesis contract:** SQI 的 peak plausibility 与共同 pulse detector 对齐。
- **Current code:** SQI 使用独立 find_peaks(distance=0.28fs,prominence=.3)，与 Aboy++-inspired detector 不同。
- **Required change:** 复用统一 peak candidate API，或明确轻量 detector 并完成与 ECG/reference 的 parity benchmark。
- **Primary locations:** `frailty_3class_classifier.py: compute_window_sqi_scores; pulse.py`

#### Q05 [P1]

- **Thesis contract:** flatline、clipping、long gap 是硬 exclusion；报告有效 coverage。
- **Current code:** 仅通过低 std 间接识别部分 flatline；没有 clipping/gap/saturation flag 和 endpoint coverage。
- **Required change:** 新增 QC flags、hard exclusion、accepted beat/window counts、coverage columns。
- **Primary locations:** `validation.py, quality.py, feature schema`

#### Q06 [P0]

- **Thesis contract:** NLMS、decomposition、spectral、dual-wavelength 模块共享同一 reducer interface。
- **Current code:** 方法分散在 funcs.py、ppg.py、pttppg_* 和 notebook；输入输出、单位、窗口和返回类型不统一。
- **Required change:** 建立 reducer registry，统一 fit/transform、输入视图、输出 x_AR、alignment、confidence 和 failure status。
- **Primary locations:** `新建 artifact/ 包；为历史实现写 adapter`

#### Q07 [P0]

- **Thesis contract:** 至少一个配置化 artifact reducer 真正接入 Frailty dynamic W/S 路线。
- **Current code:** 主 frailty classifier 不调用任何 reducer；hybrid full-waveform 路线被记录为 experimental/deprecated。
- **Required change:** 先接入 none、NLMS 和一个可验证的 spectral/wavelet baseline；其余论文列出的模块按统一 API 完成并测试。
- **Primary locations:** `pipeline.py; artifact/nlms.py, spectral.py, decomposition.py, bss.py`

#### Q08 [P0]

- **Thesis contract:** 降噪后重算 Q_rate/Q_morph，并用共同 extractor 提取 motion PPG 的 HR/PPI/PRV。
- **Current code:** 动态 peak/IBI 模型和 denoiser 输出未送入 frailty feature table；没有统一 provenance。
- **Required change:** 实现 processed-motion adapter、quality re-evaluation、common PulseResult/PRVFeatureBundle，并写入 file/role features。
- **Primary locations:** `ppg_peak_hr_gating_train.py; pttppg_denoiser_hybrid_core.py; pipeline.py/features`

### 脉搏检测与PRV

#### R01 [P1]

- **Thesis contract:** local Aboy++-inspired detector 是共同 peak extractor，并有可重复验证。
- **Current code:** 实现已存在，但与 funcs.py/ppg.py 和公开参考实现无 parity test；动态分支又使用神经 detector。
- **Required change:** 保留本地实现，建立 synthetic、人工标注和 ECG-reference benchmark；记录 detector_version。
- **Primary locations:** `frailty_3class_classifier.py: aboypp_detect_peaks; funcs.py/ppg.py; tests/test_pulse.py`

#### R02 [P1]

- **Thesis contract:** RED/IR 独立检测后输出明确的 wavelength policy、agreement 和配对结果。
- **Current code:** 各自计算 peaks/features，但没有配对容差、reference channel 选择或 shared accepted interval contract。
- **Required change:** 定义 PulseResult per wavelength、pairing/timing agreement、fallback policy。
- **Primary locations:** `pulse.py; morphology.py; quality.py`

#### R03 [P0]

- **Thesis contract:** 清洗后的 peak 序列与 PPI 序列保持一一对应，successive metrics 只在相邻有效 beat 上计算。
- **Current code:** clean_pp_intervals 直接过滤 interval values；删除间隔后仍把剩余数组视为连续序列，RMSSD/SDSD 可能跨越被删除区间。
- **Required change:** 以 peak/event mask 清洗，保留 interval index、gap boundaries 和 contiguous runs；逐域特征遵守有效邻接。
- **Primary locations:** `frailty_3class_classifier.py: clean_pp_intervals, ppi_hrv_features`

#### R04 [P1]

- **Thesis contract:** 统一输出 count/coverage、PPI center/SD/IQR/MAD/CV、rate center/SD、successive、Poincaré。
- **Current code:** 主 PRV 表缺少 PPI MAD、CV、median pulse rate；PPI CV/MAD 分散在 morphology helper。
- **Required change:** 建立唯一 PRV schema 和公式；删除跨模块重复。
- **Primary locations:** `features/prv.py; feature_schema.py`

#### R05 [P0]

- **Thesis contract:** LF/HF/VLF spectral PRV 仅在合格约五分钟 baseline/recovery 记录上计算。
- **Current code:** ppi_hrv_features 只要求 >=10 intervals 且累计 >=60 s；不检查 role/stationarity。
- **Required change:** 按 role/duration/coverage gate；短段返回 missing+validity=false，不返回伪稳定频域值。
- **Primary locations:** `frailty_3class_classifier.py: ppi_hrv_features; features/prv.py`

#### R06 [P1]

- **Thesis contract:** 输出 VLF、LF、HF、LF/HF、normalized units，并在 >=200 intervals 时输出 SampEn。
- **Current code:** 有 LF/HF/total，无独立 VLF、normalized units 或 SampEn。
- **Required change:** 实现参数化 tachogram spectral PRV 与 SampEn(m=2,r=.2SD)，保存参数/validity。
- **Primary locations:** `features/prv.py; tests/test_prv.py`

#### R07 [P0]

- **Thesis contract:** 数据不足的 feature 以 missing/validity/coverage 表示。
- **Current code:** empty_ppi_hrv_features 默认全部 0，使“无法计算”与真实零值不可区分。
- **Required change:** 内部用 NaN+valid flag；fold 内 imputer 处理；报告 coverage/invalid reason。
- **Primary locations:** `frailty_3class_classifier.py: empty_ppi_hrv_features; schema/evaluation`

### 形态学与双波长

#### M01 [P0]

- **Thesis contract:** pulse amplitude = peak − linear valley-to-valley baseline at peak。
- **Current code:** _pulse_feature_summary 使用 peak − max(left valley,right valley)。
- **Required change:** 改为在 peak 时刻插值 baseline；保留单独 preceding-valley amplitude 仅在 schema 明确时使用。
- **Primary locations:** `frailty_3class_classifier.py: _pulse_feature_summary`

#### M02 [P0]

- **Thesis contract:** 仅 Q_morph accepted beats 进入形态学。
- **Current code:** 只检查 valley/positive amplitude，没有 endpoint quality/beat template gate。
- **Required change:** 接入 Q_morph beat mask、rise-time/width/coverage validity。
- **Primary locations:** `quality.py; morphology.py`

#### M03 [P1]

- **Thesis contract:** beat features以 median 和 MAD 汇总。
- **Current code:** amplitude 等主要输出 mean/std；rise/decay/width/slope/area 只输出 mean。
- **Required change:** 统一输出 median+MAD，并按论文 schema 命名；旧 mean/std 可保留为 legacy 非默认字段。
- **Primary locations:** `frailty_3class_classifier.py: _pulse_feature_summary`

#### M04 [P0]

- **Thesis contract:** AC 是 accepted beat amplitude 的代表值，DC 是对应 beat baseline 的代表值。
- **Current code:** AC=整文件 filtered SD；DC=整文件 raw median，与表中 beatwise 定义不同。
- **Required change:** 从有效 beat 计算 per-beat AC/DC，然后 median/MAD；保留 whole-file SD 用不同名称。
- **Primary locations:** `morphology.py; dual_wavelength.py`

#### M05 [P1]

- **Thesis contract:** 每个 wavelength 输出 PI=AC/(|DC|+epsilon) 及 validity。
- **Current code:** 无显式 PI predictor。
- **Required change:** 实现 PI_RED/PI_IR；不生成未经标定的 SpO2。
- **Primary locations:** `dual_wavelength.py; feature schema`

#### M06 [P0]

- **Thesis contract:** 统一使用 RED/IR，R=(AC_RED/DC_RED)/(AC_IR/DC_IR)。
- **Current code:** 字段为 ir_red_*，ratio-of-ratios 是论文 conventional R 的倒数。
- **Required change:** 改 canonical RED/IR schema；为旧 cache 提供显式 migration alias，不静默混用。
- **Primary locations:** `frailty_3class_classifier.py: morphology_features; cache migration`

#### M07 [P1]

- **Thesis contract:** 输出 rho0、rho_max、tau* 及 cardiac-band coherence。
- **Current code:** 有 Pearson correlation 和 lag；不输出最大归一化相关系数、coherence，full correlate 未按 overlap 标准化。
- **Required change:** 实现 lag-normalized cross-correlation 和 coherence；保存 search bounds。
- **Primary locations:** `dual_wavelength.py; tests/test_dual_wavelength.py`

#### M08 [P1]

- **Thesis contract:** 所有 ratio 有 denominator checks/validity；特征物理量不混合未知单位。
- **Current code:** epsilon 强制产生数值；motion_norm_* 把不同未知单位的 acc/gyro 相加并进入 morphology selector。
- **Required change:** 加入 validity flags；默认移除 motion_norm_* 或将其归入 quality-only experimental schema。
- **Primary locations:** `frailty_3class_classifier.py: morphology_features/MORPHOLOGY_PREFIXES`

### 工程统计与频谱

#### E01 [P1]

- **Thesis contract:** PPG bands: 0.2-0.5, 0.5-3, 3-8 Hz；IMU bands: 0.1-0.5, 0.5-3, 3-8, 8-20 Hz。
- **Current code:** 所有信号统一使用 0.1-0.5、0.5-3、3-8、8-20；PPG 还生成滤波截止以上 8-20。
- **Required change:** 按 signal family 定义 band registry，更新字段名和 cache version。
- **Primary locations:** `frailty_3class_classifier.py: spectral_features/per_window_features`

#### E02 [P1]

- **Thesis contract:** spectral entropy 归一化为 -sum(p log p)/log K。
- **Current code:** 实现未除以 log K，值随 nperseg/window length 改变。
- **Required change:** 实现 normalized entropy；旧值命名为 entropy_nats_legacy，避免缓存混淆。
- **Primary locations:** `frailty_3class_classifier.py: spectral_features`

#### E03 [P0]

- **Thesis contract:** generic optical features 使用 x_analysis，motion segment 可来自 x_AR。
- **Current code:** extract_file_features 总是从原始文件重新 bandpass，不接收 artifact-reduced signal。
- **Required change:** 让 feature extractor 接收 PreparedRecording/SignalViews，不自行读取和重建 route。
- **Primary locations:** `frailty_3class_classifier.py: extract_file_features; features/engineering.py`

#### E04 [P1]

- **Thesis contract:** 保留按时间排序的 10 s engineering-window descriptor rows 及 mask。
- **Current code:** per-window rows 计算后立即被 mean/std 压缩，未缓存顺序、start time 或 validity。
- **Required change:** 输出 EngineeringFeatureSequence(window_start, values, valid_mask)，同时产生 file aggregates。
- **Primary locations:** `frailty_3class_classifier.py: extract_file_features; representations.py`

### 表征与分类器

#### C01 [P0]

- **Thesis contract:** 构造 K=32 的 ordered complete engineered-feature matrix，长记录均匀采样，短记录标准化后 mask padding。
- **Current code:** 没有 feature matrix builder；只有一行 file feature vector。
- **Required change:** 实现 fold-safe matrix builder、channel schema、mask、cache 和 serialization。
- **Primary locations:** `新建 representations.py; datasets feature cache v3`

#### C02 [P0]

- **Thesis contract:** feature matrix 可输入 InceptionTime。
- **Current code:** Inception 模型类理论上可接任意 CxT，但训练/数据入口只构造 raw 8-channel windows。
- **Required change:** 新增 feature_matrix dataset/evaluator；一个 recording=一个样本；mask/padding 正确处理。
- **Primary locations:** `models/inception.py; evaluation.py`

#### C03 [P0]

- **Thesis contract:** ROCKET 10,000 random kernels + ridge classifier，所有拟合在 train fold。
- **Current code:** 仓库无 ROCKET/MiniROCKET 实现或依赖。
- **Required change:** 新增独立 feature-matrix ROCKET route，保存 kernels/transform/ridge/schema。
- **Primary locations:** `新建 models/rocket.py, frailty_3class_rocket.py; pyproject.toml`

#### C04 [P1]

- **Thesis contract:** 可选原始 InceptionTime 五网络独立初始化概率平均 ensemble。
- **Current code:** 每 fold/repeat 只训练一个 Inception network。
- **Required change:** 实现 ensemble_size=5 wrapper、独立 seeds、成员保存和等权概率平均；raw/matrix 共用。
- **Primary locations:** `models/inception.py; training/evaluation`

#### C05 [P1]

- **Thesis contract:** 统一 representation_mode={raw, feature_vector, feature_matrix, fusion}。
- **Current code:** model/extra_input/manual_features 多组开关不能完整表达四条路线；feature-only deep route缺失。
- **Required change:** 新增单一 mode registry 和 schema validation；旧 CLI 映射为兼容 alias。
- **Primary locations:** `RunConfig/config.py, CLI, benchmark`

#### C06 [P1]

- **Thesis contract:** 固定向量由论文定义的 predictor schema 构成，并携带 feature validity/provenance。
- **Current code:** 传统模型自动选择所有 non-meta 列，包含 process fields；selector 依赖前缀/后缀，容易随 cache 漂移。
- **Required change:** 建立显式 versioned FeatureSchema；禁止自动“所有数值列”作为最终输入。
- **Primary locations:** `frailty_3class_classifier.py: evaluate/select_extra_feature_columns; feature_schema.py`

### 融合、验证与聚合

#### V01 [P0]

- **Thesis contract:** raw window embeddings 先 pool 成 file embedding；feature vector 只拼接一次。
- **Current code:** scaled_file_features[file_win] 把同一整文件向量复制到每个 raw window，逐窗口训练与预测。
- **Required change:** 实现 file batch/pooling 和 file-level fusion classifier；删除 repeated-context 主路径。
- **Primary locations:** `frailty_3class_classifier.py: FeatureFusionClassifier/evaluate_cnn; models/fusion.py`

#### V02 [P0]

- **Thesis contract:** window→file 后，再对 participant 的记录/role 做等权 role-aware mean。
- **Current code:** evaluate_cnn 另行直接 window→subject；窗口更多/文件更长的记录权重更大。
- **Required change:** 唯一聚合层级：window→file→role/participant；记录每级 coverage 和权重。
- **Primary locations:** `frailty_3class_classifier.py: aggregate_by_key_with_quality/evaluate_cnn; evaluation.py`

#### V03 [P0]

- **Thesis contract:** outer held-out participants 不用于 epoch selection；固定 epoch 或 inner validation。
- **Current code:** evaluate_cnn 默认把 outer test fold 传入 x_val，且 cnn_select_best_epoch 默认 True。
- **Required change:** 默认固定 epoch；需要 early stopping 时在 outer-train 内做 participant-grouped inner split。
- **Primary locations:** `frailty_3class_classifier.py: RunConfig/train_cnn_model/evaluate_cnn`

#### V04 [P0]

- **Thesis contract:** strict holdout/locked evaluation 必须完整重建选定 configuration。
- **Current code:** holdout CONFIG_COLUMNS/config_from_rank_row 只恢复少量字段，遗漏 SQI、aggregation、manual features、loss、weights、sampler 等。
- **Required change:** 用完整 serialized RunConfig/manifest 加载，不再从不完整 leaderboard 行猜配置。
- **Primary locations:** `frailty_3class_holdout_eval.py; run manifest`

#### V05 [P1]

- **Thesis contract:** raw/vector/matrix/fusion 使用同一 folds/seeds，完整 config identity 和统一 benchmark。
- **Current code:** analyze_sweep 的 config key 缺少多个关键字段；不同脚本协议和默认值不一致；无统一 benchmark wrapper。
- **Required change:** 扩展 config ontology，建立 frozen fold registry 与统一 benchmark，支持 paired ablation。
- **Primary locations:** `analyze_sweep.py; frailty_3class_overfitting_sweep.py; 新建 benchmark.py`

### 测试、打包与复现

#### O01 [P0]

- **Thesis contract:** 核心 pipeline 有 unit/integration/leakage/regression tests 和 CI。
- **Current code:** dev0 未发现 tests/ 或 .github/workflows；pyproject 仅列 pytest dev dependency。
- **Required change:** 建立 pytest suite、small fixtures、GitHub Actions（lint/unit/smoke）和 deterministic seeds。
- **Primary locations:** `新建 tests/, .github/workflows/ci.yml`

#### O02 [P1]

- **Thesis contract:** 可通过明确 CLI/package 复现 preprocessing、features、models、evaluation。
- **Current code:** README 仍指向 main 分支和 notebook；pyproject 描述/entry point 过时，且缺 pandas/sklearn/joblib/torch/ROCKET 等实际依赖。
- **Required change:** 更新 package structure、dependencies、CLI、README 和 environment lock。
- **Primary locations:** `README.md; pyproject.toml; 新建 cli.py`

#### O03 [P1]

- **Thesis contract:** 最终 bundle 保存完整 preprocessing/schema/scaler/model/aggregation，并做 load/inference parity。
- **Current code:** 部分 sklearn/PT 模型可保存；动态 ONNX 有独立 bundle，但没有统一 end-to-end bundle 或跨 runtime parity。
- **Required change:** 实现 versioned PipelineBundle、schema validation、CPU inference smoke 和 serialization parity。
- **Primary locations:** `pipeline.py; bundle.py; save_final_*; ONNX/export tests`

## 7. Exact feature contracts

### 7.1 PRV

- Keep peak indices and interval indices linked; mark gaps so successive-difference features never bridge a rejected interval.
- Output count, accepted duration, coverage, mean/median PPI, mean/median pulse rate, PPI SD/IQR/MAD/CV, pulse-rate SD, RMSSD, SDSD, NN50, pNN50, SD1, SD2, SD1/SD2.
- Compute spectral PRV only on eligible approximately five-minute B/R recordings: 4 Hz interpolated/detrended tachogram, VLF 0.003–0.04, LF 0.04–0.15, HF 0.15–0.40 Hz, LF/HF and normalized units.
- Compute SampEn only when `accepted_intervals >= 200`, with `m=2`, `r=0.2*SD`.
- Unavailable values are NaN + validity=false, not zero.

### 7.2 Morphology and dual wavelength

- Accepted beats are valley-to-valley and pass Q_morph.
- `A_p = x(t_p) - l(t_p)`, where `l(t)` is the linear valley-to-valley baseline.
- Rise, decay, half-prominence width, mean systolic upslope, and positive baseline-corrected area.
- Aggregate beat values by median and MAD.
- Beatwise AC and DC, PI per wavelength, RED/IR AC ratio, RED/IR DC ratio, conventional RED-over-IR ratio-of-ratios.
- Zero-lag Pearson correlation, maximum lag-normalized correlation, argmax lag, and mean 0.5–3 Hz magnitude-squared coherence.
- Every ratio has denominator validity and finite checks. Keep motion-normalized mixed-unit heuristics out of the default predictor schema.

### 7.3 Engineering features

- 10 s complete windows, 5 s hop; preserve chronological start times and validity.
- Time descriptors: mean, population SD, RMS, IQR, MAD, bias-corrected Fisher–Pearson skewness, Pearson kurtosis.
- Normalized spectral entropy: `-sum(p log p)/log(K)`.
- PPG bands: 0.2–0.5, 0.5–3, 3–8 Hz.
- IMU magnitude bands: 0.1–0.5, 0.5–3, 3–8, 8–20 Hz.
- File vector contains mean and SD of valid engineering rows.

### 7.4 Complete ordered feature matrix

- One recording = one matrix sample.
- Reference `K=32` chronological engineering positions.
- Time-varying channels = per-window engineering descriptors.
- Constant context channels = complete standardized file feature vector repeated across valid positions.
- If `W>K`, select K windows uniformly over recording progress; if `W<K`, right-pad after fold-standardization and provide a Boolean mask.
- Fit imputation/scaling only on training recordings. Save feature order and schema version.

## 8. Required test suite

### Unit tests

- Manifest: label/role mapping, exact 8-channel order, all-missing channel rejection, sampling-rate/duration contract, no predictor leakage from metadata.
- Filtering: impulse/frequency response, pass/stop bands, zero-phase path, short-signal fallback, deterministic output.
- Normalization: per-window median/IQR output, fallback scale, clipping, no mutation of amplitude-preserving feature view.
- Quality: Q_rate/Q_morph monotonic behavior, hard exclusions, train-only quantile fit, post-artifact recalculation.
- Pulse detector: synthetic pulse train, polarity inversion, known missing/extra beats, event-index/PPI adjacency preservation.
- PRV: formula fixtures for SDNN/RMSSD/SDSD/NN50/Poincare, five-minute eligibility, SampEn eligibility, NaN validity.
- Morphology: synthetic triangular/asymmetric beats with analytically known amplitude/timing/area; median/MAD aggregation.
- Dual wavelength: exact ratios, PI, conventional R direction, zero-lag/max-lag/coherence, denominator failure.
- Spectral: sinusoid peak/centroid/band power and normalized entropy at multiple lengths.
- Representations: fixed vector order, K=32 sampling, padding/mask, fold-only standardization, deterministic cache.
- Models: Inception member independence and exact probability average; ROCKET transform fit only on train; output probabilities finite/sum to one.
- Aggregation: window-to-file first; equal recording/role contribution; result invariant to duplicating a file’s windows.
- Serialization: save/reload parity for sklearn, ROCKET, neural ensemble, feature schema, and full bundle.

### Integration tests

- Low-motion internal fixture: raw file -> signal views -> quality -> pulse/features -> vector/matrix -> each classifier smoke.
- Motion fixture: x_filt -> reducer -> x_AR -> re-quality -> common HR/PPI/PRV extractor; provenance retained.
- External ECG fixture: one-to-one peak matching, HR/PPI error, coverage, raw-vs-gated-vs-reducer comparison.
- One participant-grouped 2-fold smoke with raw, vector, matrix, and fusion modes on identical frozen folds.
- CLI smoke from manifest build through report generation in a clean temporary directory.

### Leakage tests

- No participant appears in more than one split.
- Feature imputer/scaler, quality calibrator, ROCKET transform, shapelets, calibration, and early-stopping validation are fitted only from outer-train participants.
- Changing held-out values cannot change training-fold scaler/SQI references or selected epoch.
- Full-file features are never exposed to an individual raw window before file pooling in the final fusion route.

### Scientific/benchmark tests

- Peak/HR/PPI: event precision/recall/F1 at declared tolerances, timing error, HR MAE/RMSE/bias, PPI MAE, coverage by dataset/activity.
- Artifact policies: raw/no-denoise, quality-only, and each reducer under identical external subject splits.
- Frailty classification: raw/vector/matrix/fusion on the same participant folds and seeds; participant BA, macro-F1, per-class metrics, confusion, coverage.
- Paired ablations: preprocessing, SQI, artifact route, feature families, ensemble, ROCKET, and fusion; one factor at a time.
- Locked evaluation: selected full config reconstructed from a serialized manifest, not a shortened leaderboard row.

## 9. Acceptance criteria for the final aligned implementation

- One CLI/config can switch among `raw`, `feature_vector`, `feature_matrix`, and `fusion` without changing data split semantics.
- Every result stores commit, manifest hash, fold registry, preprocessing version, feature schema, complete RunConfig, seed, and model/bundle version.
- All 53 discrepancies in this document are closed or explicitly documented as thesis text that must be narrowed.
- `pytest -q` and the CI smoke workflow pass.
- The canonical feature extractor accepts both direct x_filt and artifact-reduced x_AR through the same API.
- The matrix route runs both InceptionTime and ROCKET+ridge.
- The Inception ensemble contains five separately initialized/fitted members and uses exact equal probability averaging.
- The final fusion uses one file feature vector once, after raw window pooling.
- Participant probabilities are computed through window→file→participant/role aggregation.
- No held-out participant affects fitted preprocessing, quality references, model selection, or calibration.

## 10. Deliverables Codex must produce

- Code changes and new modules described above.
- A `MIGRATION.md` mapping legacy functions/caches to canonical functions and schema versions.
- A machine-readable `feature_schema.json` and `preprocessing_profile.json`.
- A frozen fold registry for the executed benchmark.
- Unit/integration/leakage tests and a CI workflow.
- A smoke-test report showing all four representation modes.
- A gap-closure checklist keyed by D01–O03, with commit/test evidence for each closed item.
- Updated README and CLI instructions.

At the end of every phase, report: changed files, behavior change, tests run, test results, unresolved gap IDs, and whether any thesis wording still exceeds the implementation.