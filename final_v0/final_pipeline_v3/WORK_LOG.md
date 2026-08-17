# Final Pipeline V1 work log / 工作日志

> Auto-generated from records/log_entries; tracking updates are not project events.
> 由 records/log_entries 自动生成；追踪文档更新本身不计项目事件。

# Phase 01 — Spec lock and ADR 001–004 / 规格锁与前四项 ADR

- 状态 / Status: completed
- 流程 / Process: 完整读取 766 行合并规范，核对 bytes/hash/commit/branch，停止旧 TODO 子任务，并建立 V1 独立边界。
- 算法 / Algorithm: 尚未写入算法行为；先冻结入口、manifest/folds、SignalViews/单位、WindowPlan/padding/mask 四项合同。
- 结果 / Result: 新建 README、SPEC_LOCK、ADR-001–004 与自动追踪工具。
- 限制 / Limitation: 其余 ADR、代码、测试和 benchmark 尚未实现，当前不得声称 pipeline 完成。

---

# Phase 02 — ADR 005–008 / PRV、聚合、训练隔离与模型命名

- 状态 / Status: completed
- 流程 / Process: 按规范冻结 PRV 时间资格、分层聚合、outer-fold 隔离和模型命名。
- 算法 / Algorithm: 保留 interval adjacency；300 s spectral PRV；window→file→role→participant；fixed/inner-only epoch；单网络与五成员 ensemble 分名。
- 结果 / Result: 新建 ADR-005–008；记录三项 V2 待确认参数但不阻塞 V1 conservative defaults。
- 限制 / Limitation: 尚未实现对应代码与测试。

---

# Phase 03 — ADR 009–012 / 时间尺度、运动边界、表征与 Rate-only 合同

- 状态 / Status: completed
- 流程 / Process: 冻结 DL-only resampling、motion primary boundary、四种表征与非恒等 artifact rate-only 强约束。
- 算法 / Algorithm: 400 Hz audit轴不变；100/160/200/400 DL ablation；drop XOR reducer；K=32；Q_morph_post not_applicable。
- 结果 / Result: 12 份前置 ADR 全部建立，并开始累计 V2 人工确认点。
- 限制 / Limitation: ADR 是实现门而非完成证明；代码与验收测试仍待建设。

---

# Phase 04 — Configuration, typed contracts, and provenance / 配置、类型与溯源

- 状态 / Status: implemented_pending_tests
- 流程 / Process: 在 12 ADR 完成后建立 canonical package、strict YAML/JSON config、跨模块 typed containers 与 training-only provenance guard。
- 算法 / Algorithm: 四种 representation；direct/identity/non-identity routes；Q_morph not_applicable 强校验；canonical config hash；outer train/OOF 集合互斥。
- 结果 / Result: 新增 pyproject、包入口、config.py、contracts.py、provenance.py。
- 限制 / Limitation: 数据、信号、模型与 CLI 尚未接入；当前只完成底层合同。

---

# Phase 04b — Reference configs and baseline / 参考配置与基线

- Re-scanned the 766-line locked specification before this logical write batch.
- Added a generator for four fully resolved YAML configurations; runtime has no config inheritance or hidden behavior defaults.
- Added a read-only baseline-audit generator that writes only under V1 and labels historical metrics non-strict.
- Added the legacy-to-V1 migration crosswalk and explicit no-shortcut boundary.
- Validation planned immediately after materialization: strict config load, strict JSON parse, source-hash inventory, and tracking synchronization.

---

# Phase 04c.1 — Validator lock-field correction / 验证器锁字段修正

- Re-read `SPEC_LOCK.json` and recomputed the attached specification SHA-256 before editing.
- Corrected the validator from nonexistent `sha256` to the authoritative `source_sha256` field.
- The observed file hash remained `cd7c4907...3c5000`; no specification or lock content changed.

---

# Phase 04c — Test and validation harness / 测试与验证框架

- Re-scanned the V1 tree and local dependency inventory before writing.
- Confirmed pytest, ruff, mypy, and coverage are not installed; no undeclared test dependency was introduced.
- Added a standard-library unittest runner with selectable data/signal/artifact/feature/model/training/integration suites and strict JSON reporting.
- Added a deterministic validator for required paths, specification hash, Python AST, bilingual documentation, forbidden legacy runtime imports, strict JSON, and fully resolved configs.
- The validator is intentionally extensible; later batches must add data, route, OOF, bundle, and comparison invariants before the final pass is authoritative.

---

# Phase 04d — Core contract tests / 核心合同测试

- Re-scanned the V1 source tree and the locked specification before writing.
- Added standard-library tests for all four resolved configuration files, exact top-level keys, frozen memberships, and hidden outer labels.
- Added 400 Hz alignment, non-identity rate-only, explicit `Q_morph=not_applicable`, and strict JSON null tests.
- Added a byte-level specification-lock regression.

---

# Phase 04e — V1 algorithm diagrams / V1 算法图

- Re-scanned the locked specification sections 4–8 and the active V1 tree before writing.
- Added six professional Mermaid diagrams for the end-to-end route, frozen-fold boundary, signal/SQI/artifact features, representation/model families, Trainer/OOF/bundle, and paired comparisons.
- Diagrams explicitly encode training-only fitting, window→file→role→participant aggregation, and the non-identity morphology prohibition.

---

# Phase 04f — V2 decision registry / V2 决策注册表

- Re-scanned the specification, TODO, M0–M3 current status, and three implementation audits before writing.
- Consolidated 27 human-confirmation points with the conservative V1 choice, alternatives, and rerun/deployment impact.
- No point is marked user-confirmed; no `_agent` file is read as authorization to decide it.
- Changes after review require new config/schema identities and the user-requested `final_pipeline_v2` directory.

---

# Phase 04g — Matrix input-dimension contract / Matrix 输入维度合同

- Re-read the resolved configs and model factory before editing.
- Replaced the feature-matrix sentinel `input_channels=-1` with an explicit schema-derived resolution rule; negative dimensions remain invalid.
- Added `input_channels_resolution` to every resolved model section so no factory or runner silently guesses dimensions.
- Configuration human names remain canonical presentation names; the model registry records the corresponding stable machine ID.

---

# Phase 04h — Analysis-view correction / Analysis 视图修正

- Re-read contract §5.3, ADR-003, the resolved YAML generator, diagram, and current signal facade.
- Removed an unused 0.4–8 Hz secondary direct filter from the configuration.
- Frozen semantics are now explicit: direct `x_analysis=x_filter` at zero-phase 0.2–8 Hz; non-identity `x_analysis=x_ar` and is rate-only.
- This prevents configuration/runtime drift and preserves the amplitude-sensitive direct contract.

---

# Phase 04i — Raw-window padding alignment / Raw 窗口 padding 对齐

- Re-read contract §5.3, the resolved window config, `CompactCNN1D.forward_features`, and feature-matrix padding rules.
- The V1 reference raw route now emits complete 5-second windows only; it does not ask a model that rejects non-trivial masks to consume right padding.
- Feature-matrix remains explicitly right-padded after fold-local transformation with a row mask.
- A future padded-raw route requires a tested mask-propagating convolution/pooling policy and a distinct config ID.

---

# Phase 06 — Signal, artifact, quality, and features / 信号、伪影、质量与特征层

- Date / 日期: 2026-08-15
- Status / 状态: implemented_and_verified_on_synthetic_contract_tests
- Write boundary / 写入边界: `final_v0/final_pipeline_v1` only
- Tracking sync / 跟踪同步: intentionally not run; root task performs merged sync

## 1. Scope and scientific boundary / 范围与科学边界

中文：本阶段实现 400 Hz 信号视图、PPG/IMU 预处理、峰/PPI/PRV、direct-only
形态与双波长 optical、端点 SQI、伪影 reducer、工程特征、冻结十字段 registry、
FeatureVectorV1 和带显式 validity channels 的 K=32 matrix。非恒等 reducer 只产生
`x_ar` rate-only 视图；形态、幅值和 DC 相关特征不能从 `x_ar` 提取。reducer 失败
返回 no result，不回退 direct。本文记录的量化数值来自解析/合成合同测试，不是
29-subject benchmark，也不是论文性能结论。

English: This phase implements the 400 Hz signal views, PPG/IMU preprocessing,
peaks/PPI/PRV, direct-only morphology and dual optical features, endpoint SQI,
artifact reducers, engineering features, the frozen ten-field registry,
FeatureVectorV1, and a K=32 matrix with explicit validity channels. A successful
non-identity reducer creates only rate-only `x_ar`; morphology, amplitude, and
DC-dependent features cannot consume it. Failure produces no result and never
falls back to direct. Quantitative values below are analytic/synthetic contract
evidence, not a 29-subject benchmark or thesis performance claim.

## 2. Frozen authorities and copied mathematics / 冻结权威与数学迁移

| Authority / 权威输入 | SHA-256 |
|---|---|
| Merged implementation specification | `cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000` |
| M3 PPG source | `d68a807893b02b590341199822425e83ec3fcba5e44e2f6597b944e21001abe5` |
| M3 quality source | `b246d0bfae4afdf6275b92d9579ded7fde063f3d0d33b49c77d6af43d5386bdc` |
| M3 IMU entry source | `296e24b76efee4f3417a2932d818c2eea12020658b59b2bef36cdeab2421f787` |
| M3 IMU math source | `e5e1e4690ab04a184ee9838b856e6495a46f894b0d28b747d4f5e868ab8e9da9` |
| M3 IMU runtime source | `f4d214dbe0a1b52fab9c98765de7cd13ee1a920518961332713522881ead3f95` |

中文：M3 数学被复制、审阅并按 V1 合同重写；V1 运行时不 import M3。IMU 主路线
是持久 quaternion multiplicative error-state EKF（含 gyro bias、6×6 covariance、
adaptive acceleration gate、pending/tracking/prediction_only/no_estimate 状态）；对照路线
是共享 20/40 Hz causal sensor filters 后的 causal order-2 0.3 Hz gravity LPF。

English: M3 mathematics was copied, reviewed, and rewritten under V1 contracts;
V1 has no runtime import from M3. The primary IMU route is a persistent quaternion
multiplicative error-state EKF with gyro bias, 6x6 covariance, adaptive acceleration
gating, and explicit state transitions. Its comparator is a causal second-order
0.3 Hz gravity LPF after the shared causal 20/40 Hz sensor filters.

## 3. Implemented contracts and algorithms / 已实现合同与算法

### 3.1 Signal views and preprocessing / 信号视图与预处理

- `x_native`: acquisition-scale RED/IR after bounded internal gap repair;
- `x_filter`: explicit Butterworth SOS 0.2–8 Hz, linear detrend, zero phase;
- `x_analysis_rate`: exactly `x_filter` for direct/identity;
- `x_ar`: aligned successful non-identity rate-only output;
- formal config validates every signal subsection, unit, channel order, analysis
  view, gap policy, IMU profile, resampling, and normalization field;
- gap repair consumes `max_gap_samples`; flatline consumes the resolved quality
  duration; an unknown signal key fails closed;
- only `data.windows.WindowPlan` is authoritative; signal re-exports that class
  and contains no second planner.

### 3.2 Peak, PPI, and PRV / 峰、PPI 与 PRV

- dual-polarity, dual-wavelength candidate scoring with true sample timestamps;
- interval endpoints and adjacency remain on the uncompressed original timeline;
- RMSSD/SDSD/NN50 cannot cross a rejected interval;
- SampEn uses the longest contiguous valid PPI run and normalized unordered-pair
  match probabilities for both `m` and `m+1`;
- spectral PRV uses one contiguous >=300 s run, >=200 intervals, >=0.80 coverage,
  4 Hz interpolated/detrended tachogram, explicit Q_rate, and eligible long roles;
- B/R/R1/R2/R3/R4/static/reference/recovery roles are eligible when all gates pass;
- rate-qualified direct, identity, and non-identity rate-only routes may produce
  spectral PRV; the rate-only ban applies to morphology/amplitude, not PRV.

### 3.3 Morphology and dual wavelength / 形态与双波长

- route guard executes before waveform access;
- accepted beats use a local linear valley-to-valley baseline;
- morphology reports amplitude, half width, rise/decay, slopes, positive area,
  and robust median/MAD;
- optical AC is filtered peak minus baseline at the peak;
- optical DC is the aligned native local baseline at the same peak;
- canonical beat/file values include RED/IR AC, DC, PI, AC ratio, DC ratio, and
  `R=(AC_RED/DC_RED)/(AC_IR/DC_IR)`, with denominator/finite validity;
- zero-lag correlation, bounded normalized cross-correlation/lag, and cardiac-band
  coherence are direct-only.

### 3.4 Endpoint SQI / 端点 SQI

- separate Q_rate and stricter Q_morph; non-identity sets Q_morph not_applicable;
- cardiac concentration, autocorrelation periodicity, normalized spectral entropy,
  peak density, full PPI plausibility/stability, RED–IR agreement, IMU motion,
  template correlation, skewness, Pearson kurtosis, coverage, flatline, clipping,
  saturation, and long-gap slots expose raw/normalized/state/reason;
- ADC saturation is unavailable when rails are unknown; it is never inferred as pass;
- density/PPI/flatline/long-gap/endpoint thresholds are read from SqiConfig;
- `outer_train_empirical_quantiles_v1` requires a fitted SqiCalibrator whose
  participant IDs pass the outer-train membership guard;
- artifact-valid masks restrict metrics to a contiguous run while preserving full
  recording coverage.

### 3.5 Artifact reducers / 伪影 reducer

| Reducer | Implementation / 实现 | Boundary / 边界 |
|---|---|---|
| identity | exact byte-value no-op | direct/identity morphology remains eligible |
| NLMS | multi-reference ANC, explicit delay taps, leakage/update gate | IMU mask excludes prediction/update rows; rate-only |
| SSA | Hankel SVD, diagonal averaging, cardiac concentration selection | no below-threshold argmax fallback; rate-only |
| spectral_mask | formal 4 s/1 s STFT, IMU quantile soft mask, 0.5–3 Hz rate band | exact YAML keys; unknown keys reject; rate-only |
| PCA BSS | dual-channel PCA component selection | single channel fails closed; rate-only |
| FastICA BSS | deterministic FastICA component selection | single channel fails closed; rate-only |
| NMF BSS | non-negative shifted STFT magnitude factorization | single channel fails closed; rate-only |
| learned denoiser | registered unsupported | no audited model artifact, therefore no waveform |

IMU-invalid samples propagate to NLMS/STFT `output_valid_mask`, then to canonical
views, longest-contiguous-run pulse detection, and endpoint SQI. Spectral
`confidence` is retained-signal agreement on valid samples; suppression is a
separate `suppression_fraction_by_channel`, so a clean/low-suppression signal is
not mislabeled low-confidence.

### 3.6 Features and model tensors / 特征与模型 tensor

- engineering: complete 10 s windows, 5 s hop, chronological starts;
- PPG descriptors are direct/identity only; processed IMU descriptors remain
  available on rate-only routes when finite;
- invalid constant-channel skew/entropy stays NaN/false and is never forced valid;
- default file aggregation is mean plus population SD (`ddof=0`);
- every registry item records: canonical name, formula/algorithm, units, source
  signal view, endpoint/role eligibility, level, aggregation, validity, missing
  policy, and provenance/version; route and scientific group are additional fields;
- unknown/technical predictors are rejected from the default allowlist;
- FeatureVectorV1 always has complete frozen order plus parallel validity;
- K=32 matrix accepts fold-transformed EngineeringExtraction and a complete
  fold-standardized FeatureVectorV1; value channels and paired 0/1 validity
  channels both enter the model tensor, with row_mask=false right padding.

## 4. Public facade / 公共入口

```python
from ppg_frailty.signal import (
    build_signal_views, evaluate_quality, detect_pulses, compute_prv,
    extract_direct_features,
)
from ppg_frailty.artifacts import get_reducer, run_artifact_route
from ppg_frailty.features import build_feature_vector, build_ordered_matrix
```

## 5. Quantitative synthetic evidence / 合成量化证据

### 5.1 EKF versus LPF / EKF 与 LPF

Fixture: 20 s, true gravity `[0,0,9.80665]`, x-axis 0.2 Hz translation burst
from 5–15 s, amplitude 2 m/s², zero gyro. Metric uses active valid samples.

| Route | RMSE (m/s²) | MAE (m/s²) | Active valid fraction |
|---|---:|---:|---:|
| no-precalibration quaternion MEKF | 0.607805 | 0.498501 | 1.000000 |
| causal LPF 0.3 Hz | 1.232806 | 1.086527 | 1.000000 |

This fixture supports the frozen decision to keep EKF primary and LPF comparator;
it does not establish subject-level superiority.

### 5.2 Artifact comparison / 伪影对照

Fixture: 8 s, 72 bpm cardiac sine plus shared 2.3/0.7 Hz motion mixture in two
wavelengths with aligned IMU. Welch resolution makes the best dominant-HR absolute
error 1.6875 bpm for every successful route. Values are implementation smoke
evidence only; they must not be used to rank routes clinically.

| Route | Status | Confidence | Best |corr(cardiac)| | Mean |corr(motion)| | Cardiac-band concentration |
|---|---|---:|---:|---:|---:|
| identity | success | 1.000000 | 0.805586 | 0.620815 | 0.999640 |
| NLMS | success | 0.682538 | 0.723476 | 0.014444 | 0.568320 |
| SSA | success | 0.997224 | 0.805587 | 0.620817 | 0.999640 |
| spectral_mask | success | 0.973494 | 0.815796 | 0.554775 | 0.999411 |
| PCA | success | 0.992294 | 0.861163 | 0.489945 | 0.999735 |
| FastICA | success | 0.999923 | 0.999900 | 0.007078 | 0.999996 |
| NMF | success | 1.000000 | 0.774354 | 0.627682 | 0.998869 |

## 6. Verification / 验证

Focused standard-library unittest results after final patches:

- signal: 22/22 passed;
- artifacts: 11/11 passed;
- features: 6/6 passed.

Package-qualified full-tree command (required because bare discovery can collide
with Python's standard-library module named `signal`):

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=final_v0/final_pipeline_v1/src \
python3 -m unittest discover -s final_v0/final_pipeline_v1/tests \
  -t final_v0/final_pipeline_v1 -p 'test_*.py' -q
```

Final result after the central formal artifact-adapter correction: 112 tests,
112 passed, 0 failures, 0 errors, 13.631 s. Tests cover prefix/chunk parity,
EKF-vs-LPF, explicit gap
configuration, route guards, analytic optical formulas, PRV continuity and role
eligibility, calibrator leakage isolation, every reducer, strict formal spectral
YAML, single-channel BSS failure, SSA threshold failure, IMU mask propagation,
truthful feature missingness, registry fields/hash, allowlist rejection, and K=32
validity channels.

Formal CLI configuration validation:

```text
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=final_v0/final_pipeline_v1/src \
python3 -m ppg_frailty.cli validate --all-configs
```

Result: all four resolved configs passed. In particular,
`motion_benchmark_v1.yaml` resolved declared/runtime reducer
`spectral_mask -> spectral_mask`, declared/runtime version
`spectral_mask_v1 -> spectral_mask_v1`, and preserved all five formal parameters.
The central `module_registry.py` SHA-256 at this checkpoint is
`b299f742d6372b98a09ba228f139050156232f21359965ac97d59fd13f4f66d2`.

## 7. Current source identities / 当前源码身份

| Layer | File | SHA-256 |
|---|---|---|
| signal | `imu.py` | `fc0f3ca865b676e10fd650d109df7465026e39b35215d3fa7b5965139134544a` |
| signal | `views.py` | `96f8f2c8f45e32a4cb31acdf7c0dad43f4ac17c98b02db8962edc231ed4cccc2` |
| signal | `preprocess.py` | `69b9f6ae6d6d5f5dd006af278d40fb6c92417f20cdb403996d67cdf550612179` |
| signal | `peaks.py` | `009c69948fa83edd1b88efa616bb1f2ca1a0f40ec9f0fc4232dbb19635cf9b30` |
| signal | `prv.py` | `8b2714e93ae3595319feea8469df2648f705ea3e045480fc4860524263a8bb9b` |
| signal | `morphology.py` | `abc2bd83cf85415a5c031951d23a293fb92f01714562934aea8c31b3d84e5906` |
| signal | `optical.py` | `f030c202ae99e6f1b11465d5a0ca23c1cbf542f28b399d5ac72e486b6d8d5950` |
| signal | `sqi.py` | `4a600b875bdbfbf9d0e69d1cfbacc8bfb3e901782d0daf0b77fd7ce255673043` |
| artifacts | `base.py` | `c440f05b13b2f3cfbf233c7e203eb1b84525d2b6cbbbc3ded8f61ab689b67b95` |
| artifacts | `nlms.py` | `93421abed9beb78e8bb30f41f9ec0ca95991e1b2828ab55ba19e06665837bd97` |
| artifacts | `decomposition.py` | `83a0d0d020bfea35b79fe9f0495899d843787da4ce26d175da8c8441fe7e9e7f` |
| artifacts | `spectral.py` | `02e25bf600a8013405a60b1b215ebbff27f3979e266d73230a99eb9704303649` |
| artifacts | `bss.py` | `fe4d0cc35fe118eb649564386f34bf4e07253d6a8b61c39602187f81e9d13169` |
| artifacts | `router.py` | `9b6f2aceccfb27ca2c05d084d13f824ed3f51f2afa070bd95023288e826245af` |
| features | `engineering.py` | `f0650ef753b8311f0ee633c062685792f84426561952ecad7517d17f74c1cfa4` |
| features | `registry.py` | `8af704b840cbaa462aa9e6cc804ebc1c7d548450931e234b3ec779fb5c49064d` |

## 8. Known limitations and next evidence / 已知限制与后续证据

1. No 29-subject or external heartbeat benchmark has been run in this phase.
   Synthetic metrics validate code paths and formulas only.
2. The empirical SQI calibrator implementation is leakage-guarded, but no
   production fitted calibrator artifact is frozen yet; formal empirical runs
   correctly fail without one.
3. PTT pleth wavelength identity remains unresolved by the data contract;
   dual-wavelength optical/BSS must not run on those channels until confirmed.
4. The learned/ONNX denoiser stays explicitly unsupported because no audited
   model artifact is registered. It is not simulated by another reducer.
5. SSA's full CPU SVD and FastICA/NMF are deterministic references, not yet
   optimized for the proposed embedded hardware.
6. The formal spectral preserve-band and confidence definitions are implemented
   and tested, but their thresholds require the frozen grouped benchmark before
   scientific selection.
7. K=32 construction requires callers to provide a complete fold-standardized
   FeatureVectorV1 and fold-transformed engineering rows. The training pipeline
   must persist those fold-local transform artifacts and membership provenance.
8. Non-identity amplitude/morphology remains unavailable by design. Any future
   morphology-preserving learned model needs a separate reviewed contract and
   cannot silently reuse the rate-only route.

---

# Phase 07 — Models, frozen-fold training, OOF evaluation, and bundles

## English report

### Scope and source review

This phase re-scanned the locked implementation specification, the V1 contracts,
ADR-006/007/008/011, the resolved configuration names, and the reviewed historical
model definitions.  It wrote only inside `final_v0/final_pipeline_v1`.  Historical
root code and `_agent` remained read-only.  Data ingestion, signal processing,
artifact reduction, SQI, and feature extraction were deliberately outside this
phase.

### Implemented model layer

- Added the reviewed `CompactCNN1D` with exactly 79,139 trainable parameters for
  eight channels and three classes.  The implementation preserves the reviewed
  32/64/128 channel widths, 9/9/7 kernels, pool-by-four stages, dropout schedule,
  global pooling, and three-class head.
- Added reviewed full and small InceptionTime single networks.  Exact parameter
  counts are 456,579 and 57,027 for eight channels and three classes.  Full uses
  depth six and 32 filters; small uses depth three and 16 filters.  Both use
  mask-aware pooling, and padded columns are zeroed before convolution.
- Added an optional five-member InceptionTime probability ensemble.  It requires
  exactly five distinct seeds, rejects shared parameter objects, trains members
  independently, and averages probabilities rather than logits.
- Added self-contained NumPy ROCKET plus ridge.  The reference factory route is
  restricted to `feature_matrix`, defaults to 10,000 kernels, fits robust scaling,
  kernels, and ridge state on the outer-training partition, and treats the matrix
  mask as invalid support.  `MiniROCKET` is an explicitly named engineering
  ablation and does not claim reference-algorithm parity.
- Added fold-local logistic-regression-L2, RBF-SVM, and ExtraTrees feature-vector
  baselines with frozen feature names and fitted-object provenance.
- Added a self-contained effect-size shapelet discovery route and experimental
  ShapeFormer.  It imports no external PISD implementation or machine-specific
  path and is explicitly marked experimental.
- Added corrected `FileBagFusionClassifier`: signal windows are encoded and pooled
  first, and the file feature vector is encoded exactly once per file afterward.
  Padded bag slots never reach the signal encoder.
- Added one canonical model-name registry.  Human specification names such as
  `CompactCNN1D`, `InceptionTimeFull`, `InceptionTimeMatrix`, and
  `LogisticRegressionL2` map to stable machine IDs.  Unknown aliases fail closed;
  bundles record both names.

### Implemented training and evaluation layer

- Added typed datasets for all four representation modes: `raw`,
  `feature_vector`, `feature_matrix`, and `fusion`.  The matrix dataset includes a
  strict adapter from schema-compatible `OrderedFeatureMatrixV1` objects.
- Added `FrozenOuterSplit` and `UnifiedTrainer`.  The fit API has no outer-OOF
  labels or dataset argument.  The primary rule trains a pre-registered fixed
  number of epochs.  The optional inner-grouped rule selects an epoch only inside
  outer training, discards the selection model, creates a fresh model, and refits
  on all outer-training rows.
- Added independently auditable provenance for every fitted scaler, imputer,
  ROCKET transform, classifier, shapelet bank, and network state.  Learned state,
  exact participant membership, registry hash, and fold hash are bound to the fit.
- Added prediction-only evaluation, finite probability validation, balanced
  accuracy, macro-F1, multiclass log loss, and confusion matrices.
- The trainer consumes the resolved YAML training block without ignored fields.
  Canonical fixed_epoch, Adam, cross-entropy, deterministic algorithms,
  participant-to-file-to-window balanced sampling, and outer-training
  participant inverse-frequency class weights are executed exactly.  The
  historical fixed spelling is accepted only as a recorded legacy alias.
  Classical and ROCKET fits receive the equivalent combined sample weights.
- PyTorch import is lazy at the training boundary.  Without the optional deep
  dependency, estimator datasets, OOF/evaluation utilities, bundles, and
  UnifiedTrainer.fit_estimator remain usable; only tensor/deep calls fail with a
  focused dependency message.
- Added strict OOF rows and an atomic Parquet writer.  `pyarrow` is optional; when
  it is unavailable the writer fails closed and never falls back to CSV.
- Added the frozen aggregation hierarchy
  `window -> file -> role -> participant`.  Windows, files, and roles are averaged
  at their direct parent level; SQI weighting is an explicit ablation only.
- Added one-factor ablation execution and paired subject-delta comparison APIs.
- Added integrity-checked model bundles for torch and estimator models.  Bundles
  contain model/input configuration, both canonical and machine model IDs,
  provenance metadata, optional fitted transforms, payload hashes, and golden
  inputs/probabilities.  Loading rejects missing, modified, or unexpected files;
  saving reloads the bundle and enforces golden parity.

### Stable public facade

- `ppg_frailty.models.create_model(model_config, input_spec)`
- `ppg_frailty.training.UnifiedTrainer`
- `ppg_frailty.training.FrozenOuterSplit`
- `ppg_frailty.training.evaluate_predictions`
- `ppg_frailty.training.aggregate_hierarchy`
- `ppg_frailty.training.OofWriter`
- `ppg_frailty.training.run_ablation_matrix`
- `ppg_frailty.training.save_bundle`
- `ppg_frailty.training.load_bundle`
- `ppg_frailty.training.predict_bundle`

### Verification

- AST parse: 68 V1 Python files parsed successfully at the time of the check.
- Model tests: 10/10 passed with standard-library `unittest`.
- Training/OOF/bundle tests: 13/13 passed with standard-library `unittest`,
  including an actual-reference-YAML constructor test and a subprocess that
  blocks every torch import while fitting an estimator successfully.
- Exact architecture parameter tests passed: 79,139 / 456,579 / 57,027.
- V1 validator passed required paths, spec hash, legacy-import rejection, strict
  JSON, and config loading.  Its bilingual check found two unrelated files owned
  by other phases (`artifacts/__init__.py` and `features/__init__.py`); no
  models/training file failed that check.

### Remaining scientific/runtime decisions

- PyTorch is available in the current environment, but its V1 dependency status
  remains `decision_pending`; runtime availability is not treated as approval.
- ShapeFormer effect-size discovery remains an experimental comparison route.
- MiniROCKET remains a separately named ablation, not the specified ROCKET route.
- This phase ran deterministic unit/smoke tests only.  Formal five-repeat,
  five-fold candidate reruns, resource profiling at 10,000 ROCKET kernels, and
  locked participant-level benchmarks remain downstream execution work.

## 中文报告

### 范围与源文件复核

本阶段重新扫描了锁定实现规范、V1 contracts、ADR-006/007/008/011、resolved
配置名称和已审查历史模型定义。所有写入严格位于 `final_v0/final_pipeline_v1`；
根目录历史代码与 `_agent` 均保持只读。数据导入、信号处理、伪影削减、SQI 与
特征提取不属于本阶段范围。

### 已实现模型层

- 落地已审查 `CompactCNN1D`；八通道三分类下参数量精确为 79,139。保留
  32/64/128 通道、9/9/7 卷积核、四倍池化、dropout、全局池化和三分类头。
- 落地完整/小型 InceptionTime 单网络；八通道三分类参数量分别精确为 456,579
  和 57,027。完整版本 depth=6/filters=32，小型版本 depth=3/filters=16；补齐列
  在卷积前归零，最终池化显式读取 mask。
- 落地可选五成员 InceptionTime 概率集成：必须恰好五个不同 seed，拒绝共享参数
  对象，成员分别训练，并在概率空间而非 logits 空间平均。
- 落地自足 NumPy ROCKET + ridge。规范 factory 路线只允许 `feature_matrix`，默认
  10,000 个核；稳健缩放、kernel 与 ridge 均只在 outer-training 拟合，mask 定义
  无效支持。`MiniROCKET` 明确为工程消融，不宣称参考算法等价性。
- 落地 feature-vector 的 L2 逻辑回归、RBF-SVM、ExtraTrees，并冻结特征名称与
  各拟合对象 provenance。
- 落地自足效应量 shapelet 发现和实验性 ShapeFormer；不导入外部 PISD 或机器
  特定路径，并明确标为 experimental。
- 修正 FileBag 融合：先编码/汇聚信号窗口，再对每个文件仅编码并拼接一次文件
  特征；补齐袋位置不会进入信号编码器。
- 建立唯一规范名称 registry。`CompactCNN1D`、`InceptionTimeFull`、
  `InceptionTimeMatrix`、`LogisticRegressionL2` 等人类名称映射到稳定 machine ID；
  未知别名关闭失败，bundle 同时记录二者。

### 已实现训练与评估层

- 为 `raw`、`feature_vector`、`feature_matrix`、`fusion` 四种 representation
  建立类型化 dataset；matrix dataset 可从 schema 完全兼容的
  `OrderedFeatureMatrixV1` 严格构造。
- 建立 `FrozenOuterSplit` 与 `UnifiedTrainer`。fit API 不接受 outer-OOF 标签或
  dataset。主协议采用预注册固定 epoch；可选 inner-grouped 路线只在 outer-train
  内选 epoch，随后丢弃选择模型、创建新模型并在完整 outer-train 从头重训。
- 对 scaler、imputer、ROCKET transform、classifier、shapelet bank 与网络权重
  分别记录可审计 provenance，并绑定学习后状态、参与者成员、registry hash 和
  fold hash。
- 建立只预测 evaluator、有限概率校验、BA、macro-F1、多分类 log loss 与混淆矩阵。
- trainer 对 resolved YAML 的 training 区块实行零忽略字段消费：实际执行规范
  fixed_epoch、Adam、cross-entropy、确定性算法、participant→file→window
  平衡抽样和仅基于 outer-training participant 的逆频率类别权重。历史拼写 fixed
  仅作为有记录的 legacy alias；经典模型与 ROCKET 使用等价组合 sample weight。
- torch 在训练边界延迟导入。缺少可选深度依赖时，estimator dataset、OOF/评估、
  bundle 与 UnifiedTrainer.fit_estimator 仍可使用；只有张量/深度调用会给出明确
  依赖错误。
- 建立严格 OOF 行与原子 Parquet 写入器；缺少可选 `pyarrow` 时关闭失败，禁止
  静默回退 CSV。
- 固定唯一聚合层级 `window -> file -> role -> participant`；每层对直接子节点
  等权平均，SQI 加权只能作为显式消融。
- 建立单因素消融和 subject 配对差值 API。
- 建立 torch/estimator 的完整性 bundle：记录模型/输入配置、两种模型名称、
  provenance、可选 fitted transforms、payload hash 与 golden 输入/概率；加载时
  拒绝缺失、被修改或未登记文件，保存后立即重载并校验 golden parity。

### 验证结果与剩余事项

- 检查时 68 个 V1 Python 文件 AST 全部解析成功。
- 模型测试 10/10 通过；训练/OOF/bundle 测试 13/13 通过，均使用标准库
  `unittest`；后者包含实际 reference YAML 构造测试，以及阻断所有 torch import
  后仍成功拟合 estimator 的独立 subprocess 测试。
- 精确参数量测试 79,139 / 456,579 / 57,027 全部通过。
- V1 validator 的 required paths、spec hash、禁止历史 import、strict JSON、配置
  加载均通过；双语检查仅发现其他阶段的 `artifacts/__init__.py` 与
  `features/__init__.py`，models/training 本批次文件没有失败。
- 当前环境可用 PyTorch，但 V1 依赖状态仍为 `decision_pending`；ShapeFormer
  仍是实验对照；MiniROCKET 仍是具名单独消融。
- 本阶段只运行确定性 unit/smoke test。正式 5 repeats × 5 folds 全候选重跑、
  10,000 核 ROCKET 资源画像和锁定 participant-level benchmark 属于后续执行。

---

# Phase 08a — Specification/TODO comparison reports / 规范与 TODO 对照报告

- Re-read the full TODO, M0–M3 current-status documents, locked specification, and live implementation audits before writing.
- Added a requirement-level spec-vs-TODO overlap/difference/contradiction report.
- Added a milestone-level spec-vs-completed-M0–M3 overlap/gap/conflict report.
- Kept direct user authorization separate from instructions contained in the attached implementation document.

---

# Phase 08b — Remaining user-requested reports / 其余用户要求报告

- Re-scanned relevant local source, frozen M0–M3 status, audits, and current V1 interfaces before writing.
- Added the workflow-ordered local/frozen-code reuse/change matrix.
- Added an algorithm reasonableness assessment with benefits, costs, scientific limitations, and interpretation rules.
- Added a concise 27-point V1→V2 confirmation table linked to the detailed registry.

---

# Phase 09a — executable baseline regression gate / 可执行基线回归门

- Status / 状态：implemented; validation pending the next full-suite run.
- Scope / 范围：only `final_pipeline_v1/tests/audit/`; no historical source was changed.
- Process / 流程：the test reloads both audit JSON files, streams every registered root
  source as bytes, recomputes size/SHA-256, checks the 261-record/29-participant cohort,
  and verifies all historical scores remain explicitly ineligible for V1 ranking.
- Algorithm / 算法：content identity uses SHA-256 over ordered byte blocks; scientific
  eligibility is a fail-closed Boolean invariant rather than a score threshold.
- Result / 结果：four deterministic tests were added.  They will be included in the
  final all-suite report; no historical metric was reinterpreted as a V1 result.
- 中文说明：新增四项确定性测试，确保历史源码指纹、M2 roster、模型参数快照和
  历史结果 `eligible=false` 均不可静默漂移；本批次没有改动根目录历史代码。

---

# Phase 09b — generated model cards / 自动生成模型卡

- Status / 状态：implemented; model-suite validation follows in the final gate.
- Scope / 范围：one generator, thirteen model cards, one index, and two registry tests.
- Process / 流程：the generator maps the sole canonical model registry to stable machine
  IDs, eligible representation/signal routes, scientific status, deviations, and limits.
- Algorithm / 算法：coverage is exact set equality between registered machine IDs and card
  filenames; each card is checked for participant-level evaluation and the explicit absence
  of an independent frailty test.
- Result / 结果：all 13 registered routes now have a machine-traceable card.  The cards make
  no performance claim and distinguish single networks, ensembles, project deviations,
  experimental ShapeFormer, and the named MiniROCKET ablation.
- 中文说明：13 个模型路线均有生成式模型卡；模型卡明确 OOF 而非独立测试，并
  明确原论文偏离与尚未运行的计算预算，防止命名越界。

---

# Phase 09c — executable physical-time ablation / 可执行物理时间消融

- Status / 状态：implemented and tested.
- Scope / 范围：`models/time_scale.py`, optional CompactCNN/Inception constructor
  controls, and four model tests; default model identities remain unchanged.
- Process / 流程：declared physical kernel durations are converted to nearest odd
  samples for DL-only grids 100/160/200/400 Hz; 5/10 s windows, dilation 1/2 and all
  four representation labels are materialized as 64 auditable candidate conditions.
- Algorithm / 算法：ties in odd-sample conversion choose the larger kernel; the longest
  Inception branch uses `1 + depth*dilation*(max_kernel-1)` samples.  At 400 Hz the
  reference remains 39/19/9 samples and 229 samples = 0.5725 s receptive field.
- Result / 结果：model suite 16/16 passed.  Real CompactCNN and Inception forward calls
  succeeded for seconds-derived kernels with dilation 2; reference parameter counts stayed
  79,139 / 456,579 / 57,027.
- 中文说明：物理时间消融不改变 400 Hz acquisition/feature 网格，也不把 64 个候选
  视为已完成全量训练；正式比较仍须复用相同 folds/seeds 并报告指标、覆盖率、时间和内存。

---

# Phase 09c — training, evaluation, OOF and bundle protocol / 训练评估与部署协议

- Status / 状态：implemented; the isolated training suite passes 28/28 tests.
- Scope / 范围：training implementation, training tests, and this immutable phase entry;
  signal, artifact, feature, data and CLI implementations were not changed by this phase.
- Process / 流程：aggregation identity was expanded before any filtering; rejected rows
  remain explicit coverage denominators; final fits require the exact frozen train roster;
  optional content binding detects data hidden under false participant IDs.
- Algorithm / 算法：inner epoch selection now applies equal-weight
  window→file→role→participant aggregation. Evaluation reports per-class metrics,
  worst-class metrics, multiclass Brier, top-label equal-width ECE, coverage, repeat
  population/sample SD, Student-t 95% CI, and paired repeat/fold/seed deltas.
- OOF / 折外预测：the formal validator checks the exact frozen
  repeat×fold×seed×config×participant Cartesian product, complete trace fields, explicit
  rejected rows, and exact ensemble member indices 0..4.
- Bundle / 部署包：metadata is fail-closed against the complete §5.14 schema. State,
  transforms, golden arrays and the optional raw-record adapter are hashed in a
  same-filesystem staging directory; golden parity is checked before one atomic directory
  rename exposes the target. Load rejects stale expected schemas and unverified entries.
- Result / 结果：28/28 training tests pass, including identity-mix, held-out mutation,
  exact-roster, drop coverage, metric formula, failed-staging cleanup, raw adapter,
  stale-schema, and 10,000 lightweight load/predict rounds without repeated save.
- 中文说明：本阶段关闭了 §5.12–§5.14 的主要协议缺口；任何 drop 均不再从
  coverage 分母中消失，任何不同实验身份均不得混合，正式 OOF 与 bundle 均采用
  显式完整性门禁。本结果是软件协议测试，不是 29-subject 科学 benchmark。

---

# Phase 09d — canonical training facade parity / canonical 训练门面一致性

- Status / 状态：implemented; the expanded training suite passes 31/31 tests.
- Scope / 范围：three parity tests were added after the phase09c protocol gate.
- Process / 流程：canonical singular paths for aggregation, metrics, OOF and bundle were
  imported through their public APIs and compared with the plural training authorities.
- Result / 结果：drop coverage, metric formulas, exact OOF roster validation and the
  §5.14 required metadata set now have one implementation authority. Formal bundle export
  additionally requires the caller to state strict_metadata=True.
- 中文说明：新增三项门面一致性测试；canonical 单数路径不再维护第二套公式或
  metadata schema。当前 training 相关测试总计 31/31 通过。

---

# Phase 10 — Strict acceptance and CPU CI / 严格验收与纯 CPU CI

- Date / 日期：2026-08-15
- Status / 状态：`complete`, strict gate `16/16`, CPU tests `146/146`
- Scope / 范围：only V1 acceptance tools, acceptance tests, acceptance artifacts, and this immutable log
- Scientific claim / 科学声明：none; all comparator outputs are synthetic contract evidence, not frailty or external PTT performance

## Process / 流程

1. Re-read specification sections 4, 8, and 10, `AGENTS.md`, `_agent/WRITE_RULES.md`, Git status, and the current V1 tree before implementation.
2. Converted canonical paths, typed containers, public facades, manifest/fold identities, four formal configs, model cards, tests, and scientific-claim rules into 16 fail-closed machine checks.
3. Added self-negative tests proving that missing target files, changed specification bytes, JSON `NaN`, pass-only functions, and unsupported metric claims fail.
4. Added a frozen ECG-like event fixture with one-to-one matching, event P/R/F1, timing error, HR MAE/RMSE/bias, PPI MAE, coverage, and symmetric raw/quality/reducer schemas.
5. Added deterministic label-shuffle sanity and a real 10,000-kernel ROCKET fit/joblib-load parity test. The ROCKET test checks 20,000 transform values, probability parity, and outer-training-only fitted participant IDs.
6. Ran full CPU CI from a clean temporary working directory with `PYTHONWARNINGS=error`, bytecode disabled, CUDA hidden, and bounded CPU thread counts.
7. Exercised 106 package imports, 24 registered modules, all four full config preflights, a real frozen-record/fold smoke, 2 artifact controls plus 7 reducers, 13 model routes, 4 DL sample rates, 5/10-second raw windows, and the 64-case physical-time grid contract.
8. Re-ran strict acceptance after all evidence existed; no failed or pending item remained.

中文说明：全过程将“代码/接口可运行”“synthetic 定量合约”“真实科学 benchmark”三层证据严格分开。缺少结果时不会补造指标；历史或 synthetic 数字必须有不可用于独立测试/真实 benchmark 的明确范围声明。

## Algorithm and contract results / 算法与合同结果

- Canonical boundary / 规范边界：68 required files present.
- Specification identity / 规范身份：41,122 bytes, 766 lines, SHA-256 `cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000`.
- Data protocol / 数据协议：261 records, 29 participants, 9 roles; corrected grouped 5×5 folds with seeds `42,10042,20042,30042,40042`; three materialized hashes byte-exact.
- Configs / 配置：4/4 real full preflight, each resolving all 25 held-out splits.
- Registry / 注册表：4 representations + 7 artifact reducers + 13 models = 24 modules; 15 canonical facade imports.
- Types / 类型：8 required dataclass contracts including explicit `q_morph=not_applicable` semantics.
- Code checks / 代码检查：152 bilingual AST files parsed; 116 active source/tool files contain no legacy runtime import or AST-level unfinished implementation.
- Tests / 测试：146 run, 146 passed, 0 failed, 0 errors, 0 skipped, warnings treated as errors.
- Strict acceptance / 严格验收：16 passed, 0 failed, 0 pending.
- CPU CI / 纯 CPU CI：all stages passed; no unexpected warning.

## Machine evidence / 机器证据

- `artifacts/acceptance/cpu_ci_current.json`
  - SHA-256: `51f5fb2e2859bf6cec631b54cdf8fdb49ac63a84d2c8a1860c8d0512b26d5193`
- `artifacts/acceptance/cpu_ci_tests_current.json`
  - SHA-256: `5efde787ec68ded03bcecd615c7c8da757265441331d9e449866553ab1896e23`
- `artifacts/acceptance/strict_acceptance_current.json`
  - SHA-256: `19fc5b6883e006141b5a65a71fca2d5d1f6a73ec1c6b22acc858c271a3573174`

The test report embeds the path/byte/SHA-256 identity of all test sources; editing a test invalidates the green report. Quantitative artifacts are preserved under `artifacts/acceptance/runs/` and distinguish raw/no-denoise, quality-only, and non-identity reducers.

## Known boundary / 已知边界

This gate proves implementation, isolation, serialization, and synthetic quantitative contracts. It does not claim that the complete corrected 5×5 frailty benchmark or a real external ECG/PTT benchmark has been executed. Those scientific runs must emit separate provenance-complete OOF/benchmark artifacts.

---

# Phase 11 — ShapeFormer §6.1 strict repair / ShapeFormer §6.1 严格修复

- Status / 状态：implemented and CPU-regression tested; experimental status retained.
- Scope / 范围：`models/shapeformer.py`, strict model factory, model tests,
  generated ShapeFormer model card and its single generation authority.
- Scientific claim boundary / 科学声明边界：this is an implementation and protocol
  repair only. It is not PISD/original ShapeFormer parity and contains no frailty
  performance claim；本阶段只证明接口与算法约束落地，不提供衰弱分类性能结论。

## Process / 流程

1. Re-read merged specification §5.6 and §6.1 and compared every requirement with
   the existing self-contained effect-size implementation.
   重新逐条核对规格，确认旧实现虽然自足但缺少 attention 前的 patch/downsample，
   且 discovery、物理时间与 outer-fold 身份尚未成为必填契约。
2. Preserved the explicitly named local `effect_size_shapelets_v1` discovery
   method and rejected PISD/unknown values instead of falling back.
   保留具名效应量发现方法；PISD 或未知方法均关闭失败，不发生静默替换。
3. Bound every fitted bank to sorted outer-train participant IDs, a SHA-256 roster
   hash, repeat/fold indices, input sampling rate, and shapelet length in samples
   and seconds.
   每个拟合库绑定 outer-train 名单、名单哈希、repeat/fold、采样率以及样本/秒双尺度。
4. Replaced raw/local encoder use before classification with non-overlapping Conv1d
   patch embedding, deterministic positional encoding, mask-aware Transformer
   attention and mask-aware patch pooling. `patch_size_samples < 2` is rejected.
   分类前路线改为非重叠 patch embedding→位置编码→掩码 Transformer→掩码池化；
   从结构上拒绝 400 Hz 原始采样点直接作为通用注意力 token。
5. Kept trainable shapelet-distance features as a parallel experimental branch and
   applied the same validity mask before both patch and shapelet computations.
   保留可训练 shapelet 距离实验分支，并在两个分支前统一清除无效补齐值。
6. Tightened the factory: discovery method, input sampling rate, outer repeat/fold
   and outer-train roster hash must all match the fitted bank.
   工厂要求上述身份与拟合库逐项一致，并验证完整 mapping 恢复路径。
7. Regenerated all cards from the unique generator; only ShapeFormer semantic
   content changed.
   通过唯一生成器更新模型卡；语义变化仅涉及 ShapeFormer 身份与限制说明。

## Algorithm / 算法

For input `x[B,C,T]`, the reference experimental route uses non-overlapping patches
of `P>=2` samples. A Conv1d patch projection produces
`N=floor(T/P)` tokens, sinusoidal position encodings preserve order, and generic
self-attention receives a key-padding mask constructed from fully valid input
patches. Masked mean pooling yields one patch embedding. In parallel, each fitted
shapelet computes the negative minimum squared distance over fully valid sliding
windows. The two file/window embeddings are concatenated before the classifier.

对输入 `x[B,C,T]`，先以 `P>=2` 做非重叠 patch 投影，得到
`N=floor(T/P)` 个 token；确定性位置编码保留时序，完整有效 patch 构成注意力
padding mask，最后做掩码均值池化。并行 shapelet 分支只在完整有效滑窗上计算
负最小平方距离；二者拼接后进入分类器。该结构明确不是 raw sample-token attention。

## Results / 结果

- Focused model suite: **19/19 passed**, zero failures/errors/skips.
- Full V1 CPU suite at this checkpoint: **149/149 passed**, zero
  failures/errors/skips.
- Tested boundaries: patch construction, output shape, invalid-tail invariance,
  explicit discovery selection, PISD rejection, physical-time equality,
  outer repeat/fold mismatch, roster-hash mismatch, mapping restoration, generated
  card identity, and unchanged CompactCNN/Inception parameter snapshots.
- No independent test or corrected 5×5 scientific benchmark was run in this phase;
  `independent_test=false` remains unchanged.

## Review / 自审

- No external path/package was introduced; effect-size discovery remains local.
- No outer-held-out labels enter discovery or factory construction.
- Generic self-attention receives patch tokens only; `patch_size=1` fails closed.
- Invalid tail values were changed by four orders of magnitude in test and produced
  identical masked logits within `atol=1e-6, rtol=0`.
- Potential limitation: non-overlapping patch projection discards an incomplete
  terminal patch. This is deterministic and mask-safe but remains an experimental
  architectural choice for matched benchmark evaluation.
- Potential limitation / 局限：本地效应量发现并非 PISD；因此模型继续标记为
  `experimental_ineligible_for_parity_claim`，不得据此宣称原方法复现。

---

# Phase 12 — Documentation acceptance handoff / 文档验收交接

Date / 日期: 2026-08-15
Status / 状态: documentation frozen for final machine acceptance / 文档已冻结，等待最终机器验收
Scope / 范围: documentation and one dependency-envelope correction inside `final_v0`; no `_agent` write.

## Outcome / 结果

V1 now has a bilingual status authority, an operational runbook, complete M0–M3/V1
navigation, evidence-bound comparison reports, and a 28-point V2 confirmation registry.
Every statement separates engineering implementation, real-input integration, real
single-fold training smoke, synthetic contract comparison, and unexecuted scientific
benchmark evidence.

V1 已具备中英双语状态权威、可复制运行手册、M0–M3/V1 总导航、绑定机器证据的五份
对照报告及 28 项 V2 人工确认清单。全部文档严格区分工程实现、真实输入集成、真实单折
训练 smoke、合成合同对照与尚未执行的科学 benchmark。

## Files and navigation / 文件与导航

- `STATUS.md`: two-axis engineering/scientific status, implementation workflow,
  evidence taxonomy, limitations, and live acceptance pointers.
- `RUNBOOK.md`: dependency profiles, public CLI commands, validation/CI, data
  materialization, real protocol smoke, all artifact/model comparisons, EKF/LPF,
  ablations, and real outer-fold commands.
- `README.md`: concise status, frozen boundaries, quick start, public evidence
  scope and report navigation.
- `final_v0/README.md`: M0, M1, M2, M3 and final_pipeline_v1 entry map.
- `docs/comparisons/01..05`: the five user-requested specification/TODO/completed
  work/local workflow/algorithm/V2 reports.
- `records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md`: 28 decision IDs and
  confirmed/partial/pending semantics.
- `records/v2_decision_points/INITIAL_CONSERVATIVE_DEFAULTS.md`: retained only as
  superseded provenance, never as the live decision authority.

## Evidence binding / 证据绑定

1. Engineering acceptance points to the current strict-acceptance and CPU-CI JSON
   rather than freezing a prose test count or source-tree hash.
2. Real input/protocol smoke is linked separately and correctly states that it emits
   no trained prediction metric.
3. The real 60-second r0/f0 training authority is selected by
   `artifacts/experiments/reference_registry.json` and points to
   `reduced_real_r0_f0_reference_width_preserved_v2`.
4. The current real smoke retained 5/6 OOF participants, coverage 0.8333 and BA 0.5;
   every document labels it `smoke_not_scientific_benchmark`, not a final score.
5. The 12-second run remains visible as structured fail-closed evidence. Its manifest
   stores the 23-member zero-retained outer-train list and empty OOF; the separate
   29/29 post-Q_rate drop diagnostic is documented as an observation, not a manifest field.
6. Synthetic artifact/model/gravity/physical-time evidence is explicitly excluded
   from Frailty3, external-PTT, clinical or deployment ranking claims.
7. The old pre-width-preservation 60-second directory remains immutable but is
   marked superseded; hand-written navigation links only the registry and current v2 result.

## Runner and algorithm boundary / Runner 与算法边界

- `run` is an input/protocol audit; `run-experiment` is the real
  training/evaluation entry.
- Passing reduced command uses `configs/motion_benchmark_v1.yaml`, not the raw
  reference-static configuration.
- Public reduced defaults are fixed at 60 seconds, one file per participant and one
  epoch-equivalent while preserving the complete frozen roster.
- Full with no repeat/fold requests all 25 cells; an explicit pair requests one
  full-length cell and remains incomplete-5×5 scope.
- The current formal cell executor is **feature_vector-only**. Raw, matrix and fusion
  modules have construction/forward/training contract coverage but fail closed in the
  scientific runner; documentation does not hide this gap.
- ShapeFormer is described as patch/downsample → mask-aware Transformer plus a
  parallel shapelet-distance branch, with outer-train roster/repeat/fold/time binding.
  It remains effect-size experimental and not PISD/original-paper parity.
- Direct 0.2–8 Hz versus historical 0.4–8 Hz is explicit pending point V2-028.

## Human decisions / 人工决定

- V2-007 is confirmed: online no-precalibration quaternion error-state EKF primary
  with causal 0.3 Hz LPF mandatory comparator.
- V2-006, V2-008 and V2-011 are partially confirmed: signal recommendation subset,
  strict PRV behavior, and SQI-first run-locked drop XOR rate-recovery order are
  frozen while their listed deployment/device remainders stay open.
- V2-015 is partially confirmed: NumPy, SciPy, scikit-learn and ONNX Runtime are
  user-authorized; the remaining dependency profiles require a formal decision.
- Every other ID remains pending unless its detailed entry explicitly says otherwise.

## Dependency envelope correction / 依赖边界修正

`pandas` moved from core dependencies to optional
`tabular = ["pandas>=2.3,<3"]`. Source has no pandas runtime import. This removes
the prior documentation/package contradiction without installing or authorizing
additional dependencies.

## Verification and freeze rule / 验证与冻结规则

- Markdown relative links and referenced local paths are checked after the final save.
- `pyproject.toml` is parsed with the standard-library TOML parser and dependency
  membership is asserted.
- V1 and global tracking/index generators are rerun after every logical write batch.
- Final test totals, source snapshots and hashes remain machine-authoritative in
  current acceptance artifacts; this log does not hard-code a value that can stale.
- After this entry and its tracking synchronization, documentation is frozen. Any
  later source change requires a new acceptance refresh and an explicit documentation
  audit, not an undocumented edit.

## Known incomplete scientific work / 尚未完成的科学工作

- Complete 5 repeats × 5 folds candidate reruns have not been executed.
- Raw, matrix and fusion do not yet share the real formal experiment executor.
- No independent Frailty3 test cohort exists.
- External PTT reducer ranking, 29-subject motion-detector retraining, recovery/hierarchy
  routes, full mobile/ONNX parity and target-device measurements remain future work.

---

# Phase 12 — Real reduced current acceptance / 真实 reduced current 验收

- Date / 日期：2026-08-15
- Status / 状态：`complete`
- CPU CI / 纯 CPU CI：10/10 stages passed
- Strict gate / 严格门禁：18/18 passed, 0 pending, 0 failed
- Tests / 测试：159/159 passed, 0 failed, 0 errors, 0 skipped
- Scientific scope / 科学范围：real reduced smoke plus synthetic contracts; not a 5×5 frailty benchmark and not an external PTT benchmark

## Process / 流程

1. Froze the documentation tree before generating current hashes. 文档树冻结后才生成 current hash，避免报告与最终文档版本错位。
2. Extended `tools/run_cpu_ci.py` with a mandatory public-CLI execution of `motion_benchmark_v1`, repeat 0, fold 0, reduced-smoke budget. CPU CI uses `PYTHONWARNINGS=error`, disables bytecode/CUDA, limits CPU thread pools, and runs from an isolated temporary working directory.
3. Added a deterministic active-source snapshot over `src/**/*.py`, `tools/**/*.py`, `tests/**/*.py`, registered YAML configs and dependency metadata. The strict gate recomputes the same canonical path/byte/SHA-256 tree and rejects stale evidence.
4. Added a strict real-experiment validator that reads the materialized fold registry rather than regenerating folds. It verifies the exact 23-participant outer-train roster and exact six-participant r0/f0 held-out roster.
5. Validated all eight fixed experiment artifacts, exact file and participant OOF schemas, exact-once held-out rows, complete trace hashes, explicit drop rows, train-only model/SQI provenance, scientific-empty window/member tables, metrics/count/coverage consistency, and confusion-matrix consistency.
6. Added controlled negative tests proving that a reduced result cannot be relabelled as a 5×5 benchmark and that duplicate held-out subject rows fail the gate.
7. Re-ran strict acceptance after the new CPU report existed, proving that current reports remain recursively auditable and do not self-trigger the scientific-claim detector.

中文算法说明：门禁不比较或固定任何 BA 数值。它只要求指标存在、为有限值且与同一 held-out OOF 和 confusion matrix 内部一致。这样可以检出空结果、错名单、泄漏、重复 OOF、结构漂移与伪造范围声明，同时不会把一次 reduced smoke 变成性能回归阈值或论文结论。

## Results / 结果

- Active source / 活动源码：159 files; tree SHA-256 `d4ccf5a24a4c22ae5e75438d41ef6d1e5125e6b0b55230a18431bbd48fbecc13`.
- Test source / 测试源码：37 files; tree SHA-256 `4747fa8aef8d1f0baad1debca8e01ed5ecb1142247a40d981ed692922cf51d5b`.
- Full tests / 全套测试：159 run, 159 passed, 0 skipped; warnings are errors.
- CPU stages / CI 阶段：10 passed, `failed_stages=[]`.
- Strict checks / 严格检查：18 passed, 0 pending, 0 failed.
- Real current run / 真实 current run：`artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223`.
- Frozen roster / 冻结名单：23 exact outer-train participants; 6 exact held-out participants; every file/subject OOF participant appears exactly once.
- Retained evidence / 保留证据：5 of 6 held-out participants retained; at least one retained result is a gate invariant, but the exact count is not a performance threshold.
- Formal representation boundary / 正式表征边界：only `feature_vector` is supported by the current formal experiment cell executor. `raw`, `feature_matrix`, and `fusion` remain runnable through comparison/tests but fail closed in formal experiment execution.
- Metric threshold policy / 指标阈值策略：no outcome metric value, including BA, is locked by this gate.

## Machine evidence and SHA-256 / 机器证据与哈希

- `artifacts/acceptance/cpu_ci_current.json`  
  SHA-256 `0f2fcdee5096f96a65658fbe607d1f79acc2413ec6a55026ed21938f74a241f3`
- `artifacts/acceptance/cpu_ci_tests_current.json`  
  SHA-256 `819392c2d027802fcb2ff68ed19f92c727bfe26bdde46688145bef2c85a3122f`
- `artifacts/acceptance/strict_acceptance_current.json`  
  SHA-256 `1bab31fce649eda48eeddf25c8cd07fc1bd6ad94202689cb03e9dafce5210464`
- `artifacts/acceptance/source_snapshot_current.json`  
  SHA-256 `c0472928effe9ca4973dad59ba21dd0283eb5848c55a51742cc00222d91e5875`
- `artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/run_manifest.json`  
  SHA-256 `dfdf2d9ce9c68e2f9927b9bfc215b535653dcceeb23179c3c6dc8832143d0e5d`
- `artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/experiment_result.json`  
  SHA-256 `27ca402202d96357558e29dc783e97a4090054c4077629200341511c2eebfe47`

## Known boundary / 已知边界

This phase proves a real, train-only, frozen-roster feature-vector r0/f0 execution and complete current artifacts. It does not prove the unexecuted full 25-cell benchmark, raw/matrix/fusion formal training, external ECG/PTT performance, model superiority, or independent-test performance.

本阶段证明真实、train-only、冻结名单的 feature-vector r0/f0 执行及 current 工件完整性；不证明尚未执行的 25-cell 完整 benchmark、raw/matrix/fusion 正式训练、外部 ECG/PTT 性能、模型优越性或独立测试性能。

---

# Phase 05 — Data and protocol layer / 数据与协议层

- Date / 日期: 2026-08-15
- Status / 状态: completed_and_verified
- Write boundary / 写入边界: final_v0/final_pipeline_v1 only
- Tracking sync / 跟踪同步: intentionally not run; root task performs one merged sync

## 1. Scope / 范围

中文：本阶段把 M2 已审核的数据身份、recording QC、corrected subject folds、
统一切窗和 provenance-bound cache 落地为可运行代码，并物化内部与外部数据合同。
没有重新推断 frailty label，没有调用运行时 SGKF，也没有把任何外部数据称为独立
test。所有 malformed row 均聚合报告后 fail closed，不允许 silent skip。

English: This phase implements the audited M2 data identity, recording QC,
corrected subject folds, unified window planning, and provenance-bound cache.
It materializes both internal and external contracts. Frailty labels are never
re-inferred, no runtime SGKF is invoked, and no external data is called an
independent test. Malformed rows are aggregated and fail closed; silent skip is
forbidden.

## 2. Frozen authorities / 冻结权威输入

| Authority / 权威文件 | Identity / 身份 |
|---|---|
| Implementation specification | cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000 |
| M2 internal file manifest | bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90 |
| M2 dataset version | frailty3_m2_20260815_a054800abda272f6 |
| M2 corrected fold JSON file | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c |
| M2 corrected fold payload | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 |
| M2 corrected fold registry ID | frailty3_future_corrected_sgkf5_v2 |
| M2 active protocol registry | beae2a6922ae0ca840cec1a5c501cde6b6fc029afed16fc798aa2ef8e05fa394 |
| Active protocol ID | frailty3_fixed_epoch_oof_v2_corrected_sgkf |
| M2 external record manifest | 43ab3273346469e9f689ce32da9c5ad280d0a53a8bc8864adf5716f40f9f024e |
| Historical external 15/3/4 split | 6350387f086dfb289b541ff61832572d55a0bc33fa7b6fc0a2428aaec61c687f |

## 3. Implemented code / 已实现代码

### 3.1 Internal manifest / 内部清单

- src/ppg_frailty/data/schema.py
  - exact RED, IR, AX, AY, AZ, GX, GY, GZ schema;
  - exact class names and nine registered roles;
  - strict JSON CSV encoding and typed read-back;
  - machine-readable QC and fold row contracts.
- src/ppg_frailty/data/manifest.py
  - build_internal_manifest(source_csv, output_csv);
  - load_manifest(path) and load_internal_manifest(path);
  - audit_manifest(rows);
  - exact M2 path and SHA enforcement;
  - all 261 source recordings re-hashed before a successful build;
  - 261 records, 29 participants, 9/12/8 class roster, and nine-role coverage
    checked as one indivisible contract.

### 3.2 Recording QC / Recording 级质量控制

- src/ppg_frailty/data/qc.py
  - all thresholds are explicit constructor fields;
  - missing channel, all-nonfinite channel, long gap, insufficient duration,
    flatline, clipping, saturation, implausible scale, timestamp failure, and
    synchrony failure have stable reason codes;
  - parse failure remains a visible failed assessment;
  - unknown channel schemas return missing_required_channel rather than aborting
    through an unknown threshold key;
  - QC is recording eligibility, not SQI, activity, or frailty label.

### 3.3 Frozen folds and evaluation protocol / 冻结折叠与评价协议

- src/ppg_frailty/data/folds.py
  - load_frozen_memberships(json);
  - materialize_fold_csvs(...);
  - FrozenFoldRegistry.from_csv(path) and get_split(repeat_index, fold_index);
  - M2 file SHA and M2 builder-compatible pretty-JSON payload SHA both verified;
  - participant train/OOF disjointness, exact OOF partition, file inheritance,
    all-class presence, and per-class fold spread at most one verified;
  - no splitter is exposed or invoked.
- Primary and repeat protocol:
  - seeds 42, 10042, 20042, 30042, 40042;
  - 5 folds x 5 repeats;
  - subject grouped;
  - fixed epoch;
  - outer OOF invisible during fitting;
  - historical sklearn-1.4.2 membership is reproduction-only.

### 3.4 Unified WindowPlan / 统一切窗

- src/ppg_frailty/data/windows.py
  - WindowPlan.plan(n_samples, fs);
  - explicit physical window/hop, start or end alignment, short-record action,
    padded-tail policy, cap, and cap policy;
  - WindowSlice carries source_record_id, fs, exact sample boundaries,
    valid_length, and padding_mask;
  - uniform-progress cap preserves recording progress rather than using only the
    first K windows;
  - extract_window copies data and applies explicit right padding.

### 3.5 Content-addressed cache / 内容寻址缓存

- src/ppg_frailty/data/cache.py
  - ContentAddressedCache;
  - identity includes source, config, schema, producer, and fold hashes;
  - payload_sha256 is the SHA-256 of raw bytes;
  - metadata and payload tampering fail closed;
  - NPZ is loaded with allow_pickle=False.

### 3.6 External heartbeat/motion contract / 外部 heartbeat-motion 合同

- src/ppg_frailty/data/external_manifest.py
  - imports exactly the 80-row M2 external authority;
  - PTT: 66 included records, 22 subjects, sit/walk/run per subject;
  - SIM: 14 authority rows, exactly 13 included and one excluded;
  - PTT pleth_1..pleth_6 wavelength mapping remains
    unresolved_red_ir_mapping_conflict and is never inferred as RED/IR;
  - PTT single-file SHA and SIM file-snapshot JSON checksum encodings are both
    preserved and validated;
  - all external uses are heartbeat/motion benchmark candidates;
  - independence claim is none_not_an_independent_external_test.

### 3.7 Provisional external grouped folds / 暂定外部分组折叠

- Registry ID: v1_provisional_external_grouped_split_seed42.
- Scope: 22 PTT subjects only; each subject carries sit/walk/run.
- Algorithm: SHA-256 rank of seed 42 plus subject_id, then deterministic
  round-robin assignment to five folds.
- OOF subject counts: fold 0=5, fold 1=5, folds 2/3/4=4.
- Every OOF fold covers sit, walk, and run.
- CSV materializes train and OOF rows; runtime recomputation is false.
- Status: provisional_pending_v2_human_confirmation.
- It is not an independent test split and must be included in V2 human decisions.
- The legacy 15/3/4 split is recorded only as historical and is not active.

## 4. Materializer workflow / 物化流程

tools/materialize_data_contracts.py performs this exact order:

1. Verify the implementation-spec SHA and active M2 protocol registry.
2. Replace prior pass reports with materializing_incomplete_fail_closed state.
3. Import the internal M2 manifest and re-hash all 261 raw source recordings.
4. Import the external M2 manifest without changing source channel semantics.
5. Load and validate the frozen corrected fold JSON; never recompute SGKF.
6. Materialize primary seed-42 and all-five-repeat internal fold CSVs.
7. Materialize the provisional PTT grouped five-fold CSV.
8. Verify and register historical-only fold assets.
9. Write strict JSON audit reports with artifact and producer-source SHA-256.

中文：第 2 步确保若后续失败，旧的 pass 报告会先失效；因此部分生成物不能与
上一次成功报告错误配对。报告中的 artifact SHA 还必须在回读测试中逐项匹配。

English: Step 2 revokes stale success before any later failure can occur, so
partial outputs cannot be paired with a prior pass report. Every artifact digest
is also checked during report read-back tests.

## 5. Generated artifacts and byte identities / 生成物与字节身份

| Artifact / 生成物 | Rows or status / 行数或状态 | SHA-256 |
|---|---:|---|
| manifests/internal_records_v1.csv | 261 data rows | 5b5788fff09910e6c224e2548869f4085fd2bbb480adcc92e0f11b09ee0387ee |
| manifests/external_records_v1.csv | 80 data rows | e6be12bf1578553dccbcc8fa76c2c1e7be47e38b54e3581b6b03dbe9fc4cb7ee |
| splits/sgkf5_v1.csv | 29 participant assignments | 130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284 |
| splits/sgkf5_repeats_v1.csv | 145 participant assignments | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 |
| splits/v1_provisional_external_grouped_split_seed42.csv | 110 fold/subject train-OOF rows | d37926011b61184742d819951329e96f7f87bd34108733fca182a8e08469ec6b |
| reports/data_contract_report.json | pass | 8b58a84d400e4749b474ebbd37e5952a4c647331d5b048ef39a3ad4aafa9df5c |
| reports/external_data_contract_report.json | pass with provisional confirmation pending | 3a424374c727d54bc84061b917cfcfb8e2cc2c07c75cdbbd52803e2d1d45dab2 |

The materializer was executed twice without code or input changes. All seven
artifact SHA-256 values were identical on the second run.

## 6. Verification / 验证

### Syntax / 语法

- python3 -m py_compile on all data modules and the materializer: PASS.

### Standard-library unit tests / 标准库单元测试

Command:

    PYTHONPATH=final_v0/final_pipeline_v1/src +    python3 -m unittest discover -s final_v0/final_pipeline_v1/tests/data -v

Result:

- 17 tests;
- 17 passed;
- 0 failures;
- 0 errors.

Covered behaviors:

- exact internal and external roster;
- no PTT wavelength inference;
- QC pass/failure/parse/unknown-channel behavior;
- frozen registry identity, tamper rejection, and partition resolution;
- WindowPlan alignment, padding mask, short-record policy, and uniform cap;
- raw-byte cache SHA, tamper rejection, and safe NPZ round trip;
- generated CSV typed read-back;
- generated report artifact SHA and byte-size matching.

## 7. Failures found and corrected / 发现并修正的问题

1. Initial external validation assumed every checksum cell was one SHA-256.
   M2 SIM rows actually contain a strict JSON mapping from every snapshot file
   path to its SHA-256. The adapter was corrected to validate and preserve both
   authority encodings without flattening or dropping component hashes.
2. Initial frozen-registry payload verification used the V1 compact JSON hash.
   M2 defines payload identity over sorted, indented strict JSON plus a final
   newline. The loader now reproduces the M2 builder rule exactly while retaining
   the separate byte-exact file SHA check.
3. An unknown channel name could reach a missing threshold key after the schema
   mismatch was already detected. QC now returns the explicit
   missing_required_channel result and keeps the recording visible.
4. The materializer initially could leave a previous pass report during a failed
   rerun. It now writes an incomplete fail-closed state before producing outputs.

## 8. Known boundary and V2 point / 已知边界与 V2 确认点

- The provisional PTT subject five-fold registry requires explicit V2 human
  confirmation before it can become a frozen benchmark protocol.
- Even after confirmation, it remains grouped development/benchmark CV unless a
  separate independence argument is approved; it is not an independent test.
- External dataset component paths are dataset-relative in the M2 record table.
  This V1 run verifies the M2 authority table byte hash and preserves its audited
  component checksum map; it does not invent repository-relative snapshot paths.
- OOF prediction storage and hierarchical window-to-file-to-role-to-participant
  aggregation consume these fold/data contracts in the training layer; they are
  not reimplemented inside the data package.

## 9. Self-review / 自审结论

- No files outside final_v0/final_pipeline_v1 were written.
- AGENTS.md and _agent were not modified.
- No runtime SGKF or legacy split was activated.
- No class label was inferred from a source filename.
- No PTT wavelength was inferred.
- QC has no silent skip path.
- Generated artifacts round-trip and match their report hashes.
- No tracking sync was run in this phase, by explicit parent-task instruction.

---

# Phase 10 — Frozen experiment runner / 冻结实验执行器

Date / 日期: 2026-08-15  
Scientific status / 科学状态: implementation verified; reduced results are smoke only / 实现已验证；reduced 结果仅为 smoke  
Primary module / 主模块: `src/ppg_frailty/experiment.py`

## Outcome / 结果

The V1 package now has a real feature-vector outer-fold runner and an unshortened
multi-cell orchestrator. The public reduced runner preserves the complete frozen
participant roster and executes preprocessing, direct SQI, outer-train-only
empirical SQI calibration, locked quality routing, feature extraction, unified
training and complete retained/drop OOF tracing. It never relaxes SQI or switches
artifact routes after a failure.

V1 现已具备真实 feature-vector outer-fold runner 与不截短的多 cell 调度器。
reduced runner 保留完整冻结 participant roster，并依次执行预处理、direct SQI、
仅 outer-train 拟合的经验 SQI 校准、锁定质量路由、特征提取、统一训练与完整
retained/drop OOF 追踪。任何失败都不会触发 SQI 放宽或伪影路线回退。

## Public API / 公开 API

```python
run_reduced_fold_experiment(
    config_path,
    *,
    repeat_index=0,
    fold_index=0,
    max_seconds_per_record=60.0,
    max_records_per_participant=1,
    fixed_epochs_override=1,
    output_dir=None,
) -> ExperimentResult

run_full_experiment(
    config_path,
    *,
    output_dir,
    repeats=tuple(range(5)),
    folds=tuple(range(5)),
) -> ExperimentResult
```

The reduced default is 60 seconds because the unchanged formal route was measured
at 12 seconds and 60 seconds: 12 seconds retained no participant, while 60 seconds
completed training and produced nonempty OOF. The full runner always uses complete
recordings, all eligible files and the configured epoch rule; it accepts no
shortening or epoch override.

reduced 默认值设为 60 秒，因为在完全相同的正式路由下实测了 12 秒与 60 秒：
12 秒无 participant 保留，60 秒完成训练并产生非空 OOF。full runner 始终使用
完整记录、所有合格文件与配置内 epoch 规则，不接受截短或 epoch override。

## Algorithm and leakage boundary / 算法与防泄漏边界

1. Run the single canonical `preflight_pipeline(..., mode='full')` and load the
   materialized `FrozenFoldRegistry`.
   运行唯一规范 preflight，并加载已物化的冻结折叠注册表。
2. Select records only inside the exact train plus OOF participant roster. The
   reduced cap chooses the longest eligible role recording per participant; it
   never removes a participant.
   仅在精确 train+OOF roster 中选择记录；reduced 每人选择最长合格角色记录，
   但绝不删除 participant。
3. Build synchronized direct signal views. First evaluate base SQI components with
   `fixed_formula_thresholds_v1`.
   构建同步 direct views，先以固定公式计算 SQI 基础分量。
4. Fit empirical SQI quantile bounds using outer-train participant rows only.
   OOF IDs are explicitly checked absent from fitted provenance.
   经验 SQI 分位边界仅由 outer-train participant 拟合，并显式证明无 OOF ID。
5. Evaluate formal direct SQI. High-quality non-motion records return directly.
   Motion-role override is applied before the run-locked `drop XOR reducer`
   branch. Static low-quality records follow the configured locked policy.
   计算正式 direct SQI；高质量非运动记录直接返回。motion override 位于锁定的
   `drop XOR reducer` 之前；静态低质量记录遵守配置锁定策略。
6. A non-identity reducer is accepted only as `ARTIFACT_RATE_ONLY`; post-route
   `Q_morph` must be not-applicable, and only post `Q_rate` may qualify it.
   非恒等 reducer 只能产生 rate-only 路线；post `Q_morph` 必须为 NA，仅
   post `Q_rate` 可决定保留。
7. Extract pulse/PRV, engineering, morphology and dual-optical features according
   to the route. Direct-only morphology/optical fields remain unavailable for
   rate-only records rather than being fabricated.
   按路线提取 pulse/PRV、工程、形态和双光学特征；rate-only 记录的 direct-only
   形态/光学字段保持不可用，不填造数值。
8. Build the canonical feature registry and fit imputation/scaling/model transforms
   inside `UnifiedTrainer.fit_estimator` on the exact outer-train IDs. Outer labels
   are not passed to the trainer.
   使用规范特征注册表，且缺失值填补、缩放与模型变换仅在精确 outer-train ID
   内由统一训练器拟合；outer 标签不传给训练器。
9. Predict OOF and aggregate file → role → participant with the canonical
   equal-weight hierarchy. Every selected OOF file is represented as retained or
   dropped; an all-dropped participant receives an explicit empty-probability trace.
   OOF 按 file → role → participant 等权聚合；每个已选 OOF 文件都以 retained
   或 dropped 出现，全丢 participant 使用显式空概率追踪。

## Fixed artifacts and immutability / 固定产物与不可覆盖

Each reduced run and each full cell writes:

- `run_manifest.json`
- `metrics_per_fold_seed.json`
- `confusion_matrices.json`
- `oof_window_predictions.parquet`
- `oof_file_predictions.parquet`
- `oof_subject_predictions.parquet`
- `oof_member_predictions.parquet`
- `experiment_result.json`

Feature-vector prediction begins at file level, so the window parquet is a
schema-bearing scientific-empty table. The member parquet is similarly marked
`not_an_ensemble_model`. Failed-closed runs write schema-bearing empty OOF tables
and never fabricate metrics. Outputs are staged on the same filesystem, atomically
published, and an existing target is rejected rather than overwritten.

feature-vector 预测从 file level 开始，因此 window parquet 是带 schema 和原因的
科学空表；member parquet 同样明确标为非 ensemble。failed-closed 运行写带
schema 的空 OOF，绝不伪造指标。输出先在同文件系统暂存、再原子发布；已存在
目标会被拒绝，不能覆盖。

## Real frozen-fold evidence / 真实冻结折叠证据

### 12-second gate evidence / 12 秒门禁证据

Persistent output / 持久输出:
`artifacts/experiments/reduced_real_r0_f0_12s_failed_closed`

- Status: `failed_closed`
- Scope: `smoke_not_scientific_benchmark`
- A dedicated diagnostic pass observed 29/29 selected recordings at
  `dropped_post_q_rate` with reason `post_q_rate_below_threshold`.
- The persistent JSON itself records the 23 outer-train participant IDs with zero
  retained files and contains empty OOF parquet tables. The 29/29 distribution is
  a diagnostic observation, not a field currently embedded in that manifest.

- 状态：`failed_closed`
- 范围：`smoke_not_scientific_benchmark`
- 独立诊断观察到 29/29 已选记录均为 `dropped_post_q_rate`，原因均为
  `post_q_rate_below_threshold`。
- 持久 JSON 本身记录 23 名 outer-train participant 零保留并保存空 OOF；
  29/29 分布是诊断观察，当前未嵌入该 manifest 字段。

### 60-second passing reference / 60 秒通过参考

Persistent output / 持久输出:
`artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2`

Authority / 权威指针:
`artifacts/experiments/reference_registry.json`. The earlier immutable directory
`reduced_real_r0_f0_reference` is retained but marked superseded because it was
generated before the all-missing-column width-preservation fix.

- Status: `passed`
- Scope: `smoke_not_scientific_benchmark` — never cite as a benchmark result.
- Post-fix immutable verification wall time: 40.889 seconds; the cell manifest
  records 39.9477 seconds for the cell execution itself.
- OOF participants: 5 retained of 6; coverage = 0.8333333333333334.
- Balanced accuracy = 0.5; macro-F1 = 0.48888888888888893.
- These values verify execution and OOF integrity only. They do not establish model
  selection, superiority or publication performance.

- 状态：`passed`
- 范围：`smoke_not_scientific_benchmark`，禁止作为 benchmark 引用。
- 修复后不可变验证的 wall time 为 40.889 秒；cell manifest 内部记录的 cell
  执行耗时为 39.9477 秒。
- OOF participant：6 人中保留 5 人；coverage = 0.8333333333333334。
- BA = 0.5；macro-F1 = 0.48888888888888893。
- 这些数值只验证执行和 OOF 完整性，不代表模型选择、优越性或论文性能结论。

## Automated verification / 自动验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error PYTHONPATH=final_v0/final_pipeline_v1/src python3 -B -m unittest -v final_v0/final_pipeline_v1/tests/integration/test_experiment_runner.py
```

Final focused result / 最终 focused 结果:

- 4 tests passed, 0 failures, 0 errors, 0 skips, with every warning promoted
  to an error.
- Elapsed: 54.149 seconds.
- The default suite includes a clean temporary real 60-second r0/f0 frozen-fold run.
- The synthetic three-class fixture uses the same production route and asserts:
  exact train-only calibrator IDs, no OOF fit IDs, all three labels, nonempty
  three-class probabilities and retained participant OOF.
- An AST assertion enforces exactly one definition for each public runner and
  rejects the removed placeholder contract string.
- A three-model regression test proves that an all-missing outer-train feature
  column produces no warning and preserves the frozen feature width.

## Defects found by real execution / 真实执行发现并修复的问题

1. The strict JSON writer initially imported the wrong module; corrected to the
   canonical root-restricted atomic writer in `provenance.py`.
2. Dropped OOF rows initially inherited a nonempty class order despite having an
   empty probability vector; they now carry an empty class order.
3. `OofWriter.write` was initially called as a class method; it is now instantiated.
4. Cell summaries now explicitly include class order for confusion artifacts.
5. The original 12-second default was demonstrated insufficient and changed to the
   shortest tested passing duration, 60 seconds.
6. Median imputers originally warned and removed all-missing route-specific columns.
   All three allow-listed baselines now set `keep_empty_features=True`, preserving
   the frozen feature registry width under strict `PYTHONWARNINGS=error` execution.

## Known limitations / 已知限制

- The implemented scientific cell executor currently supports
  `representation_mode=feature_vector`. Raw waveform, matrix and fusion configs
  fail closed explicitly; no adapter pretends they are feature vectors.
- The full 5×5 orchestration is implemented without shortening, but executing all
  25 cells was outside this phase's runtime budget and remains unexecuted here.
- Bundle export is not part of this runner phase.
- The 60-second real smoke contains all-null route-specific feature columns. The
  train-only sklearn imputer now preserves their registered width and fills them
  deterministically; full benchmark review should still quantify availability by
  route.
- Full 5×5 metrics remain unavailable until all candidate configurations are run
  under the unified protocol. No route ranking is claimed by this phase.

---
