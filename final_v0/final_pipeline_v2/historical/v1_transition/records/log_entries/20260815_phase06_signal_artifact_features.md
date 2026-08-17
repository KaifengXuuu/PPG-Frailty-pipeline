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
