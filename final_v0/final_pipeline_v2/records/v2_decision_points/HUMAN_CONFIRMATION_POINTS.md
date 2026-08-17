# V2 human-confirmation points / V2 人工确认点

状态 / Status: **mixed: confirmed, partially confirmed, and pending**. This registry
contains 28 decision points. A V1 conservative default is not automatically a user
decision. Changing any confirmed/default answer requires a new config ID, affected rerun,
and the user-requested name `final_pipeline_v2`.

状态标记 / Status legend:

- **confirmed / 已确认**：用户已明确选择，可直接作为 V2 输入，除非用户主动重开。
- **partially confirmed / 部分确认**：原则或部分参数已确认，列出的剩余问题仍需回答。
- **pending / 待确认**：V1 仅采用保守实现值，不代表用户接受为最终路线。
- Confirmed/partial points carry an explicit line below; every other ID remains pending.
  The 28-row status table is in `docs/comparisons/05_V1_TO_V2_CONFIRMATION_SUMMARY.md`.
  / 已确认或部分确认点在下方显式标记；其余 ID 均为待确认，28 项总表见 comparison 05。

## A. Data, splits, and labels / 数据、折与标签

### V2-001 — External PTT participant split

- V1 default: materialize `v1_provisional_external_grouped_split_seed42`, five grouped
  folds over 22 PTT participants; it is explicitly not an independent test.
- Alternative: adopt the historical 15/3/4 split or define repeated grouped folds.
- Impact: every artifact-policy and heartbeat comparison must be rerun on one common split.

### V2-002 — PTT RED/IR channel mapping

- V1 default: preserve `unresolved_red_ir_mapping_conflict`; use a declared anonymous
  pleth channel for rate testing and fail closed for wavelength-dependent BSS/optical claims.
- Decision needed: authoritative pleth-to-wavelength map and evidence source.
- Impact: enables or disables PTT dual-wavelength and BSS comparisons.

### V2-003 — Display class names

- V1 default: machine IDs remain `0/1/2`; canonical display is `Pre-Frail`,
  `Robust/Non-Frail`, `Young`; M2 snake-case values remain provenance aliases.
- Alternative: use M2 snake-case names everywhere.
- Impact: presentation/schema strings only; labels and memberships must not change.

### V2-004 — Role-aware participant weights

- V1 default: ordinary mean within role and equal weight across available roles;
  missing roles are renormalized. B/R1–R4 are the static reference config.
- Alternatives: role-family weights, protocol-time weights, or a pre-registered B/R scheme.
- Impact: participant probabilities and all paired metrics change.

### V2-005 — Stage semantics beyond confirmed facts

- V1 default: only B=baseline, R=relax/recovery, S=stand-and-sit, W=walk, and S/W before
  Relax are used. No unconfirmed repetition meaning or recovery slope is a default predictor.
- Decision needed: exact sequence, R1–R4/S1–S2/W1–W2 timing, and prediction-time availability.
- Impact: required before a Base/Motion/Relax recovery classifier can become active.

## B. QC, preprocessing, and physiology / 质量、预处理与生理量

### V2-006 — Device-specific QC thresholds

- Confirmation status / 确认状态：**partially confirmed / 部分确认**。用户已冻结上一版
  推荐的 D1–D4、D6–D8 信号决策；设备饱和轨、合理光学/IMU量程与最短 endpoint
  时长仍未提供，因而不能把 V1 的 review flag 写成设备阈值。
- V1 default: internal interpolation limit 100 samples, flatline 1 s, finite/timestamp/
  synchrony failures fail closed; unknown ADC-scale clipping is a review flag rather than
  an invented absolute threshold.
- Decision needed: device saturation rails, plausible optical/IMU ranges, and minimum
  endpoint durations.
- Impact: retention, coverage, and no-result rates.

### V2-007 — Primary IMU gravity estimator

- Confirmation status / 确认状态：**confirmed / 已确认**。
- User-confirmed decision: online **no-precalibration quaternion error-state EKF** is the
  primary route; causal **0.3 Hz LPF** is a mandatory matched comparator and never a
  silent fallback.
- Reopen rule: selecting LPF as primary later is a new decision and requires a matched rerun.
- Impact: motion descriptors and downstream SQI.

### V2-008 — PRV eligibility

- Confirmation status / 确认状态：**partially confirmed / 部分确认**。V1 的严格
  duration/count/adjacency/invalid-as-unavailable 合同属于已冻结推荐路线；是否允许更短、
  具名且不得混入主结果的 exploratory PRV 输出仍待确认。
- V1 default: basic time-domain PRV requires 60 s; spectral PRV requires 300 s and at least
  200 accepted intervals; SampEn uses `m=2, r=0.2*SD` and at least 200 intervals.
- Decision needed: whether shorter named exploratory outputs may enter predictors.
- Impact: feature availability; cannot be changed after viewing outer results.

### V2-009 — SQI calibration and cut points

- V1 default: endpoint-separated train-fold empirical calibration, `Q_rate=0.50`,
  `Q_morph=0.65`; quality weighting is off until validated.
- Alternatives: fixed device thresholds or inner-fold selected thresholds.
- Impact: drop/retain route, morphology eligibility, and coverage.

### V2-010 — Motion detector inside the first-stage gate

- V1 default: SQI is mandatory; a motion detector is optional and must be explicitly
  enabled with a frozen model/threshold. Motion can override high SQI into the configured
  degraded policy.
- Decision needed: activate a retrained 29-subject detector, rule detector, or no detector.
- Impact: route distribution and feature availability; requires its own OOF training evidence.

## C. Artifact and feature routes / 伪影与特征路线

### V2-011 — Default degraded policy

- Confirmation status / 确认状态：**partially confirmed / 部分确认**。用户已确认
  `SQI → optional motion → high-quality direct return → low-quality/motion run-locked
  drop XOR denoise-to-features` 的顺序和互斥性；尚未确认 deployment reference
  最终选择 drop 还是一个具名 reducer。
- V1 default: one run preselects exactly one of `drop` or
  `denoise_then_extract_rate_features`; static reference uses drop, motion comparison uses
  spectral rate-only. No window-level post-hoc choice.
- Decision needed: which policy becomes the deployment reference after paired evidence.
- Impact: coverage versus rate recovery.

### V2-012 — Reference non-identity reducer

- V1 default: identity is the primary direct baseline; NLMS, SSA, spectral mask, PCA,
  FastICA, and NMF are comparison modules. No learned waveform denoiser is promoted.
- Decision needed: retain one reducer after PTT ECG-reference comparisons.
- Impact: selected motion route; morphology remains unavailable for every non-identity choice.

### V2-013 — BSS availability

- V1 default: internal known RED/IR can run BSS; single-channel and unresolved-wavelength
  external inputs fail closed. PCA/FastICA/NMF remain separate named methods.
- Decision needed: whether anonymous synchronized pleth channels are sufficient for a
  rate-only PTT BSS comparison without wavelength labels.
- Impact: scientific wording and eligible external rows.

### V2-014 — Feature allowlist and optional exploratory features

- V1 default: only the frozen physiological allowlist and validity fields enter models;
  duration, row/window count, file order/path, IDs, and administrative missingness are excluded.
- Decision needed: approve any exploratory diagnostic before outer evaluation.
- Impact: feature schema hash and all vector/matrix/fusion models.

## D. Models and training / 模型与训练

### V2-015 — Formal dependency envelope

- Confirmation status / 确认状态：**partially confirmed / 部分确认**。
- User-authorized: **NumPy, SciPy, scikit-learn, and ONNX Runtime**.
- V1 packaging: NumPy/SciPy/scikit-learn/PyYAML/joblib are core; PyTorch is optional
  `deep`; pandas is optional `tabular`; pyarrow is optional `parquet`; ONNX Runtime is
  optional `onnx`. Commands fail clearly when a required optional dependency is absent.
- Decision needed: formally authorize PyTorch, PyYAML, joblib, pandas, and pyarrow, and
  decide which optional profiles are mandatory for the final supported workflow.
- Impact: installation lock and supported commands.

### V2-016 — ROCKET implementation source

- V1 default: self-contained deterministic NumPy/SciPy ROCKET; MiniROCKET is a separately
  named approximation/ablation. No aeon/sktime dependency.
- Alternative: pin aeon or sktime and use its audited implementation.
- Impact: parity, performance, bundle format, and dependency footprint.

### V2-017 — ShapeFormer discovery method

- V1 default: self-contained effect-size discovery, explicitly named experimental and not
  PISD/original parity. The hard-coded external ShapeFormer repository is not imported.
- Decision needed: license and vendor/pin PISD, or retain the effect-size experimental route.
- Impact: model name, scientific comparability, and bundle contents.

### V2-018 — Epoch rule and count

- V1 default: pre-registered fixed 50 epochs, no outer early stopping.
- Alternative: inner participant-grouped selection followed by refit on all outer training
  participants for the selected epoch.
- Impact: compute cost and every deep-model result; must be fixed before comparisons.

### V2-019 — Raw-window duration and DL sampling rate

- V1 default: 5 s at 400 Hz; 10 s and 100/160/200 Hz are named one-factor ablations.
- Decision needed: which longer context and physical kernel-duration set are reference.
- Impact: architecture time scale, memory, latency, and learned weights.

### V2-020 — Five-member ensemble budget

- V1 default: wrapper and reduced CPU tests are mandatory; full 5×5×5-member training is
  not represented as completed until scheduled evidence exists.
- Decision needed: authorize full compute budget and whether ensemble is a final candidate.
- Impact: 5× training/storage cost and leaderboard eligibility.

### V2-021 — Serialized sklearn artifact format

- V1 default: version-pinned trusted joblib plus hashes; never load an untrusted bundle.
- Alternative: explicit array/JSON representation for supported estimators.
- Impact: portability, security, and round-trip implementation scope.

### V2-022 — ONNX gate

- V1 default: ONNX Runtime is optional and currently absent; V1 proves CPU Python bundle
  parity but does not claim mobile ONNX parity.
- Decision needed: make ONNX export/runtime parity mandatory for V2 or defer until the
  classifier is selected.
- Impact: deployment definition and required dependency installation.

## E. Evaluation, compute, and deployment / 评估、算力与部署

### V2-023 — Parquet as a hard output dependency

- V1 default: complete OOF Parquet is required for formal runs; missing pyarrow fails closed.
  JSON/CSV may accompany but not silently replace formal artifacts.
- Alternative: define an equally typed non-Parquet canonical format.
- Impact: formal run eligibility and environment requirements.

### V2-024 — Full benchmark compute authorization

- V1 default: code, deterministic reduced tests, and smoke matrices are completed; no
  unexecuted 5 repeats × 5 folds × all candidates score is invented.
- Decision needed: authorize the exact candidate matrix, epochs, hardware, and run budget.
- Impact: duration and whether final performance ranking can be produced.

### V2-025 — Independent frailty test claim

- V1 default: all 29-subject scores are `oof_validation_*`; `independent_test=false`.
- Decision needed: reserve or acquire an untouched participant set before publication-level
  independent performance is claimed.
- Impact: claim wording, not the existing folds.

### V2-026 — Target hardware profile

- V1 default: portable CPU code and three conceptual platform budgets; no latency/RAM/power
  figure is claimed without device measurement.
- Decision needed: high-performance x86/ARM edge versus value ARM SBC reference device.
- Impact: model choice, ONNX/quantization, memory and latency acceptance thresholds.

### V2-027 — Scope of TODO-only routes

- V1 default: implement the attached dev0 contract first. TODO-only hierarchical Young/Old,
  recovery-slope stage classifier, broad wavelet/EMD/VMD zoo, and historical Top-5 program
  are documented as differences, not silently added to the canonical V1.
- Decision needed: which become V2 modules and in what order.
- Impact: scope, new tests, and comparison matrix size.

### V2-028 — Direct PPG band: 0.2–8 Hz versus 0.4–8 Hz

- Confirmation status / 确认状态：**pending / 待确认**。
- V1 default: the attached specification's canonical direct `x_filter` remains zero-phase
  **0.2–8 Hz** at 400 Hz. Historical M3 peak-oriented profiles using **0.4–8 Hz** are
  provenance evidence or an explicitly named ablation, never a hidden secondary filter.
- Decision needed: keep 0.2–8 Hz as the final direct reference, select 0.4–8 Hz, or require
  a preregistered one-factor paired comparison before V2 selection.
- Impact: peak timing, Q_rate/Q_morph components, morphology/optical features, HR/PPI/PRV,
  feature hashes, and every affected model must be rerun under one frozen fold protocol.
