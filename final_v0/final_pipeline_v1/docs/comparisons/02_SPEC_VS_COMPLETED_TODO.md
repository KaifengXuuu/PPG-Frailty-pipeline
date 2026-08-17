# Specification vs completed M0–M3 / 规范与本会话已完成 TODO 的重合和矛盾

## Evidence boundary / 证据边界

This report uses the current status artifacts for M0–M3 and live source/test audits. It does
not treat a milestone document as proof of an unrun benchmark. In particular, M2 explicitly
says “no model rerun yet”, and M3 engineering tests are not clinical validation.

The V1 engineering checkpoint is bound to
[strict acceptance](../../artifacts/acceptance/strict_acceptance_current.json) and
[CPU CI](../../artifacts/acceptance/cpu_ci_current.json). The
[real frozen-record smoke](../../artifacts/test_reports/integration_smoke_canonical_manual.json)
proves input/protocol integration without training; reducer/model/gravity/physical-time reports
under `artifacts/test_reports` are explicitly synthetic or construction-level. None is a
Frailty3 or external-PTT scientific benchmark. A separate
[60 s real single-fold training smoke](../../artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json) (authority: [reference registry](../../artifacts/experiments/reference_registry.json); prior immutable reference is superseded)
persists train-only fitted provenance and complete retained/dropped OOF trace; its 5/6 coverage,
0.8333 coverage rate and BA 0.5 remain reduced smoke values, not a full 5×5 result. The
[12 s run](../../artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)
correctly failed closed without a metric.

## M0 — Historical motion/denoising/heartbeat audit

### Overlap / 重合

- M0 inventories NLMS/ANC, decomposition, STFT masking, hybrid/learned denoisers, motion
  detectors, direct peaks and external reference scorecards—the evidence needed to avoid
  repeating failed experiments.
- It records the central limitation adopted by the specification: the project should not
  promise full clean-waveform recovery; motion processing is a candidate for HR/PPI or
  Frailty input.
- It confirms activity supervision semantics `B/R→static`, `S/W→motion` and preserves
  historical confusion matrices/scorecards as evidence.

### Difference / 缺口

- M0 is an audit package, not a runnable `ArtifactReducer` implementation.
- It does not provide typed `x_ar`, failure/confidence/alignment, endpoint SQI, a common pulse
  backend, or the mandatory non-identity morphology prohibition.
- The 29-subject motion detector retrain/CV/threshold/SQI integration remains unimplemented.

### Conflict resolved in V1 / V1 修正

Historical functions contain misnamed wavelet/Savitzky–Golay behavior, memoryless NLMS,
full-record fitting and silent fallback. V1 reimplements algorithms behind strict interfaces;
it does not runtime-import those scripts or promote their historical scores.

## M1 — End-to-end architecture contract

### Overlap / 重合

- M1 V3 already freezes SQI-first routing, optional motion detection, high-quality bypass,
  and run-level `drop XOR denoise_then_extract` for degraded/motion data.
- It separates module status/action/reason, refuses denoiser failure fallback, and records
  mobile/offline execution modes and dependency/platform registries.
- This matches the specification's direct branch and rate-only non-identity branch.

### Difference / 缺口

- M1 is primarily machine schemas/examples; it does not implement FeatureVector/Matrix/Fusion,
  reducers, models, Trainer, OOF, metrics or bundle round trip.
- Its legacy feature schema IDs cannot be silently reused for V1's dynamic-acceleration raw8
  or PRV-v2 semantics.
- Mobile examples do not prove target-device latency, memory or ONNX parity.

### Conflict resolved in V1 / V1 修正

Earlier M1 V1/V2 language allowed action-owner/SQI weighting/coarse waveform replacement.
M1 V3 superseded it. V1 binds only the current SQI-first/rate-only semantics and treats old
registries as migration evidence.

## M2 — Data manifest and evaluation protocol

### Overlap / 重合

M2 is the strongest directly reusable milestone:

- 29 participants, 261 internal eight-channel recordings, unchanged labels and nine roles;
- exact source hashes and full finite numeric scan;
- corrected balanced subject-level SGKF registry;
- 5 repeats × 5 folds, seeds `42,10042,20042,30042,40042`;
- fixed epoch/no early stopping and `oof_validation_*` naming;
- external PTT/Sim manifests with ECG-reference and provenance status.

V1 imports the materialized memberships rather than rerunning `StratifiedGroupKFold`, converts
M2 rows to the stricter `ManifestRow`, re-hashes all 261 sources, and preserves M2 file/payload
hashes in every run identity.

### Difference / 缺口

- M2 does not provide recording-level parse/flatline/clipping/gap/time reason rows for future
  arbitrary inputs; V1 adds this QC layer.
- M2 does not define complete OOF row schemas, model fitting guards, content-addressed caches,
  or a final bundle.
- PTT RED/IR mapping remains unresolved; VitalDB is unfrozen; several external datasets are
  conditional/not eligible for the requested motion benchmark.

### No conflict / 无矛盾

The V1 corrected fold membership is exactly sourced from M2; labels and participants are not
changed. The provisional PTT split is new, clearly non-independent and pending V2 confirmation.

## M3 — Unified preprocessing and signal algorithms

### Reusable overlap / 可复用重合

- 400 Hz internal grid; static PPG 0.2–8 Hz, three-order Butterworth, no notch;
- explicit ACC/gyro units and no magnitude guessing;
- no-precalibration quaternion error-state EKF primary plus causal 0.3 Hz LPF comparator;
- source/repaired masks, timestamp validation and bounded internal gap repair;
- train-fold scaler guards;
- corrected bipolar peak/PPI backend and train-only ECG delay evaluator;
- offline/mobile profile separation and strict failure states.

V1 migrates source-hashed pure behavior into a self-contained package; it does not import the
M3 sibling directory at runtime.

### Specification gaps / 相对规范的缺口

At the time of V1 audit M3 lacked, or did not fully freeze:

- canonical `SignalViews` route semantics and `WindowPlan` shared by engineering/DL;
- endpoint `Q_rate/Q_morph/not_applicable`;
- complete valley morphology and beatwise dual-wavelength contract;
- the full requested PRV field/eligibility set;
- identity/NLMS/decomposition/spectral/BSS common reducers;
- four representation modes, models, unified Trainer, complete OOF and final bundle.

### Conflicts corrected in V1 / V1 修正

1. Some M3 peak profiles use 0.4–8 Hz, while the attached canonical direct `x_filter` is
   zero-phase 0.2–8 Hz. V1 direct analysis equals `x_filter`; no hidden secondary filter.
   Final 0.2–8 versus 0.4–8 selection is now explicit pending point V2-028 and requires a
   one-factor paired rerun if changed.
2. Earlier M3 `compute_prv` semantics and saved reports did not constitute the full spec PRV
   contract. V1 adds exact durations/counts/bands/SampEn/validity.
3. The M3 current-status test count and saved report were observed at different revisions
   (live audit 46 versus stale saved 22; one status prose says 38). V1 fingerprints its test
   source and report together and never uses an older count as current evidence.
4. A compact three-state gravity-vector EKF would contradict the frozen M3/user decision.
   V1 therefore requires the stateful quaternion+bias+covariance implementation; a simplified
   EKF cannot pass final conformance.

## Combined completion statement / 综合结论

M0–M3 supplied valuable evidence, contracts, manifests and signal primitives, but did not form
the complete package requested by the attached specification. V1 legitimately reuses M2
memberships and tested M3 mathematics while rebuilding the missing typed routes, feature
representations, models, training/evaluation, OOF, comparisons and bundle boundaries. No
historical BA or engineering test count is promoted to a new clinical/performance claim.

The formerly incomplete model surface is now engineering-complete, including an experimental
ShapeFormer repair with patch/downsample before mask-aware attention and outer-train-bound
effect-size discovery. This is implementation conformance only, not PISD/original-paper parity
or evidence that ShapeFormer is better than another candidate.
