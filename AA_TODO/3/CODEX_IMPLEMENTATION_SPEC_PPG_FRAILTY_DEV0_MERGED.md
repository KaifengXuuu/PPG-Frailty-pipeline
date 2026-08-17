# Codex Implementation Contract — PPG Frailty Pipeline (`dev0`)

**Baseline repository:** `KaifengXuuu/PPG-Frailty-pipeline`  
**Baseline branch:** `dev0`  
**Audited commit:** `2eca0ecf0e17a4deaa1d3cc8e821098e5848e421`  
**Contract date:** 2026-08-15  
**Primary objective:** make the implementation reproducibly match the accepted thesis Chapter 3 workflow without changing scientific claims merely to improve scores.

> This file is an implementation contract, not a brainstorming document or a calendar plan. Follow the implementation order and acceptance gates. Do not perform a broad rewrite, delete historical scripts, change labels, tune on outer folds, or extend scope unless a task below explicitly authorizes it.

## 1. Source-of-truth order

When sources conflict, use this order and record the resolution in an Architecture Decision Record (ADR):

1. **Scientific invariants:** participant grouping, no held-out fitting, signal units, target labels, recording roles, endpoint eligibility, and declared evaluation unit.
2. **This implementation contract and accepted ADRs.**
3. **The accepted thesis Chapter 3 workflow**, after resolving its internal contradictions, especially aggregation order, PRV eligibility, feature availability, and artifact-processed morphology.
4. **Current `dev0` behavior**, preserved behind an explicitly named `legacy_*` route only where needed for historical-result replay.
5. **Original model papers and repositories**, used to label deviations accurately; they do not override project-specific scientific requirements.

## 2. Non-negotiable scientific invariants

- The targets remain `Pre-Frail`, `Robust/Non-Frail`, and `Young`; do not relabel or merge classes in the reference three-class experiment.
- Participant identity is the grouping variable. No participant, file, raw window, feature vector, feature matrix, shapelet candidate, scaler fit, quality threshold, calibrator, ROCKET transform, or model-selection observation may cross from an outer held-out fold into training.
- All branches consume the same versioned record manifest and the same frozen participant-fold assignment.
- The internal acquisition grid remains 400 Hz for `x_native`, morphology, peak timing, and audit storage. A **separate DL-only anti-aliased resampling option** may be tested; never silently overwrite the acquisition rate.
- The external PTT-PPG dataset is used to develop and evaluate motion processing and heartbeat recovery. It must not supply frailty labels or frailty-classifier training participants.
- Outer held-out folds are evaluation-only. Epoch selection must use a training-only inner grouped split or a pre-registered fixed epoch.
- Every fitted object is fold-local: imputer, scaler, feature selector, SQI normalization and threshold, shapelet discovery, ROCKET kernels/transform, ridge hyperparameters, calibration, and network weights.
- Every reported participant prediction must be traceable to its windows, file, role, fold, seed, config hash, code commit, preprocessing schema, feature schema, signal route, and aggregation rule.
- Do not optimize toward a target balanced accuracy. Correctness, protocol comparability, and reproducibility are the completion criteria.
- **Any non-identity artifact-reduced signal `x_ar` is a rate-recovery signal, not a morphology-preserving signal.** After non-identity artifact reduction, only `Q_rate` is recomputed. `Q_morph` is `not_applicable`, and morphology/amplitude-dependent features from `x_ar` are unavailable unless a separate paired-clean validation contract is explicitly approved in a future ADR.

## 3. Required ADRs before code restructuring

Create `docs/adr/` and commit the following decisions before modifying behavior:

1. `ADR-001-canonical-experiment-entrypoint.md`
2. `ADR-002-record-manifest-and-fold-freeze.md`
3. `ADR-003-signal-views-and-units.md`
4. `ADR-004-window-planning-padding-and-masks.md`
5. `ADR-005-prv-eligibility-and-time-axis.md`
6. `ADR-006-window-file-subject-aggregation.md`
7. `ADR-007-epoch-selection-and-outer-fold-isolation.md`
8. `ADR-008-model-naming-and-original-paper-deviations.md`
9. `ADR-009-dl-sampling-rate-and-kernel-time-scales.md`
10. `ADR-010-motion-branch-status-and-primary-experiment-boundary.md`
11. `ADR-011-representation-modes-and-feature-matrix-contract.md`
12. `ADR-012-post-artifact-rate-only-feature-contract.md`

Required ADR resolutions:

- **Aggregation:** `window probabilities -> file probability -> role-aware participant probability`. Keep direct all-window participant averaging only as a named ablation.
- **PRV:** full VLF/LF/HF, normalized units, and entropy only for qualified long static/reference recordings; short task records expose only eligible HR/PPI/time-domain quantities.
- **Artifact-processed branch:**

  ```text
  x_filter
    -> ArtifactReducer
    -> x_ar
    -> recompute Q_rate only
    -> common pulse detector
    -> HR / PPI / eligible PRV
    -> file/role rate-feature representation
  ```

  `Q_morph` must not be calculated on non-identity `x_ar`. Morphology, AC/DC, PI, ratio-of-ratios, pulse width, slope, area, and other waveform-shape/amplitude features must be marked unavailable for that processed segment.

## 4. Target package boundary

Do not delete current scripts during the first implementation stage. Add a canonical package and convert old scripts to thin wrappers only after characterization tests pass.

```text
src/ppg_frailty/
  config.py
  provenance.py
  contracts.py
  pipeline.py
  data/
    schema.py
    manifest.py
    external_manifest.py
    qc.py
    folds.py
  signal/
    views.py
    ppg_preprocess.py
    imu_preprocess.py
    window_plan.py
    resample.py
  quality/
    components.py
    endpoint_sqi.py
    routing.py
  artifact/
    base.py
    identity.py
    nlms.py
    decomposition.py
    spectral.py
    bss.py
  peaks/
    aboy_project.py
    intervals.py
    pairing.py
  features/
    registry.py
    prv.py
    morphology.py
    dual_wavelength.py
    spectral.py
    engineering.py
    file_vector.py
    ordered_matrix.py
  representations/
    modes.py
    raw.py
    feature_vector.py
    feature_matrix.py
    fusion.py
  models/
    compact_cnn.py
    inception_time_port.py
    inception_ensemble.py
    shapeformer_port.py
    feature_models.py
    rocket_ridge.py
    file_fusion.py
  train/
    datasets.py
    sampling.py
    losses.py
    trainer.py
    selection.py
  evaluate/
    aggregate.py
    metrics.py
    oof.py
    calibration.py
    benchmark.py
  bundle/
    schema.py
    save.py
    load.py
    infer.py
  cli.py
configs/
  reference_static_v1.yaml
  reference_all_roles_v1.yaml
  motion_benchmark_v1.yaml
  feature_matrix_v1.yaml
manifests/
splits/
tests/
```

Use typed containers for at least:

- `ManifestRow`
- `SignalViews(x_native, x_filter, x_analysis, imu_processed, metadata)`
- `QualityResult(q_rate, q_morph, state, components, reasons, coverage)`
- `PulseResult(peaks, accepted_peak_mask, ppi, valid_interval_mask, adjacency_mask, wavelength, detector_version)`
- `FeatureVectorV1(values, validity, schema_version, provenance)`
- `EngineeringFeatureSequence(values, start_samples, valid_row_mask, schema_version)`
- `OrderedFeatureMatrixV1(values, row_mask, channel_schema, context_schema, schema_version)`
- `PredictionBundle(file_probabilities, participant_probabilities, coverage, route, model_version)`

`QualityResult.q_morph` must support the explicit state `not_applicable`. Do not encode `not_applicable` as pass, fail, or zero.

## 5. Static reference implementation

### 5.1 Baseline inventory and characterization

- Record `2eca0ecf0e17a4deaa1d3cc8e821098e5848e421`, Python/package versions, CUDA/cuDNN, file fingerprints, manifest counts, and historical result paths.
- Add characterization tests around current preprocessing, feature columns, model parameter counts, and known historical metrics before changing behavior.
- Preserve historical behavior under `legacy_v0`; never overwrite old result directories.

**Required output:**

- `artifacts/audit/baseline_inventory.json`
- `artifacts/audit/legacy_characterization.json`
- `MIGRATION.md`, mapping legacy functions/caches to canonical functions and schema versions.

### 5.2 Versioned manifest, QC, and frozen folds

Implement one manifest row per recording with at least:

`record_id, participant_id, class_id, class_name, role, source_path, source_hash, source_version, fs, n_samples, duration_s, channel_schema, channel_units, synchrony_status, reference_available, qc_status, qc_reasons, manifest_version`.

QC reason codes must cover parse failure, missing required channel, all-nonfinite channel, excessive non-finite gap, insufficient duration, flatline, clipping/saturation, implausible scale, timestamp failure, synchrony failure, and duplicate record. Never silently skip a record and never replace an entirely unavailable required channel with zeros.

Generate `splits/sgkf5_v1.csv` once from the manifest. All model and feature branches read it; they do not regenerate folds independently.

**Acceptance criteria**

- Exactly one fold per participant.
- No participant appears in two folds.
- Every eligible file inherits its participant fold.
- Fold file and manifest hashes are embedded in every run.
- Manifest count and class/role count audit matches the expected internal cohort or explicitly reports deviations.
- Process fields such as `n_samples`, `duration_s`, and window count are QC/provenance only and are excluded from the default predictor schema.

### 5.3 Unified signal views and window planner

Implement explicit signal views:

- `x_native`: interpolated acquisition-scale RED/IR, retaining the optical baseline required for DC-dependent quantities.
- `x_filter`: detrended, zero-phase 0.2–8 Hz analysis RED/IR on the 400 Hz acquisition grid.
- `x_ar`: optional, aligned, non-identity artifact-reduced RED/IR on the original time grid; valid only for the rate-recovery branch.
- processed IMU axes and derived acceleration magnitude `A`, angular-rate magnitude `Omega`, and jerk magnitude `J`.
- DL windows with explicit `fs`, `start_sample`, `end_sample`, `valid_length`, `padding_mask`, and `source_record_id`.

For offline reference processing, never silently fall back to a causal filter. Short records must be rejected, explicitly padded under a tested policy, or processed with a documented reduced-`padlen` zero-phase policy.

Use one `WindowPlan` implementation for engineering and DL windows. End alignment, overlap, cap, and padding are configuration fields, not hidden defaults.

Add DL-only anti-aliased resampling with configured `dl_fs` values. The feature branch remains on the 400 Hz analysis grid unless a feature definition explicitly states otherwise.

### 5.4 Feature Registry V1

Create a frozen, ordered registry with the following fields for every feature:

- canonical name;
- formula/algorithm;
- units;
- source signal view;
- endpoint and role eligibility;
- beat/window/file level;
- aggregation rule;
- validity rule;
- missing-value policy;
- provenance/version.

Required corrections:

- Preserve peak timestamps and interval indices; do not compress time after rejecting PPIs.
- Implement the declared PRV set or remove unsupported items from the thesis. Long-record frequency and entropy eligibility must be explicit and tested.
- Implement valley-to-valley morphology using the local linear baseline, `Q_morph`, and median/MAD robust aggregation.
- Separate PPG and IMU spectral bands. Normalize spectral entropy exactly as declared.
- Implement dual-wavelength zero-lag correlation, maximum lag-normalized cross-correlation, lag, and cardiac-band coherence for eligible direct signals.
- Exclude technical fields from the default model feature allowlist.
- Use `NaN + validity=false` for unavailable physiological quantities. Do not replace them by a physiologically valid zero.

Create:

- `FeatureVectorV1`: one complete, ordered file-level predictor vector.
- `EngineeringFeatureSequence`: chronological engineering-window descriptors with start positions and validity.
- `OrderedFeatureMatrixV1`: `K=32` ordered engineering positions, a row mask, fixed channel order, and valid recording-context channels.

### 5.5 Endpoint SQI without leakage

Implement separate `Q_rate` and `Q_morph` for the **direct amplitude-preserving analysis signal** `x_filter`.

Components expose raw value, normalized value, pass/fail, and reason:

- cardiac-band concentration;
- periodicity/peak density;
- PPI stability;
- RED–IR agreement;
- IMU motion energy;
- flatline, clipping, saturation, and long-gap exclusion.

Rules:

- SQI is calculated from amplitude-preserving analysis signals and processed IMU, not from median/IQR-normalized DL windows.
- Any learned normalization or threshold is fit on training participants only and then frozen for held-out participants.
- `Q_rate` determines whether pulse timing, HR, PPI, and eligible PRV can be extracted.
- `Q_morph` is stricter and determines whether direct-signal morphology and amplitude-dependent optical features can be extracted.
- For a non-identity `x_ar`, set `Q_morph=not_applicable`; recompute only `Q_rate`.

### 5.6 Freeze the three current model families

Do not redesign architectures before protocol parity is established.

1. Rename the current CNN to `CompactCNN1D`. Preserve 32/64/128 filters, kernels 9/9/7, two pool-4 operations, and GAP as the legacy/reference architecture. Document that it is not the Wang-FCN reproduction.
2. Preserve full and small InceptionTime ports as reference variants. Label them `single_network`; never call one network the full five-network InceptionTime ensemble.
3. ShapeFormer remains `experimental` until it is self-contained, outer-fold isolated, and processed by the same Trainer/Evaluator and input-time-scale contract.

Add architecture snapshot tests:

- CompactCNN1D trainable parameters: `79,139` for 8 channels and 3 classes.
- Full Inception port: `456,579`.
- Small Inception port: `57,027`.
- Output shape is `[batch, 3]` for all valid configured lengths.

### 5.7 Four representation modes and classifier routes

Expose one canonical configuration field:

```text
representation_mode = raw | feature_vector | feature_matrix | fusion
```

#### `raw`

- Input: robust-normalized multichannel windows `[RED, IR, AX, AY, AZ, GX, GY, GZ]`.
- Models: CompactCNN1D, full/small InceptionTime single networks, and ShapeFormer when eligible.
- Sample hierarchy: window -> file -> participant.

#### `feature_vector`

- Input: one `FeatureVectorV1` per recording.
- Models: L2 logistic regression, RBF SVM, Extra Trees, and an optional compact MLP.
- Only allowlisted predictor fields enter the model.

#### `feature_matrix`

- Input: one `OrderedFeatureMatrixV1` per recording, shape `[D, K=32]`, plus row mask.
- Models:
  - InceptionTime single network;
  - optional five-member InceptionTime ensemble;
  - ROCKET with 10,000 random convolutional kernels followed by ridge classification.
- One recording is one training sample. Do not reinterpret matrix rows as independent samples.

#### `fusion`

- Input: pooled raw file embedding plus one encoded `FeatureVectorV1`.
- Concatenate once at file level, then produce one file probability.

All four modes must use the same manifest, frozen folds, labels, role definitions, and participant-level evaluation semantics.

### 5.8 Ordered feature matrix contract

For a recording with `W` chronological engineering windows:

- Set reference length `K=32`.
- Time-varying channels are per-window engineering descriptors.
- Recording-context channels are the complete fold-standardized file feature vector repeated only across valid positions.
- Include explicit validity channels or masks for unavailable feature families.
- If `W > K`, select `K` positions uniformly over recording progress while preserving order.
- If `W < K`, right-pad **after fold-local imputation/scaling** and set `row_mask=false` for padded positions.
- Padded values are zero only because zero represents the standardized neutral value; the mask must prevent them from being interpreted as observed physiology.
- InceptionTime must use mask-aware global pooling or an equivalent tested padding policy.
- ROCKET receives the fixed matrix and an explicit mask channel or equivalent validated encoding; padded positions must not silently become physiological structure.
- Save matrix channel order, context-channel order, mask policy, schema hash, and transform version.

### 5.9 ROCKET plus ridge route

Implement a pinned, reproducible ROCKET wrapper for `OrderedFeatureMatrixV1`:

- primary reference: 10,000 random kernels;
- deterministic `random_state` per outer fold and member;
- fold-local imputation/scaling before transformation;
- kernels/transform fitted only on outer-training recordings;
- ridge alpha fixed by a pre-registered value or selected only within outer training;
- save kernel/transform state, scaler, ridge coefficients, class order, feature schema, and matrix schema;
- output finite class scores/probabilities and participant-level metrics through the common evaluator.

MiniROCKET may be added as a separately named ablation. It must not silently replace the specified ROCKET route.

### 5.10 Optional original-style five-member InceptionTime ensemble

Implement an optional ensemble wrapper for raw and feature-matrix inputs:

- `ensemble_size=5`;
- five independently initialized and trained members;
- same outer-training data, architecture, preprocessing, and training budget;
- deterministic but distinct member seeds;
- no shared trainable weights;
- exact arithmetic probability average:

  `p_ensemble = (p_1 + p_2 + p_3 + p_4 + p_5) / 5`;

- save every member and its seed;
- store member-level and averaged OOF probabilities;
- architecture/model cards must distinguish `single_network` from `five_member_probability_ensemble`.

The existing single-network InceptionTime remains the reference baseline. Do not block protocol repair on the ensemble, but do not claim original InceptionTime ensemble behavior until the five-member route has actually run.

### 5.11 Correct file-level signal–feature fusion

The current per-window repeated-feature fusion becomes a named legacy route only.

Implement `FileBagDataset`:

1. Encode each raw window.
2. Pool window embeddings to one file embedding with a mask-aware mean as the reference; attention pooling may be a named ablation.
3. Encode `FeatureVectorV1` once with the feature MLP.
4. Concatenate the two file embeddings once.
5. Produce one file probability.
6. Aggregate files to participant according to ADR-006.

Persist the exact feature imputer and scaler in the final bundle. The loaded bundle must reproduce training-side logits for golden records.

### 5.12 Unified trainer and outer-fold isolation

Create one Trainer/Evaluator used by CompactCNN1D, InceptionTime, ShapeFormer, feature-matrix InceptionTime, and the ensemble wrapper. It receives a frozen outer split and must not expose outer labels to the trainer.

Allowed epoch rules:

- pre-registered fixed epoch; or
- inner participant-grouped train/validation split, followed by refitting on all outer-training participants using the selected epoch.

Disallowed:

- selecting the best epoch on outer held-out windows;
- selecting SQI thresholds, shapelets, scalers, ROCKET parameters, sampler settings, or model variants using outer labels;
- comparing a fixed-epoch model with an outer-early-stopped model as if architecture were the only difference.

Write complete OOF artifacts:

```text
oof_window_predictions.parquet
oof_file_predictions.parquet
oof_subject_predictions.parquet
oof_member_predictions.parquet      # when an ensemble is used
metrics_per_fold_seed.json
confusion_matrices.json
run_manifest.json
```

Each row includes participant, file, role, fold, seed, config hash, probability vector, quality, retained/dropped status, representation mode, signal route, and provenance hashes.

### 5.13 Aggregation and metrics

Reference hierarchy:

```text
window -> file -> role-aware participant
```

- File aggregation is quality-weighted only when quality varies at window level and the weighting mechanism passes validation; otherwise use ordinary mean.
- Different recordings and roles contribute according to the frozen ADR, not in proportion to raw window count.
- Feature-vector and feature-matrix routes already produce file probabilities and enter at the file level.

Metrics use participant as the independent unit:

- balanced accuracy;
- macro-F1;
- per-class precision/recall/F1;
- worst-class recall/F1;
- confusion matrix;
- between-repeat mean/SD and a clearly defined confidence interval;
- paired deltas on identical folds/seeds;
- coverage and unavailable-feature rates by role/signal route.

Do not rank incomplete configurations. Preserve current ranking priorities only after all candidates share the same protocol and complete configuration identity.

### 5.14 Reproducible bundle and CPU CI

Bundle must contain:

- model class/version and state;
- class mapping and channel order;
- representation mode;
- signal/preprocessing configuration;
- DL resampler and window plan;
- feature registry hash, vector columns, matrix channels, masks, and validity policy;
- fitted imputer/scaler/selector/calibrator;
- ROCKET transform and ridge model when applicable;
- every Inception ensemble member when applicable;
- file pooler and aggregation rule;
- code commit, environment, manifest/fold hashes;
- golden inference examples.

Add CPU CI for lint/import/unit/synthetic integration tests. Full-data/GPU runs may remain manual or scheduled but must emit machine-verifiable manifests.

## 6. Model time-scale audit and dynamic branch

### 6.1 ShapeFormer repair

- Remove the hard-coded upstream path. Pin/vendor the required PISD code or implement a local equivalent.
- The discovery method is a required config field; never silently fall back from PISD to effect-size discovery.
- Remove outer-fold epoch selection.
- Apply the same SQI, sampler, loss, folds, aggregation, and OOF writer as the other models.
- Add anti-aliased downsampling or patch embedding before generic self-attention. Raw sample-token attention at 400 Hz is not the reference route.
- Record shapelet length in both samples and seconds.

### 6.2 Inception/CNN physical-time ablation

The current longest Inception kernel is 39 samples = 97.5 ms at 400 Hz. Six stacked longest branches have a theoretical local receptive field of 229 samples = 572.5 ms. Treat the current architecture as a valid local-morphology baseline, not as proof of pulse-cycle-scale suitability.

Run matched, parameter-conscious ablations:

- `dl_fs`: 100, 160, 200, 400 Hz;
- physical kernel-duration sets converted to odd sample counts;
- optional dilation to expand receptive field without excessive parameter growth;
- 5 s and one longer raw-window context, with identical participant folds;
- raw, feature-vector, feature-matrix, and corrected file-level fusion routes.

Use the same seeds and outer folds. Report participant metrics, calibration, train/held-out gap, runtime, memory, and weakest-class metrics. Do not select a configuration from one run.

### 6.3 Artifact-reduction interface and rate-only return path

Implement `ArtifactReducer` with `identity` first, then at least:

- IMU-referenced NLMS;
- one decomposition method;
- one spectral method if justified.

Every reducer returns:

- aligned RED/IR `x_ar` on the original time grid;
- reducer identity/version and parameters;
- diagnostics;
- confidence;
- failure status;
- channel availability and alignment metadata.

#### Direct branch

```text
x_filter
  -> Q_rate_pre + Q_morph_pre
  -> if Q_rate_pre passes: pulse detector -> HR/PPI/eligible PRV
  -> if Q_morph_pre passes: morphology/AC/DC/PI/dual-wavelength shape features
```

#### Non-identity artifact-reduced branch

```text
x_filter + processed IMU
  -> ArtifactReducer
  -> x_ar
  -> recompute Q_rate_post only
  -> common pulse detector
  -> HR/PPI/eligible PRV
  -> file/role rate-feature representation
```

Mandatory rules:

- `Q_morph_post = not_applicable` for every non-identity reducer.
- Do not use `x_ar` for pulse amplitude, width, rise/decay, slope, area, DC, PI, ratio-of-ratios, morphology coherence, or any claim of preserved waveform morphology.
- Do not use absolute PPG statistics or powers from `x_ar` as default physiological predictors; rate-quality spectral quantities may be used only inside `Q_rate` or as explicitly named experimental diagnostics.
- The processed file/role representation may contain HR, PPI, eligible PRV, accepted-beat coverage, rate confidence, `Q_rate_post`, reducer provenance, and processed IMU/activity descriptors.
- Morphology and amplitude-dependent feature slots remain `NaN` with validity `false/not_applicable`; never fill them with zeros or values copied from `x_filter`.
- The identity reducer is treated as the direct branch and may retain `Q_morph` because it does not alter the waveform.
- A future morphology-preserving reducer route requires a separate ADR, paired clean or equivalent morphology ground truth, landmark/area/amplitude preservation tests, and thesis wording approval. It is not part of this contract.

On PTT-PPG, lock the channel mapping and ECG R-peak reference. Report HR MAE, IBI/PPI error, event precision/recall/F1, timing error, coverage, and failure rate by sitting/walking/running. Waveform diagnostics are secondary and must not be interpreted as morphology recovery.

Internal dynamic recordings receive the same rate-only output contract. Without an external reference, report coverage, rate plausibility, RED/IR rate agreement, and route stability; do not call these ground-truth accuracy.

## 7. Exact feature contracts

### 7.1 Pulse events, HR, PPI, and PRV

- Keep peak indices, peak timestamps, interval indices, valid interval mask, and adjacency mask linked.
- Rejecting an interval must not compress the time axis or cause RMSSD/SDSD to bridge a missing interval.
- Output count, accepted duration, coverage, mean/median PPI, PPI SD/IQR/MAD/CV, mean/median pulse rate, pulse-rate SD, RMSSD, SDSD, NN50, pNN50, SD1, SD2, and SD1/SD2.
- Compute spectral PRV only on eligible approximately five-minute static/reference recordings using the declared 4 Hz interpolated/detrended tachogram and true timestamps.
- Spectral bands: VLF 0.003–0.04 Hz, LF 0.04–0.15 Hz, HF 0.15–0.40 Hz; output absolute powers, LF/HF, and normalized LF/HF units where declared.
- Compute SampEn only when `accepted_intervals >= 200`, with `m=2` and `r=0.2*SD`.
- Unavailable values are `NaN + validity=false`, not zero.
- The same rate/PRV extractor accepts eligible direct `x_filter` and rate-qualified `x_ar`.

### 7.2 Morphology — direct branch only

- Source: direct `x_filter`; DC-dependent quantities additionally use aligned `x_native`.
- Eligibility: accepted valley-to-valley beats passing `Q_morph`.
- `A_p = x(t_p) - l(t_p)`, where `l(t)` is the linear valley-to-valley baseline.
- Rise time, decay time, declared half-prominence width, mean systolic upslope, and positive baseline-corrected area.
- Aggregate beat values by median and MAD.
- Morphology is unavailable for non-identity `x_ar`.

### 7.3 Dual-wavelength optical features — direct branch only

- Compute beatwise AC and DC from accepted beats and their local baselines.
- Compute PI per wavelength: `PI = AC / (abs(DC) + epsilon)` with denominator validity.
- Use canonical RED/IR direction for AC ratio, DC ratio, and conventional ratio-of-ratios:

  `R = (AC_RED/DC_RED) / (AC_IR/DC_IR)`.

- Output zero-lag Pearson correlation, maximum lag-normalized cross-correlation, argmax lag, and mean 0.5–3 Hz magnitude-squared coherence on eligible direct signals.
- Every ratio has denominator, finite, and validity checks.
- Keep mixed-unit `motion_norm_*` heuristics out of the default predictor schema.
- These optical morphology/amplitude features are unavailable for non-identity `x_ar`.

### 7.4 Engineering statistical and spectral features

- Engineering windows: 10 s complete windows, 5 s hop; preserve chronological start positions and validity.
- Time descriptors: mean, population SD, RMS, IQR, MAD, bias-corrected Fisher–Pearson skewness, and Pearson kurtosis.
- Normalized spectral entropy: `-sum(p log p)/log(K)`.
- PPG bands for direct `x_filter`: 0.2–0.5, 0.5–3, and 3–8 Hz.
- IMU magnitude bands: 0.1–0.5, 0.5–3, 3–8, and 8–20 Hz.
- Default file-vector aggregates are mean and SD of valid engineering rows, unless the accepted thesis schema is explicitly changed by ADR.
- For non-identity `x_ar`, default optical engineering predictors are unavailable. Rate-quality diagnostics may be retained separately from the predictor registry.

### 7.5 Complete file feature vector

`FeatureVectorV1` contains the fixed, versioned allowlist of eligible features and parallel validity fields. It must not contain:

- participant ID;
- filename/path-derived identifiers;
- absolute file order;
- `n_rows`;
- file duration;
- number of generated windows;
- administrative missingness;
- any future information unavailable at the prediction point.

### 7.6 Complete ordered feature matrix

- One recording = one matrix sample.
- Reference length `K=32` chronological engineering positions.
- Time-varying channels = per-window engineering descriptors.
- Constant context channels = complete fold-standardized file feature vector repeated across valid positions.
- Validity and row-mask information must accompany the values.
- Long recordings are uniformly sampled over progress; short recordings are right-padded after fold-local transformation.
- Save feature order, context order, row mask, imputation/scaling objects, and schema version.

### 7.7 Processed motion file/role representation

For a non-identity artifact-reduced role, the default physiological block contains only:

- HR and PPI summaries;
- eligible time-domain PRV;
- spectral PRV only if the long-record eligibility rule is genuinely met;
- accepted-event count and coverage;
- `Q_rate_post` and rate confidence;
- reducer name/version/status;
- rate agreement between wavelengths where available;
- processed IMU/activity descriptors.

Morphology, AC/DC, PI, ratio-of-ratios, direct optical amplitudes, and morphology-derived context channels are unavailable and must carry validity `false/not_applicable`.

## 8. Required tests

Create the following tests before claiming Chapter 3 parity.

### 8.1 Unit tests

- Manifest uniqueness, label/role mapping, exact 8-channel order, all-missing channel rejection, QC reason codes, and technical-metadata exclusion.
- Frozen-fold determinism and no group leakage.
- Interpolation only within allowed gaps.
- Zero-phase filter impulse/peak-location tests and short-signal policy.
- IMU units, low-pass, gravity-removal, magnitude, and jerk tests.
- Window start/end/hop/end-alignment/cap/padding-mask tests.
- Robust scaling degeneracy, fallback scale, clipping, and non-mutation of the amplitude-preserving feature view.
- Sinusoid spectral-band, dominant-frequency, centroid, power, and normalized-entropy tests.
- Synthetic pulse/PPI/PRV formula tests with true timestamps, rejected intervals, adjacency gaps, five-minute eligibility, and SampEn eligibility.
- Valley-to-valley morphology tests with known amplitude, timing, width, slope, and area.
- Direct-signal RED/IR PI, ratios, conventional `R` direction, correlation, lag, and coherence tests.
- `Q_rate`/`Q_morph` component behavior and fold-local fit tests on direct signals.
- Artifact reducer alignment, no-op identity, finite output, failure status, and provenance tests.
- **Non-identity artifact test:** verify that only `Q_rate_post` is computed, `Q_morph_post=not_applicable`, and all morphology/amplitude-dependent outputs are invalid/NaN.
- Feature registry order/hash, allowlist, vector validity, matrix `K=32`, uniform sampling, padding, and row-mask tests.
- Model parameter/output-shape tests.
- Inception ensemble member independence, deterministic distinct seeds, and exact probability averaging.
- ROCKET transform fit only on training recordings; ridge fit/selection only inside outer training; serialization parity.
- Aggregation invariance and duplication-sensitivity tests.
- Bundle save/load/golden-logit round trip.

### 8.2 Integration tests

- One synthetic participant per class through manifest -> preprocessing -> direct quality -> pulse/features -> raw/vector/matrix model input.
- One real training fold/held-out fold smoke run with all fitted objects checked for training IDs only.
- Matched CompactCNN/Inception/ShapeFormer protocol smoke run.
- Raw, feature-vector, feature-matrix, and corrected fusion smoke runs on identical frozen folds.
- Feature-matrix InceptionTime smoke run.
- ROCKET 10,000-kernel reduced smoke configuration plus full-config serialization test.
- Five-member ensemble reduced smoke configuration plus exact member-average assertion.
- Motion path:

  ```text
  x_filter -> non-identity reducer -> x_ar -> Q_rate_post
           -> common pulse detector -> HR/PPI/eligible PRV
           -> processed file/role representation
  ```

  Assert that no `Q_morph` or morphology feature is produced.
- External ECG fixture: one-to-one peak matching, HR/PPI error, coverage, and raw-vs-quality-vs-reducer comparison.
- OOF writer: exactly one held-out participant probability per fold/seed/config, with member probabilities when applicable.
- CLI smoke from manifest build through report generation in a clean temporary directory.
- Stale-cache rejection after changing config, schema, source hash, fold file, or commit.

### 8.3 Leakage and statistical/regression tests

- No participant appears in more than one split.
- Imputer/scaler, quality calibrator, feature selector, ROCKET transform, ridge selection, shapelets, calibration, and early-stopping validation are fitted only from outer-training participants.
- Changing held-out values cannot change training-fold scaler/SQI references, selected epoch, ROCKET kernels, or feature selection.
- Full-file features are never exposed to an individual raw window before file pooling in the reference fusion route.
- Deterministic same-seed rerun tolerance.
- Label-shuffle performance sanity check.
- Duplicate-window challenge to verify window->file->participant aggregation.
- Technical-metadata ablation.
- Resampling/kernel physical-time ablation.
- Fixed epoch versus inner-selected epoch, never outer-selected.
- Paired fold/seed delta calculation.

### 8.4 Scientific benchmark outputs

- Peak/HR/PPI: event precision/recall/F1 at declared tolerances, timing error, HR MAE/RMSE/bias, PPI MAE, and coverage by dataset/activity.
- Artifact policies: raw/no-denoise, quality-only, and each reducer under identical external participant splits.
- Frailty classification: raw/vector/matrix/fusion on the same participant folds and seeds; participant BA, macro-F1, per-class metrics, confusion, and coverage.
- Paired ablations: preprocessing, quality, artifact rate route, feature families, ensemble, ROCKET, and fusion; one factor at a time.
- Locked evaluation: selected full config reconstructed from a serialized manifest, not a shortened leaderboard row.

## 9. Explicit no-go list

- Do not change participant labels or remove hard participants to improve accuracy.
- Do not use the outer held-out fold for early stopping, feature selection, shapelet discovery, threshold fitting, calibration, ROCKET fitting, or architecture choice.
- Do not call a single Inception network the original InceptionTime ensemble.
- Do not call the compact CNN an exact Wang-FCN implementation.
- Do not claim ShapeFormer parity while using a different discovery method without naming it.
- Do not calculate amplitude-sensitive SQI from normalized DL windows.
- Do not repeat a file feature vector per raw window in the reference fusion route.
- Do not aggregate raw windows directly to participants while bypassing file-level aggregation.
- Do not use technical metadata as default predictors.
- Do not encode unavailable physiological features as valid zeros.
- Do not compute or report `Q_morph` on non-identity `x_ar`.
- Do not extract or copy morphology, AC/DC, PI, ratio-of-ratios, slope, area, width, or amplitude-dependent optical features from non-identity `x_ar`.
- Do not claim clean-waveform or morphology recovery from PTT-PPG without paired morphology ground truth.
- Do not silently reuse a cache whose provenance hash differs.
- Do not overwrite historical results.
- Do not add unrelated architectures before the current families and required representation routes share one protocol.
- Do not edit thesis prose to conceal an implementation gap; either implement the claim or mark it proposed/exploratory.

## 10. Definition of done

The aligned implementation is complete only when all applicable items exist and pass:

- `configs/reference_static_v1.yaml` with no hidden defaults;
- versioned manifest, QC report, and frozen folds;
- unified signal views, preprocessing, window planner, and feature registry with tests;
- CompactCNN1D, full/small Inception ports, and eligible ShapeFormer under the same Trainer/Evaluator;
- canonical `representation_mode` supporting `raw`, `feature_vector`, `feature_matrix`, and `fusion`;
- `OrderedFeatureMatrixV1` classified by InceptionTime and ROCKET+ridge;
- optional five-member InceptionTime ensemble implemented with independently trained members and exact probability averaging before that name is used in results;
- corrected file-level fusion, with the file feature vector concatenated once after raw-window pooling;
- complete OOF window/file/subject probabilities and ensemble-member probabilities when applicable;
- no outer-fold fitting verified by tests;
- non-identity artifact route returns only rate-qualified HR/PPI/eligible PRV and marks morphology unavailable;
- final bundle round-trip parity;
- CPU CI green;
- generated model cards stating roles, representation mode, signal route, limitations, absence/presence of independent testing, and all deviations from original architectures;
- Chapter 3 describes only configurations and modules that have actually passed their acceptance tests.

## 11. Recommended commit sequence

1. `audit: freeze dev0 baseline and characterization artifacts`
2. `adr: freeze scientific contracts and representation modes`
3. `data: add versioned manifest qc and frozen participant folds`
4. `signal: add typed signal views unified window plan and provenance cache`
5. `features: add FeatureRegistryV1 prv morphology dual-wavelength and spectral parity`
6. `quality: add fold-local direct-signal Q_rate and Q_morph`
7. `models: snapshot compact cnn and inception single-network ports`
8. `representations: add feature vector and ordered K32 feature matrix`
9. `rocket: add fold-local ROCKET ridge feature-matrix route`
10. `ensemble: add optional five-member Inception probability ensemble`
11. `train: unify trainer and remove outer-fold early stopping`
12. `fusion: implement file-bag pooling and concatenate file features once`
13. `evaluate: add window-file-participant aggregation and complete OOF artifacts`
14. `bundle: persist complete preprocessing feature schemas and golden inference parity`
15. `test: add CPU CI and leakage cache regression suite`
16. `shapeformer: self-contain discovery and add patch/downsample route`
17. `motion: add artifact interface and rate-only x_ar return path`
18. `docs: synchronize README model cards and Chapter 3 evidence table`

## 12. Current files that must be treated carefully

- `frailty_3class_classifier.py`: current monolithic implementation; use as a characterization source, then split gradually.
- `frailty_3class_overfitting_sweep.py`: contains the stricter fixed-epoch path; migrate protocol behavior, not ad-hoc result assumptions.
- `frailty_3class_holdout_eval.py`: current locked-evaluation reconstruction is incomplete; replace shortened-row reconstruction with full serialized configuration loading.
- `analyze_sweep.py`: preserve historical reports, but extend configuration identity before matched ranking.
- `shapeformer_port.py`: experimental port; remove external-path and outer-fold dependencies.
- `frailty_3class_cnn_fusion.py`: historical/parallel fusion path; do not make it canonical without behavior comparison.
- `ppg.py`, `funcs.py`: older signal/feature implementations; use parity tests before retiring.
- `ppg_peak_hr_gating_train.py` and `pttppg_*`: motion/heartbeat prototypes; inventory and wrap behind explicit interfaces rather than importing opportunistically.
- `_agent/MODULES.md`, `_agent/TODO.md`: useful historical inventory, but this contract and accepted ADRs govern the implementation.
- `results_frailty3/`: immutable historical evidence; new runs go to a versioned output root.

---

**Completion principle:** first make the experiment impossible to run incorrectly; then compare representations and architectures; only then optimize performance.
