# Codex Implementation Contract — PPG Frailty Pipeline (`dev0`)

**Baseline repository:** `KaifengXuuu/PPG-Frailty-pipeline`  
**Baseline branch:** `dev0`  
**Audited commit:** `2eca0ecf0e17a4deaa1d3cc8e821098e5848e421`  
**Contract date:** 2026-08-14  
**Primary objective:** make the implementation reproducibly match the thesis Chapter 3 workflow without changing scientific claims merely to improve scores.

> This file is an implementation contract, not a brainstorming document. Follow the phases and acceptance criteria in order. Do not perform a broad rewrite, delete historical scripts, change labels, or tune on outer folds unless a task below explicitly authorizes it.

## 1. Source-of-truth order

When sources conflict, use this order and record the resolution in an Architecture Decision Record (ADR):

1. **Scientific invariants**: participant grouping, no held-out fitting, signal units, target labels, recording roles, and declared evaluation unit.
2. **This implementation contract and accepted ADRs.**
3. **The thesis Chapter 3 intended workflow**, after resolving its internal contradictions (notably aggregation order and PRV eligibility).
4. **Current `dev0` behavior**, preserved behind a `legacy_*` route only where needed for historical result replay.
5. **Original model papers/repositories**, used to label deviations accurately; they do not override project-specific scientific requirements.

## 2. Non-negotiable scientific invariants

- The targets remain `Pre-Frail`, `Robust/Non-Frail`, and `Young`; do not relabel or merge classes in the reference three-class experiment.
- Participant identity is the grouping variable. No participant, file, raw window, feature vector, shapelet candidate, scaler fit, quality threshold, calibrator, or model-selection observation may cross from an outer held-out fold into training.
- All branches consume the same versioned record manifest and the same frozen participant fold assignment.
- The internal acquisition grid remains 400 Hz for `x_native`, morphology, peak timing, and audit storage. A **separate DL-only anti-aliased resampling option** may be tested; never silently overwrite the acquisition rate.
- The external PTT-PPG dataset is used to develop/evaluate motion processing and heartbeat recovery. It must not supply frailty labels or classifier training participants.
- Outer held-out folds are evaluation-only. Epoch selection must use a training-only inner grouped split or a pre-registered fixed epoch.
- Every fitted object is fold-local: imputer, scaler, feature selector, SQI normalization/threshold, shapelet discovery, ROCKET transform, calibration, and network weights.
- Every reported participant prediction must be traceable to its windows, file, role, fold, seed, config hash, code commit, preprocessing schema, feature schema, and aggregation rule.
- Do not optimize toward a target balanced accuracy. Correctness and protocol comparability are the completion criteria.

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

Recommended resolution for ADR-006: `window probabilities -> file probability -> role-aware participant probability`. Keep direct all-window participant averaging only as a named ablation. Recommended PRV policy: full VLF/LF/HF and entropy only for qualified long static recordings; short task records expose only eligible time-domain metrics.

## 4. Target package boundary

Do not delete current scripts in Phase P0. Add a new package and convert old scripts to thin wrappers only after characterization tests pass.

```text
src/ppg_frailty/
  config.py
  provenance.py
  data/
    schema.py
    manifest.py
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
  peaks/
    aboy_project.py
    intervals.py
  features/
    registry.py
    prv.py
    morphology.py
    spectral.py
    file_vector.py
    ordered_matrix.py
  models/
    compact_cnn.py
    inception_time_port.py
    shapeformer_port.py
    feature_models.py
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
manifests/
splits/
tests/
```

## 5. Phase P0 — thesis-defensible static reference

### P0.1 Baseline inventory and characterization

- Record `2eca0ecf0e17a4deaa1d3cc8e821098e5848e421`, Python/package versions, CUDA/cuDNN, file fingerprints, manifest counts, and historical result paths.
- Add characterization tests around current preprocessing, model parameter counts, and known historical metrics before changing behavior.
- Preserve historical behavior under `legacy_v0`; never overwrite old result directories.

**Required output:** `artifacts/audit/baseline_inventory.json` and `artifacts/audit/legacy_characterization.json`.

### P0.2 Versioned manifest, QC, and frozen folds

Implement one manifest row per recording with at least:

`record_id, participant_id, class_id, class_name, role, source_path, source_hash, fs, n_samples, duration_s, channel_schema, qc_status, qc_reasons, manifest_version`.

QC reason codes must cover parse failure, missing required channel, non-finite gap, insufficient duration, flatline, clipping/saturation, implausible scale, and duplicate record. Never silently skip a record.

Generate `splits/sgkf5_v1.csv` once from the manifest. All models and feature branches read it; they do not regenerate folds.

**Acceptance criteria**

- Exactly one fold per participant.
- No participant appears in two folds.
- Every eligible file inherits its participant fold.
- Fold file has a content hash embedded in every run.
- Manifest count and class/role count audit matches the expected internal cohort or explicitly reports deviations.

### P0.3 Unified signal views and window planner

Implement typed objects for:

- `x_native`: interpolated acquisition-scale RED/IR.
- `x_filter`: detrended, zero-phase 0.2–8 Hz analysis RED/IR.
- `x_ar`: optional artifact-reduced analysis RED/IR on the original time grid.
- processed IMU profile and derived magnitudes/jerk.
- DL windows with explicit `fs`, `start_sample`, `end_sample`, `valid_length`, and `padding_mask`.

For offline reference processing, never silently fall back to a causal filter. Short records must be rejected, padded under an explicit tested policy, or processed with a documented reduced `padlen` policy.

Use one `WindowPlan` implementation for engineering and DL windows. End alignment, overlap, cap, and padding are configuration fields, not hidden defaults.

Add DL-only anti-aliased resampling with candidate `dl_fs` values `(100, 160, 200, 400)`. The feature branch remains on the 400 Hz analysis grid unless a feature definition explicitly states otherwise.

### P0.4 Feature Registry V1

Create a frozen, ordered registry with formula, units, source signal view, eligibility rule, aggregation rule, and missing-value policy for every feature.

Required corrections:

- Preserve peak timestamps and a valid-interval mask; do not compress time after rejecting PPIs.
- Implement the declared PRV set or remove unsupported items from the thesis. Long-record frequency/entropy eligibility must be explicit and tested.
- Implement valley-to-valley morphology using the local linear baseline; use `Q_morph`; provide median/MAD as the reference robust aggregation.
- Separate PPG and IMU spectral bands. Normalize spectral entropy exactly as declared.
- Compute dual-wavelength zero-lag correlation, maximum normalized cross-correlation, and lag under a fixed window.
- Exclude technical fields (`n_rows`, `duration_sec`, `n_windows`, filenames, path-derived fields) from model input unless a pre-registered ablation explicitly includes them.

Create:

- `FeatureVectorV1`: one complete file-level vector.
- `OrderedFeatureMatrixV1`: `K=32` ordered engineering positions, a row mask, and fixed recording-context channels.

### P0.5 Endpoint SQI without leakage

Implement separate `Q_rate` and `Q_morph`. Components must expose raw value, normalized value, pass/fail, and reason:

- cardiac-band concentration;
- periodicity / peak density;
- PPI stability;
- RED–IR agreement;
- IMU motion energy;
- flatline, clipping, long-gap exclusion.

SQI is calculated from amplitude-preserving analysis signals and processed IMU, not from median/IQR-normalized DL windows. Any learned normalization or threshold is fit on training participants only. Apply the frozen transformer to the held-out participants.

### P0.6 Freeze the three model families

Do not redesign architectures before protocol parity is established.

1. Rename current CNN to `CompactCNN1D`. Preserve 32/64/128, kernels 9/9/7, two pool-4 operations, and GAP as the legacy/reference architecture. Document that it is not the Wang-FCN reproduction.
2. Preserve full and small InceptionTime ports as reference variants. Label them `single_network`; never call one network the full five-network InceptionTime ensemble.
3. ShapeFormer remains `experimental` until P0.8 and P1.1 pass. Do not include its historical outer-fold-selected scores in a matched reference leaderboard.

Add architecture snapshot tests:

- CompactCNN1D trainable parameters: `79,139` for 8 channels / 3 classes.
- Full Inception port: `456,579`.
- Small Inception port: `57,027`.
- Output shape is `[batch, 3]` for all valid configured lengths.

### P0.7 Correct file-level signal–feature fusion

The current per-window repeated-feature fusion must become a named legacy route.

Implement `FileBagDataset`:

1. Encode each raw window.
2. Pool window embeddings to one file embedding with a mask-aware mean as the reference; attention pooling may be a later ablation.
3. Encode `FeatureVectorV1` once with the feature MLP.
4. Concatenate the two file embeddings once.
5. Produce one file probability.
6. Aggregate files to participant according to ADR-006.

Persist the exact imputer and scaler in the final bundle. The loaded bundle must reproduce training-side logits for golden records.

### P0.8 Unified trainer and outer-fold isolation

Create one Trainer/Evaluator used by CNN, Inception, and ShapeFormer. It receives a frozen outer split and must not expose outer labels to the trainer.

Allowed epoch rules:

- pre-registered fixed epoch; or
- inner participant-grouped train/validation split, after which the outer model is refit on all outer-training participants using the selected epoch.

Disallowed:

- selecting the best epoch on outer held-out windows;
- selecting SQI thresholds, shapelets, scalers, or sampler settings using outer labels;
- comparing a fixed-epoch model with an outer-early-stopped model as if architecture were the only difference.

Write complete OOF artifacts:

```text
oof_window_predictions.parquet
oof_file_predictions.parquet
oof_subject_predictions.parquet
metrics_per_fold_seed.json
confusion_matrices.json
run_manifest.json
```

Each row includes participant, file, role, fold, seed, config hash, probability vector, quality, retained/dropped status, and provenance hashes.

### P0.9 Aggregation and metrics

Reference hierarchy: window -> file -> participant. File aggregation is quality-weighted only when quality varies at window level and passes validation; otherwise report ordinary mean.

Metrics use participant as the independent unit:

- balanced accuracy;
- macro-F1;
- per-class precision/recall/F1;
- worst-class recall/F1;
- confusion matrix;
- between-repeat mean/SD and a clearly defined confidence interval;
- paired deltas on identical folds/seeds.

Do not rank incomplete configurations. Preserve the current ranking priority only after all candidates share the same protocol.

### P0.10 Reproducible bundle and CPU CI

Bundle must contain:

- model class/version and state;
- class mapping and channel order;
- signal/preprocessing configuration;
- DL resampler and window plan;
- feature registry hash and feature columns;
- fitted imputer/scaler/selector/calibrator;
- file pooler and aggregation rule;
- code commit, environment, manifest/fold hashes;
- golden inference examples.

Add CPU CI for lint/import/unit/synthetic integration tests. Full-data/GPU runs may remain manual or scheduled but must emit machine-verifiable manifests.

## 6. Phase P1 — model time-scale audit and dynamic branch

### P1.1 ShapeFormer repair

- Remove the hard-coded upstream path. Pin/vendor the required PISD code or implement a local equivalent.
- The discovery method is a required config field; never silently fall back from PISD to effect-size discovery.
- Remove outer-fold epoch selection.
- Apply the same SQI, sampler, loss, folds, aggregation, and OOF writer as the other models.
- Add anti-aliased downsampling or patch embedding before generic self-attention. Raw sample-token attention at 400 Hz is not the reference route.
- Record shapelet length in both samples and seconds.

### P1.2 Inception/CNN physical-time ablation

The current longest Inception kernel is 39 samples = 97.5 ms at 400 Hz. Six stacked longest branches have a theoretical local receptive field of 229 samples = 572.5 ms. Treat the current architecture as a valid local-morphology baseline, not as proof of pulse-cycle-scale suitability.

Run matched, parameter-conscious ablations:

- `dl_fs`: 100, 160, 200, 400 Hz;
- physical kernel-duration sets, converted to odd sample counts;
- optional dilation to expand receptive field without very large parameter growth;
- 5 s and one longer file-window context, with identical participant folds;
- raw only, complete feature only, and corrected file-level fusion.

Use the same seeds and outer folds. Report performance, calibration, train/held-out gap, runtime, memory, and weakest-class metrics. Do not select a configuration from one run.

### P1.3 Artifact-reduction interface and benchmark

Implement `ArtifactReducer` with `identity` first, then at least:

- IMU-referenced NLMS;
- one decomposition method;
- one spectral method if justified.

Every reducer returns aligned `x_ar`, diagnostics, and failure status. Recompute `Q_rate`/`Q_morph`; reject outputs that fail.

On PTT-PPG, lock the channel mapping and ECG R-peak reference. Report HR MAE, IBI/PPI error, coverage, failure rate, and waveform diagnostics by sitting/walking/running. Internal dynamic data receive visual/quality analysis only unless a valid ground truth exists.

## 7. Required tests

Create at least the following tests before claiming Chapter 3 parity:

### Unit

- manifest uniqueness, class/role counts, QC reason codes;
- frozen fold determinism and no group leakage;
- interpolation only within allowed gaps;
- zero-phase filter impulse/peak-location test;
- IMU units and gravity-removal synthetic orientation tests;
- window start/end/overlap/end-alignment/padding-mask tests;
- robust scaling degeneracy and clipping tests;
- sinusoid spectral-band and normalized entropy tests;
- synthetic peak/PPI/PRV formula tests with real timestamps and missing intervals;
- valley-to-valley morphology formula tests;
- RED/IR correlation and lag tests;
- Q_rate/Q_morph component and fold-local fit tests;
- artifact reducer alignment/no-op tests;
- feature registry order/hash and allowlist tests;
- model parameter/output shape tests;
- aggregation invariance and duplication sensitivity tests;
- bundle save/load/golden-logit round trip.

### Integration

- one synthetic participant per class through manifest -> preprocessing -> features -> model input;
- one real training fold / held-out fold smoke run with all fitted objects checked for training IDs only;
- matched CNN/Inception/ShapeFormer protocol smoke run;
- raw-only, feature-only, corrected fusion smoke run;
- OOF writer: exactly one held-out subject probability per fold/seed/config;
- stale-cache rejection after changing config, schema, source hash, or commit.

### Statistical/regression

- deterministic same-seed rerun tolerance;
- label-shuffle performance sanity check;
- duplicate-window challenge to verify selected aggregation hierarchy;
- technical-metadata ablation;
- resampling/kernel physical-time ablation;
- fixed epoch versus inner-selected epoch, never outer-selected;
- paired fold/seed delta calculation.

## 8. Explicit no-go list

- Do not change participant labels or remove hard participants to improve accuracy.
- Do not use the outer held-out fold for early stopping, feature selection, shapelet discovery, threshold fitting, calibration, or architecture choice.
- Do not call a single Inception network the original InceptionTime ensemble.
- Do not call the compact CNN an exact Wang-FCN implementation.
- Do not claim ShapeFormer parity while using a different discovery method without naming it.
- Do not calculate SQI from normalized DL windows when the component depends on amplitude, clipping, or flatline.
- Do not repeat a file feature vector per window in the reference fusion route.
- Do not silently reuse a cache whose provenance hash differs.
- Do not overwrite historical results.
- Do not add more architectures before the three current families share one protocol.
- Do not edit thesis prose to conceal an implementation gap; either implement the claim or mark it proposed/exploratory.

## 9. Definition of done

The static reference is complete only when all of the following exist and pass:

- `configs/reference_static_v1.yaml` with no hidden defaults;
- versioned manifest, QC report, and frozen folds;
- unified preprocessing and feature registry with tests;
- CompactCNN1D, full/small Inception port, and repaired ShapeFormer under the same Trainer/Evaluator;
- raw, feature-only, and corrected file-level fusion modes;
- complete OOF window/file/subject probabilities;
- no outer-fold fitting verified by tests;
- final bundle round-trip parity;
- CPU CI green;
- a generated model card that states dataset roles, limitations, absence/presence of independent testing, and all deviations from original architectures;
- Chapter 3 updated to describe only the configuration actually run.

## 10. Recommended commit sequence

1. `audit: freeze dev0 baseline and characterization artifacts`
2. `data: add versioned manifest qc and frozen participant folds`
3. `signal: add typed signal views unified window plan and provenance cache`
4. `features: add FeatureRegistryV1 prv morphology and spectral parity`
5. `quality: add fold-local endpoint SQI and routing diagnostics`
6. `models: snapshot compact cnn and inception ports`
7. `train: unify trainer and remove outer-fold early stopping`
8. `fusion: implement file-bag pooling and complete feature fusion`
9. `evaluate: add hierarchical aggregation metrics and OOF artifacts`
10. `bundle: persist complete preprocessing and golden inference parity`
11. `test: add CPU CI and leakage/cache regression suite`
12. `shapeformer: self-contain discovery and add patch/downsample route`
13. `motion: add artifact interface and PTT-PPG benchmark`
14. `docs: synchronize README model cards and Chapter 3 evidence table`

## 11. Current files that must be treated carefully

- `frailty_3class_classifier.py`: current monolithic implementation; use as a characterization source, then split gradually.
- `frailty_3class_overfitting_sweep.py`: contains the stricter fixed-epoch path; migrate protocol behavior, not ad-hoc result assumptions.
- `shapeformer_port.py`: experimental port; remove external-path and outer-fold dependencies.
- `frailty_3class_cnn_fusion.py`: historical/parallel fusion path; do not make it canonical without comparing behavior.
- `ppg.py`, `funcs.py`: older signal/feature implementations; use parity tests before retiring.
- `ppg_peak_hr_gating_train.py` and `pttppg_*`: motion/heartbeat prototypes; inventory and wrap behind explicit interfaces rather than importing opportunistically.
- `_agent/MODULES.md`, `_agent/TODO.md`: useful historical inventory, but this contract and accepted ADRs govern the implementation.
- `results_frailty3/`: immutable historical evidence; new runs go to a versioned output root.

---

**Completion principle:** first make the experiment impossible to run incorrectly; then compare architectures; only then optimize performance.
