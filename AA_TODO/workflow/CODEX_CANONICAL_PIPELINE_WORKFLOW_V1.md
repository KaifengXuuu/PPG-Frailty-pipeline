# Canonical End-to-End Workflow V1 — PPG Frailty Pipeline (`dev0`)

**Repository:** `KaifengXuuu/PPG-Frailty-pipeline`  
**Baseline branch:** `dev0`  
**Audited baseline commit:** `2eca0ecf0e17a4deaa1d3cc8e821098e5848e421`  
**Workflow version:** `canonical_workflow_v1`  
**Date:** 2026-08-16  
**Status:** binding implementation dataflow specification

This document defines the canonical pipeline dataflow that combines the reviewed thesis changes A1–A4 with the four classifier representation modes:

```text
raw
feature_vector
feature_matrix
fusion
```

It supplements `CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md`. Where that broader contract and this workflow differ on routing or representation semantics, **this workflow controls** until an explicit ADR replaces it.

This is not a brainstorming document. Do not silently invent alternative routes, substitute signal views, reuse held-out data, or change scientific semantics to improve scores.

---

## 1. Binding scientific resolutions

### 1.1 A1 — segment-level endpoint-aware routing

Replace the old top-level `static versus motion` binary logic with a segment-level state machine:

```text
hard_invalid
full_direct
rate_only_direct
rate_recovery_candidate
rate_only_processed
degraded_drop
rejected_after_reduction
```

A recording may contain segments with different states. The route is not inferred from the filename alone.

### 1.2 A2 — post-reducer quality reassessment is rate-only

A non-identity artifact reducer necessarily changes the waveform. Therefore:

```text
x_filter + processed IMU
  -> ArtifactReducer
  -> x_ar
  -> recompute Q_rate_post only
  -> common pulse detector
  -> HR / PPI / eligible PRV
```

Mandatory:

```text
Q_morph_post = not_applicable
```

A non-identity `x_ar` is a **rate-recovery signal**, not a morphology-preserving signal.

Do not calculate or retain as primary physiological predictors from non-identity `x_ar`:

- pulse amplitude;
- rise or decay time;
- pulse width;
- systolic slope;
- pulse area;
- AC or DC;
- PI;
- RED/IR AC or DC ratios;
- ratio-of-ratios;
- morphology coherence;
- absolute optical RMS, amplitude, or total power;
- any claim of recovered pulse shape.

Store these fields as:

```text
value = NaN
validity = false
reason = not_applicable_after_artifact_reduction
```

The `identity` reducer is equivalent to the direct branch and may retain `Q_morph`, because it does not change the waveform.

### 1.3 A3 — external PTT-PPG ECG-reference benchmark

The external PTT-PPG dataset is used only for motion-processing and heartbeat-recovery development. It must not supply frailty labels or frailty-classifier training observations.

Required external flow:

```text
PTT-PPG record at public 500 Hz grid
  -> locked channel mapping and preprocessing
  -> same ArtifactReducer interface
  -> x_ar
  -> Q_rate_post only
  -> same canonical pulse detector
  -> HR / PPI / eligible PRV
  -> one-to-one event matching against ECG R-peaks
  -> endpoint metrics stratified by sitting / walking / running
```

Required metrics:

- event precision, recall, and F1 under a declared timing tolerance;
- matched-event timing error;
- HR MAE;
- PPI/IBI MAE;
- accepted coverage;
- reducer failure rate;
- route counts;
- RED/IR rate agreement where applicable.

Waveform plots are secondary diagnostics and must not be interpreted as clean-waveform or morphology recovery.

### 1.4 A4 — internal dynamic evaluation includes downstream quality evidence

Internal W1/W2 and S1/S2 recordings have no clean optical waveform ground truth. Report:

- hard-invalid coverage;
- `full_direct`, `rate_only_direct`, `rate_only_processed`, and rejected coverage;
- `Q_rate_pre` and `Q_rate_post` distributions;
- accepted-event coverage;
- RED/IR rate agreement;
- reducer failure and route stability;
- HR/PPI plausibility;
- waveform/peak/route plots only as auxiliary diagnostics.

Do not report internal denoising waveform accuracy without valid ground truth.

### 1.5 Reference aggregation

The canonical hierarchy is:

```text
window -> file -> role -> participant
```

Reference behavior:

- SQI controls eligibility and routing only.
- Raw window probabilities are combined within a file by ordinary mean.
- Quality-weighted within-file aggregation is a named ablation, not the default.
- Files do not receive weight proportional to raw window count.
- Same-role files are aggregated first if more than one exists.
- Role probabilities are aggregated according to a frozen role-aware rule.
- Direct all-window participant averaging is legacy/ablation only.

### 1.6 Reference raw-signal source

The raw-signal side of `raw` and `fusion` uses:

```text
x_filter + processed IMU
```

Do not silently substitute non-identity `x_ar` for the PPG waveform.

A secondary dynamic experiment may explicitly configure:

```text
raw_motion_policy = original_x_filter | drop_degraded | processed_waveform_ablation
```

`processed_waveform_ablation` must be named as such and must not be the reference route.

---

## 2. Shared scientific invariants for all modes

- Targets remain `Pre-Frail`, `Robust/Non-Frail`, and `Young`.
- Participant identity is the grouping unit.
- All modes read the same versioned manifest and frozen fold file.
- No participant, file, window, feature, matrix row, shapelet candidate, scaler, SQI calibrator, threshold, ROCKET transform, probability calibrator, or model-selection observation may cross from outer held-out data into fitting.
- Outer held-out folds are evaluation-only.
- Epoch selection uses a pre-registered fixed epoch or an inner participant-grouped split from outer-training data.
- Internal acquisition-scale signal and feature extraction remain at 400 Hz unless a feature definition explicitly states otherwise.
- DL-only anti-aliased resampling is a separate, explicit configuration.
- Every output is traceable to participant, file, role, source route, fold, seed, config hash, code commit, manifest hash, split hash, preprocessing schema, and feature schema.
- Technical metadata is not a default predictor.

---

## 3. Canonical hierarchy and objects

```text
Participant
  -> Recording/File
       -> RoutingSegment
       -> DetectorBlock
       -> EngineeringWindow
       -> DLWindow
       -> Beat/PPI interval
```

Implement typed immutable or validation-enforced contracts.

### 3.1 `ManifestRow`

Required fields:

```text
record_id
participant_id
class_id
class_name
role
source_path
source_hash
source_version
fs
n_samples
duration_s
channel_schema
channel_units
synchrony_status
reference_available
qc_status
qc_reasons
manifest_version
```

### 3.2 `SignalViews`

```python
SignalViews(
    x_native_red,
    x_native_ir,
    x_filter_red,
    x_filter_ir,
    imu_axes,
    acceleration_magnitude,
    angular_rate_magnitude,
    jerk_magnitude,
    fs,
    record_id,
    preprocessing_version,
    provenance,
)
```

`x_ar` is not stored as the universal analysis signal. It is returned by a route-specific `ArtifactResult`.

### 3.3 `QualityResult`

```python
QualityResult(
    q_rate_score,
    q_rate_pass,
    q_shape_score,
    q_morph_state,  # pass | fail | not_applicable | missing
    hard_integrity_pass,
    components,
    reasons,
    start_sample,
    end_sample,
    calibrator_version,
)
```

`q_morph_state=not_applicable` must be a distinct state, not encoded as zero, false, or pass.

### 3.4 `RouteResult`

```python
RouteResult(
    state,
    source_signal,      # x_filter | x_ar | none
    reducer_name,
    reducer_version,
    reducer_status,
    q_pre,
    q_post,
    start_sample,
    end_sample,
    reasons,
    provenance,
)
```

### 3.5 `PulseResult`

```python
PulseResult(
    peak_indices,
    peak_times_s,
    accepted_peak_mask,
    ppi_left_peak_indices,
    ppi_right_peak_indices,
    ppi_values_s,
    valid_interval_mask,
    adjacency_mask,
    run_id_per_interval,
    source_route,
    wavelength,
    detector_version,
)
```

Rejecting intervals must not compress time or create false adjacency.

### 3.6 Feature contracts

```python
FeatureVectorV1(
    values,
    validity,
    reasons,
    ordered_names,
    units,
    source_routes,
    schema_version,
    schema_hash,
    provenance,
)
```

```python
EngineeringFeatureSequence(
    values,              # [W, D_window]
    start_samples,
    end_samples,
    valid_row_mask,
    per_family_validity,
    ordered_names,
    schema_version,
    provenance,
)
```

```python
OrderedFeatureMatrixV1(
    values,              # [D, K]
    row_mask,            # [K]
    channel_names,
    context_channel_names,
    schema_version,
    schema_hash,
    transforms_hash,
    provenance,
)
```

---

## 4. Stage 0 — manifest, QC, roles, and frozen folds

### 4.1 Internal data

Expected operational cohort:

```text
29 participants
261 recordings
8 channels: RED, IR, AX, AY, AZ, GX, GY, GZ
400 Hz
```

Roles:

```text
Static/reference: B, R1, R2, R3, R4   -> 145 recordings
Dynamic:          W1, W2, S1, S2       -> 116 recordings
```

Primary static and secondary all-role/dynamic experiments must use different named configs and output roots.

### 4.2 QC

QC reason codes must cover at least:

```text
parse_failure
missing_required_channel
all_nonfinite_channel
excessive_nonfinite_gap
insufficient_duration
flatline
clipping_or_saturation
implausible_scale
timestamp_failure
synchrony_failure
duplicate_record
```

Never silently skip a record. Never replace an unavailable required channel with zeros.

### 4.3 Folds

Create a frozen split artifact once:

```text
splits/sgkf5_v1.csv
```

All four representation modes and all ablations read this file. They do not regenerate folds.

Tests:

- exactly one fold per participant;
- no participant in multiple folds;
- all files inherit participant fold;
- split hash stored in every run;
- changing representation mode does not change folds.

---

## 5. Stage 1 — common signal preparation

### 5.1 `x_native`

Construct baseline-preserving RED and IR views:

- convert to numeric;
- interpolate only allowed isolated gaps;
- preserve acquisition scale and optical baseline;
- store units and provenance;
- do not robust-normalize.

Use only for direct DC-dependent optical quantities and audit/alignment.

### 5.2 `x_filter`

Construct direct analysis RED and IR:

```text
x_native copy
  -> least-squares linear detrending
  -> third-order Butterworth 0.2–8 Hz
  -> SOS
  -> offline zero-phase filtering
```

Do not silently fall back to a causal filter. Short records use an explicit tested drop/pad/reduced-padlen policy.

### 5.3 Processed IMU

Reference profile:

```text
accelerometer axes -> third-order low-pass 20 Hz
gyroscope axes     -> third-order low-pass 40 Hz
gravity estimate   -> second-order low-pass 0.3 Hz
dynamic acceleration = filtered acceleration - gravity estimate
```

Derived signals:

```text
A     = ||a_dynamic||
Omega = ||gyro||
J     = ||d(a_dynamic)/dt||
```

A calibrated roll–pitch EKF may exist as an explicit alternative profile. One run must use one profile consistently in routing, artifact reduction, engineering features, and raw input.

### 5.4 Window plans

Use one `WindowPlan` implementation with separate named plans.

```text
routing_plan:     explicit config; no hidden default
peak_block_plan:  10 s non-overlapping
engineering_plan: 10 s, reference hop 5 s
raw_dl_plan:      reference 5 s, 50% overlap
matrix_plan:      K=32 ordered engineering positions
```

Each window stores:

```text
record_id
participant_id
role
start_sample
end_sample
valid_length
padding_mask
source_route
```

The Dash 10 s / 2 s overlapping detector wrapper is visualization-only and must not be called by the canonical feature/classification route.

### 5.5 DL normalization

For each channel within each DL window:

```text
center = median(x)
scale  = IQR(x) / 1.349
fallback = ordinary SD if robust scale degenerates
final fallback = explicit configured finite scale
z = clip((x - center) / scale, -8, 8)
```

Handcrafted features are extracted before this normalization.

---

## 6. Stage 2 — A1/A2 endpoint-aware route state machine

### 6.1 Hard integrity gate

For each routing segment:

```python
if not hard_integrity_pass(segment):
    state = HARD_INVALID
    do_not_run_reducer()
    emit_qc_only()
```

Hard invalid includes missing data, long gaps, flatline, clipping/saturation, invalid duration, alignment failure, or other non-recoverable integrity failure.

### 6.2 Direct quality scores

Compute on amplitude-preserving `x_filter` and processed IMU:

```text
Q_rate_pre
Q_shape_pre
morph_pass = rate_pass AND Q_shape_pre >= tau_shape
```

Expose `Q_morph_pre` as a derived endpoint state if desired, but keep the underlying `Q_shape_pre` available for audit.

No SQI component is calculated from robust-normalized DL windows when it depends on amplitude, clipping, flatline, or absolute motion scale.

Any learned normalization/threshold is fitted on outer-training participants only.

### 6.3 Canonical routing pseudocode

```python
def route_segment(segment, config, fold_objects):
    integrity = hard_integrity(segment)
    if not integrity.pass_:
        return RouteResult(state="hard_invalid", source_signal=None)

    q_pre = compute_direct_endpoint_quality(
        x_filter=segment.x_filter,
        imu=segment.imu,
        fitted_calibrator=fold_objects.sqi_calibrator,
    )

    if q_pre.q_rate_pass:
        if q_pre.q_morph_pass:
            return RouteResult(
                state="full_direct",
                source_signal="x_filter",
                q_pre=q_pre,
            )
        return RouteResult(
            state="rate_only_direct",
            source_signal="x_filter",
            q_pre=q_pre,
        )

    if not recoverable_motion(segment, q_pre):
        return RouteResult(state="degraded_drop", source_signal=None, q_pre=q_pre)

    if config.artifact.reducer == "disabled":
        return RouteResult(state="degraded_drop", source_signal=None, q_pre=q_pre)

    result = artifact_reducer(segment.x_filter, segment.imu, config.artifact)
    if not result.success:
        return RouteResult(
            state="rejected_after_reduction",
            source_signal=None,
            q_pre=q_pre,
            reducer_status=result.status,
        )

    q_rate_post = compute_rate_quality_only(
        x_ar=result.x_ar,
        imu=segment.imu,
        fitted_calibrator=fold_objects.sqi_calibrator,
    )

    # Mandatory scientific boundary.
    q_morph_post = NOT_APPLICABLE

    if q_rate_post.pass_:
        return RouteResult(
            state="rate_only_processed",
            source_signal="x_ar",
            q_pre=q_pre,
            q_post=q_rate_post,
            reducer_name=result.name,
            reducer_status=result.status,
        )

    return RouteResult(
        state="rejected_after_reduction",
        source_signal=None,
        q_pre=q_pre,
        q_post=q_rate_post,
        reducer_name=result.name,
        reducer_status=result.status,
    )
```

### 6.4 Route semantics

| State | Rate endpoints | Morphology/optical endpoints | Direct optical engineering | IMU engineering | Reducer |
|---|---:|---:|---:|---:|---:|
| `full_direct` | yes | yes | yes | yes | no |
| `rate_only_direct` | yes | no | yes, subject to independent integrity rules | yes | no |
| `rate_only_processed` | yes | no / N/A | no in reference predictor schema | yes | yes |
| `hard_invalid` | no | no | no | QC only | no |
| `degraded_drop` | no | no | no | QC only | no |
| `rejected_after_reduction` | no | no | no | QC only | attempted |

Do not run a reducer merely because `Q_morph` fails when `Q_rate` passes.

Do not use a separate “motion override” that forces processing despite a valid rate endpoint. Motion is part of endpoint quality and recoverability assessment.

---

## 7. Stage 3 — common physiological and engineering analysis

### 7.1 Canonical pulse detector

The same detector API accepts:

```text
eligible direct x_filter
rate-qualified non-identity x_ar
```

The detector output must remain linked to real timestamps and route runs.

The detector must not use the Dash visualization wrapper.

### 7.2 HR, PPI, and PRV

Use `PulseResult` with true timestamps and adjacency.

Reference calculation rules:

- central and dispersion summaries may use all valid PPIs while preserving source/route coverage;
- RMSSD, SDSD, NN50, pNN50, SD1, and SD2 use only pairs where `adjacency_mask=true`;
- no successive metric crosses a gap, rejected interval, route change, reducer change, or recording boundary;
- spectral PRV uses one continuous eligible interval sequence and real timestamps;
- reference spectral PRV is restricted to qualified approximately five-minute static/reference recordings or a genuinely qualifying long processed run;
- SampEn requires at least 200 accepted intervals;
- unavailable values use `NaN + validity=false`.

Required output families:

```text
availability and coverage
mean/median PPI
mean/median pulse rate
PPI SD/IQR/MAD/CV
pulse-rate SD
RMSSD/SDSD/NN50/pNN50
SD1/SD2/SD1:SD2
VLF/LF/HF/LF:HF/normalized units when eligible
SampEn when eligible
```

### 7.3 Direct-only morphology

Source:

```text
x_filter for waveform
x_native for aligned DC-dependent values
```

Eligibility:

```text
route == full_direct
Q_morph_pre == pass
accepted valley-to-valley beat
```

Reference features:

```text
baseline-corrected amplitude
rise time
decay time
half-prominence width
mean systolic upslope
positive baseline-corrected area
```

Aggregate beat values by median and MAD.

### 7.4 Direct-only dual-wavelength optical features

Reference features:

```text
beatwise AC and DC
PI per wavelength
RED/IR AC ratio
RED/IR DC ratio
R = (AC_RED / DC_RED) / (AC_IR / DC_IR)
zero-lag Pearson correlation
maximum normalized cross-correlation
argmax lag
eligible cardiac-band coherence
```

Every denominator and finite condition has an explicit validity rule.

For `rate_only_direct`, continuous RED/IR engineering correlations may remain available under independent validity rules, but morphology-dependent beat/AC/DC features do not.

### 7.5 Engineering windows

Reference windows:

```text
10 s, 5 s hop, chronological start positions
```

Direct PPG descriptors:

```text
mean, population SD, RMS, IQR, MAD
bias-corrected skewness, Pearson kurtosis
total power, normalized spectral entropy
dominant frequency, spectral centroid
0.2–0.5, 0.5–3, 3–8 Hz powers
```

IMU descriptors:

```text
axis-specific time descriptors for AX–AZ and GX–GZ
A, Omega, J time descriptors
0.1–0.5, 0.5–3, 3–8, 8–20 Hz powers
```

For a non-identity `x_ar`, do not place optical amplitude/power/waveform descriptors into the reference predictor registry. Rate-quality diagnostics may be recorded separately.

Default file aggregates remain mean and SD across valid engineering rows unless a later accepted ADR changes the thesis schema.

### 7.6 Predictor allowlist and parallel metadata

Default predictors exclude:

```text
participant_id
record_id / filename / path
absolute file order
n_rows
file duration
number of generated windows
administrative missingness
route name
reducer identity
coverage and SQI unless a named quality-aware ablation enables them
```

Store all of the above in parallel metadata for audit and A4 diagnostics.

---

## 8. Stage 4 — four representation modes

Expose exactly one canonical field:

```yaml
representation_mode: raw | feature_vector | feature_matrix | fusion
```

Do not infer the mode from combinations of legacy booleans.

All modes use the same manifest, frozen folds, label mapping, role ontology, preprocessing contracts, and evaluator.

### 8.1 `raw`

#### Dataflow

```text
recording
  -> x_filter + processed IMU
  -> raw DL windows (reference 5 s, 50% overlap)
  -> per-channel robust normalization and clipping
  -> tensor [8, T]
  -> raw model
  -> p_window
  -> ordinary mean over eligible windows
  -> p_file
  -> role-aware participant aggregation
```

Models:

```text
CompactCNN1D
InceptionTime full/small single network
ShapeFormer only after protocol/self-contained eligibility gates pass
optional true five-member InceptionTime ensemble
```

Rules:

- Handcrafted feature extraction may be skipped for a pure raw run.
- `x_ar` is not silently substituted for PPG.
- A dynamic raw-source ablation must set `raw_motion_policy` explicitly.
- Padding must be masked or covered by an accepted tested policy.

### 8.2 `feature_vector`

#### Dataflow

```text
route-aware pulse and engineering outputs
  -> Feature Registry
  -> beat/window-to-file aggregation
  -> FeatureVectorV1 (one recording = one sample)
  -> outer-training median imputation
  -> outer-training standardization
  -> tabular classifier
  -> p_file
  -> role-aware participant aggregation
```

Models:

```text
L2 logistic regression
RBF SVM
Extra Trees
optional compact MLP
```

Rules:

- No raw waveform is supplied.
- Only the frozen allowlist enters the model.
- `x_ar` contributes rate and eligible PRV only, plus processed IMU descriptors.
- Direct-only morphology remains unavailable when no valid direct morphology exists.
- Imputer/scaler are persisted in the model bundle.

### 8.3 `feature_matrix`

#### Matrix construction

For a recording with `W` engineering windows:

```text
EngineeringFeatureSequence [W, D_window]
  + complete FeatureVectorV1 as recording-context channels
  -> outer-training imputation/scaling
  -> K=32 chronological positions
```

Rules:

```text
W > 32:
  choose 32 positions uniformly over recording progress
  preserve chronological order

W < 32:
  right-pad after fold-local transformation
  padded value = standardized zero
  row_mask = false
```

Time-varying channels contain only features valid for the corresponding route. Context channels are repeated across valid rows only.

#### Dataflow

```text
OrderedFeatureMatrixV1 [D, 32] + row mask
  -> feature-matrix classifier
  -> p_file
  -> role-aware participant aggregation
```

Models:

```text
InceptionTime single network
optional actual five-member InceptionTime probability ensemble
ROCKET with 10,000 kernels + ridge classifier
```

Rules:

- One recording is one sample.
- Matrix rows are never independent training samples.
- InceptionTime pooling must be mask-aware or use an equivalent tested padding policy.
- ROCKET must receive an explicit mask channel or another validated padding encoding.
- Padding length must not become an unintentional duration shortcut.

### 8.4 `fusion`

#### Dataflow

```text
Raw branch:
  x_filter + processed IMU
    -> normalized DL windows
    -> raw encoder
    -> window embeddings
    -> mask-aware mean pooling
    -> z_file

Feature branch:
  FeatureVectorV1
    -> outer-training imputation/scaling
    -> feature MLP
    -> e_file

Fusion:
  concat(z_file, e_file) once
    -> three-class head
    -> p_file
    -> role-aware participant aggregation
```

Rules:

- Never repeat the file feature vector once per window in the reference route.
- A file must have a valid raw bag and a valid transformed feature vector; do not silently fall back to feature-only.
- Recovered `x_ar` rate features may enter `FeatureVectorV1`.
- The raw branch remains based on `x_filter` by default.
- Persist the feature imputer/scaler, raw pooler, feature MLP, and classifier head together.

---

## 9. Stage 5 — training and outer-fold isolation

### 9.1 Required fitting order per outer fold

```python
for outer_fold in frozen_folds:
    train_records = manifest.outer_train(outer_fold)
    heldout_records = manifest.outer_heldout(outer_fold)

    # Fit only on training participants.
    fit_qc_or_sqi_calibration(train_records)
    fit_feature_imputer_scaler(train_records)
    fit_optional_feature_selector(train_records)
    fit_shapelet_discovery_if_needed(train_records)
    fit_rocket_transform_if_needed(train_records)
    select_epoch_or_hyperparameters_inside_outer_train_only(train_records)

    train_model(train_records)
    predict_heldout(heldout_records)
    write_oof_artifacts()
```

### 9.2 Allowed epoch rules

- pre-registered fixed epoch; or
- inner participant-grouped train/validation split, followed by refit on all outer-training participants for the selected epoch.

### 9.3 Disallowed

- outer held-out early stopping;
- fitting SQI bounds or thresholds on held-out windows;
- feature selection on the full cohort;
- global imputation/scaling before folds;
- shapelet discovery on held-out data;
- fitting ROCKET kernels or ridge alpha on held-out data;
- probability calibration using predictions from a model trained on the same calibration participant;
- using different folds for different representation modes in a matched comparison.

### 9.4 Sampling and loss

Use explicit mutually interpretable config fields:

```yaml
sampling:
  mode: none | participant_balanced | class_participant_balanced
  quota: all | fraction | fixed_count

loss:
  name: standard_ce | weighted_ce | balanced_softmax | focal
  class_weight_rule: none | inverse_participant_count | effective_number
  label_smoothing: 0.0
```

Do not silently stack all imbalance methods.

---

## 10. Stage 6 — prediction aggregation and metrics

### 10.1 Reference aggregation

```python
# raw mode
p_file = mean(p_window for eligible windows in file)

# feature_vector / feature_matrix / fusion
p_file = model_output_for_file

# role-aware participant aggregation
p_role = equal_mean(file probabilities within role)
p_participant = frozen_role_rule(p_role values)
```

Quality-weighted aggregation is only:

```yaml
aggregation_mode: quality_weighted_within_file_ablation
```

It is not the reference participant weighting rule.

### 10.2 Required OOF artifacts

```text
oof_window_predictions.parquet
oof_file_predictions.parquet
oof_role_predictions.parquet
oof_subject_predictions.parquet
oof_member_predictions.parquet   # ensemble only
metrics_per_fold_seed.json
confusion_matrices.json
run_manifest.json
```

Every prediction row includes:

```text
participant_id
record_id
role
fold
seed
representation_mode
model_name
config_hash
probability vector
route / source signal
quality/eligibility status
coverage
manifest/split/schema/code hashes
```

### 10.3 Primary metrics

Participant is the independent unit:

- balanced accuracy;
- macro-F1;
- per-class precision, recall, and F1;
- worst-class recall and F1;
- confusion matrix;
- between-repeat mean and SD;
- clearly defined confidence interval;
- paired differences on identical folds and seeds.

Window and file metrics are diagnostics only.

Report by route/role:

- retained coverage;
- unavailable feature rate;
- direct versus processed rate coverage;
- reducer failures;
- quality distributions.

---

## 11. Stage 7 — A3 external benchmark and A4 internal dynamic report

### 11.1 External PTT-PPG runner

Implement a separate command/config, sharing signal, routing, reducer, pulse, and rate-feature modules:

```text
python -m ppg_frailty.cli benchmark-motion --config configs/motion_benchmark_v1.yaml
```

Required configuration identity:

```text
dataset version
record IDs
500 Hz sampling grid
selected PPG channel(s)
selected IMU channels
ECG annotation source
preprocessing profile
ArtifactReducer name/version/parameters
Q_rate calibrator/threshold
peak detector version
event timing tolerance
```

Required outputs:

```text
external_event_matches.parquet
external_rate_predictions.parquet
external_metrics_by_activity.json
external_failure_report.json
external_run_manifest.json
```

### 11.2 Internal dynamic report

Implement:

```text
python -m ppg_frailty.cli analyze-dynamic --config configs/internal_dynamic_v1.yaml
```

Required outputs:

```text
segment_routes.parquet
q_rate_pre_post.parquet
dynamic_coverage_by_role.json
reducer_failures.json
rate_agreement.json
selected diagnostic plots
```

No clean-waveform accuracy claim is allowed.

---

## 12. Canonical orchestrator pseudocode

```python
def run_experiment(config_path: str) -> RunBundle:
    cfg = load_and_validate_config(config_path)
    assert cfg.representation_mode in {
        "raw", "feature_vector", "feature_matrix", "fusion"
    }

    manifest = load_versioned_manifest(cfg.manifest)
    folds = load_frozen_folds(cfg.folds)
    validate_manifest_and_folds(manifest, folds)

    for fold in folds:
        train_ids, heldout_ids = fold.participant_ids()
        assert_no_group_overlap(train_ids, heldout_ids)

        # Fit all fold-local objects using training participants only.
        fold_objects = fit_fold_objects(cfg, manifest, train_ids)

        # Build training and held-out record artifacts using identical frozen transforms.
        train_records = build_record_artifacts(
            cfg, manifest.records(train_ids), fold_objects, fit=False
        )
        heldout_records = build_record_artifacts(
            cfg, manifest.records(heldout_ids), fold_objects, fit=False
        )

        train_input = build_representation(
            mode=cfg.representation_mode,
            records=train_records,
            fold_objects=fold_objects,
        )
        heldout_input = build_representation(
            mode=cfg.representation_mode,
            records=heldout_records,
            fold_objects=fold_objects,
        )

        model = train_mode_model(cfg, train_input, fold_objects)
        predictions = predict_mode_model(cfg, model, heldout_input)

        file_predictions = aggregate_to_file(cfg, predictions)
        role_predictions = aggregate_to_role(cfg, file_predictions)
        participant_predictions = aggregate_to_participant(cfg, role_predictions)

        write_fold_oof(
            cfg,
            fold,
            predictions,
            file_predictions,
            role_predictions,
            participant_predictions,
            fold_objects,
        )

    final_report = combine_oof_and_compute_participant_metrics(cfg)
    bundle = save_reproducible_bundle(cfg, final_report)
    return bundle
```

`build_record_artifacts()` must execute:

```text
QC
-> signal views
-> routing segments
-> route state machine
-> route-specific pulse/rate extraction
-> direct-only morphology/optical extraction
-> engineering sequence
-> FeatureVectorV1
-> OrderedFeatureMatrixV1 when requested
```

For `raw`, handcrafted artifacts may be omitted for efficiency unless needed for SQI, diagnostics, or matched analysis. The signal and routing definitions must remain identical.

---

## 13. Suggested module ownership

```text
src/ppg_frailty/
  pipeline.py                 canonical orchestrator
  contracts.py                enums and typed objects
  data/manifest.py            manifest construction/loading
  data/qc.py                  file-level QC
  data/folds.py               frozen folds
  signal/views.py             x_native/x_filter containers
  signal/ppg_preprocess.py    PPG processing
  signal/imu_preprocess.py    IMU profiles and A/Omega/J
  signal/window_plan.py       all window plans
  quality/endpoint_sqi.py     Q_rate/Q_shape/Q_morph state
  quality/routing.py          A1/A2 state machine
  artifact/base.py            ArtifactReducer protocol
  peaks/aboy_project.py       canonical detector
  peaks/intervals.py          timestamp-preserving PPI object
  features/registry.py        Feature Registry V1
  features/prv.py             HR/PPI/PRV
  features/morphology.py      direct-only morphology
  features/dual_wavelength.py direct-only optical features
  features/engineering.py     10 s window descriptors
  features/file_vector.py     FeatureVectorV1
  features/ordered_matrix.py  OrderedFeatureMatrixV1
  representations/raw.py
  representations/feature_vector.py
  representations/feature_matrix.py
  representations/fusion.py
  models/rocket_ridge.py
  models/file_fusion.py
  train/trainer.py
  evaluate/aggregate.py
  evaluate/oof.py
  evaluate/benchmark.py
  bundle/save.py
  bundle/load.py
```

Legacy scripts remain characterization wrappers until parity tests pass. Do not delete historical result paths.

---

## 14. Acceptance tests

### 14.1 Data and leakage

- manifest uniqueness and expected role/class counts;
- fixed fold determinism;
- no participant crosses outer folds;
- held-out sentinel changes do not alter fitted SQI, scaler, feature selector, shapelets, or ROCKET transform;
- every mode uses the same split hash.

### 14.2 Signal preparation

- interpolation only within allowed gaps;
- zero-phase filter impulse and peak-location tests;
- 0.2–8 Hz PPG passband test;
- 20/40 Hz IMU filter tests;
- gravity-removal synthetic orientation test;
- window start/hop/end alignment and mask tests;
- robust scaling fallback and `[-8,8]` clipping tests.

### 14.3 Routing

- hard invalid never enters reducer;
- `Q_rate pass + Q_morph fail` yields `rate_only_direct` and does not run reducer;
- `Q_rate fail + recoverable motion` can enter reducer;
- non-identity reducer recomputes only `Q_rate_post`;
- `Q_morph_post` is `not_applicable`;
- no morphology/AC/DC/PI/optical-power field becomes valid for non-identity `x_ar`;
- failed reducer yields `rejected_after_reduction`.

### 14.4 Pulse and features

- synthetic pulse trains and 35–210 bpm boundary tests;
- peak timestamps preserved after interval rejection;
- no RMSSD/SDSD bridge across invalid interval or route change;
- spectral PRV uses real timestamps and correct eligibility;
- morphology formula tests on direct signals;
- RED/IR ratio and lag tests;
- engineering band-power and normalized entropy tests;
- technical metadata excluded from predictor allowlist;
- feature schema/order/hash deterministic.

### 14.5 Representation modes

#### Raw

- input shape `[batch, 8, T]`;
- no file feature repeated as raw channel;
- `x_ar` not substituted by default;
- window probabilities aggregate to exactly one file probability.

#### Feature vector

- exactly one sample per recording;
- fold-local imputer/scaler only;
- no technical metadata in predictor matrix.

#### Feature matrix

- exactly one sample per recording;
- chronological order preserved;
- `K=32` selection deterministic;
- padding rows masked;
- duplicate/padding challenge does not create duration shortcut;
- ROCKET and Inception routes serialize and reload.

#### Fusion

- raw embeddings pooled before fusion;
- feature vector encoded exactly once per file;
- duplicating a raw window does not multiply feature-branch contribution;
- no silent fallback when one branch is unavailable.

### 14.6 Aggregation and evaluation

- window duplication does not change equal-file participant weighting;
- ordinary and quality-weighted modes are distinguishable when quality varies;
- exactly one held-out participant probability per fold/seed/config;
- participant metrics reproduce from saved OOF probabilities;
- paired comparisons align fold and seed.

### 14.7 A3/A4

- external ECG matching test with synthetic known offset;
- HR/PPI errors and coverage computed by activity;
- external data never enter frailty training manifest;
- internal dynamic report contains route/SQI/coverage even without waveform ground truth.

### 14.8 Bundle

- save/load round trip reproduces golden logits/probabilities;
- changing code/config/source/schema hash rejects stale caches;
- bundle contains representation mode, transforms, feature/matrix schemas, aggregation rule, and provenance.

---

## 15. Explicit no-go list

- Do not relabel or merge the three classes.
- Do not use outer held-out data for any fitted transformation or decision.
- Do not calculate amplitude-dependent SQI from normalized DL windows.
- Do not force denoising when `Q_rate` passes and only `Q_morph` fails.
- Do not calculate `Q_morph` on non-identity `x_ar`.
- Do not produce morphology or optical-amplitude features from non-identity `x_ar`.
- Do not silently substitute `x_ar` into the raw branch.
- Do not treat feature-matrix rows as independent samples.
- Do not repeat a file feature vector per raw window in reference fusion.
- Do not aggregate all participant windows directly in the reference route.
- Do not make quality-weighted participant aggregation the hidden default.
- Do not use external PTT-PPG observations for frailty classification.
- Do not call waveform smoothness external validation.
- Do not call a single Inception network the original five-network ensemble.
- Do not change folds, labels, or exclusions to improve balanced accuracy.
- Do not conceal missing implementation by editing thesis prose.

---

## 16. Implementation order without schedule

1. Freeze manifest, QC reason codes, and participant folds.
2. Implement typed signal views and unified window plans.
3. Implement endpoint SQI and the A1/A2 route state machine with `x_ar` rate-only semantics.
4. Canonicalize the pulse detector and timestamp-preserving PPI representation.
5. Freeze Feature Registry V1 and direct-only morphology rules.
6. Implement `FeatureVectorV1` and `EngineeringFeatureSequence`.
7. Implement all four representation modes behind one config field.
8. Correct file-level fusion.
9. Implement `OrderedFeatureMatrixV1`, mask policy, feature-matrix Inception, and ROCKET+ridge.
10. Unify trainer, OOF writer, and window→file→role→participant aggregation.
11. Implement external A3 benchmark and internal A4 report.
12. Add bundle round-trip and complete acceptance tests.
13. Update thesis Chapter 3 to describe only the configuration actually run.

---

## 17. Definition of done

The canonical workflow is complete only when:

- all four modes run from the same manifest and frozen folds;
- all fitted objects are proven outer-training-only;
- the A1/A2 route state machine is tested;
- non-identity `x_ar` is rate-only by construction;
- external PTT-PPG produces ECG-reference event/rate metrics;
- internal dynamic records produce SQI/route/coverage diagnostics;
- `FeatureVectorV1` and `OrderedFeatureMatrixV1` are versioned and reproducible;
- fusion concatenates once at file level;
- reference aggregation is window→file→role→participant;
- complete OOF probabilities and provenance are saved;
- the model bundle reproduces golden predictions after reload;
- Chapter 3 and the actual locked run configuration agree.

**Completion principle:** make the dataflow impossible to run incorrectly before optimizing performance.
