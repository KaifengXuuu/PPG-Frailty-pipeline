# Codex Addendum — Revised 9-Case Legacy-to-V2 Bridge with Mandatory Phase 0

**Repository scope:** `final_v0/final_pipeline_v2`  
**Purpose:** diagnose why the historical raw-window pipeline and current V2 produce different participant-level results.  
**This addendum replaces the previous 18-case execution list.**

## 1. Frozen interpretation of old and V2 sampling

### Historical training sampler

The historical raw CNN/Inception route must be represented as:

```text
retained windows
→ DataLoader(shuffle=True)
→ one random permutation per epoch
→ every retained training window exactly once per epoch
```

This is random order **without replacement**. It must not use `WeightedRandomSampler`.

The model treats every window as an independent sample. The chronological sample order inside each window is preserved. The chronological order among different windows is not a model input and must not be claimed as a learned cross-window trajectory.

### V2 training sampler

The current V2 route uses:

```text
WeightedRandomSampler(
    weights=participant/file/window or participant/role-family/file/window weights,
    num_samples=len(training_dataset),
    replacement=True
)
```

Consequences:

- total draws per epoch equal the number of training rows;
- some windows repeat within the same epoch;
- some windows are not drawn in that epoch;
- Line B changes expected exposure by participant, role family, file, and window;
- the signal samples inside a selected window are not reordered.

For uniform replacement, the expected unique rows after `N` draws from `N` equally likely rows are approximately `0.632N`. Line-B unequal weights can reduce effective diversity further.

### Cross-window chronology

Neither historical raw CNN/Inception nor current V2 raw CNN/Inception learns chronological relations between separate windows:

- training order is randomized;
- each window receives one class label;
- file/participant aggregation is an order-invariant probability mean.

This does **not** destroy within-window temporal morphology. It means cross-window progression was never represented in these raw models.

## 2. Why input balancing and output aggregation are different

Output aggregation changes how held-out probabilities are combined. It cannot change model parameters already learned during training.

Input sampling changes the training objective by deciding how often each row contributes a gradient. V2 Line B attempts to align training exposure with the evaluation hierarchy:

```text
participant
→ B/R role family
→ file
→ window
```

However, Line B sampling is a project-specific strategy, not a mandatory literature method. It must remain an explicit ablation factor. In this cohort, every participant has one B file and four R files, so Line B changes the approximate family exposure from:

```text
ordinary rows: B ≈ 20%, R total ≈ 80%
Line B:       B = 50%, R total = 50%
```

This may be beneficial or harmful. Do not assume it is correct before the bridge result.

## 3. Historical and V2 weighted loss

### Historical approximate loss

Use inverse **training-window-count** class weights:

```text
count_c = number of retained outer-training windows with class c
weight_c = N_train_windows / (K * count_c)
```

The historical route combines this weighted cross-entropy with exhaustive shuffled traversal. It has no hierarchy sampler.

### Current V2 loss

Use inverse **unique outer-training-participant-count** class weights:

```text
count_c = number of unique outer-training participants with class c
weight_c = N_train_participants / (K * count_c)
```

Current V2 combines this class-weighted loss with the Line-B replacement sampler.

Therefore `L5 -> L6` is a named **V2 balance-policy change**, containing:

1. uniform replacement → Line-B hierarchy weights;
2. historical window-count class weights → V2 participant-count class weights.

Do not interpret a large `L5 -> L6` delta as sampler-only evidence. A future split study is allowed only after user authorization.

## 4. Phase 0 — Data, manifest, source-byte, and cache integrity audit

**Phase 0 is mandatory and must finish before any of the 9 training cases begin.**  
It is an audit stage, not a model case. It does not add folds, fits, or model-epochs to the
`9 cases / 45 fits / 450 model-epochs` budget.

### 4.1 Audit objectives

Phase 0 must determine whether the historical and V2 pipelines are actually starting from
the same recordings, labels, roles, channel schema, and current source bytes. It must also
determine whether a historical cache, if present, still represents the current raw CSV files
and the declared legacy preprocessing.

The audit must never change labels, units, files, or configuration automatically. It reports
differences and stops where identity cannot be established.

### 4.2 Revalidate the complete V2 manifest against the raw files

Use:

```text
manifests/internal_records_v2.csv
```

For all 261 manifest rows:

1. Resolve `source_path` and read the current raw file bytes.
2. Recompute the raw-file SHA-256 and compare it with the row-level `source_hash`.
3. Re-read the CSV independently and verify the required channel set and order:

```text
RED, IR, AX, AY, AZ, GX, GY, GZ
```

4. Recompute and compare, where available:
   - `n_samples`;
   - `duration_s` at the declared sampling rate;
   - participant identity;
   - class ID and class name;
   - recording role;
   - sampling rate;
   - channel schema.
5. Count:
   - non-finite values;
   - all-missing channels;
   - internal and edge gaps;
   - flat or constant runs;
   - duplicate source hashes;
   - duplicate record identities.
6. Recompute the complete manifest-file hash and record the audited manifest version.

A source-hash mismatch is not a warning that may be ignored. It means the current bytes are
not the bytes recorded by the manifest. Do not load a derived cache for that recording until
the discrepancy has been resolved.

### 4.3 Independently reconstruct the historical file-discovery result

Do not use the V2 manifest itself as proof that the historical input set was identical.
Independently reproduce the historical read-only discovery logic:

```text
StudyData/*.csv
TestDataYoungers/*.csv
included roles = B, R1, R2, R3, R4
FRAILTY-STATUS 2 -> Pre-Frail
FRAILTY-STATUS 3 -> Robust/Non-Frail
TestDataYoungers -> Young
```

Build an explicit alias table for historical participant IDs that removed suffixes from older
participant filenames. Compare the independently discovered static set with the V2 manifest
by exact source path and source hash, not by fuzzy filename matching.

The expected matched static set is:

```text
145 recordings
29 participants
9 Pre-Frail
12 Robust/Non-Frail
8 Young
one B, R1, R2, R3, and R4 recording per participant
```

For every static recording, compare:

```text
legacy source path
V2 record_id
legacy participant alias
V2 participant_id
role
class_id / class_name
source_hash
n_samples
required channel schema
```

### 4.4 Audit raw IMU units and the V2 EKF output

The manifest unit declaration must be checked against the observed signal scale; it must not
be accepted only because it appears in metadata.

For every participant's B recording, report at least:

#### Raw source scale

```text
median ||[AX,AY,AZ]||
IQR / P01 / P99 of acceleration norm
median, RMS, P01, and P99 for GX, GY, GZ
non-finite and constant-run counts
```

#### After the declared V2 unit conversion

```text
acceleration norm in m/s²
gyro axes in rad/s
```

#### After B-record calibration and roll–pitch EKF processing

```text
calibration record identity
estimated acceleration and gyro biases
initial roll and pitch
estimated gravity norm median / IQR
A_dyn_x/y/z mean, RMS, and P99 on B
EKF status and failure reason
```

Flag, but do not automatically correct, cases such as:

- raw acceleration norm already near `9.81` although the manifest declares `g`;
- converted gravity norm materially inconsistent with `9.81 m/s²`;
- static gyro offsets or ranges inconsistent with the declared unit;
- large residual dynamic acceleration in a nominally static B recording;
- EKF initialisation or numerical failure.

Any unit correction requires a separate human decision and a new manifest version.

### 4.5 Audit historical caches, but never train from them

Search for historical window caches, including files matching the old static CNN naming
pattern, for example:

```text
frailty3_cnn_windows_B_R1_R2_R3_R4_fs64_s15_h3_mf090.npz
```

Rules:

1. The 9 bridge cases must always materialise their inputs fresh from the current raw CSV
   bytes after Phase 0 passes.
2. A historical cache is audit evidence only; it is never a training source.
3. If a cache exists, record:
   - cache file SHA-256;
   - stored paths, labels, participants, roles, and file indices;
   - array names, dtypes, and shapes;
   - any stored configuration or provenance.
4. Freshly regenerate the corresponding legacy windows from the current CSV files and compare:
   - path and row identities;
   - array shape;
   - complete-array hash where feasible;
   - maximum and mean absolute difference;
   - selected windows from at least one participant in each class, covering B and R4.
5. If the cache does not contain source hashes, preprocessing hashes, or a code/schema version,
   state explicitly that byte-level equivalence to the 2026-05-25 run cannot be proven.
6. If no historical cache is available, write `historical_cache_not_available`; do not infer its
   content from filenames or old metrics.

### 4.6 Frozen split audit

Read repeat 0 directly from:

```text
splits/sgkf5_repeated_grouped_5x5_v2.csv
```

Verify:

- folds 0–4 are present;
- all 29 participants are held out exactly once in repeat 0;
- one participant never appears in both outer train and held-out data in the same fold;
- all files and windows inherit their participant's fold;
- class labels remain consistent;
- the split-file and split-payload hashes match the declared V2 registry.

Do not regenerate `StratifiedGroupKFold` at runtime.

### 4.7 Required Phase 0 outputs

Write:

```text
artifacts/audit/legacy_v2_manifest_record_diff.csv
artifacts/audit/legacy_v2_source_hash_audit.csv
artifacts/audit/legacy_v2_source_hash_audit.json
artifacts/audit/legacy_v2_channel_qc.csv
artifacts/audit/legacy_v2_participant_alias_map.csv
artifacts/audit/legacy_v2_imu_unit_ekf_audit.csv
artifacts/audit/legacy_v2_cache_audit.json
artifacts/audit/legacy_v2_split_audit.json
artifacts/audit/LEGACY_V2_PHASE0_DATA_AUDIT.md
```

The Markdown summary must contain:

- total matched and mismatched records;
- exact reasons for every mismatch;
- whether the 145-record static set is identical;
- whether current source bytes match the V2 manifest;
- whether the old cache can or cannot be traced to current source bytes;
- IMU unit/EKF red flags;
- a final `PASS`, `STOP`, or `PASS_WITH_DECLARED_LIMITATIONS` decision.

### 4.8 Mandatory stop conditions

Do not start model training if any of the following remains unresolved:

- current source bytes do not match the V2 row-level source hash;
- the independently discovered 145-record historical static set differs from the V2 static set;
- role, class, or participant identity conflicts;
- a historical participant alias cannot be mapped one-to-one;
- a required channel is missing or channel semantics are ambiguous;
- duplicate records cannot be resolved;
- participant leakage or split-registry mismatch is detected.

A historical-cache mismatch does not by itself stop fresh training, because the cache is not used.
It must, however, be reported as evidence that an exact historical byte-for-byte replay is not
established.

## 5. Revised experiment budget

Run only:

```text
9 cases
× 5 frozen outer folds in repeat 0
= 45 model fits

45 fits
× 10 fixed epochs
= 450 model-epochs
```

All cases use:

```text
repeat = 0
folds = 0,1,2,3,4
training seed = 42
fixed epochs = 10
early stopping = false
outer-label checkpoint selection = false
```

`450 model-epochs` does not mean 450 separate models; it means 45 fold-specific models trained for 10 epochs each.

## 6. Exact 9-case list

### InceptionTime — one case only

1. `inception_full__L0_legacy64_w15_fixed10`

Run only the full historical-approximate input/training profile. Use the current `InceptionTimeFull` single-network architecture, not an ensemble.

### CompactCNN — eight cases, execution order fixed by the user

2. `compact_cnn__L7_v2_training_bundle_fixed10`
3. `compact_cnn__L5_uniform_replacement_fixed10`
4. `compact_cnn__L6_v2_line_b_balance_fixed10`
5. `compact_cnn__L4_v2_imu_fold_scaled_fixed10`
6. `compact_cnn__L3_v2_imu_window_scaled_fixed10`
7. `compact_cnn__L2_legacy400_w5_fixed10`
8. `compact_cnn__L1_legacy64_w5_fixed10`
9. `compact_cnn__L0_legacy64_w15_fixed10`

Execution order is:

```text
L7 → L5 → L6 → L4 → L3 → L2 → L1 → L0
```

Reporting and causal interpretation must still sort profiles numerically and compare the predefined adjacent pairs:

```text
L0→L1
L1→L2
L2→L3
L3→L4
L4→L5
L5→L6
L6→L7
```

Do not interpret the execution-order jump `L7→L5` as one-factor evidence.

## 7. Exact profile definitions

### L0 — historical approximate baseline

```text
fresh current CSV bytes; no historical NPZ training input
roles B,R1,R2,R3,R4
PPG: detrend + order-3 zero-phase 0.2–8 Hz
IMU: order-3 20 Hz acceleration LPF; order-3 40 Hz gyro LPF
no SI conversion
no B calibration
no EKF
no gravity removal
8-channel anti-aliased resampling 400→64 Hz
window 15 s; hop 3 s; historical 90% evenly retained plan
all 8 channels per-window median/(IQR/1.349), SD fallback, clip [-8,8]
exhaustive shuffle without replacement
historical inverse-training-window-count weighted CE
AdamW, lr 0.001, wd 0.0001, batch 32
fixed 10 epochs
```

### L1 — window-plan change only

Relative to L0:

```text
15 s / 3 s / 90% retained
→
5 s / 2.5 s / complete-window plan / cap 128
```

All other fields remain L0, including 64 Hz.

### L2 — sampling-rate change only

Relative to L1:

```text
DL target fs 64 Hz → 400 Hz
```

Everything else remains legacy.

### L3 — V2 IMU semantics, legacy per-window scaling retained

Relative to L2:

```text
raw filtered IMU
→
g→m/s²
deg/s→rad/s
same-participant B calibration
calibrated roll–pitch EKF
A_dyn_x/y/z + GX/GY/GZ
```

Still apply per-window robust normalization to all eight channels.

### L4 — V2 IMU normalization

Relative to L3:

```text
IMU per-window normalization
→
outer-training-participant-only IMU median/(IQR/1.349) scaler
```

RED/IR remain per-window normalized.

### L5 — uniform replacement mechanics

Relative to L4:

```text
exhaustive shuffled traversal
→
uniform WeightedRandomSampler with replacement
```

Use all-one row weights, `num_samples=len(dataset)`. Keep historical window-count class weights. Do not use Line B.

### L6 — V2 balance policy

Relative to L5:

```text
uniform replacement
→
balance_line_weighted_v2 with equal B/R role-family mass
```

Also change:

```text
historical window-count class weights
→
V2 unique-participant-count class weights
```

Keep AdamW, batch 32, and fixed 10 epochs.

### L7 — V2 optimizer/batch/dropout bundle

Relative to L6:

```text
AdamW → Adam
batch 32 → batch 64
model dropout → current V2 registered dropout
```

Epochs remain 10 in both L6 and L7. Learning rate and weight decay remain 0.001 and 0.0001.

For CompactCNN, current and legacy stage dropout may already be identical; record the resolved diff rather than claiming a dropout change that did not occur.

## 8. Existing C0 result

Do not retrain `C0_current_v2`.

Compare L7 with the already existing current-V2 result only when all of the following match:

```text
code commit
manifest hash
split hash
model/config hash
preprocessing hash
training seed
repeat/fold roster
```

If they do not match, report `existing_C0_not_exactly_pairable`; do not create an extra case without authorization.

## 9. Aggregation output

For every case, write both post-hoc views from the same OOF window probabilities:

```text
legacy:
all participant windows → ordinary mean

V2:
window → file → role family → participant
```

This does not add training cases.

The primary bridge table must contain:

```text
model
profile
BA_legacy_aggregation
BA_v2_aggregation
macroF1_legacy_aggregation
macroF1_v2_aggregation
worst_class_F1
delta_from_previous_numeric_profile
```

## 10. Required sampling diagnostics

For every fold and epoch save:

```text
number of dataset rows
number of draws
number of unique rows drawn
duplicate draw fraction
never-drawn row fraction
draw counts by participant
draw counts by class
draw counts by B/R family
draw counts by file
class-weight vector
sampler identity
```

This evidence is required to determine whether replacement and Line B materially reduce per-epoch diversity.

## 11. Interpretation rules

- `L0→L1`: window-plan alignment only.
- `L1→L2`: DL sampling rate / physical receptive-field effect.
- `L2→L3`: unit conversion, calibration, EKF, and IMU channel semantics.
- `L3→L4`: IMU per-window normalization versus fold scaling.
- `L4→L5`: replacement mechanics only.
- `L5→L6`: V2 balance policy; not sampler-only.
- `L6→L7`: optimizer/batch/current-dropout bundle; epochs are controlled at 10.
- Inception L0 is a historical-input plausibility check only; no Inception ablation conclusions are permitted.

Do not add cases, repeats, models, early stopping, or hyperparameter search.
