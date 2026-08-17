# Specification vs local/frozen implementation by workflow / 按 workflow 对照本地与冻结实现

## Live evidence boundary / 当前证据边界

- [Strict acceptance](../../artifacts/acceptance/strict_acceptance_current.json) and
  [CPU CI](../../artifacts/acceptance/cpu_ci_current.json) are the current engineering
  authorities; prose test counts and source hashes must not outrank these live JSON files.
- [Real-input smoke](../../artifacts/test_reports/integration_smoke_canonical_manual.json)
  exercises a frozen real record and protocol integration without scientific predictions.
- The named artifact/model/gravity/physical-time reports are synthetic or construction-level
  comparisons. They establish executable parity and schema coverage, not scientific ranking.
- The [60 s real r0/f0 training smoke](../../artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json) (authority: [reference registry](../../artifacts/experiments/reference_registry.json); prior immutable reference is superseded)
  proves one feature-vector cell can fit train-only state, write complete OOF/metrics/confusion
  artifacts, and retain 5/6 OOF participants; the
  [12 s run](../../artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)
  proves zero-retention closes without a metric. Both are reduced smoke, never the full 5×5
  multi-representation benchmark.

## Classification scale / 判定等级

- **Reuse**: frozen data/membership or pure algorithm can be copied with source hash and parity.
- **Adapt**: useful implementation exists, but its interface/provenance/eligibility must change.
- **Replace**: behavior violates the contract or has outer-fold leakage/silent fallback.
- **New**: no compliant local implementation existed.

## 1. Input, manifest, and provenance

| Local evidence | Decision | Required V1 behavior |
|---|---|---|
| M2 `frailty3_file_manifest.csv` and corrected SGKF registry | Reuse | Convert to strict `ManifestRow`, bind exact file/payload hashes, materialize membership CSV; never runtime-regenerate. |
| M2 external PTT/Sim manifests | Adapt | Preserve ECG reference, activity and unresolved wavelength; add V1 record schema and explicit eligibility. |
| Legacy filename/subject parsers | Replace | Never infer labels/roles or drop `_NN` identity fragments from filenames. |
| Legacy caches keyed by a few parameters/path strings | Replace | Content-address source bytes, manifest/fold/config/schema/code; reject any mismatch. |

Implemented change: V1 re-hashes all 261 internal sources, materializes 5×5 assignments,
generates a visible row/reason for failures, and records external split provisional status.

## 2. Parse/QC and synchronized loading

| Local evidence | Decision | Required V1 behavior |
|---|---|---|
| M2 full numeric scan | Reuse as snapshot evidence | It proves the frozen 261 files are finite/eight-column, not that arbitrary future input is safe. |
| M3 signal validation/repair | Adapt | Copy bounded internal-gap/timestamp/channel logic; expose per-record status/reasons. |
| Historical NaN interpolation/all-missing→zero | Replace | Entire unavailable channel and long/boundary gap fail closed. |

Required additions: parse failure, missing channel, all nonfinite, gap, duration, flatline,
clipping/saturation, implausible scale, timestamp, synchrony and duplicate reasons. Absolute
ADC saturation rails remain review/pending rather than invented.

## 3. PPG and IMU preprocessing

| Local evidence | Decision | Required V1 behavior |
|---|---|---|
| M3 registry-bound 0.2–8 Hz static PPG | Reuse/Adapt | Copy to self-contained V1, source-hash it, retain `x_native` and zero-phase `x_filter`. |
| M3 stateful quaternion ESKF and LPF comparator | Reuse/Adapt | Preserve state, masks, q/bias/P, gating, latch/reset and causal sensor filters; no simplified EKF. |
| Root scripts' repeated filters and LPF gravity removal | Replace | Models consume one canonical signal layer, never duplicate preprocessing. |
| Historical short-input causal fallback | Replace | Reject or use an explicitly tested zero-phase short policy; never silent phase change. |

V1 direct analysis uses `x_filter` at 400 Hz. Non-identity analysis uses aligned `x_ar`.
DL-only resampling is explicit and cannot change feature/morphology time axes.

## 4. Windows and scaling

| Local evidence | Decision | Required V1 behavior |
|---|---|---|
| Multiple legacy split/window helpers | Replace | One `WindowPlan` for engineering and DL with start/end/hop/alignment/cap/padding/mask. |
| Per-window PPG median/IQR | Reuse/Adapt | Apply only to model view; preserve direct amplitude/DC views. |
| Existing sklearn scaler pipelines | Adapt | Bind exact outer-training participant IDs and persist transforms. |
| Legacy 30 s zero padding for short S/W | Replace as reference | Raw reference uses complete 5 s windows; matrix padding occurs after fold-local transform with mask. |

## 5. Quality and routing

| Local evidence | Decision | Required V1 behavior |
|---|---|---|
| Existing scalar SQI/gates | Adapt/Replace | Build separate `Q_rate` and `Q_morph`, auditable components and train-only/fixed calibration. |
| M1 V3 SQI-first router | Reuse contract | SQI required, motion optional; high quality bypass; degraded action is run-locked drop XOR reducer. |
| Legacy action-owner/coarse replacement/quality weighting | Replace | No waveform replacement claim, no post-hoc per-window best route, no unvalidated weighting. |

Every component exposes raw/normalized/state/reason. Non-identity routes instantiate
`Q_morph=not_applicable`; they do not compute it and then discard the result.

## 6. Artifact/rate recovery

| Local source | Decision | Reason / V1 change |
|---|---|---|
| `funcs.py`/`ppg.py` NLMS/CEEMD snippets | Adapt mathematics only | Correct delay taps, no full-record z-score/fitting, typed IMU references, status/provenance. |
| v7 STFT/DWT/AE | Historical comparison | Misnaming, silent fallback or protocol differences prevent active import. |
| v8/stage2 mask networks | Historical failed evidence | Code defects and no common rate/morph contract. |
| hybrid denoiser | Historical comparison | Full evaluation-record fitting, default delay and missing failure semantics. |
| M0 method audit | Reuse evidence | It prevents duplicated dead ends but is not an implementation. |

V1 provides identity, IMU-referenced NLMS, SSA, STFT/IMU masking and PCA/FastICA/NMF BSS
under one alignment/failure interface. Learned denoiser names return explicit unsupported
until a validated artifact exists; they never pretend identity is success.

## 7. Peaks, HR/PPI/PRV, morphology, dual wavelength

| Local evidence | Decision | Required V1 change |
|---|---|---|
| M3 corrected bipolar peaks and interval logic | Reuse/Adapt | Preserve event timestamps/indices, valid interval and adjacency masks. |
| Existing HR/PPI/partial PRV | Adapt/Replace final schema | Add complete count/duration/PPI/rate/time/nonlinear/frequency fields and exact eligibility. |
| Existing local Aboy++ morphology | Adapt | Enforce valley-to-valley local baseline and direct-route token; robust beat median/MAD. |
| M3 RED/IR proxies/agreement | Expand | Implement beatwise AC/DC/PI/R and correlation/lag/coherence with denominator validity. |

Rejected intervals remain in the timeline. Unavailable physiology is internal NaN plus validity
false and strict-JSON null—not a valid zero. Morphology/optical extractors reject `x_ar`.

## 8. Feature representations

| Route | Local state | Required change |
|---|---|---|
| `raw` | Existing 8-channel windows | Use canonical preprocessed/masked windows and content-addressed provenance. |
| `feature_vector` | Partial manual/file features | Freeze ordered allowlist, formula/unit/source/eligibility/aggregation/validity. |
| `feature_matrix` | Absent | New one-recording `D×32` representation, fold-transformed context and row mask. |
| `fusion` | Existing incorrect repeated file vector | Replace with raw-window file pooling then one file-vector encoding/concatenation. |

Technical metadata, duration, file order and administrative missingness stay outside the default
predictor registry. Non-identity optical engineering slots remain unavailable.

## 9. Models

| Local model | Decision | Required V1 change |
|---|---|---|
| `Cnn1DClassifier` | Reuse architecture | Rename `CompactCNN1D`; lock 79,139 params; not Wang-FCN. |
| Full/Small Inception ports | Reuse architecture | Lock 456,579/57,027; call single networks; add matrix mask policy. |
| ShapeFormer effect-size port | Adapt experimental | Self-contained effect-size discovery is fit on and bound to outer-train roster/repeat/fold; non-overlapping patch/downsample precedes mask-aware Transformer attention; local shapelet-distance is parallel; no PISD/original parity. |
| PISD external path | Not active | Requires licensed vendoring/pinning and common protocol. |
| L2 LR/RBF SVM/ExtraTrees definitions | Reuse estimator families | Only allowlisted vector features and fold-local transforms. |
| ROCKET/MiniROCKET | New | Self-contained feature-matrix transform/ridge, train-only kernels and serialization. |
| Five-member Inception ensemble | New | Five distinct seeds/weights, exact probability mean and member OOF. |

## 10. Trainer, selection, and leakage

The old deep evaluators are replaced, not wrapped. They supplied outer-held-out windows as
validation and selected best epochs. The new fit API accepts training data and a frozen split;
there is no outer-label argument. Legal rules are fixed epoch, or inner participant-grouped
selection followed by a fresh full outer-train refit.

Every fitted scaler/imputer/SQI threshold/shapelet/ROCKET kernel/ridge/calibrator/model stores
training participant IDs and hashes. A held-out mutation test must leave these artifacts unchanged.

## 11. Aggregation, OOF, and metrics

Legacy direct window→participant mean and duplicated-window weighting are replaced by:

```text
window probability → file probability → role probability → participant probability
```

Vector/matrix/fusion routes enter at file level. Formal output includes window/file/subject and
member tables; balanced accuracy, macro-F1, per-class/worst-class, confusion, coverage,
calibration, repeat mean/SD/CI and paired delta all use participant as the independent unit.

## 12. Bundle, CLI, and deployment

Legacy `.pt`/`.joblib` files lack complete environment/schemas/transforms/golden examples and
are not active bundles. V1 bundles bind class/channel order, config, signal/window/feature/matrix
schemas, every fit object, aggregation, code/data/fold hashes and golden inference. CLI commands
select modules/tests/comparisons explicitly. ONNX/mobile parity remains a V2 confirmation point;
no absent dependency or unmeasured hardware result is claimed.

The public `run` command is an input/protocol audit and its real-record smoke emits no trained
metric. Synthetic `compare`/`ablate` output is also not a benchmark. The reduced scientific
runner retains the frozen roster, reports drop/no-result coverage, and labels its single-fold
feature-vector execution as smoke. Raw/matrix/fusion have module and contract coverage but no
persisted same-runner 5×5 result; this distinction remains visible rather than being inferred
from construction tests.
