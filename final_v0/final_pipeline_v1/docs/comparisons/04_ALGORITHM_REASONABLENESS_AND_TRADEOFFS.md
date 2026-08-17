# Algorithm reasonableness, benefits, and limitations / 算法合理性、优点与缺点

## Overall assessment / 总体评价

The attached specification is scientifically reasonable and substantially safer than the
historical pipeline because its first objective is **valid comparison**, not guaranteed score
improvement. Its strongest decisions are participant-level isolation, separation of direct
morphology from artifact rate recovery, explicit feature availability, and hierarchical
aggregation. The main weakness is breadth relative to 29 participants: it creates a large
implementation/compute surface whose candidate ranking will have high uncertainty without an
independent cohort.

This assessment distinguishes evidence types. The
[real-input smoke](../../artifacts/test_reports/integration_smoke_canonical_manual.json)
shows one frozen record traversing input/protocol integration and emits no trained metric.
[Artifact](../../artifacts/test_reports/artifact_comparison_canonical_manual.json),
[model](../../artifacts/test_reports/model_comparison_all13_manual.json),
[gravity](../../artifacts/test_reports/imu_gravity_comparison_manual.json), and
[physical-time](../../artifacts/test_reports/physical_time_contract_manual.json) reports are
synthetic/construction contract evidence. The current
[strict acceptance](../../artifacts/acceptance/strict_acceptance_current.json) is an
engineering gate, not scientific validation. The
[60 s real r0/f0 training smoke](../../artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json) (authority: [reference registry](../../artifacts/experiments/reference_registry.json); prior immutable reference is superseded)
shows why coverage and fail-closed behavior matter: it retained 5/6 OOF participants
(coverage 0.8333, BA 0.5), whereas the
[12 s diagnostic](../../artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)
closed without metrics. These are one-fold engineering diagnostics, not final model evidence.

## Why the workflow is reasonable / 合理性依据

### 1. Participant outer grouping

PPG windows from one participant are highly correlated. Grouping by participant prevents the
model from seeing the same person's device placement, morphology or motion signature in train
and held-out sets. This is non-negotiable and more important than small model changes.

### 2. Direct signal versus rate-recovery signal

IMU-correlated filtering, decomposition, spectral suppression and BSS can alter amplitude,
area and landmarks even when HR becomes easier to estimate. Treating `x_ar` as rate-only is
therefore conservative and correct. It avoids the common but invalid inference that a smooth
waveform is physiologically faithful.

### 3. Endpoint-separated SQI

A segment may support beat timing but not morphology. `Q_rate` and `Q_morph` model this fact
directly. Active abstention can improve safety and reporting honesty, provided coverage is always
reported and threshold fitting remains inside the outer training fold.

### 4. Multi-representation comparison

Raw windows, file vectors, ordered feature matrices and corrected file fusion test genuinely
different inductive biases under one protocol. This is more informative than repeatedly tuning
one CNN while changing preprocessing and aggregation at the same time.

### 5. Window→file→role→participant aggregation

It prevents long recordings or high-overlap window sets from automatically receiving more
voting weight. It also matches the scientific unit: one participant, with several recording roles.

### 6. Full provenance and bundle parity

Content hashes and golden inference convert hidden scientific state into testable artifacts.
This is essential when 25 OOF cells and many candidate routes otherwise invite stale caches,
shortened configs or accidental split drift.

## Advantages over the current TODO route / 相比 TODO 的优点

1. It resolves ambiguous phrases into typed formulas and acceptance tests.
2. It narrows the artifact objective from “denoise waveform” to measurable HR/PPI recovery.
3. It names models accurately and separates one Inception network from a five-member ensemble.
4. It fixes fusion and aggregation before broad benchmark ranking.
5. It defines exact OOF and bundle outputs rather than relying on summary JSON.
6. It makes unavailable physiology explicit instead of imputing a valid zero.
7. It prevents a broad wavelet/EMD/model zoo from expanding before common protocol parity.
8. It links physical sampling rate and kernel duration, exposing a genuine time-scale issue.

## Disadvantages or omissions versus TODO / 相比 TODO 的缺点或缺失

1. It does not complete the 29-subject motion-detector retraining study.
2. It does not include the hierarchical Young/Old classifier.
3. It does not activate Base/Motion/Relax recovery features because stage timing is unresolved.
4. It does not implement every CWT/DWT/WPT/EMD/EEMD/VMD or temporal tracker candidate.
5. It does not finish the historical M5 Top-5 ontology/ranking program.
6. It stops short of the TODO's mandatory no-PyTorch mobile/ONNX target.
7. It does not create an independent frailty test cohort; OOF remains model-development evidence.

These are scope differences, not reasons to weaken the V1 leakage and route contracts. Selected
TODO-only modules can enter V2 after the user fixes their semantics and budget.

## Advantages over local frozen code/tests / 相比本地冻结代码与测试的优点

- Uses M2's corrected balanced folds instead of runtime SGKF recreation.
- Removes known outer-epoch leakage from CNN/Inception/ShapeFormer evaluators.
- Makes all fit objects train-ID auditable, not just scaler behavior.
- Replaces unsafe cache identities with content-addressed identities.
- Adds explicit route failure/no-result rather than silent identity fallback.
- Adds the missing feature matrix, ROCKET, ensemble, corrected fusion and complete OOF layers.
- Expands partial PRV/morphology/RED-IR proxies into versioned formula/validity contracts.
- Couples test reports to current source/config/data hashes instead of citing a stale count.

## Costs and scientific limitations / 成本与科学限制

### Small cohort and candidate multiplicity

With class participant counts 9/12/8, differences between large model families will be noisy.
Five repeats quantify split sensitivity but do not create new participants. A large candidate
matrix increases selection bias even without literal leakage. V1 must report paired uncertainty,
weakest-class metrics and incomplete status; it must not over-read a one-run winner.

### Feature matrix and repeated context

`D×32` gives convolution/ROCKET a fixed ordered object, but repeated file context can dominate
time-varying channels and yield optimistic complexity on few files. The context mask/order and
technical-metadata ablation are mandatory; regularization and simple vector baselines remain
important controls.

### ROCKET-10,000

ROCKET is attractive for small time-series datasets and CPU inference, but 10,000 kernels plus
ridge can still overfit 29 participants if alpha/model choices are tuned too broadly. Kernels and
alpha must be fixed or inner-only, stored, and compared with reduced/MiniROCKET routes without
silently substituting them.

### Five-member ensemble

An ensemble can reduce initialization variance but multiplies training, storage and inference by
about five. With a small cohort it does not solve domain shift or provide independent evidence.
The single network should remain the primary baseline until full member OOF is complete.

### ShapeFormer

Shapelets are interpretable, but discovery is another high-variance fit object. V1 repairs the
experimental route structurally: non-overlapping patch/downsample precedes generic mask-aware
Transformer attention; a trainable shapelet-distance branch runs in parallel; fitted shapelets
carry outer-train roster hash, repeat/fold, sampling rate, and length in samples/seconds. This
is substantially safer than raw-sample attention or external-path discovery, but local
effect-size discovery is still not PISD/original parity and does not justify a performance claim.

### SQI and abstention

Dropping low-quality windows can improve metrics by changing the evaluated population. Risk and
coverage must be reported together; a high score at very low coverage is not a better clinical
pipeline. Device-specific absolute thresholds remain a human/data decision.

### External PTT domain

PTT supplies strong ECG event reference but is not a Frailty3 label dataset and has a documented
pleth wavelength conflict. It can validate rate recovery and motion behavior, not frailty
classification or waveform morphology recovery.

### EKF observability

A no-precalibration six-axis quaternion ESKF is a reasonable primary comparator, but sustained
linear acceleration and tilt are not fully separable; yaw and parts of gyro bias are weakly or
unobservable. Confidence/state/masks must remain visible. LPF may outperform EKF on some static
proxy metrics; that is a result, not a code error.

### End-to-end executor breadth

The shared contracts cover raw, vector, ordered-matrix and fusion models, but the current real
outer-fold experiment executor supports feature-vector only. Explicit fail-closed behavior is
scientifically preferable to an implicit representation substitution, yet it remains an
engineering limitation versus the specification's intended one-switch multi-representation
benchmark. Construction/forward tests for raw, matrix and fusion do not replace their missing
same-runner OOF evidence.

### Direct PPG passband

The specification's 0.2–8 Hz direct reference retains more low-frequency baseline content than
the 0.4–8 Hz peak-oriented M3 alternative. The lower edge can help amplitude/morphology context
but may admit more drift; 0.4 Hz may stabilize peak timing while altering DC/baseline-dependent
features. Because the choice affects SQI, peaks, morphology, optical features and model hashes,
V2-028 correctly treats it as a preregistered one-factor decision rather than a hidden helper
filter.

### Runtime and deployment

400 Hz raw deep learning preserves local morphology but costs memory/latency and gives the current
Inception kernels sub-cycle receptive fields. Physical-time ablation is justified. Python CPU
parity is not equivalent to ONNX/mobile parity; those claims require a selected model and device.

## Recommended interpretation of future results / 后续结果解释原则

- Treat V1 scores as corrected OOF development estimates, never independent test performance.
- Prefer a configuration that is complete, stable, covers all classes and retains acceptable
  coverage over one with a single higher repeat.
- Keep simple vector/CompactCNN/identity baselines visible.
- Select an artifact reducer only if ECG-reference HR/PPI and coverage improve under paired
  participant splits; internal plausibility alone is insufficient.
- Do not retain morphology/optical features from a processed signal regardless of downstream BA.
- Lock V2 decisions before full candidate outcomes are inspected whenever the decision can affect
  eligibility, thresholds, model identity or aggregation.
