# Specification vs TODO / 实施规范与 TODO 的重合和差异

## Authority and comparison boundary / 权威与对照边界

本报告比较的是：

1. 用户直接指定的产品合同文件
   `AA_TODO/3/CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md`，锁定
   SHA-256 `cd7c4907...3c5000`；
2. 当前 `_agent/TODO.md`，305 行，SHA-256 `b4a5a1ac...fec113`。

The attached document is treated as the V1 product implementation specification. Its
instructions do **not** grant permission to write outside `final_v0/final_pipeline_v1`,
modify `_agent`, overwrite historical results, use outer labels, or perform external
actions. Those boundaries come from the user's direct request and workspace rules.

## High-level relationship / 总体关系

两者方向一致，但抽象层次不同：TODO 是 M0–M10 的全项目路线图；规范是把当前
`dev0` 收敛为一个可执行、无泄漏、可审计的静态/动态信号与 Frailty3 分类包。
规范不是 TODO 的逐项替代：它强化了 M1–M4/M6–M7 的科学合同，同时没有纳入
TODO 独有的所有历史排名、hierarchical、recovery 和最终移动端工作。

## Current implementation evidence / 当前落地证据

- The live engineering authorities are
  [strict_acceptance_current.json](../../artifacts/acceptance/strict_acceptance_current.json)
  and [cpu_ci_current.json](../../artifacts/acceptance/cpu_ci_current.json). They validate the
  current source/contracts/CLI/tests and claim boundaries; they are not scientific scores.
- [integration_smoke_canonical_manual.json](../../artifacts/test_reports/integration_smoke_canonical_manual.json)
  reads one real frozen Frailty3 record through the public input/protocol integration route,
  but deliberately emits no trained prediction metric.
- The persisted [60 s real r0/f0 training smoke](../../artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json) (authority: [reference registry](../../artifacts/experiments/reference_registry.json); prior immutable reference is superseded)
  completed a feature-vector/train-only-SQI/OOF path with 5/6 held-out participants retained,
  coverage 0.8333 and BA 0.5. The paired
  [12 s diagnostic](../../artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)
  failed closed with no metric. Both are explicitly `smoke_not_scientific_benchmark` and are
  evidence that the runner/abstention contract behaves honestly—not a 5×5 ranking.
- [artifact_comparison_canonical_manual.json](../../artifacts/test_reports/artifact_comparison_canonical_manual.json),
  [model_comparison_all13_manual.json](../../artifacts/test_reports/model_comparison_all13_manual.json),
  [imu_gravity_comparison_manual.json](../../artifacts/test_reports/imu_gravity_comparison_manual.json),
  and [physical_time_contract_manual.json](../../artifacts/test_reports/physical_time_contract_manual.json)
  are synthetic or construction-level contract evidence. They are not an external-PTT reducer
  ranking, a Frailty3 leaderboard, or a frozen 5×5 outcome benchmark.
- ShapeFormer §6.1 conformance is documented by
  [the phase-11 repair record](../../records/log_entries/20260815_phase11_shapeformer_spec61_repair.md):
  patch/downsample precedes mask-aware attention; the local effect-size shapelet bank is bound
  to outer-train roster/repeat/fold and physical time; PISD/original parity is not claimed.

## Exact overlap / 明确重合

| Workflow area | Shared requirement / 共同要求 | Stronger expression |
|---|---|---|
| Labels | Keep three Frailty3 classes unchanged | Spec fixes display names and forbids deleting hard participants. |
| Manifest | One versioned row per recording with subject/class/role/channels/version/reference | Spec adds source hash, QC reasons, exact typed row and predictor exclusion. |
| Participant grouping | Subject/participant is the outer independent unit | Spec requires every fitted object to carry train participant provenance. |
| Frozen evaluation | Same folds/seeds/repeats; no runtime split drift | Spec requires one materialized split file consumed by every branch. |
| OOF naming | No independent test ⇒ use `oof_validation_*` | Spec requires window/file/subject/member OOF tables and exact row provenance. |
| 400 Hz | Frailty3 acquisition/feature grid remains 400 Hz | Spec separates optional DL-only anti-aliased resampling. |
| PPG/IMU preprocessing | Shared filters, units, gravity removal and scaling | Spec makes direct amplitude/DC view and rate-only processed view different types. |
| QC | Detect parse/channel/nonfinite/gap/short/flatline/clipping/scale/time failures | Spec says every recording must produce a visible row/reason; never silently skip. |
| SQI | Combine spectral, periodicity/PPI, RED/IR, morphology and motion evidence | Spec splits `Q_rate` from stricter `Q_morph` and freezes non-identity `Q_morph=not_applicable`. |
| Heartbeat | Common peaks, HR, PPI and eligible PRV | Spec freezes timestamps, interval indices, adjacency, frequency bands and entropy eligibility. |
| Motion artifact | Raw/quality-only and comparable reducers | Spec declares every non-identity output rate-recovery only, not clean-waveform recovery. |
| Models | Compact CNN, full/small Inception, ShapeFormer, ROCKET and tabular models | Spec freezes parameter snapshots, accurate names and common Trainer/Evaluator. |
| Fusion | Correct window/file granularity and avoid repeated file-feature exposure | Spec defines one concrete `FileBagDataset` reference route. |
| Aggregation | Report participant-level metrics with intermediate levels | Spec freezes `window→file→role-aware participant`. |
| Leakage | scaler/imputer/threshold/model selection only on training participants | Spec includes SQI, shapelets, ROCKET kernels, sampler, calibration and epoch in the ban. |
| Ablation | Same folds/seeds/budget and one factor at a time | Spec requires complete config identity and paired participant/fold/seed joins. |
| Bundle | Persist preprocessing/features/model/label map/provenance | Spec adds exact schemas, ensemble members, aggregation, environment and golden parity. |
| Historical evidence | Preserve old code/results and mark comparability | Spec puts them behind `legacy_v0`/characterization and forbids corrected ranking reuse. |

## Specification requirements that are new or materially stronger / 规范新增或显著强化

1. **Four canonical representation modes**: `raw`, `feature_vector`, `feature_matrix`,
   `fusion`. TODO has these ideas across several milestones but not one typed switch.
2. **`OrderedFeatureMatrixV1`**: one recording = one `D×32` sample, chronological uniform
   sampling, post-transform padding and explicit row mask. TODO does not freeze this object.
3. **Exact architecture identity**: 79,139 / 456,579 / 57,027 parameter snapshots and
   explicit “not Wang-FCN”, “single network”, “experimental ShapeFormer” wording. The repaired
   experimental ShapeFormer uses patch/downsampling before mask-aware Transformer attention
   and binds discovery to outer-train identity; it remains non-PISD/non-original parity.
4. **Optional five-member Inception ensemble** with five independent initializations,
   member OOF and exact probability averaging. TODO lists Inception but does not define this.
5. **ROCKET-10,000 on feature matrix** with saved kernels/transform/ridge/class order.
6. **File-bag fusion reference**: encode windows, mask-aware pool once, encode the file vector
   once, concatenate once. TODO identifies leakage risk but not this exact algorithm.
7. **Signal-route type system**: `x_native`, `x_filter`, optional aligned `x_ar`, and an
   analysis view whose route determines legal features.
8. **Hard post-artifact boundary**: `x_ar` may produce rate/HR/PPI/eligible PRV only;
   morphology, AC/DC, PI, ratio-of-ratios, width/slope/area are unavailable—not merely discouraged.
9. **Exact morphology/optical formulas**: valley-to-valley linear baseline, beatwise AC/DC,
   canonical `R`, zero-lag correlation, max normalized cross-correlation/lag and 0.5–3 Hz coherence.
10. **Exact PRV contract**: VLF/LF/HF, LF/HF, normalized units, 4 Hz real-time tachogram,
    SampEn `m=2,r=.2SD,≥200`, and no adjacency compression.
11. **Complete OOF and bundle artifact names** plus row-level provenance.
12. **Physical-time ablation** for DL sampling rates, kernel duration/dilation and 5 s versus
    longer context.
13. **Golden bundle inference parity** and local CPU acceptance gates.

## TODO-only or broader requirements not made canonical by the specification / TODO 独有范围

| TODO item | Status relative to V1 specification |
|---|---|
| Retrain/compare three motion detectors on 29 subjects | Adjacent. Spec needs motion/SQI inputs but does not require this exact detector study. |
| Full wavelet/EMD/EEMD/CEEMD/VMD/CWT/DWT/WPT method zoo | Reduced. Spec requires identity, NLMS, at least one decomposition and one justified spectral method; V1 also supplies BSS comparisons. |
| Kalman/particle/Viterbi HR-trajectory tracker and TROIKA/JOSS study | Not a V1 canonical requirement. May be a future rate-recovery module. |
| Hierarchical Young-vs-Old then Pre-Frail-vs-Robust | Not in the attached V1 contract. |
| Base/Motion/Relax recovery slope/time classifier | Not active until exact stage timing/prediction availability is confirmed. |
| Multi-level OOF stacking among raw/ROCKET/stage routes | Only the corrected file-level fusion reference is canonical in V1. |
| M5 historical all-experiment Top 5 program | Spec preserves characterization but prioritizes uniform rerun, not historical leaderboard completion. |
| M8 final selection and publication-grade independent performance | V1 cannot supply an independent frailty test that does not exist. |
| M9 mobile no-PyTorch + mandatory ONNX parity | Spec requires reproducible CPU bundle/CI but not the full TODO deployment gate; retained as a V2 decision. |
| M10 archive and `_agent` synchronization | Outside current write authorization and intentionally not performed. |

## Direct contradictions or superseded semantics / 直接矛盾或已被后续决定替代

1. **Quality versus denoising order.** Original TODO M1 describes SQI and coarse denoising
   as parallel substitutes. The later user decision, M1 V3, and attached spec resolve this:
   direct quality is evaluated first; degraded/motion data follow a run-locked `drop XOR
   reducer` policy; non-identity output is requalified by `Q_rate` only.
2. **Waveform recovery claim.** Some TODO method language says “denoising” and enumerates
   waveform methods. The spec prohibits calling non-identity output morphology-preserving
   without paired clean morphology ground truth. V1 uses “rate recovery”.
3. **Single-network naming.** Historical/TODO wording sometimes uses “InceptionTime” without
   distinguishing one network from the original-style ensemble. V1 names both explicitly.
4. **ROCKET input.** TODO M4.4 proposes PPG waveform ROCKET variants. The spec reference
   ROCKET consumes `OrderedFeatureMatrixV1`; raw ROCKET may exist only as a separately named
   experimental/legacy ablation, not under the canonical ID.
5. **PRV/HRV availability.** TODO broadly requests SDNN/RMSSD/HRV; the spec makes duration,
   count, route, role and adjacency eligibility mandatory. V1 follows the stricter rule.
6. **Outer early stopping.** Historical completed routes used held-out folds to select epochs.
   Both current TODO and spec forbid it; those scores remain non-strict characterization.
7. **Full stage sequence.** One M0 status document states a more complete alternating sequence,
   while the accepted M2/user fact is only that S/W occurs before Relax. V1 uses the narrower
   confirmed fact and does not infer recovery timing.

## Implementation consequence / 实施结论

`final_pipeline_v1` implements the attached contract as its canonical boundary. TODO-only
routes are neither deleted nor falsely claimed complete. They are listed as V2 scope choices,
and adding one requires the same frozen data, fold, leakage, aggregation, OOF and bundle gates.

“Implemented” here means the engineering route and its acceptance evidence exist. It does not
mean the full candidate matrix has run or that synthetic/real-input smoke artifacts establish
Frailty3, external-PTT, clinical, or deployment performance. All four representation modules
exist, but the current real outer-fold executor dispatches `feature_vector` only; raw, matrix and
fusion scientific-runner requests fail closed. That remaining executor breadth gap must not be
hidden by their construction/forward/training contract tests.
