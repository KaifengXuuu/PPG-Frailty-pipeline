# Final Pipeline V1 status / V1 状态总览

> This page separates engineering acceptance from scientific evidence.
> 本页把“工程验收通过”与“科学性能已经得到证明”严格分开；两者不得互相替代。

## Executive status / 执行摘要

| Axis / 轴 | Current state / 当前状态 | What it means / 含义 |
|---|---|---|
| Engineering implementation / 工程实现 | Acceptance-ready at the current checkpoint / 当前检查点已具备验收条件 | The isolated package contains frozen data/fold contracts, signal/QC/SQI, artifact routes, features, four representations, thirteen model routes, training/evaluation/OOF/bundle layers, CLI comparisons, ablations, regression tests, and strict gates. |
| Scientific benchmark / 科学 benchmark | Not completed / 未完成 | No full 5 repeats × 5 folds candidate matrix has been executed. No current artifact is an independent Frailty3 test, an external-PTT reducer ranking, or a publication-grade model leaderboard. |
| Deployment / 部署 | Python CPU contract only / 仅完成 Python CPU 合同 | ONNX/mobile parity, target-device latency, memory, power, and final processor selection remain V2 decisions and measurements. |
| Historical TODO / 历史 TODO | Paused by direct user request / 已按用户要求暂停 | V1 follows the attached merged dev0 specification. TODO-only motion-retraining, recovery, hierarchy, broad method-zoo, Top-5, and mobile closure remain documented scope differences. |

## Implemented workflow / 已落地 workflow

1. **Authority and provenance / 权威与溯源** — byte-locked merged specification, versioned configuration, source hashes, exact environment/protocol identity, and fail-closed path boundaries.
2. **Data and folds / 数据与折** — 261 internal records, 29 participants, nine roles, unchanged three-class labels, and the frozen balanced subject-level 5×5 registry with seeds 42/10042/20042/30042/40042. Runtime split regeneration is forbidden.
3. **Signal layer / 信号层** — typed native/filter/analysis/artifact views, one shared window plan, bounded repair, 400 Hz engineering axis, zero-phase direct PPG filtering, no-precalibration quaternion error-state EKF, and a mandatory causal 0.3 Hz LPF comparator.
4. **Quality and routing / 质量与路由** — direct SQI first; separate Q_rate and Q_morph; optional motion override; high-quality direct bypass; one run locks drop XOR rate-recovery; non-identity output is requalified only by Q_rate and has Q_morph=not_applicable.
5. **Physiology and features / 生理量与特征** — shared peaks/PPI/PRV backend, adjacency-preserving eligibility, direct-only morphology and dual-wavelength optical features, explicit validity, ordered feature vector, and D×32 feature matrix.
6. **Artifact comparison / 伪影对照** — identity, NLMS/IMU ANC, SSA, STFT/IMU spectral mask, PCA, FastICA, and NMF under one typed no-fallback interface. Non-identity outputs are rate-recovery signals, never morphology-preserving claims.
7. **Representations and models / 表征与模型** — raw, feature_vector, feature_matrix, and fusion; thirteen registered model routes including exact CompactCNN/Inception identities, five-member ensemble, ROCKET/MiniROCKET, three classical baselines, file-bag fusion, and experimental ShapeFormer.
8. **Training and evidence / 训练与证据** — outer-participant isolation, train-only fitted objects, fixed-epoch or legal inner selection, complete retained/dropped OOF trace, window→file→role→participant aggregation, participant-unit metrics, paired ablation identity, and transactional bundle/golden parity.
9. **ShapeFormer scope / ShapeFormer 范围** — the experimental route now uses patch/downsampling before mask-aware Transformer attention, parallel trainable shapelet-distance features, and outer-train roster/repeat/fold/time-scale binding. It remains an effect-size discovery experiment, not PISD or original-paper parity.

## Evidence classes / 证据分层

| Evidence class / 证据类型 | Current artifact / 当前产物 | Valid conclusion / 可得结论 | Forbidden conclusion / 禁止结论 |
|---|---|---|---|
| Real-input integration smoke / 真实输入集成冒烟 | [integration_smoke_canonical_manual.json](artifacts/test_reports/integration_smoke_canonical_manual.json) | A frozen real record can be read, preprocessed, peak-checked, windowed, represented, and provenance-bound through the public input/protocol route. | It does not train a classifier and emits no scientific metric. |
| Real single-fold training smoke / 真实单折训练冒烟 | [authority registry](artifacts/experiments/reference_registry.json); [current 60 s passed result](artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json); [12 s fail-closed result](artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json) | The frozen r0/f0 feature-vector route can fit train-only SQI/model objects, retain a complete OOF roster with explicit drop, aggregate and persist all mandatory artifacts. The 60 s diagnostic retained 5/6 OOF participants (coverage 0.8333, BA 0.5); the 12 s diagnostic closed without metrics after zero retained outer-train subjects. | Scope is `smoke_not_scientific_benchmark`: one fold, one record per participant, one epoch/classical fit path. It is not the 5×5 candidate benchmark and the numerical result must not be ranked or published as final performance. |
| Synthetic artifact comparison / 合成伪影对照 | [artifact_comparison_canonical_manual.json](artifacts/test_reports/artifact_comparison_canonical_manual.json) | All named reducer interfaces execute under known synthetic signals and expose route/QC/runtime fields. | It is not an external PTT ECG-reference benchmark and cannot select a clinical reducer. |
| Synthetic model comparison / 合成模型对照 | [model_comparison_all13_manual.json](artifacts/test_reports/model_comparison_all13_manual.json) | All thirteen model construction/forward/fit contract paths execute at reduced scale. | It is not a Frailty3 ranking and reduced variants cannot be reported as full scientific candidates. |
| Synthetic gravity comparison / 合成重力对照 | [imu_gravity_comparison_manual.json](artifacts/test_reports/imu_gravity_comparison_manual.json) | EKF and LPF comparator paths can be quantified against known synthetic truth. | It is not a 29-subject human-motion validation. |
| Physical-time audit / 物理时间审计 | [physical_time_contract_manual.json](artifacts/test_reports/physical_time_contract_manual.json) | Sampling-rate/context/kernel physical-time identities are constructible and auditable. | It contains no frozen 5×5 outcome comparison. |
| Strict acceptance and CPU CI / 严格验收与 CPU CI | [strict_acceptance_current.json](artifacts/acceptance/strict_acceptance_current.json), [cpu_ci_current.json](artifacts/acceptance/cpu_ci_current.json) | Current source, contracts, CLI, imports, warnings-as-errors tests, real smoke, synthetic matrices, and evidence scope pass the recorded gates. | Engineering pass does not establish medical validity, generalization, or independent performance. |

## Current limitations / 当前限制

- The external PTT dataset has ECG timing reference but no Frailty3 labels and an unresolved pleth wavelength mapping. It can validate rate recovery, not frailty classification or processed-waveform morphology.
- The 29-subject cohort remains small for a broad candidate matrix. Five repeats measure split sensitivity but do not create independent participants.
- Device saturation rails and several endpoint thresholds are not invented; unresolved values remain visible review points.
- The motion detector retrain on B/R versus S/W, recovery-timing features, hierarchical classification, and broad non-stationary/trajectory routes are not silently added to canonical V1.
- Full benchmark execution requires an approved candidate/epoch/hardware budget and formal output dependencies. Unrun scores remain absent.
- The persisted end-to-end training smoke currently exercises the `feature_vector` route. Raw, feature-matrix, and fusion routes have construction/forward/training contract coverage, but no persisted same-runner 5×5 scientific result; unsupported scientific dispatch must fail closed rather than be described as complete evidence.
- The only confirmed deployment-adjacent gravity choice is no-precalibration EKF primary with LPF comparison. Target hardware and ONNX/mobile parity are not yet acceptance claims.

## Final-run updates / 最终运行更新

<!-- FINAL_RUN_UPDATES_START -->
This block must be refreshed after the last code or test write. The machine-readable authorities are the two linked current artifacts; prose here deliberately does not freeze a test count or a source-tree hash that could become stale during integration.

此区块必须在最后一次代码或测试写入后更新。权威值以当前 strict-acceptance 与 CPU-CI JSON 为准；在集成仍可能变化时，本页不把测试数量或源码树哈希写成永久常量。
<!-- FINAL_RUN_UPDATES_END -->

## Navigation / 导航

- [README and quick start / 入口与快速开始](README.md)
- [RUNBOOK / 运行手册](RUNBOOK.md)
- [Five requested comparison reports / 五份指定对照报告](docs/comparisons/01_SPEC_VS_TODO_OVERLAP_AND_DIFFERENCES.md)
- [Detailed V2 confirmation registry / V2 人工确认清单](records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md)
- [Algorithm workflow diagrams / 算法流程图](docs/algorithms/README.md)
- [Detailed file tree and hashes / 详细文件树与哈希](PROJECT_TREE.md)
