# Final Pipeline V1 runbook / V1 运行手册

> Commands below are non-interactive and are intended to be copied from the repository root.
> 以下命令均从项目根目录执行、无需交互；输出边界和科学声明边界与配置一起锁定。

## 1. Enter the isolated package / 进入隔离包

```bash
cd final_v0/final_pipeline_v1
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
python3 -B -m ppg_frailty.cli --help
```

The package reads frozen authorities outside V1 only through declared paths. Commands that write results must target a new path below `final_pipeline_v1`; historical root scripts/results are never overwritten.

本包只通过声明路径读取 V1 外的冻结权威文件；所有输出必须写入 `final_pipeline_v1` 内的新路径，禁止覆盖根目录历史脚本和结果。

## 2. Dependency profiles / 依赖配置

| Profile / 配置 | Install / 安装 | Purpose / 用途 |
|---|---|---|
| Core / 核心 | `python3 -m pip install -e .` | NumPy, SciPy, scikit-learn, PyYAML, joblib; validation, classical routes, signal/features, CLI. |
| Deep / 深度训练 | `python3 -m pip install -e ".[deep]"` | Optional PyTorch model construction/training. Formal project authorization remains a V2 point. |
| Tabular / 表格 | `python3 -m pip install -e ".[tabular]"` | Optional pandas output/inspection; not a core runtime dependency. |
| Parquet / 正式 OOF | `python3 -m pip install -e ".[parquet]"` | Optional pyarrow; required when a formal run demands canonical Parquet output. |
| ONNX / ONNX 推理 | `python3 -m pip install -e ".[onnx]"` | Optional ONNX Runtime. The user has allowed ONNX Runtime, but V1 does not claim export/mobile parity. |

User authorization currently covers NumPy, SciPy, scikit-learn, and ONNX Runtime. Optional-package installation is not itself evidence that its scientific/deployment gate has passed.

用户当前明确允许 NumPy、SciPy、scikit-learn 与 ONNX Runtime；安装一个可选包不等于对应科学或部署验收已经通过。

## 3. Inspect and validate / 检查与验证

```bash
python3 -B -m ppg_frailty.cli list-modules --family all
python3 -B -m ppg_frailty.cli validate --all-configs
python3 -B -m ppg_frailty.cli validate --config configs/reference_static_v1.yaml
```

The four registered configurations are:

- `configs/reference_static_v1.yaml` — static B/R reference, raw representation, identity/direct route.
- `configs/reference_all_roles_v1.yaml` — all confirmed roles under the same frozen split contract.
- `configs/motion_benchmark_v1.yaml` — motion/rate-recovery comparison boundary.
- `configs/feature_matrix_v1.yaml` — ordered D×32 matrix route.

## 4. Run tests and acceptance / 运行测试与验收

```bash
python3 -B -m ppg_frailty.cli test --suite all --report artifacts/test_reports/manual_all_tests.json --verbosity 1
python3 -B tools/acceptance_gate.py --write-report artifacts/acceptance/strict_acceptance_manual.json
python3 -B tools/run_cpu_ci.py --report artifacts/acceptance/cpu_ci_manual.json
```

- `test` runs the registered unit/integration suites.
- `acceptance_gate.py` checks exact target files, contracts, current evidence, claim boundaries, and pending/failure states.
- `run_cpu_ci.py` runs warnings-as-errors tests, import sweep, public CLI checks, one real frozen-record smoke, all registered synthetic reducer/model comparisons, the 5/10 s ablation, and strict acceptance from a clean temporary working directory.
- `--skip-quantitative` is diagnostic only; strict acceptance is expected to remain pending if current quantitative evidence is absent.

These commands test engineering behavior. They do not train the full 5×5 scientific benchmark.

这些命令验证工程行为，不会自动完成 5×5 科学 benchmark。

## 5. Re-materialize frozen data contracts / 重建冻结数据合同

```bash
python3 -B -m ppg_frailty.cli build-data --confirm-byte-rehash
```

This intentionally re-reads and hashes all 261 frozen internal sources. Run it only when an authority, manifest contract, or byte-level integrity audit must be refreshed. It does not invent a new split: membership is imported from the frozen corrected registry.

该命令会逐字节重读并哈希 261 份冻结内部数据，仅在权威、manifest 合同或完整性证据需要更新时执行；它不会重新随机生成折。

## 6. Real-input protocol smoke / 真实输入协议冒烟

```bash
python3 -B -m ppg_frailty.cli run \
  --config configs/reference_static_v1.yaml \
  --mode smoke \
  --output artifacts/runs/reference_static_input_smoke.json
```

This route audits the frozen manifest/fold/config, reads a real record, executes signal/peak/window/representation integration, and emits no untrained prediction metric. `--mode full` expands the input/protocol audit; it is still not the training experiment runner.

此处 `run` 是真实输入与协议集成检查，不是训练入口；不得把其 `smoke_passed` 解读为分类性能。

## 7. Parallel artifact comparisons / 并行伪影路线对照

```bash
python3 -B -m ppg_frailty.cli compare artifacts \
  --reducers identity nlms_imu_anc ssa_decomposition spectral_mask pca_bss fastica_bss nmf_bss \
  --duration-s 10 \
  --seed 42 \
  --output artifacts/runs/artifact_contract_comparison.json
```

The result is a synthetic interface/known-signal contract comparison. Use the same reducer IDs later under one frozen external-PTT split for scientific HR/PPI/coverage ranking; do not select a final reducer from the synthetic result.

输出仅是合成信号合同对照。未来科学选择必须在统一外部 PTT 折上比较 HR/PPI/coverage。

## 8. Parallel model comparisons / 并行模型路线对照

```bash
python3 -B -m ppg_frailty.cli compare models \
  --models CompactCNN1D InceptionTimeFull InceptionTimeSmall InceptionTimeMatrix \
           InceptionTimeFiveMemberEnsemble ROCKET MiniROCKET LogisticRegressionL2 \
           RBFSVM ExtraTrees ShapeFormerEffectSize FileBagFusionCompact FileBagFusionInception \
  --seed 42 \
  --output artifacts/runs/model_contract_comparison.json
```

The command exercises all thirteen registered model contracts at reduced synthetic scale. It is not a Frailty3 leaderboard. ShapeFormer remains the local effect-size experimental route with patch/downsample before mask-aware attention, not PISD parity.

该命令只验证 13 条模型合同；不得形成 Frailty3 排名。ShapeFormer 仍是本地效应量实验路线。

## 9. EKF versus LPF comparator / EKF 与 LPF 对照

```bash
python3 -B -m ppg_frailty.cli compare imu-gravity \
  --duration-s 12 \
  --seed 42 \
  --output artifacts/runs/imu_gravity_contract_comparison.json
```

This known-truth synthetic test compares the confirmed no-precalibration quaternion ESKF primary with the causal 0.3 Hz LPF gravity comparator. Human/device validation must be reported separately.

这是已确认主路线的已知真值工程对照，不是 29-subject 人体运动验证。

## 10. One-factor ablations / 单因素消融

```bash
python3 -B -m ppg_frailty.cli ablate --factor artifact --seed 42 --output artifacts/runs/ablate_artifact.json
python3 -B -m ppg_frailty.cli ablate --factor model --seed 42 --output artifacts/runs/ablate_model.json
python3 -B -m ppg_frailty.cli ablate --factor dl_fs --seed 42 --output artifacts/runs/ablate_dl_fs.json
python3 -B -m ppg_frailty.cli ablate --factor raw_window_s --seed 42 --output artifacts/runs/ablate_raw_window.json
python3 -B -m ppg_frailty.cli ablate --factor physical_time --seed 42 --output artifacts/runs/ablate_physical_time.json
```

A valid scientific ablation later must keep manifest, participant folds, seeds, epoch budget, aggregation, and evaluation identity fixed and change one declared factor only. Synthetic command output proves execution/schema, not outcome superiority.

## 11. Scientific experiment runner / 科学实验入口

The public training/evaluation command is `run-experiment`. The only currently
supported formal executor representation is `feature_vector`, so the passing reduced example
uses `motion_benchmark_v1.yaml`:

```bash
python3 -B -m ppg_frailty.cli run-experiment   --config configs/motion_benchmark_v1.yaml   --budget reduced-smoke   --repeat 0   --fold 0   --output-dir artifacts/experiments/my_reduced_r0_f0
```

Reduced-smoke invariants are not user-overridable through the CLI:

- complete frozen outer roster;
- 60 seconds maximum per selected record;
- one record per participant;
- one epoch-equivalent training budget;
- default r0/f0 when the repeat/fold pair is omitted;
- `scientific_scope=smoke_not_scientific_benchmark`;
- non-zero command exit when the result is `failed_closed`.

The authority pointer is
[reference_registry.json](artifacts/experiments/reference_registry.json); its current passing
reference is
[reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json](artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json).
The earlier immutable `reduced_real_r0_f0_reference` is retained only as a superseded pre-fix run.
The current width-preserved reference selected one record for each of 29 participants and fitted SQI/model objects on the 23 outer-train
participants only, retained 5/6 OOF participants, and reported coverage 0.8333 and BA 0.5.
Those numbers verify a real single-fold pipeline and abstention trace; they are not a final
candidate score. The
[12-second diagnostic](artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)
closed without metrics after the outer-train roster had zero retained files.

Run all 25 full-length cells only after budget and output dependencies are authorized:

```bash
python3 -B -m ppg_frailty.cli run-experiment   --config configs/motion_benchmark_v1.yaml   --budget full   --output-dir artifacts/experiments/motion_feature_vector_full_5x5
```

Run one uncapped/full-budget cell by supplying repeat and fold together:

```bash
python3 -B -m ppg_frailty.cli run-experiment   --config configs/motion_benchmark_v1.yaml   --budget full   --repeat 0   --fold 0   --output-dir artifacts/experiments/motion_feature_vector_full_r0_f0
```

For `--budget full`, omitting both indices means all 25 cells; supplying only one index is
invalid. An explicit pair is a selected full cell with scope
`selected_full_length_cells_not_complete_5x5`, not the completed benchmark.

**Current limitation / 当前限制：** `reference_static_v1.yaml` is raw and is not a passing
training-runner example. Raw, feature-matrix, and fusion modules are implemented and contract
tested, but the current formal executor rejects their representation instead of substituting
feature_vector. Therefore only feature-vector can currently be launched end-to-end through
`run-experiment`; this restriction must remain visible until a later executor expansion.

## 12. Output interpretation and failure policy / 输出解释与失败策略

- `passed` or `smoke_passed` is scoped by the artifact's `scientific_scope` and schema.
- `failed_closed` means the requested contract could not be satisfied; do not shrink a participant roster, relax SQI, substitute identity, or change folds to obtain a number.
- Dropped/no-result rows remain in coverage and OOF trace with reasons.
- Internal unavailable values are NaN plus validity=false; strict JSON serializes them as null, never as a valid zero.
- Non-identity routes expose Q_morph=not_applicable and may produce rate/HR/PPI/eligible PRV only.
- Every formal result must retain config, manifest, fold, preprocessing, feature, model, code, environment, and aggregation identity.

## 13. Where to look next / 后续导航

- [Current status / 当前状态](STATUS.md)
- [Configuration and quick start / 配置与快速开始](README.md)
- [Algorithm diagrams / 算法流程图](docs/algorithms/README.md)
- [Comparison reports / 五份对照报告](docs/comparisons/01_SPEC_VS_TODO_OVERLAP_AND_DIFFERENCES.md)
- [V2 confirmation points / V2 人工确认点](records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md)
- [Detailed package tree / 详细文件树](PROJECT_TREE.md)
