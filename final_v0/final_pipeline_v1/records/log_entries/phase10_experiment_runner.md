# Phase 10 — Frozen experiment runner / 冻结实验执行器

Date / 日期: 2026-08-15  
Scientific status / 科学状态: implementation verified; reduced results are smoke only / 实现已验证；reduced 结果仅为 smoke  
Primary module / 主模块: `src/ppg_frailty/experiment.py`

## Outcome / 结果

The V1 package now has a real feature-vector outer-fold runner and an unshortened
multi-cell orchestrator. The public reduced runner preserves the complete frozen
participant roster and executes preprocessing, direct SQI, outer-train-only
empirical SQI calibration, locked quality routing, feature extraction, unified
training and complete retained/drop OOF tracing. It never relaxes SQI or switches
artifact routes after a failure.

V1 现已具备真实 feature-vector outer-fold runner 与不截短的多 cell 调度器。
reduced runner 保留完整冻结 participant roster，并依次执行预处理、direct SQI、
仅 outer-train 拟合的经验 SQI 校准、锁定质量路由、特征提取、统一训练与完整
retained/drop OOF 追踪。任何失败都不会触发 SQI 放宽或伪影路线回退。

## Public API / 公开 API

```python
run_reduced_fold_experiment(
    config_path,
    *,
    repeat_index=0,
    fold_index=0,
    max_seconds_per_record=60.0,
    max_records_per_participant=1,
    fixed_epochs_override=1,
    output_dir=None,
) -> ExperimentResult

run_full_experiment(
    config_path,
    *,
    output_dir,
    repeats=tuple(range(5)),
    folds=tuple(range(5)),
) -> ExperimentResult
```

The reduced default is 60 seconds because the unchanged formal route was measured
at 12 seconds and 60 seconds: 12 seconds retained no participant, while 60 seconds
completed training and produced nonempty OOF. The full runner always uses complete
recordings, all eligible files and the configured epoch rule; it accepts no
shortening or epoch override.

reduced 默认值设为 60 秒，因为在完全相同的正式路由下实测了 12 秒与 60 秒：
12 秒无 participant 保留，60 秒完成训练并产生非空 OOF。full runner 始终使用
完整记录、所有合格文件与配置内 epoch 规则，不接受截短或 epoch override。

## Algorithm and leakage boundary / 算法与防泄漏边界

1. Run the single canonical `preflight_pipeline(..., mode='full')` and load the
   materialized `FrozenFoldRegistry`.
   运行唯一规范 preflight，并加载已物化的冻结折叠注册表。
2. Select records only inside the exact train plus OOF participant roster. The
   reduced cap chooses the longest eligible role recording per participant; it
   never removes a participant.
   仅在精确 train+OOF roster 中选择记录；reduced 每人选择最长合格角色记录，
   但绝不删除 participant。
3. Build synchronized direct signal views. First evaluate base SQI components with
   `fixed_formula_thresholds_v1`.
   构建同步 direct views，先以固定公式计算 SQI 基础分量。
4. Fit empirical SQI quantile bounds using outer-train participant rows only.
   OOF IDs are explicitly checked absent from fitted provenance.
   经验 SQI 分位边界仅由 outer-train participant 拟合，并显式证明无 OOF ID。
5. Evaluate formal direct SQI. High-quality non-motion records return directly.
   Motion-role override is applied before the run-locked `drop XOR reducer`
   branch. Static low-quality records follow the configured locked policy.
   计算正式 direct SQI；高质量非运动记录直接返回。motion override 位于锁定的
   `drop XOR reducer` 之前；静态低质量记录遵守配置锁定策略。
6. A non-identity reducer is accepted only as `ARTIFACT_RATE_ONLY`; post-route
   `Q_morph` must be not-applicable, and only post `Q_rate` may qualify it.
   非恒等 reducer 只能产生 rate-only 路线；post `Q_morph` 必须为 NA，仅
   post `Q_rate` 可决定保留。
7. Extract pulse/PRV, engineering, morphology and dual-optical features according
   to the route. Direct-only morphology/optical fields remain unavailable for
   rate-only records rather than being fabricated.
   按路线提取 pulse/PRV、工程、形态和双光学特征；rate-only 记录的 direct-only
   形态/光学字段保持不可用，不填造数值。
8. Build the canonical feature registry and fit imputation/scaling/model transforms
   inside `UnifiedTrainer.fit_estimator` on the exact outer-train IDs. Outer labels
   are not passed to the trainer.
   使用规范特征注册表，且缺失值填补、缩放与模型变换仅在精确 outer-train ID
   内由统一训练器拟合；outer 标签不传给训练器。
9. Predict OOF and aggregate file → role → participant with the canonical
   equal-weight hierarchy. Every selected OOF file is represented as retained or
   dropped; an all-dropped participant receives an explicit empty-probability trace.
   OOF 按 file → role → participant 等权聚合；每个已选 OOF 文件都以 retained
   或 dropped 出现，全丢 participant 使用显式空概率追踪。

## Fixed artifacts and immutability / 固定产物与不可覆盖

Each reduced run and each full cell writes:

- `run_manifest.json`
- `metrics_per_fold_seed.json`
- `confusion_matrices.json`
- `oof_window_predictions.parquet`
- `oof_file_predictions.parquet`
- `oof_subject_predictions.parquet`
- `oof_member_predictions.parquet`
- `experiment_result.json`

Feature-vector prediction begins at file level, so the window parquet is a
schema-bearing scientific-empty table. The member parquet is similarly marked
`not_an_ensemble_model`. Failed-closed runs write schema-bearing empty OOF tables
and never fabricate metrics. Outputs are staged on the same filesystem, atomically
published, and an existing target is rejected rather than overwritten.

feature-vector 预测从 file level 开始，因此 window parquet 是带 schema 和原因的
科学空表；member parquet 同样明确标为非 ensemble。failed-closed 运行写带
schema 的空 OOF，绝不伪造指标。输出先在同文件系统暂存、再原子发布；已存在
目标会被拒绝，不能覆盖。

## Real frozen-fold evidence / 真实冻结折叠证据

### 12-second gate evidence / 12 秒门禁证据

Persistent output / 持久输出:
`artifacts/experiments/reduced_real_r0_f0_12s_failed_closed`

- Status: `failed_closed`
- Scope: `smoke_not_scientific_benchmark`
- A dedicated diagnostic pass observed 29/29 selected recordings at
  `dropped_post_q_rate` with reason `post_q_rate_below_threshold`.
- The persistent JSON itself records the 23 outer-train participant IDs with zero
  retained files and contains empty OOF parquet tables. The 29/29 distribution is
  a diagnostic observation, not a field currently embedded in that manifest.

- 状态：`failed_closed`
- 范围：`smoke_not_scientific_benchmark`
- 独立诊断观察到 29/29 已选记录均为 `dropped_post_q_rate`，原因均为
  `post_q_rate_below_threshold`。
- 持久 JSON 本身记录 23 名 outer-train participant 零保留并保存空 OOF；
  29/29 分布是诊断观察，当前未嵌入该 manifest 字段。

### 60-second passing reference / 60 秒通过参考

Persistent output / 持久输出:
`artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2`

Authority / 权威指针:
`artifacts/experiments/reference_registry.json`. The earlier immutable directory
`reduced_real_r0_f0_reference` is retained but marked superseded because it was
generated before the all-missing-column width-preservation fix.

- Status: `passed`
- Scope: `smoke_not_scientific_benchmark` — never cite as a benchmark result.
- Post-fix immutable verification wall time: 40.889 seconds; the cell manifest
  records 39.9477 seconds for the cell execution itself.
- OOF participants: 5 retained of 6; coverage = 0.8333333333333334.
- Balanced accuracy = 0.5; macro-F1 = 0.48888888888888893.
- These values verify execution and OOF integrity only. They do not establish model
  selection, superiority or publication performance.

- 状态：`passed`
- 范围：`smoke_not_scientific_benchmark`，禁止作为 benchmark 引用。
- 修复后不可变验证的 wall time 为 40.889 秒；cell manifest 内部记录的 cell
  执行耗时为 39.9477 秒。
- OOF participant：6 人中保留 5 人；coverage = 0.8333333333333334。
- BA = 0.5；macro-F1 = 0.48888888888888893。
- 这些数值只验证执行和 OOF 完整性，不代表模型选择、优越性或论文性能结论。

## Automated verification / 自动验证

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONWARNINGS=error PYTHONPATH=final_v0/final_pipeline_v1/src python3 -B -m unittest -v final_v0/final_pipeline_v1/tests/integration/test_experiment_runner.py
```

Final focused result / 最终 focused 结果:

- 4 tests passed, 0 failures, 0 errors, 0 skips, with every warning promoted
  to an error.
- Elapsed: 54.149 seconds.
- The default suite includes a clean temporary real 60-second r0/f0 frozen-fold run.
- The synthetic three-class fixture uses the same production route and asserts:
  exact train-only calibrator IDs, no OOF fit IDs, all three labels, nonempty
  three-class probabilities and retained participant OOF.
- An AST assertion enforces exactly one definition for each public runner and
  rejects the removed placeholder contract string.
- A three-model regression test proves that an all-missing outer-train feature
  column produces no warning and preserves the frozen feature width.

## Defects found by real execution / 真实执行发现并修复的问题

1. The strict JSON writer initially imported the wrong module; corrected to the
   canonical root-restricted atomic writer in `provenance.py`.
2. Dropped OOF rows initially inherited a nonempty class order despite having an
   empty probability vector; they now carry an empty class order.
3. `OofWriter.write` was initially called as a class method; it is now instantiated.
4. Cell summaries now explicitly include class order for confusion artifacts.
5. The original 12-second default was demonstrated insufficient and changed to the
   shortest tested passing duration, 60 seconds.
6. Median imputers originally warned and removed all-missing route-specific columns.
   All three allow-listed baselines now set `keep_empty_features=True`, preserving
   the frozen feature registry width under strict `PYTHONWARNINGS=error` execution.

## Known limitations / 已知限制

- The implemented scientific cell executor currently supports
  `representation_mode=feature_vector`. Raw waveform, matrix and fusion configs
  fail closed explicitly; no adapter pretends they are feature vectors.
- The full 5×5 orchestration is implemented without shortening, but executing all
  25 cells was outside this phase's runtime budget and remains unexecuted here.
- Bundle export is not part of this runner phase.
- The 60-second real smoke contains all-null route-specific feature columns. The
  train-only sklearn imputer now preserves their registered width and fills them
  deterministically; full benchmark review should still quantify availability by
  route.
- Full 5×5 metrics remain unavailable until all candidate configurations are run
  under the unified protocol. No route ranking is claimed by this phase.
