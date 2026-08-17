# Phase 12 — Documentation acceptance handoff / 文档验收交接

Date / 日期: 2026-08-15
Status / 状态: documentation frozen for final machine acceptance / 文档已冻结，等待最终机器验收
Scope / 范围: documentation and one dependency-envelope correction inside `final_v0`; no `_agent` write.

## Outcome / 结果

V1 now has a bilingual status authority, an operational runbook, complete M0–M3/V1
navigation, evidence-bound comparison reports, and a 28-point V2 confirmation registry.
Every statement separates engineering implementation, real-input integration, real
single-fold training smoke, synthetic contract comparison, and unexecuted scientific
benchmark evidence.

V1 已具备中英双语状态权威、可复制运行手册、M0–M3/V1 总导航、绑定机器证据的五份
对照报告及 28 项 V2 人工确认清单。全部文档严格区分工程实现、真实输入集成、真实单折
训练 smoke、合成合同对照与尚未执行的科学 benchmark。

## Files and navigation / 文件与导航

- `STATUS.md`: two-axis engineering/scientific status, implementation workflow,
  evidence taxonomy, limitations, and live acceptance pointers.
- `RUNBOOK.md`: dependency profiles, public CLI commands, validation/CI, data
  materialization, real protocol smoke, all artifact/model comparisons, EKF/LPF,
  ablations, and real outer-fold commands.
- `README.md`: concise status, frozen boundaries, quick start, public evidence
  scope and report navigation.
- `final_v0/README.md`: M0, M1, M2, M3 and final_pipeline_v1 entry map.
- `docs/comparisons/01..05`: the five user-requested specification/TODO/completed
  work/local workflow/algorithm/V2 reports.
- `records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md`: 28 decision IDs and
  confirmed/partial/pending semantics.
- `records/v2_decision_points/INITIAL_CONSERVATIVE_DEFAULTS.md`: retained only as
  superseded provenance, never as the live decision authority.

## Evidence binding / 证据绑定

1. Engineering acceptance points to the current strict-acceptance and CPU-CI JSON
   rather than freezing a prose test count or source-tree hash.
2. Real input/protocol smoke is linked separately and correctly states that it emits
   no trained prediction metric.
3. The real 60-second r0/f0 training authority is selected by
   `artifacts/experiments/reference_registry.json` and points to
   `reduced_real_r0_f0_reference_width_preserved_v2`.
4. The current real smoke retained 5/6 OOF participants, coverage 0.8333 and BA 0.5;
   every document labels it `smoke_not_scientific_benchmark`, not a final score.
5. The 12-second run remains visible as structured fail-closed evidence. Its manifest
   stores the 23-member zero-retained outer-train list and empty OOF; the separate
   29/29 post-Q_rate drop diagnostic is documented as an observation, not a manifest field.
6. Synthetic artifact/model/gravity/physical-time evidence is explicitly excluded
   from Frailty3, external-PTT, clinical or deployment ranking claims.
7. The old pre-width-preservation 60-second directory remains immutable but is
   marked superseded; hand-written navigation links only the registry and current v2 result.

## Runner and algorithm boundary / Runner 与算法边界

- `run` is an input/protocol audit; `run-experiment` is the real
  training/evaluation entry.
- Passing reduced command uses `configs/motion_benchmark_v1.yaml`, not the raw
  reference-static configuration.
- Public reduced defaults are fixed at 60 seconds, one file per participant and one
  epoch-equivalent while preserving the complete frozen roster.
- Full with no repeat/fold requests all 25 cells; an explicit pair requests one
  full-length cell and remains incomplete-5×5 scope.
- The current formal cell executor is **feature_vector-only**. Raw, matrix and fusion
  modules have construction/forward/training contract coverage but fail closed in the
  scientific runner; documentation does not hide this gap.
- ShapeFormer is described as patch/downsample → mask-aware Transformer plus a
  parallel shapelet-distance branch, with outer-train roster/repeat/fold/time binding.
  It remains effect-size experimental and not PISD/original-paper parity.
- Direct 0.2–8 Hz versus historical 0.4–8 Hz is explicit pending point V2-028.

## Human decisions / 人工决定

- V2-007 is confirmed: online no-precalibration quaternion error-state EKF primary
  with causal 0.3 Hz LPF mandatory comparator.
- V2-006, V2-008 and V2-011 are partially confirmed: signal recommendation subset,
  strict PRV behavior, and SQI-first run-locked drop XOR rate-recovery order are
  frozen while their listed deployment/device remainders stay open.
- V2-015 is partially confirmed: NumPy, SciPy, scikit-learn and ONNX Runtime are
  user-authorized; the remaining dependency profiles require a formal decision.
- Every other ID remains pending unless its detailed entry explicitly says otherwise.

## Dependency envelope correction / 依赖边界修正

`pandas` moved from core dependencies to optional
`tabular = ["pandas>=2.3,<3"]`. Source has no pandas runtime import. This removes
the prior documentation/package contradiction without installing or authorizing
additional dependencies.

## Verification and freeze rule / 验证与冻结规则

- Markdown relative links and referenced local paths are checked after the final save.
- `pyproject.toml` is parsed with the standard-library TOML parser and dependency
  membership is asserted.
- V1 and global tracking/index generators are rerun after every logical write batch.
- Final test totals, source snapshots and hashes remain machine-authoritative in
  current acceptance artifacts; this log does not hard-code a value that can stale.
- After this entry and its tracking synchronization, documentation is frozen. Any
  later source change requires a new acceptance refresh and an explicit documentation
  audit, not an undocumented edit.

## Known incomplete scientific work / 尚未完成的科学工作

- Complete 5 repeats × 5 folds candidate reruns have not been executed.
- Raw, matrix and fusion do not yet share the real formal experiment executor.
- No independent Frailty3 test cohort exists.
- External PTT reducer ranking, 29-subject motion-detector retraining, recovery/hierarchy
  routes, full mobile/ONNX parity and target-device measurements remain future work.
