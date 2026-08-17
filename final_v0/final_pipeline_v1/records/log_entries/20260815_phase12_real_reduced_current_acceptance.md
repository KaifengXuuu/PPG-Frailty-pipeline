# Phase 12 — Real reduced current acceptance / 真实 reduced current 验收

- Date / 日期：2026-08-15
- Status / 状态：`complete`
- CPU CI / 纯 CPU CI：10/10 stages passed
- Strict gate / 严格门禁：18/18 passed, 0 pending, 0 failed
- Tests / 测试：159/159 passed, 0 failed, 0 errors, 0 skipped
- Scientific scope / 科学范围：real reduced smoke plus synthetic contracts; not a 5×5 frailty benchmark and not an external PTT benchmark

## Process / 流程

1. Froze the documentation tree before generating current hashes. 文档树冻结后才生成 current hash，避免报告与最终文档版本错位。
2. Extended `tools/run_cpu_ci.py` with a mandatory public-CLI execution of `motion_benchmark_v1`, repeat 0, fold 0, reduced-smoke budget. CPU CI uses `PYTHONWARNINGS=error`, disables bytecode/CUDA, limits CPU thread pools, and runs from an isolated temporary working directory.
3. Added a deterministic active-source snapshot over `src/**/*.py`, `tools/**/*.py`, `tests/**/*.py`, registered YAML configs and dependency metadata. The strict gate recomputes the same canonical path/byte/SHA-256 tree and rejects stale evidence.
4. Added a strict real-experiment validator that reads the materialized fold registry rather than regenerating folds. It verifies the exact 23-participant outer-train roster and exact six-participant r0/f0 held-out roster.
5. Validated all eight fixed experiment artifacts, exact file and participant OOF schemas, exact-once held-out rows, complete trace hashes, explicit drop rows, train-only model/SQI provenance, scientific-empty window/member tables, metrics/count/coverage consistency, and confusion-matrix consistency.
6. Added controlled negative tests proving that a reduced result cannot be relabelled as a 5×5 benchmark and that duplicate held-out subject rows fail the gate.
7. Re-ran strict acceptance after the new CPU report existed, proving that current reports remain recursively auditable and do not self-trigger the scientific-claim detector.

中文算法说明：门禁不比较或固定任何 BA 数值。它只要求指标存在、为有限值且与同一 held-out OOF 和 confusion matrix 内部一致。这样可以检出空结果、错名单、泄漏、重复 OOF、结构漂移与伪造范围声明，同时不会把一次 reduced smoke 变成性能回归阈值或论文结论。

## Results / 结果

- Active source / 活动源码：159 files; tree SHA-256 `d4ccf5a24a4c22ae5e75438d41ef6d1e5125e6b0b55230a18431bbd48fbecc13`.
- Test source / 测试源码：37 files; tree SHA-256 `4747fa8aef8d1f0baad1debca8e01ed5ecb1142247a40d981ed692922cf51d5b`.
- Full tests / 全套测试：159 run, 159 passed, 0 skipped; warnings are errors.
- CPU stages / CI 阶段：10 passed, `failed_stages=[]`.
- Strict checks / 严格检查：18 passed, 0 pending, 0 failed.
- Real current run / 真实 current run：`artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223`.
- Frozen roster / 冻结名单：23 exact outer-train participants; 6 exact held-out participants; every file/subject OOF participant appears exactly once.
- Retained evidence / 保留证据：5 of 6 held-out participants retained; at least one retained result is a gate invariant, but the exact count is not a performance threshold.
- Formal representation boundary / 正式表征边界：only `feature_vector` is supported by the current formal experiment cell executor. `raw`, `feature_matrix`, and `fusion` remain runnable through comparison/tests but fail closed in formal experiment execution.
- Metric threshold policy / 指标阈值策略：no outcome metric value, including BA, is locked by this gate.

## Machine evidence and SHA-256 / 机器证据与哈希

- `artifacts/acceptance/cpu_ci_current.json`  
  SHA-256 `0f2fcdee5096f96a65658fbe607d1f79acc2413ec6a55026ed21938f74a241f3`
- `artifacts/acceptance/cpu_ci_tests_current.json`  
  SHA-256 `819392c2d027802fcb2ff68ed19f92c727bfe26bdde46688145bef2c85a3122f`
- `artifacts/acceptance/strict_acceptance_current.json`  
  SHA-256 `1bab31fce649eda48eeddf25c8cd07fc1bd6ad94202689cb03e9dafce5210464`
- `artifacts/acceptance/source_snapshot_current.json`  
  SHA-256 `c0472928effe9ca4973dad59ba21dd0283eb5848c55a51742cc00222d91e5875`
- `artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/run_manifest.json`  
  SHA-256 `dfdf2d9ce9c68e2f9927b9bfc215b535653dcceeb23179c3c6dc8832143d0e5d`
- `artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/experiment_result.json`  
  SHA-256 `27ca402202d96357558e29dc783e97a4090054c4077629200341511c2eebfe47`

## Known boundary / 已知边界

This phase proves a real, train-only, frozen-roster feature-vector r0/f0 execution and complete current artifacts. It does not prove the unexecuted full 25-cell benchmark, raw/matrix/fusion formal training, external ECG/PTT performance, model superiority, or independent-test performance.

本阶段证明真实、train-only、冻结名单的 feature-vector r0/f0 执行及 current 工件完整性；不证明尚未执行的 25-cell 完整 benchmark、raw/matrix/fusion 正式训练、外部 ECG/PTT 性能、模型优越性或独立测试性能。
