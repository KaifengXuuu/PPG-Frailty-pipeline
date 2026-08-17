# Phase 10 — Strict acceptance and CPU CI / 严格验收与纯 CPU CI

- Date / 日期：2026-08-15
- Status / 状态：`complete`, strict gate `16/16`, CPU tests `146/146`
- Scope / 范围：only V1 acceptance tools, acceptance tests, acceptance artifacts, and this immutable log
- Scientific claim / 科学声明：none; all comparator outputs are synthetic contract evidence, not frailty or external PTT performance

## Process / 流程

1. Re-read specification sections 4, 8, and 10, `AGENTS.md`, `_agent/WRITE_RULES.md`, Git status, and the current V1 tree before implementation.
2. Converted canonical paths, typed containers, public facades, manifest/fold identities, four formal configs, model cards, tests, and scientific-claim rules into 16 fail-closed machine checks.
3. Added self-negative tests proving that missing target files, changed specification bytes, JSON `NaN`, pass-only functions, and unsupported metric claims fail.
4. Added a frozen ECG-like event fixture with one-to-one matching, event P/R/F1, timing error, HR MAE/RMSE/bias, PPI MAE, coverage, and symmetric raw/quality/reducer schemas.
5. Added deterministic label-shuffle sanity and a real 10,000-kernel ROCKET fit/joblib-load parity test. The ROCKET test checks 20,000 transform values, probability parity, and outer-training-only fitted participant IDs.
6. Ran full CPU CI from a clean temporary working directory with `PYTHONWARNINGS=error`, bytecode disabled, CUDA hidden, and bounded CPU thread counts.
7. Exercised 106 package imports, 24 registered modules, all four full config preflights, a real frozen-record/fold smoke, 2 artifact controls plus 7 reducers, 13 model routes, 4 DL sample rates, 5/10-second raw windows, and the 64-case physical-time grid contract.
8. Re-ran strict acceptance after all evidence existed; no failed or pending item remained.

中文说明：全过程将“代码/接口可运行”“synthetic 定量合约”“真实科学 benchmark”三层证据严格分开。缺少结果时不会补造指标；历史或 synthetic 数字必须有不可用于独立测试/真实 benchmark 的明确范围声明。

## Algorithm and contract results / 算法与合同结果

- Canonical boundary / 规范边界：68 required files present.
- Specification identity / 规范身份：41,122 bytes, 766 lines, SHA-256 `cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000`.
- Data protocol / 数据协议：261 records, 29 participants, 9 roles; corrected grouped 5×5 folds with seeds `42,10042,20042,30042,40042`; three materialized hashes byte-exact.
- Configs / 配置：4/4 real full preflight, each resolving all 25 held-out splits.
- Registry / 注册表：4 representations + 7 artifact reducers + 13 models = 24 modules; 15 canonical facade imports.
- Types / 类型：8 required dataclass contracts including explicit `q_morph=not_applicable` semantics.
- Code checks / 代码检查：152 bilingual AST files parsed; 116 active source/tool files contain no legacy runtime import or AST-level unfinished implementation.
- Tests / 测试：146 run, 146 passed, 0 failed, 0 errors, 0 skipped, warnings treated as errors.
- Strict acceptance / 严格验收：16 passed, 0 failed, 0 pending.
- CPU CI / 纯 CPU CI：all stages passed; no unexpected warning.

## Machine evidence / 机器证据

- `artifacts/acceptance/cpu_ci_current.json`
  - SHA-256: `51f5fb2e2859bf6cec631b54cdf8fdb49ac63a84d2c8a1860c8d0512b26d5193`
- `artifacts/acceptance/cpu_ci_tests_current.json`
  - SHA-256: `5efde787ec68ded03bcecd615c7c8da757265441331d9e449866553ab1896e23`
- `artifacts/acceptance/strict_acceptance_current.json`
  - SHA-256: `19fc5b6883e006141b5a65a71fca2d5d1f6a73ec1c6b22acc858c271a3573174`

The test report embeds the path/byte/SHA-256 identity of all test sources; editing a test invalidates the green report. Quantitative artifacts are preserved under `artifacts/acceptance/runs/` and distinguish raw/no-denoise, quality-only, and non-identity reducers.

## Known boundary / 已知边界

This gate proves implementation, isolation, serialization, and synthetic quantitative contracts. It does not claim that the complete corrected 5×5 frailty benchmark or a real external ECG/PTT benchmark has been executed. Those scientific runs must emit separate provenance-complete OOF/benchmark artifacts.

