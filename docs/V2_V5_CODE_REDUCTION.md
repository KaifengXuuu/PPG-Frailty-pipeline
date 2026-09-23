# V2 / V5 Code Size and Reduction Audit

This translated historical audit is retained in V6 for reference. Counts and
validation status describe the original V5 audit snapshot, not new V6 measurements.

Line count measures review effort, not algorithmic equivalence. The V5 reduction
preserved the thesis workflow, numerical algorithms, and all analysis capabilities,
while removing duplicate runners, embedded presentation, copied historical templates,
and layered wrappers. Numerical equivalence still requires comparison of all
25 folds in the same frozen environment.

## Consistent Counting Scope

Three scopes are kept separate to avoid mixing tests, documentation, or external
dependencies into core production code:

1. **Legacy sweep recursive local Python closure**: the root entry points and their
   direct and recursive dependencies within this project.
2. **V5 core production code**: `src/ppg_frailty/**/*.py`, excluding independent Dash,
   plus non-Dash production entry points and `tools/*.py`. This is the scope of the
   user's 70,000-line target.
3. **All V5 Python**: includes tests and independent Dash code; an engineering
   inventory, not a measure of the production core.

All counts are physical lines, including blank lines and comments. The core excludes
tests, docs, README, `_agent`, `src/ppg_frailty/dashboard/`, `dashboard.py`, historical
outputs, caches, the three output roots, `__pycache__`, and third-party source.
`tools/*.py` is included; Dash is reported separately.

The audited V5 core contained **66,169 lines**. The same command on the read-only
snapshot saved at the start of that refactor
(`/tmp/v5-global-refactor-before-EEAX7K/final_pipeline_v5`) returned **131,179 lines**:
a net reduction of **65,010 lines (49.56%)**, leaving 3,831 lines below 70,000.
An earlier manual inventory reported 131,213 lines, 34 more than the recountable
snapshot. Ratios and net changes here use only the consistently counted snapshot;
the two baselines are not mixed.

## Legacy Sweep Recursive Closure

Recursively scanning local Python references from root-level
`frailty_3class_overfitting_sweep.py` and `analyze_sweep.py` produced:

| File | Physical LOC | Role |
|---|---:|---|
| `frailty_3class_classifier.py` | 3,532 | Data, features, models, and training |
| `frailty_3class_overfitting_sweep.py` | 1,707 | Sweep entry point and experiment combinations |
| `analyze_sweep.py` | 1,372 | Sweep analysis and plots |
| `frailty_3class_holdout_eval.py` | 701 | Holdout/configuration helpers |
| `shapeformer_port.py` | 512 | Local ShapeFormer adapter |
| **Total recursive local closure** | **7,824** | `3,532+1,707+1,372+701+512` |

The two user-facing entry points total **3,079 lines**:
1,707 for `frailty_3class_overfitting_sweep.py` plus 1,372 for `analyze_sweep.py`.

The external ShapeFormer PISD implementation has approximately **636 lines**. It is
outside this repository's local Python closure and is not included in 7,824. Future
deployable-source-bundle comparisons must list external dependencies separately,
not silently add them to the local count.

## Core Reduction in the Audited Refactor

This file-by-file comparison of the pre-refactor snapshot and audited tree is sorted
by net reduction. It avoids double-counting moved code in informal groups. New shared
files are already included in the overall net change above.

| Rank | File | Before | Audited count | Net change | Main reduction |
|---:|---|---:|---:|---:|---|
| 1 | `quality/stage5_pre.py` | 7,454 | 1,044 | −6,410 | Removed training-path presentation templates and duplicate artifact wrappers; retained the Stage5 numerical runner |
| 2 | `experiment.py` | 10,054 | 4,112 | −5,942 | Consolidated workflow branches, fold products, and refit scheduling boilerplate |
| 3 | `reporting/analyze.py` | 5,442 | 408 | −5,034 | Replaced repeated per-figure/per-table dispatch with shared registries/specifications |
| 4 | `reporting/report.py` | 4,397 | 0 | −4,397 | Removed an unused V2/HTML compatibility entry point; retained independent analysis reporting |
| 5 | `reporting/incomplete.py` | 2,724 | 0 | −2,724 | Replaced copied failure reports with a compact execution audit |
| 6 | `study/hyperparameter.py` | 3,422 | 762 | −2,660 | Reused phase runners, statistics, and artifact writers |
| 7 | `reporting/historical_suite.py` | 2,218 | 0 | −2,218 | Removed unused legacy-suite wrappers; retained specialized-suite implementations |
| 8 | `reporting/historical.py` | 2,080 | 241 | −1,839 | Applied one collection/rendering protocol to historical comparisons |
| 9 | `models/factory.py` | 4,148 | 2,406 | −1,742 | Used table-driven construction while preserving architectures and initialization equations |
| 10 | `module_registry.py` | 2,698 | 1,220 | −1,478 | Shared one definition for the module catalog, defaults, and CLI metadata |
| 11 | `config.py` | 2,465 | 1,057 | −1,408 | Consolidated parsing, defaults, and validation |
| 12 | `study/schema.py` | 2,186 | 799 | −1,387 | Reused plan/case field handling through a declarative schema |
| 13 | `pipeline.py` | 1,715 | 352 | −1,363 | Retained shared preflight/record loading; removed an unused legacy smoke writer |
| 14 | `quality/motion_runner.py` | 2,363 | 1,220 | −1,143 | Unified internal/PTT, OOF, evidence, and bundle handling |
| 15 | `v5/specialized.py` | 1,497 | 389 | −1,108 | Shared plan, report, and provenance scheduling across specialized entry points |
| 16 | `v5/model_config_export.py` | 1,291 | 257 | −1,034 | Consolidated model selection, manifests, and reusable-parameter export |
| 17 | `reporting/tabular.py` | 1,322 | 314 | −1,008 | Unified table specifications, N/A rows, and serialization |
| 18 | `training/trainer.py` | 2,758 | 1,853 | −905 | Consolidated equivalent epoch, checkpoint, and prediction handling |
| 19 | `reporting/conclusions.py` | 1,585 | 699 | −886 | Shared conclusion tables, statistical projections, and missing-data handling |
| 20 | `study/runner.py` | 1,774 | 911 | −863 | Unified serial/thread/process, resume, and fail-fast lifecycles |

Preserving a module does not require retaining redundant code verbatim. This scope
preserved public entry points, equations, architectures, thresholds, sampling, splits,
workflow order, statistical searches, table fields, and figure types. Removed code
comprised repeated implementations of the same logic, embedded presentation, and
one-off historical wrappers.

The new 352-line `v5_reporting/execution_audit.py` restored failure/interruption
auditing from `reporting/incomplete.py` through a shared writer. It reads metadata,
not OOF predictions or weights, and makes no scientific conclusions. Compared with
the original 2,724-line copied implementation, this capability alone lost 2,372 lines.

## Major Changes, Ordered by Reduction

1. **Separate training from presentation.** The pipeline writes predictions, metrics,
   Excel, and weights only. A read-only report entry point reconstructs Stage5,
   hyperparameter, and ordinary-classification figures/HTML.
2. **Share specialized outputs.** Historical computations retain strict loaders,
   numerical runners, resume, and model export; only table/figure/text templates
   move to reusable reporters.
3. **Unify motion data paths.** Internal/PTT, transfer/reverse, thresholds, OOF, and
   window-tensor semantics remain unchanged; manifest/evidence and bundle handling
   are shared.
4. **Unify the study lifecycle.** Single/grid/ablation/catalog/legacy-bridge studies
   share expansion, cell execution, resume, and `run/comparison/repeat/fold` outputs.
5. **Make reports and statistics table-driven.** Compact specifications generate
   metrics, missing values, aggregation, and output types, avoiding repeated
   CSV/Markdown/HTML code for each analysis.
6. **Generate the configuration catalog dynamically.** One mapping describes modules,
   defaults, ranges, and paths; CLI, Dash, and help no longer maintain separate tables.

The audited refactor removed another 3,627 core production lines from the preceding
69,796-line tree. It deleted duplicate report/evaluate compatibility entry points and
singular/plural forwarding layers, bound internal calls to the sole implementation,
simplified logic-free package re-exports and 101 duplicate symbol lists, and removed
the old parity gate that only checked a source-SHA allowlist. It also removed an
unreachable smoke writer, a cache class superseded by recording cache, and an unused
Dash download helper, while retaining failed-run auditing in 352 shared lines.
The actual fold/row-identity/result comparator, `tools/compare_v2_v5_outputs.py`,
remains and uses `atol=1e-6`.

## Additional V5 Capabilities Retained Deliberately

V5 does not attempt to compress every added capability into the old entry points'
7,824 lines. These are user-requested production features:

- Complete preset/manual/study CLI and live parameter catalog.
- Per-fold multilevel predictions, learned bundles, resume, concurrency, and leakage-safe cache.
- Three output-root contracts, separate pipeline/report Excel semantics, and automatic model_config export.
- Independent, composable analysis reporting.
- Dash training/stop/inference/queue/preview/download controls.
- Reusable specialized/legacy-comparison entry points.

Thus 7,824 is a comparison with the old sweep's local closure, not a reasonable upper
bound for all of V5.

## Reproducing the Historical Count

From the repository root, this command counts core physical LOC using the scope
defined above. It targets historical V5, not the translated V6 tree:

```bash
find final_v0/final_pipeline_v5 -type f -name '*.py' \
  -not -path '*/tests/*' -not -path '*/dashboard/*' \
  -not -path '*/cache/*' -not -path '*/pipeline_output/*' \
  -not -path '*/report_output/*' -not -path '*/model_config/*' \
  -not -path '*/__pycache__/*' -not -name dashboard.py -print0 \
  | xargs -0 wc -l | tail -1
```

The audited output was `66169 total`. The independent Dash package contained 4,774
lines and `dashboard.py` 30 lines, both counted separately. Tests contained 8,026 lines
and were likewise excluded from the production core.

## Equivalence-Validation Boundary

Shorter code, similar ASTs, identical configuration hashes, or passing unit tests
cannot individually prove unchanged final outputs. Formal finalcase comparison
requires the same 29 participants, frozen 5×5 splits, GPU/CUDA/PyTorch/dependencies,
and per-fold checks of prediction identities, discrete fields, models/metrics, and
structured scientific products, with `atol=1e-6, rtol=0`. At the time of the original
audit, V5 had not completed this full comparison run.
