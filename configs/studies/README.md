# V6 study YAML entry points

This directory retains all reusable V2 study YAML and adds the thesis final
configuration, `finalcase.yaml`. Run from the V6 repository root;
do not invoke V2 top-level scripts.

Entry points for previously run thesis comparisons/ablations are indexed in
[`thesis/README.md`](thesis/README.md). The `thesis/` directory stores YAML
with historical parameters by experiment; directly reusable tests link to their
existing files. General templates evolve with default configurations. For
historical reruns, follow the indexed configuration and execution instructions
rather than choosing templates solely by old V2 filenames.

## Canonical study plans

For `schema_version: ppg_frailty.study_plan.v2`:

```bash
python sweep.py validate --plan configs/studies/PLAN.yaml
python sweep.py run --plan configs/studies/PLAN.yaml
```

Alternatively, use `python pipeline.py run-plan --plan ...`. Both invoke the
same data-only service inherited from V5. Plotting/report flags from the original
plan are disabled; training parameters, case expansion, splits, and seeds retain
their original definitions.

Thesis finalcase:

```bash
python sweep.py validate \
  --plan configs/studies/finalcase.yaml

python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v6_01
```

## Specialized study plans

Noncanonical schemas are not silently interpreted as generic grids:

- Decision-oracle and role-scope: `analyse_report.py specialized-*`.
- Stage5/static-peak/hyperparameter: `specialized_pipeline.py validate|run|complete`.
- Specialized reporting after computation: `analyse_report.py specialized-report`.

Fresh computation needs raw datasets and the files already bundled with V6,
not a V2 checkout. `assets/authority/` contains the original M2 manifests/splits;
the frozen motion evidence, all-29 weights, and five matching fold weights are
also included. Put `PPG_Testing_05_01_2026/` and `physionet.org/` at the project
root, or set `PPG_FRAILTY_DATA_ROOT` to their parent directory. Both datasets
are excluded from Git.

Motion plans disable `motion_model_comparison`, which only packages historical
result comparisons. Internal motion training, PTT evaluation, PTT training, and
Frailty29 reverse evaluation still run. The old-source Phase 0 audit is disabled
in supplied bridge plans without changing their training workflow.

Decision-oracle and role-scope YAMLs are post-hoc templates. Their dated source
paths are provenance, not bundled result directories: replace source/input and
ranking fields with compatible, completed V6 outputs before execution. The
oracle CLI also accepts `--study-dir` and `--case-id` overrides. Validation does
not supply missing inputs. No old prediction or report archive is copied.

Specialized computation writes only data and pipeline Excel under
`pipeline_output`; plans that train models automatically export learned weights
and `model_config`. Figures are generated separately by `analyse_report.py`.

See [`docs/PLAN_COMPATIBILITY.md`](../../docs/PLAN_COMPATIBILITY.md) for the
complete schema/entry-point mapping and limitations.
