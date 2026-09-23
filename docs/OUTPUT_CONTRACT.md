# V6 output contract

Training data, rebuildable presentation, and model reuse have three sibling
output roots. The pipeline generates no plots. The public reporting CLI does
not train or rewrite its input pipeline data.

## Directory hierarchy

```text
final_pipeline_v6/
├── pipeline_output/
│   └── <run>/
│       ├── <comparison>/
│       │   └── repeat_<RR>/fold_<FF>/
│       ├── tables/
│       └── models/
├── report_output/
│   └── <run-or-report-name>/
└── model_config/
    └── <run>/cases/<comparison>/
```

`<comparison>` is the directory identity of a resolved case. Its exact mapping
is recorded in `study_manifest.json`. Repeats/folds use two-digit zero padding,
for example `repeat_00/fold_03`. Single, ablation, grid, and catalog sweeps share
this hierarchy.

## Run naming and recovery

- `--run-name NAME` creates `pipeline_output/NAME`.
- Without a name, `pipeline.py run/ablation/grid` uses the source YAML stem
  (`manual` for direct manual configuration) and UTC time. Study/sweep uses
  the plan YAML stem and UTC time.
- New runs do not silently overwrite an existing same-named directory.
- Resume incomplete or interrupted runs with `--resume pipeline_output/<run>`.
- One `sweep.py` run can contain multiple comparisons; these are not split
  into separate top-level runs.

`pipeline_output` and `model_config` can be committed to Git independently.
Ordinary and Stage5/static/hyper reports can be rebuilt from pipeline outputs,
so committing `report_output` is not required. Analysis-only historical plans,
such as decision-oracle and role-scope, may also explicitly read V2 artifacts
under `--source-root`.

## Authoritative per-fold artifacts

Each successful cell is stored under:

```text
pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
```

Core data include:

| File/category | Meaning |
|---|---|
| `oof_window_predictions.parquet` | Window-level OOF probabilities and identities |
| `oof_file_predictions.parquet` | Recording/file-level OOF probabilities |
| `oof_role_predictions.parquet` | Role-level OOF probabilities |
| `oof_subject_predictions.parquet` | Participant-level OOF probabilities |
| `oof_member_predictions.parquet` | Ensemble-member probabilities; an empty-state artifact when inapplicable |
| `metrics_per_fold_seed.json` | Metrics, seed, and execution summary for the repeat/fold |
| `model_checkpoint/` | Reloadable learned weights, model-input contract, and golden sample |
| Quality/route/feature/training files | Structured audits produced by the selected modules |

Applicable fields at every level retain `case_id`, repeat, fold,
row/record/participant identities, true labels, predicted classes, and class
probabilities. File-level predictions directly support alternative
file→role→participant aggregation or file-level significance analyses.
The originally declared aggregation remains defined by the resolved configuration.

Parquet is the authoritative prediction format. Empty or inapplicable optional
levels have explicit status rather than fabricated records.

## Run-root indexes

Training finalization reads completed cell outputs to build:

```text
study_manifest.json
study_run_result.json
v5_data_manifest.json
tables/v5_fold_predictions.csv
tables/v5_fold_models.csv
tables/v5_config_parameters.csv
tables/pipeline_data.xlsx
pipeline_excel_status.json
models/<case>/median_fold/selection.json
```

The `v5` names remain unchanged for artifact compatibility.

- `study_manifest.json` defines cases, configuration paths, execution scope, and status.
- `v5_fold_predictions.csv` is a lightweight path/shape index of all per-fold Parquet.
- `v5_fold_models.csv` summarizes each fold's model, metrics, provenance, and checkpoint path.
- `v5_config_parameters.csv` flattens resolved configurations into a searchable table.
- `v5_data_manifest.json` summarizes completeness issues, published models, and table paths.

### Pipeline Excel

`tables/pipeline_data.xlsx` is a compact human-readable view, not a new
authoritative data source. It contains three compact indexes and, when size
permits, file, role, participant, and member predictions. Window predictions
are never duplicated into the workbook. Any prediction level exceeding Excel's
row limit is marked skipped in status; per-fold Parquet remains available.

Excel is recoverable postprocessing. Export failure does not delete completed
training data:

```bash
python pipeline.py export-excel \
  --pipeline-output pipeline_output/<run>
```

Add `--replace` only when explicitly rebuilding an existing export.

## Learned weights and refit

Every successful fold saves a reloadable bundle and checks a golden prediction
after saving. Each case selects its median fold after sorting
`(balanced_accuracy, repeat, fold)`; an even number of candidates uses the lower
middle. `models/<case>/median_fold/selection.json` references the authoritative
fold bundle without duplicating weights.

Refit is false by default. Adding `--refit` runs an all-29 refit separately for
each case after outer-fold training:

```text
models/<case>/all29_refit/
v5_refit_manifest.json
```

All-29 refit does not produce internal self-evaluation; performance evidence
remains outer OOF. With refit disabled, all fold weights and the median-fold
published model are still available.

## model_config

Pipeline finalization automatically creates:

```text
model_config/<run>/
├── export_manifest.json
├── available_modules.json
└── cases/<comparison>/
    ├── resolved_pipeline_config.yaml
    ├── pipeline_module_defaults.yaml
    ├── model_reuse_parameters.yaml
    ├── fold_model_parameters.csv
    └── learned_model/
```

Each case exports its resolved configuration, module/parameter defaults,
fold provenance, and one selected bundle. A successful all-29 refit takes
priority; otherwise, the median fold bundle is exported. Not every historical
representation/model supports raw new-participant inference. Capabilities and
reasons are recorded in `export_manifest.json`.

Rebuild independently from a completed run:

```bash
python export_model_config.py --pipeline-output pipeline_output/<run>
```

Existing exports are not overwritten by default; use `--replace` to rebuild.

## report_output

General `analyse_report.py run` reads one or more pipeline runs and produces:

```text
report_output/<name>/
├── analysis_manifest.json
├── outputs_index.json
├── STUDY_SUMMARY.md
├── STUDY_SUMMARY.html
├── figures/*.png or figures/*.NA.txt
└── tables/
    ├── *.csv
    ├── *.json
    └── report_tables.xlsx
```

For a single `pipeline_output/<run>` input, the default report name is the
top-level `<run>`. Thus comparison/ablation within one sweep does not need
another output name. For analysis spanning unrelated runs, use `--output-name`
to specify a new name. The target report directory must be new; existing
reports are not silently overwritten.

Stage5/static-peak/hyperparameter `specialized-report` uses
`report_manifest.json`; hyperparameter reports also contain `phases/`.
They likewise output summaries, figures, CSV/JSON, and workbooks, but do not
fabricate the general report's `analysis_manifest.json`/`outputs_index.json`.

Report Excel and pipeline Excel have different semantics:

- `pipeline_data.xlsx` is a convenience view of training data and prediction indexes.
- `report_tables.xlsx` collects derived tables produced by specified analysis parameters.

General report Excel can also be rebuilt from existing CSV files:

```bash
python analyse_report.py export-excel \
  --report-output report_output/<name>
```

This `export-excel` command requires the general report's
`analysis_manifest.json`. Specialized workbooks are generated together with
`specialized-report`.

## Completeness and numerical equivalence

Manifests, indexes, bundles, and output tables record the necessary paths,
schemas, and hashes. Recovery and rebuilding read completed cells without
redefining splits, models, or aggregation. Passing completeness checks does
not establish numerical equivalence. A formal V2/V5 conclusion requires
25-fold output comparison in the same frozen environment, with
`atol=1e-6, rtol=0` for floating-point values. The inherited V5 documentation
did not establish completion of this full finalcase equivalence run; the V6
translation adds no new numerical-equivalence claim.
