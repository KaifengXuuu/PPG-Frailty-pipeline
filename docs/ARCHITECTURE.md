# V6 architecture

V6 provides one scientific workflow for the CLI, study YAML, and Dash, while
separating expensive training from presentations that can be regenerated.

## Data flow

```text
preset / complete YAML / --manual --module --set
                         │
                         ▼
             resolve + schema validation
                         │
                         ▼
 manifest + frozen split registry
                         │
                         ▼
 signal → windows → quality/artifact → features/representation
                         │
                         ▼
                 model + training
                         │
                         ▼
 per-fold OOF predictions → file/role/participant aggregation → metrics
                         │
                         ├──────────────► optional all-cohort refit
                         │
                         ▼
         pipeline_output + model_config
                         │
                         ├──────────────► analyse_report.py → report_output
                         └──────────────► infer / Dash previews
```

Reporting and Dash do not implement another copy of the scientific algorithms.
Training, inference, and previews connect through shared configuration, module
registries, pipeline adapters, and artifact readers.

## 1. Configuration layer

The compatibility namespace `ppg_frailty.v5.configuration` resolves three input
forms into one complete configuration:

- Named presets, such as `baseline` and `finalcase`.
- Complete configuration YAML.
- `--manual` with repeated `--module`, `--set`, and `--unset`.

A module family selects an implementation and its associated defaults; dotted
paths specify exact parameters. Resolution applies the base configuration,
modules, explicit leaf overrides, unset operations, and shared schema validation,
in that order. `manual-cli` only expands a known configuration into a reviewable
command; actual training still uses the same resolver.

The module and parameter surface is projected dynamically from the registry,
configuration, and study plans:

```bash
python pipeline.py modules --help
python pipeline.py parameters --source-preset all --format markdown
```

## 2. Study and execution layers

`StudyPlan` represents single, grid, ablation, catalog, and legacy-bridge plans.
Expansion resolves a plan into ordered cases. `StudyRunner` executes each case's
declared repeats/folds using the common layout:

```text
pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
```

`pipeline.py run/ablation/grid/run-plan`, `sweep.py run`, and Dash training
all enter the same execution service retained from V5. The runner provides case
concurrency, progress, failure handling, and resume. It does not reorder folds,
recompute frozen splits, or change the call sequence inside a cell.

The request's `environment` records dependency and backend observations at
startup for provenance only. It does not enforce version, driver, or GPU-model
admission checks. `pyproject.toml` declares general dependency ranges;
`requirements/requirements-finalcase.txt` records reproduction reference versions.
Training configuration, not the dependency list, sets the numerical backend.
Concurrency file locks only prevent multiple processes from writing the same run.

## 3. Scientific workflow

The numerical stages are:

1. Read the manifest, labels, and frozen participant-grouped splits.
2. Repair gaps, filter PPG/IMU, perform static calibration, and apply configured resampling.
3. Use the shared window planner.
4. Apply optional SQI, motion, artifact/denoiser processing, and routing.
5. Extract peaks/PPI/PRV/morphology and other feature groups.
6. Build raw, feature-vector, feature-matrix, or fusion representations.
7. Materialize and train fold-local models and save learned bundles.
8. Aggregate window→file→role→participant and evaluate outer OOF predictions.

Parallel alternatives share configuration/registry interfaces. The V5 refactor
shares plumbing, data classes, serialization, and report adapters without changing
mathematical equations, model architectures, sampling rates, splits, or the stage
order above. V6 retains that scientific workflow. Such scientific changes require
a separately defined decision and a distinct comparison/ablation identity.

## 4. Cache and resume

Preprocessing cache stores only deterministic intermediate results independent
of outer-held-out labels. Keys include the relevant data and preprocessing
configuration identity. Fold-local fitted transforms, model training, and OOF
aggregation are not reused across folds. Cache supports `off`, `read_only`,
`read_write`, and namespace selection.

Resume uses the `study_manifest`, case status, and fold artifacts to continue
unfinished work. Completed cells retain their existing data and weights. Run-root
indexes, Excel, and model exports can be rebuilt, so a temporary presentation
export failure does not invalidate training data.

## 5. Data output layer

Completed cells directly write window/file/role/participant/member OOF Parquet,
metrics, training/quality audit data, and model checkpoints. Training does not
import plotting code or write HTML.

Run finalization reads cell artifacts without modifying them and builds three
CSV indexes, `v5_data_manifest.json`, and compact `pipeline_data.xlsx`. Excel
is a convenience copy; per-fold Parquet remains authoritative.

Each case publishes the median fold bundle after OOF-metric ordering by default.
`--refit` is false by default. When enabled, the existing all-cohort refit runs
for every case after outer cells. Both bundle types then enter
`model_config/<run>`.

## 6. Reporting layer

`analyse_report.py` supports `single`, `comparison`, `ablation`, and `test`.
Its declarative registry combines audit, prediction, summary, ROC/AUC, confusion,
calibration, per-class, learning, coverage, hierarchy, quality, comparison,
ablation, ensemble, operations, and historical modules.

Reporting reads pipeline artifacts without modifying them, computes requested
derived statistics and figures, and writes a new `report_output/<name>`.
The same pipeline run can be analyzed repeatedly with different statistical
seeds, significance units, or modules without affecting training results.

## 7. model_config and inference

The automatic or standalone exporter saves a resolved configuration, module
and parameter defaults, fold tables, and one selected learned bundle for each
case. The raw-inference service loads this bundle and replays its supported
preprocessing, windowing, model, and aggregation path without fitting.

Finalcase dynamic R/S/W recordings require static B calibration from the same
participant. Silent calibration without B remains a separate V5-origin TODO.
Inference does not silently replace the training contract.

## 8. Dash

Dash is the local control panel for these services:

- Configure exposes the same module and parameter catalog.
- Workflow reads manifests or completed artifacts to preview stages.
- Run constructs pipeline/sweep requests, with Train, Stop, Infer, and a comparison queue.
- Analyse constructs report requests and previews tables/figures.
- Tools exposes validation, indexes, Excel/model export, and specialized routes.

The interface displays and downloads equivalent CLI and resolved YAML.
Training requires selecting YAML first. Stop terminates the current background
training process group. Infer loads only available learned bundles from model_config.

## 9. Numerical equivalence boundary

Structural reuse and unit tests do not replace end-to-end numerical verification.
The V2/V5 equivalence protocol requires identical inputs, splits, GPU, CUDA,
PyTorch, and dependencies. It compares discrete fields and row identities over
25 finalcase outer cells and uses `atol=1e-6, rtol=0` for floating-point values.
The inherited V5 documentation did not establish completion of that full
25-fold equivalence run. V6 translation itself provides no new benchmark evidence.
