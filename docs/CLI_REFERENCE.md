# V6 CLI reference

This document describes the current public CLI. Run all examples from the
`final_pipeline_v6/` root. Training writes only `pipeline_output` and
`model_config`; `analyse_report.py` writes figures, presentation tables, and
report Excel to `report_output`.

## Before running

See the [README](../README.md#installation-and-environment-reproduction) for
reference versions and installation steps. Exact dependencies are collected in
[requirements-finalcase.txt](../requirements/requirements-finalcase.txt).
The CLI records the actual environment; it does not reject execution based on
dependency versions, GPU models, or driver versions.

```bash
python pipeline.py validate --preset finalcase --mode full
```

The trainer configures numerical backends through
`training.deterministic_algorithms`. When deterministic CUDA is selected, a
missing `CUBLAS_WORKSPACE_CONFIG=:4096:8` is set before device allocation;
no manual export is required. Formal V2/V5 comparisons use the same data,
splits, configuration, and reference environment, with `atol=1e-6, rtol=0`.
CPU or other compatible dependency versions can be used, but numerical outputs
should be revalidated.

Finalcase defaults to `cache/preprocessing`; the historical
`artifacts/studies/cache` location is also supported. Preprocessing cache stays
inside the V6 root and does not share training-fitted state across folds.

## Command map

| Entry point | Subcommand | Purpose |
|---|---|---|
| `pipeline.py` | `run` | Execute one complete configuration |
| | `ablation` | Compare multiple values of one dotted path |
| | `grid` | Cartesian comparison across multiple dotted paths |
| | `run-plan` | Execute study-plan YAML |
| | `validate`, `show-config` | Validate or display resolved configuration |
| | `modules`, `presets`, `parameters` | Inspect the live configuration catalog |
| | `manual-cli` | Generate one complete direct-parameter CLI command without training |
| | `infer` | Load a trained bundle for no-fit inference |
| | `index`, `export-excel`, `export-model-config` | Rebuild data indexes and exports |
| `sweep.py` | `validate`, `run` | Validate/execute preconfigured study YAML |
| | `export-excel` | Rebuild pipeline Excel |
| `analyse_report.py` | `list`, `validate`, `run` | List the catalog, validate read-only, and generate reports |
| | `export-excel` | Rebuild report Excel from existing general-report CSV |
| | `execution-audit` | Read-only audit of a failed/interrupted run, without reading predictions or weights |
| | `specialized-*` | Adapters for historical specialized analysis/reporting |
| `export_model_config.py` | — | Independently export model configuration from a completed run |
| `dashboard.py` | — | Start the local visual control panel |

The parser output is authoritative:

```bash
python pipeline.py --help
python pipeline.py run --help
python sweep.py run --help
python analyse_report.py run --help
```

## Modules and all parameters

Do not manually copy a static parameter table from documentation. These commands
generate current module IDs, the union of 485 leaf paths across presets/studies/
modules, types, defaults, ranges, and YAML input forms:

```bash
python pipeline.py modules --help
python pipeline.py modules --family model
python pipeline.py parameters --source-preset all --format markdown
```

To inspect only finalcase's 254 configuration leaves:

```bash
python pipeline.py parameters --source-preset finalcase --format yaml
python pipeline.py parameters --help
```

Common input forms:

```text
--module FAMILY=MODULE_ID
--set PATH=YAML_VALUE
--unset PATH
--config-id SAFE_ID
```

`--module` applies a module and its associated defaults, then `--set` overrides
individual leaves. When editing an expanded command, remove or update old
`--set` values that conflict with a newly selected module. Arrays, mappings,
strings, and `null` are parsed as YAML and usually need single quotes in a shell.

## Configuration sources

`run`, `ablation`, and `grid` require exactly one source:

```text
--preset NAME
--config path/to/complete.yaml
--manual
```

All three pass through the same schema validator and enter the same runner
with a complete resolved configuration. `--manual` also requires an explicit
`--config-id` and accepts repeated `--module`/`--set`/`--unset`.

### Finalcase using direct CLI parameters

`manual-cli` is only a command generator: it reads `finalcase` once, expands
all leaves into a shell-safe command, and exits without training. After you
review and execute the generated file, the actual training process receives
only `--manual` and explicit parameters, not a preset/config/plan selection.

```bash
python pipeline.py manual-cli \
  --source-preset finalcase \
  --run-name finalcase_cli_01 > /tmp/finalcase_cli.sh
less /tmp/finalcase_cli.sh
bash /tmp/finalcase_cli.sh
```

Run directories are not overwritten. If a failed `finalcase_cli_01` already
exists, regenerate the command with a new name, such as `finalcase_cli_02`,
and retain the old incomplete directory as audit evidence. A
`model_config/finalcase_cli_01` export created on failure contains configuration
evidence only, with no learned weights; it is not a valid model input for Dash
or `pipeline.py infer`.

The generated file includes all `--set`/`--unset` values and explicitly carries
the finalcase study YAML's 5×5 execution, CUDA, `--no-continue-on-error`,
cache root/namespaces, output, study ID, and comparison case ID.
Training no longer reads configuration or plan YAML. Refit is off by default,
so no inverse switch is needed. To review or adapt the command by module name,
you can add, for example:

```text
--module representation=raw
--module model=InceptionTimeSmall
--module imu_gravity=sensor_filter_only_no_gravity_removal
--module optimizer=adamw
--set training.batch_size=16
--set training.learning_rate=0.0003
```

These are structural excerpts from a fully expanded command, not standalone
finalcase commands. When inserting a module, reconcile subsequent explicit
`--set` values for the same paths and inspect the final values with
`show-config --manual ...` or `validate --manual ...` first.

### Preset/config entry points

```bash
python pipeline.py show-config --preset finalcase

python pipeline.py run \
  --config configs/presets/finalcase.yaml \
  --run-name finalcase_config_01 \
  --repeats all --folds all --jobs 1 --device cuda \
  --preprocessing-cache-mode read_write
```

You can append `--module`, `--set`, and `--unset` to preset/config inputs
to define explicit derived configurations. The resolved configuration is
saved in the run.

## Execution parameters

For full definitions, use `python pipeline.py run --help`. Core forms:

| Parameter | Input | Meaning |
|---|---|---|
| `--repeats` | `all` or `0,1,...,4` | Subset of outer repeats |
| `--folds` | `all` or `0,1,...,4` | Subset of outer folds |
| `--jobs` | Positive integer | Concurrent cases |
| `--device` | `cuda` / `cpu` | Also resolves the training device |
| `--continue-on-error` | BooleanOptionalAction | Continue after a case fails |
| `--measure-operational-costs` | BooleanOptionalAction | Collect non-scientific execution costs |
| `--preprocessing-cache-mode` | `off/read_only/read_write` | Leakage-safe preprocessing cache |
| `--preprocessing-cache-root` | Path | Cache root |
| `--preprocessing-cache-namespaces` | Comma-separated list | Cache namespaces |
| `--output-root` | Path | Defaults to `pipeline_output` |
| `--run-name` | Safe single-directory name | New run name; generated when omitted |
| `--case-id` | Safe single-directory name | Single `run` only; names its comparison directory |
| `--resume` | Existing run path | Resume an existing run |
| `--hash-predictions` | Flag | Index prediction-file hashes |
| `--dry-run` | Flag | Expand/check without training or writing run artifacts |
| `--refit` | Flag | Disabled by default; explicitly enables final all-cohort refit |

New runs do not overwrite same-named directories. Resume with
`--resume pipeline_output/<run>`. Without `--run-name`, config CLI and
study/sweep use the source YAML stem plus UTC time; direct `--manual` uses
`manual` plus UTC time. A readable explicit run name is recommended for
formal experiments.

## Ablation and grid

Single-factor ablation:

```bash
python pipeline.py ablation \
  --preset finalcase \
  --study-id gravity_ablation \
  --factor signal.imu.gravity_method \
  --values sensor_filter_only_no_gravity_removal profile_a_lowpass_0p3hz \
  --reference-value sensor_filter_only_no_gravity_removal \
  --run-name gravity_ablation_01 \
  --repeats all --folds all --jobs 1 --device cuda
```

Cartesian grid: the right-hand side of each `--vary` must be a YAML list
with at least two values.

```bash
python pipeline.py grid \
  --preset finalcase \
  --study-id optimizer_grid \
  --vary 'training.learning_rate=[0.0001,0.0003]' \
  --vary 'training.weight_decay=[0.0001,0.001]' \
  --reference training.learning_rate=0.0003 \
  --reference training.weight_decay=0.001 \
  --run-name optimizer_grid_01 \
  --repeats all --folds all --jobs 1 --device cuda
```

`--module` selects an implementation. `--factor`/`--vary` define experimental
axes that expand into comparison cases.

## Finalcase study YAML

Use the preconfigured plan for the complete formal 5×5 run:

```bash
python sweep.py validate \
  --plan configs/studies/finalcase.yaml

python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v6_01
```

`pipeline.py run-plan --plan ...` uses the same execution service;
`sweep.py` is the compact study entry point. All plan cases share the same
`<run>/<comparison>/repeat/fold` tree.

### Refit

Refit is disabled by default. Enable it with one switch:

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v6_refit_01 \
  --refit
```

After outer cells complete, every case in the run receives one all-29 refit.
Without refit, each fold's weights are still saved, and each case exports the
median fold bundle after OOF-metric ordering. All-29 bundles do not perform
internal self-evaluation; reported performance remains outer OOF.

## Resume, indexing, and exports

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --resume pipeline_output/finalcase_v6_01

python pipeline.py index \
  --study-dir pipeline_output/finalcase_v6_01 \
  --hash-predictions

python pipeline.py export-excel \
  --pipeline-output pipeline_output/finalcase_v6_01

python export_model_config.py \
  --pipeline-output pipeline_output/finalcase_v6_01
```

Exports do not overwrite existing targets by default. Add `--replace` only
when rebuilding is intended. Normal pipeline finalization automatically
creates indexes, Excel, and `model_config/<run>`.

## analyse_report

List available modes, 7 presets, 18 modules, 36 figures, and 74 tables in the
general-report catalog:

```bash
python analyse_report.py list
```

A sweep run already contains all its comparison cases. Pass the top-level
run as input; the report inherits its name by default. Replace
`<REFERENCE_CASE>` below with an actual reference case ID from that run:

```bash
python analyse_report.py validate \
  --mode comparison \
  --input pipeline_output/gravity_ablation_01 \
  --reference-case <REFERENCE_CASE> \
  --preset comparison

python analyse_report.py run \
  --mode comparison \
  --input pipeline_output/gravity_ablation_01 \
  --reference-case <REFERENCE_CASE> \
  --preset comparison
```

Output is `report_output/gravity_ablation_01/`. Ablation additionally takes one
or more `--factor-path` arguments. Test mode accepts only explicit independent
test evidence present in the inputs. Internal outer OOF cannot be relabeled
as an independent test.

For cross-run comparison, repeat named inputs:

```bash
python analyse_report.py run \
  --mode comparison \
  --run old=pipeline_output/run_old \
  --run new=pipeline_output/run_new \
  --reference-case <REFERENCE_CASE> \
  --preset comparison \
  --output-name old_vs_new
```

General `validate/run` selection rules:

- Repeat `--include-case` / `--exclude-case` as needed.
- Use `--preset classification|comparison|ablation|test|minimal|ensemble|full`.
- Repeat `--module` to combine analysis modules.
- Explicit `--figure` / `--table` replaces the default set; `none` selects an empty set.
- `--bootstrap-resamples`, `--permutation-resamples`, `--statistics-seed`,
  `--alpha`, and `--calibration-bins` control report statistics.
- `--on-missing error|na|skip` selects missing-input behavior.

Rebuild general report Excel from existing CSV tables:

```bash
python analyse_report.py export-excel \
  --report-output report_output/gravity_ablation_01
```

This command requires the general report's `analysis_manifest.json`.
Stage5/static-peak/hyperparameter `specialized-report` generates workbooks
from each registered complete figure/table suite. Decision-oracle and
role-scope use `specialized-run`. See
[PLAN_COMPATIBILITY.md](PLAN_COMPATIBILITY.md) for routing details.

### Execution audit for failed/interrupted runs

Failed runs do not meet the ordinary report data contract and cannot support
performance, ranking, significance, or model-selection conclusions.
Use the separate execution-only entry point to inspect scope and failure causes:

```bash
python analyse_report.py execution-audit \
  --input pipeline_output/finalcase_cli_01 \
  --output-name finalcase_cli_01_failure
```

It reads only the plan, manifest, run result, and progress metadata, not OOF
predictions or model weights, and does not modify input. Output under
`report_output` includes CSV, JSON, Excel, Markdown, HTML, and complete input
hashes, but no figures. Successful completed runs are rejected; use ordinary
`analyse_report.py validate/run` for those. Dash Tools exposes the same
parser-backed command.

## No-fit inference

```bash
python pipeline.py infer \
  --model-config model_config/finalcase_v6_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --input-manifest path/to/participant.yaml
```

The input manifest describes one or more recordings from the same participant.
Dynamic R/S/W input currently requires a static B calibration record.
Silent calibration without B remains an unimplemented V5-origin TODO.
Inference only loads the bundle and does not update weights.

## Dash

```bash
python dashboard.py --host 127.0.0.1 --port 8050
```

Dash listens only on loopback. Training requires YAML selection first.
Train and Stop are separate; Infer loads exported bundles. Configure, Workflow,
Run, Analyse, and Tools cover modules/parameters, the comparison queue,
stage previews, pipeline/report previews, and equivalent CLI/resolved YAML downloads.
