# PPG Frailty Pipeline V6

Frailty is a state of cumulative decline across multiple physiological systems,
characterized by reduced physiological reserves, diminished recovery of
homeostasis, and increased vulnerability to stressors. Even a minor stressor can
cause substantial health changes. Frailty can improve or worsen over time, making
low-cost, repeatable measurements useful for studying frailty and prefrailty.

Photoplethysmography (PPG) noninvasively records pulsatile changes in peripheral
blood volume using optical sensors. Pulse timing and shape provide information
related to cardiovascular function and autonomic regulation. An inertial
measurement unit (IMU) describes translational and rotational motion using
three-axis acceleration and three-axis angular velocity. This project uses eight
synchronously recorded channels: red PPG, infrared PPG, and six-axis IMU. These
signals connect physiological changes with concurrent movement and use motion
information to help identify and suppress PPG motion artifacts.

Traditional frailty assessment relies mainly on questionnaires, physical
performance tests, and clinical judgment, usually performed at discrete visits.
Wearable PPG and IMU offer a complementary route to repeated measurement, but
sensor contact, motion interference, and repeated recordings from the same
participant affect signal interpretation and classification evaluation. This
project therefore provides a transparent, modular Python workflow to investigate
whether synchronized signals support reproducible participant-level group
discrimination and to establish a foundation for future remote-monitoring
research. The background and three objectives follow the Introduction and
Conclusion of the project's thesis draft, `Kaifeng_Masterarbeit_draft_v1_0.docx`
(maintained separately; not included in this distribution).

## Project objectives

1. Build a transparent, reproducible workflow for synchronized PPG/IMU processing,
   covering preprocessing, pulse-peak detection, motion detection, and motion
   artifact suppression.
2. Explore physiological and motion-related information in the recordings through
   PPG-derived features and participant-level classification, comparing prefrail
   older adults, nonfrail older adults, and a young reference group.
3. Evaluate the reliability of the complete workflow through participant-independent
   splits, comparison with existing reference implementations, and assessment of
   different models and signal-processing configurations.

This project is an engineering-methods study with exploratory classification.
Young is an age reference group, not a frailty grade. Three-group classification
alone does not establish frailty-specific biomarkers or clinical diagnostic
validity.

## Data and functionality

The thesis dataset contains 29 participants: 9 Pre-Frail, 12 Robust/Non-Frail,
and 8 Young. Each provides 9 synchronized recordings, giving 261 recordings:
static baseline B, post-exercise recovery R1–R4, and movement tasks S1/S2/W1/W2.
The manifest stores raw-data paths, participants, labels, and recording roles.
Raw data are stored separately and require the appropriate access permissions.

Training and evaluation follow this workflow. YAML, direct CLI parameters, and
Dash configure the same algorithms:

```text
manifest / labels / participant-grouped splits
    → PPG and IMU preprocessing, calibration, and resampling
    → windowing, quality assessment, optional motion detection and artifact suppression
    → feature extraction or raw-signal representation
    → within-fold training and held-out predictions
    → window → recording → role → participant probability aggregation
    → per-fold data and weights → pipeline_output
         ├─ fold model / optional all-cohort refit → model_config → new-participant inference
         └─ statistical analysis and visualization → report_output
```

| Workflow stage | Modules and functionality | Main entry points or implementation |
|---|---|---|
| Data and experiment plans | Read manifests, labels, and participant-grouped splits; single, comparison, ablation, grid, and repeated cross-validation | `pipeline.py`, `sweep.py`; `data/`, `study/` |
| Signal preprocessing | Missing-segment handling, PPG/IMU filtering, static calibration, resampling, windowing, normalization, and reusable deterministic preprocessing cache | `signal/`, `data/windows.py`, `data/preprocessing_cache.py` |
| Quality and motion processing | Optional SQI, window selection, motion detection, artifact reducers, and denoisers | `quality/`, `artifacts/` |
| Features and representations | Peaks/PPI/PRV, pulse morphology, and other features; raw, feature-vector, feature-matrix, and fusion representations | `features/`, `representations/` |
| Models and evaluation | Classical models, deep models, ensembles, within-fold training, OOF prediction, hierarchical aggregation, and metrics | `models/`, `training/` |
| Data and model output | Per-fold predictions and learned weights, data Excel, model selection, optional refit, and model-reuse parameters | `pipeline.py`, `export_model_config.py` |
| Analysis and reporting | ROC/AUC, confusion matrices, learning curves, calibration, box plots, paired tests, and specialized comparisons | `analyse_report.py`; `reporting/`, `v5_reporting/` |
| Interactive operation | Configuration, training and stopping, pretrained inference, comparison queue, stage-by-stage previews, and visualization | `dashboard.py` |

Module directories in the table are under `src/ppg_frailty/`. Training-dependent
fitting uses only the current training participants. The preprocessing cache
does not share training-fitted state between folds. Predictions are saved per
fold so the same results can support different aggregation methods, analysis
units, and plots without retraining.

This V6 distribution omits generated caches, `pipeline_output`,
`report_output`, and other large generated result bundles. Small reusable
`model_config` bundles and the motion-model artifacts required by the included
plans are retained. It does not include raw datasets: configure their external
paths before running. Internal `v5` package and schema identifiers are retained
for compatibility with existing artifacts.

## Installation and environment reproduction

Run installation and CLI commands from the directory containing this README,
`pyproject.toml`, and `pipeline.py`. The contents of this V6 directory can be
used directly as a repository root. In the original multi-version checkout only,
first enter its distribution directory:

```bash
cd final_v0/final_pipeline_v6
```

No `cd final_v0/...` is needed after publishing V6 as the repository root.
Packaged configurations, manifests, splits, and outputs are located relative
to this README, independent of the checkout depth.

### Standalone data layout

By default, recordings are read from this repository root. The original
`final_v0/final_pipeline_v6` layout automatically uses its enclosing project
root instead. To keep private datasets outside the code checkout, set:

```bash
export PPG_FRAILTY_DATA_ROOT=/absolute/path/to/private-project-data
```

This directory must contain `PPG_Testing_05_01_2026/`, not be that subdirectory
itself. The same setting applies to the CLI and Dash. It does not move packaged
resources, `cache`, `pipeline_output`, `report_output`, or `model_config`.

| Use case | Required inputs |
|---|---|
| Original finalcase training | The 261 manifest-listed recordings in `PPG_Testing_05_01_2026/StudyData/` and `PPG_Testing_05_01_2026/TestDataYoungers/`; V6 already includes their materialized manifest and folds |
| PTT experiments | The 66 manifest-listed CSVs under `physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/` and `s1_sit.hea` for motion-unit provenance |
| M2 materialization and specialized motion workflows | Byte-identical authority CSVs and split JSON included in `assets/authority/`; no old M2 directory is needed |
| Motion-model reuse | Included all-participant and five fold-specific weights, evidence, and packaged splits; archived absolute paths are not used to load these weights |
| Regression tests | Small frozen expected configurations in `tests/fixtures/`; no old experiment-result archive is needed |

Only the two raw database directories need to be supplied separately. Both are
ignored by Git, including when linked from outside the checkout. Old root
scripts, notebooks, `AA_TODO`, and `final_v0` are not prerequisites. The retired
PTT script's hash remains provenance metadata, but the script is no longer read;
unit conversions and source CSV/header verification are unchanged.

The Stage5 plans run all four scientific motion training/evaluation stages and
disable only packaging a comparison against an archived run
(`motion_model_comparison.enabled: false`). The old Phase-0 archive audit is
also disabled by default. Scientific comparison/ablation plans remain available
under `configs/studies/thesis/`. Post-hoc oracle/role-scope templates require
predictions from a completed run: point their input fields at your V6 outputs
before using them; no historical predictions are bundled.

For cloning into a clean folder, publishing a new branch, and merging it into
`main`, see [Root deployment and Git publishing](docs/ROOT_DEPLOYMENT.md).

Before public release, review the participant identifiers/labels in manifests
and the verification samples in `model_config/**/learned_model/golden.npz`;
excluding raw CSVs alone does not make all bundled artifacts anonymous.

A normal installation uses the dependency ranges in
[pyproject.toml](pyproject.toml) and requires Python 3.11 or later. Install the
deep, reporting, and dashboard extras as needed:

```bash
conda create -n ppg-v6 python=3.11
conda activate ppg-v6
python -m pip install -e '.[deep,reporting,dashboard,test,legacy-artifacts]'
python -m pip check
```

The reference versions used for pipeline-module tests and numerical model-output
reproduction are listed in
[requirements/requirements-finalcase.txt](requirements/requirements-finalcase.txt).
The reference environment is Python 3.11.14, NVIDIA GeForce RTX 4080 (driver
560.81), CUDA 12.6, and PyTorch 2.9.1+cu126. To reproduce that environment, create
a separate environment and install its exact versions:

```bash
conda create -n ppg-v6-finalcase python=3.11.14
conda activate ppg-v6-finalcase
python -m pip install -r requirements/requirements-finalcase.txt
python -m pip install --no-deps -e .
python -m pip check
```

The requirements file includes the CUDA 12.6 PyTorch wheel source. The GPU and
driver must be prepared separately on the host; pip-installed CUDA dependencies
do not replace the NVIDIA driver.

These exact versions are the testing and result-reproduction reference. Runtime
does not require each software version or GPU model to match the list.
Dependencies can be upgraded by editing the installation requirements, followed
by module tests and the required numerical output comparisons. Training commands
accept `--device cpu` (sweep devices are configured in YAML); device and numerical
library changes can affect floating-point results. Deterministic CUDA training
automatically sets a missing `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Explicitly selected
backend settings are not overwritten by the environment requirements.

Test dependencies and entry point:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## Configuring modules and parameters

Each module has default parameters and can be selected through YAML, CLI, or
Dash. The CLI accepts repeated `--module FAMILY=MODULE_ID`,
`--set PATH=YAML_VALUE`, and `--unset PATH`. Module defaults are applied first;
explicit leaf parameters override them afterward. Booleans, lists, numbers,
and strings use YAML syntax. Quote values containing spaces or lists in the shell.

The complete module names, adjustable parameters, types, ranges, and defaults
are generated from the code registry:

```bash
python pipeline.py modules --help
python pipeline.py modules
python pipeline.py parameters --help
python pipeline.py parameters --source-preset all --format markdown
python pipeline.py run --help
python sweep.py --help
python analyse_report.py --help
```

## Running finalcase

The thesis final configuration is available as the optional `finalcase` preset,
corresponding to Rank 2 `tuned_all_roles_small_no_gravity`, with case ID
`tuned_all_roles__inception_small_no_gravity`. It uses all B/R/S/W roles,
64 Hz eight-channel raw input, 5-second windows, InceptionTimeSmall, and a fixed
10 epochs. Five repeats × five participant-grouped folds produce OOF predictions.
Other experiments can select their own modules and parameters.

### Direct CLI parameters

`pipeline.py run --manual` accepts complete module and parameter definitions
without reading a preset YAML during training. Because finalcase has many
parameters, first expand it into a shell command, then review, edit, and execute it:

```bash
python pipeline.py manual-cli \
  --source-preset finalcase \
  --run-name finalcase_cli_01 > /tmp/finalcase_cli.sh
less /tmp/finalcase_cli.sh
bash /tmp/finalcase_cli.sh
```

`manual-cli` only generates the command; `--source-preset` is used at generation
time. The generated command includes `--set`/`--unset` for every leaf value,
repeat/fold, cache, and output parameters, and can run independently of the
original YAML. You can also write a complete `--manual` command directly.
The following is only an editable parameter excerpt:

```text
--module representation=raw
--module model=InceptionTimeSmall
--module imu_gravity=sensor_filter_only_no_gravity_removal
--set signal.dl_resampling.target_fs_hz=64.0
--set training.batch_size=16
--set training.learning_rate=0.0003
--set 'training.classifier_role_families=[B,R,S,W]'
```

### Prepared YAML

[configs/studies/finalcase.yaml](configs/studies/finalcase.yaml) contains the
complete finalcase study plan. Validate it first, then run:

```bash
python sweep.py validate --plan configs/studies/finalcase.yaml
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v6_01
```

Comparisons and ablations define multiple cases in one study and use the same
repeat/fold execution workflow. See the
[thesis experiment YAML index](configs/studies/thesis/README.md) for experiment
groups, historical parameter mappings, and commands. General templates and
historical thesis configurations are stored separately; identical filenames
alone do not establish identical experiments. More plans and commands are in
the [CLI reference](docs/CLI_REFERENCE.md) and
[plan compatibility guide](docs/PLAN_COMPATIBILITY.md).

`refit` is off by default. Add `--refit` to produce full-cohort weights; it runs
a refit for every case after outer-fold training:

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_refit_01 \
  --refit
```

With refit disabled, all fold weights are still saved. The median fold model
after sorting OOF `(balanced_accuracy, repeat, fold)` is published. Refit weights
are for model reuse; performance evaluation still uses held-out OOF predictions.

## Preprocessing cache controls

The cache reuses label-independent preprocessing results that have not been
fitted within a fold, reducing repeated filtering, calibration, and windowing.
It is not required for correct predictions. Disabling it recomputes results
and does not delete existing cache files. The finalcase study YAML and commands
generated by `manual-cli` use `read_write`, with `cache/preprocessing` under V6.
For direct `pipeline.py run` without cache arguments, execution defaults to
`off` with directory `artifacts/studies/cache`. Cache settings are not part
of the pipeline preset itself.

| Mode | Behavior |
|---|---|
| `off` | Do not read or write preprocessing cache; recompute each time. |
| `read_only` | Read cache hits; compute misses without adding entries. |
| `read_write` | Read cache hits; compute and save misses. |

Select the mode through the CLI when running the pipeline directly.
A complete example with cache disabled:

```bash
python pipeline.py run --preset finalcase \
  --run-name finalcase_no_cache_01 \
  --preprocessing-cache-mode off \
  --preprocessing-cache-root cache/preprocessing
```

Replace `off` with `read_only` or `read_write` to change the mode.
`--manual` accepts the same arguments. The generated `manual-cli` command
already contains them; edit their values directly.
`--preprocessing-cache-root cache/preprocessing` selects a cache directory
inside V6. Select cache layers with
`--preprocessing-cache-namespaces imu_calibration,canonical_signal_views,raw_windows`.
Another available layer is `motion_windows`, used only when its motion branch
runs. Omitting a cache layer does not disable its algorithm; that stage is
simply recomputed.

For `sweep.py`, edit `preprocessing_cache` under the existing `execution`
section of the selected study YAML, retain the other execution settings, and
run the plan normally:

```yaml
execution:
  # Retain the existing repeats, folds, device, and other fields.
  preprocessing_cache:
    mode: off  # Or read_only or read_write.
    root: cache/preprocessing
    namespaces: [imu_calibration, canonical_signal_views, raw_windows]
    verify_source_sha256: true
```

In Dash, the model module's Train mode has a cache dropdown for ordinary
training; sweeps use the cache settings in the selected study YAML. Reporting
from existing pipeline results and inference with exported models do not require
these preprocessing caches. Before clearing disk space, ensure no training task
is using the cache directory. Disabling the switch alone does not free the
space occupied by existing files.

## Output structure

Three output roots sit alongside the README:

```text
final_pipeline_v6/
├── pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
├── report_output/<run-or-report-name>/
└── model_config/<run>/cases/<comparison>/
```

| Directory | Contents |
|---|---|
| `pipeline_output` | Per-fold window, recording/file, role, participant, and applicable ensemble-member predictions; metrics, learned weights, CSV indexes, and `tables/pipeline_data.xlsx` |
| `report_output` | Analysis figures, derived CSV/JSON statistical tables, HTML/Markdown, and report Excel |
| `model_config` | Per-case resolved configuration, module switches and defaults, model-reuse parameters, and selected learned bundle |

Parquet stores the authoritative predictions. Pipeline Excel offers a convenient
view of data and indexes; report Excel contains the derived statistics from the
selected analysis. The pipeline generates no plots or HTML. Existing training
results can be analyzed repeatedly in independent report runs. See the
[output contract](docs/OUTPUT_CONTRACT.md) for directories and fields.

Use `--run-name NAME` to name a run. If omitted, a name is generated from the
configuration or plan name and UTC time. Resume an existing run with
`--resume pipeline_output/<run>`. Reports use the top-level run name by default;
comparisons in one sweep share that report directory. Use `--output-name` to
give a separate analysis a new directory name.

## Analysis and reporting

`analyse_report.py` generates reports from pipeline artifacts and supports
`single`, `comparison`, `ablation`, and `test`. First inspect the composable
presets, modules, figures, and tables:

```bash
python analyse_report.py list
python analyse_report.py run --help
```

Generate a classification report for a run:

```bash
python analyse_report.py validate \
  --mode single --input pipeline_output/finalcase_v6_01 --preset classification
python analyse_report.py run \
  --mode single --input pipeline_output/finalcase_v6_01 --preset classification
```

Comparison/ablation can select cases within one run. For multiple runs, repeat
`--run NAME=PATH`. Explicit `--figure` or `--table` replaces the preset's
corresponding default selection; `none` disables that artifact type.
Stage5/static-peak/hyperparameter reports use `specialized-report`;
decision-oracle/role-scope analyses use `specialized-run`.

For interrupted or failed runs,
`execution-audit --input pipeline_output/<run>` generates tables, Excel, and
HTML/Markdown covering execution completeness and failure events. This helps
inspect completed stages and failure causes. Outputs are under `report_output`,
separate from model-performance reports.

## Model reuse and inference

The pipeline automatically exports `model_config/<run>`. It can also be
exported independently:

```bash
python export_model_config.py --pipeline-output pipeline_output/finalcase_v6_01
python pipeline.py infer \
  --model-config model_config/finalcase_v6_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --input-manifest path/to/participant.yaml
```

Inference loads the model and its preprocessing parameters without retraining.
The input manifest provides one static recording, or multiple static and dynamic
recordings, for a participant. Available inference capabilities are recorded in
the exported `export_manifest.json`. Dynamic R/S/W recordings currently use a
static B recording from the same participant for calibration. Silent-calibration
ablation without a static B record remains an unimplemented V5-origin TODO.

## Dash control panel

```bash
conda activate ppg-v6
# Run from the repository root containing dashboard.py:
python dashboard.py --host 127.0.0.1 --port 8050
```

Open `http://127.0.0.1:8050`. The page follows the workflow vertically:
input → PPG preprocessing → IMU preprocessing → motion detector → SQI →
denoiser → feature engineering → representation → classifier → aggregation →
report. Each stage has algorithm selection, switches, sliders, and exact numeric
inputs on the left, with adjacent time/frequency plots and window/feature/prediction
tables on the right. Algorithms come from the production modules; the UI does
not implement separate mathematics.

PPG preprocessing separates raw RED/IR and filtered RED/IR into two time plots
with independent vertical axes. Gap-repaired native traces remain in the raw
plot legend and can be shown by clicking. No additional normalization or
vertical shifts are applied. Time- and frequency-domain legends sit above the
plot area with room to wrap, avoiding overlap with the horizontal axis.
Raw and processed IMU traces are paired by physical unit.

Frequency plots use the full sampling-rate recording and do not change with
the time-preview start or duration. Panels overlay raw/processed PSD with the
same unit. PSD panels are collapsed by default; expand
`Frequency domain / PSD`. Frequency is in Hz, including 0 Hz; positive values
use a logarithmic vertical axis, without rewriting zeros. An all-zero panel
uses a linear vertical axis.

`Frequency domain / FFT amplitude` is expanded by default and follows the
Notebook's `abs(rFFT)/N`: no squaring, conversion to dB, additional doubling,
or detrending. A sinusoid of amplitude A has a positive-frequency peak of A/2.
FFT uses a linear vertical axis and initially displays 0.01–8 Hz. All frequencies
from DC to Nyquist are retained; use Autoscale to show them. If gaps are present,
only the longest continuous valid segment is used, without joining across gaps.
The actual sample interval and frequency resolution appear in the Stage details
`fft` field.

The feature stage overlays direct/processed RED/IR in one PPG time plot,
retaining peaks from each source and the actual-used peaks. It does not repeat
the underlying actual-used waveform. Without denoising, only the two direct
traces appear. Tables are organized by feature type and dimensionality,
retain all rows, and support pagination, filtering, and sorting. File-level
features have seven categories. Window-level tables show the actual
115-dimensional engineering quantities and 146-dimensional matrix, with separate
beat-morphology, dual-wavelength pairing, and validity/provenance tables.

Feature tables are collapsed by default into `engineering features`,
`file level features`, `time series features`, `freq domain features`,
and `morphology features`. Engineering windows/matrices belong to the
engineering group; PPI, time-domain, and nonlinear quantities belong to the
time-series group; frequency and morphology quantities have their own groups.
Other file-level summaries and dual-wavelength tables belong to the file-level
group. No table is shown twice, and collapsing does not change its data.
Matrix mode no longer computes an extra file-level vector. Products not
generated by the selected workflow are not invented, and exploratory features
in raw mode are not inputs to the raw classifier.

The peak preview's `Beatwise PPI / HR` shows beat intervals in seconds and
`60/PPI` in bpm, both positioned at the midpoint between adjacent peaks.
Dots/crosses distinguish valid/invalid intervals; the legend distinguishes
direct, processed, and intervals actually used for features. Invalid points
are hidden by default and can be shown through their legend entries; their
data are not deleted. Consecutive valid points are connected by straight
lines, with breaks at invalid intervals, missing data, or source changes.
No interpolation fills the gaps.

In every representation mode, feature engineering `Analyse` produces
`Window PPI / HR`; switching to `feature_matrix` is unnecessary.
Existing matrices are read directly. Other modes call the same `WindowPlan`
and matrix window-rate extractor, reuse detected peaks, and neither recompute
the complete 146-dimensional matrix nor change model inputs. Window length,
hop, and boundary settings come from `windows.engineering`, adjustable in
every mode without changing the raw model's separate `windows.raw_dl`.
When quality routing exists, its actual eligible intervals are used. With
routing disabled, windows span the recording without inventing routing
boundaries or quality grades.

Window plots are always expanded and show the entire recording, independent
of time-preview cropping. Mean, median, and population standard deviation
are visible by default at window centers. Adjacent valid windows are connected,
with breaks at invalid windows. Window HR summarizes beatwise `60/PPI`;
it is not the reciprocal of mean window PPI. Missing/ineligible windows are
not filled. If no peaks are available or the recording does not meet windowing
requirements, the plot frame remains and the reason appears below it and in
Stage details, without relaxing the original validity criteria.

IMU, SQI, denoiser, feature engineering, representation, and model each have
one collapsed parameter section, expanded by clicking its arrowed heading.
Main algorithm selections and switches remain visible. Analyse, Run, Stop,
existing-artifact selection, and previews remain outside the collapsible area.
Once expanded, the section stays open when switching algorithms or loading YAML.
Each parameter name includes a visible explanation of its meaning, algorithmic
role, and the effect of increasing/decreasing it or choosing another option.
Descriptions distinguish training-only settings from settings unused by the
current execution path. Collapsing changes only visibility, not values,
defaults, CLI/YAML exports, or computation.

1. **Configuration:** No YAML is selected initially, so existing function/dataclass
   defaults are used; the initial representation/model is raw/CompactCNN1D.
   Selecting pipeline YAML fills the controls. Selecting study YAML allows a
   case to be selected. Clearing YAML restores function defaults. YAML supplies
   initial values only; each subsequent Analyse uses the current controls.
2. **Input:** Select one or more recordings from the manifest dropdown and choose
   the plotted record. Recording options update when the manifest configuration
   changes. The main `roles` control shows only B/R/S/W; Resolved roles below it
   shows actual recording numbers. Newly selected families expand to all numbered
   roles. A YAML-selected subset remains unchanged until deselected and reselected.
   `training.classifier_role_families` independently controls classifier scope;
   B can be used only for calibration. Expand Custom CSV files to add paths,
   file_id, specific roles (B, R1–R4, S1–S2, W1–W2), and optional labels.
   Dynamic recordings require a B calibration record from the same participant;
   the calibration file can also be selected in the IMU section. Start/duration
   only controls the visible plot range; filtering and state estimation still
   process the complete recording.
3. **Stage Analyse:** Any downstream stage can be clicked directly. Missing or
   stale upstream stages run first, while unchanged results are reused within
   the session. Parameter changes invalidate the affected downstream results.
   Disabled optional modules bypass processing according to the original
   workflow. Whether the denoiser actually runs is determined by the original
   SQI/motion routing and reported in Stage details. Analyse does not train the
   classifier, motion detector, or SQI calibrator.
4. **Existing fitted artifacts:** Motion and SQI have separate file selectors
   supporting dropdown selection or typed paths. Click Refresh at the top after
   generating or copying artifacts. Training-quantile SQI requires existing
   calibration JSON containing bounds and fitted_on_participant_ids. Motion
   uses the model and thresholds associated with existing evidence JSON.
   Missing items are reported explicitly; no temporary fitting is performed
   on the participant being analyzed.
5. **Model Analyse:** Select a model_config case or a learned bundle directly.
   Choosing weights does not overwrite the controls. To start with the model's
   original settings, select its exported resolved_pipeline_config.yaml at
   the top; parameters remain editable afterward. Architecture, channels,
   and feature dimensions must match the selected weights, and training-fitted
   transforms must accompany the bundle. Stage details record the current config.
6. **Training:** Run appears only after switching the model section to Train.
   Select YAML before running. Current controls trains the current configuration;
   Comparison queue runs the stored combinations; Selected study YAML runs the
   selected plan's complete case sequence without overwriting every case.
   Stop terminates the background task and its child processes. Refit is off
   by default, and cache mode can be selected independently.
7. **Comparison:** Enter a unit name and click Add, then modify parameters and
   Add again. Remove last and Clear are available. Download the queue as complete
   CLI and runnable comparison YAML, or start it with the same model-section Run.
8. **Report:** Select existing pipeline output, analysis mode, plot modules, and
   statistical parameters, then Analyse. analyse_report.py writes report_output;
   the completed output is selected automatically. The panel previews figures,
   HTML, and tables, and also allows existing reports to be selected manually.
   Advanced tools expose indexes, model export, Excel, execution audit, and
   specialized studies. Training operations run only through model-section
   Train → Advanced tool request → Run. Controls come directly from the original
   CLI arguments. If a plan is required, select existing YAML and edit its
   expanded parameters. Execution and downloads read current controls; CLI text
   is read-only. Execution saves the current plan snapshot under
   `pipeline_output/.dashboard_requests/plans/` without overwriting the
   original YAML. Downloaded CLI includes the same snapshot for independent
   replay. Selecting this plan for specialized training also satisfies the
   requirement to choose YAML before training.

The page footer displays and downloads the equivalent CLI and resolved YAML
for the current stage. Place both files in the V6 directory to execute the
same stage from the CLI. Alternatively, invoke the same no-training stage
service directly without YAML:

```bash
python stage_analyse.py --stage ppg --record-id <record-id> \
  --set signal.ppg_filter.low_hz=0.2 \
  --set signal.ppg_filter.high_hz=8.0 \
  --set signal.ppg_filter.order=3
python stage_analyse.py --help
```

Stage previews retain only bounded in-session memory results. They do not
write preprocessing disk cache or modify existing runs or model weights.
Analyse can be repeated after a browser refresh or server restart. The
production feature-matrix extractor currently requires a routing timeline.
Disabling quality, motion, and denoising together does not produce that timeline
and reports the missing requirement. Select the existing diagnostics_only route
when analyzing this representation; Dash does not silently change the workflow.

Single-participant input can show quality, probabilities, and classification.
ROC/AUC, cohort confusion matrices, and significance tests require labeled
multi-participant data meeting the relevant class and sample requirements.

## Documentation

- [CLI_REFERENCE.md](docs/CLI_REFERENCE.md): Complete commands, parameters, and study/sweep usage.
- [OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md): Directories, data formats, Excel, and model weights.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): Relationships between configuration, scientific workflow, execution, reports, and Dash.
- [THESIS_CODE_CONFLICTS.md](docs/THESIS_CODE_CONFLICTS.md): Differences between thesis descriptions and implementation, and their impact.
- [PLAN_COMPATIBILITY.md](docs/PLAN_COMPATIBILITY.md): General and specialized study-plan entry points.
