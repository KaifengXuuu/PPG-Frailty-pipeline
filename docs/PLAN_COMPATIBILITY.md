# Study-plan entry points and compatibility

V6 includes the 29-plan compatibility set (28 V2 plans plus `finalcase.yaml`)
and eight dedicated thesis plans: 37 YAML files in total. Different
schemas are not forced into one grid interpretation. Each plan is validated
by its own strict loader before entering its computation or analysis route.
This preserves the V2 workflow while separating training and presentation outputs.

## Standalone inputs

Run these commands from the V6 repository root. Supply the raw
`PPG_Testing_05_01_2026/` and, for PTT experiments, `physionet.org/` directories
locally, or set `PPG_FRAILTY_DATA_ROOT` to their parent directory. Raw datasets
are excluded from Git. No V2/V5 checkout or old result archive is required for
fresh computation.

The original M2 manifest and split authorities are included byte-for-byte under
`assets/authority/`. The PTT acceleration-unit conversion uses the current
packaged implementation, with frozen source evidence retained as provenance;
it does not load an old root-level Python script. Reference motion weights and
their evidence are also included. `legacy_bridge.phase0.enabled` is false in
the supplied plans: the old-archive audit is not part of model training.

## Canonical plans: 23

The 23 `ppg_frailty.study_plan.v2` plans run through either entry point:

```bash
python pipeline.py run-plan --plan configs/studies/finalcase.yaml
python sweep.py run --plan configs/studies/finalcase.yaml
```

Here `finalcase.yaml` is a directly executable example. Substitute the other
canonical files in `--plan` to use the same route.

These plans cover single, grid, catalog, representation/model, ensemble,
SQI/motion, sequential ablation, Legacy Bridge, finalcase, gravity, and
ShapeFormer workflows. The V5-derived execution route disables only plan
HTML/plot/report policies. Scientific configuration, case expansion,
repeats/folds, seeds, splits, and training algorithms remain unchanged.
Outputs use `pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/`.

Two motion-finalist plans reference the same frozen detector bundle. The
historical artifact tree is not copied. The evidence JSON, all-29 motion model,
and five matching outer-fold model files are bundled. The latter support
`matching_outer_fold_or_all29_final` without installing old results.
The all-29 reference is unchanged:

- `motion_internal_evidence.json`: SHA-256 `10f02a9d784e06471c7109ff8dc92d28f1a8d7753f8fdf179bebce5699fb446c`.
- `formal_motion_model.pt`: SHA-256 `62a09c53fecf90dfb9388900df19efccc62facf9f72b221b09c7d06c999c6eca`.

The loader maps historical absolute provenance paths to the bundled
byte-identical copies. Evidence hashes still identify the exact trained models.
Baseline and finalcase do not depend on this bundle.

## Specialized plans: 14

### Analysis-only: 5

- Four `ppg_frailty.stage0_decision_bias_oracle.v1` plans.
- One `ppg_frailty.role_scope_decomposition.v1` plan.

These are post-hoc templates, not fresh-training commands. They consume completed
prediction/ranking artifacts and write only `report_output`. Historical paths
inside their YAML files describe original sources; those result directories are
not included. Point the inputs at new, completed V6 runs before use. For an
oracle diagnostic on a newly completed finalcase run:

```bash
python analyse_report.py specialized-run \
  --plan configs/studies/static_line_b_staged_v2/stage0_decision_bias_oracle.yaml \
  --study-dir pipeline_output/finalcase_v6_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --output-name finalcase_oracle
```

For reusable oracle YAML, update `source.study_dir`, `source.case_id`, and any
explicit `source.prediction_file`. For role-scope decomposition, update both
`sources.*.study_dir`/`case_id` entries and `ranking_evidence` to compatible new
prediction and ranking outputs. Expected ranks must describe the new ranking,
not assume the old ranks survived retraining. `specialized-validate --plan ...`
checks the edited plan; it does not create missing predictions or ranking data.
`--source-root` is optional and only selects the base for relative source paths.
Oracle results intentionally use evaluation labels and are not deployable models.

### Computation: 9

- Three `ppg_frailty.stage5_pre_motion_ptt.v1` plans.
- `ppg_frailty.stage_ablation_01_static_peaks.v3`.
- Five `ppg_frailty.hyperparameter_study_plan.v1` plans.

These use the extracted V5 computation runners, retaining V2 numerical semantics
and writing only `pipeline_output`. A Stage5 example:

```bash
python specialized_pipeline.py validate \
  --plan configs/studies/static_line_b_staged_v2/stage5_pre.yaml

python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage5_pre.yaml \
  --run-name stage5_pre_v6_01
```

`motion_model_comparison.enabled: false` disables only packaging/comparing old
result archives. All four scientific motion stages remain: internal OOF training,
frozen evaluation on PTT, PTT motion training, and reverse evaluation on Frailty29.
Fresh motion tests therefore need raw datasets, not a historical `--source-root`.

Static-peak uses the same form with
`stage_ablation_01_static_peak_detectors.yaml`. Hyperparameter plans
do not consume `--source-root`. The first stage can run directly; later stages
must explicitly reference the preceding stage's `selected_configuration.json`:

```bash
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage6_batch_LR_search.yaml \
  --run-name stage6_batch_lr_01

python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage6_regula_search.yaml \
  --upstream-study pipeline_output/stage6_batch_lr_01 \
  --run-name stage6_regularization_01
```

`stage_ablation_channels.yaml` likewise uses `--upstream-study` to reference
the completed regularization run.

These computation routes write structured data directly, rather than generating
presentation files and recursively deleting or rejecting them afterward.
After completion, the adapter indexes artifacts and writes the convenience data
view `tables/pipeline_data.xlsx`. Stage5 exports a total of 10 outer-fold and
2 final motion-weight sets across frailty29/PTT22 to `model_config/<run>`.
Every V5-derived hyperparameter phase exports all frailty fold weights and
corresponding case configurations. Static-peak trains no model and explicitly
records `model_trained=false`, `model_kind=not_applicable`.

Stage5/static-peak/hyperparameter reports read persisted data directly through
`reporting.specialized`. Hyperparameter phases reuse the ordinary classification
renderer. The public CLI does not write back into its pipeline source:

```bash
python analyse_report.py specialized-report \
  --input pipeline_output/SPECIALIZED_RUN
```

The sole existing V2 successive-halving completion workflow is retained:

```bash
python specialized_pipeline.py complete \
  --study-dir pipeline_output/SPECIALIZED_RUN
```

`specialized_pipeline.py run --resume pipeline_output/<run>` resumes the same
Stage5/static-peak run. Hyperparameter orchestration also resumes its existing
persisted V5-compatible phases. No new candidate-promotion mathematics is
introduced. `complete` performs only the originally defined V2 full-CV completion
of candidates that were not promoted.

## Numerical boundary

Presentation calls were extracted from computation runners into compatible
`reporting.specialized` entry points. Computation routes contain no PNG/HTML
writers and need no post-training presentation cleanup. Targeted golden/numerical
tests audit mathematics, schemas, ordering, thresholds, splits/seeds, and artifact
fields. Final scientific equivalence still requires full output comparisons
in the frozen environment; static source hashes cannot replace that evidence.
