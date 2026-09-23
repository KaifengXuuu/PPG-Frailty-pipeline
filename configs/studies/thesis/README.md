# Thesis experiment configurations

This directory organizes execution plans by the comparison units in
`Kaifeng_Masterarbeit_draft_v1_0.docx`. One YAML corresponds to one test;
its `cases` are that test's comparison groups, not separate tests.
Configurations prioritize `study_plan.yaml` / `resolved_plan.yaml` and
each group's `resolved_config.yaml` saved by successful V2 runs, rather
than same-named templates that may have been edited later.

Run the following commands from the V6 repository root.
`validate` only parses and expands configuration without training; it does
not establish that all dependency artifacts are present or that numerical
reproduction has been completed. Ordinary classification experiments use
`sweep.py`; peak detection, motion, denoising, and staged searches use
`specialized_pipeline.py`; post-hoc prediction decomposition uses
`analyse_report.py`. Data go to `pipeline_output/<run>`, while figures
are written separately to `report_output/<run>`.

V6 contains the small M2 manifest/split authorities in `assets/authority/` and
the frozen motion model/evidence needed by these configurations, including five
outer-fold models. Supply the raw internal and PTT datasets at the project root
or through `PPG_FRAILTY_DATA_ROOT`. Fresh experiments do not require old V2/V5
source trees or historical prediction/report directories. The old-source Phase 0
audit is disabled; this does not change the bridge training algorithms.

## Experiment index

| Thesis test | YAML | Comparison groups and execution resources | Entry point |
|---|---|---|---|
| 4.1.1 / Tables 18–19: static peak detection | [stage_ablation_01_static_peak_detectors.yaml](../static_line_b_staged_v2/stage_ablation_01_static_peak_detectors.yaml) | MSPTDfast versus Aboy project v2; sit recordings from 22 PTT participants, RED/IR separately | specialized |
| 4.1.2 / Table 21: bidirectional motion transfer | [motion_detector.yaml](motion_detector.yaml) | Frailty29 → PTT22 and PTT22 → Frailty29; motion-specific grouped five-fold evaluation, no denoiser | specialized |
| 4.1.3 / Tables 22–23: denoiser | [denoiser.yaml](denoiser.yaml) | Identity plus six reducers; PTT static/dynamic scoring separately, no motion training | specialized |
| 4.2.1 / Table 24: representation comparison | [representation_screening.yaml](representation_screening.yaml) | raw/CNN, feature-vector/logistic, feature-matrix/Small Inception; 3 × 5 repeats × 5 folds | sweep |
| 4.2.2 / Table 25: early matched three-model comparison | No numerically equivalent V5/V6 training YAML yet | Historical CNN, InceptionTime, ShapeFormer; see the historical boundary below | Decision pending |
| 4.2.3 / Tables 26–27: SQI/motion routing | [sqi_motion_routing.yaml](sqi_motion_routing.yaml) | Four combinations of two switches, all with denoising disabled; 4 × 5 × 5 | sweep |
| 4.2.4 / Tables 28–30: B0–B7 single-factor comparisons | [legacy_bridge_ablation.yaml](legacy_bridge_ablation.yaml) | CNN and InceptionTime, each B0–B7; 16 × 5 × 5 | sweep |
| 4.2.4 / Table 31, Appendix D-A: batch/LR | [batch_learning_rate_search.yaml](batch_learning_rate_search.yaml) | Six groups; screening, promotion, then full-CV completion of unpromoted groups | specialized + complete |
| 4.2.4 / Table 32, Appendix D-B: regularization | [regularization_search.yaml](regularization_search.yaml) | Nine groups; reads the selected batch/LR values; 9 × 5 × 5 | specialized |
| 4.2.4.5 / Tables 33–34: final five configurations | [final_five_configurations.yaml](final_five_configurations.yaml) | Combined rerun of thesis Ranks 1–5; 5 × 5 × 5, with no automatic winner selection | sweep |

`final_five_configurations.yaml` retains each group's historical parameters.
Groups do not all share the same role scope, epochs, batch size, class-counting
basis, or gravity processing. Rank 4 is the configuration actually run,
`s1_163_v2_port_all_roles_modules_off`, not the tuned all-role group in a
later same-named template. The selected single-configuration entry point
remains [finalcase.yaml](../finalcase.yaml), corresponding to Rank 2
`tuned_all_roles__inception_small_no_gravity`. This directory does not change
that selection.

## Ordinary classification comparisons

The three-way representation comparison is an example. Replace `--plan`
and `--run-name` to execute other sweeps:

```bash
python sweep.py validate --plan configs/studies/thesis/representation_screening.yaml
python sweep.py run --plan configs/studies/thesis/representation_screening.yaml \
  --run-name thesis_representation
python analyse_report.py run --mode comparison \
  --input pipeline_output/thesis_representation --preset classification
```

Each ordinary plan retains five repeats with five participant-isolated folds
per repeat. The SQI/motion comparison reuses the original experiment's frozen
all-29 motion weights. This auxiliary processing was fitted on the same internal
cohort and must not be described as independent outer OOF for the motion model.
V6 includes the referenced
`artifacts/studies/.../20260820_225546_.../motion_internal/motion_internal_evidence.json`
and its all-29 and matching fold weights. These are model-input artifacts, not
dependency-version locks or complete historical results.
Substituting arbitrary motion models does not reproduce the original experiment.

## Peak detection, motion, and denoising

```bash
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage_ablation_01_static_peak_detectors.yaml \
  --run-name thesis_static_peaks
python specialized_pipeline.py run --plan configs/studies/thesis/motion_detector.yaml \
  --run-name thesis_motion
python specialized_pipeline.py run --plan configs/studies/thesis/denoiser.yaml \
  --run-name thesis_denoiser
python analyse_report.py specialized-report --input pipeline_output/thesis_denoiser
```

Before running, replace `run` with `validate` and omit `--run-name`.
Motion's `motion_model_comparison.enabled: false` omits only the old-result
comparison package. All four scientific stages remain: internal grouped OOF
training, frozen PTT evaluation, PTT grouped training, and frozen reverse
evaluation on Frailty29. No historical `--source-root` is needed.
Denoising requires only the project's PTT data. Each YAML's `execution`
selects the computation stages; denoiser evaluation does not retrain unrelated
motion models. PTT data must exist locally; these commands do not download it.

## Post-hoc analyses of new runs

The decision-oracle and role-scope templates elsewhere in `configs/studies/`
still identify their historical sources. They are not runnable against absent
old results. Set their source study/case/prediction fields to newly completed
V6 outputs; role-scope analysis additionally needs matching ranking evidence
and ranks from the new report. The oracle CLI can override `--study-dir` and
`--case-id`. See the [entry-point guide](../../../docs/PLAN_COMPATIBILITY.md)
for a concrete command. These analyses never retrain models.

## Batch/LR and regularization searches

```bash
python specialized_pipeline.py run \
  --plan configs/studies/thesis/batch_learning_rate_search.yaml --run-name thesis_batch_lr
python specialized_pipeline.py complete --study-dir pipeline_output/thesis_batch_lr
python specialized_pipeline.py run \
  --plan configs/studies/thesis/regularization_search.yaml \
  --upstream-study pipeline_output/thesis_batch_lr --run-name thesis_regularization
python analyse_report.py specialized-report --input pipeline_output/thesis_batch_lr
python analyse_report.py specialized-report --input pipeline_output/thesis_regularization
```

The first step runs all six groups for 5 epochs on fold 0 of every repeat,
then retrains the top three from scratch for 10 epochs over the complete
5 × 5 CV. `complete` trains full CV for the remaining three. Without that
step, the six-group full-resource results in thesis Table 31 are unavailable.
Regularization reads batch/LR from the upstream selected configuration rather
than choosing fixed values again. If a rerun changes the upstream selection,
the downstream configuration changes accordingly. This is the original
dependent-search workflow, not a separately frozen experiment.

## Historical sources and reproduction boundaries

The following paths identify the original sources of the frozen parameters.
They were under V2's `artifacts/studies/static_line_b_staged_v2/`; they are not
runtime dependencies and are not included as result archives in V6.

| Configuration/result | Historical run directory |
|---|---|
| Static peak detection | `20260823_104616_stage-ablation-01-static-peak-detectors-v3` |
| Bidirectional motion transfer | `20260820_225546_staged-static-05-pre-motion-ptt-v1` |
| Denoising | `20260820_182324_staged-static-05-pre-motion-ptt-v1` |
| Three-way representation | First three groups of `20260823_075821_catalog_sweep_staged-static-01-representation-baselines-v2` |
| SQI/motion | First four groups of `20260824_031024_catalog_sweep_staged-static-05-sqi-motion-compact-cnn-v6` |
| B0–B7 | `20260821_162454_catalog_sweep_staged-static-03-centered-star-v1` |
| Batch/LR | `20260822_190832_hyperparameter_staged-static-06-batch-lr-successive-halving-v1` and its phases |
| Regularization | `20260823_031621_hyperparameter_staged-static-06-regularization-grid-v1` and its full_cv phase |
| Final Ranks 1/3 | `20260824_160517_catalog_sweep_final-case-all-roles-inception-architecture-comparison-v1` |
| Final Rank 2 | `20260824_175009_catalog_sweep_stage0-inception-small-no-gravity-supplement-v1` |
| Final Ranks 4/5 | `20260824_111943_catalog_sweep_final-case-comparison-inception-full-v1` |

Important distinctions:

- Historical three-way representation and both hyperparameter searches use
  `aboy_project_v1`; the current public catalog defaults to MSPTDfast.
  These plans explicitly retain the historical detector without changing
  global defaults.
- Inactive motion/feature-matrix fields in the two searches use current defaults.
  These fields do not enter the raw classification computation. Stage resources,
  actual training, inputs, and peak-detection settings retain historical values.
- Neither static-peak detector nor its alignment parameters changes algorithm.
  The current statistical configuration uses a Holm–Šidák family across five
  metrics × two channels; early historical snapshots declared only the F1 family.
  The report's multiple-comparison scope must therefore match the cited thesis table.
- Historical denoiser scoring used Aboy v1; the later `stage5_pre.yaml` switched
  to MSPTDfast. This directory retains the old scorer and does not label
  results from a replaced scorer as reproduction of the original.
- The final-five YAML combines source groups for a rerun; it does not replay
  the old seven-group merged reporting archive. Comparisons between differently
  configured groups are not all described as single-factor ablations.

The thesis's early matched three-model comparison comes from the old runner:
`results_frailty3/_overfitting_sweep/20260527_1320_cnn_inceptionTime` and
`20260528_1045_shapeformer_0extra`. V2 only reanalyzed those historical results;
see
`results_frailty3/_v2_reanalysis/20260824_historical_search_reports_v2/01_early_three_model_matched/report_manifest.json`.
Outer-evaluation labels participated in early stopping. Current fixed-epoch
V5/V6 studies cannot stand in as equivalent reruns. Retaining historical analysis
versus explicitly restoring the old training protocol requires a separate
decision. No YAML is supplied under a misleading reproducibility label.
A generic historical summary is not a numerically equivalent replacement
for that matched report.

Appendix D-C figure sources can be traced further. The window-length
(5/10/15 seconds) and overlap (30/50%) plots are byte-identical to the corresponding
PNGs under
`results_frailty3/_sweep_analyse/20260608_0659_combined_cnn_inceptiontime_shapeformer/figures/`.
The six plots for dropout, weight decay, label smoothing, retained-window
fraction, epochs, and regularization factor match corresponding PNGs under
`20260616_1139_overfitting_inceptiontime/figures/` and
`20260616_1143_overfitting_inceptiontime/figures/` (both in `_sweep_analyse/`).
The latter analyses' `clean_runs.csv` points to
`_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2`.

These are exploratory results from the old runner, not the V2 batch/LR or
nine-group regularization experiments indexed above. Fixed-epoch exploration
also differs from the early three-model early-stopping workflow. Complete
input, training, and evaluation equivalence with V5 has not been checked;
new training configurations must not be invented from individual box-plot
axis labels. Existing development plans such as
[ablation_fixed_epochs_v2.yaml](../ablation_fixed_epochs_v2.yaml) and
[stage3_v3.yaml](../static_line_b_staged_v2/stage3_v3.yaml) can run, but their
filenames alone do not establish reproduction of these older experiments.
