# Static Line B staged screening flow

The scientific protocols retain their V2 identities, but the commands below run
from a standalone V6 repository root. See the [study entry points](../README.md),
[thesis experiment index](../thesis/README.md), and
[plan compatibility guide](../../../docs/PLAN_COMPATIBILITY.md).
V6 contains the required M2 authorities in `assets/authority/` and the frozen
motion evidence, all-29 model, and five fold models. Raw datasets remain external
inputs; old V2 result directories and root-level scripts are not required for
fresh training. Figures and HTML are generated separately by `analyse_report.py`.

This folder is the compute-saving alternative to
configs/studies/static_line_b_all_models_v2.yaml. The original 39-case plan is
unchanged and remains available.

The numbered files plus the final ShapeFormer stage form a scientific workflow.
Ordinary model reports never auto-select a final winner. The Stage 6 tuning
runner is the narrow exception: it records a deterministic development-only
promotion/selection manifest so the dependent regularization and channel plans
can inherit exactly the resolved parameters without manual transcription.

For all non-historical CNN/InceptionTime cases below, `B0+B2+B7` means the
selected DL execution state: 64 Hz, 5 s/2.5 s, AdamW/batch32 and file/role Line
B reporting. The later V2-core contracts remain authoritative: calibrated
Profile-A 0.3 Hz gravity LPF physical IMU for ordinary analysis modules and
all-eight per-window robust scaling for the DL tensor. Calibrated roll-pitch
EKF is an optional ablation, not an inherited default. B0 sampling is explicit:
exhaustive shuffle without replacement and
outer-train window/row inverse-frequency class weights; B5's Line-B weighted
sampler and participant-count weighting are not silently inherited. Historical
Stage 3 bridge plans remain immutable evidence and are not rewritten.

## Order and status

0. stage0_decision_bias_oracle.yaml
   Read-only, intentionally label-leaking decision-layer ceiling analysis for
   one completed final classifier case. It averages the five participant OOF
   probabilities to exactly 29 rows, enumerates the 5,151 three-class simplex
   biases at step .01, and maximises BA on those same labels. The output is an
   upper bound only: it is ineligible for performance reporting, selection,
   calibration, deployment, CI, or P values. Incomplete fold/staging artifacts
   are rejected.

   stage0_inception_small_no_gravity_supplement_v1.yaml
   is a separate 25-cell training supplement, not part of the label-leaking
   oracle computation. It runs only the tuned all-role InceptionTimeSmall with
   the registered no-gravity-removal IMU profile. Compare its new output with
   other newly completed V6 runs, or use `thesis/final_five_configurations.yaml`
   to train the five thesis configurations together. No old merged report or
   earlier finalist output is bundled.

1. 01_representation_baselines_v2.yaml
   Runs five routes: raw CompactCNN, feature-vector Logistic Regression,
   feature-matrix Small Inception, CompactCNN Fusion, and a configured raw Full
   InceptionTime. It screens a
   representation-plus-model combination; it is not a pure causal
   representation comparison and not a final winner study. The current YAML
   uses all five repeats; reduce `execution.repeats` to `[0]` explicitly for a
   lower-cost initial screen. This development plan is distinct from the frozen
   three-case thesis `representation_screening.yaml`.

2. 02_competitive_routes_models_v2.yaml
   r0 supplement with four cases: Raw InceptionFull, Raw InceptionSmall,
   Feature-vector RBF-SVM and Feature-vector ExtraTrees. The old 400 Hz
   InceptionFull evidence is incompatible and is not reused. Review this report
   alongside a matching rerun of Stage 1.

3. stage3_star.yaml (current restart)
   Runs CompactCNN and InceptionTimeFull for B0 plus seven independent
   B0-to-Bk changes. Execution is profile-major and model-paired. Repeats 0-4,
   folds 0-4, fixed 10 epochs, seed 42, serial CUDA execution produce exactly
   16 cases / 400 outer-fold fits / 4000 model epochs. The plan has no Phase 0.

   stage3_v3.yaml (repeated CompactCNN follow-up)
   Runs only B0+B2 (64 Hz, 5/2.5 s) and B0+B1+B2 (400 Hz,
   5/2.5 s). Each configuration uses repeats 0-4 and folds 0-4, so the
   exact budget is 2 cases / 50 outer cells / 500 fixed epochs. B0+B2 is
   the within-study reference; the paired difference isolates B1 conditional
   on B2. It reuses the same field-driven bridge runtime and has no Phase 0.

   stage3_alter.yaml (preserved historical chain)
   Executable specification for the revised nine-case legacy-to-V2 bridge and
   an optional, advisory Phase 0 data/source/cache audit. It freezes repeat 0,
   folds 0-4, seed 42, ten epochs, the requested execution and numeric report
   orders, L5-to-L6 sampler-plus-class-weight bundle, post-hoc aggregation
   views, and sampling diagnostics. Phase 0 never gates or changes training;
   the supplied plan sets `legacy_bridge.phase0.enabled: false` because the
   old-source audit is not needed for fresh experiments. No existing C0 result is
   automatically selected, paired, imported, or retrained.

4. 04_selected_inception_ensemble_v2.yaml
   The checked-in route is Raw InceptionFull. The file header gives the exact
   two substitutions for Matrix Inception. Skip this stage when the promoted
   model has no registered matched ensemble.

5. 05_sqi_motion_finalists_v2.yaml
   Eight declared 64 Hz CompactCNN1D cases compare all-role off/off, a
   fixed-threshold SQI route, SQI plus the frozen Frailty29 all-29 motion
   bundle, motion-only routing, and SQI-on/SQI-off PCA/FastICA one-attempt
   HR-recovery diagnostics. Every case uses the same B/R/S/W scope. The
   classifier and frozen motion inference use CUDA. Raw-DL windows are 5 s
   with 2.5 s hop; the CNN tensor
   uses all-eight per-window robust scaling without fold-level IMU
   post-scaling. All eight cases explicitly use the same calibrated roll-pitch
   EKF profile, making EKF a stage-wide ablation relative to the ordinary V2
   Profile-A LPF default rather than a motion-on-only difference. Denoiser
   outputs remain rate-only and are compared through
   direct/post HR evidence rather than being supplied to the raw CNN. The
   all-29 detector is explicitly in-sample auxiliary evidence on Frailty29 and
   is never described as outer-OOF motion evidence. The persisted detector is
   EKF-bound, but that requirement no longer creates an IMU-profile difference
   between the eight within-stage comparisons. SQI-off denoiser cases still
   persist direct and post-denoiser HR diagnostics without running direct SQI.

5b. 05_sqi_motion_logistic_regression_l2_v2.yaml
   The complete 5x5 LogisticRegressionL2 re-test of the same eight module
   compositions. All cases use the feature-vector route, identical B/R/S/W
   scope, calibrated EKF, Line B aggregation and frozen split registry.
   PCA/FastICA outputs remain rate-only, but eligible post-denoiser pulse,
   PPI and PRV evidence enters the classifier through
   `denoise_then_extract_rate_features`. SQI-off/high-motion recovery is
   explicitly authorized only for this rate-feature policy and still requires
   a passing post-denoiser Q_rate. LogisticRegressionL2 runs on CPU; CUDA is
   used by the frozen motion detector.

6. stage6_batch_LR_search.yaml
   InceptionTimeFull batch/LR successive halving. Six candidates first run five
   epochs on fold 0 from each of the five split seeds. Top three by mean
   participant BA, then mean macro-F1, then case_id enter complete 5x5 fixed10
   tuning CV. This costs 900 model-epochs rather than 1500 (40% reduction).
   Both rungs are tuning evidence, never final-test evidence.

   stage6_regula_search.yaml
   Requires the completed batch/LR study directory. It imports the selected
   batch and LR, then runs the declared R1-R9 WD/dropout/label-smoothing grid.

   stage_ablation_channels.yaml
   Requires the completed regularization study directory. It compares the
   full eight-channel reference with RED+IR and ACC+gyro. Only the DL tensor is
   sliced; physical IMU and amplitude-preserving analysis views remain intact.

   06_sequential_single_factor_ablation_v2.yaml is retained as a separate
   three-case CompactCNN learning-rate ablation, now also locked to the selected
   state. It is not the InceptionTime tuning route.

6b. stage_ablation_s1_163_gravity_removal_v1.yaml
   Matched two-case, 5x5 S1_163 all-role InceptionTimeFull ablation. The
   reference uses Profile-A 0.3 Hz gravity removal; the candidate keeps the
   same SI conversion, participant-B sensor-bias calibration and 20/40 Hz
   sensor filtering but performs no gravity estimation or subtraction. Every
   data, model, optimizer, window, module-off, split and seed setting is shared.
   The reference is rerun in the same code snapshot for paired inference.

Last. stage_last_shapeformer_stability_v2.yaml
   ShapeFormer is intentionally deferred until every numbered stage has been
   reviewed because its fold-local discovery and model fitting are unusually
   expensive. Default execution is one cell. If stable, rerun one complete
   repeat, then full 5x5 only after another manual review. Its failure remains
   isolated from ordinary models.

## Commands

Run all commands from the V6 repository root. Ordinary plans use `sweep.py`;
its `validate` and `--dry-run` forms do not train:

```bash
python sweep.py validate \
  --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml
python sweep.py run \
  --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml \
  --dry-run
python sweep.py run \
  --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml \
  --run-name stage1_representation
python analyse_report.py run --mode comparison \
  --input pipeline_output/stage1_representation --preset classification
```

For another ordinary stage, replace `--plan` and choose a new `--run-name`.
Device, repeats, folds, jobs, and cache settings are declared in that YAML's
`execution` section. `sweep.py` does not accept the old V2 runner's resource
flags. ShapeFormer's checked-in first step is one cell; to escalate, change
`execution.folds` to `[0, 1, 2, 3, 4]`, then likewise `execution.repeats` for
complete 5 × 5 evaluation. For a cheaper Stage 1 screen, explicitly use
`execution.repeats: [0]`; the supplied current plan uses all five repeats.

The centered-star and follow-up bridge protocols use the same entry point:

```bash
python sweep.py run \
  --plan configs/studies/static_line_b_staged_v2/stage3_star.yaml \
  --run-name stage3_star
python sweep.py run \
  --plan configs/studies/static_line_b_staged_v2/stage3_v3.yaml \
  --run-name stage3_followup
python sweep.py run \
  --plan configs/studies/static_line_b_staged_v2/stage3_alter.yaml \
  --run-name stage3_chain
```

Add `--dry-run` to inspect resolved cases without training. The supplied
`stage3_alter.yaml` disables the optional old-source Phase 0 audit; its nine
training cases, inputs, and budget remain unchanged.

Run the motion/denoiser computation separately from its report:

```bash
python specialized_pipeline.py validate \
  --plan configs/studies/static_line_b_staged_v2/stage5_pre.yaml
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage5_pre.yaml \
  --run-name stage5_motion_denoiser
python analyse_report.py specialized-report \
  --input pipeline_output/stage5_motion_denoiser
```

The plan disables only `motion_model_comparison`, the old-result packaging
step. The four motion training/evaluation stages and the configured denoiser
benchmark still execute. No V2 `--source-root` is required.

Run the dependent hyperparameter stages with explicit new upstream outputs:

```bash
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage6_batch_LR_search.yaml \
  --run-name stage6_batch_lr
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage6_regula_search.yaml \
  --upstream-study pipeline_output/stage6_batch_lr --run-name stage6_regularization
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage_ablation_channels.yaml \
  --upstream-study pipeline_output/stage6_regularization --run-name stage6_channels
python analyse_report.py specialized-report --input pipeline_output/stage6_batch_lr
```

To obtain full-resource results for every batch/LR candidate, also run
`python specialized_pipeline.py complete --study-dir pipeline_output/stage6_batch_lr`.
This trains the originally unpromoted candidates; it does not change the
promotion rule or retrospectively make development results final-test evidence.

Stage 0 requires an already completed source case. The following example uses
a new finalcase run, never the absent historical archive:

```bash
python analyse_report.py specialized-run \
  --plan configs/studies/static_line_b_staged_v2/stage0_decision_bias_oracle.yaml \
  --study-dir pipeline_output/finalcase_v6_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --output-name finalcase_oracle
```

For reusable post-hoc plans, replace historical `source`/`sources` study paths,
case IDs, explicit prediction paths, and role-scope `ranking_evidence` with
compatible new outputs and their actual ranks. Validation cannot create these
inputs. Oracle-selected biases remain intentionally label-leaking diagnostics
and must not be reused for classification or model selection.

Pipeline runs produce data and Excel, not automatic plots or HTML. Reports are
created only by `analyse_report.py`; this does not retrain the source models.
Report presets select tables and figures independently from training plans.
When comparing Line A and Line B, both aggregations must use the same held-out
file probabilities. Line A sensitivity analysis is not separately trained
evidence and is not a model-selection result.

## Default budgets

- Stage 0: 0 fits; 29 repeat-mean participant probabilities and 5,151 bias
  vectors. CPU-only post-hoc analysis.
- Stage 1: 5 cases, 125 outer cells; 25 when explicitly reduced to one repeat.
- Stage 2: 4 supplemental cases, repeat 0 with all five folds, 20 outer cells.
- Current Stage 3 centered star: exactly 16 cases, 400 fits and 4000
  model-epochs; no Phase 0 execution.
- Stage 3 v3 CompactCNN follow-up: exactly 2 cases, 50 fits and 500
  model-epochs; no Phase 0 execution.
- Preserved historical Stage 3 cumulative chain: exactly 9 cases, 45 fits and
  450 model-epochs. Its optional advisory Phase 0 audit adds no fits or
  model-epochs and does not affect that historical budget.
- Stage 4: 2 scientific cases, 50 outer cells and 150 fitted networks.
- Stage 5 CompactCNN diagnostic screen: 8 cases, repeat 0 and all five folds,
  40 outer cells.
- Stage 5 LogisticRegressionL2 rate-feature re-test: 8 cases, all five repeats
  and all five folds, 200 outer cells.
- Stage 6 batch/LR: 30 five-epoch screening cells plus 75 fixed10 promoted
  cells = 900 model-epochs (versus 1500 for direct 6-case 5x5 fixed10).
- Stage 6 regularization: 9 cases, 225 fixed10 cells.
- Channel ablation: 3 cases, 75 fixed10 cells.
- Retained CompactCNN LR ablation: 3 cases, 75 fixed10 cells.
- Stage last (ShapeFormer): 1, then 5, then 25 outer cells.

Every fresh study writes to `pipeline_output/<run>`; ordinary case outputs use
`<comparison>/repeat_<RR>/fold_<FF>/`. Reports go separately to
`report_output/<run>` and reusable weights/configuration to `model_config/<run>`.
Without `--run-name`, the YAML filename and timestamp determine the run name.
An existing completed output is not silently overwritten.
