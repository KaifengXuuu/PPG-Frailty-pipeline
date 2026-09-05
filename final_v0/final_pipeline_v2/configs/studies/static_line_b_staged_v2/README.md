# Static Line B staged screening flow

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
   the registered no-gravity-removal IMU profile. After completion it is merged
   read-only with the earlier four finalists and completed Full/Small study;
   no completed case is retrained.

1. 01_representation_baselines_v2.yaml
   Runs three low-cost representatives. Raw CompactCNN and compact fusion use
   the selected V2-core plus B0/B2/B7 state. It screens a
   representation-plus-model combination; it is not a pure causal
   representation comparison and not a final winner study. The default is one
   complete repeat. If routes are close, rerun with all repeats before pruning.

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
   the current plan enables it for additional context and it may be skipped by
   setting `legacy_bridge.phase0.enabled: false`. No existing C0 result is
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

Run or dry-run any stage from the final_pipeline_v2 directory:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml --dry-run

Run the zero-training-cost Stage 0 oracle after its configured source case has
completed all five repeats:

    python decision_bias_oracle_v2.py run --plan configs/studies/static_line_b_staged_v2/stage0_decision_bias_oracle.yaml --output-root artifacts/studies/static_line_b_staged_v2

Run the separate Stage 0 Small-Inception/no-gravity supplement on CUDA:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/ppg-stage0-small-no-gravity-mpl python frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage0_inception_small_no_gravity_supplement_v1.yaml --device cuda --repeats all --folds all --jobs 1 --preprocessing-cache-mode read_write --preprocessing-cache-root artifacts/studies/cache --preprocessing-cache-namespaces imu_calibration,canonical_signal_views,raw_windows --no-measure-operational-costs --output-root artifacts/studies/static_line_b_staged_v2

After that command prints its `Study output`, paste the exact directory into
`SMALL_NOGRAV_STUDY`, then create the seven-case read-only merged report:

    SMALL_NOGRAV_STUDY=/absolute/path/from/Study-output

    python tools/recover_completed_cases_v2.py --source previous=artifacts/studies/static_line_b_staged_v2/20260824_111943_catalog_sweep_final-case-comparison-inception-full-v1 --source architecture=artifacts/studies/static_line_b_staged_v2/20260824_160517_catalog_sweep_final-case-all-roles-inception-architecture-comparison-v1 --source small_no_gravity="$SMALL_NOGRAV_STUDY" --output artifacts/studies/static_line_b_staged_v2/20260824_final_case_inception_small_no_gravity_merged_v2 --study-id final_case_inception_small_no_gravity_merged_v2 --purpose "Merged final-case comparison: four prior candidates, matched Full/Small architectures, and Small InceptionTime without gravity removal." --reference-case-id tuned_all_roles__inception_full --detailed-configuration-top-k 5

The merged Markdown and HTML summaries contain a collapsible complete resolved
configuration table for the predictive Top 5. The identical lossless long table
is written to `tables/top_model_complete_configurations.csv` and to its own
worksheet in `tables/report_tables.xlsx`; provenance hashes remain in source
manifests rather than replacing named input-data values.

Run the four-case Stage 2 supplement:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/02_competitive_routes_models_v2.yaml

Run Stage 6 batch/LR successive halving on CUDA:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/ppg-stage6-mpl python hyperparameter_studies_v2.py run --plan configs/studies/static_line_b_staged_v2/stage6_batch_LR_search.yaml --device cuda --jobs 1 --output-root artifacts/studies/static_line_b_staged_v2

Run the S1_163 all-role gravity-removal ablation on CUDA:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/ppg-s1-163-gravity-mpl python frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage_ablation_s1_163_gravity_removal_v1.yaml --device cuda --repeats all --folds all --jobs 1 --preprocessing-cache-mode read_write --preprocessing-cache-root artifacts/studies/cache --preprocessing-cache-namespaces imu_calibration,canonical_signal_views,raw_windows --no-measure-operational-costs --output-root artifacts/studies/static_line_b_staged_v2

Then pass its exact output directory to regularization, followed by channels:

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/ppg-stage6-mpl python hyperparameter_studies_v2.py run --plan configs/studies/static_line_b_staged_v2/stage6_regula_search.yaml --upstream-study PATH_TO_BATCH_LR_STUDY --device cuda --jobs 1 --output-root artifacts/studies/static_line_b_staged_v2

    CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=0 MPLCONFIGDIR=/tmp/ppg-stage6-mpl python hyperparameter_studies_v2.py run --plan configs/studies/static_line_b_staged_v2/stage_ablation_channels.yaml --upstream-study PATH_TO_REGULARIZATION_STUDY --device cuda --jobs 1 --output-root artifacts/studies/static_line_b_staged_v2

Inspect the Stage 3 protocol without expansion or training:

    python3 -c "import sys; sys.path.insert(0, 'src'); from ppg_frailty.study import load_study_plan; p=load_study_plan('configs/studies/static_line_b_staged_v2/stage3_alter.yaml'); print(p.legacy_bridge.to_dict())"

Dry-run or execute the current centered-star Stage 3:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage3_star.yaml --device cuda --repeats all --folds all --jobs 1 --no-measure-operational-costs --dry-run

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage3_star.yaml --device cuda --repeats all --folds all --jobs 1 --no-measure-operational-costs

Dry-run or execute the repeated CompactCNN Stage 3 v3 follow-up:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage3_v3.yaml --device cuda --repeats all --folds all --jobs 1 --no-measure-operational-costs --dry-run

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage3_v3.yaml --device cuda --repeats all --folds all --jobs 1 --no-measure-operational-costs

The preserved historical `stage3_alter.yaml` expands through the cumulative
Legacy Bridge runtime. With its checked-in `legacy_bridge.phase0.enabled: true`,
a `run` also executes the advisory Phase 0 source/manifest/channel/IMU/cache/split
audit. Its status is recorded but never blocks, changes, or selects training.
Set the flag to `false` to skip the audit; the nine historical cases and their
inputs remain unchanged. Its formal command is:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage3_alter.yaml

This command is the full 45-fit study, not a dry-run. Use `--dry-run` to inspect
the nine resolved cases without Phase 0 or training.

Stage 1 contains deep raw/fusion cases, so its plan disables concurrent deep
case execution. A larger command-line jobs value is accepted but is
automatically reduced to one effective case at a time to avoid memory
oversubscription.

Escalate Stage 1 to full 5x5 only when the one-repeat routes are too close:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml --repeats all --folds all

Stage-last ShapeFormer stability ladder:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage_last_shapeformer_stability_v2.yaml

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage_last_shapeformer_stability_v2.yaml --repeats 0 --folds all

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/stage_last_shapeformer_stability_v2.yaml --repeats all --folds all

Every default run generates a report, and every standalone report rebuild uses
the same reporting path:

    python3 frailty_3class_sweep_v2.py report --study-dir PATH_TO_STUDY

Stage 6 orchestration reports use:

    python3 hyperparameter_studies_v2.py report --study-dir PATH_TO_STAGE6_STUDY

All ordinary stage plans explicitly request the current modular report bundle:
paired CSV/JSON tables and plots, compact mean ± SD columns, one-sheet-per-table
XLSX export, component/input/fixed-parameter tables, and split/training-seed
reproducibility tables. Existing figures remain in place.

When file-level OOF is present, the report always includes both balanced
accuracy views: the declared canonical Line B result and Line A equal-file
post-hoc reaggregation of the same held-out file probabilities. Line A is
aggregation sensitivity only, not separately trained evidence and never a
selection or leaderboard result. An explicit --no-report skips report creation
for that run; the standalone report command can add it later without training.

## Default budgets

- Stage 0: 0 fits; 29 repeat-mean participant probabilities and 5,151 bias
  vectors. CPU-only post-hoc analysis.
- Stage 1: 3 cases, 15 outer cells by default; 75 only if manually escalated.
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

Every study writes to artifacts/studies/static_line_b_staged_v2 under its own
timestamped study directory. The existing non-overwrite and atomic publication
behavior remains active.
