# Static Line B staged screening flow

This folder is the compute-saving alternative to
configs/studies/static_line_b_all_models_v2.yaml. The original 39-case plan is
unchanged and remains available.

The numbered files plus the final ShapeFormer stage form a manual scientific workflow. Results never auto-select a
winner and no later file silently inherits an earlier result. Read each report,
record the decision, then edit only the documented selector in the next file.

## Order and status

1. 01_representation_baselines_v2.yaml
   Runs four canonical low-cost representatives. It screens a
   representation-plus-model combination; it is not a pure causal
   representation comparison and not a final winner study. The default is one
   complete repeat. If routes are close, rerun with all repeats before pruning.

2. 02_competitive_routes_models_v2.yaml
   Extreme-compute-saving r0 supplement with exactly three missing canonical
   cases: Raw InceptionSmall, Feature-vector RBF-SVM and Feature-vector
   ExtraTrees. Reuse the reviewed r0 evidence for Raw CompactCNN, Raw
   InceptionFull 400 Hz and Feature-vector Logistic; do not rerun them in this
   stage. The supplemental report does not import or relabel those earlier
   artifacts, so review the separately identified evidence alongside it.

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
   Five matched Logistic feature-vector cases compare off/off, fixed-threshold
   SQI, SQI plus the frozen Frailty29 all-29 motion bundle, and one-attempt
   PCA/FastICA recovery. The classifier stays on CPU; frozen motion inference
   uses CUDA. The all-29 detector is explicitly in-sample auxiliary evidence on
   Frailty29 and is never described as outer-OOF motion evidence.

6. 06_sequential_single_factor_ablation_v2.yaml
   This is a reusable one-axis template. Replace base_config with the selected
   finalist resolved config. Run LR, write the selected value into a new locked
   base, switch the only axis to batch size, then repeat for epochs. Change the
   study_id each time so each output remains separately archived.
   Classical finalists use their own single factors instead: Logistic C; SVM C
   then gamma; ExtraTrees max_features then min_samples_leaf. Feature-matrix
   model-specific axes remain pending after retirement of ROCKET/Ridge.
   If more than one parallel route remains, copy the template and run one
   within-route comparison per locked base; do not mix routes in one axis.

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

Run the three-case Stage 2 supplement:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/02_competitive_routes_models_v2.yaml

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

When file-level OOF is present, the report always includes both balanced
accuracy views: the declared canonical Line B result and Line A equal-file
post-hoc reaggregation of the same held-out file probabilities. Line A is
aggregation sensitivity only, not separately trained evidence and never a
selection or leaderboard result. An explicit --no-report skips report creation
for that run; the standalone report command can add it later without training.

## Default budgets

- Stage 1: 4 cases, 20 outer cells by default; 100 only if manually escalated.
- Stage 2: 3 supplemental cases, repeat 0 with all five folds, 15 outer cells.
  The reused CompactCNN, InceptionFull and Logistic r0 evidence is not counted
  again.
- Current Stage 3 centered star: exactly 16 cases, 400 fits and 4000
  model-epochs; no Phase 0 execution.
- Stage 3 v3 CompactCNN follow-up: exactly 2 cases, 50 fits and 500
  model-epochs; no Phase 0 execution.
- Preserved historical Stage 3 cumulative chain: exactly 9 cases, 45 fits and
  450 model-epochs. Its optional advisory Phase 0 audit adds no fits or
  model-epochs and does not affect that historical budget.
- Stage 4: 2 scientific cases, 50 outer cells and 150 fitted networks.
- Stage 5: 5 matched cases, repeat 0 and all five folds, 25 outer cells.
- Stage 6: 3 cases and 75 outer cells for each active factor.
- Stage last (ShapeFormer): 1, then 5, then 25 outer cells.

Every study writes to artifacts/studies/static_line_b_staged_v2 under its own
timestamped study directory. The existing non-overwrite and atomic publication
behavior remains active.
