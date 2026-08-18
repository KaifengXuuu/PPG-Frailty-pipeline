# Static Line B staged screening flow

This folder is the compute-saving alternative to
configs/studies/static_line_b_all_models_v2.yaml. The original 39-case plan is
unchanged and remains available.

The six files form a manual scientific workflow. Results never auto-select a
winner and no later file silently inherits an earlier result. Read each report,
record the decision, then edit only the documented selector in the next file.

## Order and status

1. 01_representation_baselines_v2.yaml
   Runs four canonical low-cost representatives. It screens a
   representation-plus-model combination; it is not a pure causal
   representation comparison and not a final winner study. The default is one
   complete repeat. If routes are close, rerun with all repeats before pruning.

2. 02_competitive_routes_models_v2.yaml
   Before execution, delete every complete representation block that did not
   advance from Stage 1. The checked-in ten-case file is a valid review
   superset, not an instruction to spend compute on all four routes.

3. 03_shapeformer_stability_v2.yaml
   Default execution is one cell. If stable, rerun one complete repeat, then
   full 5x5 only after another manual review. ShapeFormer failure is isolated
   from ordinary models.

4. 04_selected_inception_ensemble_v2.yaml
   The checked-in route is Raw InceptionFull. The file header gives the exact
   two substitutions for Matrix Inception. Skip this stage when the promoted
   model has no registered matched ensemble.

5. 05_sqi_motion_finalists_v2.yaml
   The runnable part is quality off versus diagnostics_only on example
   finalists. Prune or replace those examples first. diagnostics_only must not
   change predictions. Supervised SQI routing, motion override, denoiser
   efficacy and formal motion 5x5 remain planned and are not disguised as
   executable comparisons.

6. 06_sequential_single_factor_ablation_v2.yaml
   This is a reusable one-axis template. Replace base_config with the selected
   finalist resolved config. Run LR, write the selected value into a new locked
   base, switch the only axis to batch size, then repeat for epochs. Change the
   study_id each time so each output remains separately archived.
   Classical finalists use their own single factors instead: Logistic C; SVM C
   then gamma; ExtraTrees max_features then min_samples_leaf; ROCKET kernels
   then ridge alpha. The YAML header lists the registered sparse values.
   If more than one parallel route remains, copy the template and run one
   within-route comparison per locked base; do not mix routes in one axis.

## Commands

Run or dry-run any stage from the final_pipeline_v2 directory:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml --dry-run

Escalate Stage 1 to full 5x5 only when the one-repeat routes are too close:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/01_representation_baselines_v2.yaml --repeats all --folds all

ShapeFormer stability ladder:

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/03_shapeformer_stability_v2.yaml

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/03_shapeformer_stability_v2.yaml --repeats 0 --folds all

    python3 frailty_3class_sweep_v2.py run --plan configs/studies/static_line_b_staged_v2/03_shapeformer_stability_v2.yaml --repeats all --folds all

## Default budgets

- Stage 1: 4 cases, 20 outer cells by default; 100 only if manually escalated.
- Stage 2: at most 10 cases, 250 outer cells; normally fewer after pruning.
- Stage 3: 1, then 5, then 25 outer cells.
- Stage 4: 2 scientific cases, 50 outer cells and 150 fitted networks.
- Stage 5: at most 8 runnable diagnostic cases, 200 outer cells; normally
  fewer after finalist pruning. Planned motion work is not counted.
- Stage 6: 3 cases and 75 outer cells for each active factor.

Every study writes to artifacts/studies/static_line_b_staged_v2 under its own
timestamped study directory. The existing non-overwrite and atomic publication
behavior remains active.
