# V2 study commands

Run these commands from the final_pipeline_v2 directory. They resolve ordinary
pipeline YAML files and delegate every case to the canonical experiment runner;
the study layer does not reimplement any signal, feature, model, or evaluation
algorithm.

## One complete configuration

    python frailty_3class_pipeline_v2.py \
      --config configs/reference_static_role_aware_v2.yaml \
      --study-id compactcnn_role_aware_manual

The equivalent reusable-plan command is:

    python frailty_3class_sweep_v2.py run \
      --plan configs/studies/single_config_v2.yaml

## Single-factor ablation

    python frailty_3class_sweep_v2.py ablation \
      --base-config configs/reference_static_role_aware_v2.yaml \
      --factor training.fixed_epochs \
      --values 7 10 15 \
      --reference-value 10 \
      --study-id compactcnn_fixed_epochs \
      --purpose "Compare only the fixed-epoch capacity control." \
      --flow-position "Training-capacity ablation before manual candidate review."

## Cartesian grid

Quotes around list assignments prevent the shell from interpreting brackets.

    python frailty_3class_sweep_v2.py grid \
      --base-config configs/reference_static_role_aware_v2.yaml \
      --vary 'training.learning_rate=[0.0003,0.001]' \
      --vary 'training.weight_decay=[0.0001,0.001]' \
      --reference training.learning_rate=0.001 \
      --reference training.weight_decay=0.0001 \
      --study-id compactcnn_optimizer_grid \
      --purpose "Descriptive optimizer grid screening." \
      --flow-position "Screening before single-factor confirmation."

## Parallelism, resume, and output root

The --jobs option means parallel cases only. It does not start nested fold/model
worker pools. Deep-model plans default to effective jobs=1; CPU/classical case
grids may use a larger value.

    python frailty_3class_sweep_v2.py run \
      --plan configs/studies/grid_optimizer_v2.yaml \
      --jobs 4 \
      --output-root /path/to/study_archive

Resume with the same plan and the exact existing study directory:

    python frailty_3class_sweep_v2.py run \
      --plan configs/studies/grid_optimizer_v2.yaml \
      --resume /path/to/study_archive/20260817_120000_grid_compactcnn-optimizer-grid-v2

Passed cases are skipped; failed/incomplete cases get a new attempt record.
Generate or refresh the report without training:

    python frailty_3class_sweep_v2.py report \
      --study-dir /path/to/existing/study

Each new run creates a dated folder containing resolved configs, case attempts,
structured progress JSONL, CSV/JSON tables, Markdown/HTML summaries, generated
figures, N/A markers for unavailable views, and outputs_index.json.

## Seed and ensemble boundary

These plans use a one-member model. During outer CV, the canonical repeat seed
schedule is [42, 10042, 20042, 30042, 40042], with one repeat seed shared by its
five folds. A fixed seed of 42 belongs only to final-use-case refit. Five-member
outer-CV ensembles remain fail-closed until the repeat-by-member seed matrix is
manually frozen.

When jobs is greater than one, the terminal shows case-level refreshed progress;
fine-grained child events remain in each case's executor_events.jsonl.
