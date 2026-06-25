# Sweep Analysis Report

- Input sweep directory: `results_frailty3/20260528_1045_shapeformer_0extra`
- Requested model filters: `shapeformer`
- Matched `model` values: `shapeformer_pisd`
- Matched `resolved_model` values: `shapeformer`
- Expected repeats per config: `5`
- Model-filtered runs: `60`
- Configs: `12`
- Complete configs: `12`

Ranking uses config-level aggregation, not single-run ranking. Runtime is retained as a reference field only.

## Ranking Order

1. `subject_balanced_accuracy_mean` descending
2. `subject_macro_f1_mean` descending
3. `subject_balanced_accuracy_ci95_low` descending
4. `subject_macro_f1_ci95_low` descending
5. `worst_class_recall_mean` descending
6. `worst_class_f1_mean` descending
7. `subject_balanced_accuracy_std` ascending

## Top Configs

| rank | model | extra_input | cnn_epochs | cnn_patience | window_sec | overlap_pct | subject_balanced_accuracy_mean | subject_macro_f1_mean | subject_balanced_accuracy_ci95_low | worst_class_recall_mean | worst_class_f1_mean | n_repeats_done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | shapeformer_pisd | 0 | 50 | 20 | 10.0000 | 50.0000 | 0.6407 | 0.6430 | 0.5900 | 0.6000 | 0.6068 | 5 |
| 2 | shapeformer_pisd | 0 | 50 | 10 | 10.0000 | 30.0000 | 0.6157 | 0.6208 | 0.5561 | 0.5750 | 0.6139 | 5 |
| 3 | shapeformer_pisd | 0 | 50 | 20 | 5.0000 | 50.0000 | 0.6157 | 0.6052 | 0.5593 | 0.5167 | 0.5607 | 5 |
| 4 | shapeformer_pisd | 0 | 50 | 10 | 15.0000 | 50.0000 | 0.6111 | 0.6199 | 0.6070 | 0.5000 | 0.5883 | 5 |
| 5 | shapeformer_pisd | 0 | 50 | 20 | 15.0000 | 50.0000 | 0.6056 | 0.6136 | 0.5454 | 0.5500 | 0.5703 | 5 |
| 6 | shapeformer_pisd | 0 | 50 | 10 | 15.0000 | 30.0000 | 0.6056 | 0.6102 | 0.5443 | 0.5000 | 0.5695 | 5 |
| 7 | shapeformer_pisd | 0 | 50 | 10 | 10.0000 | 50.0000 | 0.6028 | 0.6123 | 0.5494 | 0.5250 | 0.6081 | 5 |
| 8 | shapeformer_pisd | 0 | 50 | 20 | 10.0000 | 30.0000 | 0.6000 | 0.6011 | 0.5150 | 0.5500 | 0.5826 | 5 |
| 9 | shapeformer_pisd | 0 | 50 | 10 | 5.0000 | 30.0000 | 0.5926 | 0.5948 | 0.5475 | 0.5500 | 0.5695 | 5 |
| 10 | shapeformer_pisd | 0 | 50 | 10 | 5.0000 | 50.0000 | 0.5870 | 0.5787 | 0.5662 | 0.5000 | 0.5340 | 5 |
| 11 | shapeformer_pisd | 0 | 50 | 20 | 15.0000 | 30.0000 | 0.5787 | 0.5844 | 0.5205 | 0.5250 | 0.5437 | 5 |
| 12 | shapeformer_pisd | 0 | 50 | 20 | 5.0000 | 30.0000 | 0.5778 | 0.5802 | 0.5216 | 0.5500 | 0.5465 | 5 |

## Output Files

- `clean_runs.csv`: filtered run-level table with resolved artifact paths and class-level metrics.
- `config_summary.csv`: repeat-aggregated config-level summary.
- `leaderboard_top_configs.csv`: top complete configs after config-level ranking.
- `leaderboard_top10_worst_class_f1_stability.csv`: top 10 resorted by worst-class F1 and repeat std.
- `incomplete_configs.csv`: configs that did not complete all expected repeats.
- `class_level_summary.csv`: class-level precision/recall/F1/support by config.
- `top_config_confusion_matrices_long.csv`: aggregated confusion matrices for top configs.
- `figures/`: leaderboard, boxplots, heatmaps, learning curves, and confusion matrix figures.
