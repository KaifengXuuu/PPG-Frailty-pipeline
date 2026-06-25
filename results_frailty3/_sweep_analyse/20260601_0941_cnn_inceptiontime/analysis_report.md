# Sweep Analysis Report

- Input sweep directory: `results_frailty3/20260527_1320_cnn_inceptionTime`
- Models analyzed: `cnn, inceptiontime`
- Expected repeats per config: `5`
- Model-filtered runs: `360`
- Configs: `72`
- Complete configs: `72`

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
| 1 | inceptiontime | 0 | 50 | 20 | 5.0000 | 30.0000 | 0.7370 | 0.7233 | 0.6767 | 0.5667 | 0.6529 | 5 |
| 2 | inceptiontime | 0 | 50 | 20 | 5.0000 | 50.0000 | 0.7269 | 0.7235 | 0.6520 | 0.6333 | 0.6607 | 5 |
| 3 | inceptiontime | 0 | 50 | 10 | 5.0000 | 30.0000 | 0.7130 | 0.7116 | 0.6473 | 0.6333 | 0.6522 | 5 |
| 4 | inceptiontime | 0 | 50 | 10 | 5.0000 | 50.0000 | 0.7111 | 0.7024 | 0.6557 | 0.5833 | 0.6201 | 5 |
| 5 | inceptiontime | 0 | 50 | 20 | 10.0000 | 30.0000 | 0.7083 | 0.6883 | 0.6303 | 0.5000 | 0.5729 | 5 |
| 6 | cnn | PPI | 50 | 20 | 10.0000 | 50.0000 | 0.7056 | 0.6930 | 0.5914 | 0.5667 | 0.6164 | 5 |
| 7 | cnn | 0 | 50 | 20 | 5.0000 | 50.0000 | 0.7028 | 0.7075 | 0.6466 | 0.6500 | 0.6440 | 5 |
| 8 | inceptiontime | 0 | 50 | 20 | 15.0000 | 50.0000 | 0.7009 | 0.7043 | 0.6139 | 0.6167 | 0.6238 | 5 |
| 9 | inceptiontime | PPI | 50 | 20 | 5.0000 | 30.0000 | 0.6963 | 0.6834 | 0.5867 | 0.5333 | 0.6100 | 5 |
| 10 | cnn | PPI | 50 | 20 | 15.0000 | 50.0000 | 0.6935 | 0.6819 | 0.6493 | 0.5333 | 0.5934 | 5 |

## Output Files

- `clean_runs.csv`: filtered run-level table with resolved artifact paths and class-level metrics.
- `config_summary.csv`: repeat-aggregated config-level summary.
- `leaderboard_top_configs.csv`: top complete configs after config-level ranking.
- `leaderboard_top10_worst_class_f1_stability.csv`: top 10 resorted by worst-class F1 and repeat std.
- `incomplete_configs.csv`: configs that did not complete all expected repeats.
- `class_level_summary.csv`: class-level precision/recall/F1/support by config.
- `top_config_confusion_matrices_long.csv`: aggregated confusion matrices for top configs.
- `figures/`: leaderboard, boxplots, heatmaps, learning curves, and confusion matrix figures.
