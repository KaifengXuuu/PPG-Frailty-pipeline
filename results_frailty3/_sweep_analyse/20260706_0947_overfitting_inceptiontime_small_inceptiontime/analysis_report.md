# Sweep Analysis Report

- Input sweep directories:
  - `results_frailty3/_overfitting_sweep/overfitting_20260608_0752`
- Requested model filters: `inceptiontime, small_inceptiontime`
- Matched `model` values: `inceptiontime`
- Matched `resolved_model` values: `inception_time`
- Expected repeats per config: `per-source inferred (overfitting_20260608_0752:10)`
- Model-filtered runs: `65`
- Configs: `7`
- Complete configs: `7`

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

| rank | model | resolved_model | source_sweep_names | source_sweep_kinds | extra_input | overfit_config_id | overfit_config_name | stage1_regularization_factor | stage1_regularization_value | cnn_epochs | cnn_patience | cnn_lr | cnn_weight_decay | cnn_dropout | cnn_label_smoothing | window_sec | overlap_pct | max_windows_fraction | subject_balanced_accuracy_mean | subject_macro_f1_mean | subject_balanced_accuracy_ci95_low | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean | val_train_loss_gap_mean | n_repeats_done |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | ref_original | original_rank_parameters_static_only |  |  | 50 | 20 | 0.0010 | 0.0001 | -1.0000 | 0.0000 | 5.0000 | 50.0000 | 0.9000 | 0.6333 | 0.5956 | 0.4602 | 0.4000 | 0.3667 | 0.3232 | 1.7571 | 5 |
| 2 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | s1_003 | lr5e-4_wd1e-3_do0p2_ls0p05_mw90 |  |  | 50 | 12 | 0.0005 | 0.0010 | 0.2000 | 0.0500 | 5.0000 | 50.0000 | 0.9000 | 0.6333 | 0.5989 | 0.4764 | 0.4500 | 0.4267 | 0.3231 | 0.8204 | 10 |
| 3 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | s1_005 | lr5e-4_wd5e-4_do0p2_ls0p10_mw90 |  |  | 50 | 12 | 0.0005 | 0.0005 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6167 | 0.5678 | 0.4904 | 0.5000 | 0.4333 | 0.3271 | 0.6972 | 10 |
| 4 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | s1_004 | lr5e-4_wd5e-4_do0p4_ls0p05_mw90 |  |  | 50 | 12 | 0.0005 | 0.0005 | 0.4000 | 0.0500 | 5.0000 | 50.0000 | 0.9000 | 0.5667 | 0.5278 | 0.4661 | 0.3000 | 0.2967 | 0.3319 | 0.8993 | 10 |
| 5 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | s1_006 | lr5e-4_wd5e-4_do0p2_ls0p05_mw50 |  |  | 50 | 12 | 0.0005 | 0.0005 | 0.2000 | 0.0500 | 5.0000 | 50.0000 | 0.5000 | 0.5500 | 0.4967 | 0.4237 | 0.2500 | 0.2567 | 0.3194 | 0.9486 | 10 |
| 6 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | s1_002 | lr5e-4_wd5e-4_do0p2_ls0p05_mw90 |  |  | 50 | 12 | 0.0005 | 0.0005 | 0.2000 | 0.0500 | 5.0000 | 50.0000 | 0.9000 | 0.5500 | 0.5124 | 0.4369 | 0.3500 | 0.3567 | 0.3270 | 0.8676 | 10 |
| 7 | inceptiontime | inception_time | overfitting_20260608_0752 | overfitting | 0 | s1_001 | lr5e-4_wd1e-4_do0_ls0_mw90 |  |  | 50 | 12 | 0.0005 | 0.0001 | 0.0000 | 0.0000 | 5.0000 | 50.0000 | 0.9000 | 0.4667 | 0.4224 | 0.3571 | 0.2500 | 0.2233 | 0.3326 | 2.1021 | 10 |

## Output Files

- `clean_runs.csv`: filtered run-level table with resolved artifact paths and class-level metrics.
- `config_summary.csv`: repeat-aggregated config-level summary.
- `leaderboard_top_configs.csv`: top complete configs after config-level ranking.
- `leaderboard_top10_worst_class_f1_stability.csv`: top 10 resorted by worst-class F1 and repeat std.
- `incomplete_configs.csv`: configs that did not complete all expected repeats.
- `class_level_summary.csv`: class-level precision/recall/F1/support by config.
- `top_config_confusion_matrices_long.csv`: aggregated confusion matrices for top configs.
- `figures/`: leaderboard, boxplots, heatmaps, learning curves, and confusion matrix figures.

## Overfitting Sweep Extras

- `overfitting_top_configs_with_reference.csv`: top ranked configs plus reference configs.
- `overfitting_epoch_summary.csv`: config-level summary grouped by fixed epoch.
- `overfitting_factor_summary.csv`: config-level summary grouped by regularization factor.
- `overfitting_factor_value_summary.csv`: config-level summary grouped by regularization factor and value.
- `overfitting_factor_epoch_summary.csv`: config-level summary grouped by factor and epoch.
- `overfitting_regularization_grid_summary.csv`: config-level summary grouped by weight decay, dropout, label smoothing, max windows fraction, and epoch.
