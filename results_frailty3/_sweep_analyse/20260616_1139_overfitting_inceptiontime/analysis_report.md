# Sweep Analysis Report

- Input sweep directories:
  - `results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2`
- Requested model filters: `inceptiontime`
- Matched `model` values: `inceptiontime`
- Matched `resolved_model` values: `inception_time`
- Expected repeats per config: `per-source inferred (20260608_1206_overfitting_sweep_stage1_rank2:5)`
- Model-filtered runs: `930`
- Configs: `186`
- Complete configs: `186`

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
| 1 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_085 | main_effect_dropout-0p0_lr0.001_wd0.0005_do0_ls0.1_mw90_ep10 | dropout | 0.0 | 10 | 0 | 0.0010 | 0.0005 | 0.0000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6231 | 0.6258 | 0.5363 | 0.5500 | 0.5398 | 0.4399 | 0.9142 | 5 |
| 2 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_091 | main_effect_dropout-0p7_lr0.001_wd0.0005_do0.7_ls0.1_mw90_ep10 | dropout | 0.7 | 10 | 0 | 0.0010 | 0.0005 | 0.7000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6213 | 0.6219 | 0.5579 | 0.5667 | 0.5499 | 0.4241 | 0.8684 | 5 |
| 3 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_105 | strong_combo_combined_regularization-wd0p005_do0p5_ls0p2_mw0p9_lr0.001_wd0.005_do0.5_ls0.2_mw90_ep10 | combined_regularization | wd0.005_do0.5_ls0.2_mw0.9 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.6157 | 0.6254 | 0.5689 | 0.5556 | 0.5818 | 0.4354 | 0.6243 | 5 |
| 4 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_163 | main_effect_dropout-0p5_lr0.001_wd0.0005_do0.5_ls0.1_mw90_ep15 | dropout | 0.5 | 15 | 0 | 0.0010 | 0.0005 | 0.5000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6120 | 0.6168 | 0.5740 | 0.5778 | 0.5668 | 0.4310 | 0.9278 | 5 |
| 5 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_079 | main_effect_weight_decay-0p0001_lr0.001_wd0.0001_do0.2_ls0.1_mw90_ep10 | weight_decay | 0.0001 | 10 | 0 | 0.0010 | 0.0001 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6065 | 0.6126 | 0.5562 | 0.5778 | 0.5688 | 0.4427 | 0.8948 | 5 |
| 6 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_075 | main_effect_baseline-base_lr0.001_wd0.0005_do0.2_ls0.1_mw90_ep10 | baseline | base | 10 | 0 | 0.0010 | 0.0005 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6046 | 0.6124 | 0.5383 | 0.5556 | 0.5482 | 0.4367 | 0.8562 | 5 |
| 7 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_077 | main_effect_weight_decay-1em05_lr0.001_wd1e-05_do0.2_ls0.1_mw90_ep10 | weight_decay | 1e-05 | 10 | 0 | 0.0010 | 0.0000 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6037 | 0.6118 | 0.5279 | 0.5778 | 0.5549 | 0.4400 | 0.8679 | 5 |
| 8 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_165 | main_effect_dropout-0p7_lr0.001_wd0.0005_do0.7_ls0.1_mw90_ep15 | dropout | 0.7 | 15 | 0 | 0.0010 | 0.0005 | 0.7000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6037 | 0.6002 | 0.5548 | 0.5333 | 0.5381 | 0.4310 | 0.8995 | 5 |
| 9 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_099 | main_effect_max_windows_fraction-0p7_lr0.001_wd0.0005_do0.2_ls0.1_mw70_ep10 | max_windows_fraction | 0.7 | 10 | 0 | 0.0010 | 0.0005 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.7000 | 0.6028 | 0.6099 | 0.4980 | 0.5333 | 0.5263 | 0.4336 | 0.8755 | 5 |
| 10 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_183 | strong_combo_combined_regularization-wd0p01_do0p6_ls0p3_mw0p7_lr0.001_wd0.01_do0.6_ls0.3_mw70_ep15 | combined_regularization | wd0.01_do0.6_ls0.3_mw0.7 | 15 | 0 | 0.0010 | 0.0100 | 0.6000 | 0.3000 | 5.0000 | 50.0000 | 0.7000 | 0.6028 | 0.5991 | 0.5460 | 0.4833 | 0.5110 | 0.4356 | 0.4455 | 5 |
| 11 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_181 | strong_combo_combined_regularization-wd0p01_do0p6_ls0p3_mw0p9_lr0.001_wd0.01_do0.6_ls0.3_mw90_ep15 | combined_regularization | wd0.01_do0.6_ls0.3_mw0.9 | 15 | 0 | 0.0010 | 0.0100 | 0.6000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.6028 | 0.5927 | 0.5261 | 0.4333 | 0.4814 | 0.4290 | 0.4532 | 5 |
| 12 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_160 | main_effect_dropout-0p1_lr0.001_wd0.0005_do0.1_ls0.1_mw90_ep15 | dropout | 0.1 | 15 | 0 | 0.0010 | 0.0005 | 0.1000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6009 | 0.6070 | 0.5493 | 0.5500 | 0.5269 | 0.4461 | 0.9175 | 5 |
| 13 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_178 | strong_combo_combined_regularization-wd0p005_do0p4_ls0p15_mw0p9_lr0.001_wd0.005_do0.4_ls0.15_mw90_ep15 | combined_regularization | wd0.005_do0.4_ls0.15_mw0.9 | 15 | 0 | 0.0010 | 0.0050 | 0.4000 | 0.1500 | 5.0000 | 50.0000 | 0.9000 | 0.6009 | 0.5990 | 0.5499 | 0.5333 | 0.5329 | 0.4515 | 0.7782 | 5 |
| 14 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_148 | strong_combo_combined_regularization-wd0p01_do0p6_ls0p3_mw0p9_lr0.0005_wd0.01_do0.6_ls0.3_mw90_ep12 | combined_regularization | wd0.01_do0.6_ls0.3_mw0.9 | 12 | 0 | 0.0005 | 0.0100 | 0.6000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.6009 | 0.5863 | 0.5015 | 0.4333 | 0.4895 | 0.4318 | 0.4690 | 5 |
| 15 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_158 | main_effect_weight_decay-0p02_lr0.001_wd0.02_do0.2_ls0.1_mw90_ep15 | weight_decay | 0.02 | 15 | 0 | 0.0010 | 0.0200 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.6000 | 0.6081 | 0.5126 | 0.5333 | 0.5315 | 0.4302 | 0.9102 | 5 |
| 16 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_185 | strong_combo_combined_regularization-wd0p01_do0p6_ls0p3_mw0p9_lr0.0005_wd0.01_do0.6_ls0.3_mw90_ep15 | combined_regularization | wd0.01_do0.6_ls0.3_mw0.9 | 15 | 0 | 0.0005 | 0.0100 | 0.6000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5981 | 0.5874 | 0.5204 | 0.4667 | 0.4912 | 0.4291 | 0.4561 | 5 |
| 17 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_034 | strong_combo_combined_regularization-wd0p005_do0p5_ls0p2_mw0p7_lr0.001_wd0.005_do0.5_ls0.2_mw70_ep5 | combined_regularization | wd0.005_do0.5_ls0.2_mw0.7 | 5 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.7000 | 0.5963 | 0.6063 | 0.5492 | 0.5556 | 0.5322 | 0.4352 | 0.6014 | 5 |
| 18 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_112 | main_effect_baseline-base_lr0.001_wd0.0005_do0.2_ls0.1_mw90_ep12 | baseline | base | 12 | 0 | 0.0010 | 0.0005 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.5963 | 0.5999 | 0.5145 | 0.4889 | 0.4913 | 0.4399 | 0.9453 | 5 |
| 19 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_157 | main_effect_weight_decay-0p01_lr0.001_wd0.01_do0.2_ls0.1_mw90_ep15 | weight_decay | 0.01 | 15 | 0 | 0.0010 | 0.0100 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.5963 | 0.5980 | 0.5098 | 0.4889 | 0.4868 | 0.4399 | 0.9446 | 5 |
| 20 | inceptiontime | inception_time | 20260608_1206_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_119 | main_effect_weight_decay-0p005_lr0.001_wd0.005_do0.2_ls0.1_mw90_ep12 | weight_decay | 0.005 | 12 | 0 | 0.0010 | 0.0050 | 0.2000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.5963 | 0.5968 | 0.4832 | 0.5333 | 0.5308 | 0.4481 | 0.9368 | 5 |

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
