# Sweep Analysis Report

- Input sweep directories:
  - `results_frailty3/_overfitting_sweep/20260625_2320_overfitting_sweep_stage1_rank2`
- Requested model filters: `inceptiontime`
- Matched `model` values: `inceptiontime`
- Matched `resolved_model` values: `inception_time`
- Expected repeats per config: `per-source inferred (20260625_2320_overfitting_sweep_stage1_rank2:5)`
- Model-filtered runs: `645`
- Configs: `129`
- Complete configs: `129`

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
| 1 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_122 | quality_combo_sqi_aggregation-top50_quality_quality_weighted_mean_lr0.001_wd0.005_do0.5_ls0.2_mw90_sqitop50_quality_aggquality_weighted_mean_mannone_lossweighted_ce_cwinverse_subject_count_ep15 | sqi_aggregation | top50_quality_quality_weighted_mean | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.6102 | 0.6040 | 0.5339 | 0.4750 | 0.5092 | 0.4633 | 0.6058 | 5 |
| 2 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_102 | main_effect_sqi_mode-top50_quality_lr0.001_wd0.005_do0.5_ls0.2_mw90_sqitop50_quality_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep15 | sqi_mode | top50_quality | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.6102 | 0.6012 | 0.5844 | 0.5250 | 0.5467 | 0.4615 | 0.6054 | 5 |
| 3 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_121 | quality_combo_sqi_aggregation-top70_quality_quality_weighted_mean_lr0.001_wd0.005_do0.5_ls0.2_mw90_sqitop70_quality_aggquality_weighted_mean_mannone_lossweighted_ce_cwinverse_subject_count_ep15 | sqi_aggregation | top70_quality_quality_weighted_mean | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5926 | 0.5873 | 0.4902 | 0.5000 | 0.5587 | 0.4798 | 0.6334 | 5 |
| 4 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_055 | main_effect_label_smoothing-0p3_lr0.001_wd0.005_do0.5_ls0.3_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | label_smoothing | 0.3 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5796 | 0.5668 | 0.5128 | 0.5000 | 0.5329 | 0.4745 | 0.4685 | 5 |
| 5 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_075 | focused_combo_combined_regularization-wd0p01_do0p5_ls0p3_mw0p9_lr0.001_wd0.01_do0.5_ls0.3_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | combined_regularization | wd0.01_do0.5_ls0.3_mw0.9 | 10 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5796 | 0.5687 | 0.5101 | 0.5000 | 0.5244 | 0.4805 | 0.4804 | 5 |
| 6 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_061 | main_effect_aggregation-quality_weighted_mean_lr0.001_wd0.005_do0.5_ls0.2_mw90_sqinone_aggquality_weighted_mean_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | aggregation | quality_weighted_mean | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5778 | 0.5755 | 0.5066 | 0.5000 | 0.5460 | 0.4740 | 0.6491 | 5 |
| 7 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_054 | main_effect_label_smoothing-0p15_lr0.001_wd0.005_do0.5_ls0.15_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | label_smoothing | 0.15 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.1500 | 5.0000 | 50.0000 | 0.9000 | 0.5750 | 0.5633 | 0.5002 | 0.4667 | 0.5095 | 0.4748 | 0.7779 | 5 |
| 8 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_033 | focused_combo_combined_regularization-wd0p01_do0p5_ls0p3_mw0p9_lr0.001_wd0.01_do0.5_ls0.3_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep9 | combined_regularization | wd0.01_do0.5_ls0.3_mw0.9 | 9 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5731 | 0.5555 | 0.5195 | 0.4333 | 0.4731 | 0.4695 | 0.4609 | 5 |
| 9 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_099 | main_effect_max_windows_fraction-0p7_lr0.001_wd0.005_do0.5_ls0.2_mw70_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep15 | max_windows_fraction | 0.7 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.7000 | 0.5731 | 0.5628 | 0.5031 | 0.4833 | 0.5102 | 0.4672 | 0.6314 | 5 |
| 10 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_067 | focused_combo_combined_regularization-wd0p005_do0p4_ls0p15_mw0p9_lr0.001_wd0.005_do0.4_ls0.15_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | combined_regularization | wd0.005_do0.4_ls0.15_mw0.9 | 10 | 0 | 0.0010 | 0.0050 | 0.4000 | 0.1500 | 5.0000 | 50.0000 | 0.9000 | 0.5722 | 0.5549 | 0.5157 | 0.4333 | 0.4877 | 0.4785 | 0.7856 | 5 |
| 11 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_065 | main_effect_loss_type-focal_loss_lr0.001_wd0.005_do0.5_ls0.2_mw90_sqinone_aggmean_prob_mannone_lossfocal_loss_cwinverse_subject_count_ep10 | loss_type | focal_loss | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5704 | 0.5704 | 0.5442 | 0.4500 | 0.5057 | 0.4917 | 1.3366 | 5 |
| 12 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_003 | main_effect_weight_decay-0p002_lr0.001_wd0.002_do0.5_ls0.2_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep9 | weight_decay | 0.002 | 9 | 0 | 0.0010 | 0.0020 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5704 | 0.5607 | 0.4715 | 0.4500 | 0.4986 | 0.4842 | 0.6631 | 5 |
| 13 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_090 | main_effect_dropout-0p3_lr0.001_wd0.005_do0.3_ls0.2_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep15 | dropout | 0.3 | 15 | 0 | 0.0010 | 0.0050 | 0.3000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5694 | 0.5469 | 0.5104 | 0.3833 | 0.4537 | 0.4674 | 0.6729 | 5 |
| 14 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_074 | focused_combo_combined_regularization-wd0p01_do0p5_ls0p15_mw0p9_lr0.001_wd0.01_do0.5_ls0.15_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | combined_regularization | wd0.01_do0.5_ls0.15_mw0.9 | 10 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.1500 | 5.0000 | 50.0000 | 0.9000 | 0.5676 | 0.5577 | 0.4755 | 0.4667 | 0.4960 | 0.4719 | 0.7607 | 5 |
| 15 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_079 | quality_combo_sqi_aggregation-top70_quality_quality_weighted_mean_lr0.001_wd0.005_do0.5_ls0.2_mw90_sqitop70_quality_aggquality_weighted_mean_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | sqi_aggregation | top70_quality_quality_weighted_mean | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5667 | 0.5578 | 0.4550 | 0.4833 | 0.5318 | 0.4785 | 0.6374 | 5 |
| 16 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_057 | main_effect_max_windows_fraction-0p7_lr0.001_wd0.005_do0.5_ls0.2_mw70_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | max_windows_fraction | 0.7 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.7000 | 0.5657 | 0.5529 | 0.4551 | 0.4667 | 0.4975 | 0.4679 | 0.6314 | 5 |
| 17 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_002 | main_effect_weight_decay-0p001_lr0.001_wd0.001_do0.5_ls0.2_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep9 | weight_decay | 0.001 | 9 | 0 | 0.0010 | 0.0010 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5657 | 0.5590 | 0.4923 | 0.4833 | 0.5058 | 0.4893 | 0.6691 | 5 |
| 18 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_068 | focused_combo_combined_regularization-wd0p005_do0p4_ls0p3_mw0p9_lr0.001_wd0.005_do0.4_ls0.3_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | combined_regularization | wd0.005_do0.4_ls0.3_mw0.9 | 10 | 0 | 0.0010 | 0.0050 | 0.4000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5657 | 0.5487 | 0.5288 | 0.4333 | 0.5066 | 0.4709 | 0.4686 | 5 |
| 19 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_051 | main_effect_dropout-0p7_lr0.001_wd0.005_do0.7_ls0.2_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | dropout | 0.7 | 10 | 0 | 0.0010 | 0.0050 | 0.7000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5648 | 0.5526 | 0.4827 | 0.4500 | 0.4733 | 0.4764 | 0.6099 | 5 |
| 20 | inceptiontime | inception_time | 20260625_2320_overfitting_sweep_stage1_rank2 | overfitting | 0 | s1_052 | main_effect_label_smoothing-0p05_lr0.001_wd0.005_do0.5_ls0.05_mw90_sqinone_aggmean_prob_mannone_lossweighted_ce_cwinverse_subject_count_ep10 | label_smoothing | 0.05 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.0500 | 5.0000 | 50.0000 | 0.9000 | 0.5639 | 0.5495 | 0.5226 | 0.4500 | 0.4880 | 0.4750 | 1.2162 | 5 |

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
