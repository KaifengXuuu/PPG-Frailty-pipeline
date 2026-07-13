# Sweep Analysis Report

- Input sweep directories:
  - `results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2`
- Requested model filters: `inceptiontime, small_inceptiontime`
- Matched `model` values: `inceptiontime, small_inceptiontime`
- Matched `resolved_model` values: `inception_time, small_inception_time`
- Expected repeats per config: `per-source inferred (20260630_0630_overfitting_sweep_generalization_rank2:5)`
- Model-filtered runs: `1160`
- Configs: `232`
- Complete configs: `232`

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
| 1 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | ref_20260625_top1_s1_122 | 20260625_s1_122_no_early_stop_fixed_reference | 20260625 | s1_122 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.6231 | 0.6146 | 0.6061 | 0.5250 | 0.5543 | 0.4609 | 0.6032 | 5 |
| 2 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | ref_20260625_top2_s1_102 | 20260625_s1_102_no_early_stop_fixed_reference | 20260625 | s1_102 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.6176 | 0.6081 | 0.5791 | 0.5250 | 0.5484 | 0.4644 | 0.6070 | 5 |
| 3 | small_inceptiontime | small_inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_212 | generalization_small_inceptiontime_ep15_wd0.01_do0.5_ls0.3_sqitop50_quality_samplernone_wpsall_trainov30 | generalization_grid | wd0.01_do0.5_ls0.3 | 15 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5815 | 0.5702 | 0.5123 | 0.4500 | 0.5067 | 0.4605 | 0.4220 | 5 |
| 4 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_080 | generalization_inceptiontime_ep15_wd0.005_do0.5_ls0.2_sqitop50_quality_samplerclass_subject_balanced_wps50pct_trainov30 | generalization_grid | wd0.005_do0.5_ls0.2 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5806 | 0.5693 | 0.5345 | 0.5250 | 0.5244 | 0.4915 | 0.6708 | 5 |
| 5 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_030 | generalization_inceptiontime_ep10_wd0.01_do0.5_ls0.3_sqinone_samplernone_wpsall_trainov30 | generalization_grid | wd0.01_do0.5_ls0.3 | 10 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5769 | 0.5603 | 0.5056 | 0.4667 | 0.5010 | 0.4753 | 0.4562 | 5 |
| 6 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_066 | generalization_inceptiontime_ep15_wd0.005_do0.5_ls0.2_sqinone_samplerclass_subject_balanced_wps50pct_trainov30 | generalization_grid | wd0.005_do0.5_ls0.2 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5769 | 0.5606 | 0.4665 | 0.4667 | 0.4896 | 0.4700 | 0.6211 | 5 |
| 7 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_074 | generalization_inceptiontime_ep15_wd0.005_do0.5_ls0.2_sqitop50_quality_samplersubject_balanced_wps50pct_trainov30 | generalization_grid | wd0.005_do0.5_ls0.2 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5759 | 0.5585 | 0.4444 | 0.4500 | 0.4646 | 0.4798 | 0.6729 | 5 |
| 8 | small_inceptiontime | small_inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_136 | generalization_small_inceptiontime_ep10_wd0.005_do0.5_ls0.2_sqitop50_quality_samplerclass_subject_balanced_wps50pct_trainov30 | generalization_grid | wd0.005_do0.5_ls0.2 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5667 | 0.5560 | 0.4660 | 0.4000 | 0.4364 | 0.4321 | 0.6281 | 5 |
| 9 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_044 | generalization_inceptiontime_ep10_wd0.01_do0.5_ls0.3_sqitop50_quality_samplernone_wpsall_trainov30 | generalization_grid | wd0.01_do0.5_ls0.3 | 10 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5667 | 0.5549 | 0.4706 | 0.5000 | 0.5425 | 0.4788 | 0.4535 | 5 |
| 10 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_093 | generalization_inceptiontime_ep15_wd0.01_do0.5_ls0.3_sqinone_samplerclass_subject_balanced_wps50pct_trainov00 | generalization_grid | wd0.01_do0.5_ls0.3 | 15 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5667 | 0.5355 | 0.4730 | 0.4333 | 0.4410 | 0.4543 | 0.4810 | 5 |
| 11 | small_inceptiontime | small_inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_164 | generalization_small_inceptiontime_ep10_wd0.01_do0.5_ls0.3_sqitop50_quality_samplerclass_subject_balanced_wps50pct_trainov30 | generalization_grid | wd0.01_do0.5_ls0.3 | 10 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5657 | 0.5520 | 0.4920 | 0.4250 | 0.4506 | 0.4328 | 0.4490 | 5 |
| 12 | small_inceptiontime | small_inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_121 | generalization_small_inceptiontime_ep10_wd0.005_do0.5_ls0.2_sqinone_samplerclass_subject_balanced_wps50pct_trainov00 | generalization_grid | wd0.005_do0.5_ls0.2 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5648 | 0.5562 | 0.5112 | 0.4500 | 0.4698 | 0.4352 | 0.5973 | 5 |
| 13 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | ref_20260608_top2_s1_091 | 20260608_s1_091_no_early_stop_fixed_reference | 20260608 | s1_091 | 10 | 0 | 0.0010 | 0.0005 | 0.7000 | 0.1000 | 5.0000 | 50.0000 | 0.9000 | 0.5648 | 0.5467 | 0.4958 | 0.4333 | 0.4813 | 0.4715 | 0.9213 | 5 |
| 14 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_024 | generalization_inceptiontime_ep10_wd0.005_do0.5_ls0.2_sqitop50_quality_samplerclass_subject_balanced_wps50pct_trainov30 | generalization_grid | wd0.005_do0.5_ls0.2 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5648 | 0.5476 | 0.4994 | 0.4000 | 0.4462 | 0.4736 | 0.6623 | 5 |
| 15 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_094 | generalization_inceptiontime_ep15_wd0.01_do0.5_ls0.3_sqinone_samplerclass_subject_balanced_wps50pct_trainov30 | generalization_grid | wd0.01_do0.5_ls0.3 | 15 | 0 | 0.0010 | 0.0100 | 0.5000 | 0.3000 | 5.0000 | 50.0000 | 0.9000 | 0.5630 | 0.5463 | 0.4637 | 0.4500 | 0.4702 | 0.4702 | 0.4591 | 5 |
| 16 | small_inceptiontime | small_inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_116 | generalization_small_inceptiontime_ep10_wd0.005_do0.5_ls0.2_sqinone_samplersubject_balanced_wps50pct_trainov30 | generalization_grid | wd0.005_do0.5_ls0.2 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5630 | 0.5584 | 0.4777 | 0.4000 | 0.4431 | 0.4451 | 0.5729 | 5 |
| 17 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_065 | generalization_inceptiontime_ep15_wd0.005_do0.5_ls0.2_sqinone_samplerclass_subject_balanced_wps50pct_trainov00 | generalization_grid | wd0.005_do0.5_ls0.2 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5630 | 0.5521 | 0.4454 | 0.5000 | 0.4998 | 0.4596 | 0.6699 | 5 |
| 18 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | ref_20260608_top3_s1_105 | 20260608_s1_105_no_early_stop_fixed_reference | 20260608 | s1_105 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5602 | 0.5511 | 0.5067 | 0.4750 | 0.5034 | 0.4734 | 0.6498 | 5 |
| 19 | inceptiontime | inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_001 | generalization_inceptiontime_ep10_wd0.005_do0.5_ls0.2_sqinone_samplernone_wpsall_trainov00 | generalization_grid | wd0.005_do0.5_ls0.2 | 10 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5593 | 0.5486 | 0.4908 | 0.4667 | 0.5082 | 0.4671 | 0.6533 | 5 |
| 20 | small_inceptiontime | small_inception_time | 20260630_0630_overfitting_sweep_generalization_rank2 | overfitting | 0 | gen_177 | generalization_small_inceptiontime_ep15_wd0.005_do0.5_ls0.2_sqinone_samplerclass_subject_balanced_wps50pct_trainov00 | generalization_grid | wd0.005_do0.5_ls0.2 | 15 | 0 | 0.0010 | 0.0050 | 0.5000 | 0.2000 | 5.0000 | 50.0000 | 0.9000 | 0.5583 | 0.5522 | 0.4265 | 0.4750 | 0.4874 | 0.4481 | 0.6035 | 5 |

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
