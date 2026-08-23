# staged_static_06_batch_lr_successive_halving_v1

Tune batch size and learning rate for InceptionTimeFull using a reduced-resource first rung and complete 5x5 tuning CV for promoted configurations.

## Selection outcome

Selected tuning configuration: `b16_lr3e-4`.

Selection evidence: `complete_six_candidate_full_5x5_grid`.

This is development/tuning evidence, not a final or independent test. The orchestration ranking is the equal-weight mean of declared fold-cell metrics. Participant-OOF repeat summaries are reported separately as a descriptive sensitivity view and do not rewrite the completed selection.

Selection agreement check: participant-OOF descriptive top is `b16_lr3e-4`; agreement with the orchestration winner is `True`.

## Phase reports

- `screen`: [20260822_190835_catalog_sweep_staged-static-06-batch-lr-successive-halving-v1--screen-5epoch-reduced-cv](phases/screen_5epoch_reduced_cv/20260822_190835_catalog_sweep_staged-static-06-batch-lr-successive-halving-v1--screen-5epoch-reduced-cv/STUDY_SUMMARY.md)
- `promotion`: [20260822_200138_catalog_sweep_staged-static-06-batch-lr-successive-halving-v1--promoted-full-cv](phases/promoted_full_cv/20260822_200138_catalog_sweep_staged-static-06-batch-lr-successive-halving-v1--promoted-full-cv/STUDY_SUMMARY.md)
- `completion`: [20260823_002609_catalog_sweep_staged-static-06-batch-lr-successive-halving-v1--nonpromoted-full-cv](phases/nonpromoted_full_cv/20260823_002609_catalog_sweep_staged-static-06-batch-lr-successive-halving-v1--nonpromoted-full-cv/STUDY_SUMMARY.md)

## Paired tables and plots

- [screen_ranking.csv](tables/screen_ranking.csv) · [plot](figures/screen_ranking.png)
- [promotion_ranking.csv](tables/promotion_ranking.csv) · [plot](figures/promotion_ranking.png)
- [nonpromoted_full_cv_ranking.csv](tables/nonpromoted_full_cv_ranking.csv) · [plot](figures/nonpromoted_full_cv_ranking.png)
- [all_candidates_full_cv_ranking.csv](tables/all_candidates_full_cv_ranking.csv) · [plot](figures/all_candidates_full_cv_ranking.png)
- [screen_participant_oof_ranking.csv](tables/screen_participant_oof_ranking.csv) · [plot](figures/screen_participant_oof_ranking.png)
- [promotion_participant_oof_ranking.csv](tables/promotion_participant_oof_ranking.csv) · [plot](figures/promotion_participant_oof_ranking.png)
- [completion_participant_oof_ranking.csv](tables/completion_participant_oof_ranking.csv) · [plot](figures/completion_participant_oof_ranking.png)
- [all_candidates_full_cv_participant_oof_ranking.csv](tables/all_candidates_full_cv_participant_oof_ranking.csv) · [plot](figures/all_candidates_full_cv_participant_oof_ranking.png)

### screen_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd |
|---|---|---|---|---|---|---|
| 1 | b16_lr1e-3 | 5 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 62.2 ± 19.1 | 55.7 ± 20.3 |
| 2 | b32_lr3e-4 | 5 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 58.9 ± 16.7 | 50.3 ± 21.4 |
| 3 | b16_lr3e-4 | 5 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 55.6 ± 23.3 | 48.8 ± 25.4 |
| 4 | b16_lr1e-4 | 5 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 54.4 ± 23.1 | 47.9 ± 23.2 |
| 5 | b32_lr1e-4 | 5 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 52.2 ± 12.0 | 44.5 ± 16.4 |
| 6 | b32_lr1e-3 | 5 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 51.1 ± 20.9 | 43.6 ± 21.6 |

### promotion_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd |
|---|---|---|---|---|---|---|
| 1 | b16_lr3e-4 | 25 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 61.3 ± 16.9 | 54.4 ± 20.7 |
| 2 | b32_lr3e-4 | 25 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 55.6 ± 15.2 | 48.4 ± 17.8 |
| 3 | b16_lr1e-3 | 25 | equal_weight_fold_cell_mean_for_selection | successive_halving_selection_evidence | 54.7 ± 15.2 | 48.8 ± 18.2 |

### nonpromoted_full_cv_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd |
|---|---|---|---|---|---|---|
| 1 | b32_lr1e-3 | 25 | equal_weight_fold_cell_mean_for_selection | completion_subset_full_cv_evidence_not_standalone_selection | 57.3 ± 18.4 | 50.5 ± 22.3 |
| 2 | b16_lr1e-4 | 25 | equal_weight_fold_cell_mean_for_selection | completion_subset_full_cv_evidence_not_standalone_selection | 55.8 ± 15.1 | 49.3 ± 17.1 |
| 3 | b32_lr1e-4 | 25 | equal_weight_fold_cell_mean_for_selection | completion_subset_full_cv_evidence_not_standalone_selection | 54.0 ± 18.2 | 47.5 ± 19.6 |

### all_candidates_full_cv_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd |
|---|---|---|---|---|---|---|
| 1 | b16_lr3e-4 | 25 | equal_weight_fold_cell_mean_for_selection | exhaustive_full_grid_selection_evidence_after_completion | 61.3 ± 16.9 | 54.4 ± 20.7 |
| 2 | b32_lr1e-3 | 25 | equal_weight_fold_cell_mean_for_selection | exhaustive_full_grid_selection_evidence_after_completion | 57.3 ± 18.4 | 50.5 ± 22.3 |
| 3 | b16_lr1e-4 | 25 | equal_weight_fold_cell_mean_for_selection | exhaustive_full_grid_selection_evidence_after_completion | 55.8 ± 15.1 | 49.3 ± 17.1 |
| 4 | b32_lr3e-4 | 25 | equal_weight_fold_cell_mean_for_selection | exhaustive_full_grid_selection_evidence_after_completion | 55.6 ± 15.2 | 48.4 ± 17.8 |
| 5 | b16_lr1e-3 | 25 | equal_weight_fold_cell_mean_for_selection | exhaustive_full_grid_selection_evidence_after_completion | 54.7 ± 15.2 | 48.8 ± 18.2 |
| 6 | b32_lr1e-4 | 25 | equal_weight_fold_cell_mean_for_selection | exhaustive_full_grid_selection_evidence_after_completion | 54.0 ± 18.2 | 47.5 ± 19.6 |

## Participant-OOF descriptive sensitivity rankings

These tables recompute participant-level OOF metrics within repeat, then report the equal-weight repeat mean. They explain why the full-CV BA value differs from the fold-cell selection mean.

### screen_participant_oof_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd | macro_roc_auc_ovr | macro_pr_auc_ovr | expected_calibration_error | worst_fold_balanced_accuracy | worst_class_f1 | balanced_accuracy_lcb95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b16_lr1e-3 | 5 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 62.2 ± 19.1 | 55.7 ± 20.3 | 0.7735185185185186 | 0.7655555555555555 | 0.169633395694954 | 0.3333333333333333 | 0.5925925925925926 | 0.4191469008108507 |
| 2 | b32_lr3e-4 | 5 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 58.9 ± 16.7 | 50.3 ± 21.4 | 0.702037037037037 | 0.7084126984126984 | 0.2823290644094943 | 0.3333333333333333 | 0.47058823529411764 | 0.41084064126594244 |
| 3 | b16_lr3e-4 | 5 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 55.6 ± 23.3 | 48.8 ± 25.4 | 0.7194444444444444 | 0.7174074074074074 | 0.27514147092543 | 0.16666666666666666 | 0.48 | 0.30712224750638983 |
| 4 | b16_lr1e-4 | 5 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 54.4 ± 23.1 | 47.9 ± 23.2 | 0.7344444444444445 | 0.7307407407407407 | 0.19357705138299475 | 0.16666666666666666 | 0.46153846153846156 | 0.29771081410440436 |
| 5 | b32_lr1e-4 | 5 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 52.2 ± 12.0 | 44.5 ± 16.4 | 0.7762962962962963 | 0.7622222222222221 | 0.341816823964229 | 0.3333333333333333 | 0.4 | 0.39466281902301037 |
| 6 | b32_lr1e-3 | 5 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 51.1 ± 20.9 | 43.6 ± 21.6 | 0.7757407407407407 | 0.7755555555555556 | 0.29736358897273363 | 0.16666666666666666 | 0.42105263157894735 | 0.2882752337459673 |

### promotion_participant_oof_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd | macro_roc_auc_ovr | macro_pr_auc_ovr | expected_calibration_error | worst_fold_balanced_accuracy | worst_class_f1 | balanced_accuracy_lcb95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b16_lr3e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 59.5 ± 5.5 | 59.4 ± 5.6 | 0.7570494864612511 | 0.6671282858613441 | 0.1510165083068288 | 0.3333333333333333 | 0.5254237288135594 | 0.4537037037037037 |
| 2 | b16_lr1e-3 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 55.2 ± 4.6 | 55.4 ± 5.7 | 0.7493152816682228 | 0.6676824243920505 | 0.17121542991941996 | 0.3333333333333333 | 0.515625 | 0.4185185185185185 |
| 3 | b32_lr3e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 54.3 ± 2.7 | 53.4 ± 3.2 | 0.7227170868347339 | 0.6304876010952792 | 0.20100495090744655 | 0.3333333333333333 | 0.5081967213114754 | 0.41481481481481486 |

### completion_participant_oof_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd | macro_roc_auc_ovr | macro_pr_auc_ovr | expected_calibration_error | worst_fold_balanced_accuracy | worst_class_f1 | balanced_accuracy_lcb95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b32_lr1e-3 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 56.1 ± 5.7 | 55.4 ± 7.2 | 0.7329691876750701 | 0.6423911998475618 | 0.2038387306164191 | 0.3333333333333333 | 0.5128205128205128 | 0.42592592592592593 |
| 2 | b16_lr1e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 54.9 ± 3.7 | 54.6 ± 4.0 | 0.7303314659197012 | 0.6470368790314367 | 0.18205995378697126 | 0.16666666666666666 | 0.4778761061946903 | 0.40092592592592585 |
| 3 | b32_lr1e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_sensitivity_not_orchestration_selection | 51.9 ± 6.4 | 51.3 ± 6.1 | 0.7144724556489263 | 0.61534574108688 | 0.20002833193395458 | 0.16666666666666666 | 0.5042016806722689 | 0.3814814814814815 |

### all_candidates_full_cv_participant_oof_ranking

| rank | case_id | cell_count | metric_source | selection_role | balanced_accuracy_mean_sd_percent | macro_f1_percent_mean_sd | macro_roc_auc_ovr | macro_pr_auc_ovr | expected_calibration_error | worst_fold_balanced_accuracy | worst_class_f1 | balanced_accuracy_lcb95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b16_lr3e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_full_grid_sensitivity_not_orchestration_selection | 59.5 ± 5.5 | 59.4 ± 5.6 | 0.7570494864612511 | 0.6671282858613441 | 0.1510165083068288 | 0.3333333333333333 | 0.5254237288135594 | 0.4537037037037037 |
| 2 | b32_lr1e-3 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_full_grid_sensitivity_not_orchestration_selection | 56.1 ± 5.7 | 55.4 ± 7.2 | 0.7329691876750701 | 0.6423911998475618 | 0.2038387306164191 | 0.3333333333333333 | 0.5128205128205128 | 0.42592592592592593 |
| 3 | b16_lr1e-3 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_full_grid_sensitivity_not_orchestration_selection | 55.2 ± 4.6 | 55.4 ± 5.7 | 0.7493152816682228 | 0.6676824243920505 | 0.17121542991941996 | 0.3333333333333333 | 0.515625 | 0.4185185185185185 |
| 4 | b16_lr1e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_full_grid_sensitivity_not_orchestration_selection | 54.9 ± 3.7 | 54.6 ± 4.0 | 0.7303314659197012 | 0.6470368790314367 | 0.18205995378697126 | 0.16666666666666666 | 0.4778761061946903 | 0.40092592592592585 |
| 5 | b32_lr3e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_full_grid_sensitivity_not_orchestration_selection | 54.3 ± 2.7 | 53.4 ± 3.2 | 0.7227170868347339 | 0.6304876010952792 | 0.20100495090744655 | 0.3333333333333333 | 0.5081967213114754 | 0.41481481481481486 |
| 6 | b32_lr1e-4 | 25 | participant_oof_recomputed_equal_repeat_mean | descriptive_full_grid_sensitivity_not_orchestration_selection | 51.9 ± 6.4 | 51.3 ± 6.1 | 0.7144724556489263 | 0.61534574108688 | 0.20002833193395458 | 0.16666666666666666 | 0.5042016806722689 | 0.3814814814814815 |

## Test models, modules, inputs, and fixed parameters

| Cases / phases | Component role | Model / module | State | Input data (values and paths; no hashes) | Detailed fixed parameters | Algorithm and kernel (≤300 chars) |
|---|---|---|---|---|---|---|
| b16_lr1e-4 | classifier_tuning_candidate | InceptionTimeFull | executed | {"dataset": "Frailty29 static roles B/R1-R4", "hop_s": 2.5, "sampling_rate_hz": 64.0, "signal_views": ["RED/IR amplitude-preserving analysis view", "profile_a_lowpass_0p3hz processed physical A_dyn/GX/GY/GZ", {"dl_only_model_input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]}], "window_s": 5.0} | {"aggregation.balance_line": "line_b_equal_role_families", "resource_contract": {"promote_count": 3, "promotion_epochs": 10, "promotion_folds": [0, 1, 2, 3, 4], "promotion_repeats": [0, 1, 2, 3, 4], "ranking_metric": "balanced_accuracy", "screen_epochs": 5, "screen_folds": [0], "screen_repeats": [0, 1, 2, 3, 4], "tie_break_metric": "macro_f1"}, "signal.dl_resampling.enabled": true, "signal.dl_resampling.target_fs_hz": 64.0, "signal.normalization.raw_imu": "none", "signal.normalization.raw_ppg": "per_window_median_iqr_over_1p349_sd_finite", "training.batch_size": 16, "training.class_count_basis": "row", "training.class_weighting": "outer_train_window_inverse_frequency", "training.learning_rate": 0.0001, "training.optimizer": "adamw", "training.sampler": "exhaustive_shuffle_without_replacement", "windows.raw_dl.hop_s": 2.5, "windows.raw_dl.length_s": 5.0} | InceptionTimeFull raw-DL candidate; candidate-specific values are combined with the selected B0+B2+B7 signal/training state; IMU gravity method=profile_a_lowpass_0p3hz. |
| b16_lr3e-4 | classifier_tuning_candidate | InceptionTimeFull | executed | {"dataset": "Frailty29 static roles B/R1-R4", "hop_s": 2.5, "sampling_rate_hz": 64.0, "signal_views": ["RED/IR amplitude-preserving analysis view", "profile_a_lowpass_0p3hz processed physical A_dyn/GX/GY/GZ", {"dl_only_model_input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]}], "window_s": 5.0} | {"aggregation.balance_line": "line_b_equal_role_families", "resource_contract": {"promote_count": 3, "promotion_epochs": 10, "promotion_folds": [0, 1, 2, 3, 4], "promotion_repeats": [0, 1, 2, 3, 4], "ranking_metric": "balanced_accuracy", "screen_epochs": 5, "screen_folds": [0], "screen_repeats": [0, 1, 2, 3, 4], "tie_break_metric": "macro_f1"}, "signal.dl_resampling.enabled": true, "signal.dl_resampling.target_fs_hz": 64.0, "signal.normalization.raw_imu": "none", "signal.normalization.raw_ppg": "per_window_median_iqr_over_1p349_sd_finite", "training.batch_size": 16, "training.class_count_basis": "row", "training.class_weighting": "outer_train_window_inverse_frequency", "training.learning_rate": 0.0003, "training.optimizer": "adamw", "training.sampler": "exhaustive_shuffle_without_replacement", "windows.raw_dl.hop_s": 2.5, "windows.raw_dl.length_s": 5.0} | InceptionTimeFull raw-DL candidate; candidate-specific values are combined with the selected B0+B2+B7 signal/training state; IMU gravity method=profile_a_lowpass_0p3hz. |
| b16_lr1e-3 | classifier_tuning_candidate | InceptionTimeFull | executed | {"dataset": "Frailty29 static roles B/R1-R4", "hop_s": 2.5, "sampling_rate_hz": 64.0, "signal_views": ["RED/IR amplitude-preserving analysis view", "profile_a_lowpass_0p3hz processed physical A_dyn/GX/GY/GZ", {"dl_only_model_input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]}], "window_s": 5.0} | {"aggregation.balance_line": "line_b_equal_role_families", "resource_contract": {"promote_count": 3, "promotion_epochs": 10, "promotion_folds": [0, 1, 2, 3, 4], "promotion_repeats": [0, 1, 2, 3, 4], "ranking_metric": "balanced_accuracy", "screen_epochs": 5, "screen_folds": [0], "screen_repeats": [0, 1, 2, 3, 4], "tie_break_metric": "macro_f1"}, "signal.dl_resampling.enabled": true, "signal.dl_resampling.target_fs_hz": 64.0, "signal.normalization.raw_imu": "none", "signal.normalization.raw_ppg": "per_window_median_iqr_over_1p349_sd_finite", "training.batch_size": 16, "training.class_count_basis": "row", "training.class_weighting": "outer_train_window_inverse_frequency", "training.learning_rate": 0.001, "training.optimizer": "adamw", "training.sampler": "exhaustive_shuffle_without_replacement", "windows.raw_dl.hop_s": 2.5, "windows.raw_dl.length_s": 5.0} | InceptionTimeFull raw-DL candidate; candidate-specific values are combined with the selected B0+B2+B7 signal/training state; IMU gravity method=profile_a_lowpass_0p3hz. |
| b32_lr1e-4 | classifier_tuning_candidate | InceptionTimeFull | executed | {"dataset": "Frailty29 static roles B/R1-R4", "hop_s": 2.5, "sampling_rate_hz": 64.0, "signal_views": ["RED/IR amplitude-preserving analysis view", "profile_a_lowpass_0p3hz processed physical A_dyn/GX/GY/GZ", {"dl_only_model_input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]}], "window_s": 5.0} | {"aggregation.balance_line": "line_b_equal_role_families", "resource_contract": {"promote_count": 3, "promotion_epochs": 10, "promotion_folds": [0, 1, 2, 3, 4], "promotion_repeats": [0, 1, 2, 3, 4], "ranking_metric": "balanced_accuracy", "screen_epochs": 5, "screen_folds": [0], "screen_repeats": [0, 1, 2, 3, 4], "tie_break_metric": "macro_f1"}, "signal.dl_resampling.enabled": true, "signal.dl_resampling.target_fs_hz": 64.0, "signal.normalization.raw_imu": "none", "signal.normalization.raw_ppg": "per_window_median_iqr_over_1p349_sd_finite", "training.batch_size": 32, "training.class_count_basis": "row", "training.class_weighting": "outer_train_window_inverse_frequency", "training.learning_rate": 0.0001, "training.optimizer": "adamw", "training.sampler": "exhaustive_shuffle_without_replacement", "windows.raw_dl.hop_s": 2.5, "windows.raw_dl.length_s": 5.0} | InceptionTimeFull raw-DL candidate; candidate-specific values are combined with the selected B0+B2+B7 signal/training state; IMU gravity method=profile_a_lowpass_0p3hz. |
| b32_lr3e-4 | classifier_tuning_candidate | InceptionTimeFull | executed | {"dataset": "Frailty29 static roles B/R1-R4", "hop_s": 2.5, "sampling_rate_hz": 64.0, "signal_views": ["RED/IR amplitude-preserving analysis view", "profile_a_lowpass_0p3hz processed physical A_dyn/GX/GY/GZ", {"dl_only_model_input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]}], "window_s": 5.0} | {"aggregation.balance_line": "line_b_equal_role_families", "resource_contract": {"promote_count": 3, "promotion_epochs": 10, "promotion_folds": [0, 1, 2, 3, 4], "promotion_repeats": [0, 1, 2, 3, 4], "ranking_metric": "balanced_accuracy", "screen_epochs": 5, "screen_folds": [0], "screen_repeats": [0, 1, 2, 3, 4], "tie_break_metric": "macro_f1"}, "signal.dl_resampling.enabled": true, "signal.dl_resampling.target_fs_hz": 64.0, "signal.normalization.raw_imu": "none", "signal.normalization.raw_ppg": "per_window_median_iqr_over_1p349_sd_finite", "training.batch_size": 32, "training.class_count_basis": "row", "training.class_weighting": "outer_train_window_inverse_frequency", "training.learning_rate": 0.0003, "training.optimizer": "adamw", "training.sampler": "exhaustive_shuffle_without_replacement", "windows.raw_dl.hop_s": 2.5, "windows.raw_dl.length_s": 5.0} | InceptionTimeFull raw-DL candidate; candidate-specific values are combined with the selected B0+B2+B7 signal/training state; IMU gravity method=profile_a_lowpass_0p3hz. |
| b32_lr1e-3 | classifier_tuning_candidate | InceptionTimeFull | executed | {"dataset": "Frailty29 static roles B/R1-R4", "hop_s": 2.5, "sampling_rate_hz": 64.0, "signal_views": ["RED/IR amplitude-preserving analysis view", "profile_a_lowpass_0p3hz processed physical A_dyn/GX/GY/GZ", {"dl_only_model_input_channel_order": ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"]}], "window_s": 5.0} | {"aggregation.balance_line": "line_b_equal_role_families", "resource_contract": {"promote_count": 3, "promotion_epochs": 10, "promotion_folds": [0, 1, 2, 3, 4], "promotion_repeats": [0, 1, 2, 3, 4], "ranking_metric": "balanced_accuracy", "screen_epochs": 5, "screen_folds": [0], "screen_repeats": [0, 1, 2, 3, 4], "tie_break_metric": "macro_f1"}, "signal.dl_resampling.enabled": true, "signal.dl_resampling.target_fs_hz": 64.0, "signal.normalization.raw_imu": "none", "signal.normalization.raw_ppg": "per_window_median_iqr_over_1p349_sd_finite", "training.batch_size": 32, "training.class_count_basis": "row", "training.class_weighting": "outer_train_window_inverse_frequency", "training.learning_rate": 0.001, "training.optimizer": "adamw", "training.sampler": "exhaustive_shuffle_without_replacement", "windows.raw_dl.hop_s": 2.5, "windows.raw_dl.length_s": 5.0} | InceptionTimeFull raw-DL candidate; candidate-specific values are combined with the selected B0+B2+B7 signal/training state; IMU gravity method=profile_a_lowpass_0p3hz. |

## Seeds and data splits

| phase | fixed_epochs | repeat_indices | fold_indices | split_seeds | training_seeds | training_seed_policy | split_group | selection_scope |
|---|---|---|---|---|---|---|---|---|
| screen_5epoch_reduced_cv | 5 | [0, 1, 2, 3, 4] | [0] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | outer_cv_repeat_seed_equals_split_seed | participant_id | development_tuning_only_not_final_test |
| promoted_full_cv | 10 | [0, 1, 2, 3, 4] | [0, 1, 2, 3, 4] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | outer_cv_repeat_seed_equals_split_seed | participant_id | development_tuning_only_not_final_test |
| nonpromoted_full_cv | 10 | [0, 1, 2, 3, 4] | [0, 1, 2, 3, 4] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | outer_cv_repeat_seed_equals_split_seed | participant_id | development_tuning_only_not_final_test |

All compact percentages use `mean ± population SD`; raw numeric columns remain available in JSON and each displayed CSV table occupies one workbook sheet.

## Reproducibility

The nested phase reports contain split seeds, training seeds, data splits, model/module names, actual input descriptions, and resolved fixed parameters.
