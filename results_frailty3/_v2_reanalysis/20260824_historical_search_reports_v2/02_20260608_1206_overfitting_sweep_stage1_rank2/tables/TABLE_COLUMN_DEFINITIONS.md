# Table column definitions and formulas

This documentation catalog describes source report tables. The catalog artifact excludes itself to prevent recursive documentation rows.

## `all_config_repeat_metrics`

Persisted root CSV report table

- **rank** (`rank`): Ordinal position after applying the table's declared sorting rule. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **seed** (`seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **extra_input** (`extra_input`): Persisted source-table value for `extra_input`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_epochs** (`cnn_epochs`): Persisted source-table value for `cnn_epochs`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_patience** (`cnn_patience`): Persisted source-table value for `cnn_patience`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **window_sec** (`window_sec`): Persisted source-table value for `window_sec`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **hop_sec** (`hop_sec`): Persisted source-table value for `hop_sec`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **overlap_pct** (`overlap_pct`): Persisted source-table value for `overlap_pct`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **max_windows_fraction** (`max_windows_fraction`): Proportion for the numerator and denominator named by this column. Formula: `rate = stated numerator count / stated eligible denominator count`
- **cnn_lr** (`cnn_lr`): Persisted source-table value for `cnn_lr`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_weight_decay** (`cnn_weight_decay`): Persisted source-table value for `cnn_weight_decay`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_dropout** (`cnn_dropout`): Persisted source-table value for `cnn_dropout`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_label_smoothing** (`cnn_label_smoothing`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **n_train_subjects** (`n_train_subjects`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_val_subjects** (`n_val_subjects`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_test_subjects** (`n_test_subjects`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_train_windows** (`n_train_windows`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_val_windows** (`n_val_windows`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_test_windows** (`n_test_windows`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **validation_window_balanced_accuracy** (`validation_window_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **validation_window_macro_f1** (`validation_window_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **validation_file_balanced_accuracy** (`validation_file_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **validation_file_macro_f1** (`validation_file_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **validation_subject_balanced_accuracy** (`validation_subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **validation_subject_macro_f1** (`validation_subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **window_balanced_accuracy** (`window_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **window_macro_f1** (`window_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **file_balanced_accuracy** (`file_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **file_macro_f1** (`file_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **best_epoch** (`best_epoch`): Persisted source-table value for `best_epoch`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_window_balanced_accuracy** (`best_window_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **duration_sec** (`duration_sec`): Persisted source-table value for `duration_sec`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **report_path** (`report_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **learning_curve_csv** (`learning_curve_csv`): Persisted source-table value for `learning_curve_csv`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **learning_curve_png** (`learning_curve_png`): Persisted source-table value for `learning_curve_png`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **pre_frail_precision** (`pre_frail_precision`): Positive predictive value for the named class. Formula: `precision = TP / (TP + FP)`
- **pre_frail_recall** (`pre_frail_recall`): Sensitivity for the named class. Formula: `recall = TP / (TP + FN)`
- **pre_frail_f1** (`pre_frail_f1`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **pre_frail_support** (`pre_frail_support`): Number of evaluated units whose true label is the named class. Formula: `support_c = TP_c + FN_c`
- **robust_non_frail_precision** (`robust_non_frail_precision`): Positive predictive value for the named class. Formula: `precision = TP / (TP + FP)`
- **robust_non_frail_recall** (`robust_non_frail_recall`): Sensitivity for the named class. Formula: `recall = TP / (TP + FN)`
- **robust_non_frail_f1** (`robust_non_frail_f1`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **robust_non_frail_support** (`robust_non_frail_support`): Number of evaluated units whose true label is the named class. Formula: `support_c = TP_c + FN_c`
- **young_precision** (`young_precision`): Positive predictive value for the named class. Formula: `precision = TP / (TP + FP)`
- **young_recall** (`young_recall`): Sensitivity for the named class. Formula: `recall = TP / (TP + FN)`
- **young_f1** (`young_f1`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **young_support** (`young_support`): Number of evaluated units whose true label is the named class. Formula: `support_c = TP_c + FN_c`
- **worst_class_recall** (`worst_class_recall`): Sensitivity for the named class. Formula: `recall = TP / (TP + FN)`
- **worst_class_f1** (`worst_class_f1`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **overfit_stage** (`overfit_stage`): Persisted source-table value for `overfit_stage`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **overfit_config_id** (`overfit_config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_config_name** (`overfit_config_name`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **stage1_screen_group** (`stage1_screen_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **stage1_regularization_factor** (`stage1_regularization_factor`): Persisted source-table value for `stage1_regularization_factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_regularization_value** (`stage1_regularization_value`): Persisted source-table value for `stage1_regularization_value`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **dynamic_data_mode** (`dynamic_data_mode`): Persisted source-table value for `dynamic_data_mode`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage2_fixed_epoch** (`stage2_fixed_epoch`): Persisted source-table value for `stage2_fixed_epoch`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage2_source_config_id** (`stage2_source_config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **is_reference** (`is_reference`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_grid_cnn_patience** (`overfit_grid_cnn_patience`): Persisted source-table value for `overfit_grid_cnn_patience`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **overfit_grid_max_windows_fraction** (`overfit_grid_max_windows_fraction`): Proportion for the numerator and denominator named by this column. Formula: `rate = stated numerator count / stated eligible denominator count`
- **train_role_mode** (`train_role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **validation_role_mode** (`validation_role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **test_role_mode** (`test_role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **eval_protocol** (`eval_protocol`): Persisted source-table value for `eval_protocol`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **requested_cv_folds** (`requested_cv_folds`): Persisted source-table value for `requested_cv_folds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **n_splits** (`n_splits`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **early_stopping_source** (`early_stopping_source`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **n_static_train_files** (`n_static_train_files`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_all_role_train_files** (`n_all_role_train_files`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_dynamic_added_train_files** (`n_dynamic_added_train_files`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **best_epoch_train_balanced_accuracy** (`best_epoch_train_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **best_epoch_val_balanced_accuracy** (`best_epoch_val_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **train_val_balanced_accuracy_gap** (`train_val_balanced_accuracy_gap`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **best_epoch_train_loss** (`best_epoch_train_loss`): Persisted source-table value for `best_epoch_train_loss`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_epoch_val_loss** (`best_epoch_val_loss`): Persisted source-table value for `best_epoch_val_loss`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **val_train_loss_gap** (`val_train_loss_gap`): Persisted source-table value for `val_train_loss_gap`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **regularization_bundle** (`regularization_bundle`): Persisted source-table value for `regularization_bundle`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`

## `all_config_summary_numeric`

Persisted root CSV report table

- **descriptive_rank** (`descriptive_rank`): Ordinal position after applying the table's declared sorting rule. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_config_id** (`overfit_config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_config_name** (`overfit_config_name`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_stage** (`overfit_stage`): Persisted source-table value for `overfit_stage`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_screen_group** (`stage1_screen_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **stage1_regularization_factor** (`stage1_regularization_factor`): Persisted source-table value for `stage1_regularization_factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_regularization_value** (`stage1_regularization_value`): Persisted source-table value for `stage1_regularization_value`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_epochs** (`cnn_epochs`): Persisted source-table value for `cnn_epochs`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_lr** (`cnn_lr`): Persisted source-table value for `cnn_lr`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_weight_decay** (`cnn_weight_decay`): Persisted source-table value for `cnn_weight_decay`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_dropout** (`cnn_dropout`): Persisted source-table value for `cnn_dropout`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_label_smoothing** (`cnn_label_smoothing`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **max_windows_fraction** (`max_windows_fraction`): Proportion for the numerator and denominator named by this column. Formula: `rate = stated numerator count / stated eligible denominator count`
- **n_runs** (`n_runs`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **seeds** (`seeds`): Persisted source-table value for `seeds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **window_balanced_accuracy_n_repeats** (`window_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **window_balanced_accuracy_mean** (`window_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **window_balanced_accuracy_sample_sd** (`window_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **window_balanced_accuracy_repeat_t_ci95_low** (`window_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_balanced_accuracy_repeat_t_ci95_high** (`window_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_balanced_accuracy_repeat_t_ci95_method** (`window_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **window_macro_f1_n_repeats** (`window_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **window_macro_f1_mean** (`window_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **window_macro_f1_sample_sd** (`window_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **window_macro_f1_repeat_t_ci95_low** (`window_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_macro_f1_repeat_t_ci95_high** (`window_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_macro_f1_repeat_t_ci95_method** (`window_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **file_balanced_accuracy_n_repeats** (`file_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **file_balanced_accuracy_mean** (`file_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **file_balanced_accuracy_sample_sd** (`file_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **file_balanced_accuracy_repeat_t_ci95_low** (`file_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_balanced_accuracy_repeat_t_ci95_high** (`file_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_balanced_accuracy_repeat_t_ci95_method** (`file_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **file_macro_f1_n_repeats** (`file_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **file_macro_f1_mean** (`file_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **file_macro_f1_sample_sd** (`file_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **file_macro_f1_repeat_t_ci95_low** (`file_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_macro_f1_repeat_t_ci95_high** (`file_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_macro_f1_repeat_t_ci95_method** (`file_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy_n_repeats** (`subject_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_balanced_accuracy_mean** (`subject_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_balanced_accuracy_sample_sd** (`subject_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_balanced_accuracy_repeat_t_ci95_low** (`subject_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_balanced_accuracy_repeat_t_ci95_high** (`subject_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_balanced_accuracy_repeat_t_ci95_method** (`subject_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_macro_f1_n_repeats** (`subject_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_f1_mean** (`subject_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_macro_f1_sample_sd** (`subject_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_macro_f1_repeat_t_ci95_low** (`subject_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_f1_repeat_t_ci95_high** (`subject_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_f1_repeat_t_ci95_method** (`subject_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **duration_sec_n_repeats** (`duration_sec_n_repeats`): Persisted source-table value for `duration_sec_n_repeats`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **duration_sec_mean** (`duration_sec_mean`): Arithmetic mean of the reported statistic over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **duration_sec_sample_sd** (`duration_sec_sample_sd`): Sample standard deviation of the reported statistic Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **duration_sec_repeat_t_ci95_low** (`duration_sec_repeat_t_ci95_low`): Reported 95% confidence bound or interval for the reported statistic Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **duration_sec_repeat_t_ci95_high** (`duration_sec_repeat_t_ci95_high`): Reported 95% confidence bound or interval for the reported statistic Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **duration_sec_repeat_t_ci95_method** (`duration_sec_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **formal_v2_selection_eligible** (`formal_v2_selection_eligible`): Persisted source-table value for `formal_v2_selection_eligible`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `archived_parameter_contract`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **row_type** (`row_type`): Persisted source-table value for `row_type`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **unique_count** (`unique_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **values** (`values`): Persisted source-table value for `values`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **role** (`role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `classifier_per_class_results`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **classifier_id** (`classifier_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **case_id** (`case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_name** (`config_name`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **class_label** (`class_label`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **metric_scope** (`metric_scope`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **expected_repeat_count** (`expected_repeat_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **split_seeds** (`split_seeds`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **precision_n_repeats_available** (`precision_n_repeats_available`): Positive predictive value for the named class. Formula: `precision = TP / (TP + FP)`
- **precision_mean** (`precision_mean`): Arithmetic mean of Positive predictive value for the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **precision_sample_sd** (`precision_sample_sd`): Sample standard deviation of Positive predictive value for the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **precision_repeat_t_ci95_low** (`precision_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Positive predictive value for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **precision_repeat_t_ci95_high** (`precision_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Positive predictive value for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **precision_repeat_t_ci95_method** (`precision_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **precision_applicability** (`precision_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **sensitivity_n_repeats_available** (`sensitivity_n_repeats_available`): True-positive rate for the positive class. Formula: `sensitivity = TP / (TP + FN)`
- **sensitivity_mean** (`sensitivity_mean`): Arithmetic mean of True-positive rate for the positive class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **sensitivity_sample_sd** (`sensitivity_sample_sd`): Sample standard deviation of True-positive rate for the positive class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **sensitivity_repeat_t_ci95_low** (`sensitivity_repeat_t_ci95_low`): Reported 95% confidence bound or interval for True-positive rate for the positive class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **sensitivity_repeat_t_ci95_high** (`sensitivity_repeat_t_ci95_high`): Reported 95% confidence bound or interval for True-positive rate for the positive class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **sensitivity_repeat_t_ci95_method** (`sensitivity_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **sensitivity_applicability** (`sensitivity_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **specificity_n_repeats_available** (`specificity_n_repeats_available`): True-negative rate for the negative class. Formula: `specificity = TN / (TN + FP)`
- **specificity_mean** (`specificity_mean`): Arithmetic mean of True-negative rate for the negative class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **specificity_sample_sd** (`specificity_sample_sd`): Sample standard deviation of True-negative rate for the negative class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **specificity_repeat_t_ci95_low** (`specificity_repeat_t_ci95_low`): Reported 95% confidence bound or interval for True-negative rate for the negative class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **specificity_repeat_t_ci95_high** (`specificity_repeat_t_ci95_high`): Reported 95% confidence bound or interval for True-negative rate for the negative class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **specificity_repeat_t_ci95_method** (`specificity_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **specificity_applicability** (`specificity_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **balanced_accuracy_n_repeats_available** (`balanced_accuracy_n_repeats_available`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **balanced_accuracy_mean** (`balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **balanced_accuracy_sample_sd** (`balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **balanced_accuracy_repeat_t_ci95_low** (`balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **balanced_accuracy_repeat_t_ci95_high** (`balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **balanced_accuracy_repeat_t_ci95_method** (`balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **balanced_accuracy_applicability** (`balanced_accuracy_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **f1_n_repeats_available** (`f1_n_repeats_available`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **f1_mean** (`f1_mean`): Arithmetic mean of Harmonic mean of precision and recall for the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **f1_sample_sd** (`f1_sample_sd`): Sample standard deviation of Harmonic mean of precision and recall for the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **f1_repeat_t_ci95_low** (`f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Harmonic mean of precision and recall for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **f1_repeat_t_ci95_high** (`f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Harmonic mean of precision and recall for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **f1_repeat_t_ci95_method** (`f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **f1_applicability** (`f1_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **support_n_repeats_available** (`support_n_repeats_available`): Number of evaluated units whose true label is the named class. Formula: `support_c = TP_c + FN_c`
- **support_mean** (`support_mean`): Arithmetic mean of Number of evaluated units whose true label is the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **support_sample_sd** (`support_sample_sd`): Sample standard deviation of Number of evaluated units whose true label is the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **support_repeat_t_ci95_low** (`support_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Number of evaluated units whose true label is the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **support_repeat_t_ci95_high** (`support_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Number of evaluated units whose true label is the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **support_repeat_t_ci95_method** (`support_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **support_applicability** (`support_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **predicted_support_n_repeats_available** (`predicted_support_n_repeats_available`): Number of evaluated units predicted as the named class. Formula: `predicted support_c = TP_c + FP_c`
- **predicted_support_mean** (`predicted_support_mean`): Arithmetic mean of Number of evaluated units predicted as the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **predicted_support_sample_sd** (`predicted_support_sample_sd`): Sample standard deviation of Number of evaluated units predicted as the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **predicted_support_repeat_t_ci95_low** (`predicted_support_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Number of evaluated units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **predicted_support_repeat_t_ci95_high** (`predicted_support_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Number of evaluated units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **predicted_support_repeat_t_ci95_method** (`predicted_support_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **predicted_support_applicability** (`predicted_support_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **true_positive_n_repeats_available** (`true_positive_n_repeats_available`): Count of named-class units predicted as that class. Formula: `TP_c = sum_i 1[y_i=c and predicted_i=c]`
- **true_positive_mean** (`true_positive_mean`): Arithmetic mean of Count of named-class units predicted as that class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **true_positive_sample_sd** (`true_positive_sample_sd`): Sample standard deviation of Count of named-class units predicted as that class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **true_positive_repeat_t_ci95_low** (`true_positive_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of named-class units predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_positive_repeat_t_ci95_high** (`true_positive_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of named-class units predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_positive_repeat_t_ci95_method** (`true_positive_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **true_positive_applicability** (`true_positive_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **false_positive_n_repeats_available** (`false_positive_n_repeats_available`): Count of non-class units predicted as the named class. Formula: `FP_c = sum_i 1[y_i!=c and predicted_i=c]`
- **false_positive_mean** (`false_positive_mean`): Arithmetic mean of Count of non-class units predicted as the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **false_positive_sample_sd** (`false_positive_sample_sd`): Sample standard deviation of Count of non-class units predicted as the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **false_positive_repeat_t_ci95_low** (`false_positive_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of non-class units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_positive_repeat_t_ci95_high** (`false_positive_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of non-class units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_positive_repeat_t_ci95_method** (`false_positive_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **false_positive_applicability** (`false_positive_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **false_negative_n_repeats_available** (`false_negative_n_repeats_available`): Count of named-class units not predicted as that class. Formula: `FN_c = sum_i 1[y_i=c and predicted_i!=c]`
- **false_negative_mean** (`false_negative_mean`): Arithmetic mean of Count of named-class units not predicted as that class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **false_negative_sample_sd** (`false_negative_sample_sd`): Sample standard deviation of Count of named-class units not predicted as that class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **false_negative_repeat_t_ci95_low** (`false_negative_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of named-class units not predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_negative_repeat_t_ci95_high** (`false_negative_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of named-class units not predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_negative_repeat_t_ci95_method** (`false_negative_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **false_negative_applicability** (`false_negative_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **true_negative_n_repeats_available** (`true_negative_n_repeats_available`): Count of non-class units not predicted as the named class. Formula: `TN_c = sum_i 1[y_i!=c and predicted_i!=c]`
- **true_negative_mean** (`true_negative_mean`): Arithmetic mean of Count of non-class units not predicted as the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **true_negative_sample_sd** (`true_negative_sample_sd`): Sample standard deviation of Count of non-class units not predicted as the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **true_negative_repeat_t_ci95_low** (`true_negative_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of non-class units not predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_negative_repeat_t_ci95_high** (`true_negative_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of non-class units not predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_negative_repeat_t_ci95_method** (`true_negative_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **true_negative_applicability** (`true_negative_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **roc_auc** (`roc_auc`): Area under the empirical receiver-operating-characteristic curve. Formula: `ROC-AUC = integral_0^1 TPR(FPR) dFPR (empirical trapezoidal area)`
- **pr_auc** (`pr_auc`): Area/average precision under the empirical precision-recall curve. Formula: `AP = sum_n (recall_n - recall_(n-1)) * precision_n`
- **confusion_count_applicability** (`confusion_count_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **roc_auc_applicability** (`roc_auc_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **pr_auc_applicability** (`pr_auc_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `configuration_parameters`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_name** (`config_name`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **input_data** (`input_data`): Persisted source-table value for `input_data`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **extra_input** (`extra_input`): Persisted source-table value for `extra_input`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **window_sec** (`window_sec`): Persisted source-table value for `window_sec`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **hop_sec** (`hop_sec`): Persisted source-table value for `hop_sec`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **overlap_pct** (`overlap_pct`): Persisted source-table value for `overlap_pct`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **max_windows_fraction** (`max_windows_fraction`): Proportion for the numerator and denominator named by this column. Formula: `rate = stated numerator count / stated eligible denominator count`
- **cnn_epochs** (`cnn_epochs`): Persisted source-table value for `cnn_epochs`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_patience** (`cnn_patience`): Persisted source-table value for `cnn_patience`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **early_stopping_source** (`early_stopping_source`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **cnn_lr** (`cnn_lr`): Persisted source-table value for `cnn_lr`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_weight_decay** (`cnn_weight_decay`): Persisted source-table value for `cnn_weight_decay`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_dropout** (`cnn_dropout`): Persisted source-table value for `cnn_dropout`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_label_smoothing** (`cnn_label_smoothing`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **regularization_bundle** (`regularization_bundle`): Persisted source-table value for `regularization_bundle`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **overfit_stage** (`overfit_stage`): Persisted source-table value for `overfit_stage`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_screen_group** (`stage1_screen_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **stage1_regularization_factor** (`stage1_regularization_factor`): Persisted source-table value for `stage1_regularization_factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_regularization_value** (`stage1_regularization_value`): Persisted source-table value for `stage1_regularization_value`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **is_reference** (`is_reference`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **dynamic_data_mode** (`dynamic_data_mode`): Persisted source-table value for `dynamic_data_mode`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **train_role_mode** (`train_role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **validation_role_mode** (`validation_role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **test_role_mode** (`test_role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **eval_protocol** (`eval_protocol`): Persisted source-table value for `eval_protocol`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **n_splits** (`n_splits`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **report_config__cnn_batch_size** (`report_config__cnn_batch_size`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_dropout** (`report_config__cnn_dropout`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_epochs** (`report_config__cnn_epochs`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_hop_sec** (`report_config__cnn_hop_sec`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_label_smoothing** (`report_config__cnn_label_smoothing`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_lr** (`report_config__cnn_lr`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_max_windows_fraction** (`report_config__cnn_max_windows_fraction`): Proportion for the numerator and denominator named by this column. Formula: `rate = stated numerator count / stated eligible denominator count`
- **report_config__cnn_max_windows_per_file** (`report_config__cnn_max_windows_per_file`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_num_workers** (`report_config__cnn_num_workers`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_patience** (`report_config__cnn_patience`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_select_best_epoch** (`report_config__cnn_select_best_epoch`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_seq_sec** (`report_config__cnn_seq_sec`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_target_fs** (`report_config__cnn_target_fs`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__cnn_weight_decay** (`report_config__cnn_weight_decay`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__data_root** (`report_config__data_root`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__extra_input** (`report_config__extra_input`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__folds** (`report_config__folds`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__fs** (`report_config__fs`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__hop_sec** (`report_config__hop_sec`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__label_csv** (`report_config__label_csv`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__role_mode** (`report_config__role_mode`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_candidates_per_class_channel** (`report_config__shapeformer_candidates_per_class_channel`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_dim_ff** (`report_config__shapeformer_dim_ff`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_discovery_method** (`report_config__shapeformer_discovery_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_discovery_windows** (`report_config__shapeformer_discovery_windows`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_dropout** (`report_config__shapeformer_dropout`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_heads** (`report_config__shapeformer_heads`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_local_embed_dim** (`report_config__shapeformer_local_embed_dim`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_local_window** (`report_config__shapeformer_local_window`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_num_pip** (`report_config__shapeformer_num_pip`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_num_shapelets** (`report_config__shapeformer_num_shapelets`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_pisd_verbose** (`report_config__shapeformer_pisd_verbose`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_processes** (`report_config__shapeformer_processes`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_shape_embed_dim** (`report_config__shapeformer_shape_embed_dim`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_shapelet_len** (`report_config__shapeformer_shapelet_len`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__shapeformer_shapelet_stride** (`report_config__shapeformer_shapelet_stride`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__study_dir** (`report_config__study_dir`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__win_sec** (`report_config__win_sec`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **report_config__young_dir** (`report_config__young_dir`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `factor_paired_participant_inference`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_family** (`comparison_family`): Persisted source-table value for `comparison_family`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparison_id** (`comparison_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_role** (`comparison_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **factor** (`factor`): Persisted source-table value for `factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **epoch** (`epoch`): Persisted source-table value for `epoch`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **reference_case_id** (`reference_case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **candidate_case_id** (`candidate_case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **metric** (`metric`): Persisted source-table value for `metric`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **candidate_minus_reference** (`candidate_minus_reference`): Paired candidate-minus-reference difference in the reported statistic Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **point_delta_source** (`point_delta_source`): Paired candidate-minus-reference difference in the reported statistic Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **participant_cluster_delta_ci95_low** (`participant_cluster_delta_ci95_low`): Participant-cluster percentile-bootstrap 95% CI lower endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b = metric_candidate,b - metric_reference,b from the same participant draw; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_delta_ci95_high** (`participant_cluster_delta_ci95_high`): Participant-cluster percentile-bootstrap 95% CI upper endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b = metric_candidate,b - metric_reference,b from the same participant draw; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_ci_applicability** (`participant_cluster_ci_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **participant_cluster_ci_unavailability_reason** (`participant_cluster_ci_unavailability_reason`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **raw_two_sided_p_value** (`raw_two_sided_p_value`): Null-hypothesis tail probability from the table's declared test. Formula: `two-sided p = Pr_H0(\|T*\| >= \|T_observed\|); exact statistic and resampling/rank distribution follow the declared test_method`
- **holm_adjusted_p_value** (`holm_adjusted_p_value`): Holm step-down multiplicity-adjusted P value. Formula: `ordered adjusted p_(i) = max_(j<=i) [(m-j+1) * p_(j)], capped at 1`
- **p_value_applicability** (`p_value_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **inference_role** (`inference_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **automatic_selection** (`automatic_selection`): Persisted source-table value for `automatic_selection`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `factor_pairwise_repeat_metric_deltas`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_family** (`comparison_family`): Persisted source-table value for `comparison_family`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparison_id** (`comparison_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_role** (`comparison_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **factor** (`factor`): Persisted source-table value for `factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **epoch** (`epoch`): Persisted source-table value for `epoch`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **reference_case_id** (`reference_case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **candidate_case_id** (`candidate_case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_contract_status** (`comparison_contract_status`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **difference_direction** (`difference_direction`): Persisted source-table value for `difference_direction`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **reference_balanced_accuracy** (`reference_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **candidate_balanced_accuracy** (`candidate_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **balanced_accuracy_delta** (`balanced_accuracy_delta`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **reference_macro_f1** (`reference_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **candidate_macro_f1** (`candidate_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **macro_f1_delta** (`macro_f1_delta`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **reference_macro_roc_auc_ovr** (`reference_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **candidate_macro_roc_auc_ovr** (`candidate_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_delta** (`macro_roc_auc_ovr_delta`): Paired candidate-minus-reference difference in Unweighted mean of valid one-vs-rest class ROC areas. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **macro_roc_auc_ovr_unavailability_reason** (`macro_roc_auc_ovr_unavailability_reason`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **automatic_selection** (`automatic_selection`): Persisted source-table value for `automatic_selection`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `factor_signals`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **epoch** (`epoch`): Persisted source-table value for `epoch`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **factor** (`factor`): Persisted source-table value for `factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_observed_value** (`best_observed_value`): Persisted source-table value for `best_observed_value`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_observed_config_id** (`best_observed_config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **baseline_config_id** (`baseline_config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **baseline_subject_ba_mean** (`baseline_subject_ba_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **candidate_subject_ba_mean** (`candidate_subject_ba_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **descriptive_delta_ba** (`descriptive_delta_ba`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **candidate_subject_ba_repeat_t_ci95_low** (`candidate_subject_ba_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **candidate_subject_ba_repeat_t_ci95_high** (`candidate_subject_ba_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **interpretation** (`interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `historical_absolute_participant_cluster_ci`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **classifier_id** (`classifier_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **case_id** (`case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **metric** (`metric`): Persisted source-table value for `metric`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **point_estimate** (`point_estimate`): Persisted source-table value for `point_estimate`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **point_estimate_source** (`point_estimate_source`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **participant_cluster_ci95_low** (`participant_cluster_ci95_low`): Participant-cluster percentile-bootstrap 95% CI lower endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b is the metric recomputed from bootstrap draw b; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_ci95_high** (`participant_cluster_ci95_high`): Participant-cluster percentile-bootstrap 95% CI upper endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b is the metric recomputed from bootstrap draw b; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_ci_applicability** (`participant_cluster_ci_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **participant_cluster_ci_unavailability_reason** (`participant_cluster_ci_unavailability_reason`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **formal_v2_inference** (`formal_v2_inference`): Persisted source-table value for `formal_v2_inference`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `leaderboard_display`

Persisted root CSV report table

- **rank** (`rank`): Ordinal position after applying the table's declared sorting rule. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_or_model** (`config_or_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_BA_mean_sd_percent** (`subject_BA_mean_sd_percent`): Compact mean and sample-standard-deviation display for Macro-average recall across the K declared classes. Formula: `display = 100 * mean +/- 100 * sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_BA_repeat_t_CI95_percent** (`subject_BA_repeat_t_CI95_percent`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 100 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_F1_mean_sd_percent** (`subject_macro_F1_mean_sd_percent`): Compact mean and sample-standard-deviation display for Unweighted mean of the K class-specific F1 scores. Formula: `display = 100 * mean +/- 100 * sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_macro_F1_repeat_t_CI95_percent** (`subject_macro_F1_repeat_t_CI95_percent`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 100 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_ROC_AUC** (`subject_macro_ROC_AUC`): Area under the empirical receiver-operating-characteristic curve. Formula: `ROC-AUC = integral_0^1 TPR(FPR) dFPR (empirical trapezoidal area)`
- **ROC_AUC_applicability** (`ROC_AUC_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **scientific_role** (`scientific_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `missing_v2_statistics`

Persisted root CSV report table

- **requested_output** (`requested_output`): Persisted source-table value for `requested_output`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **status** (`status`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **reason** (`reason`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **required_source_fields** (`required_source_fields`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **v2_action** (`v2_action`): Persisted source-table value for `v2_action`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `parameter_dependency_warnings`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **design_factor** (`design_factor`): Persisted source-table value for `design_factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **coupled_parameters** (`coupled_parameters`): Persisted source-table value for `coupled_parameters`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **status** (`status`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **reporting_rule** (`reporting_rule`): Persisted source-table value for `reporting_rule`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `parameter_inventory`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **unique_value_count** (`unique_value_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **observed_values** (`observed_values`): Persisted source-table value for `observed_values`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_role** (`parameter_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_interpretation** (`comparison_interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `parameter_value_metric_long`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `parameter_value_metric_summary`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_count** (`config_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **config_repeat_row_count** (`config_repeat_row_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **repeat_seed_count** (`repeat_seed_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **within_seed_aggregation** (`within_seed_aggregation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **interpretation** (`interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy_n_repeats** (`subject_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_balanced_accuracy_mean** (`subject_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_balanced_accuracy_sample_sd** (`subject_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_balanced_accuracy_repeat_t_ci95_low** (`subject_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_balanced_accuracy_repeat_t_ci95_high** (`subject_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_balanced_accuracy_repeat_t_ci95_method** (`subject_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_macro_f1_n_repeats** (`subject_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_f1_mean** (`subject_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_macro_f1_sample_sd** (`subject_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_macro_f1_repeat_t_ci95_low** (`subject_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_f1_repeat_t_ci95_high** (`subject_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_f1_repeat_t_ci95_method** (`subject_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_macro_roc_auc_ovr_mean** (`subject_macro_roc_auc_ovr_mean`): Arithmetic mean of Unweighted mean of valid one-vs-rest class ROC areas. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_macro_roc_auc_ovr_sample_sd** (`subject_macro_roc_auc_ovr_sample_sd`): Sample standard deviation of Unweighted mean of valid one-vs-rest class ROC areas. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_macro_roc_auc_ovr_repeat_t_ci95_low** (`subject_macro_roc_auc_ovr_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of valid one-vs-rest class ROC areas. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_roc_auc_ovr_repeat_t_ci95_high** (`subject_macro_roc_auc_ovr_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of valid one-vs-rest class ROC areas. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_roc_auc_ovr_applicability** (`subject_macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **formal_v2_p_value** (`formal_v2_p_value`): Null-hypothesis tail probability from the table's declared test. Formula: `two-sided p = Pr_H0(\|T*\| >= \|T_observed\|); exact statistic and resampling/rank distribution follow the declared test_method`
- **formal_v2_p_value_applicability** (`formal_v2_p_value_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `per_class_repeat_results`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **classifier_id** (`classifier_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **case_id** (`case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_name** (`config_name`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **metric_scope** (`metric_scope`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **class_label** (`class_label`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **precision** (`precision`): Positive predictive value for the named class. Formula: `precision = TP / (TP + FP)`
- **sensitivity** (`sensitivity`): True-positive rate for the positive class. Formula: `sensitivity = TP / (TP + FN)`
- **f1** (`f1`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **support** (`support`): Number of evaluated units whose true label is the named class. Formula: `support_c = TP_c + FN_c`
- **true_positive** (`true_positive`): Count of named-class units predicted as that class. Formula: `TP_c = sum_i 1[y_i=c and predicted_i=c]`
- **false_positive** (`false_positive`): Count of non-class units predicted as the named class. Formula: `FP_c = sum_i 1[y_i!=c and predicted_i=c]`
- **false_negative** (`false_negative`): Count of named-class units not predicted as that class. Formula: `FN_c = sum_i 1[y_i=c and predicted_i!=c]`
- **true_negative** (`true_negative`): Count of non-class units not predicted as the named class. Formula: `TN_c = sum_i 1[y_i!=c and predicted_i!=c]`
- **predicted_support** (`predicted_support`): Number of evaluated units predicted as the named class. Formula: `predicted support_c = TP_c + FP_c`
- **specificity** (`specificity`): True-negative rate for the negative class. Formula: `specificity = TN / (TN + FP)`
- **balanced_accuracy** (`balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **roc_auc** (`roc_auc`): Area under the empirical receiver-operating-characteristic curve. Formula: `ROC-AUC = integral_0^1 TPR(FPR) dFPR (empirical trapezoidal area)`
- **pr_auc** (`pr_auc`): Area/average precision under the empirical precision-recall curve. Formula: `AP = sum_n (recall_n - recall_(n-1)) * precision_n`
- **result_applicability** (`result_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **missing_archived_fields** (`missing_archived_fields`): Persisted source-table value for `missing_archived_fields`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **precision_applicability** (`precision_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **sensitivity_applicability** (`sensitivity_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **f1_applicability** (`f1_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **support_applicability** (`support_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **confusion_count_applicability** (`confusion_count_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **specificity_applicability** (`specificity_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **balanced_accuracy_applicability** (`balanced_accuracy_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **roc_auc_applicability** (`roc_auc_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **pr_auc_applicability** (`pr_auc_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `plot_01_top15_configs`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_02_cnn_epochs_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_03_cnn_lr_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_04_cnn_weight_decay_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_05_cnn_dropout_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_06_cnn_label_smoothing_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_07_max_windows_fraction_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_08_overfit_stage_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_09_stage1_screen_group_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_10_stage1_regularization_factor_all_archived_config_repeats`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_11_cnn_dropout_stage1_regularization_factor_dropout`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_12_cnn_label_smoothing_stage1_regularization_factor_label_smoothing`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_13_max_windows_fraction_stage1_regularization_factor_max_windows_fraction`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_14_cnn_weight_decay_stage1_regularization_factor_weight_decay`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subset_contract** (`subset_contract`): Persisted source-table value for `subset_contract`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_value** (`parameter_value`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **plot_category** (`plot_category`): Persisted source-table value for `plot_category`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **config_id** (`config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`
- **macro_roc_auc_ovr_applicability** (`macro_roc_auc_ovr_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **observation_unit** (`observation_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_warning** (`independence_warning`): Persisted source-table value for `independence_warning`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `source_evidence`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_kind** (`source_kind`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_file** (`source_file`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **size_bytes** (`size_bytes`): Persisted source-table value for `size_bytes`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **sha256** (`sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `split_audit`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_count** (`config_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **fold_count** (`fold_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **fold_sizes** (`fold_sizes`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **held_out_participant_count** (`held_out_participant_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **fold_roster_sha256** (`fold_roster_sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **all_configs_share_roster** (`all_configs_share_roster`): Persisted source-table value for `all_configs_share_roster`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `table_figure_pairs`

Persisted root CSV report table

- **figure_id** (`figure_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **title** (`title`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **figure_path** (`figure_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **table_path** (`table_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **interpretation** (`interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `top15_config_summary_numeric`

Persisted root CSV report table

- **descriptive_rank** (`descriptive_rank`): Ordinal position after applying the table's declared sorting rule. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_config_id** (`overfit_config_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_config_name** (`overfit_config_name`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **overfit_stage** (`overfit_stage`): Persisted source-table value for `overfit_stage`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_screen_group** (`stage1_screen_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **stage1_regularization_factor** (`stage1_regularization_factor`): Persisted source-table value for `stage1_regularization_factor`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **stage1_regularization_value** (`stage1_regularization_value`): Persisted source-table value for `stage1_regularization_value`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_epochs** (`cnn_epochs`): Persisted source-table value for `cnn_epochs`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_lr** (`cnn_lr`): Persisted source-table value for `cnn_lr`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_weight_decay** (`cnn_weight_decay`): Persisted source-table value for `cnn_weight_decay`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_dropout** (`cnn_dropout`): Persisted source-table value for `cnn_dropout`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **cnn_label_smoothing** (`cnn_label_smoothing`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **max_windows_fraction** (`max_windows_fraction`): Proportion for the numerator and denominator named by this column. Formula: `rate = stated numerator count / stated eligible denominator count`
- **n_runs** (`n_runs`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **seeds** (`seeds`): Persisted source-table value for `seeds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **window_balanced_accuracy_n_repeats** (`window_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **window_balanced_accuracy_mean** (`window_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **window_balanced_accuracy_sample_sd** (`window_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **window_balanced_accuracy_repeat_t_ci95_low** (`window_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_balanced_accuracy_repeat_t_ci95_high** (`window_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_balanced_accuracy_repeat_t_ci95_method** (`window_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **window_macro_f1_n_repeats** (`window_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **window_macro_f1_mean** (`window_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **window_macro_f1_sample_sd** (`window_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **window_macro_f1_repeat_t_ci95_low** (`window_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_macro_f1_repeat_t_ci95_high** (`window_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **window_macro_f1_repeat_t_ci95_method** (`window_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **file_balanced_accuracy_n_repeats** (`file_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **file_balanced_accuracy_mean** (`file_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **file_balanced_accuracy_sample_sd** (`file_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **file_balanced_accuracy_repeat_t_ci95_low** (`file_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_balanced_accuracy_repeat_t_ci95_high** (`file_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_balanced_accuracy_repeat_t_ci95_method** (`file_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **file_macro_f1_n_repeats** (`file_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **file_macro_f1_mean** (`file_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **file_macro_f1_sample_sd** (`file_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **file_macro_f1_repeat_t_ci95_low** (`file_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_macro_f1_repeat_t_ci95_high** (`file_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **file_macro_f1_repeat_t_ci95_method** (`file_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_balanced_accuracy_n_repeats** (`subject_balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_balanced_accuracy_mean** (`subject_balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_balanced_accuracy_sample_sd** (`subject_balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_balanced_accuracy_repeat_t_ci95_low** (`subject_balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_balanced_accuracy_repeat_t_ci95_high** (`subject_balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_balanced_accuracy_repeat_t_ci95_method** (`subject_balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_macro_f1_n_repeats** (`subject_macro_f1_n_repeats`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_macro_f1_mean** (`subject_macro_f1_mean`): Arithmetic mean of Unweighted mean of the K class-specific F1 scores. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **subject_macro_f1_sample_sd** (`subject_macro_f1_sample_sd`): Sample standard deviation of Unweighted mean of the K class-specific F1 scores. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_macro_f1_repeat_t_ci95_low** (`subject_macro_f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_f1_repeat_t_ci95_high** (`subject_macro_f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_f1_repeat_t_ci95_method** (`subject_macro_f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **duration_sec_n_repeats** (`duration_sec_n_repeats`): Persisted source-table value for `duration_sec_n_repeats`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **duration_sec_mean** (`duration_sec_mean`): Arithmetic mean of the reported statistic over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **duration_sec_sample_sd** (`duration_sec_sample_sd`): Sample standard deviation of the reported statistic Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **duration_sec_repeat_t_ci95_low** (`duration_sec_repeat_t_ci95_low`): Reported 95% confidence bound or interval for the reported statistic Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **duration_sec_repeat_t_ci95_high** (`duration_sec_repeat_t_ci95_high`): Reported 95% confidence bound or interval for the reported statistic Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **duration_sec_repeat_t_ci95_method** (`duration_sec_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **formal_v2_selection_eligible** (`formal_v2_selection_eligible`): Persisted source-table value for `formal_v2_selection_eligible`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
