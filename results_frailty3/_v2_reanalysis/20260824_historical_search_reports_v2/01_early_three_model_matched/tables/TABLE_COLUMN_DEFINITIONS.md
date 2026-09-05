# Table column definitions and formulas

This documentation catalog describes source report tables. The catalog artifact excludes itself to prevent recursive documentation rows.

## `early_three_model_exploratory_paired_tests`

Persisted root CSV report table

- **metric** (`metric`): Persisted source-table value for `metric`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **reference** (`reference`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **candidate** (`candidate`): Persisted source-table value for `candidate`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **candidate_minus_reference_mean_delta** (`candidate_minus_reference_mean_delta`): Paired candidate-minus-reference difference in the reported statistic Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **n_paired_repeat_seeds** (`n_paired_repeat_seeds`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **exact_sign_patterns** (`exact_sign_patterns`): Persisted source-table value for `exact_sign_patterns`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **raw_two_sided_exact_sign_flip_p** (`raw_two_sided_exact_sign_flip_p`): Null-hypothesis tail probability from the table's declared test. Formula: `two-sided p = Pr_H0(\|T*\| >= \|T_observed\|); exact statistic and resampling/rank distribution follow the declared test_method`
- **exchange_unit** (`exchange_unit`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **formal_v2_inference** (`formal_v2_inference`): Persisted source-table value for `formal_v2_inference`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **interpretation** (`interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **holm_adjusted_p_within_metric_three_pairs** (`holm_adjusted_p_within_metric_three_pairs`): Holm step-down multiplicity-adjusted P value. Formula: `ordered adjusted p_(i) = max_(j<=i) [(m-j+1) * p_(j)], capped at 1`

## `early_three_model_parameters`

Persisted root CSV report table

- **study** (`study`): Persisted source-table value for `study`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **model_or_module** (`model_or_module`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **value** (`value`): Persisted source-table value for `value`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **provenance** (`provenance`): Persisted source-table value for `provenance`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `early_three_model_per_class_summary`

Persisted root CSV report table

- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **class** (`class`): Persisted source-table value for `class`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **precision_n_repeats** (`precision_n_repeats`): Positive predictive value for the named class. Formula: `precision = TP / (TP + FP)`
- **precision_mean** (`precision_mean`): Arithmetic mean of Positive predictive value for the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **precision_sample_sd** (`precision_sample_sd`): Sample standard deviation of Positive predictive value for the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **precision_repeat_t_ci95_low** (`precision_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Positive predictive value for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **precision_repeat_t_ci95_high** (`precision_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Positive predictive value for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **precision_repeat_t_ci95_method** (`precision_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **sensitivity_n_repeats** (`sensitivity_n_repeats`): True-positive rate for the positive class. Formula: `sensitivity = TP / (TP + FN)`
- **sensitivity_mean** (`sensitivity_mean`): Arithmetic mean of True-positive rate for the positive class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **sensitivity_sample_sd** (`sensitivity_sample_sd`): Sample standard deviation of True-positive rate for the positive class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **sensitivity_repeat_t_ci95_low** (`sensitivity_repeat_t_ci95_low`): Reported 95% confidence bound or interval for True-positive rate for the positive class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **sensitivity_repeat_t_ci95_high** (`sensitivity_repeat_t_ci95_high`): Reported 95% confidence bound or interval for True-positive rate for the positive class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **sensitivity_repeat_t_ci95_method** (`sensitivity_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **specificity_n_repeats** (`specificity_n_repeats`): True-negative rate for the negative class. Formula: `specificity = TN / (TN + FP)`
- **specificity_mean** (`specificity_mean`): Arithmetic mean of True-negative rate for the negative class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **specificity_sample_sd** (`specificity_sample_sd`): Sample standard deviation of True-negative rate for the negative class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **specificity_repeat_t_ci95_low** (`specificity_repeat_t_ci95_low`): Reported 95% confidence bound or interval for True-negative rate for the negative class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **specificity_repeat_t_ci95_high** (`specificity_repeat_t_ci95_high`): Reported 95% confidence bound or interval for True-negative rate for the negative class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **specificity_repeat_t_ci95_method** (`specificity_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **balanced_accuracy_n_repeats** (`balanced_accuracy_n_repeats`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **balanced_accuracy_mean** (`balanced_accuracy_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **balanced_accuracy_sample_sd** (`balanced_accuracy_sample_sd`): Sample standard deviation of Macro-average recall across the K declared classes. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **balanced_accuracy_repeat_t_ci95_low** (`balanced_accuracy_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **balanced_accuracy_repeat_t_ci95_high** (`balanced_accuracy_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **balanced_accuracy_repeat_t_ci95_method** (`balanced_accuracy_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **f1_n_repeats** (`f1_n_repeats`): Harmonic mean of precision and recall for the named class. Formula: `F1 = 2 * precision * recall / (precision + recall)`
- **f1_mean** (`f1_mean`): Arithmetic mean of Harmonic mean of precision and recall for the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **f1_sample_sd** (`f1_sample_sd`): Sample standard deviation of Harmonic mean of precision and recall for the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **f1_repeat_t_ci95_low** (`f1_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Harmonic mean of precision and recall for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **f1_repeat_t_ci95_high** (`f1_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Harmonic mean of precision and recall for the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **f1_repeat_t_ci95_method** (`f1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **support_n_repeats** (`support_n_repeats`): Number of evaluated units whose true label is the named class. Formula: `support_c = TP_c + FN_c`
- **support_mean** (`support_mean`): Arithmetic mean of Number of evaluated units whose true label is the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **support_sample_sd** (`support_sample_sd`): Sample standard deviation of Number of evaluated units whose true label is the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **support_repeat_t_ci95_low** (`support_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Number of evaluated units whose true label is the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **support_repeat_t_ci95_high** (`support_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Number of evaluated units whose true label is the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **support_repeat_t_ci95_method** (`support_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **predicted_support_n_repeats** (`predicted_support_n_repeats`): Number of evaluated units predicted as the named class. Formula: `predicted support_c = TP_c + FP_c`
- **predicted_support_mean** (`predicted_support_mean`): Arithmetic mean of Number of evaluated units predicted as the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **predicted_support_sample_sd** (`predicted_support_sample_sd`): Sample standard deviation of Number of evaluated units predicted as the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **predicted_support_repeat_t_ci95_low** (`predicted_support_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Number of evaluated units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **predicted_support_repeat_t_ci95_high** (`predicted_support_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Number of evaluated units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **predicted_support_repeat_t_ci95_method** (`predicted_support_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **true_positive_n_repeats** (`true_positive_n_repeats`): Count of named-class units predicted as that class. Formula: `TP_c = sum_i 1[y_i=c and predicted_i=c]`
- **true_positive_mean** (`true_positive_mean`): Arithmetic mean of Count of named-class units predicted as that class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **true_positive_sample_sd** (`true_positive_sample_sd`): Sample standard deviation of Count of named-class units predicted as that class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **true_positive_repeat_t_ci95_low** (`true_positive_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of named-class units predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_positive_repeat_t_ci95_high** (`true_positive_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of named-class units predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_positive_repeat_t_ci95_method** (`true_positive_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **false_positive_n_repeats** (`false_positive_n_repeats`): Count of non-class units predicted as the named class. Formula: `FP_c = sum_i 1[y_i!=c and predicted_i=c]`
- **false_positive_mean** (`false_positive_mean`): Arithmetic mean of Count of non-class units predicted as the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **false_positive_sample_sd** (`false_positive_sample_sd`): Sample standard deviation of Count of non-class units predicted as the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **false_positive_repeat_t_ci95_low** (`false_positive_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of non-class units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_positive_repeat_t_ci95_high** (`false_positive_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of non-class units predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_positive_repeat_t_ci95_method** (`false_positive_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **false_negative_n_repeats** (`false_negative_n_repeats`): Count of named-class units not predicted as that class. Formula: `FN_c = sum_i 1[y_i=c and predicted_i!=c]`
- **false_negative_mean** (`false_negative_mean`): Arithmetic mean of Count of named-class units not predicted as that class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **false_negative_sample_sd** (`false_negative_sample_sd`): Sample standard deviation of Count of named-class units not predicted as that class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **false_negative_repeat_t_ci95_low** (`false_negative_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of named-class units not predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_negative_repeat_t_ci95_high** (`false_negative_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of named-class units not predicted as that class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **false_negative_repeat_t_ci95_method** (`false_negative_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **true_negative_n_repeats** (`true_negative_n_repeats`): Count of non-class units not predicted as the named class. Formula: `TN_c = sum_i 1[y_i!=c and predicted_i!=c]`
- **true_negative_mean** (`true_negative_mean`): Arithmetic mean of Count of non-class units not predicted as the named class. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **true_negative_sample_sd** (`true_negative_sample_sd`): Sample standard deviation of Count of non-class units not predicted as the named class. Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **true_negative_repeat_t_ci95_low** (`true_negative_repeat_t_ci95_low`): Reported 95% confidence bound or interval for Count of non-class units not predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_negative_repeat_t_ci95_high** (`true_negative_repeat_t_ci95_high`): Reported 95% confidence bound or interval for Count of non-class units not predicted as the named class. Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **true_negative_repeat_t_ci95_method** (`true_negative_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **roc_auc** (`roc_auc`): Area under the empirical receiver-operating-characteristic curve. Formula: `ROC-AUC = integral_0^1 TPR(FPR) dFPR (empirical trapezoidal area)`
- **pr_auc** (`pr_auc`): Area/average precision under the empirical precision-recall curve. Formula: `AP = sum_n (recall_n - recall_(n-1)) * precision_n`
- **roc_auc_applicability** (`roc_auc_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **pr_auc_applicability** (`pr_auc_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `early_three_model_repeat_metrics`

Persisted root CSV report table

- **timestamp** (`timestamp`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **status** (`status`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **error** (`error`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **group_id** (`group_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
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
- **n_windows** (`n_windows`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **n_extra_features** (`n_extra_features`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **window_balanced_accuracy** (`window_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **window_macro_f1** (`window_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **file_balanced_accuracy** (`file_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **file_macro_f1** (`file_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **subject_balanced_accuracy** (`subject_balanced_accuracy`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **subject_macro_f1** (`subject_macro_f1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **best_val_loss** (`best_val_loss`): Persisted source-table value for `best_val_loss`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_val_loss_epoch** (`best_val_loss_epoch`): Persisted source-table value for `best_val_loss_epoch`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_val_accuracy** (`best_val_accuracy`): Fraction of evaluated units assigned to their true class. Formula: `accuracy = (sum_c TP_c) / N`
- **best_val_accuracy_epoch** (`best_val_accuracy_epoch`): Fraction of evaluated units assigned to their true class. Formula: `accuracy = (sum_c TP_c) / N`
- **duration_sec** (`duration_sec`): Persisted source-table value for `duration_sec`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **report_path** (`report_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **learning_curve_png** (`learning_curve_png`): Persisted source-table value for `learning_curve_png`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **learning_curve_csv** (`learning_curve_csv`): Persisted source-table value for `learning_curve_csv`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_path** (`source_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model_display** (`model_display`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_macro_roc_auc_ovr** (`subject_macro_roc_auc_ovr`): Unweighted mean of valid one-vs-rest class ROC areas. Formula: `macro ROC-AUC = (1/K_valid) * sum_c ROC-AUC_c`

## `early_three_model_split_audit`

Persisted root CSV report table

- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **fold_count** (`fold_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **held_out_participant_count** (`held_out_participant_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **fold_sizes** (`fold_sizes`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **fold_roster_sha256** (`fold_roster_sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **roster** (`roster`): Persisted source-table value for `roster`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `early_three_model_summary_numeric`

Persisted root CSV report table

- **descriptive_rank** (`descriptive_rank`): Ordinal position after applying the table's declared sorting rule. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model_display** (`model_display`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **resolved_model** (`resolved_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
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
- **ranking_scope** (`ranking_scope`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **formal_v2_selection_eligible** (`formal_v2_selection_eligible`): Persisted source-table value for `formal_v2_selection_eligible`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

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

## `paired_participant_inference`

Persisted root CSV report table

- **comparison_family** (`comparison_family`): Persisted source-table value for `comparison_family`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparison_id** (`comparison_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **reference_case_id** (`reference_case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **candidate_case_id** (`candidate_case_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **metric** (`metric`): Persisted source-table value for `metric`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **candidate_minus_reference** (`candidate_minus_reference`): Paired candidate-minus-reference difference in the reported statistic Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **participant_cluster_delta_ci95_low** (`participant_cluster_delta_ci95_low`): Participant-cluster percentile-bootstrap 95% CI lower endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b = metric_candidate,b - metric_reference,b from the same participant draw; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_delta_ci95_high** (`participant_cluster_delta_ci95_high`): Participant-cluster percentile-bootstrap 95% CI upper endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b = metric_candidate,b - metric_reference,b from the same participant draw; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **raw_two_sided_p_value** (`raw_two_sided_p_value`): Null-hypothesis tail probability from the table's declared test. Formula: `two-sided p = Pr_H0(\|T*\| >= \|T_observed\|); exact statistic and resampling/rank distribution follow the declared test_method`
- **holm_adjusted_p_value** (`holm_adjusted_p_value`): Holm step-down multiplicity-adjusted P value. Formula: `ordered adjusted p_(i) = max_(j<=i) [(m-j+1) * p_(j)], capped at 1`
- **comparison_contract_status** (`comparison_contract_status`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **p_value_applicability** (`p_value_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **inference_role** (`inference_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **automatic_selection** (`automatic_selection`): Persisted source-table value for `automatic_selection`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `pairwise_repeat_metric_deltas`

Persisted root CSV report table

- **comparison_family** (`comparison_family`): Persisted source-table value for `comparison_family`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparison_id** (`comparison_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_role** (`comparison_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
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
- **automatic_selection** (`automatic_selection`): Persisted source-table value for `automatic_selection`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `parameter_inventory`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter_group** (`parameter_group`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **parameter** (`parameter`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **unique_value_count** (`unique_value_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **observed_values** (`observed_values`): Persisted source-table value for `observed_values`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **parameter_role** (`parameter_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **comparison_interpretation** (`comparison_interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `plot_01_three_model_metrics`

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
- **evaluation_level** (`evaluation_level`): Persisted source-table value for `evaluation_level`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_03_three_model_file_metrics`

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
- **evaluation_level** (`evaluation_level`): Persisted source-table value for `evaluation_level`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_04_three_model_window_metrics`

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
- **evaluation_level** (`evaluation_level`): Persisted source-table value for `evaluation_level`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `plot_05_shapeformer_paired_deltas`

Persisted root CSV report table

- **comparison_id** (`comparison_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **shapeformer_subject_BA** (`shapeformer_subject_BA`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **comparator_subject_BA** (`comparator_subject_BA`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **balanced_accuracy_delta** (`balanced_accuracy_delta`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_subject_macro_F1** (`shapeformer_subject_macro_F1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **comparator_subject_macro_F1** (`comparator_subject_macro_F1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **macro_f1_delta** (`macro_f1_delta`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_runtime_seconds** (`shapeformer_runtime_seconds`): Persisted source-table value for `shapeformer_runtime_seconds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparator_runtime_seconds** (`comparator_runtime_seconds`): Persisted source-table value for `comparator_runtime_seconds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **shapeformer_runtime_ratio** (`shapeformer_runtime_ratio`): Persisted source-table value for `shapeformer_runtime_ratio`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **macro_roc_auc_ovr_delta** (`macro_roc_auc_ovr_delta`): Paired candidate-minus-reference difference in Unweighted mean of valid one-vs-rest class ROC areas. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`

## `shapeformer_decision_evidence`

Persisted root CSV report table

- **comparison** (`comparison`): Persisted source-table value for `comparison`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **repeat** (`repeat`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **shapeformer_subject_BA** (`shapeformer_subject_BA`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **comparator_subject_BA** (`comparator_subject_BA`): Macro-average recall across the K declared classes. Formula: `BA = (1/K) * sum_c [TP_c / (TP_c + FN_c)]`
- **shapeformer_minus_comparator_BA** (`shapeformer_minus_comparator_BA`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_subject_macro_F1** (`shapeformer_subject_macro_F1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **comparator_subject_macro_F1** (`comparator_subject_macro_F1`): Unweighted mean of the K class-specific F1 scores. Formula: `macro-F1 = (1/K) * sum_c F1_c`
- **shapeformer_minus_comparator_macro_F1** (`shapeformer_minus_comparator_macro_F1`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_runtime_seconds** (`shapeformer_runtime_seconds`): Persisted source-table value for `shapeformer_runtime_seconds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparator_runtime_seconds** (`comparator_runtime_seconds`): Persisted source-table value for `comparator_runtime_seconds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **shapeformer_runtime_ratio** (`shapeformer_runtime_ratio`): Persisted source-table value for `shapeformer_runtime_ratio`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `shapeformer_decision_summary`

Persisted root CSV report table

- **comparison** (`comparison`): Persisted source-table value for `comparison`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **n_matched_repeat_seeds** (`n_matched_repeat_seeds`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **difference_direction** (`difference_direction`): Persisted source-table value for `difference_direction`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **formal_v2_inference** (`formal_v2_inference`): Persisted source-table value for `formal_v2_inference`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **shapeformer_minus_comparator_BA_n_repeats** (`shapeformer_minus_comparator_BA_n_repeats`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_BA_mean** (`shapeformer_minus_comparator_BA_mean`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_BA_sample_sd** (`shapeformer_minus_comparator_BA_sample_sd`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_BA_repeat_t_ci95_low** (`shapeformer_minus_comparator_BA_repeat_t_ci95_low`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_BA_repeat_t_ci95_high** (`shapeformer_minus_comparator_BA_repeat_t_ci95_high`): Paired candidate-minus-reference difference in Macro-average recall across the K declared classes. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_BA_repeat_t_ci95_method** (`shapeformer_minus_comparator_BA_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **shapeformer_minus_comparator_macro_F1_n_repeats** (`shapeformer_minus_comparator_macro_F1_n_repeats`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_macro_F1_mean** (`shapeformer_minus_comparator_macro_F1_mean`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_macro_F1_sample_sd** (`shapeformer_minus_comparator_macro_F1_sample_sd`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_macro_F1_repeat_t_ci95_low** (`shapeformer_minus_comparator_macro_F1_repeat_t_ci95_low`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_macro_F1_repeat_t_ci95_high** (`shapeformer_minus_comparator_macro_F1_repeat_t_ci95_high`): Paired candidate-minus-reference difference in Unweighted mean of the K class-specific F1 scores. Formula: `delta = metric_candidate - metric_reference on the declared matched unit`
- **shapeformer_minus_comparator_macro_F1_repeat_t_ci95_method** (`shapeformer_minus_comparator_macro_F1_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **shapeformer_runtime_ratio_n_repeats** (`shapeformer_runtime_ratio_n_repeats`): Persisted source-table value for `shapeformer_runtime_ratio_n_repeats`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **shapeformer_runtime_ratio_mean** (`shapeformer_runtime_ratio_mean`): Arithmetic mean of the reported statistic over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **shapeformer_runtime_ratio_sample_sd** (`shapeformer_runtime_ratio_sample_sd`): Sample standard deviation of the reported statistic Formula: `sample SD = sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **shapeformer_runtime_ratio_repeat_t_ci95_low** (`shapeformer_runtime_ratio_repeat_t_ci95_low`): Reported 95% confidence bound or interval for the reported statistic Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **shapeformer_runtime_ratio_repeat_t_ci95_high** (`shapeformer_runtime_ratio_repeat_t_ci95_high`): Reported 95% confidence bound or interval for the reported statistic Formula: `repeat CI95 = 1 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **shapeformer_runtime_ratio_repeat_t_ci95_method** (`shapeformer_runtime_ratio_repeat_t_ci95_method`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **participant_cluster_delta_ci95_low** (`participant_cluster_delta_ci95_low`): Participant-cluster percentile-bootstrap 95% CI lower endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b = metric_candidate,b - metric_reference,b from the same participant draw; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_delta_ci95_high** (`participant_cluster_delta_ci95_high`): Participant-cluster percentile-bootstrap 95% CI upper endpoint for the reported statistic Formula: `CI95 = [Q_0.025(T_b), Q_0.975(T_b)], b=1..B; T_b = metric_candidate,b - metric_reference,b from the same participant draw; each draw resamples participant IDs with replacement under the declared strata and carries every repeat/row belonging to each sampled participant cluster`
- **participant_cluster_ci_applicability** (`participant_cluster_ci_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `source_evidence`

Persisted root CSV report table

- **source_study** (`source_study`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_kind** (`source_kind`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_file** (`source_file`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **size_bytes** (`size_bytes`): Persisted source-table value for `size_bytes`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **sha256** (`sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `table_figure_pairs`

Persisted root CSV report table

- **figure_id** (`figure_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **title** (`title`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **figure_path** (`figure_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **table_path** (`table_path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **interpretation** (`interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
