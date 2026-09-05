# Table column definitions and formulas

This documentation catalog describes source report tables. The catalog artifact excludes itself to prevent recursive documentation rows.

## `cross_study_contract_summary`

Persisted root CSV report table

- **report_id** (`report_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_studies** (`source_studies`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **models** (`models`): Persisted source-table value for `models`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **native_fs_hz** (`native_fs_hz`): Persisted source-table value for `native_fs_hz`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **dl_target_fs_hz** (`dl_target_fs_hz`): Persisted source-table value for `dl_target_fs_hz`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **batch_size** (`batch_size`): Persisted source-table value for `batch_size`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **all_config_evaluation_window_seconds** (`all_config_evaluation_window_seconds`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **all_config_evaluation_hop_seconds** (`all_config_evaluation_hop_seconds`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **analysis_grid_evaluation_hop_seconds** (`analysis_grid_evaluation_hop_seconds`): Persisted source-table value for `analysis_grid_evaluation_hop_seconds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **analysis_grid_train_overlap_percent** (`analysis_grid_train_overlap_percent`): Persisted source-table value for `analysis_grid_train_overlap_percent`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **derived_analysis_grid_training_hop_seconds** (`derived_analysis_grid_training_hop_seconds`): Persisted source-table value for `derived_analysis_grid_training_hop_seconds`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **best_epoch_selection** (`best_epoch_selection`): Persisted source-table value for `best_epoch_selection`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **roles** (`roles`): Persisted source-table value for `roles`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **comparison_boundary** (`comparison_boundary`): Persisted source-table value for `comparison_boundary`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`

## `cross_study_split_audit`

Persisted root CSV report table

- **split_seed** (`split_seed`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **01_early_three_model_matched_fold_roster_sha256** (`01_early_three_model_matched_fold_roster_sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **02_20260608_1206_overfitting_sweep_stage1_rank2_fold_roster_sha256** (`02_20260608_1206_overfitting_sweep_stage1_rank2_fold_roster_sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **03_20260625_2320_overfitting_sweep_stage1_rank2_fold_roster_sha256** (`03_20260625_2320_overfitting_sweep_stage1_rank2_fold_roster_sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **04_20260630_0630_overfitting_sweep_generalization_rank2_fold_roster_sha256** (`04_20260630_0630_overfitting_sweep_generalization_rank2_fold_roster_sha256`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **all_report_units_share_exact_participant_fold_roster** (`all_report_units_share_exact_participant_fold_roster`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **independence_interpretation** (`independence_interpretation`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

## `report_index`

Persisted root CSV report table

- **report_id** (`report_id`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **path** (`path`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **source_studies** (`source_studies`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **configuration_count** (`configuration_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **run_count** (`run_count`): Number of units satisfying the column's stated inclusion condition. Formula: `count = sum_i 1[unit i satisfies the stated condition]`
- **leader** (`leader`): Persisted source-table value for `leader`; producer-specific semantics are not reinterpreted by the shared table renderer. Formula: `N/A — direct persisted/source-defined value; the table renderer applies no additional arithmetic.`
- **leader_subject_ba_mean** (`leader_subject_ba_mean`): Arithmetic mean of Macro-average recall across the K declared classes. over the table's declared units Formula: `mean = (1/n) * sum_i x_i`
- **scientific_status** (`scientific_status`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
