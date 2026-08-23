# V2 study summary — staged_static_06_batch_lr_successive_halving_v1__screen_5epoch_reduced_cv

> This report is descriptive evidence for manual review. It does not automatically select a final use case or winner.

## Scientific context

- Study kind: catalog_sweep
- Purpose: Tune batch size and learning rate for InceptionTimeFull using a reduced-resource first rung and complete 5x5 tuning CV for promoted configurations. Phase=screen_5epoch_reduced_cv.
- Position in use-case selection flow: Stage 6 after B0/B2/B7 selection and before regularization/channel ablations.
- Decision role: screening
- Thesis sections: ["Successive-halving batch-size and learning-rate tuning"]
- Catalog: configs/formal_experiment_catalog_v2.yaml (scope=selected_ordinary, balance=line_b)
- Reference case: N/A

## Run controls and completeness

- Repeats requested: [0, 1, 2, 3, 4]
- Folds requested: [0]
- Case-level jobs requested: 1
- Effective jobs: 1
- Planned / passed / failed / not-run cases: 6 / 6 / 0 / 0
- Planned / reported / passed / failed / not-run cells: 30 / 30 / 30 / 0 / 0
- Resume-skipped passed cases: 0

## Test models, modules, inputs, and fixed parameters

The identical standalone table is in [TEST_COMPONENTS.md](TEST_COMPONENTS.md); machine-readable copies are `tables/test_components.csv` and `.json`. Input data are reported as dataset/path, signal view, channels, units, rate, and windows—not hashes.

| Cases / phases | Component role | Model / module | State | Input data (values and paths; no hashes) | Detailed fixed parameters | Algorithm and kernel (≤300 chars) |
|---|---|---|---|---|---|---|
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | aggregation | line_b_equal_role_families | enabled | {"input_data":"held-out window/file probabilities","roles":["B","R1","R2","R3","R4"]} | {"balance_line":"line_b_equal_role_families","direct_all_window_participant_mean":false,"file_to_role":"ordinary_mean","hierarchy":["window","file","role","participant"],"missing_role_policy":"mean_available_roles","quality_weight_levels":[],"quality_weight_source":"none","quality_weighting":false,"role_to_participant":"ordinary_mean","window_to_file":"ordinary_mean"} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | dataset_adapter | frailty3_m2_20260815_a054800abda272f6 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"allow_qc_excluded_records":false,"channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"class_id_order":[0,1,2],"class_name_order":["Pre-Frail","Robust/Non-Frail","Young"],"expected_participant_count":29,"expected_record_count":261,"manifest_version":"internal_records_v2","path":"manifests/internal_records_v2.csv","source_dataset_id":"frailty3_m2_20260815_a054800abda272f6"} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | denoiser | identity | identity_or_disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["filtered RED/IR","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"declared_reducer_version":"identity_v1","degraded_policy":"drop","denoiser_enabled":false,"failure_action":"no_result_no_fallback","reducer":"identity","resolved_parameters":{},"runtime_reducer_version":"identity_exact_v1"} | 逐样本复制双波长 PPG，不估计或抑制伪影；内核：恒等映射与同时间网格校验，作为未去噪直接对照。 |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | evaluation | balanced_accuracy | enabled | {"class_order":["Pre-Frail","Robust/Non-Frail","Young"],"input_data":"held-out participant predictions and frailty labels"} | {"calibration_metrics":["multiclass_brier","expected_calibration_error"],"confidence_interval":"participant_cluster_bootstrap_two_sided_95","independent_test_available":false,"metric_prefix":"oof_validation_","metrics":["balanced_accuracy","macro_f1","per_class_precision_recall_f1","worst_class_recall","worst_class_f1","confusion_matrix","coverage"],"paired_delta_key":["repeat_index","fold_index","participant_id"],"primary_metric":"balanced_accuracy","rank_incomplete_configs":false,"ranking":{"automatic_final_selection":false,"manual_multiple_final_versions_allowed":true,"max_qualified_per_comparison_group":10,"preserve_ablation_provenance":true,"sort_key":"participant_level_mean_balanced_accuracy"},"statistics":{"affects_automatic_selection":false,"bootstrap_replicates":10000,"cluster_unit":"participant_with_all_five_repeat_oof_predictions","confidence_interval":"two_sided_95_percentile","lcb95_metrics":["participant_level_mean_balanced_accuracy","participant_level_mean_macro_f1"],"lcb95_percentile":2.5,"multiplicity_correction":"holm_within_comparison_family","paired_exchange_unit":"participant","paired_permutation_replicates":100000,"seed":42},"unit":"participant"} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | feature_extractor | feature_vector_282_v3 | auxiliary_not_classifier_input | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","engineering_window":{"cap_fraction_per_file":null,"cap_per_file":null,"end_alignment":"left_start_regular_grid","hop_s":2.0,"length_s":10.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"input_views":["x_analysis/x_native","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"enabled_groups":["ppi_basic_rate","hrv_time_domain","hrv_spectral","hrv_nonlinear","morphology","dual_optical","engineering_summary"],"engineering_sequence_schema":"engineering_10s_hop2s_thesis_115_v3","file_aggregation":["mean","population_sd"],"file_vector_schema":"feature_vector_282_v3","matrix_k":150,"matrix_schema":"ordered_feature_matrix_d115_by_150_engineering_v4","missing_physiology_encoding":"nan_and_validity_false","prv_library_comparison_scope":"fixed_ppi_vectors_only_no_classifier","prv_primary_backend":"local_manual","rate_prv_min_duration_s":8.0,"rate_prv_min_peaks":5,"registry_id":"feature_vector_282_v3","sample_entropy":{"m":2,"min_intervals":200,"r_sd_fraction":0.2},"spectral_bands_hz":{"hf":[0.15,0.4],"lf":[0.04,0.15],"vlf":[0.003,0.04]},"spectral_prv_min_coverage":0.8,"spectral_prv_min_duration_s":300.0,"spectral_prv_min_intervals":200,"tachogram_fs_hz":4.0,"technical_metadata_allowed":false,"time_prv_min_coverage":0.8,"time_prv_min_duration_s":60.0,"time_prv_min_intervals":30} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | imu_preprocessing | profile_a_lowpass_0p3hz | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["AX","AY","AZ","GX","GY","GZ"],"manifest_path":"manifests/internal_records_v2.csv","output_view":"processed_imu_physical","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"calibration_start_s":5.0,"calibration_stop_s":100.0,"comparison_method":"calibrated_roll_pitch_ekf","failure_action":"fail_closed","gravity_filter_order":4,"gravity_lowpass_hz":0.3,"gravity_method":"profile_a_lowpass_0p3hz","gravity_mps2":9.81,"initialization":"same_participant_static_calibration","output_units":{"acceleration":"m/s^2","gyroscope":"rad/s","jerk":"m/s^3"},"required_axes":6,"sensor_filter_order":3,"sensor_lowpass_acc_hz":20.0,"sensor_lowpass_gyro_hz":40.0} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | motion_detector | formal_local_supervised_motion_detector_v2 | disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_view":"RED/IR + processed physical A_dyn/GX/GY/GZ","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":64,"device":"cuda","enabled":false,"evidence_path":null,"threshold_source":"bundle_frozen","window_probability_aggregation":"median"} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | peak_detector | aboy_project_v1 | enabled | {"channels":["RED","IR"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_view":"x_analysis/x_native","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"detector_id":"aboy_project_v1","failure_action":"fail_closed_no_fallback","min_observation_sec":8.0,"min_peaks":5} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | ppg_preprocessing | butterworth_sos | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["RED","IR"],"input_view":"repaired native PPG","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"analysis_view":{"additional_filter":"none","direct_source":"x_filter_0p2_to_8hz","non_identity_semantics":"rate_only","non_identity_source":"aligned_x_ar"},"gap_repair":{"all_missing_channel_action":"reject_record","edge_extrapolation":false,"max_gap_samples":100,"method":"linear_inside_only"},"ppg_filter":{"family":"butterworth_sos","high_hz":8.0,"low_hz":0.2,"notch_enabled":false,"order":3,"phase":"zero_phase","short_signal_policy":"reject"}} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | representation | raw | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"input_contract":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"representation_mode":"raw"} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | signal_views_and_scaling | parallel_physical_analysis_and_dl_views | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"views":["processed_imu_physical","x_dl_all8_window_norm","x_analysis/x_native"]} | {"dl_resampling":{"enabled":true,"method":"polyphase_anti_alias","preserve_feature_grid_hz":400.0,"target_fs_hz":64.0},"normalization":{"clip_after_scale":[-8.0,8.0],"iqr_fallback":"standard_deviation_then_finite_one","mad_consistency_divisor":0.6744897501960817,"raw_imu":"none","raw_ppg":"per_window_robust","robust_iqr_divisor":1.349,"scale_epsilon":1e-08,"standard_ddof":0}} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | split_registry | frailty3_future_corrected_sgkf5_v2 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","groups":"participant_id","labels":"frailty_class","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"n_repeats":5,"n_splits":5,"path":"splits/sgkf5_repeated_grouped_5x5_v2.csv","registry_id":"frailty3_future_corrected_sgkf5_v2","runtime_recompute":false,"split_seeds":[42,10042,20042,30042,40042]} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | sqi | quality_off | disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["x_analysis","pulse train","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"components":[],"failure_action":"fail_closed","fit_scope":"not_applied_off","flatline_duration_s":1.0,"high_quality_rule":"not_applied","long_gap_max_samples":100,"mode":"off","window_selection":{"application_scope":"outer_train_only","keep_fraction":1.0,"policy":"none","score_algorithm":"legacy_cardiac_motion_window_sqi_v1"}} |  |
| b16_lr1e-3 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":16,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} |  |
| b16_lr1e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":16,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} |  |
| b16_lr3e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":16,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0003,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} |  |
| b32_lr1e-3 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} |  |
| b32_lr1e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} |  |
| b32_lr3e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0003,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} |  |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | window_planner | window_plan_v1 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["x_dl_all8_window_norm","x_analysis/x_native","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"engineering":{"cap_fraction_per_file":null,"cap_per_file":null,"end_alignment":"left_start_regular_grid","hop_s":2.0,"length_s":10.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"raw_dl":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"shared_planner_version":"window_plan_v1"} |  |

## Seed and data-split reproducibility

- Audit status: **PASS**
- Scope: manifest_cases_and_selected_artifact_roots_only
- Planned / observed selected cells: 30 / 30
- Split seeds by repeat: {"0": [42], "1": [10042], "2": [20042], "3": [30042], "4": [40042]}
- Errors / not-verifiable items: 0 / 0
- This is report-only evidence; it never gates training or report generation.

| Case | Selected status | Selected attempt | Excluded attempts | Planned cells | Observed cells | Declared seed policy | Effective seed policy | Split seeds | Model seeds | Orchestration seeds | Evaluation seeds | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-4 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["outer_cv_repeat_seed_equals_split_seed"] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42] | PASS |
| b16_lr3e-4 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["outer_cv_repeat_seed_equals_split_seed"] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42] | PASS |
| b16_lr1e-3 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["outer_cv_repeat_seed_equals_split_seed"] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42] | PASS |
| b32_lr1e-4 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["outer_cv_repeat_seed_equals_split_seed"] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42] | PASS |
| b32_lr3e-4 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["outer_cv_repeat_seed_equals_split_seed"] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42] | PASS |
| b32_lr1e-3 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["outer_cv_repeat_seed_equals_split_seed"] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42, 10042, 20042, 30042, 40042] | [42] | PASS |

<details><summary>Per-cell seed and split evidence</summary>

| Case | Repeat | Fold | Cell status | Attempt | Declared policy | Effective policy | Split seed | Orchestration seed | Training seed | Model/member seeds | Member-seed semantics | Evaluation seed | Epoch RNG rows | Split CSV SHA256 | Fold membership SHA256 | Train participants | OOF participants | Train/OOF overlap | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-4 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| b16_lr1e-4 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 10042 | 10042 | 10042 | [10042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| b16_lr1e-4 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 20042 | 20042 | 20042 | [20042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| b16_lr1e-4 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 30042 | 30042 | 30042 | [30042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| b16_lr1e-4 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 40042 | 40042 | 40042 | [40042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| b16_lr3e-4 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| b16_lr3e-4 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 10042 | 10042 | 10042 | [10042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| b16_lr3e-4 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 20042 | 20042 | 20042 | [20042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| b16_lr3e-4 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 30042 | 30042 | 30042 | [30042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| b16_lr3e-4 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 40042 | 40042 | 40042 | [40042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| b16_lr1e-3 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| b16_lr1e-3 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 10042 | 10042 | 10042 | [10042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| b16_lr1e-3 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 20042 | 20042 | 20042 | [20042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| b16_lr1e-3 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 30042 | 30042 | 30042 | [30042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| b16_lr1e-3 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 40042 | 40042 | 40042 | [40042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| b32_lr1e-4 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| b32_lr1e-4 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 10042 | 10042 | 10042 | [10042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| b32_lr1e-4 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 20042 | 20042 | 20042 | [20042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| b32_lr1e-4 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 30042 | 30042 | 30042 | [30042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| b32_lr1e-4 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 40042 | 40042 | 40042 | [40042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| b32_lr3e-4 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| b32_lr3e-4 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 10042 | 10042 | 10042 | [10042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| b32_lr3e-4 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 20042 | 20042 | 20042 | [20042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| b32_lr3e-4 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 30042 | 30042 | 30042 | [30042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| b32_lr3e-4 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 40042 | 40042 | 40042 | [40042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| b32_lr1e-3 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| b32_lr1e-3 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 10042 | 10042 | 10042 | [10042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| b32_lr1e-3 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 20042 | 20042 | 20042 | [20042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| b32_lr1e-3 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 30042 | 30042 | 30042 | [30042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| b32_lr1e-3 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | outer_cv_repeat_seed_equals_split_seed | 40042 | 40042 | 40042 | [40042] | N/A_single_model_training_seed_alias | 42 | 5 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |

</details>

### Frozen split roster

| Repeat | Fold | Split seed | Split CSV SHA256 | Declared authority JSON SHA256 | Declared authority payload SHA256 | Train participants | OOF participants | Overlap | Matching cases | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["b16_lr1e-3", "b16_lr1e-4", "b16_lr3e-4", "b32_lr1e-3", "b32_lr1e-4", "b32_lr3e-4"] | PASS |
| 1 | 0 | 10042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 22 | 7 | 0 | ["b16_lr1e-3", "b16_lr1e-4", "b16_lr3e-4", "b32_lr1e-3", "b32_lr1e-4", "b32_lr3e-4"] | PASS |
| 2 | 0 | 20042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["b16_lr1e-3", "b16_lr1e-4", "b16_lr3e-4", "b32_lr1e-3", "b32_lr1e-4", "b32_lr3e-4"] | PASS |
| 3 | 0 | 30042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["b16_lr1e-3", "b16_lr1e-4", "b16_lr3e-4", "b32_lr1e-3", "b32_lr1e-4", "b32_lr3e-4"] | PASS |
| 4 | 0 | 40042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["b16_lr1e-3", "b16_lr1e-4", "b16_lr3e-4", "b32_lr1e-3", "b32_lr1e-4", "b32_lr3e-4"] | PASS |

## Varied and controlled parameters

- Explicit deterministic sparse catalog profiles; this is a screening comparison, not a single-factor causal ablation.
- Search method: deterministic_sparse_profiles
- Runtime parameter sampling: False
- Profile-design seed: 42
- Interpretation: Declared screen_5epoch_reduced_cv tuning phase; no final-test claim and no automatic final-model selection.

The complete resolved varied/controlled tables are [varied_parameters.csv](tables/varied_parameters.csv) and [controlled_parameters.csv](tables/controlled_parameters.csv). Execution controls such as jobs are not scientific grid variables.

<details><summary>Complete controlled-parameter list (226 rows)</summary>

| Controlled parameter | Resolved value |
|---|---|
| aggregation.balance_line | line_b_equal_role_families |
| aggregation.direct_all_window_participant_mean | False |
| aggregation.file_to_role | ordinary_mean |
| aggregation.hierarchy | ["window", "file", "role", "participant"] |
| aggregation.missing_role_policy | mean_available_roles |
| aggregation.quality_weight_levels | [] |
| aggregation.quality_weight_source | none |
| aggregation.quality_weighting | False |
| aggregation.role_to_participant | ordinary_mean |
| aggregation.window_to_file | ordinary_mean |
| artifact.degraded_policy | drop |
| artifact.denoiser_enabled | False |
| artifact.failure_action | no_result_no_fallback |
| artifact.motion_detector.batch_size | 64 |
| artifact.motion_detector.device | cuda |
| artifact.motion_detector.evidence_path |  |
| artifact.motion_detector.expected_evidence_sha256 |  |
| artifact.motion_detector.threshold_source | bundle_frozen |
| artifact.motion_detector.window_probability_aggregation | median |
| artifact.motion_detector_enabled | False |
| artifact.non_identity_output_contract | rate_only |
| artifact.reducer | identity |
| artifact.reducer_version | identity_v1 |
| artifact.selection_scope | run_before_evaluation |
| evaluation.calibration_metrics | ["multiclass_brier", "expected_calibration_error"] |
| evaluation.confidence_interval | participant_cluster_bootstrap_two_sided_95 |
| evaluation.independent_test_available | False |
| evaluation.metric_prefix | oof_validation_ |
| evaluation.metrics | ["balanced_accuracy", "macro_f1", "per_class_precision_recall_f1", "worst_class_recall", "worst_class_f1", "confusion_matrix", "coverage"] |
| evaluation.paired_delta_key | ["repeat_index", "fold_index", "participant_id"] |
| evaluation.primary_metric | balanced_accuracy |
| evaluation.rank_incomplete_configs | False |
| evaluation.ranking.automatic_final_selection | False |
| evaluation.ranking.manual_multiple_final_versions_allowed | True |
| evaluation.ranking.max_qualified_per_comparison_group | 10 |
| evaluation.ranking.preserve_ablation_provenance | True |
| evaluation.ranking.sort_key | participant_level_mean_balanced_accuracy |
| evaluation.statistics.affects_automatic_selection | False |
| evaluation.statistics.bootstrap_replicates | 10000 |
| evaluation.statistics.cluster_unit | participant_with_all_five_repeat_oof_predictions |
| evaluation.statistics.confidence_interval | two_sided_95_percentile |
| evaluation.statistics.lcb95_metrics | ["participant_level_mean_balanced_accuracy", "participant_level_mean_macro_f1"] |
| evaluation.statistics.lcb95_percentile | 2.5 |
| evaluation.statistics.multiplicity_correction | holm_within_comparison_family |
| evaluation.statistics.paired_exchange_unit | participant |
| evaluation.statistics.paired_permutation_replicates | 100000 |
| evaluation.statistics.seed | 42 |
| evaluation.unit | participant |
| features.enabled_groups | ["ppi_basic_rate", "hrv_time_domain", "hrv_spectral", "hrv_nonlinear", "morphology", "dual_optical", "engineering_summary"] |
| features.engineering_sequence_schema | engineering_10s_hop2s_thesis_115_v3 |
| features.file_aggregation | ["mean", "population_sd"] |
| features.file_vector_schema | feature_vector_282_v3 |
| features.matrix_k | 150 |
| features.matrix_schema | ordered_feature_matrix_d115_by_150_engineering_v4 |
| features.missing_physiology_encoding | nan_and_validity_false |
| features.prv_library_comparison_scope | fixed_ppi_vectors_only_no_classifier |
| features.prv_primary_backend | local_manual |
| features.rate_prv_min_duration_s | 8.0 |
| features.rate_prv_min_peaks | 5 |
| features.registry_id | feature_vector_282_v3 |
| features.sample_entropy.m | 2 |
| features.sample_entropy.min_intervals | 200 |
| features.sample_entropy.r_sd_fraction | 0.2 |
| features.spectral_bands_hz.hf | [0.15, 0.4] |
| features.spectral_bands_hz.lf | [0.04, 0.15] |
| features.spectral_bands_hz.vlf | [0.003, 0.04] |
| features.spectral_prv_min_coverage | 0.8 |
| features.spectral_prv_min_duration_s | 300.0 |
| features.spectral_prv_min_intervals | 200 |
| features.tachogram_fs_hz | 4.0 |
| features.technical_metadata_allowed | False |
| features.time_prv_min_coverage | 0.8 |
| features.time_prv_min_duration_s | 60.0 |
| features.time_prv_min_intervals | 30 |
| manifest.allow_qc_excluded_records | False |
| manifest.channel_order | ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"] |
| manifest.class_id_order | [0, 1, 2] |
| manifest.class_name_order | ["Pre-Frail", "Robust/Non-Frail", "Young"] |
| manifest.expected_participant_count | 29 |
| manifest.expected_record_count | 261 |
| manifest.manifest_version | internal_records_v2 |
| manifest.path | manifests/internal_records_v2.csv |
| manifest.source_dataset_id | frailty3_m2_20260815_a054800abda272f6 |
| manifest.source_manifest_sha256 | bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90 |
| model.dilation | 1 |
| model.dropout | 0.2 |
| model.input_channel_order | ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"] |
| model.input_channels | 8 |
| model.input_channels_resolution | canonical_frailty_raw_8 |
| model.kernel_sizes | [39, 19, 9] |
| model.mask_aware_pooling | True |
| model.model_id | InceptionTimeFull |
| model.n_classes | 3 |
| model.seed_policy | outer_cv_repeat_seed_equals_split_seed |
| model.variant | single_network |
| output.overwrite_existing | False |
| output.parquet_missing_dependency_action | fail_closed |
| output.root | artifacts/runs |
| output.strict_json | True |
| output.write_file_oof | True |
| output.write_member_oof | False |
| output.write_parquet | True |
| output.write_subject_oof | True |
| output.write_window_oof | True |
| quality.components | [] |
| quality.failure_action | fail_closed |
| quality.fit_scope | not_applied_off |
| quality.flatline_duration_s | 1.0 |
| quality.high_quality_rule | not_applied |
| quality.long_gap_max_samples | 100 |
| quality.mode | off |
| quality.window_selection.application_scope | outer_train_only |
| quality.window_selection.keep_fraction | 1.0 |
| quality.window_selection.policy | none |
| quality.window_selection.score_algorithm | legacy_cardiac_motion_window_sqi_v1 |
| representation_mode | raw |
| roles | ["B", "R1", "R2", "R3", "R4"] |
| schema_version | ppg_frailty.pipeline_config.v2 |
| signal.accelerometer_input_unit | g |
| signal.analysis_view.additional_filter | none |
| signal.analysis_view.direct_source | x_filter_0p2_to_8hz |
| signal.analysis_view.non_identity_semantics | rate_only |
| signal.analysis_view.non_identity_source | aligned_x_ar |
| signal.channel_order | ["RED", "IR", "AX", "AY", "AZ", "GX", "GY", "GZ"] |
| signal.dl_resampling.enabled | True |
| signal.dl_resampling.method | polyphase_anti_alias |
| signal.dl_resampling.preserve_feature_grid_hz | 400.0 |
| signal.dl_resampling.target_fs_hz | 64.0 |
| signal.gap_repair.all_missing_channel_action | reject_record |
| signal.gap_repair.edge_extrapolation | False |
| signal.gap_repair.max_gap_samples | 100 |
| signal.gap_repair.method | linear_inside_only |
| signal.gyroscope_input_unit | deg/s |
| signal.imu.calibration_start_s | 5.0 |
| signal.imu.calibration_stop_s | 100.0 |
| signal.imu.comparison_method | calibrated_roll_pitch_ekf |
| signal.imu.failure_action | fail_closed |
| signal.imu.gravity_filter_order | 4 |
| signal.imu.gravity_lowpass_hz | 0.3 |
| signal.imu.gravity_method | profile_a_lowpass_0p3hz |
| signal.imu.gravity_mps2 | 9.81 |
| signal.imu.initialization | same_participant_static_calibration |
| signal.imu.output_units.acceleration | m/s^2 |
| signal.imu.output_units.gyroscope | rad/s |
| signal.imu.output_units.jerk | m/s^3 |
| signal.imu.required_axes | 6 |
| signal.imu.sensor_filter_order | 3 |
| signal.imu.sensor_lowpass_acc_hz | 20.0 |
| signal.imu.sensor_lowpass_gyro_hz | 40.0 |
| signal.internal_fs_hz | 400.0 |
| signal.normalization.clip_after_scale | [-8.0, 8.0] |
| signal.normalization.iqr_fallback | standard_deviation_then_finite_one |
| signal.normalization.mad_consistency_divisor | 0.6744897501960817 |
| signal.normalization.raw_imu | none |
| signal.normalization.raw_ppg | per_window_robust |
| signal.normalization.robust_iqr_divisor | 1.349 |
| signal.normalization.scale_epsilon | 1e-08 |
| signal.normalization.standard_ddof | 0 |
| signal.peak_detector.detector_id | aboy_project_v1 |
| signal.peak_detector.failure_action | fail_closed_no_fallback |
| signal.peak_detector.min_observation_sec | 8.0 |
| signal.peak_detector.min_peaks | 5 |
| signal.ppg_filter.family | butterworth_sos |
| signal.ppg_filter.high_hz | 8.0 |
| signal.ppg_filter.low_hz | 0.2 |
| signal.ppg_filter.notch_enabled | False |
| signal.ppg_filter.order | 3 |
| signal.ppg_filter.phase | zero_phase |
| signal.ppg_filter.short_signal_policy | reject |
| signal.ppg_native_unit | raw_counts |
| splits.n_repeats | 5 |
| splits.n_splits | 5 |
| splits.path | splits/sgkf5_repeated_grouped_5x5_v2.csv |
| splits.registry_id | frailty3_future_corrected_sgkf5_v2 |
| splits.runtime_recompute | False |
| splits.source_registry_file_sha256 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c |
| splits.source_registry_payload_sha256 | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 |
| splits.split_seeds | [42, 10042, 20042, 30042, 40042] |
| training.cache_policy | disabled |
| training.class_count_basis | row |
| training.class_weight_beta | 0.999 |
| training.class_weighting | inverse_frequency |
| training.classifier_role_families | ["B", "R"] |
| training.deterministic_algorithms | True |
| training.device | cuda |
| training.epoch_profile | configured_5 |
| training.epoch_rule | fixed_epoch |
| training.execution_mode | formal |
| training.fixed_epochs | 5 |
| training.focal_gamma | 2.0 |
| training.gradient_clip_norm |  |
| training.inner_grouped_folds | 0 |
| training.inner_patience | 0 |
| training.label_smoothing | 0.0 |
| training.loss | cross_entropy |
| training.maximum_inner_epochs | 0 |
| training.n_classes | 3 |
| training.num_workers | 0 |
| training.optimizer | adamw |
| training.optimizer_parameters.amsgrad | False |
| training.optimizer_parameters.betas | [0.9, 0.999] |
| training.optimizer_parameters.eps | 1e-08 |
| training.optimizer_parameters.maximize | False |
| training.outer_labels_visible_to_trainer | False |
| training.participant_window_quota | all |
| training.refit_on_all_outer_training | True |
| training.sampler | exhaustive_shuffle_without_replacement |
| training.samples_per_epoch |  |
| training.seed | 42 |
| training.training_balance | equal_role_families |
| training.weight_decay | 0.0001 |
| windows.engineering.cap_fraction_per_file |  |
| windows.engineering.cap_per_file |  |
| windows.engineering.end_alignment | left_start_regular_grid |
| windows.engineering.hop_s | 2.0 |
| windows.engineering.length_s | 10.0 |
| windows.engineering.min_valid_fraction | 1.0 |
| windows.engineering.padding | none_complete_windows_only |
| windows.raw_dl.cap_fraction_per_file |  |
| windows.raw_dl.cap_per_file | 128 |
| windows.raw_dl.end_alignment | include_right_aligned_if_distinct |
| windows.raw_dl.hop_s | 2.5 |
| windows.raw_dl.length_s | 5.0 |
| windows.raw_dl.min_valid_fraction | 1.0 |
| windows.raw_dl.padding | none_complete_windows_only |
| windows.shared_planner_version | window_plan_v1 |

</details>

## Predictive ranking

Primary ranking is by participant-level, repeat-recomputed abstention-aware balanced accuracy, then participant coverage and abstention-aware Macro-F1. Conditional retained-only metrics remain visible but never lead the ranking; deployment measurements do not filter this table.

| Rank | Case | Abstention-aware BA, mean ± SD (%) | Abstention-aware precision | Abstention-aware recall | Abstention-aware Macro-F1, mean ± SD (%) | Participant coverage | Abstentions | Abstentions by class | Conditional BA, mean ± SD (%) | Conditional Macro-F1, mean ± SD (%) | Macro ROC AUC, mean ± SD (%) | Macro PR AUC, mean ± SD (%) | Conditional BA LCB95 | Conditional Macro-F1 LCB95 | Aware worst-fold BA | Conditional worst-fold BA | Worst recall | Worst F1 | Source | Frailty endpoint | Motion auxiliary outer-OOF | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b16_lr1e-3 | 62.2 ± 21.3 | 0.5456 | 0.6222 | 55.7 ± 22.7 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 62.2 ± 21.3 | 55.7 ± 22.7 | 77.4 ± 11.7 | 76.6 ± 7.7 | 0.4191 | 0.3405 | 0.3333 | 0.3333 | 0.5000 | 0.5926 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 2 | b32_lr3e-4 | 58.9 ± 18.7 | 0.5289 | 0.5889 | 50.3 ± 23.9 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 58.9 ± 18.7 | 50.3 ± 23.9 | 70.2 ± 12.4 | 70.8 ± 11.8 | 0.4108 | 0.2746 | 0.3333 | 0.3333 | 0.4000 | 0.4706 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 3 | b16_lr3e-4 | 55.6 ± 26.1 | 0.5244 | 0.5556 | 48.8 ± 28.4 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 55.6 ± 26.1 | 48.8 ± 28.4 | 71.9 ± 13.6 | 71.7 ± 11.8 | 0.3071 | 0.2170 | 0.1667 | 0.1667 | 0.4615 | 0.4800 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 4 | b16_lr1e-4 | 54.4 ± 25.9 | 0.5022 | 0.5444 | 47.9 ± 25.9 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 54.4 ± 25.9 | 47.9 ± 25.9 | 73.4 ± 14.0 | 73.1 ± 12.1 | 0.2977 | 0.2322 | 0.1667 | 0.1667 | 0.3750 | 0.4615 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 5 | b32_lr1e-4 | 52.2 ± 13.4 | 0.5100 | 0.5222 | 44.5 ± 18.3 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 52.2 ± 13.4 | 44.5 ± 18.3 | 77.6 ± 10.9 | 76.2 ± 11.2 | 0.3947 | 0.2707 | 0.3333 | 0.3333 | 0.3750 | 0.4000 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 6 | b32_lr1e-3 | 51.1 ± 23.4 | 0.4300 | 0.5111 | 43.6 ± 24.2 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 51.1 ± 23.4 | 43.6 ± 24.2 | 77.6 ± 9.0 | 77.6 ± 10.1 | 0.2883 | 0.2059 | 0.1667 | 0.1667 | 0.4000 | 0.4211 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |

## Repeat-level predictive distributions

Mean and sample SD are shown in one percentage column; lossless CI, range, mean, and SD values remain in the matching JSON table.

| Case | Metric | Repeats | Mean ± SD (%) | Source |
|---|---|---|---|---|
| b16_lr1e-4 | balanced_accuracy | 5 | 54.4 ± 25.9 | participant_oof |
| b16_lr1e-4 | macro_f1 | 5 | 47.9 ± 25.9 | participant_oof |
| b16_lr1e-4 | macro_roc_auc_ovr | 5 | 73.4 ± 14.0 | participant_oof |
| b16_lr1e-4 | macro_pr_auc_ovr | 5 | 73.1 ± 12.1 | participant_oof |
| b16_lr1e-4 | abstention_aware_balanced_accuracy | 5 | 54.4 ± 25.9 | participant_oof |
| b16_lr1e-4 | abstention_aware_macro_f1 | 5 | 47.9 ± 25.9 | participant_oof |
| b16_lr3e-4 | balanced_accuracy | 5 | 55.6 ± 26.1 | participant_oof |
| b16_lr3e-4 | macro_f1 | 5 | 48.8 ± 28.4 | participant_oof |
| b16_lr3e-4 | macro_roc_auc_ovr | 5 | 71.9 ± 13.6 | participant_oof |
| b16_lr3e-4 | macro_pr_auc_ovr | 5 | 71.7 ± 11.8 | participant_oof |
| b16_lr3e-4 | abstention_aware_balanced_accuracy | 5 | 55.6 ± 26.1 | participant_oof |
| b16_lr3e-4 | abstention_aware_macro_f1 | 5 | 48.8 ± 28.4 | participant_oof |
| b16_lr1e-3 | balanced_accuracy | 5 | 62.2 ± 21.3 | participant_oof |
| b16_lr1e-3 | macro_f1 | 5 | 55.7 ± 22.7 | participant_oof |
| b16_lr1e-3 | macro_roc_auc_ovr | 5 | 77.4 ± 11.7 | participant_oof |
| b16_lr1e-3 | macro_pr_auc_ovr | 5 | 76.6 ± 7.7 | participant_oof |
| b16_lr1e-3 | abstention_aware_balanced_accuracy | 5 | 62.2 ± 21.3 | participant_oof |
| b16_lr1e-3 | abstention_aware_macro_f1 | 5 | 55.7 ± 22.7 | participant_oof |
| b32_lr1e-4 | balanced_accuracy | 5 | 52.2 ± 13.4 | participant_oof |
| b32_lr1e-4 | macro_f1 | 5 | 44.5 ± 18.3 | participant_oof |
| b32_lr1e-4 | macro_roc_auc_ovr | 5 | 77.6 ± 10.9 | participant_oof |
| b32_lr1e-4 | macro_pr_auc_ovr | 5 | 76.2 ± 11.2 | participant_oof |
| b32_lr1e-4 | abstention_aware_balanced_accuracy | 5 | 52.2 ± 13.4 | participant_oof |
| b32_lr1e-4 | abstention_aware_macro_f1 | 5 | 44.5 ± 18.3 | participant_oof |
| b32_lr3e-4 | balanced_accuracy | 5 | 58.9 ± 18.7 | participant_oof |
| b32_lr3e-4 | macro_f1 | 5 | 50.3 ± 23.9 | participant_oof |
| b32_lr3e-4 | macro_roc_auc_ovr | 5 | 70.2 ± 12.4 | participant_oof |
| b32_lr3e-4 | macro_pr_auc_ovr | 5 | 70.8 ± 11.8 | participant_oof |
| b32_lr3e-4 | abstention_aware_balanced_accuracy | 5 | 58.9 ± 18.7 | participant_oof |
| b32_lr3e-4 | abstention_aware_macro_f1 | 5 | 50.3 ± 23.9 | participant_oof |
| b32_lr1e-3 | balanced_accuracy | 5 | 51.1 ± 23.4 | participant_oof |
| b32_lr1e-3 | macro_f1 | 5 | 43.6 ± 24.2 | participant_oof |
| b32_lr1e-3 | macro_roc_auc_ovr | 5 | 77.6 ± 9.0 | participant_oof |
| b32_lr1e-3 | macro_pr_auc_ovr | 5 | 77.6 ± 10.1 | participant_oof |
| b32_lr1e-3 | abstention_aware_balanced_accuracy | 5 | 51.1 ± 23.4 | participant_oof |
| b32_lr1e-3 | abstention_aware_macro_f1 | 5 | 43.6 ± 24.2 | participant_oof |

<details><summary>Per-class repeat distributions</summary>

| Case | Class | Metric | Repeats | Mean ± SD (%) |
|---|---|---|---|---|
| b16_lr1e-3 | Pre-Frail | balanced_accuracy_ovr | 5 | 73.0 ± 9.9 |
| b16_lr1e-3 | Pre-Frail | f1 | 5 | 62.7 ± 12.8 |
| b16_lr1e-3 | Pre-Frail | recall | 5 | 70.0 ± 27.4 |
| b16_lr1e-3 | Pre-Frail | specificity | 5 | 76.0 ± 17.8 |
| b16_lr1e-3 | Pre-Frail | roc_auc_ovr | 5 | 86.0 ± 12.9 |
| b16_lr1e-3 | Pre-Frail | pr_auc_ovr | 5 | 78.3 ± 20.9 |
| b16_lr1e-3 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 62.5 ± 18.6 |
| b16_lr1e-3 | Robust/Non-Frail | f1 | 5 | 51.0 ± 33.8 |
| b16_lr1e-3 | Robust/Non-Frail | recall | 5 | 56.7 ± 36.5 |
| b16_lr1e-3 | Robust/Non-Frail | specificity | 5 | 68.3 ± 32.5 |
| b16_lr1e-3 | Robust/Non-Frail | roc_auc_ovr | 5 | 68.1 ± 13.1 |
| b16_lr1e-3 | Robust/Non-Frail | pr_auc_ovr | 5 | 74.7 ± 16.1 |
| b16_lr1e-3 | Young | balanced_accuracy_ovr | 5 | 78.0 ± 25.9 |
| b16_lr1e-3 | Young | f1 | 5 | 53.3 ± 50.6 |
| b16_lr1e-3 | Young | recall | 5 | 60.0 ± 54.8 |
| b16_lr1e-3 | Young | specificity | 5 | 96.0 ± 8.9 |
| b16_lr1e-3 | Young | roc_auc_ovr | 5 | 78.0 ± 22.8 |
| b16_lr1e-3 | Young | pr_auc_ovr | 5 | 76.7 ± 22.4 |
| b16_lr1e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 68.5 ± 21.8 |
| b16_lr1e-4 | Pre-Frail | f1 | 5 | 53.3 ± 34.0 |
| b16_lr1e-4 | Pre-Frail | recall | 5 | 60.0 ± 41.8 |
| b16_lr1e-4 | Pre-Frail | specificity | 5 | 77.0 ± 14.4 |
| b16_lr1e-4 | Pre-Frail | roc_auc_ovr | 5 | 79.0 ± 14.9 |
| b16_lr1e-4 | Pre-Frail | pr_auc_ovr | 5 | 72.3 ± 19.9 |
| b16_lr1e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 55.8 ± 17.6 |
| b16_lr1e-4 | Robust/Non-Frail | f1 | 5 | 50.5 ± 14.0 |
| b16_lr1e-4 | Robust/Non-Frail | recall | 5 | 53.3 ± 13.9 |
| b16_lr1e-4 | Robust/Non-Frail | specificity | 5 | 58.3 ± 37.3 |
| b16_lr1e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 65.8 ± 11.9 |
| b16_lr1e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 75.2 ± 7.0 |
| b16_lr1e-4 | Young | balanced_accuracy_ovr | 5 | 71.0 ± 20.1 |
| b16_lr1e-4 | Young | f1 | 5 | 40.0 ± 36.5 |
| b16_lr1e-4 | Young | recall | 5 | 50.0 ± 50.0 |
| b16_lr1e-4 | Young | specificity | 5 | 92.0 ± 11.0 |
| b16_lr1e-4 | Young | roc_auc_ovr | 5 | 75.5 ± 26.9 |
| b16_lr1e-4 | Young | pr_auc_ovr | 5 | 71.7 ± 27.4 |
| b16_lr3e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 64.5 ± 18.7 |
| b16_lr3e-4 | Pre-Frail | f1 | 5 | 49.3 ± 30.3 |
| b16_lr3e-4 | Pre-Frail | recall | 5 | 60.0 ± 41.8 |
| b16_lr3e-4 | Pre-Frail | specificity | 5 | 69.0 ± 29.5 |
| b16_lr3e-4 | Pre-Frail | roc_auc_ovr | 5 | 74.5 ± 13.2 |
| b16_lr3e-4 | Pre-Frail | pr_auc_ovr | 5 | 68.3 ± 14.9 |
| b16_lr3e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.7 ± 19.9 |
| b16_lr3e-4 | Robust/Non-Frail | f1 | 5 | 50.4 ± 16.7 |
| b16_lr3e-4 | Robust/Non-Frail | recall | 5 | 46.7 ± 13.9 |
| b16_lr3e-4 | Robust/Non-Frail | specificity | 5 | 66.7 ± 40.8 |
| b16_lr3e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 65.8 ± 11.9 |
| b16_lr3e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 75.2 ± 7.0 |
| b16_lr3e-4 | Young | balanced_accuracy_ovr | 5 | 76.0 ± 24.1 |
| b16_lr3e-4 | Young | f1 | 5 | 46.7 ± 44.7 |
| b16_lr3e-4 | Young | recall | 5 | 60.0 ± 54.8 |
| b16_lr3e-4 | Young | specificity | 5 | 92.0 ± 11.0 |
| b16_lr3e-4 | Young | roc_auc_ovr | 5 | 75.5 ± 26.9 |
| b16_lr3e-4 | Young | pr_auc_ovr | 5 | 71.7 ± 27.4 |
| b32_lr1e-3 | Pre-Frail | balanced_accuracy_ovr | 5 | 58.5 ± 25.7 |
| b32_lr1e-3 | Pre-Frail | f1 | 5 | 38.0 ± 41.5 |
| b32_lr1e-3 | Pre-Frail | recall | 5 | 40.0 ± 41.8 |
| b32_lr1e-3 | Pre-Frail | specificity | 5 | 77.0 ± 14.4 |
| b32_lr1e-3 | Pre-Frail | roc_auc_ovr | 5 | 79.0 ± 14.9 |
| b32_lr1e-3 | Pre-Frail | pr_auc_ovr | 5 | 68.3 ± 22.4 |
| b32_lr1e-3 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 55.0 ± 25.6 |
| b32_lr1e-3 | Robust/Non-Frail | f1 | 5 | 49.5 ± 22.4 |
| b32_lr1e-3 | Robust/Non-Frail | recall | 5 | 53.3 ± 27.4 |
| b32_lr1e-3 | Robust/Non-Frail | specificity | 5 | 56.7 ± 39.7 |
| b32_lr1e-3 | Robust/Non-Frail | roc_auc_ovr | 5 | 64.7 ± 20.2 |
| b32_lr1e-3 | Robust/Non-Frail | pr_auc_ovr | 5 | 75.3 ± 13.9 |
| b32_lr1e-3 | Young | balanced_accuracy_ovr | 5 | 74.0 ± 23.0 |
| b32_lr1e-3 | Young | f1 | 5 | 43.3 ± 43.5 |
| b32_lr1e-3 | Young | recall | 5 | 60.0 ± 54.8 |
| b32_lr1e-3 | Young | specificity | 5 | 88.0 ± 17.9 |
| b32_lr1e-3 | Young | roc_auc_ovr | 5 | 89.0 ± 15.2 |
| b32_lr1e-3 | Young | pr_auc_ovr | 5 | 89.0 ± 15.2 |
| b32_lr1e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 62.0 ± 19.8 |
| b32_lr1e-4 | Pre-Frail | f1 | 5 | 39.3 ± 37.4 |
| b32_lr1e-4 | Pre-Frail | recall | 5 | 50.0 ± 50.0 |
| b32_lr1e-4 | Pre-Frail | specificity | 5 | 74.0 ± 32.7 |
| b32_lr1e-4 | Pre-Frail | roc_auc_ovr | 5 | 83.5 ± 10.5 |
| b32_lr1e-4 | Pre-Frail | pr_auc_ovr | 5 | 75.0 ± 17.7 |
| b32_lr1e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 61.7 ± 6.8 |
| b32_lr1e-4 | Robust/Non-Frail | f1 | 5 | 52.9 ± 3.9 |
| b32_lr1e-4 | Robust/Non-Frail | recall | 5 | 56.7 ± 27.9 |
| b32_lr1e-4 | Robust/Non-Frail | specificity | 5 | 66.7 ± 35.8 |
| b32_lr1e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 73.9 ± 15.9 |
| b32_lr1e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 82.0 ± 12.6 |
| b32_lr1e-4 | Young | balanced_accuracy_ovr | 5 | 66.5 ± 24.1 |
| b32_lr1e-4 | Young | f1 | 5 | 41.3 ± 43.3 |
| b32_lr1e-4 | Young | recall | 5 | 50.0 ± 50.0 |
| b32_lr1e-4 | Young | specificity | 5 | 83.0 ± 26.4 |
| b32_lr1e-4 | Young | roc_auc_ovr | 5 | 75.5 ± 26.9 |
| b32_lr1e-4 | Young | pr_auc_ovr | 5 | 71.7 ± 27.4 |
| b32_lr3e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 63.5 ± 16.9 |
| b32_lr3e-4 | Pre-Frail | f1 | 5 | 37.3 ± 37.0 |
| b32_lr3e-4 | Pre-Frail | recall | 5 | 40.0 ± 41.8 |
| b32_lr3e-4 | Pre-Frail | specificity | 5 | 87.0 ± 18.6 |
| b32_lr3e-4 | Pre-Frail | roc_auc_ovr | 5 | 78.5 ± 16.3 |
| b32_lr3e-4 | Pre-Frail | pr_auc_ovr | 5 | 71.7 ± 21.7 |
| b32_lr3e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 69.2 ± 13.7 |
| b32_lr3e-4 | Robust/Non-Frail | f1 | 5 | 65.5 ± 15.7 |
| b32_lr3e-4 | Robust/Non-Frail | recall | 5 | 76.7 ± 32.5 |
| b32_lr3e-4 | Robust/Non-Frail | specificity | 5 | 61.7 ± 43.9 |
| b32_lr3e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 56.1 ± 21.9 |
| b32_lr3e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 64.7 ± 18.8 |
| b32_lr3e-4 | Young | balanced_accuracy_ovr | 5 | 74.0 ± 25.1 |
| b32_lr3e-4 | Young | f1 | 5 | 48.0 ± 50.2 |
| b32_lr3e-4 | Young | recall | 5 | 60.0 ± 54.8 |
| b32_lr3e-4 | Young | specificity | 5 | 88.0 ± 26.8 |
| b32_lr3e-4 | Young | roc_auc_ovr | 5 | 76.0 ± 25.1 |
| b32_lr3e-4 | Young | pr_auc_ovr | 5 | 76.2 ± 22.7 |

</details>

## Aggregation sensitivity from the same file-level OOF

The declared-source row reproduces the aggregation used by the fitted model and, when eligible, the primary leaderboard. The other row reaggregates the same held-out file probabilities post hoc. It is not a separately retrained Line A/Line B experiment and is not selection evidence.

| Case | Aggregation view | Role | Mean BA | Mean Macro-F1 | Line A − Line B BA | Line A − Line B Macro-F1 | Worst recall | Worst F1 | ECE | Repeats | Retained participant OOF n | All participant units n | Dropped participant units n | All file OOF n | Dropped files n | Source replay | Primary ranking eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | line_a_equal_files | posthoc_aggregation_only | 0.6333 | 0.5595 | 0.0111 | 0.0029 | 0.1667 | 0.2333 | 0.2465 | 5 | 31 | 31 | 0 | 155 | 0 | not_applicable_posthoc_view | False |
| b16_lr1e-3 | line_b_equal_role_families | declared_source_line | 0.6222 | 0.5567 | 0.0111 | 0.0029 | 0.2333 | 0.2933 | 0.3251 | 5 | 31 | 31 | 0 | 155 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| b16_lr1e-4 | line_a_equal_files | posthoc_aggregation_only | 0.5556 | 0.4562 | 0.0111 | -0.0232 | 0.2000 | 0.2333 | 0.2851 | 5 | 31 | 31 | 0 | 155 | 0 | not_applicable_posthoc_view | False |
| b16_lr1e-4 | line_b_equal_role_families | declared_source_line | 0.5444 | 0.4794 | 0.0111 | -0.0232 | 0.2667 | 0.3333 | 0.3252 | 5 | 31 | 31 | 0 | 155 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| b16_lr3e-4 | line_a_equal_files | posthoc_aggregation_only | 0.4556 | 0.3394 | -0.1000 | -0.1486 | 0.0000 | 0.0000 | 0.3772 | 5 | 31 | 31 | 0 | 155 | 0 | not_applicable_posthoc_view | False |
| b16_lr3e-4 | line_b_equal_role_families | declared_source_line | 0.5556 | 0.4879 | -0.1000 | -0.1486 | 0.2667 | 0.3467 | 0.3566 | 5 | 31 | 31 | 0 | 155 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| b32_lr1e-3 | line_a_equal_files | posthoc_aggregation_only | 0.5333 | 0.4597 | 0.0222 | 0.0235 | 0.1667 | 0.1800 | 0.3740 | 5 | 31 | 31 | 0 | 155 | 0 | not_applicable_posthoc_view | False |
| b32_lr1e-3 | line_b_equal_role_families | declared_source_line | 0.5111 | 0.4362 | 0.0222 | 0.0235 | 0.1667 | 0.2000 | 0.3855 | 5 | 31 | 31 | 0 | 155 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| b32_lr1e-4 | line_a_equal_files | posthoc_aggregation_only | 0.5222 | 0.4451 | 0.0000 | 0.0000 | 0.1667 | 0.1800 | 0.4076 | 5 | 31 | 31 | 0 | 155 | 0 | not_applicable_posthoc_view | False |
| b32_lr1e-4 | line_b_equal_role_families | declared_source_line | 0.5222 | 0.4451 | 0.0000 | 0.0000 | 0.1667 | 0.1800 | 0.4146 | 5 | 31 | 31 | 0 | 155 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| b32_lr3e-4 | line_a_equal_files | posthoc_aggregation_only | 0.5889 | 0.5022 | 0.0000 | -0.0005 | 0.1667 | 0.2133 | 0.3621 | 5 | 31 | 31 | 0 | 155 | 0 | not_applicable_posthoc_view | False |
| b32_lr3e-4 | line_b_equal_role_families | declared_source_line | 0.5889 | 0.5027 | 0.0000 | -0.0005 | 0.1667 | 0.2133 | 0.3733 | 5 | 31 | 31 | 0 | 155 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |

## Parallel window/file/role-balanced participant views

All three rows reuse the same fitted held-out OOF probabilities; they are not three training runs. `window_balanced_to_participant` gives every retained window equal report weight, Line A gives every file equal weight after window→file, and Line B gives every canonical role family equal weight after window→file→role. Only the declared training aggregation may support the primary leaderboard; the other views are post-hoc sensitivity plots.

| Case | Aggregation view | Evidence role | Mean BA | Mean Macro-F1 | Worst recall | Worst F1 | Repeats | Participant OOF n | Primary ranking eligible |
|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.6333 | 0.5595 | 0.1667 | 0.2333 | 5 | 31 | False |
| b16_lr1e-3 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.6333 | 0.5595 | 0.1667 | 0.2333 | 5 | 31 | False |
| b16_lr1e-3 | line_b_equal_role_families | declared_training_aggregation | 0.6222 | 0.5567 | 0.2333 | 0.2933 | 5 | 31 | True |
| b16_lr1e-4 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5556 | 0.4562 | 0.2000 | 0.2333 | 5 | 31 | False |
| b16_lr1e-4 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.5556 | 0.4562 | 0.2000 | 0.2333 | 5 | 31 | False |
| b16_lr1e-4 | line_b_equal_role_families | declared_training_aggregation | 0.5444 | 0.4794 | 0.2667 | 0.3333 | 5 | 31 | True |
| b16_lr3e-4 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4556 | 0.3394 | 0.0000 | 0.0000 | 5 | 31 | False |
| b16_lr3e-4 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.4556 | 0.3394 | 0.0000 | 0.0000 | 5 | 31 | False |
| b16_lr3e-4 | line_b_equal_role_families | declared_training_aggregation | 0.5556 | 0.4879 | 0.2667 | 0.3467 | 5 | 31 | True |
| b32_lr1e-3 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5333 | 0.4597 | 0.1667 | 0.1800 | 5 | 31 | False |
| b32_lr1e-3 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.5333 | 0.4597 | 0.1667 | 0.1800 | 5 | 31 | False |
| b32_lr1e-3 | line_b_equal_role_families | declared_training_aggregation | 0.5111 | 0.4362 | 0.1667 | 0.2000 | 5 | 31 | True |
| b32_lr1e-4 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5222 | 0.4451 | 0.1667 | 0.1800 | 5 | 31 | False |
| b32_lr1e-4 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.5222 | 0.4451 | 0.1667 | 0.1800 | 5 | 31 | False |
| b32_lr1e-4 | line_b_equal_role_families | declared_training_aggregation | 0.5222 | 0.4451 | 0.1667 | 0.1800 | 5 | 31 | True |
| b32_lr3e-4 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5889 | 0.5022 | 0.1667 | 0.2133 | 5 | 31 | False |
| b32_lr3e-4 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.5889 | 0.5022 | 0.1667 | 0.2133 | 5 | 31 | False |
| b32_lr3e-4 | line_b_equal_role_families | declared_training_aggregation | 0.5889 | 0.5027 | 0.1667 | 0.2133 | 5 | 31 | True |

<details><summary>Hierarchy coverage: B/R1–R4 window/file views and B/R role-balanced view</summary>

| Case | Repeat | Level | View | Group | OOF units | Retained units | Participants |
|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | 0 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 6 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 6 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | B | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | participant | participant_balanced_endpoint | participant | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 7 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 7 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 7 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 7 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 7 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 7 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 6 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 6 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 6 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 6 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 6 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 6 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 6 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 6 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 6 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | participant | participant_balanced_endpoint | participant | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 7 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 7 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 7 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 7 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 7 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 7 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 6 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 6 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 6 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 6 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 6 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 6 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 6 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 6 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 6 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | participant | participant_balanced_endpoint | participant | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 7 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 7 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 7 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 7 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 7 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 7 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 6 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 6 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 6 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 6 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 6 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 6 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 6 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 6 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 6 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | B | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | participant | participant_balanced_endpoint | participant | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 7 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 7 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 7 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 7 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 7 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 7 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 6 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 6 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 6 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 6 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 6 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 6 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 6 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 6 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 6 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | participant | participant_balanced_endpoint | participant | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 7 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 7 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 7 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 7 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 7 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 7 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 6 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 6 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 6 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 6 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 6 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 6 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 6 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 6 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 6 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | participant | participant_balanced_endpoint | participant | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 7 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 7 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 7 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 7 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 7 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 7 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 6 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 6 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 6 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 6 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 6 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 6 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 6 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | participant | participant_balanced_endpoint | participant | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 6 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 6 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 6 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 6 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 6 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 6 |

</details>

## Worst-class F1 stability review

This secondary view reorders complete cases by abstention-aware worst-class F1, then abstention-aware repeat variability. Conditional retained-only values remain visible for comparison.

| Stability rank | Aware-BA rank | Case | Aware worst F1 | Aware worst recall | Aware BA, mean ± SD (%) | Worst F1 | Worst recall | Conditional BA, mean ± SD (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | b16_lr1e-3 | 0.5926 | 0.5000 | 62.2 ± 19.1 | 0.5926 | 0.5000 | 62.2 ± 19.1 |
| 2 | 3 | b16_lr3e-4 | 0.4800 | 0.4615 | 55.6 ± 23.3 | 0.4800 | 0.4615 | 55.6 ± 23.3 |
| 3 | 2 | b32_lr3e-4 | 0.4706 | 0.4000 | 58.9 ± 16.7 | 0.4706 | 0.4000 | 58.9 ± 16.7 |
| 4 | 4 | b16_lr1e-4 | 0.4615 | 0.3750 | 54.4 ± 23.1 | 0.4615 | 0.3750 | 54.4 ± 23.1 |
| 5 | 6 | b32_lr1e-3 | 0.4211 | 0.4000 | 51.1 ± 20.9 | 0.4211 | 0.4000 | 51.1 ± 20.9 |
| 6 | 5 | b32_lr1e-4 | 0.4000 | 0.3750 | 52.2 ± 12.0 | 0.4000 | 0.3750 | 52.2 ± 12.0 |

## Incomplete cases excluded from ranking

N/A — no rows were available.

## Deployment measurements (separate from predictive ranking)

| Case | Parameters | Inference cost | Status | Reported note |
|---|---|---|---|---|
| b16_lr1e-4 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| b16_lr3e-4 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| b16_lr1e-3 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| b32_lr1e-4 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| b32_lr3e-4 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| b32_lr1e-3 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |

## Route × role coverage and feature availability

This table separates direct and processed rate paths, retained coverage, unavailable predictors, and reducer failures for each role/route state.

| Case | Role | Quality tier | Motion | Route state | Signal route | Retained coverage | Abstention | Abstention reasons | Direct | Processed | Unavailable predictors | Denoiser attempts | Denoiser successes | Reducer failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | B | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-3 | R1 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-3 | R2 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-3 | R3 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-3 | R4 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-4 | B | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-4 | R1 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-4 | R2 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-4 | R3 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr1e-4 | R4 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr3e-4 | B | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr3e-4 | R1 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr3e-4 | R2 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr3e-4 | R3 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b16_lr3e-4 | R4 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-3 | B | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-3 | R1 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-3 | R2 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-3 | R3 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-3 | R4 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-4 | B | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-4 | R1 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-4 | R2 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-4 | R3 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr1e-4 | R4 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr3e-4 | B | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr3e-4 | R1 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr3e-4 | R2 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr3e-4 | R3 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| b32_lr3e-4 | R4 | excellent | off | full_direct | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |

## SQI state, score, and coverage provenance by each route

Direct and post-denoiser coverage are reported separately so the configured minimum-coverage decision remains auditable.

| Case | Role | Tier | Direct Q_rate state | Mean direct Q_rate | Direct Q_rate coverage | Direct Q_morph state | Mean direct Q_morph | Direct Q_morph coverage | Post Q_rate state | Mean post Q_rate | Post Q_rate coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | B | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-3 | R1 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-3 | R2 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-3 | R3 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-3 | R4 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-4 | B | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-4 | R1 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-4 | R2 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-4 | R3 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr1e-4 | R4 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr3e-4 | B | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr3e-4 | R1 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr3e-4 | R2 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr3e-4 | R3 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b16_lr3e-4 | R4 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-3 | B | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-3 | R1 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-3 | R2 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-3 | R3 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-3 | R4 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-4 | B | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-4 | R1 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-4 | R2 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-4 | R3 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr1e-4 | R4 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr3e-4 | B | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr3e-4 | R1 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr3e-4 | R2 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr3e-4 | R3 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| b32_lr3e-4 | R4 | excellent | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |

## Denoiser paired heart-rate comparison

HR is calculated as `60 / median(valid PPI seconds)` from the same registered peak detector before and after the single denoiser attempt. Rows are paired within recording and averaged within participant before the participant-macro summary. Use the `outer_oof` rows for the primary held-out comparison; outer-train rows remain audit-only.

N/A — no rows were available.

<details><summary>Per-record paired denoiser HR evidence</summary>

N/A — no rows were available.

</details>

## Frozen motion evidence used by each route

Frailty29 reuse is in-sample auxiliary motion-preprocessing evidence, not valid outer-OOF motion-detector evidence. The downstream frailty classification outcome is still evaluated on each outer held-out fold.

| Case | Role | Tier | Motion | Mean p(motion) | Threshold | Mean windows | Evidence SHA-256 | Model SHA-256 | Training scope | Frailty29 relation | Valid outer-OOF motion evidence | Denoiser | Denoiser status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | B | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-3 | R1 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-3 | R2 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-3 | R3 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-3 | R4 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-4 | B | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-4 | R1 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-4 | R2 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-4 | R3 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr1e-4 | R4 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr3e-4 | B | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr3e-4 | R1 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr3e-4 | R2 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr3e-4 | R3 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b16_lr3e-4 | R4 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-3 | B | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-3 | R1 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-3 | R2 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-3 | R3 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-3 | R4 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-4 | B | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-4 | R1 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-4 | R2 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-4 | R3 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr1e-4 | R4 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr3e-4 | B | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr3e-4 | R1 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr3e-4 | R2 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr3e-4 | R3 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| b32_lr3e-4 | R4 | excellent | off | N/A | N/A | 0.0000 | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |

## Quality-component distributions by route and role

N/A — no rows were available.

## Classification score, t-SNE, and ROC–AUC diagnostics

Every classifier with persisted participant OOF probabilities is represented in the three paired figure modules. t-SNE embeds the prediction-probability vector, not hidden features. Multiclass frailty decisions use argmax and therefore have no single scalar threshold.

| Classifier | Score/threshold | Prediction t-SNE | ROC–AUC curve | OOF rows | t-SNE points | ROC points |
|---|---|---|---|---|---|---|
| b16_lr1e-4 | available | available | available | 31 | 31 | 244 |
| b16_lr3e-4 | available | available | available | 31 | 31 | 248 |
| b16_lr1e-3 | available | available | available | 31 | 31 | 241 |
| b32_lr1e-4 | available | available | available | 31 | 31 | 244 |
| b32_lr3e-4 | available | available | available | 31 | 31 | 247 |
| b32_lr1e-3 | available | available | available | 31 | 31 | 251 |

## Failed or incomplete cases

N/A — no rows were available.

## Figure status

| Figure | Status | Path | Reason |
|---|---|---|---|
| classification_prediction_scores | generated | figures/classification_prediction_scores.png |  |
| classification_prediction_tsne | generated | figures/classification_prediction_tsne.png |  |
| classification_roc_auc_curves | generated | figures/classification_roc_auc_curves.png |  |
| leaderboard | generated | figures/leaderboard.png |  |
| stability | generated | figures/stability.png |  |
| macro_f1_stability | generated | figures/macro_f1_stability.png |  |
| roc_pr_auc_stability | generated | figures/roc_pr_auc_stability.png |  |
| per_class_metric_stability | generated | figures/per_class_metric_stability.png |  |
| worst_class_f1_stability | generated | figures/worst_class_f1_stability.png |  |
| fold_heatmap | generated | figures/fold_heatmap.png |  |
| paired_deltas | N/A | figures/paired_deltas.NA.txt | ValueError: reference-paired repeat deltas unavailable |
| ablation_sensitivity_metrics | N/A | figures/ablation_sensitivity_metrics.NA.txt | ValueError: configured repeat-level metrics unavailable |
| coverage | generated | figures/coverage.png |  |
| route_role_coverage | generated | figures/route_role_coverage.png |  |
| denoiser_hr_comparison | N/A | figures/denoiser_hr_comparison.NA.txt | ValueError: paired direct/post-denoiser HR evidence unavailable |
| quality_distributions | N/A | figures/quality_distributions.NA.txt | ValueError: route/role quality-component distributions unavailable |
| calibration | generated | figures/calibration.png |  |
| confusion_matrices | generated | figures/confusion_matrices.png |  |
| confusion_matrices_row_normalized | generated | figures/confusion_matrices_row_normalized.png |  |
| per_class | generated | figures/per_class.png |  |
| aggregation_view_metrics | generated | figures/aggregation_view_metrics.png |  |
| aggregation_hierarchy_coverage | generated | figures/aggregation_hierarchy_coverage.png |  |
| aggregation_view_confusion_matrices | generated | figures/aggregation_view_confusion_matrices.png |  |
| aggregation_view_confusion_matrices_row_normalized | generated | figures/aggregation_view_confusion_matrices_row_normalized.png |  |
| aggregation_view_per_class | generated | figures/aggregation_view_per_class.png |  |
| learning_curves | generated | figures/learning_curves.png |  |
| top_learning_curves | generated | figures/top_learning_curves.png |  |
| balanced_accuracy_learning_curves | generated | figures/balanced_accuracy_learning_curves.png |  |
| top_balanced_accuracy_learning_curves | generated | figures/top_balanced_accuracy_learning_curves.png |  |
| parameter_effects | N/A | figures/parameter_effects.NA.txt | ValueError: no declared varied axis |
| parameter_interaction | N/A | figures/parameter_interaction.NA.txt | ValueError: requires at least two declared varied axes |

## Limitations and N/A items

- Classification t-SNE is a report-only embedding of persisted OOF prediction-probability vectors, not a hidden-feature embedding and not evidence of separability in the model representation space.
- paired deltas are N/A because no reference case was declared
- Report-only ROC AUC is one-vs-rest macro AUC; PR AUC is one-vs-rest average precision. Per-class BA is (sensitivity + specificity) / 2. All are recomputed from retained outer participant OOF probabilities.

## Output navigation

- [outputs_index.json](outputs_index.json): machine-readable inventory
- [study_summary.json](study_summary.json): report context and tables
- [tables/reproducibility_summary.csv](tables/reproducibility_summary.csv)
- [tables/reproducibility_cases.csv](tables/reproducibility_cases.csv)
- [tables/reproducibility_cells.csv](tables/reproducibility_cells.csv)
- [tables/reproducibility_splits.csv](tables/reproducibility_splits.csv)
- [tables/reproducibility_issues.csv](tables/reproducibility_issues.csv)
- [tables/predictive_leaderboard.csv](tables/predictive_leaderboard.csv)
- [tables/aggregation_line_comparison.csv](tables/aggregation_line_comparison.csv)
- [tables/aggregation_line_repeat_metrics.csv](tables/aggregation_line_repeat_metrics.csv)
- [tables/aggregation_line_per_class_metrics.csv](tables/aggregation_line_per_class_metrics.csv)
- [tables/aggregation_view_comparison.csv](tables/aggregation_view_comparison.csv)
- [tables/aggregation_view_confusion_matrices.csv](tables/aggregation_view_confusion_matrices.csv)
- [tables/aggregation_hierarchy_coverage.csv](tables/aggregation_hierarchy_coverage.csv)
- [tables/metric_distribution_summary.csv](tables/metric_distribution_summary.csv)
- [tables/repeat_per_class_metrics.csv](tables/repeat_per_class_metrics.csv)
- [tables/per_class_metric_distribution_summary.csv](tables/per_class_metric_distribution_summary.csv)
- [tables/classification_prediction_scores.csv](tables/classification_prediction_scores.csv)
- [tables/classification_prediction_tsne.csv](tables/classification_prediction_tsne.csv)
- [tables/classification_roc_curves.csv](tables/classification_roc_curves.csv)
- [tables/classification_diagnostic_status.csv](tables/classification_diagnostic_status.csv)
- [tables/table_figure_pairs.csv](tables/table_figure_pairs.csv)
- [tables/report_tables.xlsx](tables/report_tables.xlsx): one table per worksheet
- [tables/worst_class_f1_stability.csv](tables/worst_class_f1_stability.csv)
- [tables/incomplete_cases.csv](tables/incomplete_cases.csv)
- [tables/confusion_counts.csv](tables/confusion_counts.csv)
- [tables/confusion_row_normalized.csv](tables/confusion_row_normalized.csv)
- [tables/top_confusion_matrices/](tables/top_confusion_matrices/): top-case count and row-normalized CSVs
- [tables/deployment_measurements.csv](tables/deployment_measurements.csv)
- [figures/plot_status.json](figures/plot_status.json)
