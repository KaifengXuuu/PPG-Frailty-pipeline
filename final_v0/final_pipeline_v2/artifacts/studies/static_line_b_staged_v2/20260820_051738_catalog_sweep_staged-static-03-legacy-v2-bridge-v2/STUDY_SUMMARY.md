# V2 study summary — staged_static_03_legacy_v2_bridge_v2

> This report is descriptive evidence for manual review. It does not automatically select a final use case or winner.

## Scientific context

- Study kind: catalog_sweep
- Purpose: Diagnose the historical-to-V2 result gap with an advisory data-integrity audit and a frozen adjacent-profile ablation chain.
- Position in use-case selection flow: Stage 3 after ordinary route screening and before ensemble, SQI/motion, sequential finalist ablations, and stage-last ShapeFormer.
- Decision role: ablation
- Thesis sections: ["Legacy-to-V2 data and source integrity audit", "Raw-model preprocessing and balance-policy bridge"]
- Catalog: configs/formal_experiment_catalog_v2.yaml (scope=selected_ordinary, balance=line_b)
- Reference case: compact_cnn__l0_legacy64_w15_fixed10

## Run controls and completeness

- Repeats requested: [0]
- Folds requested: [0, 1, 2, 3, 4]
- Case-level jobs requested: 1
- Effective jobs: 1
- Planned / passed / failed / not-run cases: 9 / 9 / 0 / 0
- Planned / reported / passed / failed / not-run cells: 45 / 45 / 45 / 0 / 0
- Resume-skipped passed cases: 0

## Test models, modules, inputs, and fixed parameters

The identical standalone table is in [TEST_COMPONENTS.md](TEST_COMPONENTS.md); machine-readable copies are `tables/test_components.csv` and `.json`. Input data are reported as dataset/path, signal view, channels, units, rate, and windows—not hashes.

| Cases / phases | Component role | Model / module | State | Input data (values and paths; no hashes) | Detailed fixed parameters | Algorithm and kernel (≤300 chars) | Reporter profile | Model reporter extension | Algorithm / literature source |
|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | aggregation | line_b_equal_role_families | enabled | {"input_data":"held-out window/file probabilities","roles":["B","R1","R2","R3","R4"]} | {"balance_line":"line_b_equal_role_families","direct_all_window_participant_mean":false,"file_to_role":"ordinary_mean","hierarchy":["window","file","role","participant"],"missing_role_policy":"mean_available_roles","quality_weight_levels":[],"quality_weight_source":"none","quality_weighting":false,"role_to_participant":"ordinary_mean","window_to_file":"ordinary_mean"} | executable line_b_equal_role_families hierarchy ending in participant-balanced output | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.training.aggregation.aggregate_hierarchy; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10 | classifier | CompactCNN1D | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"architecture_parameters":{"classifier_dropout":0.2,"dilations":[1,1,1],"global_pooling":"adaptive_average_1","kernel_sizes":[9,9,7],"model_id":"compact_cnn","n_classes":3,"pool_sizes":[4,4],"representation_mode":"raw","stage_channels":[32,64,128],"stage_dropouts":[0.1,0.15]},"dilations":[1,1,1],"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[9,9,7],"mask_aware_pooling":true,"model_id":"CompactCNN1D","n_classes":3,"pool_sizes":[4,4],"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"reference_not_wang_fcn"} | Project CompactCNN1D: three temporal convolution stages with a pooled classification head. | multiclass_participant_oof_v1 | compactcnn_model_v1 | Project implementation: src/ppg_frailty/models/compact_cnn.py |
| inception_full__l0_legacy64_w15_fixed10 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} | Six-block, single-network InceptionTime adaptation with bottleneck and parallel fixed-sample kernels; not the paper's five-member ensemble. | multiclass_participant_oof_v1 | inceptiontime_single_network_model_v1 | Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | dataset_adapter | frailty3_m2_20260815_a054800abda272f6 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"allow_qc_excluded_records":false,"channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"class_id_order":[0,1,2],"class_name_order":["Pre-Frail","Robust/Non-Frail","Young"],"expected_participant_count":29,"expected_record_count":261,"manifest_version":"internal_records_v2","path":"manifests/internal_records_v2.csv","source_dataset_id":"frailty3_m2_20260815_a054800abda272f6"} | Persisted dataset_adapter contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: dataset_adapter; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | denoiser | identity | identity_or_disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["filtered RED/IR","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"declared_reducer_version":"identity_v1","degraded_policy":"drop","denoiser_enabled":null,"failure_action":"no_result_no_fallback","reducer":"identity","resolved_parameters":{},"runtime_reducer_version":"identity_exact_v1"} | 逐样本复制双波长 PPG，不估计或抑制伪影；内核：恒等映射与同时间网格校验，作为未去噪直接对照。 | audit_provenance_v1 | not_applicable | N/A — component was not executed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | evaluation | balanced_accuracy | enabled | {"class_order":["Pre-Frail","Robust/Non-Frail","Young"],"input_data":"held-out participant predictions and frailty labels"} | {"calibration_metrics":["multiclass_brier","expected_calibration_error"],"confidence_interval":"participant_cluster_bootstrap_two_sided_95","independent_test_available":false,"metric_prefix":"oof_validation_","metrics":["balanced_accuracy","macro_f1","per_class_precision_recall_f1","worst_class_recall","worst_class_f1","confusion_matrix","coverage"],"paired_delta_key":["repeat_index","fold_index","participant_id"],"primary_metric":"balanced_accuracy","rank_incomplete_configs":false,"ranking":{"automatic_final_selection":false,"manual_multiple_final_versions_allowed":true,"max_qualified_per_comparison_group":10,"preserve_ablation_provenance":true,"sort_key":"participant_level_mean_balanced_accuracy"},"statistics":{"affects_automatic_selection":false,"bootstrap_replicates":10000,"cluster_unit":"participant_with_all_five_repeat_oof_predictions","confidence_interval":"two_sided_95_percentile","lcb95_metrics":["participant_level_mean_balanced_accuracy","participant_level_mean_macro_f1"],"lcb95_percentile":2.5,"multiplicity_correction":"holm_within_comparison_family","paired_exchange_unit":"participant","paired_permutation_replicates":100000,"seed":42},"unit":"participant"} | Persisted evaluation contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: evaluation; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | feature_extractor | feature_vector_282_v3 | auxiliary_not_classifier_input | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","engineering_window":{"cap_fraction_per_file":null,"cap_per_file":null,"end_alignment":"left_start_regular_grid","hop_s":5.0,"length_s":10.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"input_views":["x_analysis/x_native","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"enabled_groups":["ppi_basic_rate","hrv_time_domain","hrv_spectral","hrv_nonlinear","morphology","dual_optical","engineering_summary"],"engineering_sequence_schema":"engineering_10s_hop5s_thesis_115_v2","file_aggregation":["mean","population_sd"],"file_vector_schema":"feature_vector_282_v3","matrix_k":32,"matrix_schema":"ordered_feature_matrix_d794_by_32_registry-0bea68a2058d_v3","missing_physiology_encoding":"nan_and_validity_false","prv_library_comparison_scope":"fixed_ppi_vectors_only_no_classifier","prv_primary_backend":"local_manual","rate_prv_min_duration_s":8.0,"rate_prv_min_peaks":5,"registry_id":"feature_vector_282_v3","sample_entropy":{"m":2,"min_intervals":200,"r_sd_fraction":0.2},"spectral_bands_hz":{"hf":[0.15,0.4],"lf":[0.04,0.15],"vlf":[0.003,0.04]},"spectral_prv_min_coverage":0.8,"spectral_prv_min_duration_s":300.0,"spectral_prv_min_intervals":200,"tachogram_fs_hz":4.0,"technical_metadata_allowed":false,"time_prv_min_coverage":0.8,"time_prv_min_duration_s":60.0,"time_prv_min_intervals":30} | Persisted feature_extractor contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: feature_extractor; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | imu_preprocessing | calibrated_roll_pitch_ekf | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["AX","AY","AZ","GX","GY","GZ"],"manifest_path":"manifests/internal_records_v2.csv","output_view":"processed_imu_physical","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"calibration_start_s":5.0,"calibration_stop_s":100.0,"comparison_method":"profile_a_lowpass_0p3hz","dynamic_observation_scale":3.0,"failure_action":"fail_closed","gravity_method":"calibrated_roll_pitch_ekf","initial_covariance_diagonal":[1.0,1.0,0.5,0.5,0.5],"initialization":"same_participant_static_calibration","observation_covariance_diagonal_rad2":[0.5,0.5],"output_units":{"acceleration":"m/s^2","gyroscope":"rad/s","jerk":"m/s^3"},"process_covariance_diagonal_per_second":[5.0,5.0,0.05,0.05,0.05],"required_axes":6,"sensor_filter_order":3,"sensor_lowpass_acc_hz":20.0,"sensor_lowpass_gyro_hz":40.0} | Calibrated roll-pitch EKF gravity compensation retaining physical-unit dynamic acceleration and gyroscope views. | audit_provenance_v1 | audit_provenance_v1 | Roll/pitch EKF context: Sabatini (2011), DOI:10.3390/s110201482; project equations/noise are persisted in resolved config |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | motion_detector | formal_local_supervised_motion_detector_v2 | disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_view":"RED/IR + processed physical A_dyn/GX/GY/GZ","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"enabled":false} |  | audit_provenance_v1 | not_applicable | N/A — component was not executed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | peak_detector | aboy_project_v1 | enabled | {"channels":["RED","IR"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_view":"x_analysis/x_native","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"detector_id":"aboy_project_v1","failure_action":"fail_closed_no_fallback","min_observation_sec":8.0,"min_peaks":5} | Historical shared-preprocessing Aboy-family project detector. | audit_provenance_v1 | audit_provenance_v1 | Historical project adaptation; algorithm family: Aboy et al. (2005), DOI:10.1109/TBME.2005.855725 |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | ppg_preprocessing | butterworth_sos | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["RED","IR"],"input_view":"repaired native PPG","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"analysis_view":{"additional_filter":"none","direct_source":"x_filter_0p2_to_8hz","non_identity_semantics":"rate_only","non_identity_source":"aligned_x_ar"},"gap_repair":{"all_missing_channel_action":"reject_record","edge_extrapolation":false,"max_gap_samples":100,"method":"linear_inside_only"},"ppg_filter":{"family":"butterworth_sos","high_hz":8.0,"low_hz":0.2,"notch_enabled":false,"order":3,"phase":"zero_phase","short_signal_policy":"reject"}} | executable PPG filter family; order, passband, phase, notch and short-signal policy are validated config parameters rather than separate profile IDs | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | representation | raw | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"input_contract":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"representation_mode":"raw"} | Line A window->file->participant; Line B window->file->role_family->participant | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.representations.raw; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | signal_views_and_scaling | parallel_physical_analysis_and_dl_views | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"views":["processed_imu_physical","x_dl_all8_window_norm","x_analysis/x_native"]} | {"dl_resampling":{"enabled":false,"method":"polyphase_anti_alias","preserve_feature_grid_hz":400.0,"target_fs_hz":400.0},"normalization":{"clip_after_scale":[-8.0,8.0],"iqr_fallback":"standard_deviation_then_finite_one","mad_consistency_divisor":0.6744897501960817,"raw_imu":"outer_train_robust","raw_ppg":"per_window_robust","robust_iqr_divisor":1.349,"scale_epsilon":1e-08,"standard_ddof":0}} | Persisted signal_views_and_scaling contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: signal_views_and_scaling; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | split_registry | frailty3_future_corrected_sgkf5_v2 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","groups":"participant_id","labels":"frailty_class","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"n_repeats":5,"n_splits":5,"path":"splits/sgkf5_repeated_grouped_5x5_v2.csv","registry_id":"frailty3_future_corrected_sgkf5_v2","runtime_recompute":false,"split_seeds":[42,10042,20042,30042,40042]} | Persisted split_registry contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: split_registry; no separate external literature source claimed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | sqi | quality_off | disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["x_analysis","pulse train","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"components":[],"failure_action":"fail_closed","fit_scope":"not_applied_off","flatline_duration_s":1.0,"high_quality_rule":"not_applied","long_gap_max_samples":100,"mode":"off","window_selection":{"application_scope":"outer_train_only","keep_fraction":1.0,"policy":"none","score_algorithm":"legacy_cardiac_motion_window_sqi_v1"}} |  | audit_provenance_v1 | not_applicable | N/A — component was not executed |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | trainer | adam | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":64,"cache_policy":"disabled","class_count_basis":"participant","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"default_10","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":10,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adam","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"balance_line_weighted_v2","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | Adam optimizer with persisted learning-rate, beta and epsilon parameters. | audit_provenance_v1 | audit_provenance_v1 | Kingma & Ba (2015), Adam, DOI:10.48550/arXiv.1412.6980 |
| compact_cnn__l0_legacy64_w15_fixed10; compact_cnn__l1_legacy64_w5_fixed10; compact_cnn__l2_legacy400_w5_fixed10; compact_cnn__l3_v2_imu_window_scaled_fixed10; compact_cnn__l4_v2_imu_fold_scaled_fixed10; compact_cnn__l5_uniform_replacement_fixed10; compact_cnn__l6_v2_line_b_balance_fixed10; compact_cnn__l7_v2_training_bundle_fixed10; inception_full__l0_legacy64_w15_fixed10 | window_planner | window_plan_v1 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["x_dl_all8_window_norm","x_analysis/x_native","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"engineering":{"cap_fraction_per_file":null,"cap_per_file":null,"end_alignment":"left_start_regular_grid","hop_s":5.0,"length_s":10.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"raw_dl":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"shared_planner_version":"window_plan_v1"} | Persisted window_planner contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: window_planner; no separate external literature source claimed |

## Model/module-owned reporter contracts and literature

The complete generated methods record is in [REPORT_METHODS.md](REPORT_METHODS.md). Profiles are selected from the persisted component identities and affect presentation only—not training, predictions, thresholds, or ranking.

| Profile | Kind | Scope | Components | Required tables | Required figures | Literature | Module sources |
|---|---|---|---|---|---|---|---|
| audit_provenance_v1 | endpoint_or_module | Configuration and provenance audit | ["aggregation:line_b_equal_role_families", "dataset_adapter:frailty3_m2_20260815_a054800abda272f6", "denoiser:identity", "evaluation:balanced_accuracy", "feature_extractor:feature_vector_282_v3", "imu_preprocessing:calibrated_roll_pitch_ekf", "motion_detector:formal_local_supervised_motion_detector_v2", "peak_detector:aboy_project_v1", "ppg_preprocessing:butterworth_sos", "representation:raw", "signal_views_and_scaling:parallel_physical_analysis_and_dl_views", "split_registry:frailty3_future_corrected_sgkf5_v2", "sqi:quality_off", "trainer:adam", "window_planner:window_plan_v1"] | ["test_components", "reproducibility_summary"] | [] | [] | ["Historical project adaptation; algorithm family: Aboy et al. (2005), DOI:10.1109/TBME.2005.855725", "Kingma & Ba (2015), Adam, DOI:10.48550/arXiv.1412.6980", "N/A — component was not executed", "Project component-role audit binding: dataset_adapter; no separate external literature source claimed", "Project component-role audit binding: evaluation; no separate external literature source claimed", "Project component-role audit binding: feature_extractor; no separate external literature source claimed", "Project component-role audit binding: signal_views_and_scaling; no separate external literature source claimed", "Project component-role audit binding: split_registry; no separate external literature source claimed", "Project component-role audit binding: window_planner; no separate external literature source claimed", "Project registry implementation: ppg_frailty.representations.raw; no separate external literature source claimed", "Project registry implementation: ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config; no separate external literature source claimed", "Project registry implementation: ppg_frailty.training.aggregation.aggregate_hierarchy; no separate external literature source claimed", "Roll/pitch EKF context: Sabatini (2011), DOI:10.3390/s110201482; project equations/noise are persisted in resolved config"] |
| compactcnn_model_v1 | model_or_module_extension | CompactCNN1D model extension | ["classifier:CompactCNN1D"] | ["training_history_raw", "test_components"] | ["learning_curves", "top_learning_curves", "balanced_accuracy_learning_curves"] | [] | ["Project implementation: src/ppg_frailty/models/compact_cnn.py"] |
| inceptiontime_single_network_model_v1 | model_or_module_extension | InceptionTime single-network model extension | ["classifier:InceptionTimeFull"] | ["training_history_raw", "test_components"] | ["learning_curves", "top_learning_curves", "balanced_accuracy_learning_curves", "top_balanced_accuracy_learning_curves"] | [] | ["Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y"] |
| multiclass_participant_oof_v1 | endpoint_or_module | Multiclass frailty classifier | ["classifier:CompactCNN1D", "classifier:InceptionTimeFull"] | ["case_summary", "metric_distribution_summary", "repeat_metrics", "repeat_per_class_metrics", "per_class_metrics", "confusion_matrices", "classification_prediction_scores", "classification_prediction_tsne", "classification_roc_curves", "classification_diagnostic_status", "paired_participant_inference", "comparison_conclusions"] | ["classification_prediction_scores", "classification_prediction_tsne", "classification_roc_auc_curves", "leaderboard", "stability", "macro_f1_stability", "roc_pr_auc_stability", "per_class_metric_stability", "confusion_matrices", "confusion_matrices_row_normalized", "per_class", "calibration"] | ["Brodersen et al. (2010), balanced accuracy, DOI:10.1109/ICPR.2010.764", "Sokolova & Lapalme (2009), classification measures including F-score, DOI:10.1016/j.ipm.2009.03.002", "Fawcett (2006), ROC analysis, DOI:10.1016/j.patrec.2005.10.010", "Efron & Tibshirani (1993), An Introduction to the Bootstrap, DOI:10.1007/978-1-4899-4541-9", "Holm (1979), sequentially rejective multiple testing, DOI:10.2307/4615733"] | ["Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y", "Project implementation: src/ppg_frailty/models/compact_cnn.py"] |

## Comprehensive comparison and confidence-qualified conclusion

P values are null-hypothesis tail probabilities, not the probability that a model is best. Repeat Student-t CIs and participant-cluster bootstrap CIs are kept separate. The lossless table and full narrative are in [RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md) and `tables/comparison_conclusions.json`.

| Rank | Case | BA mean ± SD (%) | BA participant-cluster 95% CI (%) | Macro-F1 mean ± SD (%) | Macro-F1 participant-cluster 95% CI (%) | Macro ROC-AUC mean ± SD (%) | Worst-fold BA (%) | Worst-class F1 (%) | BA Holm P | F1 Holm P | P-value role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | inception_full__l0_legacy64_w15_fixed10 | 57.9 (SD N/A; n=1 repeat) | N/A | 58.0 (SD N/A; n=1 repeat) | N/A | 69.7 (SD N/A; n=1 repeat) | 44.4 | 50.0 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 2 | compact_cnn__l2_legacy400_w5_fixed10 | 53.2 (SD N/A; n=1 repeat) | N/A | 54.0 (SD N/A; n=1 repeat) | N/A | 69.7 (SD N/A; n=1 repeat) | 33.3 | 46.2 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 3 | compact_cnn__l7_v2_training_bundle_fixed10 | 50.0 (SD N/A; n=1 repeat) | N/A | 48.5 (SD N/A; n=1 repeat) | N/A | 60.3 (SD N/A; n=1 repeat) | 33.3 | 40.0 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 4 | compact_cnn__l6_v2_line_b_balance_fixed10 | 49.5 (SD N/A; n=1 repeat) | N/A | 49.4 (SD N/A; n=1 repeat) | N/A | 63.9 (SD N/A; n=1 repeat) | 22.2 | 42.1 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 5 | compact_cnn__l1_legacy64_w5_fixed10 | 45.8 (SD N/A; n=1 repeat) | N/A | 46.3 (SD N/A; n=1 repeat) | N/A | 68.3 (SD N/A; n=1 repeat) | 16.7 | 37.5 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 6 | compact_cnn__l0_legacy64_w15_fixed10 | 42.6 (SD N/A; n=1 repeat) | N/A | 41.8 (SD N/A; n=1 repeat) | N/A | 62.7 (SD N/A; n=1 repeat) | 16.7 | 36.4 | N/A | N/A | N/A_no_eligible_paired_comparison |
| 7 | compact_cnn__l3_v2_imu_window_scaled_fixed10 | 39.8 (SD N/A; n=1 repeat) | N/A | 40.8 (SD N/A; n=1 repeat) | N/A | 62.5 (SD N/A; n=1 repeat) | 33.3 | 36.4 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 8 | compact_cnn__l5_uniform_replacement_fixed10 | 38.0 (SD N/A; n=1 repeat) | N/A | 36.8 (SD N/A; n=1 repeat) | N/A | 59.7 (SD N/A; n=1 repeat) | 11.1 | 26.7 | 1.0000 | 1.0000 | declared_reference_confirmatory |
| 9 | compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 29.2 (SD N/A; n=1 repeat) | N/A | 28.4 (SD N/A; n=1 repeat) | N/A | 54.7 (SD N/A; n=1 repeat) | 0.0 | 18.2 | 1.0000 | 1.0000 | declared_reference_confirmatory |

### Conclusions by evidence angle

| Angle | Case | Finding | Confidence | Selection effect |
|---|---|---|---|---|
| point_estimates | inception_full__l0_legacy64_w15_fixed10 | Highest participant-OOF BA is inception_full__l0_legacy64_w15_fixed10: 57.9 (SD N/A; n=1 repeat) percent; Macro-F1 58.0 (SD N/A; n=1 repeat) percent; macro ROC-AUC 69.7 (SD N/A; n=1 repeat) percent. | descriptive | none_by_itself |
| uncertainty | inception_full__l0_legacy64_w15_fixed10 | Repeat t-CI and participant-cluster percentile CI are reported separately; marginal CI overlap is not used as a significance test. | evidence_completeness_moderate_reduced_resource_or_ci | supports_precision_audit_only |
| paired_inference | N/A | 8 candidate contrasts use declared_reference_confirmatory paired participant-cluster P values with metric-wise Holm correction. | confirmatory | none_automatic |
| robustness | inception_full__l0_legacy64_w15_fixed10 | Worst-fold BA=44.4%; worst-class F1=50.0%. These stress metrics can disagree with mean BA ranking. | evidence_completeness_moderate_reduced_resource_or_ci | secondary_review |
| selection | N/A | Persisted choice=none by manual review only; no automatic selection in ordinary study reporter; participant-OOF point-estimate top=inception_full__l0_legacy64_w15_fixed10; agreement=N/A. This is a ablation choice, not an independent final-test winner. | not_established_no_selection | retain_persisted_choice_without_rewriting_history |

## Seed and data-split reproducibility

- Audit status: **PASS**
- Scope: manifest_cases_and_selected_artifact_roots_only
- Planned / observed selected cells: 45 / 45
- Split seeds by repeat: {"0": [42]}
- Errors / not-verifiable items: 0 / 0
- This is report-only evidence; it never gates training or report generation.

| Case | Selected status | Selected attempt | Excluded attempts | Planned cells | Observed cells | Declared seed policy | Effective seed policy | Split seeds | Model seeds | Orchestration seeds | Evaluation seeds | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inception_full__l0_legacy64_w15_fixed10 | passed | 2 | ["attempt_001"] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l7_v2_training_bundle_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l5_uniform_replacement_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l6_v2_line_b_balance_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l2_legacy400_w5_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l1_legacy64_w5_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |
| compact_cnn__l0_legacy64_w15_fixed10 | passed | 1 | [] | 5 | 5 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42] | [42] | [42] | [42] | PASS |

<details><summary>Per-cell seed and split evidence</summary>

| Case | Repeat | Fold | Cell status | Attempt | Declared policy | Effective policy | Split seed | Orchestration seed | Training seed | Model/member seeds | Member-seed semantics | Evaluation seed | Epoch RNG rows | Split CSV SHA256 | Fold membership SHA256 | Train participants | OOF participants | Train/OOF overlap | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inception_full__l0_legacy64_w15_fixed10 | 0 | 0 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__l0_legacy64_w15_fixed10 | 0 | 1 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__l0_legacy64_w15_fixed10 | 0 | 2 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__l0_legacy64_w15_fixed10 | 0 | 3 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__l0_legacy64_w15_fixed10 | 0 | 4 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |

</details>

### Frozen split roster

| Repeat | Fold | Split seed | Split CSV SHA256 | Declared authority JSON SHA256 | Declared authority payload SHA256 | Train participants | OOF participants | Overlap | Matching cases | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__l0_legacy64_w15_fixed10", "compact_cnn__l1_legacy64_w5_fixed10", "compact_cnn__l2_legacy400_w5_fixed10", "compact_cnn__l3_v2_imu_window_scaled_fixed10", "compact_cnn__l4_v2_imu_fold_scaled_fixed10", "compact_cnn__l5_uniform_replacement_fixed10", "compact_cnn__l6_v2_line_b_balance_fixed10", "compact_cnn__l7_v2_training_bundle_fixed10", "inception_full__l0_legacy64_w15_fixed10"] | PASS |
| 0 | 1 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__l0_legacy64_w15_fixed10", "compact_cnn__l1_legacy64_w5_fixed10", "compact_cnn__l2_legacy400_w5_fixed10", "compact_cnn__l3_v2_imu_window_scaled_fixed10", "compact_cnn__l4_v2_imu_fold_scaled_fixed10", "compact_cnn__l5_uniform_replacement_fixed10", "compact_cnn__l6_v2_line_b_balance_fixed10", "compact_cnn__l7_v2_training_bundle_fixed10", "inception_full__l0_legacy64_w15_fixed10"] | PASS |
| 0 | 2 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__l0_legacy64_w15_fixed10", "compact_cnn__l1_legacy64_w5_fixed10", "compact_cnn__l2_legacy400_w5_fixed10", "compact_cnn__l3_v2_imu_window_scaled_fixed10", "compact_cnn__l4_v2_imu_fold_scaled_fixed10", "compact_cnn__l5_uniform_replacement_fixed10", "compact_cnn__l6_v2_line_b_balance_fixed10", "compact_cnn__l7_v2_training_bundle_fixed10", "inception_full__l0_legacy64_w15_fixed10"] | PASS |
| 0 | 3 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__l0_legacy64_w15_fixed10", "compact_cnn__l1_legacy64_w5_fixed10", "compact_cnn__l2_legacy400_w5_fixed10", "compact_cnn__l3_v2_imu_window_scaled_fixed10", "compact_cnn__l4_v2_imu_fold_scaled_fixed10", "compact_cnn__l5_uniform_replacement_fixed10", "compact_cnn__l6_v2_line_b_balance_fixed10", "compact_cnn__l7_v2_training_bundle_fixed10", "inception_full__l0_legacy64_w15_fixed10"] | PASS |
| 0 | 4 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__l0_legacy64_w15_fixed10", "compact_cnn__l1_legacy64_w5_fixed10", "compact_cnn__l2_legacy400_w5_fixed10", "compact_cnn__l3_v2_imu_window_scaled_fixed10", "compact_cnn__l4_v2_imu_fold_scaled_fixed10", "compact_cnn__l5_uniform_replacement_fixed10", "compact_cnn__l6_v2_line_b_balance_fixed10", "compact_cnn__l7_v2_training_bundle_fixed10", "inception_full__l0_legacy64_w15_fixed10"] | PASS |

## Varied and controlled parameters

- Explicit deterministic sparse catalog profiles; this is a screening comparison, not a single-factor causal ablation.
- Search method: deterministic_sparse_profiles
- Runtime parameter sampling: False
- Profile-design seed: 42
- Interpretation: The catalog supplies architecture provenance only. Legacy bridge profiles are declared below and are not canonical V2 overrides.

The complete resolved varied/controlled tables are [varied_parameters.csv](tables/varied_parameters.csv) and [controlled_parameters.csv](tables/controlled_parameters.csv). Execution controls such as jobs are not scientific grid variables.

<details><summary>Complete controlled-parameter list (218 rows)</summary>

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
| artifact.failure_action | no_result_no_fallback |
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
| features.engineering_sequence_schema | engineering_10s_hop5s_thesis_115_v2 |
| features.file_aggregation | ["mean", "population_sd"] |
| features.file_vector_schema | feature_vector_282_v3 |
| features.matrix_k | 32 |
| features.matrix_schema | ordered_feature_matrix_d794_by_32_registry-0bea68a2058d_v3 |
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
| model.dropout | 0.2 |
| model.input_channel_order | ["RED", "IR", "A_dyn_x", "A_dyn_y", "A_dyn_z", "GX", "GY", "GZ"] |
| model.input_channels | 8 |
| model.input_channels_resolution | canonical_frailty_raw_8 |
| model.mask_aware_pooling | True |
| model.n_classes | 3 |
| model.seed_policy | outer_cv_repeat_seed_equals_split_seed |
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
| signal.dl_resampling.enabled | False |
| signal.dl_resampling.method | polyphase_anti_alias |
| signal.dl_resampling.preserve_feature_grid_hz | 400.0 |
| signal.dl_resampling.target_fs_hz | 400.0 |
| signal.gap_repair.all_missing_channel_action | reject_record |
| signal.gap_repair.edge_extrapolation | False |
| signal.gap_repair.max_gap_samples | 100 |
| signal.gap_repair.method | linear_inside_only |
| signal.gyroscope_input_unit | deg/s |
| signal.imu.calibration_start_s | 5.0 |
| signal.imu.calibration_stop_s | 100.0 |
| signal.imu.comparison_method | profile_a_lowpass_0p3hz |
| signal.imu.dynamic_observation_scale | 3.0 |
| signal.imu.failure_action | fail_closed |
| signal.imu.gravity_method | calibrated_roll_pitch_ekf |
| signal.imu.initial_covariance_diagonal | [1.0, 1.0, 0.5, 0.5, 0.5] |
| signal.imu.initialization | same_participant_static_calibration |
| signal.imu.observation_covariance_diagonal_rad2 | [0.5, 0.5] |
| signal.imu.output_units.acceleration | m/s^2 |
| signal.imu.output_units.gyroscope | rad/s |
| signal.imu.output_units.jerk | m/s^3 |
| signal.imu.process_covariance_diagonal_per_second | [5.0, 5.0, 0.05, 0.05, 0.05] |
| signal.imu.required_axes | 6 |
| signal.imu.sensor_filter_order | 3 |
| signal.imu.sensor_lowpass_acc_hz | 20.0 |
| signal.imu.sensor_lowpass_gyro_hz | 40.0 |
| signal.internal_fs_hz | 400.0 |
| signal.normalization.clip_after_scale | [-8.0, 8.0] |
| signal.normalization.iqr_fallback | standard_deviation_then_finite_one |
| signal.normalization.mad_consistency_divisor | 0.6744897501960817 |
| signal.normalization.raw_imu | outer_train_robust |
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
| training.batch_size | 64 |
| training.cache_policy | disabled |
| training.class_count_basis | participant |
| training.class_weight_beta | 0.999 |
| training.class_weighting | inverse_frequency |
| training.classifier_role_families | ["B", "R"] |
| training.deterministic_algorithms | True |
| training.device | cuda |
| training.epoch_profile | default_10 |
| training.epoch_rule | fixed_epoch |
| training.execution_mode | formal |
| training.fixed_epochs | 10 |
| training.focal_gamma | 2.0 |
| training.gradient_clip_norm |  |
| training.inner_grouped_folds | 0 |
| training.inner_patience | 0 |
| training.label_smoothing | 0.0 |
| training.learning_rate | 0.001 |
| training.loss | cross_entropy |
| training.maximum_inner_epochs | 0 |
| training.n_classes | 3 |
| training.num_workers | 0 |
| training.optimizer | adam |
| training.optimizer_parameters.amsgrad | False |
| training.optimizer_parameters.betas | [0.9, 0.999] |
| training.optimizer_parameters.eps | 1e-08 |
| training.optimizer_parameters.maximize | False |
| training.outer_labels_visible_to_trainer | False |
| training.participant_window_quota | all |
| training.refit_on_all_outer_training | True |
| training.sampler | balance_line_weighted_v2 |
| training.samples_per_epoch |  |
| training.seed | 42 |
| training.training_balance | equal_role_families |
| training.weight_decay | 0.0001 |
| windows.engineering.cap_fraction_per_file |  |
| windows.engineering.cap_per_file |  |
| windows.engineering.end_alignment | left_start_regular_grid |
| windows.engineering.hop_s | 5.0 |
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

| Rank | Case | Abstention-aware BA, mean ± SD (%) | Abstention-aware precision | Abstention-aware recall | Abstention-aware Macro-F1, mean ± SD (%) | Participant coverage | Abstentions | Abstentions by class | Conditional BA, mean ± SD (%) | Conditional BA cluster 95% CI (%) | Conditional Macro-F1, mean ± SD (%) | Conditional Macro-F1 cluster 95% CI (%) | Macro ROC AUC, mean ± SD (%) | Macro PR AUC, mean ± SD (%) | Conditional BA LCB95 | Conditional Macro-F1 LCB95 | Aware worst-fold BA | Conditional worst-fold BA | Worst recall | Worst F1 | Source | Frailty endpoint | Motion auxiliary outer-OOF | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | inception_full__l0_legacy64_w15_fixed10 | 57.9 | 0.5893 | 0.5787 | 58.0 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 57.9 | N/A | 58.0 | N/A | 69.7 | 61.7 | N/A | N/A | N/A | 0.4444 | 0.4444 | 0.5000 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 2 | compact_cnn__l2_legacy400_w5_fixed10 | 53.2 | 0.5750 | 0.5324 | 54.0 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 53.2 | N/A | 54.0 | N/A | 69.7 | 61.4 | N/A | N/A | N/A | 0.3333 | 0.3750 | 0.4615 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 3 | compact_cnn__l7_v2_training_bundle_fixed10 | 50.0 | 0.5000 | 0.5000 | 48.5 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 50.0 | N/A | 48.5 | N/A | 60.3 | 48.7 | N/A | N/A | N/A | 0.3333 | 0.3333 | 0.4000 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 4 | compact_cnn__l6_v2_line_b_balance_fixed10 | 49.5 | 0.4932 | 0.4954 | 49.4 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 49.5 | N/A | 49.4 | N/A | 63.9 | 50.2 | N/A | N/A | N/A | 0.2222 | 0.4167 | 0.4211 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 5 | compact_cnn__l1_legacy64_w5_fixed10 | 45.8 | 0.4997 | 0.4583 | 46.3 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 45.8 | N/A | 46.3 | N/A | 68.3 | 59.5 | N/A | N/A | N/A | 0.1667 | 0.3333 | 0.3750 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 6 | compact_cnn__l0_legacy64_w15_fixed10 | 42.6 | 0.4212 | 0.4259 | 41.8 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 42.6 | N/A | 41.8 | N/A | 62.7 | 51.0 | N/A | N/A | N/A | 0.1667 | 0.3333 | 0.3636 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 7 | compact_cnn__l3_v2_imu_window_scaled_fixed10 | 39.8 | 0.5788 | 0.3981 | 40.8 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 39.8 | N/A | 40.8 | N/A | 62.5 | 55.0 | N/A | N/A | N/A | 0.3333 | 0.2500 | 0.3636 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 8 | compact_cnn__l5_uniform_replacement_fixed10 | 38.0 | 0.3712 | 0.3796 | 36.8 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 38.0 | N/A | 36.8 | N/A | 59.7 | 51.3 | N/A | N/A | N/A | 0.1111 | 0.2222 | 0.2667 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 9 | compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 29.2 | 0.3135 | 0.2917 | 28.4 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 29.2 | N/A | 28.4 | N/A | 54.7 | 42.5 | N/A | N/A | N/A | 0.0000 | 0.1250 | 0.1818 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |

## Repeat-level predictive distributions

Mean and sample SD are shown in one percentage column and the two-sided repeat-level Student-t 95% CI is shown beside it. Lossless bounds, range, mean, and SD remain in the matching JSON table.

| Case | Metric | Repeats | Mean ± SD (%) | Repeat 95% CI (%) | Source |
|---|---|---|---|---|---|
| inception_full__l0_legacy64_w15_fixed10 | balanced_accuracy | 1 | 57.9 | N/A | participant_oof |
| inception_full__l0_legacy64_w15_fixed10 | macro_f1 | 1 | 58.0 | N/A | participant_oof |
| inception_full__l0_legacy64_w15_fixed10 | macro_roc_auc_ovr | 1 | 69.7 | N/A | participant_oof |
| inception_full__l0_legacy64_w15_fixed10 | macro_pr_auc_ovr | 1 | 61.7 | N/A | participant_oof |
| inception_full__l0_legacy64_w15_fixed10 | abstention_aware_balanced_accuracy | 1 | 57.9 | N/A | participant_oof |
| inception_full__l0_legacy64_w15_fixed10 | abstention_aware_macro_f1 | 1 | 58.0 | N/A | participant_oof |
| compact_cnn__l7_v2_training_bundle_fixed10 | balanced_accuracy | 1 | 50.0 | N/A | participant_oof |
| compact_cnn__l7_v2_training_bundle_fixed10 | macro_f1 | 1 | 48.5 | N/A | participant_oof |
| compact_cnn__l7_v2_training_bundle_fixed10 | macro_roc_auc_ovr | 1 | 60.3 | N/A | participant_oof |
| compact_cnn__l7_v2_training_bundle_fixed10 | macro_pr_auc_ovr | 1 | 48.7 | N/A | participant_oof |
| compact_cnn__l7_v2_training_bundle_fixed10 | abstention_aware_balanced_accuracy | 1 | 50.0 | N/A | participant_oof |
| compact_cnn__l7_v2_training_bundle_fixed10 | abstention_aware_macro_f1 | 1 | 48.5 | N/A | participant_oof |
| compact_cnn__l5_uniform_replacement_fixed10 | balanced_accuracy | 1 | 38.0 | N/A | participant_oof |
| compact_cnn__l5_uniform_replacement_fixed10 | macro_f1 | 1 | 36.8 | N/A | participant_oof |
| compact_cnn__l5_uniform_replacement_fixed10 | macro_roc_auc_ovr | 1 | 59.7 | N/A | participant_oof |
| compact_cnn__l5_uniform_replacement_fixed10 | macro_pr_auc_ovr | 1 | 51.3 | N/A | participant_oof |
| compact_cnn__l5_uniform_replacement_fixed10 | abstention_aware_balanced_accuracy | 1 | 38.0 | N/A | participant_oof |
| compact_cnn__l5_uniform_replacement_fixed10 | abstention_aware_macro_f1 | 1 | 36.8 | N/A | participant_oof |
| compact_cnn__l6_v2_line_b_balance_fixed10 | balanced_accuracy | 1 | 49.5 | N/A | participant_oof |
| compact_cnn__l6_v2_line_b_balance_fixed10 | macro_f1 | 1 | 49.4 | N/A | participant_oof |
| compact_cnn__l6_v2_line_b_balance_fixed10 | macro_roc_auc_ovr | 1 | 63.9 | N/A | participant_oof |
| compact_cnn__l6_v2_line_b_balance_fixed10 | macro_pr_auc_ovr | 1 | 50.2 | N/A | participant_oof |
| compact_cnn__l6_v2_line_b_balance_fixed10 | abstention_aware_balanced_accuracy | 1 | 49.5 | N/A | participant_oof |
| compact_cnn__l6_v2_line_b_balance_fixed10 | abstention_aware_macro_f1 | 1 | 49.4 | N/A | participant_oof |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | balanced_accuracy | 1 | 29.2 | N/A | participant_oof |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | macro_f1 | 1 | 28.4 | N/A | participant_oof |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | macro_roc_auc_ovr | 1 | 54.7 | N/A | participant_oof |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | macro_pr_auc_ovr | 1 | 42.5 | N/A | participant_oof |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | abstention_aware_balanced_accuracy | 1 | 29.2 | N/A | participant_oof |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | abstention_aware_macro_f1 | 1 | 28.4 | N/A | participant_oof |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | balanced_accuracy | 1 | 39.8 | N/A | participant_oof |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | macro_f1 | 1 | 40.8 | N/A | participant_oof |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | macro_roc_auc_ovr | 1 | 62.5 | N/A | participant_oof |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | macro_pr_auc_ovr | 1 | 55.0 | N/A | participant_oof |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | abstention_aware_balanced_accuracy | 1 | 39.8 | N/A | participant_oof |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | abstention_aware_macro_f1 | 1 | 40.8 | N/A | participant_oof |
| compact_cnn__l2_legacy400_w5_fixed10 | balanced_accuracy | 1 | 53.2 | N/A | participant_oof |
| compact_cnn__l2_legacy400_w5_fixed10 | macro_f1 | 1 | 54.0 | N/A | participant_oof |
| compact_cnn__l2_legacy400_w5_fixed10 | macro_roc_auc_ovr | 1 | 69.7 | N/A | participant_oof |
| compact_cnn__l2_legacy400_w5_fixed10 | macro_pr_auc_ovr | 1 | 61.4 | N/A | participant_oof |
| compact_cnn__l2_legacy400_w5_fixed10 | abstention_aware_balanced_accuracy | 1 | 53.2 | N/A | participant_oof |
| compact_cnn__l2_legacy400_w5_fixed10 | abstention_aware_macro_f1 | 1 | 54.0 | N/A | participant_oof |
| compact_cnn__l1_legacy64_w5_fixed10 | balanced_accuracy | 1 | 45.8 | N/A | participant_oof |
| compact_cnn__l1_legacy64_w5_fixed10 | macro_f1 | 1 | 46.3 | N/A | participant_oof |
| compact_cnn__l1_legacy64_w5_fixed10 | macro_roc_auc_ovr | 1 | 68.3 | N/A | participant_oof |
| compact_cnn__l1_legacy64_w5_fixed10 | macro_pr_auc_ovr | 1 | 59.5 | N/A | participant_oof |
| compact_cnn__l1_legacy64_w5_fixed10 | abstention_aware_balanced_accuracy | 1 | 45.8 | N/A | participant_oof |
| compact_cnn__l1_legacy64_w5_fixed10 | abstention_aware_macro_f1 | 1 | 46.3 | N/A | participant_oof |
| compact_cnn__l0_legacy64_w15_fixed10 | balanced_accuracy | 1 | 42.6 | N/A | participant_oof |
| compact_cnn__l0_legacy64_w15_fixed10 | macro_f1 | 1 | 41.8 | N/A | participant_oof |
| compact_cnn__l0_legacy64_w15_fixed10 | macro_roc_auc_ovr | 1 | 62.7 | N/A | participant_oof |
| compact_cnn__l0_legacy64_w15_fixed10 | macro_pr_auc_ovr | 1 | 51.0 | N/A | participant_oof |
| compact_cnn__l0_legacy64_w15_fixed10 | abstention_aware_balanced_accuracy | 1 | 42.6 | N/A | participant_oof |
| compact_cnn__l0_legacy64_w15_fixed10 | abstention_aware_macro_f1 | 1 | 41.8 | N/A | participant_oof |

<details><summary>Per-class repeat distributions</summary>

| Case | Class | Metric | Repeats | Mean ± SD (%) | Repeat 95% CI (%) |
|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 62.2 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Pre-Frail | f1 | 1 | 47.1 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Pre-Frail | recall | 1 | 44.4 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Pre-Frail | specificity | 1 | 80.0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 67.8 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 62.8 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 49.0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Robust/Non-Frail | f1 | 1 | 36.4 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Robust/Non-Frail | recall | 1 | 33.3 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Robust/Non-Frail | specificity | 1 | 64.7 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 50.0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 45.2 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Young | balanced_accuracy_ovr | 1 | 58.3 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Young | f1 | 1 | 42.1 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Young | recall | 1 | 50.0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Young | specificity | 1 | 66.7 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Young | roc_auc_ovr | 1 | 70.2 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | Young | pr_auc_ovr | 1 | 45.1 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 56.7 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Pre-Frail | f1 | 1 | 37.5 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Pre-Frail | recall | 1 | 33.3 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Pre-Frail | specificity | 1 | 80.0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 66.1 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 60.2 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 56.9 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Robust/Non-Frail | f1 | 1 | 55.2 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Robust/Non-Frail | recall | 1 | 66.7 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Robust/Non-Frail | specificity | 1 | 47.1 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 57.4 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 47.3 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Young | balanced_accuracy_ovr | 1 | 64.0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Young | f1 | 1 | 46.2 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Young | recall | 1 | 37.5 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Young | specificity | 1 | 90.5 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Young | roc_auc_ovr | 1 | 81.5 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | Young | pr_auc_ovr | 1 | 71.1 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 70.3 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Pre-Frail | f1 | 1 | 58.8 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Pre-Frail | recall | 1 | 55.6 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Pre-Frail | specificity | 1 | 85.0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 77.8 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 71.4 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 59.8 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Robust/Non-Frail | f1 | 1 | 57.1 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Robust/Non-Frail | recall | 1 | 66.7 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Robust/Non-Frail | specificity | 1 | 52.9 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 59.8 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 57.1 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Young | balanced_accuracy_ovr | 1 | 64.0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Young | f1 | 1 | 46.2 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Young | recall | 1 | 37.5 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Young | specificity | 1 | 90.5 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Young | roc_auc_ovr | 1 | 71.4 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | Young | pr_auc_ovr | 1 | 55.8 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 49.7 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Pre-Frail | f1 | 1 | 36.4 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Pre-Frail | recall | 1 | 44.4 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Pre-Frail | specificity | 1 | 55.0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 60.6 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 54.5 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 51.5 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Robust/Non-Frail | f1 | 1 | 46.2 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Robust/Non-Frail | recall | 1 | 50.0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Robust/Non-Frail | specificity | 1 | 52.9 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 54.9 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 46.1 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Young | balanced_accuracy_ovr | 1 | 62.5 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Young | f1 | 1 | 40.0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Young | recall | 1 | 25.0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Young | specificity | 1 | 100.0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Young | roc_auc_ovr | 1 | 72.0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | Young | pr_auc_ovr | 1 | 64.4 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 44.2 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Pre-Frail | f1 | 1 | 28.6 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Pre-Frail | recall | 1 | 33.3 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Pre-Frail | specificity | 1 | 55.0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 56.7 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 35.2 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 44.4 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Robust/Non-Frail | f1 | 1 | 38.5 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Robust/Non-Frail | recall | 1 | 41.7 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Robust/Non-Frail | specificity | 1 | 47.1 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 43.6 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 45.0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Young | balanced_accuracy_ovr | 1 | 51.5 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Young | f1 | 1 | 18.2 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Young | recall | 1 | 12.5 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Young | specificity | 1 | 90.5 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Young | roc_auc_ovr | 1 | 63.7 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | Young | pr_auc_ovr | 1 | 47.4 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 51.1 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Pre-Frail | f1 | 1 | 26.7 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Pre-Frail | recall | 1 | 22.2 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Pre-Frail | specificity | 1 | 80.0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 62.8 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 38.2 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 50.2 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Robust/Non-Frail | f1 | 1 | 41.7 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Robust/Non-Frail | recall | 1 | 41.7 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Robust/Non-Frail | specificity | 1 | 58.8 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 48.5 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 57.9 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Young | balanced_accuracy_ovr | 1 | 58.3 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Young | f1 | 1 | 42.1 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Young | recall | 1 | 50.0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Young | specificity | 1 | 66.7 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Young | roc_auc_ovr | 1 | 67.9 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | Young | pr_auc_ovr | 1 | 57.9 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 57.2 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Pre-Frail | f1 | 1 | 42.1 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Pre-Frail | recall | 1 | 44.4 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Pre-Frail | specificity | 1 | 70.0 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 67.8 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 41.4 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 53.2 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Robust/Non-Frail | f1 | 1 | 43.5 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Robust/Non-Frail | recall | 1 | 41.7 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Robust/Non-Frail | specificity | 1 | 64.7 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 47.1 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 50.2 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Young | balanced_accuracy_ovr | 1 | 74.1 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Young | f1 | 1 | 62.5 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Young | recall | 1 | 62.5 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Young | specificity | 1 | 85.7 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Young | roc_auc_ovr | 1 | 76.8 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | Young | pr_auc_ovr | 1 | 58.9 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 63.3 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Pre-Frail | f1 | 1 | 52.2 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Pre-Frail | recall | 1 | 66.7 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Pre-Frail | specificity | 1 | 60.0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 64.4 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 40.1 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 54.9 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Robust/Non-Frail | f1 | 1 | 40.0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Robust/Non-Frail | recall | 1 | 33.3 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Robust/Non-Frail | specificity | 1 | 76.5 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 43.1 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 50.8 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Young | balanced_accuracy_ovr | 1 | 67.9 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Young | f1 | 1 | 53.3 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Young | recall | 1 | 50.0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Young | specificity | 1 | 85.7 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Young | roc_auc_ovr | 1 | 73.2 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | Young | pr_auc_ovr | 1 | 55.3 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Pre-Frail | balanced_accuracy_ovr | 1 | 64.7 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Pre-Frail | f1 | 1 | 50.0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Pre-Frail | recall | 1 | 44.4 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Pre-Frail | specificity | 1 | 85.0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Pre-Frail | roc_auc_ovr | 1 | 68.9 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Pre-Frail | pr_auc_ovr | 1 | 57.0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 1 | 65.7 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Robust/Non-Frail | f1 | 1 | 61.5 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Robust/Non-Frail | recall | 1 | 66.7 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Robust/Non-Frail | specificity | 1 | 64.7 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Robust/Non-Frail | roc_auc_ovr | 1 | 61.8 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Robust/Non-Frail | pr_auc_ovr | 1 | 55.6 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Young | balanced_accuracy_ovr | 1 | 74.1 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Young | f1 | 1 | 62.5 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Young | recall | 1 | 62.5 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Young | specificity | 1 | 85.7 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Young | roc_auc_ovr | 1 | 78.6 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | Young | pr_auc_ovr | 1 | 72.4 | N/A |

</details>

## Paired participant-cluster inference

Each candidate is compared with the declared reference on the exact participant/repeat/fold/split roster. P values are two-sided participant-cluster permutation results; Holm adjustment is applied separately within BA and Macro-F1. These comparisons do not select a winner and do not turn this representation screen into a causal ablation.

| Reference | Candidate | Metric | Candidate − reference | Raw P | Holm P | Holm P ≤ 0.05 | Participants | Repeats | Permutations |
|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l1_legacy64_w5_fixed10 | balanced_accuracy | 0.0324 | 0.7342 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l1_legacy64_w5_fixed10 | macro_f1 | 0.0443 | 0.7025 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l2_legacy400_w5_fixed10 | balanced_accuracy | 0.1065 | 0.3677 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l2_legacy400_w5_fixed10 | macro_f1 | 0.1220 | 0.3012 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l3_v2_imu_window_scaled_fixed10 | balanced_accuracy | -0.0278 | 0.7995 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l3_v2_imu_window_scaled_fixed10 | macro_f1 | -0.0100 | 0.8802 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l4_v2_imu_fold_scaled_fixed10 | balanced_accuracy | -0.1343 | 0.2586 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l4_v2_imu_fold_scaled_fixed10 | macro_f1 | -0.1344 | 0.2396 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l5_uniform_replacement_fixed10 | balanced_accuracy | -0.0463 | 0.7510 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l5_uniform_replacement_fixed10 | macro_f1 | -0.0503 | 0.6637 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l6_v2_line_b_balance_fixed10 | balanced_accuracy | 0.0694 | 0.6523 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l6_v2_line_b_balance_fixed10 | macro_f1 | 0.0752 | 0.5274 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l7_v2_training_bundle_fixed10 | balanced_accuracy | 0.0741 | 0.5085 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | compact_cnn__l7_v2_training_bundle_fixed10 | macro_f1 | 0.0666 | 0.6316 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | inception_full__l0_legacy64_w15_fixed10 | balanced_accuracy | 0.1528 | 0.2263 | 1.0000 | False | 29 | 1 | 100000 |
| compact_cnn__l0_legacy64_w15_fixed10 | inception_full__l0_legacy64_w15_fixed10 | macro_f1 | 0.1617 | 0.2050 | 1.0000 | False | 29 | 1 | 100000 |

## Legacy/V2 bridge report A — numeric adjacent ablations (L0→L7)

This is the causal-interpretation table: L0 is the baseline and the next seven rows are only the predefined adjacent profile contrasts L0→L1 through L6→L7. Deltas are never taken from run order.

| Numeric order | Model | Profile | Previous numeric profile | Predefined comparison | BA legacy W | BA Line A | BA Line B | Δ BA legacy W | Δ BA Line A | Δ BA Line B | Macro-F1 legacy W | Macro-F1 Line A | Macro-F1 Line B | Δ Macro-F1 legacy W | Δ Macro-F1 Line A | Δ Macro-F1 Line B | Worst-class F1 Line B | Δ worst-class F1 | Contrast available | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | CompactCNN1D | L0 | N/A | baseline | 0.4259 | 0.4259 | 0.4213 | N/A | N/A | N/A | 0.4184 | 0.4184 | 0.4143 | N/A | N/A | N/A | 0.3529 | N/A | False | baseline_for_seven_predefined_adjacent_ablation_contrasts |
| 1 | CompactCNN1D | L1 | L0 | L0->L1 | 0.4583 | 0.4583 | 0.4028 | 0.0324 | 0.0324 | -0.0185 | 0.4628 | 0.4628 | 0.4087 | 0.0443 | 0.0443 | -0.0056 | 0.3529 | 0.0000 | True | predefined_adjacent_numeric_profile_ablation_only |
| 2 | CompactCNN1D | L2 | L1 | L1->L2 | 0.5324 | 0.5324 | 0.4537 | 0.0741 | 0.0741 | 0.0509 | 0.5404 | 0.5404 | 0.4497 | 0.0776 | 0.0776 | 0.0411 | 0.3333 | -0.0196 | True | predefined_adjacent_numeric_profile_ablation_only |
| 3 | CompactCNN1D | L3 | L2 | L2->L3 | 0.3981 | 0.3981 | 0.3704 | -0.1343 | -0.1343 | -0.0833 | 0.4084 | 0.4084 | 0.3684 | -0.1320 | -0.1320 | -0.0813 | 0.3077 | -0.0256 | True | predefined_adjacent_numeric_profile_ablation_only |
| 4 | CompactCNN1D | L4 | L3 | L3->L4 | 0.2917 | 0.2917 | 0.4028 | -0.1065 | -0.1065 | 0.0324 | 0.2840 | 0.2840 | 0.3722 | -0.1243 | -0.1243 | 0.0038 | 0.2000 | -0.1077 | True | predefined_adjacent_numeric_profile_ablation_only |
| 5 | CompactCNN1D | L5 | L4 | L4->L5 | 0.3796 | 0.3796 | 0.4583 | 0.0880 | 0.0880 | 0.0556 | 0.3681 | 0.3681 | 0.4593 | 0.0841 | 0.0841 | 0.0871 | 0.3529 | 0.1529 | True | predefined_adjacent_numeric_profile_ablation_only |
| 6 | CompactCNN1D | L6 | L5 | L5->L6 | 0.4954 | 0.4954 | 0.4954 | 0.1157 | 0.1157 | 0.0370 | 0.4879 | 0.4879 | 0.4936 | 0.1198 | 0.1198 | 0.0343 | 0.4211 | 0.0681 | True | predefined_adjacent_numeric_profile_ablation_only |
| 7 | CompactCNN1D | L7 | L6 | L6->L7 | 0.3750 | 0.3750 | 0.5000 | -0.1204 | -0.1204 | 0.0046 | 0.3815 | 0.3815 | 0.4850 | -0.1065 | -0.1065 | -0.0086 | 0.4000 | -0.0211 | True | predefined_adjacent_numeric_profile_ablation_only |

## Legacy/V2 bridge report B — CompactCNN execution order

This table lists absolute W/A/B metrics in the requested L7→L5→L6→L4→L3→L2→L1→L0 run order. It deliberately has no execution-order delta: L7→L5 and every other neighbouring run pair are scheduling transitions, not causal ablations.

| Execution order | Model | Profile | Previous execution profile | Execution transition | BA legacy W | BA Line A | BA Line B | Macro-F1 legacy W | Macro-F1 Line A | Macro-F1 Line B | Worst-class F1 Line B | Transition is ablation | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | CompactCNN1D | L7 | N/A | start | 0.3750 | 0.3750 | 0.5000 | 0.3815 | 0.3815 | 0.4850 | 0.4000 | False | execution_start_absolute_metrics_only |
| 2 | CompactCNN1D | L5 | L7 | L7->L5 | 0.3796 | 0.3796 | 0.4583 | 0.3681 | 0.3681 | 0.4593 | 0.3529 | False | execution_order_only_not_a_causal_ablation |
| 3 | CompactCNN1D | L6 | L5 | L5->L6 | 0.4954 | 0.4954 | 0.4954 | 0.4879 | 0.4879 | 0.4936 | 0.4211 | False | execution_order_only_not_a_causal_ablation |
| 4 | CompactCNN1D | L4 | L6 | L6->L4 | 0.2917 | 0.2917 | 0.4028 | 0.2840 | 0.2840 | 0.3722 | 0.2000 | False | execution_order_only_not_a_causal_ablation |
| 5 | CompactCNN1D | L3 | L4 | L4->L3 | 0.3981 | 0.3981 | 0.3704 | 0.4084 | 0.4084 | 0.3684 | 0.3077 | False | execution_order_only_not_a_causal_ablation |
| 6 | CompactCNN1D | L2 | L3 | L3->L2 | 0.5324 | 0.5324 | 0.4537 | 0.5404 | 0.5404 | 0.4497 | 0.3333 | False | execution_order_only_not_a_causal_ablation |
| 7 | CompactCNN1D | L1 | L2 | L2->L1 | 0.4583 | 0.4583 | 0.4028 | 0.4628 | 0.4628 | 0.4087 | 0.3529 | False | execution_order_only_not_a_causal_ablation |
| 8 | CompactCNN1D | L0 | L1 | L1->L0 | 0.4259 | 0.4259 | 0.4213 | 0.4184 | 0.4184 | 0.4143 | 0.3529 | False | execution_order_only_not_a_causal_ablation |

## Aggregation sensitivity from the same file-level OOF

The declared-source row reproduces the aggregation used by the fitted model and, when eligible, the primary leaderboard. The other row reaggregates the same held-out file probabilities post hoc. It is not a separately retrained Line A/Line B experiment and is not selection evidence.

| Case | Aggregation view | Role | Mean BA | Mean Macro-F1 | Line A − Line B BA | Line A − Line B Macro-F1 | Worst recall | Worst F1 | ECE | Repeats | Retained participant OOF n | All participant units n | Dropped participant units n | All file OOF n | Dropped files n | Source replay | Primary ranking eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | line_a_equal_files | declared_source_line | 0.4259 | 0.4184 | 0.0046 | 0.0041 | 0.3333 | 0.3636 | 0.3299 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l0_legacy64_w15_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4213 | 0.4143 | 0.0046 | 0.0041 | 0.3333 | 0.3529 | 0.3147 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l1_legacy64_w5_fixed10 | line_a_equal_files | declared_source_line | 0.4583 | 0.4628 | 0.0556 | 0.0541 | 0.3333 | 0.3750 | 0.4280 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l1_legacy64_w5_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4028 | 0.4087 | 0.0556 | 0.0541 | 0.3333 | 0.3529 | 0.3481 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l2_legacy400_w5_fixed10 | line_a_equal_files | declared_source_line | 0.5324 | 0.5404 | 0.0787 | 0.0907 | 0.3750 | 0.4615 | 0.2742 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l2_legacy400_w5_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4537 | 0.4497 | 0.0787 | 0.0907 | 0.2500 | 0.3333 | 0.2321 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | line_a_equal_files | declared_source_line | 0.3981 | 0.4084 | 0.0278 | 0.0400 | 0.2500 | 0.3636 | 0.3057 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.3704 | 0.3684 | 0.0278 | 0.0400 | 0.2500 | 0.3077 | 0.3625 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | line_a_equal_files | declared_source_line | 0.2917 | 0.2840 | -0.1111 | -0.0882 | 0.1250 | 0.1818 | 0.4389 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4028 | 0.3722 | -0.1111 | -0.0882 | 0.1250 | 0.2000 | 0.3594 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l5_uniform_replacement_fixed10 | line_a_equal_files | declared_source_line | 0.3796 | 0.3681 | -0.0787 | -0.0912 | 0.2222 | 0.2667 | 0.3357 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l5_uniform_replacement_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4583 | 0.4593 | -0.0787 | -0.0912 | 0.3333 | 0.3529 | 0.2926 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l6_v2_line_b_balance_fixed10 | line_a_equal_files | posthoc_aggregation_only | 0.4954 | 0.4879 | 0.0000 | -0.0057 | 0.4167 | 0.4211 | 0.2327 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l6_v2_line_b_balance_fixed10 | line_b_equal_role_families | declared_source_line | 0.4954 | 0.4936 | 0.0000 | -0.0057 | 0.4167 | 0.4211 | 0.2873 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__l7_v2_training_bundle_fixed10 | line_a_equal_files | posthoc_aggregation_only | 0.3750 | 0.3815 | -0.1250 | -0.1036 | 0.3333 | 0.3158 | 0.4027 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__l7_v2_training_bundle_fixed10 | line_b_equal_role_families | declared_source_line | 0.5000 | 0.4850 | -0.1250 | -0.1036 | 0.3333 | 0.4000 | 0.3608 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__l0_legacy64_w15_fixed10 | line_a_equal_files | declared_source_line | 0.5787 | 0.5801 | 0.0000 | -0.0124 | 0.4444 | 0.5000 | 0.2360 | 1 | 29 | 29 | 0 | 145 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__l0_legacy64_w15_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5787 | 0.5925 | 0.0000 | -0.0124 | 0.4444 | 0.4706 | 0.2257 | 1 | 29 | 29 | 0 | 145 | 0 | not_applicable_posthoc_view | False |

## Parallel window/file/role-balanced participant views

All three rows reuse the same fitted held-out OOF probabilities; they are not three training runs. `window_balanced_to_participant` gives every retained window equal report weight, Line A gives every file equal weight after window→file, and Line B gives every canonical role family equal weight after window→file→role. Only the declared training aggregation may support the primary leaderboard; the other views are post-hoc sensitivity plots.

| Case | Aggregation view | Evidence role | Mean BA | Mean Macro-F1 | Worst recall | Worst F1 | Repeats | Participant OOF n | Primary ranking eligible |
|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4259 | 0.4184 | 0.3333 | 0.3636 | 1 | 29 | False |
| compact_cnn__l0_legacy64_w15_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4259 | 0.4184 | 0.3333 | 0.3636 | 1 | 29 | True |
| compact_cnn__l0_legacy64_w15_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4213 | 0.4143 | 0.3333 | 0.3529 | 1 | 29 | False |
| compact_cnn__l1_legacy64_w5_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4583 | 0.4628 | 0.3333 | 0.3750 | 1 | 29 | False |
| compact_cnn__l1_legacy64_w5_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4583 | 0.4628 | 0.3333 | 0.3750 | 1 | 29 | True |
| compact_cnn__l1_legacy64_w5_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4028 | 0.4087 | 0.3333 | 0.3529 | 1 | 29 | False |
| compact_cnn__l2_legacy400_w5_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5324 | 0.5404 | 0.3750 | 0.4615 | 1 | 29 | False |
| compact_cnn__l2_legacy400_w5_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5324 | 0.5404 | 0.3750 | 0.4615 | 1 | 29 | True |
| compact_cnn__l2_legacy400_w5_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4537 | 0.4497 | 0.2500 | 0.3333 | 1 | 29 | False |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.3981 | 0.4084 | 0.2500 | 0.3636 | 1 | 29 | False |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | line_a_equal_files | declared_training_aggregation | 0.3981 | 0.4084 | 0.2500 | 0.3636 | 1 | 29 | True |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.3704 | 0.3684 | 0.2500 | 0.3077 | 1 | 29 | False |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.2917 | 0.2840 | 0.1250 | 0.1818 | 1 | 29 | False |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | line_a_equal_files | declared_training_aggregation | 0.2917 | 0.2840 | 0.1250 | 0.1818 | 1 | 29 | True |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4028 | 0.3722 | 0.1250 | 0.2000 | 1 | 29 | False |
| compact_cnn__l5_uniform_replacement_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.3796 | 0.3681 | 0.2222 | 0.2667 | 1 | 29 | False |
| compact_cnn__l5_uniform_replacement_fixed10 | line_a_equal_files | declared_training_aggregation | 0.3796 | 0.3681 | 0.2222 | 0.2667 | 1 | 29 | True |
| compact_cnn__l5_uniform_replacement_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4583 | 0.4593 | 0.3333 | 0.3529 | 1 | 29 | False |
| compact_cnn__l6_v2_line_b_balance_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4954 | 0.4879 | 0.4167 | 0.4211 | 1 | 29 | False |
| compact_cnn__l6_v2_line_b_balance_fixed10 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.4954 | 0.4879 | 0.4167 | 0.4211 | 1 | 29 | False |
| compact_cnn__l6_v2_line_b_balance_fixed10 | line_b_equal_role_families | declared_training_aggregation | 0.4954 | 0.4936 | 0.4167 | 0.4211 | 1 | 29 | True |
| compact_cnn__l7_v2_training_bundle_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.3750 | 0.3815 | 0.3333 | 0.3158 | 1 | 29 | False |
| compact_cnn__l7_v2_training_bundle_fixed10 | line_a_equal_files | posthoc_same_oof_sensitivity_only | 0.3750 | 0.3815 | 0.3333 | 0.3158 | 1 | 29 | False |
| compact_cnn__l7_v2_training_bundle_fixed10 | line_b_equal_role_families | declared_training_aggregation | 0.5000 | 0.4850 | 0.3333 | 0.4000 | 1 | 29 | True |
| inception_full__l0_legacy64_w15_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5787 | 0.5801 | 0.4444 | 0.5000 | 1 | 29 | False |
| inception_full__l0_legacy64_w15_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5787 | 0.5801 | 0.4444 | 0.5000 | 1 | 29 | True |
| inception_full__l0_legacy64_w15_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5787 | 0.5925 | 0.4444 | 0.4706 | 1 | 29 | False |

<details><summary>Hierarchy coverage: B/R1–R4 window/file views and B/R role-balanced view</summary>

| Case | Repeat | Level | View | Group | OOF units | Retained units | Dropped units | Retained coverage | All participants | Retained participants | Dropped participants |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | ALL | 12706 | 12706 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l1_legacy64_w5_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l2_legacy400_w5_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l5_uniform_replacement_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | window | window_balanced_to_participant | ALL | 17333 | 17333 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | window | window_balanced_to_participant | B | 3479 | 3479 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3447 | 3447 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3452 | 3452 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3472 | 3472 | 0 | 1.0000 | 29 | 29 | 0 |
| compact_cnn__l7_v2_training_bundle_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3483 | 3483 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | ALL | 145 | 145 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | participant | participant_balanced_endpoint | ALL | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | role | line_b_equal_role_families | ALL | 58 | 58 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | ALL | 12706 | 12706 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 0 | 1.0000 | 29 | 29 | 0 |
| inception_full__l0_legacy64_w15_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 0 | 1.0000 | 29 | 29 | 0 |

</details>

## Worst-class F1 stability review

This secondary view reorders complete cases by abstention-aware worst-class F1, then abstention-aware repeat variability. Conditional retained-only values remain visible for comparison.

| Stability rank | Aware-BA rank | Case | Aware worst F1 | Aware worst recall | Aware BA, mean ± SD (%) | Worst F1 | Worst recall | Conditional BA, mean ± SD (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | inception_full__l0_legacy64_w15_fixed10 | 0.5000 | 0.4444 | 57.9 ± 0.0 | 0.5000 | 0.4444 | 57.9 ± 0.0 |
| 2 | 2 | compact_cnn__l2_legacy400_w5_fixed10 | 0.4615 | 0.3750 | 53.2 ± 0.0 | 0.4615 | 0.3750 | 53.2 ± 0.0 |
| 3 | 4 | compact_cnn__l6_v2_line_b_balance_fixed10 | 0.4211 | 0.4167 | 49.5 ± 0.0 | 0.4211 | 0.4167 | 49.5 ± 0.0 |
| 4 | 3 | compact_cnn__l7_v2_training_bundle_fixed10 | 0.4000 | 0.3333 | 50.0 ± 0.0 | 0.4000 | 0.3333 | 50.0 ± 0.0 |
| 5 | 5 | compact_cnn__l1_legacy64_w5_fixed10 | 0.3750 | 0.3333 | 45.8 ± 0.0 | 0.3750 | 0.3333 | 45.8 ± 0.0 |
| 6 | 7 | compact_cnn__l3_v2_imu_window_scaled_fixed10 | 0.3636 | 0.2500 | 39.8 ± 0.0 | 0.3636 | 0.2500 | 39.8 ± 0.0 |
| 7 | 6 | compact_cnn__l0_legacy64_w15_fixed10 | 0.3636 | 0.3333 | 42.6 ± 0.0 | 0.3636 | 0.3333 | 42.6 ± 0.0 |
| 8 | 8 | compact_cnn__l5_uniform_replacement_fixed10 | 0.2667 | 0.2222 | 38.0 ± 0.0 | 0.2667 | 0.2222 | 38.0 ± 0.0 |
| 9 | 9 | compact_cnn__l4_v2_imu_fold_scaled_fixed10 | 0.1818 | 0.1250 | 29.2 ± 0.0 | 0.1818 | 0.1250 | 29.2 ± 0.0 |

## Incomplete cases excluded from ranking

N/A — no rows were available.

## Deployment measurements (separate from predictive ranking)

| Case | Parameters | Inference cost | Status | Reported note |
|---|---|---|---|---|
| inception_full__l0_legacy64_w15_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l7_v2_training_bundle_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l5_uniform_replacement_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l6_v2_line_b_balance_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l2_legacy400_w5_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l1_legacy64_w5_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__l0_legacy64_w15_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |

## Route × role coverage and feature availability

This table separates direct and processed rate paths, retained coverage, unavailable predictors, and reducer failures for each role/route state.

| Case | Role | Quality tier | Motion | Route state | Signal route | Files | Retained files | Dropped files | Retained coverage | Abstention | Abstention reasons | Direct | Processed | Unavailable predictors | Denoiser attempts | Denoiser successes | Reducer failures | Reducer failure rate | Post Q_rate pass rate | Recovery eligible | Q_rate recovered | Q_rate recovery rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | B | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R1 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R2 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R3 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R4 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | B | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R1 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R2 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R3 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R4 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | B | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R1 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R2 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R3 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R4 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | B | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R1 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R2 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R3 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R4 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | B | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R1 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R2 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R3 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R4 | not_reported | not_reported | reference_direct_quality_nonrouting | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | not_reported | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |

## SQI state, score, and coverage provenance by each route

Direct and post-denoiser coverage are reported separately so the configured minimum-coverage decision remains auditable.

| Case | Role | Tier | Direct Q_rate state | Mean direct Q_rate | Direct Q_rate coverage | Direct Q_morph state | Mean direct Q_morph | Direct Q_morph coverage | Post Q_rate state | Mean post Q_rate | Post Q_rate coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l0_legacy64_w15_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l1_legacy64_w5_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l2_legacy400_w5_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l5_uniform_replacement_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__l7_v2_training_bundle_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__l0_legacy64_w15_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__l0_legacy64_w15_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |

## Denoiser paired HR/PPI endpoint audit

HR is calculated as `60 / median(valid PPI seconds)` from the same registered peak detector before and after the single denoiser attempt. Rows are paired within recording and averaged within participant before the participant-macro summary. Use the `outer_oof` rows for the primary held-out comparison; outer-train rows remain audit-only. HR/PPI endpoint error here is absolute post-denoise minus same-record direct-PPG change; Frailty29 has no ECG reference, so it is not physiological accuracy.

N/A — no rows were available.

<details><summary>Per-record paired denoiser HR evidence</summary>

N/A — no rows were available.

</details>

## Frozen motion evidence used by each route

Frailty29 reuse is in-sample auxiliary motion-preprocessing evidence, not valid outer-OOF motion-detector evidence. The downstream frailty classification outcome is still evaluated on each outer held-out fold.

| Case | Role | Tier | Motion | Mean p(motion) | Threshold | Mean windows | Evidence SHA-256 | Model SHA-256 | Training scope | Frailty29 relation | Valid outer-OOF motion evidence | Denoiser | Denoiser status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__l0_legacy64_w15_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l0_legacy64_w15_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l0_legacy64_w15_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l0_legacy64_w15_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l0_legacy64_w15_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l1_legacy64_w5_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l1_legacy64_w5_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l1_legacy64_w5_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l1_legacy64_w5_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l1_legacy64_w5_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l2_legacy400_w5_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l2_legacy400_w5_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l2_legacy400_w5_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l2_legacy400_w5_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l2_legacy400_w5_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l5_uniform_replacement_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l5_uniform_replacement_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l5_uniform_replacement_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l5_uniform_replacement_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l5_uniform_replacement_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l6_v2_line_b_balance_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l6_v2_line_b_balance_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l7_v2_training_bundle_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l7_v2_training_bundle_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l7_v2_training_bundle_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l7_v2_training_bundle_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__l7_v2_training_bundle_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__l0_legacy64_w15_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__l0_legacy64_w15_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__l0_legacy64_w15_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__l0_legacy64_w15_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__l0_legacy64_w15_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |

## Quality-component distributions by route and role

N/A — no rows were available.

## Classification score, t-SNE, and ROC–AUC diagnostics

Every classifier with persisted participant OOF probabilities is represented in the three paired figure modules. t-SNE embeds the prediction-probability vector, not hidden features. Multiclass frailty decisions use argmax and therefore have no single scalar threshold.

| Classifier | Score/threshold | Prediction t-SNE | ROC–AUC curve | OOF rows | t-SNE points | ROC points |
|---|---|---|---|---|---|---|
| inception_full__l0_legacy64_w15_fixed10 | available | available | available | 29 | 29 | 245 |
| compact_cnn__l7_v2_training_bundle_fixed10 | available | available | available | 29 | 29 | 241 |
| compact_cnn__l5_uniform_replacement_fixed10 | available | available | available | 29 | 29 | 240 |
| compact_cnn__l6_v2_line_b_balance_fixed10 | available | available | available | 29 | 29 | 246 |
| compact_cnn__l4_v2_imu_fold_scaled_fixed10 | available | available | available | 29 | 29 | 243 |
| compact_cnn__l3_v2_imu_window_scaled_fixed10 | available | available | available | 29 | 29 | 251 |
| compact_cnn__l2_legacy400_w5_fixed10 | available | available | available | 29 | 29 | 240 |
| compact_cnn__l1_legacy64_w5_fixed10 | available | available | available | 29 | 29 | 249 |
| compact_cnn__l0_legacy64_w15_fixed10 | available | available | available | 29 | 29 | 246 |

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
| paired_deltas | generated | figures/paired_deltas.png |  |
| ablation_sensitivity_metrics | generated | figures/ablation_sensitivity_metrics.png |  |
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
| legacy_bridge_numeric_ablation_report | generated | figures/legacy_bridge_numeric_ablation_report.png |  |
| legacy_bridge_execution_order_report | generated | figures/legacy_bridge_execution_order_report.png |  |

## Limitations and N/A items

- Classification t-SNE is a report-only embedding of persisted OOF prediction-probability vectors, not a hidden-feature embedding and not evidence of separability in the model representation space.
- Paired P values use the registered participant-cluster permutation budget and Holm correction separately within BA and Macro-F1; they do not trigger automatic selection.
- legacy bridge report A uses only the seven predefined numeric CompactCNN contrasts L0->L1 through L6->L7; report B lists absolute metrics in execution order and never interprets execution jumps, including L7->L5, as ablations
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

- [tables/legacy_bridge_numeric_ablation_report.csv](tables/legacy_bridge_numeric_ablation_report.csv)
- [tables/legacy_bridge_execution_order_report.csv](tables/legacy_bridge_execution_order_report.csv)
