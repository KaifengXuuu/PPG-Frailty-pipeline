# V2 study summary — staged_static_03_centered_star_v1

> This report is descriptive evidence for manual review. It does not automatically select a final use case or winner.

## Scientific context

- Study kind: catalog_sweep
- Purpose: Estimate seven single-factor B0-to-Bk effects independently for CompactCNN and InceptionTimeFull.
- Position in use-case selection flow: Restarted Stage 3 before ensemble, SQI/motion, finalist ablations, and stage-last ShapeFormer.
- Decision role: ablation
- Thesis sections: ["Centered legacy-to-V2 raw-model ablation", "Architecture-specific paired-fold sensitivity"]
- Catalog: configs/formal_experiment_catalog_v2.yaml (scope=selected_ordinary, balance=line_b)
- Reference case: compact_cnn__b0_star_fixed10

## Run controls and completeness

- Repeats requested: [0, 1, 2, 3, 4]
- Folds requested: [0, 1, 2, 3, 4]
- Case-level jobs requested: 1
- Effective jobs: 1
- Planned / passed / failed / not-run cases: 16 / 16 / 0 / 0
- Planned / reported / passed / failed / not-run cells: 400 / 400 / 400 / 0 / 0
- Resume-skipped passed cases: 0

## Test models, modules, inputs, and fixed parameters

The identical standalone table is in [TEST_COMPONENTS.md](TEST_COMPONENTS.md); machine-readable copies are `tables/test_components.csv` and `.json`. Input data are reported as dataset/path, signal view, channels, units, rate, and windows—not hashes.

| Cases / phases | Component role | Model / module | State | Input data (values and paths; no hashes) | Detailed fixed parameters | Algorithm and kernel (≤300 chars) |
|---|---|---|---|---|---|---|
| compact_cnn__b7_star_fixed10; inception_full__b7_star_fixed10 | aggregation | line_b_equal_role_families | enabled | {"input_data":"held-out window/file probabilities","roles":["B","R1","R2","R3","R4"]} | {"primary_report_aggregation_view":"line_b_equal_role_families","training_metric_aggregation_rule":"line_a_equal_files"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b1_star_fixed10; compact_cnn__b2_star_fixed10; compact_cnn__b3_star_fixed10; compact_cnn__b4_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; inception_full__b0_star_fixed10; inception_full__b1_star_fixed10; inception_full__b2_star_fixed10; inception_full__b3_star_fixed10; inception_full__b4_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10 | aggregation | window_balanced_to_participant | enabled | {"input_data":"held-out window/file probabilities","roles":["B","R1","R2","R3","R4"]} | {"primary_report_aggregation_view":"window_balanced_to_participant","training_metric_aggregation_rule":"line_a_equal_files"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10 | classifier | CompactCNN1D | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"classifier_dropout":0.2,"dilations":[1,1,1],"global_pooling":"adaptive_average_1","kernel_sizes":[9,9,7],"model_id":"compact_cnn","n_classes":3,"pool_sizes":[4,4],"representation_mode":"raw","stage_channels":[32,64,128],"stage_dropouts":[0.1,0.15]},"dilations":[1,1,1],"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[9,9,7],"mask_aware_pooling":true,"model_id":"CompactCNN1D","n_classes":3,"pool_sizes":[4,4],"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"reference_not_wang_fcn"} |  |
| compact_cnn__b1_star_fixed10 | classifier | CompactCNN1D | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"classifier_dropout":0.2,"dilations":[1,1,1],"global_pooling":"adaptive_average_1","kernel_sizes":[9,9,7],"model_id":"compact_cnn","n_classes":3,"pool_sizes":[4,4],"representation_mode":"raw","stage_channels":[32,64,128],"stage_dropouts":[0.1,0.15]},"dilations":[1,1,1],"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[9,9,7],"mask_aware_pooling":true,"model_id":"CompactCNN1D","n_classes":3,"pool_sizes":[4,4],"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"reference_not_wang_fcn"} |  |
| compact_cnn__b2_star_fixed10 | classifier | CompactCNN1D | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}} | {"architecture_parameters":{"classifier_dropout":0.2,"dilations":[1,1,1],"global_pooling":"adaptive_average_1","kernel_sizes":[9,9,7],"model_id":"compact_cnn","n_classes":3,"pool_sizes":[4,4],"representation_mode":"raw","stage_channels":[32,64,128],"stage_dropouts":[0.1,0.15]},"dilations":[1,1,1],"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[9,9,7],"mask_aware_pooling":true,"model_id":"CompactCNN1D","n_classes":3,"pool_sizes":[4,4],"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"reference_not_wang_fcn"} |  |
| compact_cnn__b3_star_fixed10 | classifier | CompactCNN1D | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"classifier_dropout":0.2,"dilations":[1,1,1],"global_pooling":"adaptive_average_1","kernel_sizes":[9,9,7],"model_id":"compact_cnn","n_classes":3,"pool_sizes":[4,4],"representation_mode":"raw","stage_channels":[32,64,128],"stage_dropouts":[0.1,0.15]},"dilations":[1,1,1],"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[9,9,7],"mask_aware_pooling":true,"model_id":"CompactCNN1D","n_classes":3,"pool_sizes":[4,4],"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"reference_not_wang_fcn"} |  |
| compact_cnn__b4_star_fixed10 | classifier | CompactCNN1D | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"classifier_dropout":0.2,"dilations":[1,1,1],"global_pooling":"adaptive_average_1","kernel_sizes":[9,9,7],"model_id":"compact_cnn","n_classes":3,"pool_sizes":[4,4],"representation_mode":"raw","stage_channels":[32,64,128],"stage_dropouts":[0.1,0.15]},"dilations":[1,1,1],"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[9,9,7],"mask_aware_pooling":true,"model_id":"CompactCNN1D","n_classes":3,"pool_sizes":[4,4],"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"reference_not_wang_fcn"} |  |
| inception_full__b0_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} |  |
| inception_full__b1_star_fixed10 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} |  |
| inception_full__b2_star_fixed10 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} |  |
| inception_full__b3_star_fixed10 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} |  |
| inception_full__b4_star_fixed10 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b1_star_fixed10; compact_cnn__b2_star_fixed10; compact_cnn__b3_star_fixed10; compact_cnn__b4_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b1_star_fixed10; inception_full__b2_star_fixed10; inception_full__b3_star_fixed10; inception_full__b4_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | dataset_adapter | frailty3_m2_20260815_a054800abda272f6 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"allow_qc_excluded_records":false,"channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"class_id_order":[0,1,2],"class_name_order":["Pre-Frail","Robust/Non-Frail","Young"],"expected_participant_count":29,"expected_record_count":261,"manifest_version":"internal_records_v2","path":"manifests/internal_records_v2.csv","source_dataset_id":"frailty3_m2_20260815_a054800abda272f6"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b1_star_fixed10; compact_cnn__b2_star_fixed10; compact_cnn__b3_star_fixed10; compact_cnn__b4_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b1_star_fixed10; inception_full__b2_star_fixed10; inception_full__b3_star_fixed10; inception_full__b4_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | evaluation | balanced_accuracy | enabled | {"class_order":["Pre-Frail","Robust/Non-Frail","Young"],"input_data":"held-out participant predictions and frailty labels"} | {"calibration_metrics":["multiclass_brier","expected_calibration_error"],"confidence_interval":"participant_cluster_bootstrap_two_sided_95","independent_test_available":false,"metric_prefix":"oof_validation_","metrics":["balanced_accuracy","macro_f1","per_class_precision_recall_f1","worst_class_recall","worst_class_f1","confusion_matrix","coverage"],"paired_delta_key":["repeat_index","fold_index","participant_id"],"primary_metric":"balanced_accuracy","rank_incomplete_configs":false,"ranking":{"automatic_final_selection":false,"manual_multiple_final_versions_allowed":true,"max_qualified_per_comparison_group":10,"preserve_ablation_provenance":true,"sort_key":"participant_level_mean_balanced_accuracy"},"statistics":{"affects_automatic_selection":false,"bootstrap_replicates":10000,"cluster_unit":"participant_with_all_five_repeat_oof_predictions","confidence_interval":"two_sided_95_percentile","lcb95_metrics":["participant_level_mean_balanced_accuracy","participant_level_mean_macro_f1"],"lcb95_percentile":2.5,"multiplicity_correction":"holm_within_comparison_family","paired_exchange_unit":"participant","paired_permutation_replicates":100000,"seed":42},"unit":"participant"} |  |
| compact_cnn__b3_star_fixed10; inception_full__b3_star_fixed10 | imu_preprocessing | calibrated_ekf_adyn | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["AX","AY","AZ","GX","GY","GZ"],"manifest_path":"manifests/internal_records_v2.csv","output_channels":["A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"method":"calibrated_ekf_adyn"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b1_star_fixed10; compact_cnn__b2_star_fixed10; compact_cnn__b4_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b1_star_fixed10; inception_full__b2_star_fixed10; inception_full__b4_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | imu_preprocessing | legacy_filtered_axes | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["AX","AY","AZ","GX","GY","GZ"],"manifest_path":"manifests/internal_records_v2.csv","output_channels":["AX","AY","AZ","GX","GY","GZ"],"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"method":"legacy_filtered_axes"} |  |
| compact_cnn__b0_star_fixed10; inception_full__b0_star_fixed10 | legacy_bridge_effective_profile | B0 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":[],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"baseline","interpretation":"complete legacy baseline"} |  |
| compact_cnn__b1_star_fixed10; inception_full__b1_star_fixed10 | legacy_bridge_effective_profile | B1 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":["controls.target_fs_hz"],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":400,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"sampling_rate","interpretation":"Target sample rate only: 64 to 400 Hz."} |  |
| compact_cnn__b2_star_fixed10; inception_full__b2_star_fixed10 | legacy_bridge_effective_profile | B2 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}} | {"changed_control_paths":["controls.window_seconds","controls.hop_seconds"],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":2.5,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":5.0},"factor_id":"window_plan","interpretation":"Window plan only: 15/3 to 5/2.5 seconds; retain fraction 0.9 and no cap."} |  |
| compact_cnn__b3_star_fixed10; inception_full__b3_star_fixed10 | legacy_bridge_effective_profile | B3 | executed_hash_bound_controls | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":["controls.imu_preprocessing"],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"calibrated_ekf_adyn","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"imu_preprocessing","interpretation":"IMU semantics only: legacy axes to calibrated EKF dynamic acceleration."} |  |
| compact_cnn__b4_star_fixed10; inception_full__b4_star_fixed10 | legacy_bridge_effective_profile | B4 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":["controls.normalization"],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"ppg_window_imu_outer_train_fold","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"normalization","interpretation":"Normalization only: all-eight per-window to PPG-window plus outer-train IMU-fold scaling."} |  |
| compact_cnn__b5_star_fixed10; inception_full__b5_star_fixed10 | legacy_bridge_effective_profile | B5 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":["controls.sampler"],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"balance_line_weighted_v2","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"sampler","interpretation":"Sampler only: exhaustive uniform shuffle to Line B weighted sampling."} |  |
| compact_cnn__b6_star_fixed10; inception_full__b6_star_fixed10 | legacy_bridge_effective_profile | B6 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":["controls.optimizer","controls.batch_size"],"controls":{"allow_short_record_padding":true,"batch_size":64,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adam","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"window_balanced_to_participant","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"optimizer_and_batch_size","interpretation":"Declared training bundle only: AdamW/batch32 to Adam/batch64."} |  |
| compact_cnn__b7_star_fixed10; inception_full__b7_star_fixed10 | legacy_bridge_effective_profile | B7 | executed_hash_bound_controls | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"changed_control_paths":["controls.primary_report_aggregation_view"],"controls":{"allow_short_record_padding":true,"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","fixed_epochs":10,"historical_retained_fraction":0.9,"hop_seconds":3.0,"imu_preprocessing":"legacy_filtered_axes","learning_rate":0.001,"max_windows_per_file":null,"normalization":"per_window_all_eight","optimizer":"adamw","ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","primary_report_aggregation_view":"line_b_equal_role_families","sampler":"exhaustive_shuffle_without_replacement","target_fs_hz":64,"training_metric_aggregation_rule":"line_a_equal_files","weight_decay":0.0001,"window_seconds":15.0},"factor_id":"primary_aggregation","interpretation":"Native primary reporting only: direct-window participant mean to file/role participant mean."} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b1_star_fixed10; compact_cnn__b2_star_fixed10; compact_cnn__b3_star_fixed10; compact_cnn__b4_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b1_star_fixed10; inception_full__b2_star_fixed10; inception_full__b3_star_fixed10; inception_full__b4_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | ppg_preprocessing | legacy_detrend_bandpass_0p2_8 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["RED","IR"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"method":"legacy_detrend_bandpass_0p2_8"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | representation | raw | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"input_contract":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"representation_mode":"raw"} |  |
| compact_cnn__b1_star_fixed10; inception_full__b1_star_fixed10 | representation | raw | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"input_contract":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"representation_mode":"raw"} |  |
| compact_cnn__b2_star_fixed10; inception_full__b2_star_fixed10 | representation | raw | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}} | {"input_contract":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}},"representation_mode":"raw"} |  |
| compact_cnn__b3_star_fixed10; inception_full__b3_star_fixed10 | representation | raw | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"input_contract":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"representation_mode":"raw"} |  |
| compact_cnn__b4_star_fixed10; inception_full__b4_star_fixed10 | representation | raw | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"input_contract":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"representation_mode":"raw"} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | signal_views_and_scaling | per_window_all_eight | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"normalization":"per_window_all_eight","sampling_rate_hz":64} |  |
| compact_cnn__b1_star_fixed10; inception_full__b1_star_fixed10 | signal_views_and_scaling | per_window_all_eight | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"normalization":"per_window_all_eight","sampling_rate_hz":400} |  |
| compact_cnn__b2_star_fixed10; inception_full__b2_star_fixed10 | signal_views_and_scaling | per_window_all_eight | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}} | {"normalization":"per_window_all_eight","sampling_rate_hz":64} |  |
| compact_cnn__b3_star_fixed10; inception_full__b3_star_fixed10 | signal_views_and_scaling | per_window_all_eight | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"normalization":"per_window_all_eight","sampling_rate_hz":64} |  |
| compact_cnn__b4_star_fixed10; inception_full__b4_star_fixed10 | signal_views_and_scaling | ppg_window_imu_outer_train_fold | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"normalization":"ppg_window_imu_outer_train_fold","sampling_rate_hz":64} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b1_star_fixed10; compact_cnn__b2_star_fixed10; compact_cnn__b3_star_fixed10; compact_cnn__b4_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b1_star_fixed10; inception_full__b2_star_fixed10; inception_full__b3_star_fixed10; inception_full__b4_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | split_registry | frailty3_future_corrected_sgkf5_v2 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","groups":"participant_id","labels":"frailty_class","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"n_repeats":5,"n_splits":5,"path":"splits/sgkf5_repeated_grouped_5x5_v2.csv","registry_id":"frailty3_future_corrected_sgkf5_v2","runtime_recompute":false,"split_seeds":[42,10042,20042,30042,40042]} |  |
| compact_cnn__b6_star_fixed10; inception_full__b6_star_fixed10 | trainer | adam | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":64,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adam","outer_labels_visible_to_trainer":false,"sampler":"exhaustive_shuffle_without_replacement","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b7_star_fixed10 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adamw","outer_labels_visible_to_trainer":false,"sampler":"exhaustive_shuffle_without_replacement","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b1_star_fixed10; inception_full__b1_star_fixed10 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adamw","outer_labels_visible_to_trainer":false,"sampler":"exhaustive_shuffle_without_replacement","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b2_star_fixed10; inception_full__b2_star_fixed10 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adamw","outer_labels_visible_to_trainer":false,"sampler":"exhaustive_shuffle_without_replacement","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b3_star_fixed10; inception_full__b3_star_fixed10 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adamw","outer_labels_visible_to_trainer":false,"sampler":"exhaustive_shuffle_without_replacement","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b4_star_fixed10; inception_full__b4_star_fixed10 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adamw","outer_labels_visible_to_trainer":false,"sampler":"exhaustive_shuffle_without_replacement","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b5_star_fixed10; inception_full__b5_star_fixed10 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"class_weighting":"outer_train_window_inverse_frequency","device":"cuda","fixed_epochs":10,"learning_rate":0.001,"optimizer":"adamw","outer_labels_visible_to_trainer":false,"sampler":"balance_line_weighted_v2","training_metric_aggregation_rule":"line_a_equal_files","training_seed":42,"weight_decay":0.0001} |  |
| compact_cnn__b0_star_fixed10; compact_cnn__b5_star_fixed10; compact_cnn__b6_star_fixed10; compact_cnn__b7_star_fixed10; inception_full__b0_star_fixed10; inception_full__b5_star_fixed10; inception_full__b6_star_fixed10; inception_full__b7_star_fixed10 | window_planner | legacy_bridge_reviewed_window_plan_v1 | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0} |  |
| compact_cnn__b1_star_fixed10; inception_full__b1_star_fixed10 | window_planner | legacy_bridge_reviewed_window_plan_v1 | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":400,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0} |  |
| compact_cnn__b2_star_fixed10; inception_full__b2_star_fixed10 | window_planner | legacy_bridge_reviewed_window_plan_v1 | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0}} | {"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":2.5,"length_s":5.0} |  |
| compact_cnn__b3_star_fixed10; inception_full__b3_star_fixed10 | window_planner | legacy_bridge_reviewed_window_plan_v1 | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"calibrated_ekf_adyn","manifest_path":"manifests/internal_records_v2.csv","normalization":"per_window_all_eight","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0} |  |
| compact_cnn__b4_star_fixed10; inception_full__b4_star_fixed10 | window_planner | legacy_bridge_reviewed_window_plan_v1 | enabled | {"channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","imu_preprocessing":"legacy_filtered_axes","manifest_path":"manifests/internal_records_v2.csv","normalization":"ppg_window_imu_outer_train_fold","participants":29,"pipeline_fs_hz":400.0,"ppg_preprocessing":"legacy_detrend_bandpass_0p2_8","records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"sampling_rate_hz":64,"signal_view":"legacy_bridge_effective_DL_tensor","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0}} | {"allow_short_record_padding":true,"cap_per_file":null,"historical_retained_fraction":0.9,"hop_s":3.0,"length_s":15.0} |  |

## Seed and data-split reproducibility

- Audit status: **PASS**
- Scope: manifest_cases_and_selected_artifact_roots_only
- Planned / observed selected cells: 400 / 400
- Split seeds by repeat: {"0": [42], "1": [10042], "2": [20042], "3": [30042], "4": [40042]}
- Errors / not-verifiable items: 0 / 0
- This is report-only evidence; it never gates training or report generation.

| Case | Selected status | Selected attempt | Excluded attempts | Planned cells | Observed cells | Declared seed policy | Effective seed policy | Split seeds | Model seeds | Orchestration seeds | Evaluation seeds | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | passed | 2 | ["attempt_001"] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b0_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b1_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b1_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b2_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b2_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b3_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b3_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b4_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b4_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b5_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b5_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b6_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b6_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| compact_cnn__b7_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |
| inception_full__b7_star_fixed10 | passed | 1 | [] | 25 | 25 | ["outer_cv_repeat_seed_equals_split_seed"] | ["legacy_bridge_fixed_training_seed_42"] | [42, 10042, 20042, 30042, 40042] | [42] | [42] | [42] | PASS |

<details><summary>Per-cell seed and split evidence</summary>

| Case | Repeat | Fold | Cell status | Attempt | Declared policy | Effective policy | Split seed | Orchestration seed | Training seed | Model/member seeds | Member-seed semantics | Evaluation seed | Epoch RNG rows | Split CSV SHA256 | Fold membership SHA256 | Train participants | OOF participants | Train/OOF overlap | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | 0 | 0 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 0 | 1 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 0 | 2 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 0 | 3 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 0 | 4 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 1 | 0 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 1 | 1 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 1 | 2 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 1 | 3 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 1 | 4 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 2 | 0 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 2 | 1 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 2 | 2 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 2 | 3 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 2 | 4 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 3 | 0 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 3 | 1 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 3 | 2 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 3 | 3 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 3 | 4 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 4 | 0 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 4 | 1 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 4 | 2 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 4 | 3 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b0_star_fixed10 | 4 | 4 | passed | 2 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b0_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b0_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b0_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b0_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b0_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b0_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b0_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b0_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b1_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b1_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b1_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b1_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b1_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b1_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b1_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b1_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b1_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b2_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b2_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b2_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b2_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b2_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b2_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b2_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b2_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b2_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b3_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b3_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b3_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b3_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b3_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b3_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b3_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b3_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b3_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b4_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b4_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b4_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b4_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b4_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b4_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b4_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b4_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b4_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b5_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b5_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b5_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b5_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b5_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b5_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b5_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b5_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b5_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b6_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b6_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b6_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b6_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b6_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b6_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b6_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b6_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b6_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| compact_cnn__b7_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 0 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 74a9abe341803e19561e3f8325621de387e0a6f3cc76e79a468dca12d814052a | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 0 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 57bb106ec2ff5d3c168e33998b4a7936c39cbbb1e996b929652dd6e1a36da8e8 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 0 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dc46fdbd11329275083d417b2d5d720396b21cbc072994433e7dc7a6b06a666d | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 0 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0f885f271eff150fced6d2c9acdddbf0fcdf8f994d2e0bed36169eee28c2bc90 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 0 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 42 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d94e3292b845a0950210e2e3294bf00e43a1f33c65c16b35be9d37b8362422ad | 24 | 5 | 0 | PASS |
| inception_full__b7_star_fixed10 | 1 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 91233bf0f9ab6ae43b78c5014b5d48d26b6d091b34f83211383bcaeef1e148e7 | 22 | 7 | 0 | PASS |
| inception_full__b7_star_fixed10 | 1 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 39abfdc29e01293bf55542e427ffc608689093236594d1fc0ae1b3b2694dd544 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 1 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 49933d7fb3f72995803ccefc8aedbfdfe0cbff92366fbcb3f76f4f4cf6946e56 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 1 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c842847d07fa69f86821292aded7d0e3377607446cdce92996ab0445430ef39d | 24 | 5 | 0 | PASS |
| inception_full__b7_star_fixed10 | 1 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 10042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | fbcbb0d4b376f34633ea4c4f3db887e665f87ea21ed8e7dcd0931ec726aad916 | 24 | 5 | 0 | PASS |
| inception_full__b7_star_fixed10 | 2 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 317be2c61e13790a922ca40e02831c8730373692242fb7adf3f8ad0fda5999f9 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 2 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0a29df73e1697ec9e031f16297ff66d2370ddf13b89c92492a983c713ff8a0d4 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 2 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 02ad1d4cbb85e5db5014b25cfcb1eb9c8db74cf4916412dc85cdfa1a0d1577a9 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 2 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c170d07d5d33c7991fa9056e84f431ba4c60a842416dfd667bb62acccb47f490 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 2 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 20042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | d8dc10a2e86af78b2813932ab8667b729a7bb77054b0fd256eddc433327ca922 | 24 | 5 | 0 | PASS |
| inception_full__b7_star_fixed10 | 3 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 0c98ab483932ce51b926081d2b42799ed4ca42eb0eb153a6cfd22cd68f8f21e7 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 3 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | cdbdb34a24464d19e1f01ec7d83fb193cb6bff99a4220999c5dfefd88a36c4b8 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 3 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | dd5e378526a22ad6eaa842da6f4b5abcd61f44517e2cf695265849c4d1d1c9eb | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 3 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 54ba5fd44ff3da1ed8c0688cb59795298fcbd9be0dfe2b1c6fd61036f5b8e458 | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 3 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 30042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 25406defed8613b4981edbb84837216322d1d5dd61051bf1b9306feb63c934c8 | 24 | 5 | 0 | PASS |
| inception_full__b7_star_fixed10 | 4 | 0 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 3e2e68f893d4f2e6a8be3f8273e21520ff9372870fc6efbf7e1a8d04feb8155a | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 4 | 1 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 4a77d21ac5aff1d8b65abb41db29a534dd0f53c6c3f6629e524b4f7c141bf9da | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 4 | 2 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | ffe61a7e1c5a563e266fe3ad9c711d059f85117e9e0000e902db944c49d7de6e | 23 | 6 | 0 | PASS |
| inception_full__b7_star_fixed10 | 4 | 3 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 24cec4022da4da297401a99fc04d00df08feba450ecd5e45066d768076789daf | 24 | 5 | 0 | PASS |
| inception_full__b7_star_fixed10 | 4 | 4 | passed | 1 | outer_cv_repeat_seed_equals_split_seed | legacy_bridge_fixed_training_seed_42 | 40042 | 42 | 42 | [42] | N/A_single_model_training_seed_alias | 42 | 10 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | 460dfbfca41f06353efb5a9f7a36be56046872c88b8390e49c5176864eb56316 | 23 | 6 | 0 | PASS |

</details>

### Frozen split roster

| Repeat | Fold | Split seed | Split CSV SHA256 | Declared authority JSON SHA256 | Declared authority payload SHA256 | Train participants | OOF participants | Overlap | Matching cases | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 0 | 1 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 0 | 2 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 0 | 3 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 0 | 4 | 42 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 1 | 0 | 10042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 22 | 7 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 1 | 1 | 10042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 1 | 2 | 10042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 1 | 3 | 10042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 1 | 4 | 10042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 2 | 0 | 20042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 2 | 1 | 20042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 2 | 2 | 20042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 2 | 3 | 20042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 2 | 4 | 20042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 3 | 0 | 30042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 3 | 1 | 30042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 3 | 2 | 30042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 3 | 3 | 30042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 3 | 4 | 30042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 4 | 0 | 40042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 4 | 1 | 40042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 4 | 2 | 40042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 4 | 3 | 40042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 24 | 5 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |
| 4 | 4 | 40042 | 1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702 | c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c | 0bca827fa8e4f9679b6ee9435b5497a60d91e74ddb366e058fc69614fb007f46 | 23 | 6 | 0 | ["compact_cnn__b0_star_fixed10", "compact_cnn__b1_star_fixed10", "compact_cnn__b2_star_fixed10", "compact_cnn__b3_star_fixed10", "compact_cnn__b4_star_fixed10", "compact_cnn__b5_star_fixed10", "compact_cnn__b6_star_fixed10", "compact_cnn__b7_star_fixed10", "inception_full__b0_star_fixed10", "inception_full__b1_star_fixed10", "inception_full__b2_star_fixed10", "inception_full__b3_star_fixed10", "inception_full__b4_star_fixed10", "inception_full__b5_star_fixed10", "inception_full__b6_star_fixed10", "inception_full__b7_star_fixed10"] | PASS |

## Varied and controlled parameters

- Explicit deterministic sparse catalog profiles; this is a screening comparison, not a single-factor causal ablation.
- Search method: deterministic_sparse_profiles
- Runtime parameter sampling: False
- Profile-design seed: 42
- Interpretation: B0 is the centre. Every B1-B7 case changes only its declared factor from B0; no adjacent Bk-to-B(k+1) or cross-model contrast is permitted.

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
| signal.imu.gravity_mps2 | 9.81 |
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
| 1 | compact_cnn__b2_star_fixed10 | 56.9 ± 5.9 | 0.6104 | 0.5685 | 57.2 ± 6.9 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 56.9 ± 5.9 | 57.2 ± 6.9 | 74.7 ± 4.4 | 66.1 ± 5.6 | 0.5120 | 0.5057 | 0.1667 | 0.1667 | 0.5500 | 0.5376 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 2 | inception_full__b0_star_fixed10 | 56.8 ± 2.4 | 0.5865 | 0.5676 | 57.2 ± 2.5 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 56.8 ± 2.4 | 57.2 ± 2.5 | 72.8 ± 2.5 | 64.6 ± 4.0 | 0.5447 | 0.5490 | 0.3333 | 0.3333 | 0.5000 | 0.4918 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 3 | inception_full__b7_star_fixed10 | 56.8 ± 2.4 | 0.5865 | 0.5676 | 57.2 ± 2.5 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 56.8 ± 2.4 | 57.2 ± 2.5 | 72.8 ± 2.5 | 64.6 ± 4.0 | 0.5447 | 0.5490 | 0.3333 | 0.3333 | 0.5000 | 0.4918 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 4 | inception_full__b6_star_fixed10 | 56.0 ± 3.8 | 0.5717 | 0.5602 | 56.2 ± 4.7 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 56.0 ± 3.8 | 56.2 ± 4.7 | 72.5 ± 4.0 | 63.6 ± 5.0 | 0.5239 | 0.5175 | 0.1667 | 0.1667 | 0.5000 | 0.4959 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 5 | compact_cnn__b0_star_fixed10 | 53.8 ± 6.8 | 0.5635 | 0.5380 | 52.9 ± 6.6 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 53.8 ± 6.8 | 52.9 ± 6.6 | 69.2 ± 4.3 | 61.0 ± 7.0 | 0.4731 | 0.4660 | 0.1667 | 0.1667 | 0.4667 | 0.4828 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 6 | compact_cnn__b7_star_fixed10 | 53.8 ± 6.8 | 0.5635 | 0.5380 | 52.9 ± 6.6 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 53.8 ± 6.8 | 52.9 ± 6.6 | 69.2 ± 4.3 | 61.0 ± 7.0 | 0.4731 | 0.4660 | 0.1667 | 0.1667 | 0.4667 | 0.4828 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 7 | compact_cnn__b5_star_fixed10 | 52.3 ± 3.5 | 0.5650 | 0.5231 | 53.4 ± 4.5 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 52.3 ± 3.5 | 53.4 ± 4.5 | 69.4 ± 3.1 | 60.3 ± 4.7 | 0.4894 | 0.4905 | 0.0000 | 0.0000 | 0.5111 | 0.4923 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 8 | inception_full__b5_star_fixed10 | 51.8 ± 3.9 | 0.5228 | 0.5176 | 51.5 ± 3.7 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 51.8 ± 3.9 | 51.5 ± 3.7 | 69.5 ± 2.7 | 61.3 ± 3.4 | 0.4802 | 0.4802 | 0.2778 | 0.2778 | 0.4167 | 0.4237 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 9 | inception_full__b2_star_fixed10 | 50.9 ± 5.2 | 0.5755 | 0.5093 | 51.8 ± 4.7 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 50.9 ± 5.2 | 51.8 ± 4.7 | 71.5 ± 2.9 | 64.2 ± 1.2 | 0.4596 | 0.4731 | 0.1667 | 0.1667 | 0.4444 | 0.4545 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 10 | inception_full__b1_star_fixed10 | 50.3 ± 2.3 | 0.5281 | 0.5028 | 50.5 ± 2.3 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 50.3 ± 2.3 | 50.5 ± 2.3 | 68.0 ± 6.6 | 61.5 ± 5.0 | 0.4812 | 0.4829 | 0.1667 | 0.1667 | 0.4250 | 0.4857 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 11 | inception_full__b3_star_fixed10 | 49.7 ± 5.9 | 0.5416 | 0.4972 | 50.3 ± 7.3 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 49.7 ± 5.9 | 50.3 ± 7.3 | 68.8 ± 1.4 | 60.7 ± 4.1 | 0.4411 | 0.4336 | 0.1667 | 0.1667 | 0.4750 | 0.4640 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 12 | compact_cnn__b6_star_fixed10 | 49.2 ± 6.9 | 0.5014 | 0.4917 | 48.7 ± 6.7 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 49.2 ± 6.9 | 48.7 ± 6.7 | 67.4 ± 3.7 | 58.3 ± 4.8 | 0.4262 | 0.4229 | 0.1667 | 0.1667 | 0.4000 | 0.4211 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 13 | compact_cnn__b3_star_fixed10 | 47.2 ± 9.4 | 0.5345 | 0.4722 | 47.6 ± 9.5 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 47.2 ± 9.4 | 47.6 ± 9.5 | 66.9 ± 4.0 | 57.7 ± 4.0 | 0.3827 | 0.3857 | 0.1667 | 0.1667 | 0.4000 | 0.4468 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 14 | compact_cnn__b1_star_fixed10 | 44.9 ± 9.3 | 0.4492 | 0.4491 | 43.9 ± 11.0 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 44.9 ± 9.3 | 43.9 ± 11.0 | 63.7 ± 2.9 | 54.2 ± 4.5 | 0.3604 | 0.3343 | 0.1667 | 0.1667 | 0.2750 | 0.3492 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 15 | compact_cnn__b4_star_fixed10 | 43.3 ± 3.7 | 0.4400 | 0.4333 | 42.8 ± 3.4 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 43.3 ± 3.7 | 42.8 ± 3.4 | 59.5 ± 6.3 | 51.8 ± 6.3 | 0.3985 | 0.3951 | 0.0000 | 0.0000 | 0.4000 | 0.4200 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 16 | inception_full__b4_star_fixed10 | 43.0 ± 7.3 | 0.4464 | 0.4296 | 43.1 ± 7.3 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 43.0 ± 7.3 | 43.1 ± 7.3 | 57.2 ± 6.3 | 47.7 ± 7.1 | 0.3597 | 0.3613 | 0.0000 | 0.0000 | 0.3500 | 0.3958 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |

## Repeat-level predictive distributions

Mean and sample SD are shown in one percentage column; lossless CI, range, mean, and SD values remain in the matching JSON table.

| Case | Metric | Repeats | Mean ± SD (%) | Source |
|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | balanced_accuracy | 5 | 53.8 ± 6.8 | participant_oof |
| compact_cnn__b0_star_fixed10 | macro_f1 | 5 | 52.9 ± 6.6 | participant_oof |
| compact_cnn__b0_star_fixed10 | macro_roc_auc_ovr | 5 | 69.2 ± 4.3 | participant_oof |
| compact_cnn__b0_star_fixed10 | macro_pr_auc_ovr | 5 | 61.0 ± 7.0 | participant_oof |
| compact_cnn__b0_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 53.8 ± 6.8 | participant_oof |
| compact_cnn__b0_star_fixed10 | abstention_aware_macro_f1 | 5 | 52.9 ± 6.6 | participant_oof |
| inception_full__b0_star_fixed10 | balanced_accuracy | 5 | 56.8 ± 2.4 | participant_oof |
| inception_full__b0_star_fixed10 | macro_f1 | 5 | 57.2 ± 2.5 | participant_oof |
| inception_full__b0_star_fixed10 | macro_roc_auc_ovr | 5 | 72.8 ± 2.5 | participant_oof |
| inception_full__b0_star_fixed10 | macro_pr_auc_ovr | 5 | 64.6 ± 4.0 | participant_oof |
| inception_full__b0_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 56.8 ± 2.4 | participant_oof |
| inception_full__b0_star_fixed10 | abstention_aware_macro_f1 | 5 | 57.2 ± 2.5 | participant_oof |
| compact_cnn__b1_star_fixed10 | balanced_accuracy | 5 | 44.9 ± 9.3 | participant_oof |
| compact_cnn__b1_star_fixed10 | macro_f1 | 5 | 43.9 ± 11.0 | participant_oof |
| compact_cnn__b1_star_fixed10 | macro_roc_auc_ovr | 5 | 63.7 ± 2.9 | participant_oof |
| compact_cnn__b1_star_fixed10 | macro_pr_auc_ovr | 5 | 54.2 ± 4.5 | participant_oof |
| compact_cnn__b1_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 44.9 ± 9.3 | participant_oof |
| compact_cnn__b1_star_fixed10 | abstention_aware_macro_f1 | 5 | 43.9 ± 11.0 | participant_oof |
| inception_full__b1_star_fixed10 | balanced_accuracy | 5 | 50.3 ± 2.3 | participant_oof |
| inception_full__b1_star_fixed10 | macro_f1 | 5 | 50.5 ± 2.3 | participant_oof |
| inception_full__b1_star_fixed10 | macro_roc_auc_ovr | 5 | 68.0 ± 6.6 | participant_oof |
| inception_full__b1_star_fixed10 | macro_pr_auc_ovr | 5 | 61.5 ± 5.0 | participant_oof |
| inception_full__b1_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 50.3 ± 2.3 | participant_oof |
| inception_full__b1_star_fixed10 | abstention_aware_macro_f1 | 5 | 50.5 ± 2.3 | participant_oof |
| compact_cnn__b2_star_fixed10 | balanced_accuracy | 5 | 56.9 ± 5.9 | participant_oof |
| compact_cnn__b2_star_fixed10 | macro_f1 | 5 | 57.2 ± 6.9 | participant_oof |
| compact_cnn__b2_star_fixed10 | macro_roc_auc_ovr | 5 | 74.7 ± 4.4 | participant_oof |
| compact_cnn__b2_star_fixed10 | macro_pr_auc_ovr | 5 | 66.1 ± 5.6 | participant_oof |
| compact_cnn__b2_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 56.9 ± 5.9 | participant_oof |
| compact_cnn__b2_star_fixed10 | abstention_aware_macro_f1 | 5 | 57.2 ± 6.9 | participant_oof |
| inception_full__b2_star_fixed10 | balanced_accuracy | 5 | 50.9 ± 5.2 | participant_oof |
| inception_full__b2_star_fixed10 | macro_f1 | 5 | 51.8 ± 4.7 | participant_oof |
| inception_full__b2_star_fixed10 | macro_roc_auc_ovr | 5 | 71.5 ± 2.9 | participant_oof |
| inception_full__b2_star_fixed10 | macro_pr_auc_ovr | 5 | 64.2 ± 1.2 | participant_oof |
| inception_full__b2_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 50.9 ± 5.2 | participant_oof |
| inception_full__b2_star_fixed10 | abstention_aware_macro_f1 | 5 | 51.8 ± 4.7 | participant_oof |
| compact_cnn__b3_star_fixed10 | balanced_accuracy | 5 | 47.2 ± 9.4 | participant_oof |
| compact_cnn__b3_star_fixed10 | macro_f1 | 5 | 47.6 ± 9.5 | participant_oof |
| compact_cnn__b3_star_fixed10 | macro_roc_auc_ovr | 5 | 66.9 ± 4.0 | participant_oof |
| compact_cnn__b3_star_fixed10 | macro_pr_auc_ovr | 5 | 57.7 ± 4.0 | participant_oof |
| compact_cnn__b3_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 47.2 ± 9.4 | participant_oof |
| compact_cnn__b3_star_fixed10 | abstention_aware_macro_f1 | 5 | 47.6 ± 9.5 | participant_oof |
| inception_full__b3_star_fixed10 | balanced_accuracy | 5 | 49.7 ± 5.9 | participant_oof |
| inception_full__b3_star_fixed10 | macro_f1 | 5 | 50.3 ± 7.3 | participant_oof |
| inception_full__b3_star_fixed10 | macro_roc_auc_ovr | 5 | 68.8 ± 1.4 | participant_oof |
| inception_full__b3_star_fixed10 | macro_pr_auc_ovr | 5 | 60.7 ± 4.1 | participant_oof |
| inception_full__b3_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 49.7 ± 5.9 | participant_oof |
| inception_full__b3_star_fixed10 | abstention_aware_macro_f1 | 5 | 50.3 ± 7.3 | participant_oof |
| compact_cnn__b4_star_fixed10 | balanced_accuracy | 5 | 43.3 ± 3.7 | participant_oof |
| compact_cnn__b4_star_fixed10 | macro_f1 | 5 | 42.8 ± 3.4 | participant_oof |
| compact_cnn__b4_star_fixed10 | macro_roc_auc_ovr | 5 | 59.5 ± 6.3 | participant_oof |
| compact_cnn__b4_star_fixed10 | macro_pr_auc_ovr | 5 | 51.8 ± 6.3 | participant_oof |
| compact_cnn__b4_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 43.3 ± 3.7 | participant_oof |
| compact_cnn__b4_star_fixed10 | abstention_aware_macro_f1 | 5 | 42.8 ± 3.4 | participant_oof |
| inception_full__b4_star_fixed10 | balanced_accuracy | 5 | 43.0 ± 7.3 | participant_oof |
| inception_full__b4_star_fixed10 | macro_f1 | 5 | 43.1 ± 7.3 | participant_oof |
| inception_full__b4_star_fixed10 | macro_roc_auc_ovr | 5 | 57.2 ± 6.3 | participant_oof |
| inception_full__b4_star_fixed10 | macro_pr_auc_ovr | 5 | 47.7 ± 7.1 | participant_oof |
| inception_full__b4_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 43.0 ± 7.3 | participant_oof |
| inception_full__b4_star_fixed10 | abstention_aware_macro_f1 | 5 | 43.1 ± 7.3 | participant_oof |
| compact_cnn__b5_star_fixed10 | balanced_accuracy | 5 | 52.3 ± 3.5 | participant_oof |
| compact_cnn__b5_star_fixed10 | macro_f1 | 5 | 53.4 ± 4.5 | participant_oof |
| compact_cnn__b5_star_fixed10 | macro_roc_auc_ovr | 5 | 69.4 ± 3.1 | participant_oof |
| compact_cnn__b5_star_fixed10 | macro_pr_auc_ovr | 5 | 60.3 ± 4.7 | participant_oof |
| compact_cnn__b5_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 52.3 ± 3.5 | participant_oof |
| compact_cnn__b5_star_fixed10 | abstention_aware_macro_f1 | 5 | 53.4 ± 4.5 | participant_oof |
| inception_full__b5_star_fixed10 | balanced_accuracy | 5 | 51.8 ± 3.9 | participant_oof |
| inception_full__b5_star_fixed10 | macro_f1 | 5 | 51.5 ± 3.7 | participant_oof |
| inception_full__b5_star_fixed10 | macro_roc_auc_ovr | 5 | 69.5 ± 2.7 | participant_oof |
| inception_full__b5_star_fixed10 | macro_pr_auc_ovr | 5 | 61.3 ± 3.4 | participant_oof |
| inception_full__b5_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 51.8 ± 3.9 | participant_oof |
| inception_full__b5_star_fixed10 | abstention_aware_macro_f1 | 5 | 51.5 ± 3.7 | participant_oof |
| compact_cnn__b6_star_fixed10 | balanced_accuracy | 5 | 49.2 ± 6.9 | participant_oof |
| compact_cnn__b6_star_fixed10 | macro_f1 | 5 | 48.7 ± 6.7 | participant_oof |
| compact_cnn__b6_star_fixed10 | macro_roc_auc_ovr | 5 | 67.4 ± 3.7 | participant_oof |
| compact_cnn__b6_star_fixed10 | macro_pr_auc_ovr | 5 | 58.3 ± 4.8 | participant_oof |
| compact_cnn__b6_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 49.2 ± 6.9 | participant_oof |
| compact_cnn__b6_star_fixed10 | abstention_aware_macro_f1 | 5 | 48.7 ± 6.7 | participant_oof |
| inception_full__b6_star_fixed10 | balanced_accuracy | 5 | 56.0 ± 3.8 | participant_oof |
| inception_full__b6_star_fixed10 | macro_f1 | 5 | 56.2 ± 4.7 | participant_oof |
| inception_full__b6_star_fixed10 | macro_roc_auc_ovr | 5 | 72.5 ± 4.0 | participant_oof |
| inception_full__b6_star_fixed10 | macro_pr_auc_ovr | 5 | 63.6 ± 5.0 | participant_oof |
| inception_full__b6_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 56.0 ± 3.8 | participant_oof |
| inception_full__b6_star_fixed10 | abstention_aware_macro_f1 | 5 | 56.2 ± 4.7 | participant_oof |
| compact_cnn__b7_star_fixed10 | balanced_accuracy | 5 | 53.8 ± 6.8 | participant_oof |
| compact_cnn__b7_star_fixed10 | macro_f1 | 5 | 52.9 ± 6.6 | participant_oof |
| compact_cnn__b7_star_fixed10 | macro_roc_auc_ovr | 5 | 69.2 ± 4.3 | participant_oof |
| compact_cnn__b7_star_fixed10 | macro_pr_auc_ovr | 5 | 61.0 ± 7.0 | participant_oof |
| compact_cnn__b7_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 53.8 ± 6.8 | participant_oof |
| compact_cnn__b7_star_fixed10 | abstention_aware_macro_f1 | 5 | 52.9 ± 6.6 | participant_oof |
| inception_full__b7_star_fixed10 | balanced_accuracy | 5 | 56.8 ± 2.4 | participant_oof |
| inception_full__b7_star_fixed10 | macro_f1 | 5 | 57.2 ± 2.5 | participant_oof |
| inception_full__b7_star_fixed10 | macro_roc_auc_ovr | 5 | 72.8 ± 2.5 | participant_oof |
| inception_full__b7_star_fixed10 | macro_pr_auc_ovr | 5 | 64.6 ± 4.0 | participant_oof |
| inception_full__b7_star_fixed10 | abstention_aware_balanced_accuracy | 5 | 56.8 ± 2.4 | participant_oof |
| inception_full__b7_star_fixed10 | abstention_aware_macro_f1 | 5 | 57.2 ± 2.5 | participant_oof |

<details><summary>Per-class repeat distributions</summary>

| Case | Class | Metric | Repeats | Mean ± SD (%) |
|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 69.1 ± 5.3 |
| compact_cnn__b0_star_fixed10 | Pre-Frail | f1 | 5 | 57.0 ± 7.0 |
| compact_cnn__b0_star_fixed10 | Pre-Frail | recall | 5 | 62.2 ± 16.9 |
| compact_cnn__b0_star_fixed10 | Pre-Frail | specificity | 5 | 76.0 ± 8.2 |
| compact_cnn__b0_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 74.8 ± 6.7 |
| compact_cnn__b0_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 68.6 ± 7.3 |
| compact_cnn__b0_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.9 ± 6.0 |
| compact_cnn__b0_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.1 ± 8.4 |
| compact_cnn__b0_star_fixed10 | Robust/Non-Frail | recall | 5 | 46.7 ± 9.5 |
| compact_cnn__b0_star_fixed10 | Robust/Non-Frail | specificity | 5 | 67.1 ± 3.2 |
| compact_cnn__b0_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 58.7 ± 5.9 |
| compact_cnn__b0_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 52.3 ± 5.7 |
| compact_cnn__b0_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 68.6 ± 8.1 |
| compact_cnn__b0_star_fixed10 | Young | f1 | 5 | 53.7 ± 12.0 |
| compact_cnn__b0_star_fixed10 | Young | recall | 5 | 52.5 ± 18.5 |
| compact_cnn__b0_star_fixed10 | Young | specificity | 5 | 84.8 ± 12.3 |
| compact_cnn__b0_star_fixed10 | Young | roc_auc_ovr | 5 | 74.0 ± 2.7 |
| compact_cnn__b0_star_fixed10 | Young | pr_auc_ovr | 5 | 62.0 ± 10.5 |
| compact_cnn__b1_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 62.8 ± 9.6 |
| compact_cnn__b1_star_fixed10 | Pre-Frail | f1 | 5 | 51.0 ± 10.5 |
| compact_cnn__b1_star_fixed10 | Pre-Frail | recall | 5 | 55.6 ± 7.9 |
| compact_cnn__b1_star_fixed10 | Pre-Frail | specificity | 5 | 70.0 ± 15.8 |
| compact_cnn__b1_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 70.3 ± 5.9 |
| compact_cnn__b1_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 59.6 ± 7.6 |
| compact_cnn__b1_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 54.7 ± 5.8 |
| compact_cnn__b1_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.5 ± 7.3 |
| compact_cnn__b1_star_fixed10 | Robust/Non-Frail | recall | 5 | 51.7 ± 10.9 |
| compact_cnn__b1_star_fixed10 | Robust/Non-Frail | specificity | 5 | 57.6 ± 7.7 |
| compact_cnn__b1_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 58.0 ± 2.1 |
| compact_cnn__b1_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 52.4 ± 4.9 |
| compact_cnn__b1_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 58.0 ± 11.1 |
| compact_cnn__b1_star_fixed10 | Young | f1 | 5 | 32.1 ± 24.2 |
| compact_cnn__b1_star_fixed10 | Young | recall | 5 | 27.5 ± 22.4 |
| compact_cnn__b1_star_fixed10 | Young | specificity | 5 | 88.6 ± 2.6 |
| compact_cnn__b1_star_fixed10 | Young | roc_auc_ovr | 5 | 62.7 ± 5.1 |
| compact_cnn__b1_star_fixed10 | Young | pr_auc_ovr | 5 | 50.5 ± 10.6 |
| compact_cnn__b2_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 66.3 ± 9.3 |
| compact_cnn__b2_star_fixed10 | Pre-Frail | f1 | 5 | 52.6 ± 14.0 |
| compact_cnn__b2_star_fixed10 | Pre-Frail | recall | 5 | 55.6 ± 22.2 |
| compact_cnn__b2_star_fixed10 | Pre-Frail | specificity | 5 | 77.0 ± 11.5 |
| compact_cnn__b2_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 78.8 ± 6.2 |
| compact_cnn__b2_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 68.4 ± 9.0 |
| compact_cnn__b2_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 60.6 ± 6.6 |
| compact_cnn__b2_star_fixed10 | Robust/Non-Frail | f1 | 5 | 54.4 ± 12.0 |
| compact_cnn__b2_star_fixed10 | Robust/Non-Frail | recall | 5 | 60.0 ± 20.7 |
| compact_cnn__b2_star_fixed10 | Robust/Non-Frail | specificity | 5 | 61.2 ± 9.8 |
| compact_cnn__b2_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 64.3 ± 6.0 |
| compact_cnn__b2_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 59.4 ± 6.5 |
| compact_cnn__b2_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 74.6 ± 6.8 |
| compact_cnn__b2_star_fixed10 | Young | f1 | 5 | 64.5 ± 11.8 |
| compact_cnn__b2_star_fixed10 | Young | recall | 5 | 55.0 ± 11.2 |
| compact_cnn__b2_star_fixed10 | Young | specificity | 5 | 94.3 ± 4.0 |
| compact_cnn__b2_star_fixed10 | Young | roc_auc_ovr | 5 | 81.0 ± 4.6 |
| compact_cnn__b2_star_fixed10 | Young | pr_auc_ovr | 5 | 70.6 ± 11.8 |
| compact_cnn__b3_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 59.3 ± 9.8 |
| compact_cnn__b3_star_fixed10 | Pre-Frail | f1 | 5 | 43.8 ± 12.8 |
| compact_cnn__b3_star_fixed10 | Pre-Frail | recall | 5 | 46.7 ± 19.9 |
| compact_cnn__b3_star_fixed10 | Pre-Frail | specificity | 5 | 72.0 ± 10.4 |
| compact_cnn__b3_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 72.0 ± 4.9 |
| compact_cnn__b3_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 57.1 ± 3.5 |
| compact_cnn__b3_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 53.4 ± 10.0 |
| compact_cnn__b3_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.4 ± 13.6 |
| compact_cnn__b3_star_fixed10 | Robust/Non-Frail | recall | 5 | 55.0 ± 19.2 |
| compact_cnn__b3_star_fixed10 | Robust/Non-Frail | specificity | 5 | 51.8 ± 9.7 |
| compact_cnn__b3_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 56.1 ± 2.3 |
| compact_cnn__b3_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 48.8 ± 3.7 |
| compact_cnn__b3_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 67.1 ± 9.1 |
| compact_cnn__b3_star_fixed10 | Young | f1 | 5 | 50.7 ± 17.0 |
| compact_cnn__b3_star_fixed10 | Young | recall | 5 | 40.0 ± 16.3 |
| compact_cnn__b3_star_fixed10 | Young | specificity | 5 | 94.3 ± 5.2 |
| compact_cnn__b3_star_fixed10 | Young | roc_auc_ovr | 5 | 72.5 ± 6.3 |
| compact_cnn__b3_star_fixed10 | Young | pr_auc_ovr | 5 | 67.3 ± 6.5 |
| compact_cnn__b4_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 56.3 ± 4.7 |
| compact_cnn__b4_star_fixed10 | Pre-Frail | f1 | 5 | 41.4 ± 7.5 |
| compact_cnn__b4_star_fixed10 | Pre-Frail | recall | 5 | 46.7 ± 14.5 |
| compact_cnn__b4_star_fixed10 | Pre-Frail | specificity | 5 | 66.0 ± 10.2 |
| compact_cnn__b4_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 60.8 ± 10.6 |
| compact_cnn__b4_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 45.7 ± 10.3 |
| compact_cnn__b4_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 53.4 ± 4.6 |
| compact_cnn__b4_star_fixed10 | Robust/Non-Frail | f1 | 5 | 43.5 ± 9.4 |
| compact_cnn__b4_star_fixed10 | Robust/Non-Frail | recall | 5 | 43.3 ± 14.9 |
| compact_cnn__b4_star_fixed10 | Robust/Non-Frail | specificity | 5 | 63.5 ± 11.3 |
| compact_cnn__b4_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 53.0 ± 8.0 |
| compact_cnn__b4_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 50.8 ± 10.0 |
| compact_cnn__b4_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 61.9 ± 5.7 |
| compact_cnn__b4_star_fixed10 | Young | f1 | 5 | 43.5 ± 9.5 |
| compact_cnn__b4_star_fixed10 | Young | recall | 5 | 40.0 ± 10.5 |
| compact_cnn__b4_star_fixed10 | Young | specificity | 5 | 83.8 ± 2.6 |
| compact_cnn__b4_star_fixed10 | Young | roc_auc_ovr | 5 | 64.6 ± 5.7 |
| compact_cnn__b4_star_fixed10 | Young | pr_auc_ovr | 5 | 58.8 ± 6.9 |
| compact_cnn__b5_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 64.6 ± 5.0 |
| compact_cnn__b5_star_fixed10 | Pre-Frail | f1 | 5 | 51.3 ± 7.0 |
| compact_cnn__b5_star_fixed10 | Pre-Frail | recall | 5 | 51.1 ± 6.1 |
| compact_cnn__b5_star_fixed10 | Pre-Frail | specificity | 5 | 78.0 ± 7.6 |
| compact_cnn__b5_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 74.0 ± 3.5 |
| compact_cnn__b5_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 62.3 ± 4.2 |
| compact_cnn__b5_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 54.3 ± 4.2 |
| compact_cnn__b5_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.7 ± 7.3 |
| compact_cnn__b5_star_fixed10 | Robust/Non-Frail | recall | 5 | 53.3 ± 12.6 |
| compact_cnn__b5_star_fixed10 | Robust/Non-Frail | specificity | 5 | 55.3 ± 5.3 |
| compact_cnn__b5_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 57.2 ± 5.3 |
| compact_cnn__b5_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 50.9 ± 6.4 |
| compact_cnn__b5_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 72.0 ± 6.6 |
| compact_cnn__b5_star_fixed10 | Young | f1 | 5 | 60.1 ± 11.5 |
| compact_cnn__b5_star_fixed10 | Young | recall | 5 | 52.5 ± 10.5 |
| compact_cnn__b5_star_fixed10 | Young | specificity | 5 | 91.4 ± 6.2 |
| compact_cnn__b5_star_fixed10 | Young | roc_auc_ovr | 5 | 77.1 ± 6.8 |
| compact_cnn__b5_star_fixed10 | Young | pr_auc_ovr | 5 | 67.9 ± 11.5 |
| compact_cnn__b6_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 64.0 ± 7.0 |
| compact_cnn__b6_star_fixed10 | Pre-Frail | f1 | 5 | 51.7 ± 8.6 |
| compact_cnn__b6_star_fixed10 | Pre-Frail | recall | 5 | 60.0 ± 12.7 |
| compact_cnn__b6_star_fixed10 | Pre-Frail | specificity | 5 | 68.0 ± 8.4 |
| compact_cnn__b6_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 74.2 ± 6.4 |
| compact_cnn__b6_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 61.6 ± 6.6 |
| compact_cnn__b6_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 52.4 ± 4.4 |
| compact_cnn__b6_star_fixed10 | Robust/Non-Frail | f1 | 5 | 41.8 ± 7.3 |
| compact_cnn__b6_star_fixed10 | Robust/Non-Frail | recall | 5 | 40.0 ± 9.1 |
| compact_cnn__b6_star_fixed10 | Robust/Non-Frail | specificity | 5 | 64.7 ± 8.3 |
| compact_cnn__b6_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 54.6 ± 4.6 |
| compact_cnn__b6_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 48.5 ± 3.9 |
| compact_cnn__b6_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 67.6 ± 5.9 |
| compact_cnn__b6_star_fixed10 | Young | f1 | 5 | 52.6 ± 9.6 |
| compact_cnn__b6_star_fixed10 | Young | recall | 5 | 47.5 ± 10.5 |
| compact_cnn__b6_star_fixed10 | Young | specificity | 5 | 87.6 ± 4.3 |
| compact_cnn__b6_star_fixed10 | Young | roc_auc_ovr | 5 | 73.3 ± 9.8 |
| compact_cnn__b6_star_fixed10 | Young | pr_auc_ovr | 5 | 64.8 ± 9.7 |
| compact_cnn__b7_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 69.1 ± 5.3 |
| compact_cnn__b7_star_fixed10 | Pre-Frail | f1 | 5 | 57.0 ± 7.0 |
| compact_cnn__b7_star_fixed10 | Pre-Frail | recall | 5 | 62.2 ± 16.9 |
| compact_cnn__b7_star_fixed10 | Pre-Frail | specificity | 5 | 76.0 ± 8.2 |
| compact_cnn__b7_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 74.8 ± 6.7 |
| compact_cnn__b7_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 68.6 ± 7.3 |
| compact_cnn__b7_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.9 ± 6.0 |
| compact_cnn__b7_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.1 ± 8.4 |
| compact_cnn__b7_star_fixed10 | Robust/Non-Frail | recall | 5 | 46.7 ± 9.5 |
| compact_cnn__b7_star_fixed10 | Robust/Non-Frail | specificity | 5 | 67.1 ± 3.2 |
| compact_cnn__b7_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 58.7 ± 5.9 |
| compact_cnn__b7_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 52.3 ± 5.7 |
| compact_cnn__b7_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 68.6 ± 8.1 |
| compact_cnn__b7_star_fixed10 | Young | f1 | 5 | 53.7 ± 12.0 |
| compact_cnn__b7_star_fixed10 | Young | recall | 5 | 52.5 ± 18.5 |
| compact_cnn__b7_star_fixed10 | Young | specificity | 5 | 84.8 ± 12.3 |
| compact_cnn__b7_star_fixed10 | Young | roc_auc_ovr | 5 | 74.0 ± 2.7 |
| compact_cnn__b7_star_fixed10 | Young | pr_auc_ovr | 5 | 62.0 ± 10.5 |
| inception_full__b0_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 67.4 ± 2.5 |
| inception_full__b0_star_fixed10 | Pre-Frail | f1 | 5 | 55.1 ± 3.9 |
| inception_full__b0_star_fixed10 | Pre-Frail | recall | 5 | 57.8 ± 9.3 |
| inception_full__b0_star_fixed10 | Pre-Frail | specificity | 5 | 77.0 ± 5.7 |
| inception_full__b0_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 75.9 ± 5.0 |
| inception_full__b0_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 62.1 ± 5.5 |
| inception_full__b0_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.2 ± 5.9 |
| inception_full__b0_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.9 ± 7.8 |
| inception_full__b0_star_fixed10 | Robust/Non-Frail | recall | 5 | 50.0 ± 10.2 |
| inception_full__b0_star_fixed10 | Robust/Non-Frail | specificity | 5 | 62.4 ± 3.2 |
| inception_full__b0_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 59.6 ± 2.0 |
| inception_full__b0_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 53.8 ± 4.4 |
| inception_full__b0_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 77.0 ± 2.0 |
| inception_full__b0_star_fixed10 | Young | f1 | 5 | 67.7 ± 3.8 |
| inception_full__b0_star_fixed10 | Young | recall | 5 | 62.5 ± 0.0 |
| inception_full__b0_star_fixed10 | Young | specificity | 5 | 91.4 ± 4.0 |
| inception_full__b0_star_fixed10 | Young | roc_auc_ovr | 5 | 83.0 ± 5.7 |
| inception_full__b0_star_fixed10 | Young | pr_auc_ovr | 5 | 77.7 ± 5.1 |
| inception_full__b1_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 64.7 ± 4.3 |
| inception_full__b1_star_fixed10 | Pre-Frail | f1 | 5 | 51.6 ± 5.4 |
| inception_full__b1_star_fixed10 | Pre-Frail | recall | 5 | 53.3 ± 9.3 |
| inception_full__b1_star_fixed10 | Pre-Frail | specificity | 5 | 76.0 ± 9.6 |
| inception_full__b1_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 75.0 ± 6.1 |
| inception_full__b1_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 67.8 ± 3.9 |
| inception_full__b1_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 57.5 ± 2.1 |
| inception_full__b1_star_fixed10 | Robust/Non-Frail | f1 | 5 | 51.5 ± 5.3 |
| inception_full__b1_star_fixed10 | Robust/Non-Frail | recall | 5 | 55.0 ± 12.6 |
| inception_full__b1_star_fixed10 | Robust/Non-Frail | specificity | 5 | 60.0 ± 9.7 |
| inception_full__b1_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 60.7 ± 5.2 |
| inception_full__b1_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 56.6 ± 6.8 |
| inception_full__b1_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 65.1 ± 6.3 |
| inception_full__b1_star_fixed10 | Young | f1 | 5 | 48.4 ± 11.4 |
| inception_full__b1_star_fixed10 | Young | recall | 5 | 42.5 ± 11.2 |
| inception_full__b1_star_fixed10 | Young | specificity | 5 | 87.6 ± 5.4 |
| inception_full__b1_star_fixed10 | Young | roc_auc_ovr | 5 | 68.3 ± 10.1 |
| inception_full__b1_star_fixed10 | Young | pr_auc_ovr | 5 | 60.0 ± 10.4 |
| inception_full__b2_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 60.7 ± 6.0 |
| inception_full__b2_star_fixed10 | Pre-Frail | f1 | 5 | 44.8 ± 9.1 |
| inception_full__b2_star_fixed10 | Pre-Frail | recall | 5 | 44.4 ± 15.7 |
| inception_full__b2_star_fixed10 | Pre-Frail | specificity | 5 | 77.0 ± 12.0 |
| inception_full__b2_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 72.4 ± 5.9 |
| inception_full__b2_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 61.1 ± 2.3 |
| inception_full__b2_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 55.6 ± 6.8 |
| inception_full__b2_star_fixed10 | Robust/Non-Frail | f1 | 5 | 51.2 ± 9.8 |
| inception_full__b2_star_fixed10 | Robust/Non-Frail | recall | 5 | 58.3 ± 15.6 |
| inception_full__b2_star_fixed10 | Robust/Non-Frail | specificity | 5 | 52.9 ± 11.8 |
| inception_full__b2_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 58.6 ± 4.5 |
| inception_full__b2_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 55.2 ± 7.7 |
| inception_full__b2_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 71.7 ± 2.1 |
| inception_full__b2_star_fixed10 | Young | f1 | 5 | 59.5 ± 3.4 |
| inception_full__b2_star_fixed10 | Young | recall | 5 | 50.0 ± 8.8 |
| inception_full__b2_star_fixed10 | Young | specificity | 5 | 93.3 ± 5.4 |
| inception_full__b2_star_fixed10 | Young | roc_auc_ovr | 5 | 83.3 ± 6.8 |
| inception_full__b2_star_fixed10 | Young | pr_auc_ovr | 5 | 76.3 ± 5.9 |
| inception_full__b3_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 62.2 ± 3.9 |
| inception_full__b3_star_fixed10 | Pre-Frail | f1 | 5 | 48.9 ± 4.7 |
| inception_full__b3_star_fixed10 | Pre-Frail | recall | 5 | 53.3 ± 9.3 |
| inception_full__b3_star_fixed10 | Pre-Frail | specificity | 5 | 71.0 ± 8.9 |
| inception_full__b3_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 74.4 ± 2.1 |
| inception_full__b3_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 61.0 ± 4.6 |
| inception_full__b3_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 53.0 ± 7.3 |
| inception_full__b3_star_fixed10 | Robust/Non-Frail | f1 | 5 | 45.6 ± 11.7 |
| inception_full__b3_star_fixed10 | Robust/Non-Frail | recall | 5 | 48.3 ± 16.0 |
| inception_full__b3_star_fixed10 | Robust/Non-Frail | specificity | 5 | 57.6 ± 7.7 |
| inception_full__b3_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 55.9 ± 3.9 |
| inception_full__b3_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 50.7 ± 6.6 |
| inception_full__b3_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 69.9 ± 8.4 |
| inception_full__b3_star_fixed10 | Young | f1 | 5 | 56.6 ± 15.3 |
| inception_full__b3_star_fixed10 | Young | recall | 5 | 47.5 ± 13.7 |
| inception_full__b3_star_fixed10 | Young | specificity | 5 | 92.4 ± 7.2 |
| inception_full__b3_star_fixed10 | Young | roc_auc_ovr | 5 | 76.2 ± 3.8 |
| inception_full__b3_star_fixed10 | Young | pr_auc_ovr | 5 | 70.5 ± 6.6 |
| inception_full__b4_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 55.1 ± 9.3 |
| inception_full__b4_star_fixed10 | Pre-Frail | f1 | 5 | 39.3 ± 12.6 |
| inception_full__b4_star_fixed10 | Pre-Frail | recall | 5 | 42.2 ± 14.5 |
| inception_full__b4_star_fixed10 | Pre-Frail | specificity | 5 | 68.0 ± 5.7 |
| inception_full__b4_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 58.6 ± 11.9 |
| inception_full__b4_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 40.8 ± 7.7 |
| inception_full__b4_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 55.8 ± 5.3 |
| inception_full__b4_star_fixed10 | Robust/Non-Frail | f1 | 5 | 49.4 ± 6.3 |
| inception_full__b4_star_fixed10 | Robust/Non-Frail | recall | 5 | 51.7 ± 9.1 |
| inception_full__b4_star_fixed10 | Robust/Non-Frail | specificity | 5 | 60.0 ± 7.7 |
| inception_full__b4_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 58.2 ± 8.2 |
| inception_full__b4_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 55.6 ± 7.9 |
| inception_full__b4_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 60.4 ± 6.6 |
| inception_full__b4_star_fixed10 | Young | f1 | 5 | 40.5 ± 10.5 |
| inception_full__b4_star_fixed10 | Young | recall | 5 | 35.0 ± 10.5 |
| inception_full__b4_star_fixed10 | Young | specificity | 5 | 85.7 ± 5.8 |
| inception_full__b4_star_fixed10 | Young | roc_auc_ovr | 5 | 54.8 ± 6.8 |
| inception_full__b4_star_fixed10 | Young | pr_auc_ovr | 5 | 46.8 ± 11.2 |
| inception_full__b5_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 65.6 ± 5.1 |
| inception_full__b5_star_fixed10 | Pre-Frail | f1 | 5 | 52.4 ± 6.6 |
| inception_full__b5_star_fixed10 | Pre-Frail | recall | 5 | 51.1 ± 6.1 |
| inception_full__b5_star_fixed10 | Pre-Frail | specificity | 5 | 80.0 ± 6.1 |
| inception_full__b5_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 73.1 ± 5.6 |
| inception_full__b5_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 62.3 ± 4.6 |
| inception_full__b5_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 51.4 ± 4.8 |
| inception_full__b5_star_fixed10 | Robust/Non-Frail | f1 | 5 | 42.2 ± 5.9 |
| inception_full__b5_star_fixed10 | Robust/Non-Frail | recall | 5 | 41.7 ± 8.3 |
| inception_full__b5_star_fixed10 | Robust/Non-Frail | specificity | 5 | 61.2 ± 9.8 |
| inception_full__b5_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 53.1 ± 3.9 |
| inception_full__b5_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 50.6 ± 8.1 |
| inception_full__b5_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 72.2 ± 5.8 |
| inception_full__b5_star_fixed10 | Young | f1 | 5 | 60.0 ± 8.7 |
| inception_full__b5_star_fixed10 | Young | recall | 5 | 62.5 ± 8.8 |
| inception_full__b5_star_fixed10 | Young | specificity | 5 | 81.9 ± 9.2 |
| inception_full__b5_star_fixed10 | Young | roc_auc_ovr | 5 | 82.4 ± 4.2 |
| inception_full__b5_star_fixed10 | Young | pr_auc_ovr | 5 | 71.0 ± 6.6 |
| inception_full__b6_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 66.8 ± 1.4 |
| inception_full__b6_star_fixed10 | Pre-Frail | f1 | 5 | 54.4 ± 1.6 |
| inception_full__b6_star_fixed10 | Pre-Frail | recall | 5 | 55.6 ± 0.0 |
| inception_full__b6_star_fixed10 | Pre-Frail | specificity | 5 | 78.0 ± 2.7 |
| inception_full__b6_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 78.1 ± 5.5 |
| inception_full__b6_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 67.2 ± 3.9 |
| inception_full__b6_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.8 ± 2.8 |
| inception_full__b6_star_fixed10 | Robust/Non-Frail | f1 | 5 | 49.2 ± 6.5 |
| inception_full__b6_star_fixed10 | Robust/Non-Frail | recall | 5 | 50.0 ± 10.2 |
| inception_full__b6_star_fixed10 | Robust/Non-Frail | specificity | 5 | 63.5 ± 4.9 |
| inception_full__b6_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 57.9 ± 2.9 |
| inception_full__b6_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 50.8 ± 5.6 |
| inception_full__b6_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 75.5 ± 5.7 |
| inception_full__b6_star_fixed10 | Young | f1 | 5 | 65.2 ± 8.6 |
| inception_full__b6_star_fixed10 | Young | recall | 5 | 62.5 ± 8.8 |
| inception_full__b6_star_fixed10 | Young | specificity | 5 | 88.6 ± 7.2 |
| inception_full__b6_star_fixed10 | Young | roc_auc_ovr | 5 | 81.4 ± 6.8 |
| inception_full__b6_star_fixed10 | Young | pr_auc_ovr | 5 | 72.7 ± 10.6 |
| inception_full__b7_star_fixed10 | Pre-Frail | balanced_accuracy_ovr | 5 | 67.4 ± 2.5 |
| inception_full__b7_star_fixed10 | Pre-Frail | f1 | 5 | 55.1 ± 3.9 |
| inception_full__b7_star_fixed10 | Pre-Frail | recall | 5 | 57.8 ± 9.3 |
| inception_full__b7_star_fixed10 | Pre-Frail | specificity | 5 | 77.0 ± 5.7 |
| inception_full__b7_star_fixed10 | Pre-Frail | roc_auc_ovr | 5 | 75.9 ± 5.0 |
| inception_full__b7_star_fixed10 | Pre-Frail | pr_auc_ovr | 5 | 62.1 ± 5.5 |
| inception_full__b7_star_fixed10 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.2 ± 5.9 |
| inception_full__b7_star_fixed10 | Robust/Non-Frail | f1 | 5 | 48.9 ± 7.8 |
| inception_full__b7_star_fixed10 | Robust/Non-Frail | recall | 5 | 50.0 ± 10.2 |
| inception_full__b7_star_fixed10 | Robust/Non-Frail | specificity | 5 | 62.4 ± 3.2 |
| inception_full__b7_star_fixed10 | Robust/Non-Frail | roc_auc_ovr | 5 | 59.6 ± 2.0 |
| inception_full__b7_star_fixed10 | Robust/Non-Frail | pr_auc_ovr | 5 | 53.8 ± 4.4 |
| inception_full__b7_star_fixed10 | Young | balanced_accuracy_ovr | 5 | 77.0 ± 2.0 |
| inception_full__b7_star_fixed10 | Young | f1 | 5 | 67.7 ± 3.8 |
| inception_full__b7_star_fixed10 | Young | recall | 5 | 62.5 ± 0.0 |
| inception_full__b7_star_fixed10 | Young | specificity | 5 | 91.4 ± 4.0 |
| inception_full__b7_star_fixed10 | Young | roc_auc_ovr | 5 | 83.0 ± 5.7 |
| inception_full__b7_star_fixed10 | Young | pr_auc_ovr | 5 | 77.7 ± 5.1 |

</details>

## Stage 3 InceptionTime B0–B7 comparison

One InceptionTime table: B0 is the baseline and B1–B7 are paired to B0 by repeat. Values are native participant-OOF repeat mean ± population SD. B2 and B6 are declared coupled bundles; B7 is a reporting-aggregation ablation.

| Profile | Factor | Native endpoint | BA mean ± SD (%) | Δ BA vs B0 mean ± SD (pp) | Macro-F1 mean ± SD (%) | Δ Macro-F1 vs B0 mean ± SD (pp) | Worst-class F1 mean ± SD (%) | Δ worst-class F1 vs B0 mean ± SD (pp) | Repeats | Changed controls | Factor audit | Available |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | window_balanced_to_participant | 56.8 ± 2.1 | 0.0 ± 0.0 | 57.2 ± 2.2 | 0.0 ± 0.0 | 46.6 ± 3.4 | 0.0 ± 0.0 | 5 | [] | baseline_no_contrast | True |
| B1 | sampling_rate | window_balanced_to_participant | 50.3 ± 2.0 | -6.5 ± 3.5 | 50.4 ± 2.0 | -6.8 ± 3.1 | 43.4 ± 6.6 | -3.3 ± 9.3 | 5 | ["controls.target_fs_hz"] | pass_exact_declared_paths | True |
| B2 | window_plan | window_balanced_to_participant | 50.9 ± 4.3 | -5.8 ± 4.7 | 51.8 ± 3.7 | -5.4 ± 3.9 | 41.6 ± 6.6 | -5.0 ± 7.1 | 5 | ["controls.hop_seconds", "controls.window_seconds"] | pass_exact_declared_paths | True |
| B3 | imu_preprocessing | window_balanced_to_participant | 49.7 ± 5.3 | -7.0 ± 5.9 | 50.3 ± 6.6 | -6.9 ± 7.1 | 40.3 ± 7.8 | -6.3 ± 10.1 | 5 | ["controls.imu_preprocessing"] | pass_exact_declared_paths | True |
| B4 | normalization | window_balanced_to_participant | 43.0 ± 6.6 | -13.8 ± 6.3 | 43.1 ± 6.5 | -14.2 ± 5.9 | 34.7 ± 10.7 | -12.0 ± 9.8 | 5 | ["controls.normalization"] | pass_exact_declared_paths | True |
| B5 | sampler | window_balanced_to_participant | 52.3 ± 3.3 | -4.4 ± 2.8 | 52.1 ± 3.4 | -5.2 ± 2.3 | 43.1 ± 5.2 | -3.6 ± 3.0 | 5 | ["controls.sampler"] | pass_exact_declared_paths | True |
| B6 | optimizer_and_batch_size | window_balanced_to_participant | 56.0 ± 3.4 | -0.7 ± 5.0 | 56.2 ± 4.2 | -1.0 ± 5.7 | 49.2 ± 5.8 | 2.5 ± 8.0 | 5 | ["controls.batch_size", "controls.optimizer"] | pass_exact_declared_paths | True |
| B7 | primary_aggregation | line_b_equal_role_families | 57.3 ± 3.3 | 0.6 ± 1.9 | 57.9 ± 4.0 | 0.7 ± 2.4 | 47.0 ± 7.3 | 0.4 ± 6.6 | 5 | ["controls.primary_report_aggregation_view"] | pass_exact_declared_paths | True |

## Stage 3 CompactCNN B0–B7 comparison

One CompactCNN table: B0 is the baseline and B1–B7 are paired to B0 by repeat. Values are native participant-OOF repeat mean ± population SD. B2 and B6 are declared coupled bundles; B7 is a reporting-aggregation ablation.

| Profile | Factor | Native endpoint | BA mean ± SD (%) | Δ BA vs B0 mean ± SD (pp) | Macro-F1 mean ± SD (%) | Δ Macro-F1 vs B0 mean ± SD (pp) | Worst-class F1 mean ± SD (%) | Δ worst-class F1 vs B0 mean ± SD (pp) | Repeats | Changed controls | Factor audit | Available |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | window_balanced_to_participant | 53.8 ± 6.1 | 0.0 ± 0.0 | 52.9 ± 5.9 | 0.0 ± 0.0 | 44.4 ± 5.9 | 0.0 ± 0.0 | 5 | [] | baseline_no_contrast | True |
| B1 | sampling_rate | window_balanced_to_participant | 44.9 ± 8.3 | -8.9 ± 13.2 | 43.8 ± 9.8 | -9.1 ± 14.5 | 29.4 ± 19.4 | -15.0 ± 22.8 | 5 | ["controls.target_fs_hz"] | pass_exact_declared_paths | True |
| B2 | window_plan | window_balanced_to_participant | 56.9 ± 5.3 | 3.1 ± 11.2 | 57.2 ± 6.2 | 4.2 ± 11.8 | 44.3 ± 9.7 | -0.1 ± 14.8 | 5 | ["controls.hop_seconds", "controls.window_seconds"] | pass_exact_declared_paths | True |
| B3 | imu_preprocessing | window_balanced_to_participant | 47.2 ± 8.4 | -6.6 ± 9.8 | 47.6 ± 8.5 | -5.3 ± 9.2 | 37.1 ± 7.3 | -7.3 ± 8.0 | 5 | ["controls.imu_preprocessing"] | pass_exact_declared_paths | True |
| B4 | normalization | window_balanced_to_participant | 43.3 ± 3.3 | -10.5 ± 4.4 | 42.8 ± 3.1 | -10.1 ± 4.4 | 34.3 ± 4.8 | -10.1 ± 7.9 | 5 | ["controls.normalization"] | pass_exact_declared_paths | True |
| B5 | sampler | window_balanced_to_participant | 52.3 ± 3.2 | -1.5 ± 6.0 | 53.4 ± 4.1 | 0.4 ± 6.0 | 44.9 ± 2.7 | 0.5 ± 6.4 | 5 | ["controls.sampler"] | pass_exact_declared_paths | True |
| B6 | optimizer_and_batch_size | window_balanced_to_participant | 49.2 ± 6.1 | -4.6 ± 6.0 | 48.7 ± 6.0 | -4.2 ± 6.0 | 41.1 ± 6.0 | -3.3 ± 5.1 | 5 | ["controls.batch_size", "controls.optimizer"] | pass_exact_declared_paths | True |
| B7 | primary_aggregation | line_b_equal_role_families | 54.7 ± 6.4 | 0.9 ± 3.7 | 54.0 ± 6.5 | 1.1 ± 3.4 | 47.8 ± 8.6 | 3.4 ± 4.8 | 5 | ["controls.primary_report_aggregation_view"] | pass_exact_declared_paths | True |

## Stage 3 B0–B7 InceptionTime versus CompactCNN

Each row horizontally matches the two models under the same B-profile, repeat split and native endpoint. InceptionTime − CNN is a descriptive matched architecture comparison, not one of the fourteen B0-centered ablations and carries no significance claim.

| Profile | Factor | Native endpoint | InceptionTime BA mean ± SD (%) | CNN BA mean ± SD (%) | InceptionTime − CNN Δ BA mean ± SD (pp) | InceptionTime Macro-F1 mean ± SD (%) | CNN Macro-F1 mean ± SD (%) | InceptionTime − CNN Δ Macro-F1 mean ± SD (pp) | InceptionTime worst-class F1 mean ± SD (%) | CNN worst-class F1 mean ± SD (%) | InceptionTime − CNN Δ worst-class F1 mean ± SD (pp) | Paired repeats | Controls match | Available |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B0 | baseline | window_balanced_to_participant | 56.8 ± 2.1 | 53.8 ± 6.1 | 3.0 ± 6.6 | 57.2 ± 2.2 | 52.9 ± 5.9 | 4.3 ± 6.3 | 46.6 ± 3.4 | 44.4 ± 5.9 | 2.2 ± 6.2 | 5 | True | True |
| B1 | sampling_rate | window_balanced_to_participant | 50.3 ± 2.0 | 44.9 ± 8.3 | 5.4 ± 9.0 | 50.4 ± 2.0 | 43.8 ± 9.8 | 6.6 ± 10.3 | 43.4 ± 6.6 | 29.4 ± 19.4 | 14.0 ± 23.8 | 5 | True | True |
| B2 | window_plan | window_balanced_to_participant | 50.9 ± 4.3 | 56.9 ± 5.3 | -5.9 ± 9.2 | 51.8 ± 3.7 | 57.2 ± 6.2 | -5.4 ± 9.6 | 41.6 ± 6.6 | 44.3 ± 9.7 | -2.7 ± 15.3 | 5 | True | True |
| B3 | imu_preprocessing | window_balanced_to_participant | 49.7 ± 5.3 | 47.2 ± 8.4 | 2.5 ± 12.1 | 50.3 ± 6.6 | 47.6 ± 8.5 | 2.7 ± 12.3 | 40.3 ± 7.8 | 37.1 ± 7.3 | 3.2 ± 10.6 | 5 | True | True |
| B4 | normalization | window_balanced_to_participant | 43.0 ± 6.6 | 43.3 ± 3.3 | -0.4 ± 6.0 | 43.1 ± 6.5 | 42.8 ± 3.1 | 0.3 ± 6.4 | 34.7 ± 10.7 | 34.3 ± 4.8 | 0.3 ± 11.1 | 5 | True | True |
| B5 | sampler | window_balanced_to_participant | 52.3 ± 3.3 | 52.3 ± 3.2 | -0.0 ± 4.9 | 52.1 ± 3.4 | 53.4 ± 4.1 | -1.3 ± 5.5 | 43.1 ± 5.2 | 44.9 ± 2.7 | -1.8 ± 7.6 | 5 | True | True |
| B6 | optimizer_and_batch_size | window_balanced_to_participant | 56.0 ± 3.4 | 49.2 ± 6.1 | 6.9 ± 4.5 | 56.2 ± 4.2 | 48.7 ± 6.0 | 7.5 ± 4.9 | 49.2 ± 5.8 | 41.1 ± 6.0 | 8.1 ± 8.1 | 5 | True | True |
| B7 | primary_aggregation | line_b_equal_role_families | 57.3 ± 3.3 | 54.7 ± 6.4 | 2.6 ± 7.4 | 57.9 ± 4.0 | 54.0 ± 6.5 | 3.9 ± 8.2 | 47.0 ± 7.3 | 47.8 ± 8.6 | -0.8 ± 12.4 | 5 | True | True |

## Stage 3 centered-star detailed absolute endpoints

Sixteen absolute model/profile endpoints. W/A/B are same-OOF sensitivity views; each row declares its native endpoint.

| Model | Profile | Factor | Native endpoint | Native BA | Native Macro-F1 | Native worst-class F1 | BA W | BA A | BA B | Passed cells | Factor audit | Cross-model controls match |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CompactCNN1D | B0 | baseline | window_balanced_to_participant | 0.5380 | 0.5293 | 0.4440 | 0.5380 | 0.5380 | 0.5472 | 25 | baseline_no_contrast | True |
| InceptionTimeFull | B0 | baseline | window_balanced_to_participant | 0.5676 | 0.5725 | 0.4663 | 0.5676 | 0.5676 | 0.5731 | 25 | baseline_no_contrast | True |
| CompactCNN1D | B1 | sampling_rate | window_balanced_to_participant | 0.4491 | 0.4383 | 0.2938 | 0.4491 | 0.4491 | 0.4694 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B1 | sampling_rate | window_balanced_to_participant | 0.5028 | 0.5045 | 0.4336 | 0.5028 | 0.5028 | 0.4935 | 25 | pass_exact_declared_paths | True |
| CompactCNN1D | B2 | window_plan | window_balanced_to_participant | 0.5685 | 0.5718 | 0.4427 | 0.5685 | 0.5685 | 0.5194 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B2 | window_plan | window_balanced_to_participant | 0.5093 | 0.5180 | 0.4161 | 0.5093 | 0.5093 | 0.5000 | 25 | pass_exact_declared_paths | True |
| CompactCNN1D | B3 | imu_preprocessing | window_balanced_to_participant | 0.4722 | 0.4764 | 0.3707 | 0.4722 | 0.4722 | 0.4639 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B3 | imu_preprocessing | window_balanced_to_participant | 0.4972 | 0.5034 | 0.4030 | 0.4972 | 0.4972 | 0.5120 | 25 | pass_exact_declared_paths | True |
| CompactCNN1D | B4 | normalization | window_balanced_to_participant | 0.4333 | 0.4279 | 0.3433 | 0.4333 | 0.4333 | 0.4204 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B4 | normalization | window_balanced_to_participant | 0.4296 | 0.4308 | 0.3467 | 0.4296 | 0.4296 | 0.4157 | 25 | pass_exact_declared_paths | True |
| CompactCNN1D | B5 | sampler | window_balanced_to_participant | 0.5231 | 0.5338 | 0.4491 | 0.5231 | 0.5231 | 0.4861 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B5 | sampler | window_balanced_to_participant | 0.5231 | 0.5209 | 0.4308 | 0.5231 | 0.5176 | 0.5111 | 25 | pass_exact_declared_paths | True |
| CompactCNN1D | B6 | optimizer_and_batch_size | window_balanced_to_participant | 0.4917 | 0.4872 | 0.4108 | 0.4917 | 0.4917 | 0.4963 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B6 | optimizer_and_batch_size | window_balanced_to_participant | 0.5602 | 0.5623 | 0.4916 | 0.5602 | 0.5602 | 0.5370 | 25 | pass_exact_declared_paths | True |
| CompactCNN1D | B7 | primary_aggregation | line_b_equal_role_families | 0.5472 | 0.5403 | 0.4783 | 0.5380 | 0.5380 | 0.5472 | 25 | pass_exact_declared_paths | True |
| InceptionTimeFull | B7 | primary_aggregation | line_b_equal_role_families | 0.5731 | 0.5794 | 0.4699 | 0.5676 | 0.5676 | 0.5731 | 25 | pass_exact_declared_paths | True |

## Stage 3 centered-star detailed contrast audit

Fourteen same-model B0→variant contrasts. Availability requires all declared repeat×fold cells plus matching seeds, split hashes, held-out rosters, native metrics, and exact factor paths. B0/B7 also audits training-control and window-OOF identity.

| Model | Factor | Reference | Variant | Reference endpoint | Variant endpoint | Native Δ BA | Native Δ Macro-F1 | Native Δ worst-class F1 | Sensitivity-only Δ BA W | Sensitivity-only Δ BA A | Sensitivity-only Δ BA B | Actual changed paths | Factor audit | Seeds match | Split hashes match | Held-out rosters match | Available | N/A reasons | B0/B7 training controls identical | B0/B7 window OOF identical | B0/B7 matched window rows | B0/B7 max absolute probability diff | B0/B7 identity audit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CompactCNN1D | sampling_rate | B0 | B1 | window_balanced_to_participant | window_balanced_to_participant | -0.0889 | -0.0910 | -0.1502 | -0.0889 | -0.0889 | -0.0778 | ["controls.target_fs_hz"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| InceptionTimeFull | sampling_rate | B0 | B1 | window_balanced_to_participant | window_balanced_to_participant | -0.0648 | -0.0680 | -0.0326 | -0.0648 | -0.0648 | -0.0796 | ["controls.target_fs_hz"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| CompactCNN1D | window_plan | B0 | B2 | window_balanced_to_participant | window_balanced_to_participant | 0.0306 | 0.0424 | -0.0013 | 0.0306 | 0.0306 | -0.0278 | ["controls.hop_seconds", "controls.window_seconds"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| InceptionTimeFull | window_plan | B0 | B2 | window_balanced_to_participant | window_balanced_to_participant | -0.0583 | -0.0545 | -0.0502 | -0.0583 | -0.0583 | -0.0731 | ["controls.hop_seconds", "controls.window_seconds"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| CompactCNN1D | imu_preprocessing | B0 | B3 | window_balanced_to_participant | window_balanced_to_participant | -0.0657 | -0.0529 | -0.0733 | -0.0657 | -0.0657 | -0.0833 | ["controls.imu_preprocessing"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | window_balanced_to_participant | window_balanced_to_participant | -0.0704 | -0.0690 | -0.0633 | -0.0704 | -0.0704 | -0.0611 | ["controls.imu_preprocessing"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| CompactCNN1D | normalization | B0 | B4 | window_balanced_to_participant | window_balanced_to_participant | -0.1046 | -0.1014 | -0.1008 | -0.1046 | -0.1046 | -0.1269 | ["controls.normalization"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| InceptionTimeFull | normalization | B0 | B4 | window_balanced_to_participant | window_balanced_to_participant | -0.1380 | -0.1416 | -0.1196 | -0.1380 | -0.1380 | -0.1574 | ["controls.normalization"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| CompactCNN1D | sampler | B0 | B5 | window_balanced_to_participant | window_balanced_to_participant | -0.0148 | 0.0045 | 0.0051 | -0.0148 | -0.0148 | -0.0611 | ["controls.sampler"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| InceptionTimeFull | sampler | B0 | B5 | window_balanced_to_participant | window_balanced_to_participant | -0.0444 | -0.0516 | -0.0355 | -0.0444 | -0.0500 | -0.0620 | ["controls.sampler"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | window_balanced_to_participant | window_balanced_to_participant | -0.0463 | -0.0422 | -0.0332 | -0.0463 | -0.0463 | -0.0509 | ["controls.batch_size", "controls.optimizer"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | window_balanced_to_participant | window_balanced_to_participant | -0.0074 | -0.0101 | 0.0253 | -0.0074 | -0.0074 | -0.0361 | ["controls.batch_size", "controls.optimizer"] | pass_exact_declared_paths | True | True | True | True | [] | N/A | N/A | N/A | N/A | not_applicable_non_report_view_factor |
| CompactCNN1D | primary_aggregation | B0 | B7 | window_balanced_to_participant | line_b_equal_role_families | 0.0093 | 0.0109 | 0.0343 | 0.0000 | 0.0000 | 0.0000 | ["controls.primary_report_aggregation_view"] | pass_exact_declared_paths | True | True | True | True | [] | True | True | 63530 | 0.0000 | exact_row_identity_and_bitwise_probability_match |
| InceptionTimeFull | primary_aggregation | B0 | B7 | window_balanced_to_participant | line_b_equal_role_families | 0.0056 | 0.0069 | 0.0036 | 0.0000 | 0.0000 | 0.0000 | ["controls.primary_report_aggregation_view"] | pass_exact_declared_paths | True | True | True | True | [] | True | True | 63530 | 0.0000 | exact_row_identity_and_bitwise_probability_match |

## Stage 3 centered-star detailed matched-fold deltas

Every declared repeat×fold delta is descriptive only: no CI or significance claim. Seven contrasts within each model share the same correlated B0.

| Model | Factor | Reference | Variant | Repeat | Fold | Native Δ BA | Native Δ Macro-F1 | Native Δ worst-class F1 | Available | Inference |
|---|---|---|---|---|---|---|---|---|---|---|
| CompactCNN1D | sampling_rate | B0 | B1 | 0 | 0 | 0.0000 | -0.0556 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 0 | 1 | 0.1111 | 0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 0 | 2 | 0.1667 | 0.1825 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 0 | 3 | 0.3333 | 0.3222 | 0.1667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 0 | 4 | -0.1111 | 0.0556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 1 | 0 | -0.1667 | -0.1310 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 1 | 1 | -0.1111 | -0.1444 | -0.2667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 1 | 3 | -0.3333 | -0.3222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 1 | 4 | -0.5000 | -0.6317 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 2 | 0 | -0.2222 | -0.1095 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 2 | 1 | -0.3333 | -0.4333 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 2 | 2 | -0.3333 | -0.3000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 2 | 3 | -0.1667 | -0.2778 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 2 | 4 | 0.1667 | 0.2429 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 3 | 0 | -0.1111 | -0.2778 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 3 | 2 | 0.1667 | 0.2540 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 3 | 3 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 3 | 4 | 0.2222 | 0.3056 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 4 | 0 | -0.3333 | -0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 4 | 1 | 0.0556 | 0.1151 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 4 | 2 | -0.5556 | -0.6270 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 4 | 3 | -0.1667 | -0.0571 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampling_rate | B0 | B1 | 4 | 4 | 0.1667 | 0.1444 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 0 | 0 | 0.0000 | 0.0317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 0 | 1 | 0.0000 | -0.0206 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 0 | 2 | -0.1667 | -0.2333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 0 | 3 | -0.3333 | -0.3444 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 0 | 4 | 0.1111 | 0.1444 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 1 | 0 | -0.1667 | -0.1905 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 1 | 1 | -0.1111 | -0.1444 | -0.2667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 1 | 2 | 0.0000 | -0.0206 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 1 | 3 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 1 | 4 | -0.1667 | -0.1111 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 2 | 0 | 0.0556 | 0.1095 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 2 | 1 | -0.1667 | -0.2667 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 2 | 2 | 0.1111 | 0.1889 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 2 | 3 | -0.1667 | -0.2778 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 2 | 4 | 0.0000 | 0.0778 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 3 | 0 | -0.1111 | -0.1389 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 3 | 1 | 0.0000 | 0.0556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 3 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 3 | 3 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 3 | 4 | -0.1111 | -0.0635 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 4 | 1 | 0.1111 | 0.1151 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 4 | 2 | 0.1111 | 0.1222 | 0.1000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 4 | 3 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampling_rate | B0 | B1 | 4 | 4 | -0.1667 | -0.1889 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 0 | 0 | -0.3333 | -0.4048 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 0 | 1 | 0.1111 | 0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 0 | 2 | 0.5000 | 0.5603 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 0 | 3 | 0.5000 | 0.5000 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 0 | 4 | 0.0000 | 0.2302 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 1 | 0 | 0.0000 | 0.0556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 1 | 2 | 0.0000 | 0.0873 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 1 | 3 | 0.1667 | 0.2333 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 1 | 4 | -0.3333 | -0.3778 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 2 | 0 | -0.0556 | -0.1095 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 2 | 1 | -0.3333 | -0.4095 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 2 | 2 | -0.1667 | -0.1000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 2 | 3 | 0.0000 | -0.1333 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 2 | 4 | 0.0000 | 0.0317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 3 | 0 | -0.2222 | -0.1984 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 3 | 1 | 0.0000 | 0.0317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 3 | 2 | 0.1667 | 0.2540 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 3 | 3 | 0.1667 | 0.2222 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 3 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 4 | 1 | 0.1111 | 0.1317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 4 | 2 | 0.0556 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 4 | 3 | 0.3333 | 0.4762 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | window_plan | B0 | B2 | 4 | 4 | 0.0000 | -0.0778 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 0 | 0 | -0.3333 | -0.3175 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 0 | 1 | 0.0000 | 0.0905 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 0 | 2 | 0.0000 | 0.0127 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 0 | 3 | -0.3333 | -0.3333 | -0.3333 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 1 | 0 | -0.0556 | -0.1349 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 1 | 2 | -0.1667 | -0.1556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 1 | 3 | -0.1667 | -0.2762 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 1 | 4 | 0.1667 | 0.2667 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 2 | 1 | -0.3333 | -0.3000 | -0.2667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 2 | 2 | 0.1667 | 0.1000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 2 | 3 | 0.0000 | -0.1111 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 2 | 4 | 0.1667 | 0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 3 | 0 | -0.0556 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 3 | 2 | -0.1667 | -0.2778 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 3 | 3 | 0.1667 | 0.2222 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 3 | 4 | -0.3333 | -0.3690 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 4 | 0 | -0.1667 | -0.1111 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 4 | 1 | 0.1667 | 0.1317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 4 | 2 | -0.4444 | -0.4889 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 4 | 3 | 0.3333 | 0.3111 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | window_plan | B0 | B2 | 4 | 4 | 0.0000 | -0.1222 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 0 | 0 | -0.1667 | -0.3095 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 0 | 1 | 0.1111 | 0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 0 | 3 | 0.1667 | 0.1556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 0 | 4 | -0.1111 | 0.0556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 1 | 0 | -0.2222 | -0.1881 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 1 | 1 | -0.4444 | -0.3889 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 1 | 2 | -0.1667 | -0.0238 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 1 | 3 | -0.3333 | -0.3222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 1 | 4 | -0.1667 | -0.3333 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 2 | 0 | 0.0556 | 0.0611 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 2 | 2 | -0.3333 | -0.3000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 2 | 3 | 0.0000 | -0.1333 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 2 | 4 | 0.0000 | 0.0317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 3 | 0 | -0.2222 | -0.1984 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 3 | 2 | 0.1667 | 0.1206 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 3 | 3 | 0.0000 | 0.0778 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 3 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 4 | 1 | -0.1111 | -0.1905 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 4 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | imu_preprocessing | B0 | B3 | 4 | 4 | 0.1667 | 0.0571 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 0 | 0 | 0.0000 | -0.0238 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 0 | 1 | -0.0556 | -0.1333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 0 | 2 | -0.1667 | -0.1222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 0 | 3 | -0.3333 | -0.4667 | -1.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 0 | 4 | -0.1111 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 1 | 0 | -0.0556 | -0.1071 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 1 | 1 | -0.3333 | -0.3095 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 1 | 2 | 0.1667 | 0.2222 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 1 | 3 | 0.1667 | 0.0889 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 1 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 2 | 1 | -0.3333 | -0.3778 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 2 | 2 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 2 | 3 | 0.0000 | -0.1333 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 2 | 4 | 0.0000 | 0.0778 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 3 | 0 | -0.0556 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 3 | 1 | 0.1667 | 0.1349 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 3 | 2 | -0.1667 | -0.2333 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 3 | 3 | -0.1667 | -0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 3 | 4 | -0.3333 | -0.3690 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 4 | 1 | -0.1111 | -0.1127 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 4 | 2 | 0.1111 | 0.1222 | 0.1000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 4 | 3 | 0.5000 | 0.5222 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | imu_preprocessing | B0 | B3 | 4 | 4 | -0.3333 | -0.4111 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 0 | 0 | -0.3333 | -0.3667 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 0 | 1 | 0.1111 | 0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 0 | 2 | 0.3333 | 0.4270 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 0 | 3 | -0.1667 | -0.1444 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 0 | 4 | -0.3333 | -0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 1 | 0 | 0.0000 | 0.0357 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 1 | 1 | -0.3333 | -0.2540 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 1 | 2 | 0.0000 | 0.1095 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 1 | 3 | 0.0000 | 0.0333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 1 | 4 | -0.3333 | -0.3889 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 2 | 0 | -0.0556 | -0.1333 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 2 | 1 | -0.3333 | -0.3000 | -0.2667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 2 | 2 | -0.0556 | 0.0667 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 2 | 3 | -0.3333 | -0.4000 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 2 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 3 | 1 | -0.1667 | -0.1349 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 3 | 2 | -0.1667 | -0.1127 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 3 | 3 | -0.1667 | -0.2540 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 3 | 4 | 0.2222 | 0.1944 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 4 | 0 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 4 | 1 | -0.1667 | -0.0905 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 4 | 2 | -0.1111 | -0.1222 | -0.1000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 4 | 3 | 0.5000 | 0.5873 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | normalization | B0 | B4 | 4 | 4 | -0.3333 | -0.3556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 0 | 0 | -0.5000 | -0.4127 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 0 | 1 | 0.0000 | 0.0111 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 0 | 2 | 0.0000 | 0.1222 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 0 | 3 | -0.5000 | -0.5556 | -1.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 1 | 0 | -0.3333 | -0.3889 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 1 | 1 | -0.3333 | -0.2540 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 1 | 2 | 0.0000 | -0.0444 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 1 | 3 | -0.3333 | -0.3333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 1 | 4 | -0.3333 | -0.2556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 2 | 0 | 0.0000 | -0.0238 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 2 | 1 | -0.1667 | -0.1556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 2 | 2 | 0.1111 | 0.1889 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 2 | 3 | -0.3333 | -0.4000 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 2 | 4 | 0.1667 | 0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 3 | 0 | -0.1667 | -0.2000 | -0.1000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 3 | 1 | 0.1667 | 0.2444 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 3 | 2 | 0.0000 | 0.0000 | 0.1667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 3 | 3 | -0.1667 | -0.2222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 3 | 4 | -0.1111 | -0.1746 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 4 | 0 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 4 | 1 | -0.1111 | -0.0571 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 4 | 3 | 0.1667 | 0.1889 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | normalization | B0 | B4 | 4 | 4 | -0.3333 | -0.3000 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 0 | 0 | -0.1667 | -0.2333 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 0 | 1 | -0.0556 | 0.0238 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 0 | 2 | 0.3333 | 0.3048 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 0 | 3 | 0.1667 | 0.0556 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 0 | 4 | -0.2222 | -0.0889 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 1 | 0 | 0.0000 | 0.0278 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 1 | 1 | -0.2222 | -0.1944 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 1 | 2 | 0.1667 | 0.2762 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 1 | 3 | -0.1667 | -0.1333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 1 | 4 | -0.3333 | -0.3889 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 2 | 1 | -0.1667 | -0.2667 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 2 | 2 | -0.1111 | -0.0206 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 2 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 2 | 4 | -0.3333 | -0.1905 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 3 | 0 | -0.2222 | -0.1984 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 3 | 1 | -0.1667 | -0.0794 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 3 | 2 | 0.3333 | 0.4095 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 3 | 3 | 0.1667 | 0.2222 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 3 | 4 | 0.2222 | 0.1944 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 4 | 1 | -0.1667 | -0.0905 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 4 | 2 | -0.1111 | -0.1222 | -0.1000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 4 | 3 | 0.3333 | 0.3651 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | sampler | B0 | B5 | 4 | 4 | 0.1667 | 0.0571 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 0 | 0 | -0.1667 | -0.0794 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 0 | 1 | 0.0000 | 0.0905 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 0 | 3 | -0.1667 | -0.1778 | -0.3333 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 1 | 0 | -0.2222 | -0.4312 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 1 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 1 | 4 | 0.1667 | 0.2667 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 2 | 0 | 0.1667 | 0.2540 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 2 | 1 | -0.3333 | -0.4095 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 2 | 2 | 0.1111 | 0.1889 | 0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 2 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 2 | 4 | 0.0000 | 0.0778 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 3 | 0 | -0.1667 | -0.1222 | -0.1000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 3 | 2 | -0.3333 | -0.3333 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 3 | 3 | 0.3333 | 0.3778 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 3 | 4 | 0.1111 | 0.0476 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 4 | 1 | -0.1111 | -0.0794 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 4 | 2 | -0.1111 | -0.3000 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 4 | 3 | 0.1667 | 0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | sampler | B0 | B5 | 4 | 4 | 0.0000 | -0.1095 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 0 | 0 | 0.0000 | -0.0556 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 0 | 1 | 0.0000 | 0.1000 | -0.1000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 0 | 3 | 0.1667 | 0.1667 | 0.1667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 1 | 0 | -0.3333 | -0.3333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 1 | 2 | 0.0000 | 0.0317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 1 | 3 | 0.0000 | 0.0333 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 1 | 4 | -0.1667 | -0.3222 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 2 | 0 | 0.0000 | -0.0222 | 0.1000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 2 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 2 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 2 | 4 | 0.0000 | 0.0317 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 3 | 1 | -0.1667 | -0.1127 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 3 | 2 | 0.0000 | -0.0238 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 3 | 3 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 3 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 4 | 1 | -0.1111 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 4 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | optimizer_and_batch_size | B0 | B6 | 4 | 4 | 0.0000 | -0.0222 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 0 | 0 | 0.0000 | -0.0238 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 0 | 1 | -0.0556 | -0.1000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 0 | 3 | -0.3333 | -0.4444 | -1.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 1 | 0 | 0.0000 | -0.0016 | -0.1000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 1 | 1 | -0.3333 | -0.3095 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 1 | 2 | 0.1667 | 0.2222 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 1 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 1 | 4 | 0.0000 | -0.0667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 2 | 0 | 0.1667 | 0.2540 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 2 | 1 | -0.1667 | -0.2667 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 2 | 2 | 0.1667 | 0.1000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 2 | 3 | 0.1667 | 0.1556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 2 | 4 | -0.1667 | -0.0889 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 3 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 3 | 1 | 0.1667 | 0.1349 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 3 | 2 | -0.1667 | -0.2778 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 3 | 3 | -0.1667 | -0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 3 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 4 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 4 | 2 | 0.1111 | 0.1222 | 0.1000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 4 | 3 | 0.3333 | 0.3667 | 0.5000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | optimizer_and_batch_size | B0 | B6 | 4 | 4 | 0.0000 | -0.1095 | -0.4000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 0 | 0 | -0.1667 | -0.2000 | -0.5000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 0 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 0 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 0 | 3 | 0.1667 | 0.1556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 1 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 1 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 1 | 3 | 0.1667 | 0.1667 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 1 | 4 | -0.1667 | -0.2667 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 2 | 0 | 0.0000 | -0.0222 | 0.1000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 2 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 2 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 2 | 3 | -0.1667 | -0.2778 | -0.6667 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 2 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 3 | 0 | 0.1111 | 0.1444 | 0.3000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 3 | 2 | 0.0000 | -0.0238 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 3 | 3 | -0.1667 | -0.2540 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 3 | 4 | 0.3333 | 0.3690 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 4 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 4 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| CompactCNN1D | primary_aggregation | B0 | B7 | 4 | 4 | 0.3333 | 0.3111 | 0.6667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 0 | 0 | -0.1667 | -0.1349 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 0 | 1 | 0.0000 | 0.0111 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 0 | 2 | 0.1667 | 0.1556 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 0 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 0 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 1 | 0 | 0.1111 | 0.1151 | 0.1667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 1 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 1 | 2 | -0.1667 | -0.1000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 1 | 3 | 0.1667 | 0.0889 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 1 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 2 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 2 | 1 | -0.1667 | -0.1667 | -0.1667 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 2 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 2 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 2 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 3 | 0 | 0.1111 | 0.1444 | 0.3000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 3 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 3 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 3 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 4 | 0 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 4 | 1 | 0.0556 | -0.0016 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 4 | 2 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 4 | 3 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |
| InceptionTimeFull | primary_aggregation | B0 | B7 | 4 | 4 | 0.0000 | 0.0000 | 0.0000 | True | descriptive_only_no_ci_no_significance |

## Stage 3 centered-star execution order

Absolute scheduling rows only; neighbouring execution rows are not ablation contrasts.

| Execution order | Model | Profile | Factor | Native endpoint | Native BA | Native Macro-F1 | Native worst-class F1 | Scheduling transition | Transition is ablation |
|---|---|---|---|---|---|---|---|---|---|
| 1 | CompactCNN1D | B0 | baseline | window_balanced_to_participant | 0.5380 | 0.5293 | 0.4440 | start | False |
| 2 | InceptionTimeFull | B0 | baseline | window_balanced_to_participant | 0.5676 | 0.5725 | 0.4663 | compact_cnn__B0_star_fixed10->inception_full__B0_star_fixed10 | False |
| 3 | CompactCNN1D | B1 | sampling_rate | window_balanced_to_participant | 0.4491 | 0.4383 | 0.2938 | inception_full__B0_star_fixed10->compact_cnn__B1_star_fixed10 | False |
| 4 | InceptionTimeFull | B1 | sampling_rate | window_balanced_to_participant | 0.5028 | 0.5045 | 0.4336 | compact_cnn__B1_star_fixed10->inception_full__B1_star_fixed10 | False |
| 5 | CompactCNN1D | B2 | window_plan | window_balanced_to_participant | 0.5685 | 0.5718 | 0.4427 | inception_full__B1_star_fixed10->compact_cnn__B2_star_fixed10 | False |
| 6 | InceptionTimeFull | B2 | window_plan | window_balanced_to_participant | 0.5093 | 0.5180 | 0.4161 | compact_cnn__B2_star_fixed10->inception_full__B2_star_fixed10 | False |
| 7 | CompactCNN1D | B3 | imu_preprocessing | window_balanced_to_participant | 0.4722 | 0.4764 | 0.3707 | inception_full__B2_star_fixed10->compact_cnn__B3_star_fixed10 | False |
| 8 | InceptionTimeFull | B3 | imu_preprocessing | window_balanced_to_participant | 0.4972 | 0.5034 | 0.4030 | compact_cnn__B3_star_fixed10->inception_full__B3_star_fixed10 | False |
| 9 | CompactCNN1D | B4 | normalization | window_balanced_to_participant | 0.4333 | 0.4279 | 0.3433 | inception_full__B3_star_fixed10->compact_cnn__B4_star_fixed10 | False |
| 10 | InceptionTimeFull | B4 | normalization | window_balanced_to_participant | 0.4296 | 0.4308 | 0.3467 | compact_cnn__B4_star_fixed10->inception_full__B4_star_fixed10 | False |
| 11 | CompactCNN1D | B5 | sampler | window_balanced_to_participant | 0.5231 | 0.5338 | 0.4491 | inception_full__B4_star_fixed10->compact_cnn__B5_star_fixed10 | False |
| 12 | InceptionTimeFull | B5 | sampler | window_balanced_to_participant | 0.5231 | 0.5209 | 0.4308 | compact_cnn__B5_star_fixed10->inception_full__B5_star_fixed10 | False |
| 13 | CompactCNN1D | B6 | optimizer_and_batch_size | window_balanced_to_participant | 0.4917 | 0.4872 | 0.4108 | inception_full__B5_star_fixed10->compact_cnn__B6_star_fixed10 | False |
| 14 | InceptionTimeFull | B6 | optimizer_and_batch_size | window_balanced_to_participant | 0.5602 | 0.5623 | 0.4916 | compact_cnn__B6_star_fixed10->inception_full__B6_star_fixed10 | False |
| 15 | CompactCNN1D | B7 | primary_aggregation | line_b_equal_role_families | 0.5472 | 0.5403 | 0.4783 | inception_full__B6_star_fixed10->compact_cnn__B7_star_fixed10 | False |
| 16 | InceptionTimeFull | B7 | primary_aggregation | line_b_equal_role_families | 0.5731 | 0.5794 | 0.4699 | compact_cnn__B7_star_fixed10->inception_full__B7_star_fixed10 | False |

## Aggregation sensitivity from the same file-level OOF

The declared-source row reproduces the aggregation used by the fitted model and, when eligible, the primary leaderboard. The other row reaggregates the same held-out file probabilities post hoc. It is not a separately retrained Line A/Line B experiment and is not selection evidence.

| Case | Aggregation view | Role | Mean BA | Mean Macro-F1 | Line A − Line B BA | Line A − Line B Macro-F1 | Worst recall | Worst F1 | ECE | Repeats | Retained participant OOF n | All participant units n | Dropped participant units n | All file OOF n | Dropped files n | Source replay | Primary ranking eligible |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | line_a_equal_files | declared_source_line | 0.5380 | 0.5293 | -0.0093 | -0.0109 | 0.4000 | 0.4440 | 0.3023 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b0_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5472 | 0.5403 | -0.0093 | -0.0109 | 0.4167 | 0.4783 | 0.2928 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b1_star_fixed10 | line_a_equal_files | declared_source_line | 0.4491 | 0.4388 | -0.0204 | -0.0225 | 0.2750 | 0.2990 | 0.2978 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b1_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4694 | 0.4613 | -0.0204 | -0.0225 | 0.2750 | 0.3513 | 0.3334 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b2_star_fixed10 | line_a_equal_files | declared_source_line | 0.5685 | 0.5718 | 0.0491 | 0.0517 | 0.3944 | 0.4427 | 0.2015 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b2_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5194 | 0.5200 | 0.0491 | 0.0517 | 0.3528 | 0.4219 | 0.2359 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b3_star_fixed10 | line_a_equal_files | declared_source_line | 0.4722 | 0.4764 | 0.0083 | 0.0111 | 0.3167 | 0.3707 | 0.3360 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b3_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4639 | 0.4653 | 0.0083 | 0.0111 | 0.3194 | 0.3543 | 0.3173 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b4_star_fixed10 | line_a_equal_files | declared_source_line | 0.4333 | 0.4279 | 0.0130 | 0.0057 | 0.3083 | 0.3433 | 0.3566 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b4_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4204 | 0.4222 | 0.0130 | 0.0057 | 0.3194 | 0.3305 | 0.3605 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b5_star_fixed10 | line_a_equal_files | declared_source_line | 0.5231 | 0.5338 | 0.0370 | 0.0385 | 0.4306 | 0.4491 | 0.2935 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b5_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4861 | 0.4953 | 0.0370 | 0.0385 | 0.3972 | 0.4330 | 0.3493 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b6_star_fixed10 | line_a_equal_files | declared_source_line | 0.4917 | 0.4872 | -0.0046 | -0.0037 | 0.3806 | 0.4108 | 0.2801 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b6_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4963 | 0.4909 | -0.0046 | -0.0037 | 0.3972 | 0.4211 | 0.2918 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| compact_cnn__b7_star_fixed10 | line_a_equal_files | declared_source_line | 0.5380 | 0.5293 | -0.0093 | -0.0109 | 0.4000 | 0.4440 | 0.3023 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| compact_cnn__b7_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5472 | 0.5403 | -0.0093 | -0.0109 | 0.4167 | 0.4783 | 0.2928 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b0_star_fixed10 | line_a_equal_files | declared_source_line | 0.5676 | 0.5725 | -0.0056 | -0.0069 | 0.4556 | 0.4663 | 0.2626 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b0_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5731 | 0.5794 | -0.0056 | -0.0069 | 0.4667 | 0.4699 | 0.2648 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b1_star_fixed10 | line_a_equal_files | declared_source_line | 0.5028 | 0.5048 | 0.0093 | 0.0067 | 0.3972 | 0.4336 | 0.2467 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b1_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4935 | 0.4981 | 0.0093 | 0.0067 | 0.4111 | 0.4615 | 0.2908 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b2_star_fixed10 | line_a_equal_files | declared_source_line | 0.5093 | 0.5182 | 0.0093 | 0.0090 | 0.3667 | 0.4162 | 0.2616 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b2_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5000 | 0.5092 | 0.0093 | 0.0090 | 0.3667 | 0.4102 | 0.2855 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b3_star_fixed10 | line_a_equal_files | declared_source_line | 0.4972 | 0.5034 | -0.0148 | -0.0140 | 0.3778 | 0.4030 | 0.2762 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b3_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5120 | 0.5174 | -0.0148 | -0.0140 | 0.3889 | 0.4090 | 0.2968 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b4_star_fixed10 | line_a_equal_files | declared_source_line | 0.4296 | 0.4308 | 0.0139 | 0.0164 | 0.3194 | 0.3467 | 0.3798 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b4_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.4157 | 0.4145 | 0.0139 | 0.0164 | 0.3028 | 0.3175 | 0.3692 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b5_star_fixed10 | line_a_equal_files | declared_source_line | 0.5176 | 0.5152 | 0.0065 | 0.0076 | 0.4056 | 0.4219 | 0.2965 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b5_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5111 | 0.5076 | 0.0065 | 0.0076 | 0.3833 | 0.4071 | 0.3279 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b6_star_fixed10 | line_a_equal_files | declared_source_line | 0.5602 | 0.5623 | 0.0231 | 0.0227 | 0.4778 | 0.4916 | 0.2520 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b6_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5370 | 0.5397 | 0.0231 | 0.0227 | 0.4250 | 0.4666 | 0.2456 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |
| inception_full__b7_star_fixed10 | line_a_equal_files | declared_source_line | 0.5676 | 0.5725 | -0.0056 | -0.0069 | 0.4556 | 0.4663 | 0.2626 | 5 | 145 | 145 | 0 | 725 | 0 | exact_match_persisted_subject_oof_atol_1e-7_with_dropped_coverage | True |
| inception_full__b7_star_fixed10 | line_b_equal_role_families | posthoc_aggregation_only | 0.5731 | 0.5794 | -0.0056 | -0.0069 | 0.4667 | 0.4699 | 0.2648 | 5 | 145 | 145 | 0 | 725 | 0 | not_applicable_posthoc_view | False |

## Parallel window/file/role-balanced participant views

All three rows reuse the same fitted held-out OOF probabilities; they are not three training runs. `window_balanced_to_participant` gives every retained window equal report weight, Line A gives every file equal weight after window→file, and Line B gives every canonical role family equal weight after window→file→role. Only the declared training aggregation may support the primary leaderboard; the other views are post-hoc sensitivity plots.

| Case | Aggregation view | Evidence role | Mean BA | Mean Macro-F1 | Worst recall | Worst F1 | Repeats | Participant OOF n | Primary ranking eligible |
|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5380 | 0.5293 | 0.4000 | 0.4440 | 5 | 145 | False |
| compact_cnn__b0_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5380 | 0.5293 | 0.4000 | 0.4440 | 5 | 145 | True |
| compact_cnn__b0_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5472 | 0.5403 | 0.4167 | 0.4783 | 5 | 145 | False |
| compact_cnn__b1_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4491 | 0.4383 | 0.2750 | 0.2938 | 5 | 145 | False |
| compact_cnn__b1_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4491 | 0.4388 | 0.2750 | 0.2990 | 5 | 145 | True |
| compact_cnn__b1_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4694 | 0.4613 | 0.2750 | 0.3513 | 5 | 145 | False |
| compact_cnn__b2_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5685 | 0.5718 | 0.3944 | 0.4427 | 5 | 145 | False |
| compact_cnn__b2_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5685 | 0.5718 | 0.3944 | 0.4427 | 5 | 145 | True |
| compact_cnn__b2_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5194 | 0.5200 | 0.3528 | 0.4219 | 5 | 145 | False |
| compact_cnn__b3_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4722 | 0.4764 | 0.3167 | 0.3707 | 5 | 145 | False |
| compact_cnn__b3_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4722 | 0.4764 | 0.3167 | 0.3707 | 5 | 145 | True |
| compact_cnn__b3_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4639 | 0.4653 | 0.3194 | 0.3543 | 5 | 145 | False |
| compact_cnn__b4_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4333 | 0.4279 | 0.3083 | 0.3433 | 5 | 145 | False |
| compact_cnn__b4_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4333 | 0.4279 | 0.3083 | 0.3433 | 5 | 145 | True |
| compact_cnn__b4_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4204 | 0.4222 | 0.3194 | 0.3305 | 5 | 145 | False |
| compact_cnn__b5_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5231 | 0.5338 | 0.4306 | 0.4491 | 5 | 145 | False |
| compact_cnn__b5_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5231 | 0.5338 | 0.4306 | 0.4491 | 5 | 145 | True |
| compact_cnn__b5_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4861 | 0.4953 | 0.3972 | 0.4330 | 5 | 145 | False |
| compact_cnn__b6_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4917 | 0.4872 | 0.3806 | 0.4108 | 5 | 145 | False |
| compact_cnn__b6_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4917 | 0.4872 | 0.3806 | 0.4108 | 5 | 145 | True |
| compact_cnn__b6_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4963 | 0.4909 | 0.3972 | 0.4211 | 5 | 145 | False |
| compact_cnn__b7_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5380 | 0.5293 | 0.4000 | 0.4440 | 5 | 145 | False |
| compact_cnn__b7_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5380 | 0.5293 | 0.4000 | 0.4440 | 5 | 145 | True |
| compact_cnn__b7_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5472 | 0.5403 | 0.4167 | 0.4783 | 5 | 145 | False |
| inception_full__b0_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5676 | 0.5725 | 0.4556 | 0.4663 | 5 | 145 | False |
| inception_full__b0_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5676 | 0.5725 | 0.4556 | 0.4663 | 5 | 145 | True |
| inception_full__b0_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5731 | 0.5794 | 0.4667 | 0.4699 | 5 | 145 | False |
| inception_full__b1_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5028 | 0.5045 | 0.3972 | 0.4336 | 5 | 145 | False |
| inception_full__b1_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5028 | 0.5048 | 0.3972 | 0.4336 | 5 | 145 | True |
| inception_full__b1_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4935 | 0.4981 | 0.4111 | 0.4615 | 5 | 145 | False |
| inception_full__b2_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5093 | 0.5180 | 0.3667 | 0.4161 | 5 | 145 | False |
| inception_full__b2_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5093 | 0.5182 | 0.3667 | 0.4162 | 5 | 145 | True |
| inception_full__b2_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5000 | 0.5092 | 0.3667 | 0.4102 | 5 | 145 | False |
| inception_full__b3_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4972 | 0.5034 | 0.3778 | 0.4030 | 5 | 145 | False |
| inception_full__b3_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4972 | 0.5034 | 0.3778 | 0.4030 | 5 | 145 | True |
| inception_full__b3_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5120 | 0.5174 | 0.3889 | 0.4090 | 5 | 145 | False |
| inception_full__b4_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.4296 | 0.4308 | 0.3194 | 0.3467 | 5 | 145 | False |
| inception_full__b4_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.4296 | 0.4308 | 0.3194 | 0.3467 | 5 | 145 | True |
| inception_full__b4_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.4157 | 0.4145 | 0.3028 | 0.3175 | 5 | 145 | False |
| inception_full__b5_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5231 | 0.5209 | 0.4111 | 0.4308 | 5 | 145 | False |
| inception_full__b5_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5176 | 0.5152 | 0.4056 | 0.4219 | 5 | 145 | True |
| inception_full__b5_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5111 | 0.5076 | 0.3833 | 0.4071 | 5 | 145 | False |
| inception_full__b6_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5602 | 0.5623 | 0.4778 | 0.4916 | 5 | 145 | False |
| inception_full__b6_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5602 | 0.5623 | 0.4778 | 0.4916 | 5 | 145 | True |
| inception_full__b6_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5370 | 0.5397 | 0.4250 | 0.4666 | 5 | 145 | False |
| inception_full__b7_star_fixed10 | window_balanced_to_participant | posthoc_same_oof_sensitivity_only | 0.5676 | 0.5725 | 0.4556 | 0.4663 | 5 | 145 | False |
| inception_full__b7_star_fixed10 | line_a_equal_files | declared_training_aggregation | 0.5676 | 0.5725 | 0.4556 | 0.4663 | 5 | 145 | True |
| inception_full__b7_star_fixed10 | line_b_equal_role_families | posthoc_same_oof_sensitivity_only | 0.5731 | 0.5794 | 0.4667 | 0.4699 | 5 | 145 | False |

<details><summary>Hierarchy coverage: B/R1–R4 window/file views and B/R role-balanced view</summary>

| Case | Repeat | Level | View | Group | OOF units | Retained units | Participants |
|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| compact_cnn__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| compact_cnn__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| compact_cnn__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| compact_cnn__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| compact_cnn__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| compact_cnn__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b0_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b0_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b0_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b0_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b0_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b0_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b0_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b0_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b0_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b0_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b0_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b1_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b1_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b1_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b1_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b1_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b1_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b1_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b1_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b1_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b1_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b1_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b2_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 0 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| inception_full__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| inception_full__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| inception_full__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| inception_full__b2_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| inception_full__b2_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 1 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| inception_full__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| inception_full__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| inception_full__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| inception_full__b2_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| inception_full__b2_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 2 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| inception_full__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| inception_full__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| inception_full__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| inception_full__b2_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| inception_full__b2_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 3 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| inception_full__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| inception_full__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| inception_full__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| inception_full__b2_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| inception_full__b2_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b2_star_fixed10 | 4 | window | window_balanced_to_participant | B | 3155 | 3155 | 29 |
| inception_full__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 3110 | 3110 | 29 |
| inception_full__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 3115 | 3115 | 29 |
| inception_full__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 3133 | 3133 | 29 |
| inception_full__b2_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 3144 | 3144 | 29 |
| inception_full__b3_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b3_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b3_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b3_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b3_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b3_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b3_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b3_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b3_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b3_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b3_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b4_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b4_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b4_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b4_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b4_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b4_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b4_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b4_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b4_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b4_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b4_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b5_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b5_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b5_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b5_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b5_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b5_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b5_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b5_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b5_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b5_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b5_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b6_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b6_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b6_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b6_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b6_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b6_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b6_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b6_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b6_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b6_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b6_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b7_star_fixed10 | 0 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 0 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b7_star_fixed10 | 0 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b7_star_fixed10 | 1 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 1 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b7_star_fixed10 | 1 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b7_star_fixed10 | 2 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 2 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b7_star_fixed10 | 2 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b7_star_fixed10 | 3 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 3 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b7_star_fixed10 | 3 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |
| inception_full__b7_star_fixed10 | 4 | file | line_a_equal_files | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | file | line_a_equal_files | R1 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | file | line_a_equal_files | R2 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | file | line_a_equal_files | R3 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | file | line_a_equal_files | R4 | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | participant | participant_balanced_endpoint | participant | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | role | line_b_equal_role_families | B | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | role | line_b_equal_role_families | R | 29 | 29 | 29 |
| inception_full__b7_star_fixed10 | 4 | window | window_balanced_to_participant | B | 2560 | 2560 | 29 |
| inception_full__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R1 | 2525 | 2525 | 29 |
| inception_full__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R2 | 2528 | 2528 | 29 |
| inception_full__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R3 | 2544 | 2544 | 29 |
| inception_full__b7_star_fixed10 | 4 | window | window_balanced_to_participant | R4 | 2549 | 2549 | 29 |

</details>

## Worst-class F1 stability review

This secondary view reorders complete cases by abstention-aware worst-class F1, then abstention-aware repeat variability. Conditional retained-only values remain visible for comparison.

| Stability rank | Aware-BA rank | Case | Aware worst F1 | Aware worst recall | Aware BA, mean ± SD (%) | Worst F1 | Worst recall | Conditional BA, mean ± SD (%) |
|---|---|---|---|---|---|---|---|---|
| 1 | 1 | compact_cnn__b2_star_fixed10 | 0.5376 | 0.5500 | 56.9 ± 5.3 | 0.5376 | 0.5500 | 56.9 ± 5.3 |
| 2 | 4 | inception_full__b6_star_fixed10 | 0.4959 | 0.5000 | 56.0 ± 3.4 | 0.4959 | 0.5000 | 56.0 ± 3.4 |
| 3 | 7 | compact_cnn__b5_star_fixed10 | 0.4923 | 0.5111 | 52.3 ± 3.2 | 0.4923 | 0.5111 | 52.3 ± 3.2 |
| 4 | 2 | inception_full__b0_star_fixed10 | 0.4918 | 0.5000 | 56.8 ± 2.1 | 0.4918 | 0.5000 | 56.8 ± 2.1 |
| 5 | 3 | inception_full__b7_star_fixed10 | 0.4918 | 0.5000 | 56.8 ± 2.1 | 0.4918 | 0.5000 | 56.8 ± 2.1 |
| 6 | 10 | inception_full__b1_star_fixed10 | 0.4857 | 0.4250 | 50.3 ± 2.0 | 0.4857 | 0.4250 | 50.3 ± 2.0 |
| 7 | 5 | compact_cnn__b0_star_fixed10 | 0.4828 | 0.4667 | 53.8 ± 6.1 | 0.4828 | 0.4667 | 53.8 ± 6.1 |
| 8 | 6 | compact_cnn__b7_star_fixed10 | 0.4828 | 0.4667 | 53.8 ± 6.1 | 0.4828 | 0.4667 | 53.8 ± 6.1 |
| 9 | 11 | inception_full__b3_star_fixed10 | 0.4640 | 0.4750 | 49.7 ± 5.3 | 0.4640 | 0.4750 | 49.7 ± 5.3 |
| 10 | 9 | inception_full__b2_star_fixed10 | 0.4545 | 0.4444 | 50.9 ± 4.7 | 0.4545 | 0.4444 | 50.9 ± 4.7 |

## Incomplete cases excluded from ranking

N/A — no rows were available.

## Deployment measurements (separate from predictive ranking)

| Case | Parameters | Inference cost | Status | Reported note |
|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b0_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b1_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b1_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b2_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b2_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b3_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b3_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b4_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b4_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b5_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b5_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b6_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b6_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| compact_cnn__b7_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |
| inception_full__b7_star_fixed10 | N/A | {} | N/A_pending_hardware_evidence_V2_026 |  |

## Route × role coverage and feature availability

This table separates direct and processed rate paths, retained coverage, unavailable predictors, and reducer failures for each role/route state.

| Case | Role | Quality tier | Motion | Route state | Signal route | Retained coverage | Abstention | Abstention reasons | Direct | Processed | Unavailable predictors | Denoiser attempts | Denoiser successes | Reducer failures |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b0_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b0_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b0_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b0_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b1_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b1_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b1_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b1_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b1_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b2_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b2_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b2_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b2_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b2_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b3_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b3_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b3_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b3_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b3_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b4_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b4_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b4_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b4_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b4_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b5_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b5_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b5_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b5_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b5_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b6_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b6_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b6_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b6_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b6_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b7_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b7_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b7_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b7_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| compact_cnn__b7_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b0_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b0_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b0_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b0_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b0_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b1_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b1_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b1_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b1_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b1_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b2_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b2_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b2_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b2_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b2_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b3_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b3_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b3_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b3_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b3_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b4_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b4_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b4_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b4_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b4_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b5_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b5_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b5_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b5_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b5_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b6_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b6_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b6_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b6_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b6_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b7_star_fixed10 | B | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b7_star_fixed10 | R1 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b7_star_fixed10 | R2 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b7_star_fixed10 | R3 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |
| inception_full__b7_star_fixed10 | R4 | not_reported | not_reported | direct_fresh_raw_csv | direct_x_filter | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 |

## SQI state, score, and coverage provenance by each route

Direct and post-denoiser coverage are reported separately so the configured minimum-coverage decision remains auditable.

| Case | Role | Tier | Direct Q_rate state | Mean direct Q_rate | Direct Q_rate coverage | Direct Q_morph state | Mean direct Q_morph | Direct Q_morph coverage | Post Q_rate state | Mean post Q_rate | Post Q_rate coverage |
|---|---|---|---|---|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b0_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b0_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b0_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b0_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b1_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b1_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b1_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b1_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b1_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b2_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b2_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b2_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b2_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b2_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b3_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b3_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b3_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b3_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b3_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b4_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b4_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b4_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b4_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b4_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b5_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b5_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b5_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b5_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b5_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b6_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b6_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b6_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b6_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b6_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b7_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b7_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b7_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b7_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| compact_cnn__b7_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b0_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b0_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b0_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b0_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b0_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b1_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b1_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b1_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b1_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b1_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b2_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b2_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b2_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b2_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b2_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b3_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b3_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b3_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b3_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b3_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b4_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b4_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b4_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b4_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b4_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b5_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b5_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b5_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b5_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b5_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b6_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b6_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b6_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b6_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b6_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b7_star_fixed10 | B | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b7_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b7_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b7_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |
| inception_full__b7_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | not_reported | N/A | N/A | not_reported | N/A | N/A |

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
| compact_cnn__b0_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b0_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b0_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b0_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b0_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b1_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b1_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b1_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b1_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b1_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b2_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b2_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b2_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b2_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b2_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b3_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b3_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b3_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b3_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b3_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b4_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b4_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b4_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b4_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b4_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b5_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b5_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b5_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b5_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b5_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b6_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b6_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b6_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b6_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b6_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b7_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b7_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b7_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b7_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| compact_cnn__b7_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b0_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b0_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b0_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b0_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b0_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b1_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b1_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b1_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b1_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b1_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b2_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b2_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b2_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b2_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b2_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b3_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b3_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b3_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b3_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b3_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b4_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b4_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b4_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b4_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b4_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b5_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b5_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b5_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b5_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b5_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b6_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b6_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b6_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b6_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b6_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b7_star_fixed10 | B | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b7_star_fixed10 | R1 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b7_star_fixed10 | R2 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b7_star_fixed10 | R3 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |
| inception_full__b7_star_fixed10 | R4 | not_reported | not_reported | N/A | N/A | N/A | not_reported | not_reported | not_reported | not_reported | N/A | not_reported | not_reported |

## Quality-component distributions by route and role

N/A — no rows were available.

## Classification score, t-SNE, and ROC–AUC diagnostics

Every classifier with persisted participant OOF probabilities is represented in the three paired figure modules. t-SNE embeds the prediction-probability vector, not hidden features. Multiclass frailty decisions use argmax and therefore have no single scalar threshold.

| Classifier | Score/threshold | Prediction t-SNE | ROC–AUC curve | OOF rows | t-SNE points | ROC points |
|---|---|---|---|---|---|---|
| compact_cnn__b0_star_fixed10 | available | available | available | 145 | 145 | 383 |
| inception_full__b0_star_fixed10 | available | available | available | 145 | 145 | 354 |
| compact_cnn__b1_star_fixed10 | available | available | available | 145 | 145 | 399 |
| inception_full__b1_star_fixed10 | available | available | available | 145 | 145 | 395 |
| compact_cnn__b2_star_fixed10 | available | available | available | 145 | 145 | 369 |
| inception_full__b2_star_fixed10 | available | available | available | 145 | 145 | 384 |
| compact_cnn__b3_star_fixed10 | available | available | available | 145 | 145 | 382 |
| inception_full__b3_star_fixed10 | available | available | available | 145 | 145 | 374 |
| compact_cnn__b4_star_fixed10 | available | available | available | 145 | 145 | 386 |
| inception_full__b4_star_fixed10 | available | available | available | 145 | 145 | 387 |
| compact_cnn__b5_star_fixed10 | available | available | available | 145 | 145 | 389 |
| inception_full__b5_star_fixed10 | available | available | available | 145 | 145 | 387 |
| compact_cnn__b6_star_fixed10 | available | available | available | 145 | 145 | 382 |
| inception_full__b6_star_fixed10 | available | available | available | 145 | 145 | 381 |
| compact_cnn__b7_star_fixed10 | available | available | available | 145 | 145 | 383 |
| inception_full__b7_star_fixed10 | available | available | available | 145 | 145 | 354 |

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
| stage3_star_model_deltas | generated | figures/stage3_star_model_deltas.png |  |
| stage3_star_fold_delta_heatmap | generated | figures/stage3_star_fold_delta_heatmap.png |  |

## Limitations and N/A items

- Classification t-SNE is a report-only embedding of persisted OOF prediction-probability vectors, not a hidden-feature embedding and not evidence of separability in the model representation space.
- generic single-reference paired deltas are disabled for the two-model centered star; use only the fourteen same-model Stage-3 star contrasts
- Stage-3 has 5 repeat(s) and 5 grouped folds; fold deltas are descriptive only, with no per-fold independence or significance claim.
- Each model reuses one B0 across seven correlated contrasts.
- B7 compares the B0 W endpoint with Line B; W/A/B same-view deltas remain sensitivity-only.
- B0/B7 OOF uses sorted row identities and zero probability tolerance, not Parquet bytes or row order.
- The 16-case two-model B0-B7 design contains 400 repeat/fold cells.
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

- [tables/stage3_star_inception_comparison.csv](tables/stage3_star_inception_comparison.csv)
- [tables/stage3_star_cnn_comparison.csv](tables/stage3_star_cnn_comparison.csv)
- [tables/stage3_star_model_comparison.csv](tables/stage3_star_model_comparison.csv)
- [tables/stage3_star_absolute.csv](tables/stage3_star_absolute.csv)
- [tables/stage3_star_contrasts.csv](tables/stage3_star_contrasts.csv)
- [tables/stage3_star_fold_contrasts.csv](tables/stage3_star_fold_contrasts.csv)
- [tables/stage3_star_execution.csv](tables/stage3_star_execution.csv)
