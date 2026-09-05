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

| Cases / phases | Component role | Model / module | State | Input data (values and paths; no hashes) | Detailed fixed parameters | Algorithm and kernel (≤300 chars) | Reporter profile | Model reporter extension | Algorithm / literature source |
|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | aggregation | line_b_equal_role_families | enabled | {"input_data":"held-out window/file probabilities","roles":["B","R1","R2","R3","R4"]} | {"balance_line":"line_b_equal_role_families","direct_all_window_participant_mean":false,"file_to_role":"ordinary_mean","hierarchy":["window","file","role","participant"],"missing_role_policy":"mean_available_roles","quality_weight_levels":[],"quality_weight_source":"none","quality_weighting":false,"role_to_participant":"ordinary_mean","window_to_file":"ordinary_mean"} | executable line_b_equal_role_families hierarchy ending in participant-balanced output | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.training.aggregation.aggregate_hierarchy; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | classifier | InceptionTimeFull | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"architecture_parameters":{"bottleneck_channels":32,"branch_count":4,"classifier_dropout":0.2,"depth":6,"dilation":1,"global_pooling":"mask_aware_global_average","kernel_sizes":[39,19,9],"model_id":"inception_full","n_classes":3,"out_channels":32,"pool_size":3,"representation_mode":"raw","residual_interval":3,"variant":"full"},"dilation":1,"dropout":0.2,"ensemble_size":1,"input_channel_order":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"input_channels":8,"input_channels_resolution":"canonical_frailty_raw_8","kernel_sizes":[39,19,9],"mask_aware_pooling":true,"model_id":"InceptionTimeFull","n_classes":3,"seed_policy":"outer_cv_repeat_seed_equals_split_seed","variant":"single_network"} | Six-block, single-network InceptionTime adaptation with bottleneck and parallel fixed-sample kernels; not the paper's five-member ensemble. | multiclass_participant_oof_v1 | inceptiontime_single_network_model_v1 | Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | dataset_adapter | frailty3_m2_20260815_a054800abda272f6 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"allow_qc_excluded_records":false,"channel_order":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"class_id_order":[0,1,2],"class_name_order":["Pre-Frail","Robust/Non-Frail","Young"],"expected_participant_count":29,"expected_record_count":261,"manifest_version":"internal_records_v2","path":"manifests/internal_records_v2.csv","source_dataset_id":"frailty3_m2_20260815_a054800abda272f6"} | Persisted dataset_adapter contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: dataset_adapter; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | denoiser | identity | identity_or_disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["filtered RED/IR","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"declared_reducer_version":"identity_v1","degraded_policy":"drop","denoiser_enabled":false,"failure_action":"no_result_no_fallback","reducer":"identity","resolved_parameters":{},"runtime_reducer_version":"identity_exact_v1"} | 逐样本复制双波长 PPG，不估计或抑制伪影；内核：恒等映射与同时间网格校验，作为未去噪直接对照。 | audit_provenance_v1 | not_applicable | N/A — component was not executed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | evaluation | balanced_accuracy | enabled | {"class_order":["Pre-Frail","Robust/Non-Frail","Young"],"input_data":"held-out participant predictions and frailty labels"} | {"calibration_metrics":["multiclass_brier","expected_calibration_error"],"confidence_interval":"participant_cluster_bootstrap_two_sided_95","independent_test_available":false,"metric_prefix":"oof_validation_","metrics":["balanced_accuracy","macro_f1","per_class_precision_recall_f1","worst_class_recall","worst_class_f1","confusion_matrix","coverage"],"paired_delta_key":["repeat_index","fold_index","participant_id"],"primary_metric":"balanced_accuracy","rank_incomplete_configs":false,"ranking":{"automatic_final_selection":false,"manual_multiple_final_versions_allowed":true,"max_qualified_per_comparison_group":10,"preserve_ablation_provenance":true,"sort_key":"participant_level_mean_balanced_accuracy"},"statistics":{"affects_automatic_selection":false,"bootstrap_replicates":10000,"cluster_unit":"participant_with_all_five_repeat_oof_predictions","confidence_interval":"two_sided_95_percentile","lcb95_metrics":["participant_level_mean_balanced_accuracy","participant_level_mean_macro_f1"],"lcb95_percentile":2.5,"multiplicity_correction":"holm_within_comparison_family","paired_exchange_unit":"participant","paired_permutation_replicates":100000,"seed":42},"unit":"participant"} | Persisted evaluation contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: evaluation; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | feature_extractor | feature_vector_282_v3 | auxiliary_not_classifier_input | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","engineering_window":{"cap_fraction_per_file":null,"cap_per_file":null,"end_alignment":"left_start_regular_grid","hop_s":2.0,"length_s":10.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"input_views":["x_analysis/x_native","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"enabled_groups":["ppi_basic_rate","hrv_time_domain","hrv_spectral","hrv_nonlinear","morphology","dual_optical","engineering_summary"],"engineering_sequence_schema":"engineering_10s_hop2s_thesis_115_v3","file_aggregation":["mean","population_sd"],"file_vector_schema":"feature_vector_282_v3","matrix_k":150,"matrix_schema":"ordered_feature_matrix_d115_by_150_engineering_v4","missing_physiology_encoding":"nan_and_validity_false","prv_library_comparison_scope":"fixed_ppi_vectors_only_no_classifier","prv_primary_backend":"local_manual","rate_prv_min_duration_s":8.0,"rate_prv_min_peaks":5,"registry_id":"feature_vector_282_v3","sample_entropy":{"m":2,"min_intervals":200,"r_sd_fraction":0.2},"spectral_bands_hz":{"hf":[0.15,0.4],"lf":[0.04,0.15],"vlf":[0.003,0.04]},"spectral_prv_min_coverage":0.8,"spectral_prv_min_duration_s":300.0,"spectral_prv_min_intervals":200,"tachogram_fs_hz":4.0,"technical_metadata_allowed":false,"time_prv_min_coverage":0.8,"time_prv_min_duration_s":60.0,"time_prv_min_intervals":30} | Persisted feature_extractor contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: feature_extractor; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | imu_preprocessing | profile_a_lowpass_0p3hz | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["AX","AY","AZ","GX","GY","GZ"],"manifest_path":"manifests/internal_records_v2.csv","output_view":"processed_imu_physical","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"calibration_start_s":5.0,"calibration_stop_s":100.0,"comparison_method":"calibrated_roll_pitch_ekf","failure_action":"fail_closed","gravity_filter_order":4,"gravity_lowpass_hz":0.3,"gravity_method":"profile_a_lowpass_0p3hz","gravity_mps2":9.81,"initialization":"same_participant_static_calibration","output_units":{"acceleration":"m/s^2","gyroscope":"rad/s","jerk":"m/s^3"},"required_axes":6,"sensor_filter_order":3,"sensor_lowpass_acc_hz":20.0,"sensor_lowpass_gyro_hz":40.0} | executable gravity-separation profile with profile-specific numerical parameters and no silent fallback | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | motion_detector | formal_local_supervised_motion_detector_v2 | disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_view":"RED/IR + processed physical A_dyn/GX/GY/GZ","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":64,"device":"cuda","enabled":false,"evidence_path":null,"threshold_source":"bundle_frozen","window_probability_aggregation":"median"} |  | audit_provenance_v1 | not_applicable | N/A — component was not executed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | peak_detector | aboy_project_v1 | enabled | {"channels":["RED","IR"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_view":"x_analysis/x_native","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"detector_id":"aboy_project_v1","failure_action":"fail_closed_no_fallback","min_observation_sec":8.0,"min_peaks":5} | Historical shared-preprocessing Aboy-family project detector. | audit_provenance_v1 | audit_provenance_v1 | Historical project adaptation; algorithm family: Aboy et al. (2005), DOI:10.1109/TBME.2005.855725 |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | ppg_preprocessing | butterworth_sos | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_channels":["RED","IR"],"input_view":"repaired native PPG","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"analysis_view":{"additional_filter":"none","direct_source":"x_filter_0p2_to_8hz","non_identity_semantics":"rate_only","non_identity_source":"aligned_x_ar"},"gap_repair":{"all_missing_channel_action":"reject_record","edge_extrapolation":false,"max_gap_samples":100,"method":"linear_inside_only"},"ppg_filter":{"family":"butterworth_sos","high_hz":8.0,"low_hz":0.2,"notch_enabled":false,"order":3,"phase":"zero_phase","short_signal_policy":"reject"}} | executable PPG filter family; order, passband, phase, notch and short-signal policy are validated config parameters rather than separate profile IDs | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | representation | raw | enabled | {"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"representation_mode":"raw","roles":["B","R1","R2","R3","R4"],"signal_view":"x_dl_all8_window_norm","source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}} | {"input_contract":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"representation_mode":"raw"} | Line A window->file->participant; Line B window->file->role_family->participant | audit_provenance_v1 | audit_provenance_v1 | Project registry implementation: ppg_frailty.representations.raw; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | signal_views_and_scaling | parallel_physical_analysis_and_dl_views | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"},"views":["processed_imu_physical","x_dl_all8_window_norm","x_analysis/x_native"]} | {"dl_resampling":{"enabled":true,"method":"polyphase_anti_alias","preserve_feature_grid_hz":400.0,"target_fs_hz":64.0},"normalization":{"clip_after_scale":[-8.0,8.0],"iqr_fallback":"standard_deviation_then_finite_one","mad_consistency_divisor":0.6744897501960817,"raw_imu":"none","raw_ppg":"per_window_robust","robust_iqr_divisor":1.349,"scale_epsilon":1e-08,"standard_ddof":0}} | Persisted signal_views_and_scaling contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: signal_views_and_scaling; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | split_registry | frailty3_future_corrected_sgkf5_v2 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","groups":"participant_id","labels":"frailty_class","manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"n_repeats":5,"n_splits":5,"path":"splits/sgkf5_repeated_grouped_5x5_v2.csv","registry_id":"frailty3_future_corrected_sgkf5_v2","runtime_recompute":false,"split_seeds":[42,10042,20042,30042,40042]} | Persisted split_registry contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: split_registry; no separate external literature source claimed |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | sqi | quality_off | disabled_control | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["x_analysis","pulse train","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"components":[],"failure_action":"fail_closed","fit_scope":"not_applied_off","flatline_duration_s":1.0,"high_quality_rule":"not_applied","long_gap_max_samples":100,"mode":"off","window_selection":{"application_scope":"outer_train_only","keep_fraction":1.0,"policy":"none","score_algorithm":"legacy_cardiac_motion_window_sqi_v1"}} |  | audit_provenance_v1 | not_applicable | N/A — component was not executed |
| b16_lr1e-3 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":16,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | AdamW optimizer with decoupled weight decay and persisted hyperparameters. | audit_provenance_v1 | audit_provenance_v1 | Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101 |
| b16_lr1e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":16,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | AdamW optimizer with decoupled weight decay and persisted hyperparameters. | audit_provenance_v1 | audit_provenance_v1 | Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101 |
| b16_lr3e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":16,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0003,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | AdamW optimizer with decoupled weight decay and persisted hyperparameters. | audit_provenance_v1 | audit_provenance_v1 | Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101 |
| b32_lr1e-3 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | AdamW optimizer with decoupled weight decay and persisted hyperparameters. | audit_provenance_v1 | audit_provenance_v1 | Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101 |
| b32_lr1e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0001,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | AdamW optimizer with decoupled weight decay and persisted hyperparameters. | audit_provenance_v1 | audit_provenance_v1 | Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101 |
| b32_lr3e-4 | trainer | adamw | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","labels":"participant frailty class","manifest_path":"manifests/internal_records_v2.csv","model_input":{"channels":["RED","IR","A_dyn_x","A_dyn_y","A_dyn_z","GX","GY","GZ"],"representation_mode":"raw","signal_view":"x_dl_all8_window_norm","window":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"}},"participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"batch_size":32,"cache_policy":"disabled","class_count_basis":"row","class_weight_beta":0.999,"class_weighting":"inverse_frequency","classifier_role_families":["B","R"],"deterministic_algorithms":true,"device":"cuda","epoch_profile":"configured_5","epoch_rule":"fixed_epoch","execution_mode":"formal","fixed_epochs":5,"focal_gamma":2.0,"gradient_clip_norm":null,"inner_grouped_folds":0,"inner_patience":0,"label_smoothing":0.0,"learning_rate":0.0003,"loss":"cross_entropy","maximum_inner_epochs":0,"n_classes":3,"num_workers":0,"optimizer":"adamw","optimizer_parameters":{"amsgrad":false,"betas":[0.9,0.999],"eps":1e-08,"maximize":false},"outer_labels_visible_to_trainer":false,"participant_window_quota":"all","refit_on_all_outer_training":true,"sampler":"exhaustive_shuffle_without_replacement","samples_per_epoch":null,"seed":42,"training_balance":"equal_role_families","weight_decay":0.0001} | AdamW optimizer with decoupled weight decay and persisted hyperparameters. | audit_provenance_v1 | audit_provenance_v1 | Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101 |
| b16_lr1e-3; b16_lr1e-4; b16_lr3e-4; b32_lr1e-3; b32_lr1e-4; b32_lr3e-4 | window_planner | window_plan_v1 | enabled | {"dataset_id":"frailty3_m2_20260815_a054800abda272f6","input_views":["x_dl_all8_window_norm","x_analysis/x_native","processed_imu_physical"],"manifest_path":"manifests/internal_records_v2.csv","participants":29,"pipeline_fs_hz":400.0,"records":261,"roles":["B","R1","R2","R3","R4"],"source_channels":["RED","IR","AX","AY","AZ","GX","GY","GZ"],"source_units":{"ACC":"g","GYRO":"deg/s","PPG":"raw_counts"}} | {"engineering":{"cap_fraction_per_file":null,"cap_per_file":null,"end_alignment":"left_start_regular_grid","hop_s":2.0,"length_s":10.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"raw_dl":{"cap_fraction_per_file":null,"cap_per_file":128,"end_alignment":"include_right_aligned_if_distinct","hop_s":2.5,"length_s":5.0,"min_valid_fraction":1.0,"padding":"none_complete_windows_only"},"shared_planner_version":"window_plan_v1"} | Persisted window_planner contract; detailed values remain in the component input and fixed-parameter fields. | audit_provenance_v1 | audit_provenance_v1 | Project component-role audit binding: window_planner; no separate external literature source claimed |

## Model/module-owned reporter contracts and literature

The complete generated methods record is in [REPORT_METHODS.md](REPORT_METHODS.md). Profiles are selected from the persisted component identities and affect presentation only—not training, predictions, thresholds, or ranking.

| Profile | Kind | Scope | Components | Required tables | Required figures | Literature | Module sources |
|---|---|---|---|---|---|---|---|
| audit_provenance_v1 | endpoint_or_module | Configuration and provenance audit | ["aggregation:line_b_equal_role_families", "dataset_adapter:frailty3_m2_20260815_a054800abda272f6", "denoiser:identity", "evaluation:balanced_accuracy", "feature_extractor:feature_vector_282_v3", "imu_preprocessing:profile_a_lowpass_0p3hz", "motion_detector:formal_local_supervised_motion_detector_v2", "peak_detector:aboy_project_v1", "ppg_preprocessing:butterworth_sos", "representation:raw", "signal_views_and_scaling:parallel_physical_analysis_and_dl_views", "split_registry:frailty3_future_corrected_sgkf5_v2", "sqi:quality_off", "trainer:adamw", "window_planner:window_plan_v1"] | ["test_components", "reproducibility_summary"] | [] | [] | ["Historical project adaptation; algorithm family: Aboy et al. (2005), DOI:10.1109/TBME.2005.855725", "Loshchilov & Hutter (2019), DOI:10.48550/arXiv.1711.05101", "N/A — component was not executed", "Project component-role audit binding: dataset_adapter; no separate external literature source claimed", "Project component-role audit binding: evaluation; no separate external literature source claimed", "Project component-role audit binding: feature_extractor; no separate external literature source claimed", "Project component-role audit binding: signal_views_and_scaling; no separate external literature source claimed", "Project component-role audit binding: split_registry; no separate external literature source claimed", "Project component-role audit binding: window_planner; no separate external literature source claimed", "Project registry implementation: ppg_frailty.representations.raw; no separate external literature source claimed", "Project registry implementation: ppg_frailty.signal.preprocess.materialize_signal_preprocessing_config; no separate external literature source claimed", "Project registry implementation: ppg_frailty.training.aggregation.aggregate_hierarchy; no separate external literature source claimed"] |
| inceptiontime_single_network_model_v1 | model_or_module_extension | InceptionTime single-network model extension | ["classifier:InceptionTimeFull"] | ["training_history_raw", "test_components"] | ["learning_curves", "top_learning_curves", "balanced_accuracy_learning_curves", "top_balanced_accuracy_learning_curves"] | [] | ["Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y"] |
| multiclass_participant_oof_v1 | endpoint_or_module | Multiclass frailty classifier | ["classifier:InceptionTimeFull"] | ["case_summary", "metric_distribution_summary", "repeat_metrics", "repeat_per_class_metrics", "per_class_metrics", "confusion_matrices", "classification_prediction_scores", "classification_prediction_tsne", "classification_roc_curves", "classification_diagnostic_status", "paired_participant_inference", "comparison_conclusions"] | ["classification_prediction_scores", "classification_prediction_tsne", "classification_roc_auc_curves", "leaderboard", "stability", "macro_f1_stability", "roc_pr_auc_stability", "per_class_metric_stability", "confusion_matrices", "confusion_matrices_row_normalized", "per_class", "calibration"] | ["Brodersen et al. (2010), balanced accuracy, DOI:10.1109/ICPR.2010.764", "Sokolova & Lapalme (2009), classification measures including F-score, DOI:10.1016/j.ipm.2009.03.002", "Fawcett (2006), ROC analysis, DOI:10.1016/j.patrec.2005.10.010", "Efron & Tibshirani (1993), An Introduction to the Bootstrap, DOI:10.1007/978-1-4899-4541-9", "Holm (1979), sequentially rejective multiple testing, DOI:10.2307/4615733"] | ["Fawaz et al. (2020), InceptionTime, DOI:10.1007/s10618-020-00710-y"] |

## Comprehensive comparison and confidence-qualified conclusion

P values are null-hypothesis tail probabilities, not the probability that a model is best. Repeat Student-t CIs and participant-cluster bootstrap CIs are kept separate. The lossless table and full narrative are in [RESULT_INTERPRETATION.md](RESULT_INTERPRETATION.md) and `tables/comparison_conclusions.json`.

| Rank | Case | BA mean ± SD (%) | BA participant-cluster 95% CI (%) | Macro-F1 mean ± SD (%) | Macro-F1 participant-cluster 95% CI (%) | Macro ROC-AUC mean ± SD (%) | Worst-fold BA (%) | Worst-class F1 (%) | BA Holm P | F1 Holm P | P-value role |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b16_lr1e-3 | 62.2 ± 21.3 | N/A | 55.7 ± 22.7 | N/A | 77.4 ± 11.7 | 33.3 | 59.3 | N/A | N/A | N/A_no_eligible_paired_comparison |
| 2 | b32_lr3e-4 | 58.9 ± 18.7 | N/A | 50.3 ± 23.9 | N/A | 70.2 ± 12.4 | 33.3 | 47.1 | N/A | N/A | N/A_no_eligible_paired_comparison |
| 3 | b16_lr3e-4 | 55.6 ± 26.1 | N/A | 48.8 ± 28.4 | N/A | 71.9 ± 13.6 | 16.7 | 48.0 | N/A | N/A | N/A_no_eligible_paired_comparison |
| 4 | b16_lr1e-4 | 54.4 ± 25.9 | N/A | 47.9 ± 25.9 | N/A | 73.4 ± 14.0 | 16.7 | 46.2 | N/A | N/A | N/A_no_eligible_paired_comparison |
| 5 | b32_lr1e-4 | 52.2 ± 13.4 | N/A | 44.5 ± 18.3 | N/A | 77.6 ± 10.9 | 33.3 | 40.0 | N/A | N/A | N/A_no_eligible_paired_comparison |
| 6 | b32_lr1e-3 | 51.1 ± 23.4 | N/A | 43.6 ± 24.2 | N/A | 77.6 ± 9.0 | 16.7 | 42.1 | N/A | N/A | N/A_no_eligible_paired_comparison |

### Conclusions by evidence angle

| Angle | Case | Finding | Confidence | Selection effect |
|---|---|---|---|---|
| point_estimates | b16_lr1e-3 | Highest participant-OOF BA is b16_lr1e-3: 62.2 ± 21.3 percent; Macro-F1 55.7 ± 22.7 percent; macro ROC-AUC 77.4 ± 11.7 percent. | descriptive | none_by_itself |
| uncertainty | b16_lr1e-3 | Repeat t-CI and participant-cluster percentile CI are reported separately; marginal CI overlap is not used as a significance test. | evidence_completeness_moderate_reduced_resource_or_ci | supports_precision_audit_only |
| paired_inference | N/A | No eligible paired P-value family is available; superiority is not established. | exploratory_or_unavailable | none_automatic |
| robustness | b16_lr1e-3 | Worst-fold BA=33.3%; worst-class F1=59.3%. These stress metrics can disagree with mean BA ranking. | evidence_completeness_moderate_reduced_resource_or_ci | secondary_review |
| selection | N/A | Persisted choice=none by manual review only; no automatic selection in ordinary study reporter; participant-OOF point-estimate top=b16_lr1e-3; agreement=N/A. This is a screening choice, not an independent final-test winner. | not_established_no_selection | retain_persisted_choice_without_rewriting_history |

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

| Rank | Case | Abstention-aware BA, mean ± SD (%) | Abstention-aware precision | Abstention-aware recall | Abstention-aware Macro-F1, mean ± SD (%) | Participant coverage | Abstentions | Abstentions by class | Conditional BA, mean ± SD (%) | Conditional BA cluster 95% CI (%) | Conditional Macro-F1, mean ± SD (%) | Conditional Macro-F1 cluster 95% CI (%) | Macro ROC AUC, mean ± SD (%) | Macro PR AUC, mean ± SD (%) | Conditional BA LCB95 | Conditional Macro-F1 LCB95 | Aware worst-fold BA | Conditional worst-fold BA | Worst recall | Worst F1 | Source | Frailty endpoint | Motion auxiliary outer-OOF | Interpretation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | b16_lr1e-3 | 62.2 ± 21.3 | 0.5456 | 0.6222 | 55.7 ± 22.7 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 62.2 ± 21.3 | N/A | 55.7 ± 22.7 | N/A | 77.4 ± 11.7 | 76.6 ± 7.7 | 0.4191 | 0.3405 | 0.3333 | 0.3333 | 0.5000 | 0.5926 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 2 | b32_lr3e-4 | 58.9 ± 18.7 | 0.5289 | 0.5889 | 50.3 ± 23.9 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 58.9 ± 18.7 | N/A | 50.3 ± 23.9 | N/A | 70.2 ± 12.4 | 70.8 ± 11.8 | 0.4108 | 0.2746 | 0.3333 | 0.3333 | 0.4000 | 0.4706 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 3 | b16_lr3e-4 | 55.6 ± 26.1 | 0.5244 | 0.5556 | 48.8 ± 28.4 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 55.6 ± 26.1 | N/A | 48.8 ± 28.4 | N/A | 71.9 ± 13.6 | 71.7 ± 11.8 | 0.3071 | 0.2170 | 0.1667 | 0.1667 | 0.4615 | 0.4800 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 4 | b16_lr1e-4 | 54.4 ± 25.9 | 0.5022 | 0.5444 | 47.9 ± 25.9 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 54.4 ± 25.9 | N/A | 47.9 ± 25.9 | N/A | 73.4 ± 14.0 | 73.1 ± 12.1 | 0.2977 | 0.2322 | 0.1667 | 0.1667 | 0.3750 | 0.4615 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 5 | b32_lr1e-4 | 52.2 ± 13.4 | 0.5100 | 0.5222 | 44.5 ± 18.3 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 52.2 ± 13.4 | N/A | 44.5 ± 18.3 | N/A | 77.6 ± 10.9 | 76.2 ± 11.2 | 0.3947 | 0.2707 | 0.3333 | 0.3333 | 0.3750 | 0.4000 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |
| 6 | b32_lr1e-3 | 51.1 ± 23.4 | 0.4300 | 0.5111 | 43.6 ± 24.2 | 1.0000 | 0 | [[0, 0], [1, 0], [2, 0]] | 51.1 ± 23.4 | N/A | 43.6 ± 24.2 | N/A | 77.6 ± 9.0 | 77.6 ± 10.1 | 0.2883 | 0.2059 | 0.1667 | 0.1667 | 0.4000 | 0.4211 | participant_oof | outer_heldout_participant_oof | N/A | not_applicable_no_auxiliary_motion_evidence |

## Repeat-level predictive distributions

Mean and sample SD are shown in one percentage column and the two-sided repeat-level Student-t 95% CI is shown beside it. Lossless bounds, range, mean, and SD remain in the matching JSON table.

| Case | Metric | Repeats | Mean ± SD (%) | Repeat 95% CI (%) | Source |
|---|---|---|---|---|---|
| b16_lr1e-4 | balanced_accuracy | 5 | 54.4 ± 25.9 | [22.3, 86.6] | participant_oof |
| b16_lr1e-4 | macro_f1 | 5 | 47.9 ± 25.9 | [15.7, 80.1] | participant_oof |
| b16_lr1e-4 | macro_roc_auc_ovr | 5 | 73.4 ± 14.0 | [56.1, 90.8] | participant_oof |
| b16_lr1e-4 | macro_pr_auc_ovr | 5 | 73.1 ± 12.1 | [58.0, 88.1] | participant_oof |
| b16_lr1e-4 | abstention_aware_balanced_accuracy | 5 | 54.4 ± 25.9 | [22.3, 86.6] | participant_oof |
| b16_lr1e-4 | abstention_aware_macro_f1 | 5 | 47.9 ± 25.9 | [15.7, 80.1] | participant_oof |
| b16_lr3e-4 | balanced_accuracy | 5 | 55.6 ± 26.1 | [23.2, 87.9] | participant_oof |
| b16_lr3e-4 | macro_f1 | 5 | 48.8 ± 28.4 | [13.5, 84.1] | participant_oof |
| b16_lr3e-4 | macro_roc_auc_ovr | 5 | 71.9 ± 13.6 | [55.1, 88.8] | participant_oof |
| b16_lr3e-4 | macro_pr_auc_ovr | 5 | 71.7 ± 11.8 | [57.0, 86.4] | participant_oof |
| b16_lr3e-4 | abstention_aware_balanced_accuracy | 5 | 55.6 ± 26.1 | [23.2, 87.9] | participant_oof |
| b16_lr3e-4 | abstention_aware_macro_f1 | 5 | 48.8 ± 28.4 | [13.5, 84.1] | participant_oof |
| b16_lr1e-3 | balanced_accuracy | 5 | 62.2 ± 21.3 | [35.8, 88.7] | participant_oof |
| b16_lr1e-3 | macro_f1 | 5 | 55.7 ± 22.7 | [27.5, 83.8] | participant_oof |
| b16_lr1e-3 | macro_roc_auc_ovr | 5 | 77.4 ± 11.7 | [62.9, 91.8] | participant_oof |
| b16_lr1e-3 | macro_pr_auc_ovr | 5 | 76.6 ± 7.7 | [67.1, 86.1] | participant_oof |
| b16_lr1e-3 | abstention_aware_balanced_accuracy | 5 | 62.2 ± 21.3 | [35.8, 88.7] | participant_oof |
| b16_lr1e-3 | abstention_aware_macro_f1 | 5 | 55.7 ± 22.7 | [27.5, 83.8] | participant_oof |
| b32_lr1e-4 | balanced_accuracy | 5 | 52.2 ± 13.4 | [35.6, 68.8] | participant_oof |
| b32_lr1e-4 | macro_f1 | 5 | 44.5 ± 18.3 | [21.8, 67.2] | participant_oof |
| b32_lr1e-4 | macro_roc_auc_ovr | 5 | 77.6 ± 10.9 | [64.1, 91.2] | participant_oof |
| b32_lr1e-4 | macro_pr_auc_ovr | 5 | 76.2 ± 11.2 | [62.3, 90.1] | participant_oof |
| b32_lr1e-4 | abstention_aware_balanced_accuracy | 5 | 52.2 ± 13.4 | [35.6, 68.8] | participant_oof |
| b32_lr1e-4 | abstention_aware_macro_f1 | 5 | 44.5 ± 18.3 | [21.8, 67.2] | participant_oof |
| b32_lr3e-4 | balanced_accuracy | 5 | 58.9 ± 18.7 | [35.7, 82.1] | participant_oof |
| b32_lr3e-4 | macro_f1 | 5 | 50.3 ± 23.9 | [20.6, 80.0] | participant_oof |
| b32_lr3e-4 | macro_roc_auc_ovr | 5 | 70.2 ± 12.4 | [54.8, 85.6] | participant_oof |
| b32_lr3e-4 | macro_pr_auc_ovr | 5 | 70.8 ± 11.8 | [56.1, 85.5] | participant_oof |
| b32_lr3e-4 | abstention_aware_balanced_accuracy | 5 | 58.9 ± 18.7 | [35.7, 82.1] | participant_oof |
| b32_lr3e-4 | abstention_aware_macro_f1 | 5 | 50.3 ± 23.9 | [20.6, 80.0] | participant_oof |
| b32_lr1e-3 | balanced_accuracy | 5 | 51.1 ± 23.4 | [22.1, 80.1] | participant_oof |
| b32_lr1e-3 | macro_f1 | 5 | 43.6 ± 24.2 | [13.6, 73.6] | participant_oof |
| b32_lr1e-3 | macro_roc_auc_ovr | 5 | 77.6 ± 9.0 | [66.4, 88.8] | participant_oof |
| b32_lr1e-3 | macro_pr_auc_ovr | 5 | 77.6 ± 10.1 | [65.0, 90.1] | participant_oof |
| b32_lr1e-3 | abstention_aware_balanced_accuracy | 5 | 51.1 ± 23.4 | [22.1, 80.1] | participant_oof |
| b32_lr1e-3 | abstention_aware_macro_f1 | 5 | 43.6 ± 24.2 | [13.6, 73.6] | participant_oof |

<details><summary>Per-class repeat distributions</summary>

| Case | Class | Metric | Repeats | Mean ± SD (%) | Repeat 95% CI (%) |
|---|---|---|---|---|---|
| b16_lr1e-3 | Pre-Frail | balanced_accuracy_ovr | 5 | 73.0 ± 9.9 | [60.7, 85.3] |
| b16_lr1e-3 | Pre-Frail | f1 | 5 | 62.7 ± 12.8 | [46.8, 78.5] |
| b16_lr1e-3 | Pre-Frail | recall | 5 | 70.0 ± 27.4 | [36.0, 104.0] |
| b16_lr1e-3 | Pre-Frail | specificity | 5 | 76.0 ± 17.8 | [53.9, 98.1] |
| b16_lr1e-3 | Pre-Frail | roc_auc_ovr | 5 | 86.0 ± 12.9 | [69.9, 102.1] |
| b16_lr1e-3 | Pre-Frail | pr_auc_ovr | 5 | 78.3 ± 20.9 | [52.4, 104.3] |
| b16_lr1e-3 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 62.5 ± 18.6 | [39.4, 85.6] |
| b16_lr1e-3 | Robust/Non-Frail | f1 | 5 | 51.0 ± 33.8 | [9.0, 93.0] |
| b16_lr1e-3 | Robust/Non-Frail | recall | 5 | 56.7 ± 36.5 | [11.3, 102.0] |
| b16_lr1e-3 | Robust/Non-Frail | specificity | 5 | 68.3 ± 32.5 | [28.0, 108.7] |
| b16_lr1e-3 | Robust/Non-Frail | roc_auc_ovr | 5 | 68.1 ± 13.1 | [51.8, 84.3] |
| b16_lr1e-3 | Robust/Non-Frail | pr_auc_ovr | 5 | 74.7 ± 16.1 | [54.7, 94.6] |
| b16_lr1e-3 | Young | balanced_accuracy_ovr | 5 | 78.0 ± 25.9 | [45.9, 110.1] |
| b16_lr1e-3 | Young | f1 | 5 | 53.3 ± 50.6 | [-9.4, 116.1] |
| b16_lr1e-3 | Young | recall | 5 | 60.0 ± 54.8 | [-8.0, 128.0] |
| b16_lr1e-3 | Young | specificity | 5 | 96.0 ± 8.9 | [84.9, 107.1] |
| b16_lr1e-3 | Young | roc_auc_ovr | 5 | 78.0 ± 22.8 | [49.7, 106.3] |
| b16_lr1e-3 | Young | pr_auc_ovr | 5 | 76.7 ± 22.4 | [48.9, 104.4] |
| b16_lr1e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 68.5 ± 21.8 | [41.4, 95.6] |
| b16_lr1e-4 | Pre-Frail | f1 | 5 | 53.3 ± 34.0 | [11.1, 95.5] |
| b16_lr1e-4 | Pre-Frail | recall | 5 | 60.0 ± 41.8 | [8.1, 111.9] |
| b16_lr1e-4 | Pre-Frail | specificity | 5 | 77.0 ± 14.4 | [59.1, 94.9] |
| b16_lr1e-4 | Pre-Frail | roc_auc_ovr | 5 | 79.0 ± 14.9 | [60.6, 97.4] |
| b16_lr1e-4 | Pre-Frail | pr_auc_ovr | 5 | 72.3 ± 19.9 | [47.6, 97.1] |
| b16_lr1e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 55.8 ± 17.6 | [34.0, 77.7] |
| b16_lr1e-4 | Robust/Non-Frail | f1 | 5 | 50.5 ± 14.0 | [33.1, 67.9] |
| b16_lr1e-4 | Robust/Non-Frail | recall | 5 | 53.3 ± 13.9 | [36.0, 70.6] |
| b16_lr1e-4 | Robust/Non-Frail | specificity | 5 | 58.3 ± 37.3 | [12.1, 104.6] |
| b16_lr1e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 65.8 ± 11.9 | [51.0, 80.6] |
| b16_lr1e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 75.2 ± 7.0 | [66.6, 83.9] |
| b16_lr1e-4 | Young | balanced_accuracy_ovr | 5 | 71.0 ± 20.1 | [46.0, 96.0] |
| b16_lr1e-4 | Young | f1 | 5 | 40.0 ± 36.5 | [-5.3, 85.3] |
| b16_lr1e-4 | Young | recall | 5 | 50.0 ± 50.0 | [-12.1, 112.1] |
| b16_lr1e-4 | Young | specificity | 5 | 92.0 ± 11.0 | [78.4, 105.6] |
| b16_lr1e-4 | Young | roc_auc_ovr | 5 | 75.5 ± 26.9 | [42.0, 109.0] |
| b16_lr1e-4 | Young | pr_auc_ovr | 5 | 71.7 ± 27.4 | [37.7, 105.7] |
| b16_lr3e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 64.5 ± 18.7 | [41.3, 87.7] |
| b16_lr3e-4 | Pre-Frail | f1 | 5 | 49.3 ± 30.3 | [11.7, 87.0] |
| b16_lr3e-4 | Pre-Frail | recall | 5 | 60.0 ± 41.8 | [8.1, 111.9] |
| b16_lr3e-4 | Pre-Frail | specificity | 5 | 69.0 ± 29.5 | [32.4, 105.6] |
| b16_lr3e-4 | Pre-Frail | roc_auc_ovr | 5 | 74.5 ± 13.2 | [58.2, 90.8] |
| b16_lr3e-4 | Pre-Frail | pr_auc_ovr | 5 | 68.3 ± 14.9 | [49.8, 86.8] |
| b16_lr3e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 56.7 ± 19.9 | [32.0, 81.4] |
| b16_lr3e-4 | Robust/Non-Frail | f1 | 5 | 50.4 ± 16.7 | [29.7, 71.1] |
| b16_lr3e-4 | Robust/Non-Frail | recall | 5 | 46.7 ± 13.9 | [29.4, 64.0] |
| b16_lr3e-4 | Robust/Non-Frail | specificity | 5 | 66.7 ± 40.8 | [16.0, 117.4] |
| b16_lr3e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 65.8 ± 11.9 | [51.0, 80.6] |
| b16_lr3e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 75.2 ± 7.0 | [66.6, 83.9] |
| b16_lr3e-4 | Young | balanced_accuracy_ovr | 5 | 76.0 ± 24.1 | [46.1, 105.9] |
| b16_lr3e-4 | Young | f1 | 5 | 46.7 ± 44.7 | [-8.9, 102.2] |
| b16_lr3e-4 | Young | recall | 5 | 60.0 ± 54.8 | [-8.0, 128.0] |
| b16_lr3e-4 | Young | specificity | 5 | 92.0 ± 11.0 | [78.4, 105.6] |
| b16_lr3e-4 | Young | roc_auc_ovr | 5 | 75.5 ± 26.9 | [42.0, 109.0] |
| b16_lr3e-4 | Young | pr_auc_ovr | 5 | 71.7 ± 27.4 | [37.7, 105.7] |
| b32_lr1e-3 | Pre-Frail | balanced_accuracy_ovr | 5 | 58.5 ± 25.7 | [26.6, 90.4] |
| b32_lr1e-3 | Pre-Frail | f1 | 5 | 38.0 ± 41.5 | [-13.5, 89.5] |
| b32_lr1e-3 | Pre-Frail | recall | 5 | 40.0 ± 41.8 | [-11.9, 91.9] |
| b32_lr1e-3 | Pre-Frail | specificity | 5 | 77.0 ± 14.4 | [59.1, 94.9] |
| b32_lr1e-3 | Pre-Frail | roc_auc_ovr | 5 | 79.0 ± 14.9 | [60.6, 97.4] |
| b32_lr1e-3 | Pre-Frail | pr_auc_ovr | 5 | 68.3 ± 22.4 | [40.6, 96.1] |
| b32_lr1e-3 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 55.0 ± 25.6 | [23.2, 86.8] |
| b32_lr1e-3 | Robust/Non-Frail | f1 | 5 | 49.5 ± 22.4 | [21.7, 77.4] |
| b32_lr1e-3 | Robust/Non-Frail | recall | 5 | 53.3 ± 27.4 | [19.3, 87.3] |
| b32_lr1e-3 | Robust/Non-Frail | specificity | 5 | 56.7 ± 39.7 | [7.4, 106.0] |
| b32_lr1e-3 | Robust/Non-Frail | roc_auc_ovr | 5 | 64.7 ± 20.2 | [39.6, 89.8] |
| b32_lr1e-3 | Robust/Non-Frail | pr_auc_ovr | 5 | 75.3 ± 13.9 | [58.1, 92.5] |
| b32_lr1e-3 | Young | balanced_accuracy_ovr | 5 | 74.0 ± 23.0 | [45.4, 102.6] |
| b32_lr1e-3 | Young | f1 | 5 | 43.3 ± 43.5 | [-10.6, 97.3] |
| b32_lr1e-3 | Young | recall | 5 | 60.0 ± 54.8 | [-8.0, 128.0] |
| b32_lr1e-3 | Young | specificity | 5 | 88.0 ± 17.9 | [65.8, 110.2] |
| b32_lr1e-3 | Young | roc_auc_ovr | 5 | 89.0 ± 15.2 | [70.2, 107.8] |
| b32_lr1e-3 | Young | pr_auc_ovr | 5 | 89.0 ± 15.2 | [70.2, 107.8] |
| b32_lr1e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 62.0 ± 19.8 | [37.4, 86.6] |
| b32_lr1e-4 | Pre-Frail | f1 | 5 | 39.3 ± 37.4 | [-7.2, 85.8] |
| b32_lr1e-4 | Pre-Frail | recall | 5 | 50.0 ± 50.0 | [-12.1, 112.1] |
| b32_lr1e-4 | Pre-Frail | specificity | 5 | 74.0 ± 32.7 | [33.4, 114.6] |
| b32_lr1e-4 | Pre-Frail | roc_auc_ovr | 5 | 83.5 ± 10.5 | [70.4, 96.6] |
| b32_lr1e-4 | Pre-Frail | pr_auc_ovr | 5 | 75.0 ± 17.7 | [53.1, 96.9] |
| b32_lr1e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 61.7 ± 6.8 | [53.2, 70.2] |
| b32_lr1e-4 | Robust/Non-Frail | f1 | 5 | 52.9 ± 3.9 | [48.0, 57.7] |
| b32_lr1e-4 | Robust/Non-Frail | recall | 5 | 56.7 ± 27.9 | [22.0, 91.3] |
| b32_lr1e-4 | Robust/Non-Frail | specificity | 5 | 66.7 ± 35.8 | [22.2, 111.2] |
| b32_lr1e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 73.9 ± 15.9 | [54.2, 93.6] |
| b32_lr1e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 82.0 ± 12.6 | [66.3, 97.7] |
| b32_lr1e-4 | Young | balanced_accuracy_ovr | 5 | 66.5 ± 24.1 | [36.6, 96.4] |
| b32_lr1e-4 | Young | f1 | 5 | 41.3 ± 43.3 | [-12.4, 95.1] |
| b32_lr1e-4 | Young | recall | 5 | 50.0 ± 50.0 | [-12.1, 112.1] |
| b32_lr1e-4 | Young | specificity | 5 | 83.0 ± 26.4 | [50.3, 115.7] |
| b32_lr1e-4 | Young | roc_auc_ovr | 5 | 75.5 ± 26.9 | [42.0, 109.0] |
| b32_lr1e-4 | Young | pr_auc_ovr | 5 | 71.7 ± 27.4 | [37.7, 105.7] |
| b32_lr3e-4 | Pre-Frail | balanced_accuracy_ovr | 5 | 63.5 ± 16.9 | [42.5, 84.5] |
| b32_lr3e-4 | Pre-Frail | f1 | 5 | 37.3 ± 37.0 | [-8.6, 83.3] |
| b32_lr3e-4 | Pre-Frail | recall | 5 | 40.0 ± 41.8 | [-11.9, 91.9] |
| b32_lr3e-4 | Pre-Frail | specificity | 5 | 87.0 ± 18.6 | [63.9, 110.1] |
| b32_lr3e-4 | Pre-Frail | roc_auc_ovr | 5 | 78.5 ± 16.3 | [58.3, 98.7] |
| b32_lr3e-4 | Pre-Frail | pr_auc_ovr | 5 | 71.7 ± 21.7 | [44.7, 98.6] |
| b32_lr3e-4 | Robust/Non-Frail | balanced_accuracy_ovr | 5 | 69.2 ± 13.7 | [52.2, 86.2] |
| b32_lr3e-4 | Robust/Non-Frail | f1 | 5 | 65.5 ± 15.7 | [46.0, 84.9] |
| b32_lr3e-4 | Robust/Non-Frail | recall | 5 | 76.7 ± 32.5 | [36.3, 117.0] |
| b32_lr3e-4 | Robust/Non-Frail | specificity | 5 | 61.7 ± 43.9 | [7.1, 116.2] |
| b32_lr3e-4 | Robust/Non-Frail | roc_auc_ovr | 5 | 56.1 ± 21.9 | [28.9, 83.3] |
| b32_lr3e-4 | Robust/Non-Frail | pr_auc_ovr | 5 | 64.7 ± 18.8 | [41.3, 88.1] |
| b32_lr3e-4 | Young | balanced_accuracy_ovr | 5 | 74.0 ± 25.1 | [42.8, 105.2] |
| b32_lr3e-4 | Young | f1 | 5 | 48.0 ± 50.2 | [-14.3, 110.3] |
| b32_lr3e-4 | Young | recall | 5 | 60.0 ± 54.8 | [-8.0, 128.0] |
| b32_lr3e-4 | Young | specificity | 5 | 88.0 ± 26.8 | [54.7, 121.3] |
| b32_lr3e-4 | Young | roc_auc_ovr | 5 | 76.0 ± 25.1 | [44.8, 107.2] |
| b32_lr3e-4 | Young | pr_auc_ovr | 5 | 76.2 ± 22.7 | [48.1, 104.3] |

</details>

## Paired participant-cluster inference

Each candidate is compared with the declared reference on the exact participant/repeat/fold/split roster. P values are two-sided participant-cluster permutation results; Holm adjustment is applied separately within BA and Macro-F1. These comparisons do not select a winner and do not turn this representation screen into a causal ablation.

N/A — no rows were available.

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

| Case | Repeat | Level | View | Group | OOF units | Retained units | Dropped units | Retained coverage | All participants | Retained participants | Dropped participants |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | 0 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | ALL | 3596 | 3596 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | ALL | 35 | 35 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | participant | participant_balanced_endpoint | ALL | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | role | line_b_equal_role_families | ALL | 14 | 14 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | ALL | 4195 | 4195 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | ALL | 3601 | 3601 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | ALL | 3604 | 3604 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | ALL | 3541 | 3541 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-3 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | ALL | 3596 | 3596 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | ALL | 35 | 35 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | participant | participant_balanced_endpoint | ALL | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | role | line_b_equal_role_families | ALL | 14 | 14 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | ALL | 4195 | 4195 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | ALL | 3601 | 3601 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | ALL | 3604 | 3604 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | ALL | 3541 | 3541 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr1e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | ALL | 3596 | 3596 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | ALL | 35 | 35 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | participant | participant_balanced_endpoint | ALL | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | role | line_b_equal_role_families | ALL | 14 | 14 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | ALL | 4195 | 4195 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | ALL | 3601 | 3601 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | ALL | 3604 | 3604 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | ALL | 3541 | 3541 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b16_lr3e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | ALL | 3596 | 3596 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | ALL | 35 | 35 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | participant | participant_balanced_endpoint | ALL | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | role | line_b_equal_role_families | ALL | 14 | 14 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | ALL | 4195 | 4195 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | ALL | 3601 | 3601 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | ALL | 3604 | 3604 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | ALL | 3541 | 3541 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-3 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | ALL | 3596 | 3596 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | ALL | 35 | 35 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | participant | participant_balanced_endpoint | ALL | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | role | line_b_equal_role_families | ALL | 14 | 14 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | ALL | 4195 | 4195 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | ALL | 3601 | 3601 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | ALL | 3604 | 3604 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | ALL | 3541 | 3541 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr1e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | ALL | 3596 | 3596 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R2 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 0 | window | window_balanced_to_participant | R4 | 723 | 723 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | ALL | 35 | 35 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R1 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R2 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R3 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | file | line_a_equal_files | R4 | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | participant | participant_balanced_endpoint | ALL | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | role | line_b_equal_role_families | ALL | 14 | 14 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | role | line_b_equal_role_families | B | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | role | line_b_equal_role_families | R | 7 | 7 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | ALL | 4195 | 4195 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | B | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R1 | 839 | 839 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R2 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R3 | 838 | 838 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 1 | window | window_balanced_to_participant | R4 | 840 | 840 | 0 | 1.0000 | 7 | 7 | 0 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | ALL | 3601 | 3601 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | B | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R2 | 720 | 720 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 2 | window | window_balanced_to_participant | R4 | 725 | 725 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | ALL | 3604 | 3604 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | B | 728 | 728 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R1 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R2 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R3 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 3 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | ALL | 30 | 30 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R1 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R2 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R3 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | file | line_a_equal_files | R4 | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | participant | participant_balanced_endpoint | ALL | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | role | line_b_equal_role_families | ALL | 12 | 12 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | role | line_b_equal_role_families | B | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | role | line_b_equal_role_families | R | 6 | 6 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | ALL | 3541 | 3541 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | B | 717 | 717 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R1 | 693 | 693 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R2 | 694 | 694 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R3 | 718 | 718 | 0 | 1.0000 | 6 | 6 | 0 |
| b32_lr3e-4 | 4 | window | window_balanced_to_participant | R4 | 719 | 719 | 0 | 1.0000 | 6 | 6 | 0 |

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

| Case | Role | Quality tier | Motion | Route state | Signal route | Files | Retained files | Dropped files | Retained coverage | Abstention | Abstention reasons | Direct | Processed | Unavailable predictors | Denoiser attempts | Denoiser successes | Reducer failures | Reducer failure rate | Post Q_rate pass rate | Recovery eligible | Q_rate recovered | Q_rate recovery rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b16_lr1e-3 | B | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-3 | R1 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-3 | R2 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-3 | R3 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-3 | R4 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-4 | B | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-4 | R1 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-4 | R2 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-4 | R3 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr1e-4 | R4 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr3e-4 | B | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr3e-4 | R1 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr3e-4 | R2 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr3e-4 | R3 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b16_lr3e-4 | R4 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-3 | B | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-3 | R1 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-3 | R2 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-3 | R3 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-3 | R4 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-4 | B | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-4 | R1 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-4 | R2 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-4 | R3 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr1e-4 | R4 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr3e-4 | B | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr3e-4 | R1 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr3e-4 | R2 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr3e-4 | R3 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |
| b32_lr3e-4 | R4 | excellent | off | full_direct | direct_x_filter | 145 | 145 | 0 | 1.0000 | 0.0000 | not_reported | 0 | 0 | N/A | 0 | 0 | 0 | 0.0000 | N/A | 0 | 0 | N/A |

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
