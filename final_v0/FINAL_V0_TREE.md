# `final_v0` 文件树与逐文件详细说明 / File Tree and Detailed Per-file Descriptions

> 本文件由 `tools/update_final_v0_index_detailed.py` 自动生成。索引自身不记录 SHA-256，以避免自引用。

## 树状结构 / Tree

```text
final_v0/
├── algorithm_diagrams
│   ├── baseline
│   │   ├── 00_ARCHIVED_CODE_LINEAGE.md
│   │   ├── 01_NON_M0_ROOT_SCRIPT_ATLAS.md
│   │   └── 02_ARCHIVED_SCRIPT_ATLAS.md
│   ├── m0
│   │   ├── 01_FOUNDATION_FUNCS_PPG.md
│   │   ├── 02_V7_TO_STAGE2_EVOLUTION.md
│   │   ├── 03_HYBRID_SUITE.md
│   │   ├── 04_HEARTBEAT_AND_MOTION_AB.md
│   │   ├── 05_SCRIPT_ALGORITHM_ATLAS.md
│   │   ├── 06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md
│   │   ├── 07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md
│   │   └── 08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md
│   ├── m1
│   │   ├── 00_END_TO_END_MOBILE_PIPELINE.md
│   │   ├── 01_END_TO_END_MOBILE_PIPELINE_V2.md
│   │   └── 02_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md
│   ├── m2
│   │   ├── 00_DATA_MANIFEST_DUAL_FOLD_AND_PROTOCOL.md
│   │   └── 01_EXTERNAL_SYNCHRONIZED_REFERENCE_MANIFEST.md
│   ├── m3
│   │   ├── 00_UNIFIED_PREPROCESSING_AND_SIGNAL_API.md
│   │   ├── 01_IMU_EKF_PRIMARY_AND_LPF_COMPARATOR.md
│   │   ├── 02_PEAK_PPI_HR_PRV_COMMON_BACKEND.md
│   │   └── 03_REFERENCE_TEST_AND_PARITY_MATRIX.md
│   ├── 00_PROJECT_HISTORICAL_SIGNAL_FLOW.md
│   ├── 01_PROJECT_END_TO_END_PIPELINE.md
│   └── README.md
├── final_pipeline_v1
│   ├── artifacts
│   │   ├── acceptance
│   │   │   ├── runs
│   │   │   │   ├── experiment_reduced_r0_f0_20260815T165715Z_517223
│   │   │   │   │   ├── confusion_matrices.json
│   │   │   │   │   ├── experiment_result.json
│   │   │   │   │   ├── metrics_per_fold_seed.json
│   │   │   │   │   ├── oof_file_predictions.parquet
│   │   │   │   │   ├── oof_member_predictions.parquet
│   │   │   │   │   ├── oof_subject_predictions.parquet
│   │   │   │   │   ├── oof_window_predictions.parquet
│   │   │   │   │   └── run_manifest.json
│   │   │   │   ├── artifact_parallel_20260815T154022Z_492045.json
│   │   │   │   ├── artifact_parallel_20260815T154410Z_493798.json
│   │   │   │   ├── artifact_parallel_20260815T165715Z_517223.json
│   │   │   │   ├── cli_smoke_20260815T154022Z_492045.json
│   │   │   │   ├── cli_smoke_20260815T154410Z_493798.json
│   │   │   │   ├── cli_smoke_20260815T165715Z_517223.json
│   │   │   │   ├── model_parallel_20260815T154022Z_492045.json
│   │   │   │   ├── model_parallel_20260815T154410Z_493798.json
│   │   │   │   ├── model_parallel_20260815T165715Z_517223.json
│   │   │   │   ├── raw_window_ablation_20260815T154022Z_492045.json
│   │   │   │   ├── raw_window_ablation_20260815T154410Z_493798.json
│   │   │   │   └── raw_window_ablation_20260815T165715Z_517223.json
│   │   │   ├── cpu_ci_current.json
│   │   │   ├── cpu_ci_tests_current.json
│   │   │   ├── source_snapshot_current.json
│   │   │   ├── strict_acceptance_current.json
│   │   │   └── strict_acceptance_pending.json
│   │   ├── audit
│   │   │   ├── baseline_inventory.json
│   │   │   └── legacy_characterization.json
│   │   ├── experiments
│   │   │   ├── reduced_real_r0_f0_12s_failed_closed
│   │   │   │   ├── confusion_matrices.json
│   │   │   │   ├── experiment_result.json
│   │   │   │   ├── metrics_per_fold_seed.json
│   │   │   │   ├── oof_file_predictions.parquet
│   │   │   │   ├── oof_member_predictions.parquet
│   │   │   │   ├── oof_subject_predictions.parquet
│   │   │   │   ├── oof_window_predictions.parquet
│   │   │   │   └── run_manifest.json
│   │   │   ├── reduced_real_r0_f0_reference
│   │   │   │   ├── confusion_matrices.json
│   │   │   │   ├── experiment_result.json
│   │   │   │   ├── metrics_per_fold_seed.json
│   │   │   │   ├── oof_file_predictions.parquet
│   │   │   │   ├── oof_member_predictions.parquet
│   │   │   │   ├── oof_subject_predictions.parquet
│   │   │   │   ├── oof_window_predictions.parquet
│   │   │   │   └── run_manifest.json
│   │   │   ├── reduced_real_r0_f0_reference_width_preserved_v2
│   │   │   │   ├── confusion_matrices.json
│   │   │   │   ├── experiment_result.json
│   │   │   │   ├── metrics_per_fold_seed.json
│   │   │   │   ├── oof_file_predictions.parquet
│   │   │   │   ├── oof_member_predictions.parquet
│   │   │   │   ├── oof_subject_predictions.parquet
│   │   │   │   ├── oof_window_predictions.parquet
│   │   │   │   └── run_manifest.json
│   │   │   └── reference_registry.json
│   │   └── test_reports
│   │       ├── artifact_comparison_canonical_manual.json
│   │       ├── artifact_comparison_manual.json
│   │       ├── audit_current.json
│   │       ├── cli_via_public_command.json
│   │       ├── contracts_current.json
│   │       ├── data_current.json
│   │       ├── dl_fs_ablation_manual.json
│   │       ├── imu_gravity_comparison_manual.json
│   │       ├── integration_smoke_canonical_manual.json
│   │       ├── integration_smoke_manual.json
│   │       ├── model_comparison_all13_manual.json
│   │       ├── model_comparison_manual.json
│   │       ├── models_current.json
│   │       ├── phase08_cli_all_tests.json
│   │       ├── physical_time_contract_manual.json
│   │       └── training_current.json
│   ├── configs
│   │   ├── feature_matrix_v1.yaml
│   │   ├── motion_benchmark_v1.yaml
│   │   ├── reference_all_roles_v1.yaml
│   │   └── reference_static_v1.yaml
│   ├── docs
│   │   ├── adr
│   │   │   ├── ADR-001-canonical-experiment-entrypoint.md
│   │   │   ├── ADR-002-record-manifest-and-fold-freeze.md
│   │   │   ├── ADR-003-signal-views-and-units.md
│   │   │   ├── ADR-004-window-planning-padding-and-masks.md
│   │   │   ├── ADR-005-prv-eligibility-and-time-axis.md
│   │   │   ├── ADR-006-window-file-subject-aggregation.md
│   │   │   ├── ADR-007-epoch-selection-and-outer-fold-isolation.md
│   │   │   ├── ADR-008-model-naming-and-original-paper-deviations.md
│   │   │   ├── ADR-009-dl-sampling-rate-and-kernel-time-scales.md
│   │   │   ├── ADR-010-motion-branch-status-and-primary-experiment-boundary.md
│   │   │   ├── ADR-011-representation-modes-and-feature-matrix-contract.md
│   │   │   └── ADR-012-post-artifact-rate-only-feature-contract.md
│   │   ├── algorithms
│   │   │   ├── 00_END_TO_END_PIPELINE.md
│   │   │   ├── 01_DATA_MANIFEST_FOLDS_AND_LEAKAGE.md
│   │   │   ├── 02_SIGNAL_QUALITY_ARTIFACT_FEATURES.md
│   │   │   ├── 03_REPRESENTATIONS_AND_PARALLEL_MODELS.md
│   │   │   ├── 04_TRAIN_OOF_BUNDLE.md
│   │   │   ├── 05_ABLATION_AND_COMPARISON_EXECUTION.md
│   │   │   └── README.md
│   │   ├── comparisons
│   │   │   ├── 01_SPEC_VS_TODO_OVERLAP_AND_DIFFERENCES.md
│   │   │   ├── 02_SPEC_VS_COMPLETED_TODO.md
│   │   │   ├── 03_SPEC_VS_LOCAL_FROZEN_WORKFLOW.md
│   │   │   ├── 04_ALGORITHM_REASONABLENESS_AND_TRADEOFFS.md
│   │   │   └── 05_V1_TO_V2_CONFIRMATION_SUMMARY.md
│   │   └── spec
│   │       └── SPEC_LOCK.json
│   ├── manifests
│   │   ├── external_records_v1.csv
│   │   └── internal_records_v1.csv
│   ├── model_cards
│   │   ├── compact_cnn.md
│   │   ├── extra_trees.md
│   │   ├── fusion_compact.md
│   │   ├── fusion_inception.md
│   │   ├── inception_five_member_ensemble.md
│   │   ├── inception_full.md
│   │   ├── inception_matrix.md
│   │   ├── inception_small.md
│   │   ├── logistic_regression.md
│   │   ├── minirocket_ablation.md
│   │   ├── rbf_svm.md
│   │   ├── README.md
│   │   ├── rocket_numpy.md
│   │   └── shapeformer_effect_size.md
│   ├── records
│   │   ├── log_entries
│   │   │   ├── 20260815_phase01_spec_lock_and_adr_001_004.md
│   │   │   ├── 20260815_phase02_adr_005_008.md
│   │   │   ├── 20260815_phase03_adr_009_012.md
│   │   │   ├── 20260815_phase04_config_contracts_provenance.md
│   │   │   ├── 20260815_phase04b_reference_configs_and_baseline.md
│   │   │   ├── 20260815_phase04c1_validator_spec_lock_field_fix.md
│   │   │   ├── 20260815_phase04c_standard_library_test_and_validator.md
│   │   │   ├── 20260815_phase04d_core_contract_tests.md
│   │   │   ├── 20260815_phase04e_algorithm_diagrams.md
│   │   │   ├── 20260815_phase04f_v2_decision_registry.md
│   │   │   ├── 20260815_phase04g_matrix_input_dimension_contract.md
│   │   │   ├── 20260815_phase04h_analysis_view_correction.md
│   │   │   ├── 20260815_phase04i_raw_window_padding_alignment.md
│   │   │   ├── 20260815_phase06_signal_artifact_features.md
│   │   │   ├── 20260815_phase07_models_training.md
│   │   │   ├── 20260815_phase08a_spec_todo_comparison_reports.md
│   │   │   ├── 20260815_phase08b_remaining_user_reports.md
│   │   │   ├── 20260815_phase09a_baseline_regression_gate.md
│   │   │   ├── 20260815_phase09b_generated_model_cards.md
│   │   │   ├── 20260815_phase09c_physical_time_ablation.md
│   │   │   ├── 20260815_phase09c_training_evaluation_bundle_protocol.md
│   │   │   ├── 20260815_phase09d_training_canonical_facade_parity.md
│   │   │   ├── 20260815_phase10_strict_acceptance_cpu_ci.md
│   │   │   ├── 20260815_phase11_shapeformer_spec61_repair.md
│   │   │   ├── 20260815_phase12_documentation_acceptance_handoff.md
│   │   │   ├── 20260815_phase12_real_reduced_current_acceptance.md
│   │   │   ├── phase05_data_protocol.md
│   │   │   └── phase10_experiment_runner.md
│   │   └── v2_decision_points
│   │       ├── HUMAN_CONFIRMATION_POINTS.md
│   │       └── INITIAL_CONSERVATIVE_DEFAULTS.md
│   ├── reports
│   │   ├── data_contract_report.json
│   │   └── external_data_contract_report.json
│   ├── splits
│   │   ├── sgkf5_repeats_v1.csv
│   │   ├── sgkf5_v1.csv
│   │   └── v1_provisional_external_grouped_split_seed42.csv
│   ├── src
│   │   └── ppg_frailty
│   │       ├── artifact
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   ├── bss.py
│   │       │   ├── decomposition.py
│   │       │   ├── identity.py
│   │       │   ├── nlms.py
│   │       │   ├── router.py
│   │       │   └── spectral.py
│   │       ├── artifacts
│   │       │   ├── __init__.py
│   │       │   ├── base.py
│   │       │   ├── bss.py
│   │       │   ├── decomposition.py
│   │       │   ├── identity.py
│   │       │   ├── nlms.py
│   │       │   ├── router.py
│   │       │   └── spectral.py
│   │       ├── bundle
│   │       │   ├── __init__.py
│   │       │   ├── infer.py
│   │       │   ├── load.py
│   │       │   ├── save.py
│   │       │   └── schema.py
│   │       ├── data
│   │       │   ├── __init__.py
│   │       │   ├── cache.py
│   │       │   ├── external_manifest.py
│   │       │   ├── folds.py
│   │       │   ├── manifest.py
│   │       │   ├── qc.py
│   │       │   ├── schema.py
│   │       │   └── windows.py
│   │       ├── evaluate
│   │       │   ├── __init__.py
│   │       │   ├── aggregate.py
│   │       │   ├── benchmark.py
│   │       │   ├── calibration.py
│   │       │   ├── metrics.py
│   │       │   └── oof.py
│   │       ├── features
│   │       │   ├── __init__.py
│   │       │   ├── dual_wavelength.py
│   │       │   ├── engineering.py
│   │       │   ├── file_vector.py
│   │       │   ├── morphology.py
│   │       │   ├── ordered_matrix.py
│   │       │   ├── prv.py
│   │       │   ├── registry.py
│   │       │   └── spectral.py
│   │       ├── models
│   │       │   ├── __init__.py
│   │       │   ├── compact_cnn.py
│   │       │   ├── factory.py
│   │       │   ├── feature_baselines.py
│   │       │   ├── feature_models.py
│   │       │   ├── file_fusion.py
│   │       │   ├── fusion.py
│   │       │   ├── inception.py
│   │       │   ├── inception_ensemble.py
│   │       │   ├── inception_time_port.py
│   │       │   ├── rocket.py
│   │       │   ├── rocket_ridge.py
│   │       │   ├── shapeformer.py
│   │       │   ├── shapeformer_port.py
│   │       │   └── time_scale.py
│   │       ├── peaks
│   │       │   ├── __init__.py
│   │       │   ├── aboy_project.py
│   │       │   ├── intervals.py
│   │       │   └── pairing.py
│   │       ├── quality
│   │       │   ├── __init__.py
│   │       │   ├── components.py
│   │       │   ├── endpoint_sqi.py
│   │       │   └── routing.py
│   │       ├── representations
│   │       │   ├── __init__.py
│   │       │   ├── feature_matrix.py
│   │       │   ├── feature_vector.py
│   │       │   ├── fusion.py
│   │       │   ├── modes.py
│   │       │   └── raw.py
│   │       ├── signal
│   │       │   ├── __init__.py
│   │       │   ├── imu.py
│   │       │   ├── imu_preprocess.py
│   │       │   ├── morphology.py
│   │       │   ├── optical.py
│   │       │   ├── peaks.py
│   │       │   ├── ppg_preprocess.py
│   │       │   ├── preprocess.py
│   │       │   ├── prv.py
│   │       │   ├── resample.py
│   │       │   ├── sqi.py
│   │       │   ├── views.py
│   │       │   └── window_plan.py
│   │       ├── train
│   │       │   ├── __init__.py
│   │       │   ├── datasets.py
│   │       │   ├── losses.py
│   │       │   ├── sampling.py
│   │       │   ├── selection.py
│   │       │   └── trainer.py
│   │       ├── training
│   │       │   ├── __init__.py
│   │       │   ├── ablation.py
│   │       │   ├── aggregation.py
│   │       │   ├── bundle.py
│   │       │   ├── datasets.py
│   │       │   ├── evaluator.py
│   │       │   ├── oof.py
│   │       │   └── trainer.py
│   │       ├── __init__.py
│   │       ├── cli.py
│   │       ├── config.py
│   │       ├── contracts.py
│   │       ├── experiment.py
│   │       ├── module_registry.py
│   │       ├── pipeline.py
│   │       └── provenance.py
│   ├── tests
│   │   ├── acceptance
│   │   │   ├── __init__.py
│   │   │   ├── test_acceptance_gate.py
│   │   │   ├── test_external_ecg_and_regression_guards.py
│   │   │   └── test_rocket_10000_serialization.py
│   │   ├── artifacts
│   │   │   ├── __init__.py
│   │   │   ├── test_reducers.py
│   │   │   └── test_router_rate_only.py
│   │   ├── audit
│   │   │   ├── __init__.py
│   │   │   └── test_baseline_characterization.py
│   │   ├── cli
│   │   │   ├── __init__.py
│   │   │   └── test_cli_commands.py
│   │   ├── contracts
│   │   │   ├── __init__.py
│   │   │   └── test_core_contracts.py
│   │   ├── data
│   │   │   ├── __init__.py
│   │   │   ├── test_folds.py
│   │   │   ├── test_manifest_qc.py
│   │   │   ├── test_materialized_outputs.py
│   │   │   └── test_windows_cache.py
│   │   ├── features
│   │   │   ├── __init__.py
│   │   │   └── test_engineering_registry.py
│   │   ├── integration
│   │   │   ├── __init__.py
│   │   │   ├── test_experiment_runner.py
│   │   │   └── test_pipeline_facades.py
│   │   ├── models
│   │   │   ├── __init__.py
│   │   │   ├── test_architectures.py
│   │   │   ├── test_model_cards.py
│   │   │   ├── test_rocket_and_fusion.py
│   │   │   └── test_time_scale_ablation.py
│   │   ├── signal
│   │   │   ├── __init__.py
│   │   │   ├── test_morphology_optical_sqi.py
│   │   │   ├── test_peaks_prv.py
│   │   │   └── test_views_preprocess_imu.py
│   │   ├── training
│   │   │   ├── __init__.py
│   │   │   ├── test_formal_protocol_guards.py
│   │   │   ├── test_oof_aggregation_ablation_bundle.py
│   │   │   └── test_training_isolation.py
│   │   └── __init__.py
│   ├── tools
│   │   ├── acceptance_gate.py
│   │   ├── build_baseline_audit.py
│   │   ├── generate_model_cards.py
│   │   ├── materialize_data_contracts.py
│   │   ├── materialize_reference_configs.py
│   │   ├── run_cpu_ci.py
│   │   ├── run_test_suite.py
│   │   ├── sync_tracking.py
│   │   └── validate_v1.py
│   ├── MIGRATION.md
│   ├── PROJECT_TREE.md
│   ├── pyproject.toml
│   ├── README.md
│   ├── RUNBOOK.md
│   ├── STATUS.md
│   └── WORK_LOG.md
├── M0_history_MA_denoising_detector_HR_feature
│   ├── evidence
│   │   ├── current_binary_detector_balanced_v2
│   │   │   ├── A_external_motion_confusion_matrix.png
│   │   │   ├── A_holdout_motion_confusion_matrix.png
│   │   │   ├── B_external_motion_confusion_matrix.png
│   │   │   ├── B_holdout_motion_confusion_matrix.png
│   │   │   ├── config_summary.json
│   │   │   ├── detector_ab_f1_comparison.png
│   │   │   └── detector_benchmark_summary.json
│   │   ├── current_binary_detector_smoke
│   │   │   ├── A_external_motion_confusion_matrix.png
│   │   │   ├── B_external_motion_confusion_matrix.png
│   │   │   ├── config_summary.json
│   │   │   └── detector_benchmark_summary.json
│   │   ├── EARLY_MULTICLASS_SEARCH_AUDIT.json
│   │   └── MOTION29_DATA_AUDIT.json
│   ├── snapshots
│   │   ├── algorithm_diagrams
│   │   │   ├── 00_PROJECT_HISTORICAL_SIGNAL_FLOW.md
│   │   │   ├── 01_FOUNDATION_FUNCS_PPG.md
│   │   │   ├── 02_V7_TO_STAGE2_EVOLUTION.md
│   │   │   ├── 03_HYBRID_SUITE.md
│   │   │   ├── 04_HEARTBEAT_AND_MOTION_AB.md
│   │   │   ├── 05_SCRIPT_ALGORITHM_ATLAS.md
│   │   │   ├── 06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md
│   │   │   ├── 07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md
│   │   │   └── 08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md
│   │   ├── records
│   │   │   ├── decisions
│   │   │   │   ├── 20260803_m0_activity_motion_supervision.md
│   │   │   │   └── 20260803_m0_madenoiser_route.md
│   │   │   ├── ARCHIVED_CODE_IO_INVENTORY.md
│   │   │   ├── CODE_IO_MASTER_INDEX.md
│   │   │   ├── HUMAN_DECISION_GATES.md
│   │   │   ├── M0_ARCHIVED_LINEAGE_EVIDENCE.md
│   │   │   ├── M0_CODE_OUTPUT_CROSSWALK.md
│   │   │   ├── M0_EXECUTIVE_REPORT.md
│   │   │   ├── M0_METHOD_REGISTRY.md
│   │   │   ├── M0_PAPER_EVIDENCE.md
│   │   │   ├── M0_RISK_REGISTER.md
│   │   │   ├── PROJECT_WIDE_SCAN_FINDINGS.md
│   │   │   ├── ROOT_FILE_IO_INVENTORY.md
│   │   │   └── SCAN_PROTOCOL.md
│   │   └── verification
│   │       ├── inputs
│   │       │   ├── physionet.org.summary.json
│   │       │   └── PPG_Testing_05_01_2026.summary.json
│   │       ├── outputs
│   │       │   ├── CNN_RESULTS.summary.json
│   │       │   ├── denoiser_preview_output.summary.json
│   │       │   ├── results.summary.json
│   │       │   ├── results_denoiser_v8.summary.json
│   │       │   ├── results_frailty3.summary.json
│   │       │   ├── results_hybrid_denoiser.summary.json
│   │       │   ├── results_hybrid_denoiser_raw_imu.summary.json
│   │       │   ├── results_hybrid_denoiser_raw_imu_baseline.summary.json
│   │       │   ├── results_stage1.summary.json
│   │       │   ├── results_stage2.summary.json
│   │       │   ├── results_v72_noleak.summary.json
│   │       │   ├── results_v7_4.summary.json
│   │       │   └── results_v8_audit.summary.json
│   │       ├── ALGORITHM_DIAGRAM_VERIFICATION.json
│   │       ├── ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V2.json
│   │       ├── ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V3.json
│   │       ├── BASELINE_SUMMARY.json
│   │       ├── CODE_DIAGRAM_COVERAGE.json
│   │       ├── CODE_FILES.jsonl
│   │       ├── CODE_PATH_REFERENCES.jsonl
│   │       ├── ROOT_FILES.jsonl
│   │       ├── SCAN_RUNS.jsonl
│   │       ├── SCAN_VERIFICATION.json
│   │       └── TOP_LEVEL_DIRECTORIES.json
│   ├── 00_CURRENT_STATUS_V3.md
│   ├── 01_M0_COMPLETE_RESULTS_AND_DECISIONS.md
│   ├── 02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md
│   ├── 03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md
│   ├── 04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md
│   ├── 05_EVIDENCE_INDEX_AND_PROVENANCE.md
│   ├── 06_M0_PACKAGE_TREE.md
│   ├── 07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md
│   ├── 08_M0_PACKAGE_TREE_V2.md
│   ├── 09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md
│   ├── 10_M0_PACKAGE_TREE_V3.md
│   ├── M0_PACKAGE_VERIFICATION.json
│   ├── M0_PACKAGE_VERIFICATION_V2.json
│   ├── M0_PACKAGE_VERIFICATION_V3.json
│   ├── M0_SOURCE_SNAPSHOT_MANIFEST.json
│   ├── M0_SOURCE_SNAPSHOT_MANIFEST_V2.json
│   ├── M0_SOURCE_SNAPSHOT_MANIFEST_V3.json
│   └── README.md
├── M1_end_to_end_architecture_contract
│   ├── examples
│   │   ├── pipeline_accelerated_arm64.json
│   │   ├── pipeline_high_performance_x86.json
│   │   └── pipeline_value_arm64.json
│   ├── examples_v2
│   │   ├── pipeline_accelerated_arm64_v2.json
│   │   ├── pipeline_high_performance_x86_v2.json
│   │   └── pipeline_value_arm64_v2.json
│   ├── examples_v3
│   │   ├── pipeline_accelerated_arm64_v3.json
│   │   ├── pipeline_high_performance_x86_v3.json
│   │   └── pipeline_value_arm64_v3.json
│   ├── registries
│   │   ├── classifier_registry.json
│   │   ├── feature_extractor_registry.json
│   │   ├── platform_profiles.json
│   │   └── quality_policy_registry.json
│   ├── registries_v2
│   │   ├── classifier_registry_v2.json
│   │   ├── feature_extractor_registry_v2.json
│   │   ├── platform_profiles_v2.json
│   │   └── quality_policy_registry_v2.json
│   ├── registries_v3
│   │   ├── quality_routing_registry_v3.json
│   │   └── quality_routing_registry_v3_active.json
│   ├── schemas
│   │   ├── inference_output.schema.json
│   │   ├── pipeline_config.schema.json
│   │   └── signal_input.schema.json
│   ├── schemas_v2
│   │   ├── inference_output_v2.schema.json
│   │   ├── pipeline_config_v2.schema.json
│   │   └── signal_input_v2.schema.json
│   ├── schemas_v3
│   │   ├── inference_output_v3.schema.json
│   │   └── pipeline_config_v3.schema.json
│   ├── tools
│   │   ├── bootstrap_m1_contract_report.py
│   │   ├── validate_m1_contracts.py
│   │   ├── validate_m1_contracts_v2.py
│   │   ├── validate_m1_contracts_v3.py
│   │   ├── validate_m1_contracts_v3_current.py
│   │   ├── validate_m1_v2_semantic_invariants.py
│   │   └── validate_m1_v3_routing_invariants.py
│   ├── 00_CURRENT_STATUS_V2.md
│   ├── 00_CURRENT_STATUS_V3.md
│   ├── 00_CURRENT_STATUS_V3_1.md
│   ├── 01_END_TO_END_ARCHITECTURE_AND_API.md
│   ├── 02_MOBILE_PLATFORM_PROFILES.md
│   ├── 03_TRAINING_VS_MOBILE_INFERENCE_BOUNDARY.md
│   ├── 04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md
│   ├── 05_VALIDATION_LIMITATIONS_AND_SEMANTIC_GATES.md
│   ├── 06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md
│   ├── M1_CONTRACT_VERIFICATION.json
│   ├── M1_CONTRACT_VERIFICATION_V2.json
│   ├── M1_CONTRACT_VERIFICATION_V3_CURRENT.json
│   ├── M1_PACKAGE_TREE.md
│   ├── M1_PACKAGE_TREE_V2.md
│   ├── M1_PACKAGE_TREE_V3_CURRENT.md
│   ├── M1_ROUTING_INVARIANTS_V3.json
│   ├── M1_SEMANTIC_INVARIANTS_V2.json
│   └── README.md
├── M2_data_manifest_and_evaluation_protocol
│   ├── examples
│   │   └── result_provenance_fixed_epoch_oof_template.json
│   ├── manifests
│   │   ├── external_dataset_manifest.csv
│   │   ├── external_record_manifest.csv
│   │   ├── frailty3_file_manifest.csv
│   │   ├── frailty3_source_anomalies.json
│   │   └── frailty3_subject_manifest.csv
│   ├── registries
│   │   ├── external_dataset_registry.json
│   │   ├── protocol_registry.json
│   │   └── stage_role_registry.json
│   ├── schemas
│   │   ├── dataset_manifest.schema.json
│   │   ├── fold_registry.schema.json
│   │   └── result_provenance.schema.json
│   ├── splits
│   │   ├── frailty3_future_corrected_sgkf5_v2.json
│   │   └── frailty3_historical_sgkf5_sklearn142_bug_v1.json
│   ├── tools
│   │   ├── build_m2_manifests.py
│   │   └── validate_m2_contracts.py
│   ├── 00_CURRENT_STATUS.md
│   ├── 01_DATASET_MANIFEST_AND_PROVENANCE.md
│   ├── 02_STAGE_ROLE_MAPPING.md
│   ├── 03_DUAL_FOLD_REGISTRY_AND_MAIN_PROTOCOL.md
│   ├── 04_EXTERNAL_SYNCHRONIZED_DATA_MANIFEST.md
│   ├── 05_RESULT_PROVENANCE_AND_NAMING_CONTRACT.md
│   ├── M2_BUILD_REPORT.json
│   ├── M2_CONTRACT_VERIFICATION.json
│   ├── M2_PACKAGE_TREE.md
│   └── README.md
├── M3_unified_preprocessing_and_signal_algorithms
│   ├── docs
│   │   ├── 00_CURRENT_STATUS.md
│   │   ├── 01_HISTORICAL_PREPROCESSING_CROSSWALK.md
│   │   ├── 02_PPG_AND_CLEANING_CONTRACT.md
│   │   ├── 03_EKF_PRIMARY_AND_LPF_COMPARATOR_CONTRACT.md
│   │   ├── 04_FOLD_SCALING_AND_M1_M2_BINDING.md
│   │   ├── 05_PEAK_PPI_HR_PRV_CONTRACT.md
│   │   ├── 06_TEST_RESULTS_AND_LIMITATIONS.md
│   │   ├── 07_PROFILE_AND_MOBILE_PLATFORM_BINDING.md
│   │   └── 08_EKF_VS_LPF_RESULTS.md
│   ├── evidence
│   │   ├── ekf_lpf_frailty3_role_proxy.json
│   │   ├── ekf_lpf_synthetic_comparison.json
│   │   ├── filter_response_comparison.json
│   │   ├── frailty3_signal_integrity_summary.json
│   │   ├── historical_preprocessing_crosswalk_v1.json
│   │   └── legacy_peak_parity.json
│   ├── examples
│   │   ├── m1_pipeline_config_m3_mobile.json
│   │   ├── m1_pipeline_config_m3_offline.json
│   │   └── m2_result_provenance_m3_bound.json
│   ├── fixtures
│   │   ├── imu_reference_v1.npy
│   │   ├── ppg_expected_peaks_v1.npy
│   │   ├── ppg_reference_v1.npy
│   │   └── reference_fixture_manifest.json
│   ├── registries
│   │   ├── feature_schemas_v1.json
│   │   ├── historical_preprocessing_crosswalk_v1.json
│   │   ├── module_bindings_v1.json
│   │   ├── physiology_algorithms_v1.json
│   │   ├── preprocessing_profiles_v1.json
│   │   ├── reason_codes_v1.json
│   │   └── status_mapping_v1.json
│   ├── schemas
│   │   ├── fold_fitted_artifact.schema.json
│   │   ├── module_binding.schema.json
│   │   ├── physiology_result.schema.json
│   │   ├── preprocessing_profile.schema.json
│   │   ├── preprocessing_result.schema.json
│   │   ├── ptt_reference_evaluation.schema.json
│   │   ├── reference_fixture_manifest.schema.json
│   │   └── transit_delay_artifact.schema.json
│   ├── src
│   │   └── m3_signal_core
│   │       ├── __pycache__
│   │       │   ├── __init__.cpython-310.pyc
│   │       │   ├── contracts.cpython-310.pyc
│   │       │   ├── fold_contract.cpython-310.pyc
│   │       │   ├── imu.cpython-310.pyc
│   │       │   ├── imu_math.cpython-310.pyc
│   │       │   ├── imu_runtime.cpython-310.pyc
│   │       │   ├── physiology.cpython-310.pyc
│   │       │   ├── ppg.cpython-310.pyc
│   │       │   ├── quality.cpython-310.pyc
│   │       │   ├── reference_evaluation.cpython-310.pyc
│   │       │   ├── registry.cpython-310.pyc
│   │       │   └── scaling.cpython-310.pyc
│   │       ├── __init__.py
│   │       ├── contracts.py
│   │       ├── fold_contract.py
│   │       ├── imu.py
│   │       ├── imu_math.py
│   │       ├── imu_runtime.py
│   │       ├── physiology.py
│   │       ├── ppg.py
│   │       ├── quality.py
│   │       ├── reference_evaluation.py
│   │       ├── registry.py
│   │       └── scaling.py
│   ├── tests
│   │   ├── __init__.py
│   │   ├── _support.py
│   │   ├── test_contract_edges.py
│   │   ├── test_fold_reference.py
│   │   ├── test_imu_physiology.py
│   │   ├── test_legacy_peak_parity.py
│   │   └── test_quality_ppg_scaling.py
│   ├── tools
│   │   ├── __pycache__
│   │   │   └── validate_m3_contracts.cpython-310.pyc
│   │   ├── build_m3_core_evidence.py
│   │   ├── build_m3_frailty_imu_proxy.py
│   │   ├── build_m3_reference_fixtures.py
│   │   ├── legacy_peak_parity.py
│   │   ├── run_m3_reference_tests.py
│   │   └── validate_m3_contracts.py
│   ├── M3_BUILD_REPORT.json
│   ├── M3_REFERENCE_TEST_RESULTS.json
│   └── README.md
├── records
│   ├── decisions
│   │   ├── 20260803_m0_activity_motion_supervision.md
│   │   ├── 20260803_m0_madenoiser_route.md
│   │   ├── 20260814_m1_architecture_mobile_profiles.md
│   │   ├── 20260814_m1_architecture_mobile_profiles_v2.md
│   │   ├── 20260815_m1_quality_routing_v3.md
│   │   ├── 20260815_m2_dual_registry_and_stage_mapping.md
│   │   └── 20260815_m3_unified_preprocessing_contract.md
│   ├── generated
│   │   ├── inputs
│   │   │   ├── datasets.jsonl
│   │   │   ├── datasets.summary.json
│   │   │   ├── physionet.org.jsonl
│   │   │   ├── physionet.org.summary.json
│   │   │   ├── PPG_Testing_05_01_2026.jsonl
│   │   │   ├── PPG_Testing_05_01_2026.summary.json
│   │   │   ├── train_labeled.jsonl
│   │   │   ├── train_labeled.summary.json
│   │   │   ├── train_raw.jsonl
│   │   │   ├── train_raw.summary.json
│   │   │   ├── train_val.jsonl
│   │   │   ├── train_val.summary.json
│   │   │   ├── train_window.jsonl
│   │   │   └── train_window.summary.json
│   │   ├── outputs
│   │   │   ├── .CNN_results.jsonl
│   │   │   ├── .CNN_results.summary.json
│   │   │   ├── denoiser_preview_output.jsonl
│   │   │   ├── denoiser_preview_output.summary.json
│   │   │   ├── models.jsonl
│   │   │   ├── models.summary.json
│   │   │   ├── results.jsonl
│   │   │   ├── results.summary.json
│   │   │   ├── results_denoiser_v8.jsonl
│   │   │   ├── results_denoiser_v8.summary.json
│   │   │   ├── results_detector_v8.jsonl
│   │   │   ├── results_detector_v8.summary.json
│   │   │   ├── results_frailty3.jsonl
│   │   │   ├── results_frailty3.summary.json
│   │   │   ├── results_hybrid_denoiser.jsonl
│   │   │   ├── results_hybrid_denoiser.summary.json
│   │   │   ├── results_hybrid_denoiser_raw_imu.jsonl
│   │   │   ├── results_hybrid_denoiser_raw_imu.summary.json
│   │   │   ├── results_hybrid_denoiser_raw_imu_baseline.jsonl
│   │   │   ├── results_hybrid_denoiser_raw_imu_baseline.summary.json
│   │   │   ├── results_stage1.jsonl
│   │   │   ├── results_stage1.summary.json
│   │   │   ├── results_stage2.jsonl
│   │   │   ├── results_stage2.summary.json
│   │   │   ├── results_v72_noleak.jsonl
│   │   │   ├── results_v72_noleak.summary.json
│   │   │   ├── results_v7_3.jsonl
│   │   │   ├── results_v7_3.summary.json
│   │   │   ├── results_v7_4.jsonl
│   │   │   ├── results_v7_4.summary.json
│   │   │   ├── results_v8_audit.jsonl
│   │   │   ├── results_v8_audit.summary.json
│   │   │   ├── test_asa_classifier.jsonl
│   │   │   └── test_asa_classifier.summary.json
│   │   ├── ALGORITHM_DIAGRAM_VERIFICATION.json
│   │   ├── BASELINE_SUMMARY.json
│   │   ├── CODE_DIAGRAM_COVERAGE.json
│   │   ├── CODE_FILES.jsonl
│   │   ├── CODE_PATH_REFERENCES.jsonl
│   │   ├── FINAL_V0_VERIFICATION.json
│   │   ├── ROOT_FILES.jsonl
│   │   ├── SCAN_RUNS.jsonl
│   │   ├── SCAN_VERIFICATION.json
│   │   ├── TOP_LEVEL_DIRECTORIES.json
│   │   └── WORKSPACE_FILES.jsonl
│   ├── log_entries
│   │   ├── 20260802_all_code_diagrams_verified.md
│   │   ├── 20260802_archived_code_inventory_and_diagrams.md
│   │   ├── 20260802_code_master_index_and_coverage_tool.md
│   │   ├── 20260802_delivery_verifier_added.md
│   │   ├── 20260802_detailed_tree_indexer_added.md
│   │   ├── 20260802_m0_algorithm_atlas_added.md
│   │   ├── 20260802_m0_algorithm_diagrams_verified.md
│   │   ├── 20260802_m0_archived_lineage_evidence.md
│   │   ├── 20260802_m0_crosswalk_counts_corrected.md
│   │   ├── 20260802_m0_crosswalk_risks_gates.md
│   │   ├── 20260802_m0_dash_filename_corrected.md
│   │   ├── 20260802_m0_final_verification.md
│   │   ├── 20260802_m0_full_scan_verified.md
│   │   ├── 20260802_m0_method_registry.md
│   │   ├── 20260802_m0_reports_batch1.md
│   │   ├── 20260802_project_and_non_m0_root_diagrams.md
│   │   ├── 20260802_root_file_io_inventory.md
│   │   ├── 20260802_scan_and_delivery_preverification.md
│   │   ├── 20260802_scan_verifier_added.md
│   │   ├── 20260802_scanner_added.md
│   │   ├── 20260802_session_baseline.md
│   │   ├── 20260803_activity_motion_supervision_and_history.md
│   │   ├── 20260803_m0_candidate_future_direction_heading.md
│   │   ├── 20260803_m0_candidate_routes_catalog.md
│   │   ├── 20260803_m0_evidence_provenance_index.md
│   │   ├── 20260803_m0_five_family_diagrams.md
│   │   ├── 20260803_m0_five_method_families_audit.md
│   │   ├── 20260803_m0_history_package_core_results.md
│   │   ├── 20260803_m0_history_package_snapshots.md
│   │   ├── 20260803_m0_unified_benchmark_contract.md
│   │   ├── 20260803_madenoiser_confirmed_route.md
│   │   ├── 20260814_m1_architecture_contract.md
│   │   ├── 20260814_m1_architecture_contract_v2.md
│   │   ├── 20260814_m1_architecture_contract_v2_verification_correction.md
│   │   ├── 20260815_m1_quality_routing_v3.md
│   │   ├── 20260815_m2_manifest_dual_fold_protocol.md
│   │   ├── 20260815_m3_contract_edge_tests_phase7.md
│   │   ├── 20260815_m3_core_evidence_builder_phase8.md
│   │   ├── 20260815_m3_core_phase1.md
│   │   ├── 20260815_m3_d8_symmetric_scorecard_phase21.md
│   │   ├── 20260815_m3_d8_training_split_identity_phase24.md
│   │   ├── 20260815_m3_decision_contract_phase16.md
│   │   ├── 20260815_m3_deprecated_profile_fail_closed_phase29.md
│   │   ├── 20260815_m3_deprecated_profile_tests_phase30.md
│   │   ├── 20260815_m3_evidence_authority_phase17.md
│   │   ├── 20260815_m3_fixture_manifest_contract_phase25.md
│   │   ├── 20260815_m3_fixture_manifest_regeneration_phase26.md
│   │   ├── 20260815_m3_fold_and_reference_tests_phase12.md
│   │   ├── 20260815_m3_fold_artifact_envelope_phase22.md
│   │   ├── 20260815_m3_fold_registry_field_correction.md
│   │   ├── 20260815_m3_fold_robust_scaling_phase20.md
│   │   ├── 20260815_m3_fold_schema_alignment_phase23.md
│   │   ├── 20260815_m3_frailty_imu_proxy_builder_phase9.md
│   │   ├── 20260815_m3_historical_discovery_phase19.md
│   │   ├── 20260815_m3_imu_core_phase2a.md
│   │   ├── 20260815_m3_legacy_peak_parity_phase15.md
│   │   ├── 20260815_m3_m2_fold_artifact_binding_phase10.md
│   │   ├── 20260815_m3_physiology_core_phase2b.md
│   │   ├── 20260815_m3_physiology_provenance_phase13.md
│   │   ├── 20260815_m3_ppg_raw_repaired_views_phase18.md
│   │   ├── 20260815_m3_profile_and_physiology_corrections_phase5.md
│   │   ├── 20260815_m3_profile_locked_peak_and_resampling_phase27.md
│   │   ├── 20260815_m3_profile_locked_peak_and_resampling_tests_phase28.md
│   │   ├── 20260815_m3_ptt_ecg_reference_evaluator_phase11.md
│   │   ├── 20260815_m3_reference_report_snapshot_phase14.md
│   │   ├── 20260815_m3_reference_test_corrections.md
│   │   ├── 20260815_m3_reference_tests_phase3.md
│   │   ├── 20260815_m3_sim_256_resampling_fixture_phase31.md
│   │   ├── 20260815_m3_sim_256_resampling_tests_phase32.md
│   │   └── 20260815_m3_stateful_imu_runtime_corrections_phase6.md
│   ├── pending_agent_updates
│   │   ├── BASELINE_AND_M0_FINAL_TOPICS.md
│   │   ├── M0_ACTIVITY_MOTION_SUPERVISION_DRAFT.md
│   │   ├── M0_DRAFT_TOPICS.md
│   │   ├── M0_FIVE_FAMILY_EXTENSION_DRAFT.md
│   │   ├── M0_MADENOISER_CONFIRMED_ROUTE_DRAFT.md
│   │   ├── M1_ARCHITECTURE_CONTRACT_DRAFT.md
│   │   ├── M1_ARCHITECTURE_CONTRACT_V2_DRAFT.md
│   │   ├── M1_QUALITY_ROUTING_V3_DRAFT.md
│   │   ├── M2_DATA_MANIFEST_PROTOCOL_DRAFT.md
│   │   └── M3_UNIFIED_PREPROCESSING_DRAFT.md
│   ├── ARCHIVED_CODE_IO_INVENTORY.md
│   ├── CODE_IO_MASTER_INDEX.md
│   ├── HUMAN_DECISION_GATES.md
│   ├── M0_ARCHIVED_LINEAGE_EVIDENCE.md
│   ├── M0_CODE_OUTPUT_CROSSWALK.md
│   ├── M0_EXECUTIVE_REPORT.md
│   ├── M0_METHOD_REGISTRY.md
│   ├── M0_PAPER_EVIDENCE.md
│   ├── M0_RISK_REGISTER.md
│   ├── PENDING_AGENT_UPDATES.md
│   ├── PROJECT_WIDE_SCAN_FINDINGS.md
│   ├── ROOT_FILE_IO_INVENTORY.md
│   ├── SCAN_PROTOCOL.md
│   └── WORK_LOG.md
├── tools
│   ├── add_bilingual_inline_comments_to_scan_verifier.py
│   ├── build_m0_history_package.py
│   ├── build_m0_history_package_v3.py
│   ├── correct_archived_inventory_details.py
│   ├── correct_m0_crosswalk_manifest_counts.py
│   ├── correct_m0_dash_filename_reference.py
│   ├── sync_algorithm_index.py
│   ├── sync_tracking_docs.py
│   ├── update_final_v0_index.py
│   ├── update_final_v0_index_detailed.py
│   ├── verify_algorithm_diagrams.py
│   ├── verify_code_diagram_coverage.py
│   ├── verify_final_v0_delivery.py
│   ├── verify_scan_evidence.py
│   └── workspace_audit.py
├── FINAL_V0_TREE.md
└── README.md
```

## 逐文件内容与完整性 / Per-file content and integrity

| 文件 / File | 字节 / Bytes | SHA-256 | 内容详细说明 / Detailed content description |
|---|---:|---|---|
| `M0_history_MA_denoising_detector_HR_feature/00_CURRENT_STATUS_V3.md` | 2434 | `17d35975bd1ae6e717e555a14ca1877306e3ccc9582f725aee183fd7d47900b1` | 文档《M0 当前状态 v3：Activity/Motion 监督已确认》；1. 监督已确认：`B/R→static`，`S/W→motion`；S 为 stand-and-sit 往复，W 为 walking。；2. 目标名称是 activity/motion state，不是 optical-artifact ground truth。 |
| `M0_history_MA_denoising_detector_HR_feature/01_M0_COMPLETE_RESULTS_AND_DECISIONS.md` | 14337 | `af2c221f18aa6740149ab923454961ea354e003a9303942528e8dfb0fd1e4ead` | 文档《M0 完整结果、算法结论与路线决定》；M0 对 workspace 的历史 motion artifact、动态降噪、heartbeat/IBI/HR 和相关 detector 路线进行了代码—输入—输出—历史记录四方审计。本归档是在既有 M0 完成报告基础上的扩展，补充五类用户指定方向及其可实现性；它不重跑训练、不修复根目录代码，也不提前执行 M1–M10。；当前状态：`M0_local_audit_complete_pending_user_acceptance`。 |
| `M0_history_MA_denoising_detector_HR_feature/02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md` | 21353 | `adae474bcff301454b75a0b9872c04ba45614e54a8712cd48aa916f4481630cc` | 文档《Motion Detector、Denoising 与动态 HR 候选脚本总表》；本文件是用户指定的新候选路线文档。它登记三个问题中所有仍有工程或 benchmark 价值的脚本，并明确区分：；**主候选**：值得在统一协议下优先实现/复测； |
| `M0_history_MA_denoising_detector_HR_feature/03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md` | 29845 | `99a728f93c38b159a4b430b21c27d1427d1ce9886f86dbd58e96842556fa031c` | 文档《五类方法：代码、算法、应用、测试与可实现性审计》；本报告对 52 份现行/归档代码与 notebook 做完整字节校验，并以关键函数逐行复核、全项目符号检索、实际输入 header、输出 JSON/CSV/Markdown 和既有 M0 证据交叉确认。三项并行只读审计分别覆盖：；1. 自适应滤波 + 非平稳分解； |
| `M0_history_MA_denoising_detector_HR_feature/04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md` | 15105 | `ea1f3f1ee7a7d9778e9fd7cd06a4ea189ca3a290416a6f2f5684653bb5f11b1e` | 文档《五类路线统一实现与 Benchmark 合同》；本合同把“可以实现”转成可执行的模块边界、测试先决条件、数据切分、指标、输出 schema 和路线淘汰门。它不是新算法结果，也不授权立即编码；实际实施必须等待用户确认当前 M0 扩展。；核心原则：先统一数据与测试，再比较路线；测试不能晚于模型写完才临时补。所有路线允许拒绝输出，拒绝率/coverage 是主指标之一。 |
| `M0_history_MA_denoising_detector_HR_feature/05_EVIDENCE_INDEX_AND_PROVENANCE.md` | 11465 | `7de41e07b9abaf3af10a91110e44597222f587a10720f2dd11b2e68e9c91f91d` | 文档《M0 证据索引与来源链》；冲突时按以下顺序处理：；1. 当前源代码的实际执行分支、参数与写出语句； |
| `M0_history_MA_denoising_detector_HR_feature/06_M0_PACKAGE_TREE.md` | 15519 | `f64eaf6447835f74852ac0eae8d16585a3c6ce7b8095ba78a9dc03c30f8344ba` | 文档《M0 历史归档文件树与逐文件说明 / M0 Package Tree and Per-file Descriptions》；永久文件数（含本索引）/ Permanent files including this index：**52**。；历史快照刷新必须显式 `--refresh` 且应先取得用户确认。 |
| `M0_history_MA_denoising_detector_HR_feature/07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md` | 12054 | `84b5ee74268985ab0871699b90d36bdc1c378e84b7ac25e3d272b3afc9a211da` | 文档《MAdenoiser 已确认后续路线：从 SQI、Motion-29 到 Frailty 特征选择》；决策 ID：`M0-MAD-001`；决策日期：2026-08-03 |
| `M0_history_MA_denoising_detector_HR_feature/08_M0_PACKAGE_TREE_V2.md` | 17644 | `d41d1599dbcdf2844e343ac85a3d4a4e49ba261ceefe17b97d2c5adadca64e3e` | 文档《M0 历史归档文件树与逐文件说明 / M0 Package Tree and Per-file Descriptions》；永久文件数（含本索引）/ Permanent files including this index：**59**。；历史快照刷新必须显式 `--refresh` 且应先取得用户确认。 |
| `M0_history_MA_denoising_detector_HR_feature/09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md` | 16700 | `81b82f55729b5f66ce4b6f977e249205ca5d03a37c5a2c728275c2986d800928` | 文档《29 人 Activity/Motion 监督合同、早期三分类历史与运动后恢复特征路线》；决策 ID：`M0-MOT-001`；日期：2026-08-03 |
| `M0_history_MA_denoising_detector_HR_feature/10_M0_PACKAGE_TREE_V3.md` | 23879 | `9a92c92c8101ce5f2141a4cdf7b4249b2f471bdf216bec74eabe6eb522fb3d4c` | 文档《M0 v3 文件树与逐文件说明 / M0 v3 Tree and Per-file Descriptions》；永久文件数（含本索引）：**80**。；v1/v2历史文件保持原字节；v3只追加新决定、算法图、验证和证据。 |
| `M0_history_MA_denoising_detector_HR_feature/M0_PACKAGE_VERIFICATION.json` | 302 | `0c580a7b316d9cfb34e2920fb3bf5d7907f3dad531766b9735e68752cb2efb7c` | 机器可读 JSON 证据 `M0_PACKAGE_VERIFICATION.json`；status=pass; mermaid_block_count=35 |
| `M0_history_MA_denoising_detector_HR_feature/M0_PACKAGE_VERIFICATION_V2.json` | 302 | `83fd3894c5060a5bb243359f31a40378c080157d1526c6f38a5e8d7c11bb90fb` | 机器可读 JSON 证据 `M0_PACKAGE_VERIFICATION_V2.json`；status=pass; mermaid_block_count=39 |
| `M0_history_MA_denoising_detector_HR_feature/M0_PACKAGE_VERIFICATION_V3.json` | 319 | `8b04a1f0dbab74197221c6cdb26639b452d47a2ff8fbccba18babc0a496243f1` | 机器可读 JSON 证据 `M0_PACKAGE_VERIFICATION_V3.json`；status=pass; mermaid_block_count=42 |
| `M0_history_MA_denoising_detector_HR_feature/M0_SOURCE_SNAPSHOT_MANIFEST.json` | 16815 | `4e09ca1f82c97cf82b4b63c0f03884a0e4338237e5fec21af0fbc6811532ba07` | 机器可读 JSON 证据 `M0_SOURCE_SNAPSHOT_MANIFEST.json`；status=pass; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/M0_SOURCE_SNAPSHOT_MANIFEST_V2.json` | 17655 | `ec86d163cfd78b46d024f8ad875cc6fb4c51dd01c89cacb3ce847da88b12f551` | 机器可读 JSON 证据 `M0_SOURCE_SNAPSHOT_MANIFEST_V2.json`；status=pass; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/M0_SOURCE_SNAPSHOT_MANIFEST_V3.json` | 18432 | `5225b2653112c4595541bc8d72226d43dd44e6a1ea6962ddddd8517dff28cc7b` | 机器可读 JSON 证据 `M0_SOURCE_SNAPSHOT_MANIFEST_V3.json`；status=pass; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/README.md` | 6152 | `1571c8ba8b19f7afc41bce1aae011b61840cdb55ac9f81dc49810c5b20b1de2f` | 文档《M0 历史 Motion Artifact、降噪、检测器与动态 HR 归档》；用户指定显示名称：`M0_**history\_MA\_denoising\_detector\_HR\_feature**`；实际安全目录名：`M0_history_MA_denoising_detector_HR_feature` |
| `M0_history_MA_denoising_detector_HR_feature/evidence/EARLY_MULTICLASS_SEARCH_AUDIT.json` | 1780 | `a7f9e6e9431cc6d7921a3081927ea1cacbf4814233d1ac4fb1b541cd0a667ad8` | 机器可读 JSON 证据 `EARLY_MULTICLASS_SEARCH_AUDIT.json`；status=verified_svm_assets_three_class_cnn_not_found |
| `M0_history_MA_denoising_detector_HR_feature/evidence/MOTION29_DATA_AUDIT.json` | 2465 | `14cac2f40806b8d53d2438186799de2a1558e1b2e38e94a11b7f9f5b120b1c5b` | 机器可读 JSON 证据 `MOTION29_DATA_AUDIT.json`；status=pass |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/A_external_motion_confusion_matrix.png` | 39752 | `4151bbe6a161078f01560958afe9d038165ec26d2b2b827222983ceacdfb04de` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/A_holdout_motion_confusion_matrix.png` | 33922 | `1db8c723594c0953be349786514f509bb3c7e417bfe57032bf4f62a21e20115d` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/B_external_motion_confusion_matrix.png` | 39449 | `bcdb9e8d3a293ea6f52cc20bda715be44ca368fdac312b4d749ef375059ba956` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/B_holdout_motion_confusion_matrix.png` | 32856 | `8980dfb801ec2cca3f6c3247aeaacd0af29f936dbd2ca27a47bced22b9c9a74c` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/config_summary.json` | 55902 | `3b6d6cf3f4f0e0cef1c7d33ce3e01dc4379bfeb761d4408d6fd716cc02b1e3ba` | 机器可读 JSON 证据 `config_summary.json`；keys=results_root,run_name,cv_folds,final_train_epochs,target_fs,win_sec,hop_sec,batch_size,lr,seed |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/detector_ab_f1_comparison.png` | 31849 | `4f75d3969a25e7dd0b526ae76f55d520b83551e9663756e46c44a59a7a23cd6b` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_balanced_v2/detector_benchmark_summary.json` | 29140 | `0a34ae2656aac00a3cdeb789e582981745c5322ba90d3cf36d108ae925c8f7a0` | 机器可读 JSON 证据 `detector_benchmark_summary.json`；status=completed |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_smoke/A_external_motion_confusion_matrix.png` | 38584 | `8b6df160dc7695b92c9cc057cda8f39c019fc8e58bddea25c0dee3f5f79471a4` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_smoke/B_external_motion_confusion_matrix.png` | 32815 | `dea2591f62ae90555a8b035f47898b36ba19ca1debf6aaedc2df29fc27155744` | PNG 文件；用途由路径和相邻审计记录定义 |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_smoke/config_summary.json` | 5518 | `80fe885e246e3381dca0d6b0977e596417b3c7e62259ed4502aa5f47d885ce3b` | 机器可读 JSON 证据 `config_summary.json`；keys=results_root,run_name,cv_folds,final_train_epochs,target_fs,win_sec,hop_sec,batch_size,lr,seed |
| `M0_history_MA_denoising_detector_HR_feature/evidence/current_binary_detector_smoke/detector_benchmark_summary.json` | 26326 | `8eeeab19041d9738db9c9e976cbdf02f4dd3aaf3b245406da21f15137a27dabc` | 机器可读 JSON 证据 `detector_benchmark_summary.json`；status=completed |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md` | 2684 | `1ed28714c1f7f46a779f259a31f940dc26e18deaf4c08e3b20e5aea1caf5a9e4` | 文档《项目历史信号处理总图 / Historical Signal-Processing Map》；本图把 M0 审计到的历史输入、五类 motion/heartbeat 路线、实际证据和最终研究决策放在同一条可追溯链上。实线表示运行数据流；虚线表示监督、评价或审计引用，不表示部署时输入。；ECG 在图中均以虚线进入历史路线，表示它应当只作为监督/评价reference；v7 setup2 把它变成实质推理输入，是已登记的critical leakage。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/01_FOUNDATION_FUNCS_PPG.md` | 1807 | `c8586eb04bfe58fc3ff56f09e1d0221da600aa853312144ad3cc03638018f431` | 文档《M0 基础函数与 Dash 算法图 / Foundation Functions and Dash Flow》；基础算法透明且可作为 M3 候选，但当前双实现、参数错位、边界coverage和runtime默认路径使其处于 `implemented_unverified`；不得直接当作统一公共实现。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/02_V7_TO_STAGE2_EVOLUTION.md` | 3068 | `53d16384e83b8ce98ffafd2dc3107be803f5dee76a19a5a2f0175d5e981efb88` | 文档《v7 至 Stage-2 演化图 / v7-to-Stage-2 Evolution》 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/03_HYBRID_SUITE.md` | 2256 | `de5bb69fc9fe0d2513986380f2c9d87b9ef88aa53b631af56a6311a1b671944f` | 文档《Hybrid 去噪、导出与运行图 / Hybrid Denoiser Suite》 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/04_HEARTBEAT_AND_MOTION_AB.md` | 2515 | `1d9a75b196e673209d6867468a90e96bb6c99c4893deb62e6690822582cae835` | 文档《Heartbeat 与 PPG+IMU Motion A/B 图 / Heartbeat and Motion A/B》；P01 与 P02 虽在同一训练脚本中，但输入、target和证据不同：P01 是 PPG-only 的历史 heartbeat尝试；P02 才是 PPG+IMU motion-state benchmark。后续不得把 P02 的外部性能解释为 P01 gate 或 peak 的性能。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/05_SCRIPT_ALGORITHM_ATLAS.md` | 6934 | `d6681b04fc2c0235be1669ef026001774ddee3f681056e47b77e3629963685d4` | 文档《M0 逐脚本算法结构图册 / Per-script Algorithm Atlas》；本图册覆盖 M0 范围内承担算法、训练、评价、导出或运行职责的每个脚本。每节只画该文件的直接职责；跨脚本关系见上级总图。；以上16个入口覆盖 M0 中实际承担算法或运行职责的脚本。`funcs.py` 与 `ppg.py` 的重复实现被分别绘制；同一大脚本中的 PPG-only heartbeat 与 PPG+IMU A/B 被画成两条分支，防止证据混用。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md` | 5558 | `dec93e8d4901361ec78c512658ca08ce9bc17657851a7a0d9d3036a40aac1a12` | 文档《M0 五类方法、统一实现与 Benchmark 算法图》；本图把本轮五类扩展审计连接到三个工程问题：motion detector、denoising 和动态 HR。实线表示未来可执行数据流；虚线表示监督、评价、安全门或失败历史。所有“未实现”节点只是已定义的实现路线，不是现有结果。；已存在且可复用：STFT/ISTFT utilities、部分 IMU preprocessing、P02 detector benchmark、hybrid工程链、当前SQI消融框架。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md` | 3042 | `1cb90ecec9db32867f424c3db82aac91bac53281e075067a26807119690c5905` | 文档《MAdenoiser 已确认路线到 Frailty 特征选择算法图》；状态：路线已确认；实现、训练与 benchmark 尚未开始。；实线表示用户确认的未来执行顺序，不表示已完成。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/algorithm_diagrams/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md` | 2776 | `3dafee689821990a7318cb2b4c5510a99623e8c4d2855107324bef59da2b7e3a` | 文档《Activity/Motion 迁移重训、SQI 与恢复特征流程图》；`Rk` 与它前面的实际活动配对，不按 S/W 编号硬编码。；`p_active` 是 activity probability，不是 optical-artifact probability。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/ARCHIVED_CODE_IO_INVENTORY.md` | 6837 | `cf81c62086fa40812905f28ec7c27e52ad67409e69e699b390e7183345ded08a` | 文档《非根归档代码逐文件 I/O 与版本关系 / Archived Code I/O and Lineage Inventory》；状态 / Status：`complete; historical_only`；覆盖 / Coverage：`CODE_FILES.jsonl` 中全部23个非根代码/Notebook，逐字节复扫并核对SHA；所有 `.py` 静态编译通过。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/CODE_IO_MASTER_INDEX.md` | 3761 | `78e60dda2d6d166b5e90e0e861ee47a2e1137f066adbc192bfcbd867fce95938` | 文档《52份代码/Notebook I/O 总索引 / Master Code and Notebook I/O Index》；状态 / Status：`complete`；证据 / Evidence：`CODE_FILES.jsonl` 52行，全部逐字节至EOF并记录SHA；根目录29份、非根归档23份。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/HUMAN_DECISION_GATES.md` | 3170 | `093a37653cde9afba230c8dcd1a4a3e8f2cb1aa2ff81b500620d8a9f66eff4dc` | 文档《人工决策门 / Human Decision Gates》；状态 / Status：`awaiting_user_decisions`；规则 / Rule：发现会改变研究主线、论文口径、数据cohort或依赖范围的选择时停止；这里只记录选项与影响，不代替用户决定。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/M0_ARCHIVED_LINEAGE_EVIDENCE.md` | 3106 | `17e5f4b33e2d225dc72a2e9a42ae5c7e592030023debbcce2ced7182ac6cc331` | 文档《M0 归档版本与实际输出生产关系 / Archived Lineage and Output Provenance》；状态 / Status：`complete_supplement`；目的 / Purpose：解释中间修复版本与现存目录的精确关系；不把修复链重复登记为独立科学方法。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/M0_CODE_OUTPUT_CROSSWALK.md` | 9419 | `432718ff0f46c140e3a0ca6729d6e9548fb822be2520d00c0232f65e4d89a159` | 文档《M0 代码—输入—输出对应表 / Code–Input–Output Crosswalk》；状态 / Status：`complete`；对应范围 / Scope：M0 Motion Artifact、动态降噪、Heartbeat 与其公共基础函数。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/M0_EXECUTIVE_REPORT.md` | 9014 | `cc297c4e12f68d0858f0b53591d299708993f402085f8a5fba809c00d4bda0d6` | 文档《M0 执行、算法与结果总报告 / M0 Execution, Algorithm, and Results Report》；TODO：M0 完整审计历史 Motion Artifact、动态降噪和 Heartbeat 路线；状态 / Status：`complete` |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/M0_METHOD_REGISTRY.md` | 17119 | `7b4e4f13bebc964fcf2ebe87a258d08629c2922c9ad7895756712d63944f3cd6` | 文档《M0 Motion Processing Method Registry》；状态 / Status：`complete`；证据来源 / Evidence：逐字节代码读取、AST/逐行审计、输入头部 manifests、输出 EOF manifests、实际 JSON/CSV/Markdown、历史项目记录。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/M0_PAPER_EVIDENCE.md` | 5977 | `c2e6c988bd827a500dbdca97b1dd6be13cc036088def9d45167200b13b5a70f1` | 文档《M0 论文证据、结果评价与表述边界 / Paper Evidence and Claim Boundaries》；状态 / Status：`confirmed_by_code_and_outputs`；目的 / Purpose：区分可写入论文的事实、只能作为探索性结果的证据，以及禁止作出的性能声明。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/M0_RISK_REGISTER.md` | 7541 | `14d212d20bff712802b55696918cdb34fb7d37da94366055f645ec83c192e540` | 文档《M0 风险登记 / Risk Register》；状态 / Status：`complete`；分级 / Severity：`critical` 会使结论无效或运行阻断；`high` 会显著偏置结果/部署；`medium` 限制泛化、可复现性或解释；`low` 为文档/工程质量风险。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/PROJECT_WIDE_SCAN_FINDINGS.md` | 4315 | `a38e5401967d3b56034ea283dc9bc069f70fa2afc9fd3ad3dfbfe3b090dcf2d6` | 文档《Workspace 全项目扫描发现 / Project-wide Scan Findings》；状态 / Status：`baseline_complete; future_TODO_items_not_executed`；目的 / Purpose：保存 M0 前置全量扫描中发现、但属于 M1–M10 的事实，避免后续重复扫描后遗失上下文。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/ROOT_FILE_IO_INVENTORY.md` | 11314 | `9a1550d161a251839187d41f00ccfe0d2022be19f3f0f6e79c872655effdae2e` | 文档《根目录逐文件 I/O 与内容清单 / Root-file I/O and Content Inventory》；状态 / Status：`complete`；覆盖 / Coverage：workspace 根目录 45 个文件，逐份完整读取或按非文本规则登记；其中29个代码/Notebook也在 `CODE_FILES.jsonl` 中逐字节校验。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/SCAN_PROTOCOL.md` | 2143 | `82444b736ceedac4293f62ebee2d2e59c01394986a367faef0b8fca13e749180` | 文档《扫描协议与证据要求 / Scan Protocol and Evidence Requirements》；1. 根目录代码和文本文件逐文件、逐字节完整读取。；2. 从代码中提取并记录全部可识别的输入路径、输出路径、输出内容及输出结构。 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/decisions/20260803_m0_activity_motion_supervision.md` | 1905 | `f4f10640e4926a70b9e470bcf33f65aafa22b2ca3534f0b499abbc1495e75580` | 文档《M0-MOT-001 — 29-subject Activity/Motion 监督与时序特征决定》；日期 / Date：2026-08-03；状态 / Status：`confirmed` |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/records/decisions/20260803_m0_madenoiser_route.md` | 1834 | `ca93169b3a0167598ddebbaa7c8d0d95ee8ff50f30703c2ae6791d0653ac57a3` | 文档《M0-MAD-001 — MAdenoiser 后续路线决定》；日期 / Date：2026-08-03；状态 / Status：`confirmed` |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION.json` | 2850 | `c5b8d606ddaee50c3c366d2f720fe70f724ea2c97823f4ab5e38b5acd7b3fec1` | 机器可读 JSON 证据 `ALGORITHM_DIAGRAM_VERIFICATION.json`；status=pass; diagram_file_count=11; mermaid_block_count=73; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V2.json` | 3112 | `9aa45ec2a7d7ea11a3d1ff5240bbfe00dcea578bbb34f5eb46d482b6cbf17517` | 机器可读 JSON 证据 `ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V2.json`；status=pass; diagram_file_count=12; mermaid_block_count=77; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V3.json` | 3375 | `5d275bc757e44fa145ca466d2e20684d5b2124cc9d25407eb5b368e7f9bc8983` | 机器可读 JSON 证据 `ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V3.json`；status=pass; diagram_file_count=13; mermaid_block_count=80; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/BASELINE_SUMMARY.json` | 375 | `8cb2a6ead19c107cefb2dd1f4e1f65fc8533576a1d87749e360a58aa252af881` | 机器可读 JSON 证据 `BASELINE_SUMMARY.json`；error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/CODE_DIAGRAM_COVERAGE.json` | 6904 | `44f522b54d717a099543298506094b2fb4c02713678a3bfc44faacd0f3b965ff` | 机器可读 JSON 证据 `CODE_DIAGRAM_COVERAGE.json`；status=pass; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/CODE_FILES.jsonl` | 57269 | `92f9b0328fced5a72922cfc8eea8c0bd8fcc44f97090d5b0ac4f0a09d50ec2d0` | 全部代码/notebook逐字节读取与结构 manifest；52 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/CODE_PATH_REFERENCES.jsonl` | 767183 | `af487aca29c73018988af74e0805ff4e84c15142cfb61775b8d6d76e714b8132` | 代码静态输入/输出路径字符串引用清单；2387 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/ROOT_FILES.jsonl` | 37555 | `e2843bac57e526da14c8cd7ed9fce604ceb48591997008ba732e58d1e1956bd1` | 根目录逐文件完整读取 manifest；45 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/SCAN_RUNS.jsonl` | 8377 | `c43da31b5de360796888bbc61b51539664490000d62bc98ab8202b8af66c9fbf` | baseline、输入和输出扫描事务账本；25 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/SCAN_VERIFICATION.json` | 4259 | `e562b99a17854050defefef45c4caacc52470420706fc27429d697fe0c52e0df` | 机器可读 JSON 证据 `SCAN_VERIFICATION.json`；status=pass; failures=list[0] |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/TOP_LEVEL_DIRECTORIES.json` | 7081 | `6b79fd6b4f78a71e8cb4ed7a67292e3917162c530c5e3e2e7157e492c4c72ebf` | 机器可读 JSON 证据 `TOP_LEVEL_DIRECTORIES.json`；top-level=list, items=32 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/inputs/PPG_Testing_05_01_2026.summary.json` | 277 | `7f16b8e6c23016576fe547476813cd34980255caf3388527a4e89d225415e135` | 机器可读 JSON 证据 `PPG_Testing_05_01_2026.summary.json`；file_count=1134; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/inputs/physionet.org.summary.json` | 288 | `1875d00e52c9329d04d81b8819b366ae5a5a1716b234876118b25c9ea570c151` | 机器可读 JSON 证据 `physionet.org.summary.json`；file_count=4920; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/CNN_RESULTS.summary.json` | 352 | `e2e864632044318a4ceb8773dcf136528192e5fc8d1602970ba1180d3f6a63ba` | 机器可读 JSON 证据 `CNN_RESULTS.summary.json`；file_count=687; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/denoiser_preview_output.summary.json` | 258 | `f2ca400b7ca021c7df49f046cf4a71978da2f0a7eadff6aba9454dfba78e58cb` | 机器可读 JSON 证据 `denoiser_preview_output.summary.json`；file_count=8; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results.summary.json` | 240 | `d642178e58439226705345a0bf88044dc90c395d65f8d487ae48c52f8e1c323f` | 机器可读 JSON 证据 `results.summary.json`；file_count=5; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_denoiser_v8.summary.json` | 231 | `ebf1f8d3f80e47c68f689c58251d9b03833330ad45c88e8f6e0454f061ca9e59` | 机器可读 JSON 证据 `results_denoiser_v8.summary.json`；file_count=0; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_frailty3.summary.json` | 319 | `e4432c91530a63009313e819cbdb0dabc40e9fa3af8468ee039b93b02fbec241` | 机器可读 JSON 证据 `results_frailty3.summary.json`；file_count=14496; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_hybrid_denoiser.summary.json` | 273 | `0d5f98bcf7072f0fa7ce3cf0323b6fe502f2a5941858ae03b3183c0c17f6a7c9` | 机器可读 JSON 证据 `results_hybrid_denoiser.summary.json`；file_count=6; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_hybrid_denoiser_raw_imu.summary.json` | 313 | `ef876dc94e26df82cf3fd0eadaa715092220143e97c42b5ac123720698d83f4d` | 机器可读 JSON 证据 `results_hybrid_denoiser_raw_imu.summary.json`；file_count=8; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_hybrid_denoiser_raw_imu_baseline.summary.json` | 322 | `5649e32d09cbd3502540065e95a8f9ee03e55dfe44186fce5016f9be93112cab` | 机器可读 JSON 证据 `results_hybrid_denoiser_raw_imu_baseline.summary.json`；file_count=8; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_stage1.summary.json` | 282 | `490692f73a7ec03885bfb020efe3c2917643c365b2009286302f3a78209f92be` | 机器可读 JSON 证据 `results_stage1.summary.json`；file_count=17; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_stage2.summary.json` | 226 | `869828aa6fe291ab74b98ec13fc54e630c04318850d1b6914eb20db606e8e6bd` | 机器可读 JSON 证据 `results_stage2.summary.json`；file_count=0; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_v72_noleak.summary.json` | 285 | `ee4a88a2c718e76533f189d81e02340490e6c1917f81fca311e818578674134f` | 机器可读 JSON 证据 `results_v72_noleak.summary.json`；file_count=16; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_v7_4.summary.json` | 327 | `72e65eceddad71d9089aff63758a7d1f5b562cfe10e8ab878f47c640e5c57d6f` | 机器可读 JSON 证据 `results_v7_4.summary.json`；file_count=55; error_count=0 |
| `M0_history_MA_denoising_detector_HR_feature/snapshots/verification/outputs/results_v8_audit.summary.json` | 284 | `f032ac9f5d97d53d2c5c2c6bf5ff20d07fee8b2e30e377176f612dafa4d6a994` | 机器可读 JSON 证据 `results_v8_audit.summary.json`；file_count=30; error_count=0 |
| `M1_end_to_end_architecture_contract/00_CURRENT_STATUS_V2.md` | 4302 | `781734d3b24cbfd814fba017085d2fe341286b8e25629b2690be8535ddb91e2b` | 文档《M1 当前权威合同 V2 / Current Authoritative Contract V2》；当前权威版本：`m1.architecture.v2`。；`README.md` 与 V1 schema/registry/example 保留为初始合同历史，不删除、不覆盖。 |
| `M1_end_to_end_architecture_contract/00_CURRENT_STATUS_V3.md` | 4921 | `5e057aaa7aa7c9daeae9b0a750ac3f2fb530158aca3cedf0190c65f915cf063d` | 文档《M1 当前权威合同 V3 / Current Authoritative Contract V3》；当前权威版本：`m1.architecture.v3`。；V3 只替换 V1/V2 中冲突的 quality routing / denoiser routing 语义；V2 的 `m1.signal_input.v2`、有界流式、窗口坐标、bundle 完整性、平台 profile、provider/CPU fallback、分类器输出与 no-result 规则继续有效。 |
| `M1_end_to_end_architecture_contract/00_CURRENT_STATUS_V3_1.md` | 1615 | `db4856cb44db5cef133c32d7e670fe06687423ad44afc4d159943a7486636c6d` | 文档《M1 V3 当前验证入口修正 / Current V3 Validation Entry》；架构语义仍为 `m1.architecture.v3`；本文件不改变路由算法、schema 或三档配置。；本文件仅修正 V3 首版 validator 对迁移元数据的误报。当前机器验证入口为： |
| `M1_end_to_end_architecture_contract/01_END_TO_END_ARCHITECTURE_AND_API.md` | 7282 | `a1043a67fa0fd6b688948f2374852bb4c4df30f72787ce0ffad9b34b8528a725` | 文档《M1 端到端模块架构与统一 Python API》；`SQI` 还有一个不改变波形的共同诊断出口。部署动作互斥规则是：；`sqi_gate`：SQI 决定接受、降权或拒绝原始/基础预处理窗口；signal frontend 必须是 `identity`。 |
| `M1_end_to_end_architecture_contract/02_MOBILE_PLATFORM_PROFILES.md` | 4887 | `492f5cbdfb50ff09b74e4dd65c9ccc4f5387391b2e5ab8f731d183d615803150` | 文档《M1 血压仪大小中心屏显处理设备：平台分档与工程预算》；允许软件依赖：NumPy、SciPy、ONNX Runtime、scikit-learn。中心设备是主要计算节点；穿戴端初期只负责采集、时间戳、缓存和传输，不承担 Frailty3 模型推理。；64-bit Linux 作为 M1/M9 的参考操作系统；其他系统必须通过相同 bundle parity。 |
| `M1_end_to_end_architecture_contract/03_TRAINING_VS_MOBILE_INFERENCE_BOUNDARY.md` | 4218 | `5b04465ca5496b9d5ddf363134eea0de180756e552cb1e2506a3f7cb908a87d8` | 文档《M1 训练/评估 Pipeline 与移动推理 Runtime 边界》；所有路径必须相对 bundle 根；manifest 保存每个 artifact 的 SHA-256、字节数、producer code version、训练数据 manifest ID 和 protocol ID。移动端启动时先校验 schema、hash、依赖与 provider，再加载模型。；模型不能导出或 parity 不通过时标记 `training_only`，不得以 Python 中可运行作为移动可部署证据。 |
| `M1_end_to_end_architecture_contract/04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md` | 4722 | `27f52db07505ea4dcf9fc26189b4c088d7fdff678d2eb3651707a77f13a1696d` | 文档《M1 现有实现接口审计与移动部署风险》；本文件记录对 Motion detector、hybrid denoiser、peak/IBI gate、Frailty3 classifier 与 ONNX runtime 的只读接口审计。已有算法部件可复用，但当前没有任何一条路线形成“原始 8 通道输入 → 统一预处理 → Motion/SQI/HR/PPI → Frailty3 → 屏显输出”的完整移动 bundle。；M1 只冻结新合同；旧根目录文件保持只读。具体修复、adapter、单元测试与 smoke test 留到 M3/M4。 |
| `M1_end_to_end_architecture_contract/05_VALIDATION_LIMITATIONS_AND_SEMANTIC_GATES.md` | 2295 | `ab0e5bb8c8f48318579e0d98f7bc4801cd489356f4d69053e22762e6bc3e1c7c` | 文档《M1 V2 验证限制与补充语义门》；V2 主验证器已完成 JSON 解析、schema 结构检查、registry/config 交叉引用、动作互斥、provider fallback 和有界 buffer 检查。本机 WSL、系统 Python 与工作区自带 Python 均未安装 `jsonschema`，因此没有运行第三方 Draft 2020-12 引擎，也没有下载或安装依赖。；`M1_CONTRACT_VERIFICATION_V2.json` 已如实记录： |
| `M1_end_to_end_architecture_contract/06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md` | 7000 | `45959509dce135a506f5cae360118513932d475eb6ab1bc205a393d8636a5faf` | 文档《M1 V3 顺序 SQI、可选 Motion 与 Denoiser 路由合同》；本合同落实用户确认的新顺序：先做 SQI 与可选 Motion detector，再把窗口分流。high-quality 窗口绕过 denoiser；low-quality 或 motion 窗口只执行预先选定的 `drop` 或 `denoise_then_extract_features`。它废止 V1/V2 的 `sqi_gate/coarse_denoise` 动作所有者模型。；`quality_state`：`high / low / unrecoverable / unknown`，来自 SQI，不把 activity label 当作质量真值。 |
| `M1_end_to_end_architecture_contract/M1_CONTRACT_VERIFICATION.json` | 648 | `0cac91e1b3b9587e27e54c24bec98c6e2da4932e15099c3fb01e76f8b121fda5` | 机器可读 JSON 证据 `M1_CONTRACT_VERIFICATION.json`；status=pass; failures=list[0] |
| `M1_end_to_end_architecture_contract/M1_CONTRACT_VERIFICATION_V2.json` | 779 | `b42a04f32aab347fb1b3354ef4ed51a4adb5215c0c6f9a1a0370530ce1f73032` | 机器可读 JSON 证据 `M1_CONTRACT_VERIFICATION_V2.json`；status=pass; failures=list[0] |
| `M1_end_to_end_architecture_contract/M1_CONTRACT_VERIFICATION_V3_CURRENT.json` | 920 | `2bb9e0be3833b35bc3ce838a52d900d118fd1455a149295912aa6a25ac553d85` | 机器可读 JSON 证据 `M1_CONTRACT_VERIFICATION_V3_CURRENT.json`；status=pass; failures=list[0] |
| `M1_end_to_end_architecture_contract/M1_PACKAGE_TREE.md` | 13587 | `1d8472d4e8a205b7326b5c21cb4244072d36c3420317905e33128f46371ec169` | 文档《M1 包文件树与逐文件说明 / M1 Package Tree》；Permanent files including generated indexes: **52**.；所有写入均位于 `final_v0/M1_end_to_end_architecture_contract/`。 |
| `M1_end_to_end_architecture_contract/M1_PACKAGE_TREE_V2.md` | 3267 | `c78a8896b1e0689bb2fc1fe0af8d4c50004b142adcd6676fee6338d4475c34db` | 文档《M1 V2 权威文件树与完整性 / M1 V2 Integrity Tree》；V2 authoritative files including generated indexes: **15**.；All V2 writes remain under `final_v0/`. |
| `M1_end_to_end_architecture_contract/M1_PACKAGE_TREE_V3_CURRENT.md` | 3821 | `8968dd7180d0a9595ea0678f97726d1206ac1934f82357a11646e7548407db8b` | 文档《M1 V3 CURRENT 权威文件树与完整性》；CURRENT authority files including generated indexes: **18**.；首版 V3 validator/迁移 registry 保留为历史，CURRENT 使用 active registry。 |
| `M1_end_to_end_architecture_contract/M1_ROUTING_INVARIANTS_V3.json` | 4836 | `58769053d31dfd611a64e6317d25f310e277e65022e5fe49e5b5c1d2653b04d1` | 机器可读 JSON 证据 `M1_ROUTING_INVARIANTS_V3.json`；status=pass; failures=list[0] |
| `M1_end_to_end_architecture_contract/M1_SEMANTIC_INVARIANTS_V2.json` | 2857 | `a937775022748a0d49188ee6cb7d9f638389d305045799ccbce1ed5848112a5b` | 机器可读 JSON 证据 `M1_SEMANTIC_INVARIANTS_V2.json`；status=pass; failures=list[0] |
| `M1_end_to_end_architecture_contract/README.md` | 2324 | `bae17fcf9c610768a6d8eefbc8ddb7fe2caead1df72e18801a2fa15d207ee1c0` | 文档《M1 端到端架构、数据契约与移动处理中心约束》；里程碑：`M1`；状态：`contract_defined_implementation_not_started` |
| `M1_end_to_end_architecture_contract/examples/pipeline_accelerated_arm64.json` | 1530 | `7c1a5d35a9a82f5dde4de6ca116b01ec46757b51e09edd24cc8adda394121959` | 机器可读 JSON 证据 `pipeline_accelerated_arm64.json`；keys=schema_version,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions |
| `M1_end_to_end_architecture_contract/examples/pipeline_high_performance_x86.json` | 1535 | `92f51376b0f1c96df9193d97b84e2140532476d9fe3dcb917313c30c8104e1e0` | 机器可读 JSON 证据 `pipeline_high_performance_x86.json`；keys=schema_version,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions |
| `M1_end_to_end_architecture_contract/examples/pipeline_value_arm64.json` | 1473 | `0beaafb1ad36aa926257ebc5322b63ec7c04cdffd78d017839cf4d4408dd3c8a` | 机器可读 JSON 证据 `pipeline_value_arm64.json`；keys=schema_version,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions |
| `M1_end_to_end_architecture_contract/examples_v2/pipeline_accelerated_arm64_v2.json` | 1928 | `b37961cc016d083860a37df6145eec995461f336f02cd66ca1e7f257c2511e5a` | 机器可读 JSON 证据 `pipeline_accelerated_arm64_v2.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M1_end_to_end_architecture_contract/examples_v2/pipeline_high_performance_x86_v2.json` | 1876 | `0e32946c35b9a914af10424bf814b7a455707acb9b6c2ed20de1b2db284918aa` | 机器可读 JSON 证据 `pipeline_high_performance_x86_v2.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M1_end_to_end_architecture_contract/examples_v2/pipeline_value_arm64_v2.json` | 1789 | `18505c10fcc5851af3ab022ea4a40e8cb07bdbd6494c8b85ede317850b83de9b` | 机器可读 JSON 证据 `pipeline_value_arm64_v2.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M1_end_to_end_architecture_contract/examples_v3/pipeline_accelerated_arm64_v3.json` | 3220 | `61a9bbf212817e1a93e69e5c3b5f4f8cd530e66b9ba1c1e4066da7a21786abf2` | 机器可读 JSON 证据 `pipeline_accelerated_arm64_v3.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M1_end_to_end_architecture_contract/examples_v3/pipeline_high_performance_x86_v3.json` | 3169 | `1c2936ac0d2cd0a9da85122cb15e17cf231b3a357627bb2ee969895db28da6ea` | 机器可读 JSON 证据 `pipeline_high_performance_x86_v3.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M1_end_to_end_architecture_contract/examples_v3/pipeline_value_arm64_v3.json` | 2967 | `93e876ad843957b15811c768696c37ee14b68c42ff3687e3b07ff152323da496` | 机器可读 JSON 证据 `pipeline_value_arm64_v3.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M1_end_to_end_architecture_contract/registries/classifier_registry.json` | 2053 | `a3f0b0a58813af7a0ccaef9a3397d0833f1f855df0c60040d245c8d571f6761d` | 机器可读 JSON 证据 `classifier_registry.json`；keys=registry_version,output_adapter,label_order,classifiers |
| `M1_end_to_end_architecture_contract/registries/feature_extractor_registry.json` | 1065 | `db274e89b735aee89cc964004ff804bad52f56a4a7db6e9e5e8e3cb36c2291fe` | 机器可读 JSON 证据 `feature_extractor_registry.json`；keys=registry_version,extractors |
| `M1_end_to_end_architecture_contract/registries/platform_profiles.json` | 2201 | `a7f1e6fd921487061ecff2de794f68d9b4e19c52c18b5edf79f323abcddf2a62` | 机器可读 JSON 证据 `platform_profiles.json`；status=candidate_profiles_not_benchmarked |
| `M1_end_to_end_architecture_contract/registries/quality_policy_registry.json` | 705 | `706f858406e5461f6b8637e8b0daf9bc41e3814cb7c8a1bfb35eac205631fa6e` | 机器可读 JSON 证据 `quality_policy_registry.json`；keys=registry_version,sqi_monitor_required,action_mode_cardinality,policies |
| `M1_end_to_end_architecture_contract/registries_v2/classifier_registry_v2.json` | 2945 | `a5b0ccb4103eee8de2983c9e19a5cb3103c82f2dc283d70555b9337455c29c18` | 机器可读 JSON 证据 `classifier_registry_v2.json`；keys=registry_version,output_adapter,label_order,entry_contract,classifiers |
| `M1_end_to_end_architecture_contract/registries_v2/feature_extractor_registry_v2.json` | 1446 | `04d9c6759f04f9423493e709039318d42dc5b51b6194146e9c5fe26e719582dd` | 机器可读 JSON 证据 `feature_extractor_registry_v2.json`；keys=registry_version,entry_contract,extractors |
| `M1_end_to_end_architecture_contract/registries_v2/platform_profiles_v2.json` | 2419 | `108c800e6ccefa4dbe2f65fc78062828cd4f097b9c5662299f08c6395a27c6a5` | 机器可读 JSON 证据 `platform_profiles_v2.json`；status=candidate_profiles_not_benchmarked |
| `M1_end_to_end_architecture_contract/registries_v2/quality_policy_registry_v2.json` | 1049 | `968cbb3e5bcc5776bf2a5b3439acda6360ac23490f92f7e88f246fc5139d4a21` | 机器可读 JSON 证据 `quality_policy_registry_v2.json`；keys=registry_version,sqi_monitor_required,action_mode_cardinality,diagnostic_candidates_may_run_in_parallel,allowed_final_action_codes,policies |
| `M1_end_to_end_architecture_contract/registries_v3/quality_routing_registry_v3.json` | 3488 | `32d257cbc89c0eb690d1cd53eea13a3af4e99243d504fd61ca9ca97f9919ee34` | 机器可读 JSON 证据 `quality_routing_registry_v3.json`；keys=registry_version,architecture_version,sqi_required,motion_detector_optional,first_stage_parallelism,state_axes,manual_policy_contract,allowed_terminal_action_codes,policies,denoiser_frontends |
| `M1_end_to_end_architecture_contract/registries_v3/quality_routing_registry_v3_active.json` | 3163 | `0d5643de7a19f9c612fdbd29afa31273a37fc91fbc716727092a82a67fa6411e` | 机器可读 JSON 证据 `quality_routing_registry_v3_active.json`；keys=registry_version,architecture_version,sqi_required,motion_detector_optional,first_stage_parallelism,state_axes,manual_policy_contract,allowed_terminal_action_codes,policies,denoiser_frontends |
| `M1_end_to_end_architecture_contract/schemas/inference_output.schema.json` | 1960 | `217f6b8f8ba36c6d083704d40ca49025753994793dcd0e3676cdcb080026e3f8` | 机器可读 JSON 证据 `inference_output.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties,allOf |
| `M1_end_to_end_architecture_contract/schemas/pipeline_config.schema.json` | 3608 | `b5afc25c5e2888d6a5f80c802062eba12095da4e47b55884bb7834eb9317e145` | 机器可读 JSON 证据 `pipeline_config.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties,$defs |
| `M1_end_to_end_architecture_contract/schemas/signal_input.schema.json` | 2691 | `447ddc7684a0825efe37f04abbc3887417be99b7f26e0ff874ac28fc70c4c149` | 机器可读 JSON 证据 `signal_input.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties |
| `M1_end_to_end_architecture_contract/schemas_v2/inference_output_v2.schema.json` | 5086 | `c41dab571d544d340cac3290215b3312cf53123218b6095770acf4e3d19cce67` | 机器可读 JSON 证据 `inference_output_v2.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties,allOf |
| `M1_end_to_end_architecture_contract/schemas_v2/pipeline_config_v2.schema.json` | 6028 | `c15e854bca5a0d40cd98f183cba93237036c9be8408240083c73ed17a14a644c` | 机器可读 JSON 证据 `pipeline_config_v2.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties,$defs |
| `M1_end_to_end_architecture_contract/schemas_v2/signal_input_v2.schema.json` | 2813 | `6c4cbff49ee39ff04a385b5bc2a106926217f1bf591361fd39f8bce84f7dbfba` | 机器可读 JSON 证据 `signal_input_v2.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties |
| `M1_end_to_end_architecture_contract/schemas_v3/inference_output_v3.schema.json` | 9023 | `e9f6870c36e3e82c14dca17cc27eab90e6001558719d8b2d67e7c847e0a7486f` | 机器可读 JSON 证据 `inference_output_v3.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties,allOf,$defs |
| `M1_end_to_end_architecture_contract/schemas_v3/pipeline_config_v3.schema.json` | 11504 | `1a06d416e312507997d13be440e0aca3712957f9d6d7ffa05f0ae602a51ea5ce` | 机器可读 JSON 证据 `pipeline_config_v3.schema.json`；keys=$schema,$id,title,type,additionalProperties,required,properties,allOf,$defs |
| `M1_end_to_end_architecture_contract/tools/bootstrap_m1_contract_report.py` | 1981 | `06086c237c3ec698b5cc21cf6d65760c2e034058b0ff8e2da830618df3b7145c` | 带中英文说明的 final_v0 工具；首次生成 M1 报告与包树；bootstrap the first M1 report and package tree.；主要入口：main |
| `M1_end_to_end_architecture_contract/tools/validate_m1_contracts.py` | 18601 | `21230f930c08c281585eb985d42a6d005522b0dddceed4f4a7b6b7b2975a57eb` | 带中英文说明的 final_v0 工具；验证 M1 架构合同并生成可追溯索引；validate M1 contracts and build traceable indexes.；主要入口：checked_relative, load_json, sha256_bytes, atomic_write, atomic_write_json, atomic_write_text, add_failure, index_by_id |
| `M1_end_to_end_architecture_contract/tools/validate_m1_contracts_v2.py` | 19439 | `451fe054cc21e8e6f8f767feea37bf409b3f60a7819d42c61a05d48c90da031b` | 带中英文说明的 final_v0 工具；验证 M1 V2 架构合同并生成可追溯报告；validate M1 V2 contracts.；主要入口：checked_relative, load_json, sha256_bytes, atomic_write, atomic_write_json, atomic_write_text, add_failure, index_by_id |
| `M1_end_to_end_architecture_contract/tools/validate_m1_contracts_v3.py` | 27634 | `659e91ef58b2b4fa6fa5513bf4b5efe2ef9b6a342c8328497386a649d2a3f1e8` | 带中英文说明的 final_v0 工具；验证 M1 V3 顺序质量路由合同；validate M1 V3 sequential routing contracts.；主要入口：checked_relative, load_json, sha256_bytes, atomic_write, atomic_write_json, atomic_write_text, add_failure, index_by_id |
| `M1_end_to_end_architecture_contract/tools/validate_m1_contracts_v3_current.py` | 6129 | `6b3724dfcc367c3b6802e0ed7b3b7fd0558989e89fddc4b991d8d0f7d3882e58` | 带中英文说明的 final_v0 工具；运行 M1 V3 当前合同验证入口；run the current M1 V3 contract validator.；主要入口：configure_base_validator, validate_contracts, render_tree, parse_args, main |
| `M1_end_to_end_architecture_contract/tools/validate_m1_v2_semantic_invariants.py` | 12064 | `88a03099fd1cf26eb2555cf018a41a1b514128d72d1d162041f8daa3f3e48351` | 带中英文说明的 final_v0 工具；验证 M1 V2 的运行时语义不变量；validate M1 V2 runtime semantic invariants.；主要入口：checked_relative, atomic_write_json, validate_bundle_relative_path, validate_output, validate_config, base_output, base_locked_config, run_tests |
| `M1_end_to_end_architecture_contract/tools/validate_m1_v3_routing_invariants.py` | 21392 | `7241e213195a2ec1ae63b65d829871202ad2eafb801ae8693d9812914fb14f85` | 带中英文说明的 final_v0 工具；验证 M1 V3 顺序路由语义；validate M1 V3 sequential routing semantics.；主要入口：checked_relative, atomic_write_json, action_for_reason, route_window, validate_route_shape, validate_run_policy, validate_routing_summary, base_summary |
| `M2_data_manifest_and_evaluation_protocol/00_CURRENT_STATUS.md` | 1268 | `355ea3bd0ec88e85be25b76d02ea7ce447b871344c49b839883534bf3fdfaca9` | 文档《M2 当前状态 / Current Status》；里程碑：`M2`；状态：`contract_and_registries_defined_no_model_rerun_yet` |
| `M2_data_manifest_and_evaluation_protocol/01_DATASET_MANIFEST_AND_PROVENANCE.md` | 2430 | `70b5bd1de15dce98c492d750e9314066086cf0f56aa40cf576b910b97d414ac1` | 文档《Frailty3 数据 Manifest 与溯源》；`PPG_Testing_05_01_2026/StudyData/`：21 名 older subjects，189 个 CSV。；`PPG_Testing_05_01_2026/TestDataYoungers/`：8 名 young subjects，72 个 CSV。 |
| `M2_data_manifest_and_evaluation_protocol/02_STAGE_ROLE_MAPPING.md` | 1390 | `bf74982797b1e65cd7e32fc60fbe5dd8c36aec9d8377d9058586fea1d3fdcec5` | 文档《B/R/S/W 阶段映射与部分时序合同》；当前只冻结下列部分顺序：；该关系允许后续定义运动后恢复特征，但不代表已知全部采集 protocol。 |
| `M2_data_manifest_and_evaluation_protocol/03_DUAL_FOLD_REGISTRY_AND_MAIN_PROTOCOL.md` | 2343 | `346827511b35c741160809015e6035078a37281215868bfe3ab085dadf8e23df` | 文档《Frailty3 双 Fold 注册表与唯一未来主协议》；历史运行调用 scikit-learn 1.4.2 的 `StratifiedGroupKFold(..., shuffle=True)`。该版本在 shuffle 后移动了 group class-count rows，却没有同步重映射 `groups_inv`，因此受试者仍不跨 fold，但类别计数被错误关联到别的 group。结果是 25 个历史 OOF folds 中 6 个缺少至少一个类别。；M2 不静默改写历史 membership： |
| `M2_data_manifest_and_evaluation_protocol/04_EXTERNAL_SYNCHRONIZED_DATA_MANIFEST.md` | 2218 | `014c3621acfeb1ff02d135306314e73509afa702d2e0f9e0d13b332d9fe4fbeb` | 文档《外部同步 ECG/PPG/IMU 数据 Manifest》；22 subjects × sit/walk/run = 66 records；CSV container grid 500 Hz。；左食指 distal/proximal 各含多色 PPG；ECG、accelerometer、gyroscope 同步；`peaks`/`.atr` 为自动检测后人工复核 R peaks。 |
| `M2_data_manifest_and_evaluation_protocol/05_RESULT_PROVENANCE_AND_NAMING_CONTRACT.md` | 1233 | `8c93fc7845aeb1b94555c87897f5b90945d349b6a50859ba250d4dad6a8e65d4` | 文档《结果溯源与命名合同》；1. `dataset_version_id`、manifest SHA-256、纳入/排除 filter。；2. `fold_registry_id`、registry payload SHA-256、repeat、split seed、fold、train/OOF subject IDs。 |
| `M2_data_manifest_and_evaluation_protocol/M2_BUILD_REPORT.json` | 3289 | `9c7903b4ce0594a1cb53be2835aeba36453f04835b95d45bd6935d63da6d8a8e` | 机器可读 JSON 证据 `M2_BUILD_REPORT.json`；status=pass |
| `M2_data_manifest_and_evaluation_protocol/M2_CONTRACT_VERIFICATION.json` | 18883 | `a8cf5f0ac60635a6c81577627a7e1f7331b9b973053d68ddd860c0ac4bcd7cec` | 机器可读 JSON 证据 `M2_CONTRACT_VERIFICATION.json`；status=pass; failures=list[0] |
| `M2_data_manifest_and_evaluation_protocol/M2_PACKAGE_TREE.md` | 3784 | `79c8af5edf1bee4def054311846b90424ca57b11a35c9fbabcb8f8410834445a` | 文档《M2 包文件树与完整性 / Package Tree and Integrity》；永久文件 / Permanent files：26。 |
| `M2_data_manifest_and_evaluation_protocol/README.md` | 2426 | `880349da90e811a138faa2213e12084e2a08556662aa26ef17f8ac187e82b0d2` | 文档《M2 数据 Manifest、阶段映射与评估协议》；M2 冻结数据身份、文件/受试者清单、经用户确认的阶段语义、外部同步 ECG/PPG/IMU 证据、协议命名和 Frailty3 双 fold 注册表。这里不训练模型，也不产生性能结论。；唯一未来主协议为：原始 400 Hz Frailty3、subject-level 5-fold、5 repeats、seeds `42, 10042, 20042, 30042, 40042`、fixed epoch、no early stopping、仅训练完成后计算 OOF validation。历史 scikit-learn 1.4.2 shuffle 映射错误的 SGKF membership 只保留用于复现，所有候选必须在修正且类别均衡的注册表上统一重跑。 |
| `M2_data_manifest_and_evaluation_protocol/examples/result_provenance_fixed_epoch_oof_template.json` | 1504 | `5c3b2c59ae07c1247f901577fac343f55504108522bec0821ae320d93e5b5f72` | 机器可读 JSON 证据 `result_provenance_fixed_epoch_oof_template.json`；keys=config_hash,coverage,dataset_manifest_sha256,dataset_version_id,early_stopping,evaluation_role,feature_schema_version,fixed_epoch,fold_index,fold_registry_id |
| `M2_data_manifest_and_evaluation_protocol/manifests/external_dataset_manifest.csv` | 2362 | `b1188d76d9897d5b707c8efd5c1b6665da38bea62c05904d47e29dd1d91b40e1` | CSV 文件；用途由路径和相邻审计记录定义 |
| `M2_data_manifest_and_evaluation_protocol/manifests/external_record_manifest.csv` | 56917 | `43ab3273346469e9f689ce32da9c5ad280d0a53a8bc8864adf5716f40f9f024e` | CSV 文件；用途由路径和相邻审计记录定义 |
| `M2_data_manifest_and_evaluation_protocol/manifests/frailty3_file_manifest.csv` | 218952 | `bd429ae9c56974ba9ffcb924dfbad0ed930f7d2d47418365754a1929ada06e90` | CSV 文件；用途由路径和相邻审计记录定义 |
| `M2_data_manifest_and_evaluation_protocol/manifests/frailty3_source_anomalies.json` | 1445 | `6b000b66ac63e66d6643a564a95b3c655c88813017d20ff2bb8c0b61f51b5626` | 机器可读 JSON 证据 `frailty3_source_anomalies.json`；keys=duration_seconds_by_role,label_only_ids_without_raw_files,label_source_conflicts,metadata_unknown,reference_limit,schema_version,shorter_than_30s_counts |
| `M2_data_manifest_and_evaluation_protocol/manifests/frailty3_subject_manifest.csv` | 10097 | `e8f712447aa1d7d0f8e564ae8b80ba0901cfaad85287a9961362928ca3d1cd03` | CSV 文件；用途由路径和相邻审计记录定义 |
| `M2_data_manifest_and_evaluation_protocol/registries/external_dataset_registry.json` | 4594 | `14ecb306d234f9d191950f150b2ce4682253aa6baf381bdcefdd70373ed27696` | 机器可读 JSON 证据 `external_dataset_registry.json`；keys=schema_version,datasets |
| `M2_data_manifest_and_evaluation_protocol/registries/protocol_registry.json` | 2423 | `beae2a6922ae0ca840cec1a5c501cde6b6fc029afed16fc798aa2ef8e05fa394` | 机器可读 JSON 证据 `protocol_registry.json`；keys=schema_version,active_protocol_id,protocols |
| `M2_data_manifest_and_evaluation_protocol/registries/stage_role_registry.json` | 1348 | `4bfe93b78985db8c279f80b966a46f2f2ac1df44971fd6c1a25a6d4cddbc81bb` | 机器可读 JSON 证据 `stage_role_registry.json`；status=confirmed_partial_order_only |
| `M2_data_manifest_and_evaluation_protocol/schemas/dataset_manifest.schema.json` | 1211 | `bf1499828c304ccb5e3ea2cf6e22a3d8176d416ea7b35325ca25ee3d3d1ca2d0` | 机器可读 JSON 证据 `dataset_manifest.schema.json`；keys=$schema,$id,title,type,required,properties,additionalProperties |
| `M2_data_manifest_and_evaluation_protocol/schemas/fold_registry.schema.json` | 777 | `863ca6e8c4bf9d754fb23a4f5554a1bf4b4ba99837e598dc67f44e000db8034f` | 机器可读 JSON 证据 `fold_registry.schema.json`；keys=$schema,$id,title,type,required,properties,additionalProperties |
| `M2_data_manifest_and_evaluation_protocol/schemas/result_provenance.schema.json` | 1249 | `5aa489a50e2ff974b78ace388e44b134e9445f8e3dbf4eb275d1d5eaf40f5119` | 机器可读 JSON 证据 `result_provenance.schema.json`；keys=$schema,$id,title,type,required,properties,additionalProperties |
| `M2_data_manifest_and_evaluation_protocol/splits/frailty3_future_corrected_sgkf5_v2.json` | 295604 | `c80e780d60162ff1132101ee368ee3a55d95f3c7a6d51a5c9a8feec8605d388c` | 机器可读 JSON 证据 `frailty3_future_corrected_sgkf5_v2.json`；status=active_future_benchmark_only |
| `M2_data_manifest_and_evaluation_protocol/splits/frailty3_historical_sgkf5_sklearn142_bug_v1.json` | 295704 | `dc31683f7e35df28ec69282c98e8d56be3616809a8514831f5b0f3daaec3bd90` | 机器可读 JSON 证据 `frailty3_historical_sgkf5_sklearn142_bug_v1.json`；status=historical_reproduction_only |
| `M2_data_manifest_and_evaluation_protocol/tools/build_m2_manifests.py` | 41134 | `59ca997b4b6e22fd80ed2c364e1285d992d058c114d0c66e2c00ba60c1a38f74` | 带中英文说明的 final_v0 工具；构建 M2 数据 manifests、外部记录清单与双 fold 注册表。；主要入口：sha256_bytes, stable_json_bytes, checked_target, atomic_write_bytes, write_json, write_csv, relative_repo, permission_snapshot |
| `M2_data_manifest_and_evaluation_protocol/tools/validate_m2_contracts.py` | 19608 | `19a9c3ccc23f6fa381a929077372cca418244a6d9083c59e10553336bb48dce2` | 带中英文说明的 final_v0 工具；验证 M2 manifests、双 fold registry、协议和结果溯源合同。；主要入口：sha256_bytes, stable_json_bytes, checked_target, atomic_write, load_json, load_csv, check, validate_manifests |
| `M3_unified_preprocessing_and_signal_algorithms/M3_BUILD_REPORT.json` | 1286 | `4aa5b66bf1f864275eab9ab11ad32514b9c943a39cd300c84e922b1acadbde3f` | 机器可读 JSON 证据 `M3_BUILD_REPORT.json`；status=pass |
| `M3_unified_preprocessing_and_signal_algorithms/M3_REFERENCE_TEST_RESULTS.json` | 4152 | `7003e6758a7cbc6d3c448f4605b27b7ffa737f68f05ced0eaf6d5d33c6871c11` | 机器可读 JSON 证据 `M3_REFERENCE_TEST_RESULTS.json`；status=pass; error_count=0; failures=list[0] |
| `M3_unified_preprocessing_and_signal_algorithms/README.md` | 1581 | `7c22fba4b4d53aaacb7ae9c15371c9b90cc707dcb83f909ba5d902aafdb39fd0` | 文档《M3 统一预处理与信号算法 / Unified Preprocessing and Signal Algorithms》；本包是 M3 的唯一未来活动实现入口。根目录历史脚本保持只读，只用于历史复现和；parity 审计；M4 之后的新模块必须调用本包公共 API 或读取本包版本化注册表。 |
| `M3_unified_preprocessing_and_signal_algorithms/docs/00_CURRENT_STATUS.md` | 6895 | `c9bb78aa6d7e4f6170a45afc9e6fe36ac94c1d05ca0b8d2258b0b25215f03f39` | 文档《M3 当前状态 / Current Status》；里程碑 / Milestone: **M3 — unified preprocessing and signal algorithms**；状态 / Status: **核心合同与参考实现已建立；工程参考测试通过；真实监督 benchmark 尚未完成。 / Core contracts and reference implementation are established; engineering reference tests pass; real supervised benchmarks remain pending.** |
| `M3_unified_preprocessing_and_signal_algorithms/docs/01_HISTORICAL_PREPROCESSING_CROSSWALK.md` | 5892 | `39311a1d4d69b54ea282ce445d84db5385bee626d952634cb5655aba95ca8e42` | 文档《历史预处理 Crosswalk / Historical Preprocessing Crosswalk》；本 crosswalk 将根目录历史脚本映射到 M3 公共实现。所有列出的根目录脚本状态均为 `historical_reproduction_only`：允许读取、复现和解释旧结果，但不得成为 M4 以后第二个活动预处理入口。机器证据位于 `evidence/historical_preprocessing_crosswalk_v1.json`，其中保存逐文件 SHA-256、字节数和存在性。；This crosswalk maps root-level historical scripts to the M3 public implementation. Every listed root script is reproduction-only. The evidence JSON, not this pros… |
| `M3_unified_preprocessing_and_signal_algorithms/docs/02_PPG_AND_CLEANING_CONTRACT.md` | 5884 | `191feb0c099f5c145c8d7a60916956f80642c0e6612c7e345e72e08f91e4e822` | 文档《PPG 与清洗合同 / PPG and Cleaning Contract》；PPG 处理在任何滤波、归一化、SQI、denoising 或 feature extraction 之前执行输入合同。；canonical 双波长顺序：RED, IR； |
| `M3_unified_preprocessing_and_signal_algorithms/docs/03_EKF_PRIMARY_AND_LPF_COMPARATOR_CONTRACT.md` | 6389 | `6af3ec66fb696b72183b73bf4019a4a99d16c340c5ebdb407bb7b25d8756fb1a` | 文档《EKF 主路线与 LPF 对照合同 / EKF Primary and LPF Comparator Contract》；EKF 与 LPF 只能改变 gravity estimator。两路线必须共享：；同一 400 Hz 六轴输入 AX, AY, AZ, GX, GY, GZ； |
| `M3_unified_preprocessing_and_signal_algorithms/docs/04_FOLD_SCALING_AND_M1_M2_BINDING.md` | 5123 | `73ed5005bd91aa0cd294be9b21157957f9ce91ce3543deb5381b4dc775f896ec` | 文档《Fold Scaling 及 M1/M2 绑定 / Fold Scaling and M1/M2 Binding》；M3 的所有可学习 preprocessing statistics 必须绑定：；fold registry: `frailty3_future_corrected_sgkf5_v2`; |
| `M3_unified_preprocessing_and_signal_algorithms/docs/05_PEAK_PPI_HR_PRV_CONTRACT.md` | 5630 | `8ac23958faf79329ed28ce1658506f405d88565c1125e0da68dc9474586ce9b6` | 文档《Peak、PPI、HR 与 PRV 合同 / Peak, PPI, HR, and PRV Contract》；high-quality bypass 路线和 denoised 路线必须调用同一个 `m3_physiology_corrected_v1` 后端。算法输入 profile 固定为 `frailty3_peak_ppg_400_offline_v1`、400 Hz、0.4–8 Hz。若 denoiser 输出不是该 profile 的 canonical waveform/feature adapter，必须显式转换并保存 provenance，不能在 denoiser 内另写 peak helper。；The shared backend prevents a route from winning merely because it used a more favorable peak or PPI def… |
| `M3_unified_preprocessing_and_signal_algorithms/docs/06_TEST_RESULTS_AND_LIMITATIONS.md` | 5788 | `efd4b06683c814e304fea4db3986966a2f08663b61f4d509b53c50f1ed2527e9` | 文档《测试结果与局限 / Test Results and Limitations》；`M3_REFERENCE_TEST_RESULTS.json` 当前记录：；report ID `m3_reference_tests_v1`; |
| `M3_unified_preprocessing_and_signal_algorithms/docs/07_PROFILE_AND_MOBILE_PLATFORM_BINDING.md` | 5063 | `a140b57b3115205bb1ed292f7163e813919bf265b510ab0c206f15209956c135` | 文档《Profile 与移动平台绑定 / Profile and Mobile-Platform Binding》；用户确认的形态为：RED/IR PPG + 3-axis ACC + 3-axis GYRO 穿戴采集端，连接血压仪大小、带中心屏显的处理中心。穿戴端初期负责采集、timestamp、packet sequence、缓存和传输；中心端执行 M3 preprocessing、M1 routing、feature/model inference 与屏显。；The wearable is a sensor/transport endpoint; the central unit is the primary compute node. |
| `M3_unified_preprocessing_and_signal_algorithms/docs/08_EKF_VS_LPF_RESULTS.md` | 5707 | `6a1114af86853662443f37d557a8ec7fd5fa914ef422249925377a0f4c31de0d` | 文档《EKF 与 LPF 结果对比 / EKF versus LPF Results》；在有真值的固定合成 IMU fixture 上，无预校准 quaternion ESKF 的 gravity/dynamic-acceleration error 明显低于 0.3 Hz LPF，但 coverage 因显式在线初始化略低。Frailty3 没有 gravity 或姿态真值；其首 6 s 结果只能说明两路线产生不同的信号分解与 coverage，不能用于计算姿态/重力准确率，也不能单独决定临床或分类优越性。；The current evidence supports keeping ESKF as the engineering primary and LPF as a controlled comparator. It does not establish clinical superiority. |
| `M3_unified_preprocessing_and_signal_algorithms/evidence/ekf_lpf_frailty3_role_proxy.json` | 257991 | `afc2de50e55168f1c2e2df3f0ab38076f861490c162edcb89b2ceda8d5b6b74e` | 机器可读 JSON 证据 `ekf_lpf_frailty3_role_proxy.json`；keys=dataset_version_id,evidence_id,limitations,m2_manifest_sha256,paired_upstream,record_count,records,segment_definition,subject_count,summary_by_role_family_and_route |
| `M3_unified_preprocessing_and_signal_algorithms/evidence/ekf_lpf_synthetic_comparison.json` | 1521 | `404638f7530da9b9ea9d478293a94dad2f0d5315ddbe7f1a790d455c9fc30e68` | 机器可读 JSON 证据 `ekf_lpf_synthetic_comparison.json`；keys=engineering_gate_status,evidence_id,fixture,fixture_columns,fixture_sha256,limitations,route_metrics |
| `M3_unified_preprocessing_and_signal_algorithms/evidence/filter_response_comparison.json` | 2700 | `752ea092d896a6d6329d825a5b17389dd147d4b73c54b25f40f1baf0ea0b99f7` | 机器可读 JSON 证据 `filter_response_comparison.json`；keys=evidence_id,frequencies_hz,interpretation,notch,profiles,sampling_rate_hz |
| `M3_unified_preprocessing_and_signal_algorithms/evidence/frailty3_signal_integrity_summary.json` | 756 | `2db148d796b8a77fc2509303dc5d80871485dadf284f64ca72c93813dc5d552a` | 机器可读 JSON 证据 `frailty3_signal_integrity_summary.json`；file_count=261 |
| `M3_unified_preprocessing_and_signal_algorithms/evidence/historical_preprocessing_crosswalk_v1.json` | 21142 | `7403c15960894d5215c9c40a5d2bae929e23c2106bf26b4a0c5d9a805950a96e` | 机器可读 JSON 证据 `historical_preprocessing_crosswalk_v1.json`；status=root_read_only_historical_reproduction_only |
| `M3_unified_preprocessing_and_signal_algorithms/evidence/legacy_peak_parity.json` | 1415 | `42dcc66cae91342d02e16b7098ac50f85f5fd1a0938990f0874e5070913dcb3d` | 机器可读 JSON 证据 `legacy_peak_parity.json`；status=pass_with_expected_cross_implementation_difference |
| `M3_unified_preprocessing_and_signal_algorithms/examples/m1_pipeline_config_m3_mobile.json` | 3205 | `0d7577c24ffa272fbc0f40fafc339640bcdab7c7e3aaa3295a060f8139a4d23e` | 机器可读 JSON 证据 `m1_pipeline_config_m3_mobile.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M3_unified_preprocessing_and_signal_algorithms/examples/m1_pipeline_config_m3_offline.json` | 3208 | `d44615696a27125e5a4b7ba59beba24d143736feadca1de42286ebb20cba4f2f` | 机器可读 JSON 证据 `m1_pipeline_config_m3_offline.json`；keys=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime,modules,versions,thresholds |
| `M3_unified_preprocessing_and_signal_algorithms/examples/m2_result_provenance_m3_bound.json` | 3722 | `8e99a6d39659afac9f8eadc66c92d8f8fbaa07dff4273624012a1513fcbbc3ec` | 机器可读 JSON 证据 `m2_result_provenance_m3_bound.json`；keys=config_hash,config_path,coverage,dataset_manifest_sha256,dataset_version_id,early_stopping,evaluation_role,feature_schema_version,fixed_epoch,fold_index |
| `M3_unified_preprocessing_and_signal_algorithms/fixtures/imu_reference_v1.npy` | 460928 | `bcb0e796b97d10e96a32b61c4fe17eb31de8729724411735158c2abd82e00f24` | NPY 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/fixtures/ppg_expected_peaks_v1.npy` | 424 | `30778dfa9961e1518f28a476b28fc643bb90f55d7623964288de81b2cc874674` | NPY 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/fixtures/ppg_reference_v1.npy` | 96128 | `0bd9ba8e3c6385e4d972c20afd5f7dd945e38bf55a849e29941de4ebd1ad0d2c` | NPY 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/fixtures/reference_fixture_manifest.json` | 2102 | `fd99d6c5c92821c10314e15e30df5d9ea82611fb8088863c18b5d7ae07f59925` | 机器可读 JSON 证据 `reference_fixture_manifest.json`；status=deterministic_synthetic_truth |
| `M3_unified_preprocessing_and_signal_algorithms/registries/feature_schemas_v1.json` | 6365 | `68484c7fd33d65b43d91ee68ca973e1264eea8c4add003f232f1631240bbc8c9` | 机器可读 JSON 证据 `feature_schemas_v1.json`；status=contract_defined_M4_adapters_and_retraining_pending |
| `M3_unified_preprocessing_and_signal_algorithms/registries/historical_preprocessing_crosswalk_v1.json` | 15245 | `d1f62105139e801003c105645a6a6f034138c0e0f1f39937c1941443f2719bc8` | 机器可读 JSON 证据 `historical_preprocessing_crosswalk_v1.json`；status=historical_authority_future_benchmark_forbidden |
| `M3_unified_preprocessing_and_signal_algorithms/registries/module_bindings_v1.json` | 17446 | `1dd152ef6a0056fa12ffbaea9cbfcaab123b321f0653761bec9e095f906afdfa` | 机器可读 JSON 证据 `module_bindings_v1.json`；status=contract_defined_core_bindings_present_profile_lock_adapter_pending |
| `M3_unified_preprocessing_and_signal_algorithms/registries/physiology_algorithms_v1.json` | 3619 | `a5ff455bfe9ad38a59bb04508da319a39ccb3236761cf9bbbdd6b4cb3f73062b` | 机器可读 JSON 证据 `physiology_algorithms_v1.json`；status=future_active |
| `M3_unified_preprocessing_and_signal_algorithms/registries/preprocessing_profiles_v1.json` | 12765 | `178b7df672431954a66359cd499c05bf9bc95d30b0c494725f79447593aa399e` | 机器可读 JSON 证据 `preprocessing_profiles_v1.json`；status=future_active |
| `M3_unified_preprocessing_and_signal_algorithms/registries/reason_codes_v1.json` | 10767 | `36221e3ffc5bd4cde560ad93088ac8e3e10ba823410d1253f26e609f87da90bf` | 机器可读 JSON 证据 `reason_codes_v1.json`；status=future_active |
| `M3_unified_preprocessing_and_signal_algorithms/registries/status_mapping_v1.json` | 6006 | `5af820950e7bd8617f0ea62419be30c7567a2c280952e92fe10ecb2893a8caee` | 机器可读 JSON 证据 `status_mapping_v1.json`；status=future_active |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/fold_fitted_artifact.schema.json` | 12166 | `59f7dd6c6e80ae59e73ff545acb49560acb2af1325b869b96efed7236ec1174a` | 机器可读 JSON 证据 `fold_fitted_artifact.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,allOf,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/module_binding.schema.json` | 10958 | `d59f6d3076fa13be1b003e10ee3bcab520cc55574a1d7f7a600ea5704fc15f40` | 机器可读 JSON 证据 `module_binding.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/physiology_result.schema.json` | 18900 | `e7dbb0b25c6883c58b6bc1e32fa08803839432ba73f0718973f8768cfba72622` | 机器可读 JSON 证据 `physiology_result.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,allOf,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/preprocessing_profile.schema.json` | 3105 | `30ae2ed5a0553d7947153850eec8c337a827d592b434c821cb9409d001460d22` | 机器可读 JSON 证据 `preprocessing_profile.schema.json`；keys=$schema,$id,title,description,type,required,properties,additionalProperties |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/preprocessing_result.schema.json` | 21045 | `56795d4af1eb4ce76e458fc483701f90cf71c9349dc20ee1fdab0b046030a132` | 机器可读 JSON 证据 `preprocessing_result.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/ptt_reference_evaluation.schema.json` | 5840 | `55f1421d6f427d1f512c4048ad9c8696fb37030a0ac71aec6b738fe50338b0cd` | 机器可读 JSON 证据 `ptt_reference_evaluation.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/reference_fixture_manifest.schema.json` | 4748 | `fd0ed4e80ff2aa4df9c37989143416db2ae749cc4c59a8368baa78d472c785be` | 机器可读 JSON 证据 `reference_fixture_manifest.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/schemas/transit_delay_artifact.schema.json` | 4393 | `688e8905299bd41a2c4b559507dd586d9992a83b10d9d601c6bb15e40710c5ee` | 机器可读 JSON 证据 `transit_delay_artifact.schema.json`；keys=$schema,$id,title,description,type,additionalProperties,required,properties,$defs |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__init__.py` | 2766 | `08fce71c0496a6039b4c77809af40b26d5b1f874bfb9c0ea92ca58dbc96c75e2` | 带中英文说明的 final_v0 工具；M3 唯一未来活动信号算法入口 / Sole future-active M3 signal API.；主要入口：script entry only |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/__init__.cpython-310.pyc` | 2190 | `d0201fa6c4e98bca72108709f1fd35c8cafb0eeaa0402e1b4450d8e95f28ed4c` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/contracts.cpython-310.pyc` | 5896 | `1fe5bfdee4538d6c47a8be4590ef7bc50aac52d108aa73a036e1c61496c2b5d8` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/fold_contract.cpython-310.pyc` | 7830 | `e2febc0514ec38343dd637ebf64740209908e9519dbd933b1f515ef70758da05` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/imu.cpython-310.pyc` | 5489 | `7da4fcf566804e1d24ce45a243012d01a9f0d874c1a645b5dea5e0526a1c89c1` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/imu_math.cpython-310.pyc` | 13611 | `56effa2862bc23a63f7dfb3a3d91a82b71953fd845aea03daf27285ca9d0b24a` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/imu_runtime.cpython-310.pyc` | 10104 | `9ac23834903627bf1765b64fbbb93aebedb7866483e28ea47ad2cf1dc4358747` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/physiology.cpython-310.pyc` | 13698 | `c30f3819691293a94957a391d623009bb456c0f1f56102fefdfeef44dfe49a9b` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/ppg.cpython-310.pyc` | 9736 | `8c408af8ae68948e0ca357315fabbde407f23249a135ba1a5e7b54df0ebe17e7` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/quality.cpython-310.pyc` | 10941 | `cf484be5085cf7cac55d9544833b27ac66ee560a20db333458fac840958f53c8` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/reference_evaluation.cpython-310.pyc` | 8968 | `29f9a5779211e624a1c05333a3f7a31b231cc26138d519e3360fa097d5c9dd78` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/registry.cpython-310.pyc` | 2018 | `77538c151ba8236743cc589190a739a65f73c8d80c8f8e72000ffe99c156e075` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/__pycache__/scaling.cpython-310.pyc` | 8891 | `fd06b94af54ace9c593343215163ff185a03b611deb0c4eeb5d5894d08aa4bd0` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/contracts.py` | 5953 | `a22e8072d9a34244f70e54ee077f2b7790bc64e256d1d33af25e25d68a049d82` | 带中英文说明的 final_v0 工具；M3 公共返回合同与状态类型 / M3 shared result contracts and status types.；主要入口：ProcessingStatus, QualityIssue, QualityAssessment, PpgPreprocessResult, ExternalResampleResult, PeakResult, HrvResult, ImuPreprocessResult |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/fold_contract.py` | 8694 | `ec66fe44d03915f0d2f1b9ff6fa661ab83d9e0c5c989f420e766b36bb2b7eb03` | 带中英文说明的 final_v0 工具；M2 物化 fold 与拟合 artifact 绑定 / Bind fitted artifacts to M2 folds.；主要入口：_sha256_file, _ids_hash, _canonical_json_hash, resolve_m2_fold, fit_fold_scaler |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/imu.py` | 8169 | `296e24b76efee4f3417a2932d818c2eea12020658b59b2bef36cdeab2421f787` | 带中英文说明的 final_v0 工具；统一 IMU 公共入口 / Unified public IMU entry point.；主要入口：vector_jerk, _invalid_shape_result, preprocess_imu |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/imu_math.py` | 17770 | `e5e1e4690ab04a184ee9838b856e6495a46f894b0d28b747d4f5e868ab8e9da9` | 带中英文说明的 final_v0 工具；ESKF 数学与共享多轴滤波 / ESKF mathematics and shared axis filtering.；主要入口：convert_imu_to_si, skew, quat_normalize, quat_multiply, quat_exp, quat_to_rotation, quat_from_two_vectors, tangent_basis |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/imu_runtime.py` | 14360 | `f4d214dbe0a1b52fab9c98765de7cd13ee1a920518961332713522881ead3f95` | 带中英文说明的 final_v0 工具；注册表驱动的 stateful IMU runtime / Registry-bound stateful IMU runtime.；主要入口：_eskf_config, _resolve_profile, _failed_result, _common_jerk, CausalImuProcessor, preprocess_imu |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/physiology.py` | 19714 | `34d01586ecd99c08740c8d80b67ae97dbebecbb2aa3cf67cb70ad334c538b703` | 带中英文说明的 final_v0 工具；统一 peak/PPI/HR/PPG-derived PRV 后端 / Shared physiology backend.；主要入口：_window_bounds, _merge_events, derive_ppi, _polarity_score, detect_peaks_corrected, _contiguous_valid_differences, _longest_valid_run, compute_prv |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/ppg.py` | 18468 | `d68a807893b02b590341199822425e83ec3fcba5e44e2f6597b944e21001abe5` | 带中英文说明的 final_v0 工具；版本化 PPG 清洗、滤波与幅值保留 / Versioned PPG cleaning and filtering.；主要入口：design_ppg_sos, raw_ppg_metrics, _source_and_repaired_metrics, dual_ppg_raw_metrics, preprocess_ppg, resample_poly_explicit, resample_external_ppg_to_400, normalized_spectral_entropy |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/quality.py` | 15695 | `b246d0bfae4afdf6275b92d9579ded7fde063f3d0d33b49c77d6af43d5386bdc` | 带中英文说明的 final_v0 工具；输入质量检查与有限缺口修复 / Input quality checks and bounded-gap repair.；主要入口：validate_timestamp_grid, _true_runs, _longest_low_change_run, validate_channel_contract, inspect_and_repair_signal, with_contract_issues |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/reference_evaluation.py` | 11621 | `372dae28e5a0d49365acb8fbe28412ba5dc8806c79a9bb20a2dff39f3667747b` | 带中英文说明的 final_v0 工具；PTT ECG 监督的延迟拟合与评价 / PTT ECG-supervised delay evaluation.；主要入口：_subject_hash, _match_pairs, _following_delays, TransitDelayArtifact, fit_transit_delay, _score, _complete_scorecard, evaluate_ppg_against_ecg |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/registry.py` | 1503 | `7ba93768f6dd7bfa5a475975f17b0d975a2f389caa41683e3790fafff35f5316` | 带中英文说明的 final_v0 工具；M3 版本化 profile 注册表加载器 / M3 versioned profile-registry loader.；主要入口：load_registry, registry_sha256, get_profile |
| `M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core/scaling.py` | 10003 | `701c9cc6278ac0f0cf35bd068c3308da942b54c16d3df9cb2e1b1d0b009faec1` | 带中英文说明的 final_v0 工具；仅训练折拟合的缩放与 raw8 混合视图 / Training-fold-only scaling and raw8 view.；主要入口：_stable_id_hash, FoldScaler, FoldAmplitudeRiskModel, robust_window_scale, build_raw8_model_view |
| `M3_unified_preprocessing_and_signal_algorithms/tests/__init__.py` | 53 | `1032889f42cfa570e66e4e8ce2ca619b728498a62e8e968763b8ebaf508e7dff` | 带中英文说明的 final_v0 工具；M3 reference tests / M3 固定参考测试。；主要入口：script entry only |
| `M3_unified_preprocessing_and_signal_algorithms/tests/_support.py` | 576 | `a3d452e55b2c79d563804ae35689f81b1a958dcc5458837340267dc32e74a28b` | 带中英文说明的 final_v0 工具；测试路径和 fixture helpers / Test paths and fixture helpers.；主要入口：load_fixture |
| `M3_unified_preprocessing_and_signal_algorithms/tests/test_contract_edges.py` | 7803 | `0690151ccb478bb199e73759d19611a07350492cb74be2841db080ced7c83398` | 带中英文说明的 final_v0 工具；M3 合同边界回归 / M3 contract-edge regressions.；主要入口：peak_result, ContractEdgeTests |
| `M3_unified_preprocessing_and_signal_algorithms/tests/test_fold_reference.py` | 6583 | `66b8abca309d2d5563102be0cc3cf6c6b75b594b91f70ff6f237409b5d512b10` | 带中英文说明的 final_v0 工具；M2 fold artifact 与 PTT ECG 评价测试 / Fold and ECG-reference tests.；主要入口：FoldBindingTests, PttReferenceTests |
| `M3_unified_preprocessing_and_signal_algorithms/tests/test_imu_physiology.py` | 13859 | `8ad8310969ed052bbe56c22326d818784c28d6a2208c4c2e7b5d09e2dd11b820` | 带中英文说明的 final_v0 工具；ESKF/LPF、peak/PPI/PRV 固定测试 / IMU and physiology reference tests.；主要入口：ImuReferenceTests, manual_peak_result, PhysiologyReferenceTests |
| `M3_unified_preprocessing_and_signal_algorithms/tests/test_legacy_peak_parity.py` | 2011 | `c270ddd0669f81119aac403e6f397e192e1015eb8095be0a58074e8ec7be7e9d` | 带中英文说明的 final_v0 工具；历史 peak 同输入 parity / Same-input parity for legacy peak implementations.；主要入口：LegacyPeakParityTests |
| `M3_unified_preprocessing_and_signal_algorithms/tests/test_quality_ppg_scaling.py` | 10441 | `394d0d4fc9bda4018e84541bb288db018f692a4a24e519a505fd0c34712dd755` | 带中英文说明的 final_v0 工具；质量门、PPG 与 fold scaling 测试 / Quality, PPG, and scaling tests.；主要入口：QualityGateTests, PpgProfileTests, ScalingTests |
| `M3_unified_preprocessing_and_signal_algorithms/tools/__pycache__/validate_m3_contracts.cpython-310.pyc` | 56698 | `7d696d04d1b8cfca8194274172d340bf3e2301c3d6393f026e7224d2980da5a0` | PYC 文件；用途由路径和相邻审计记录定义 |
| `M3_unified_preprocessing_and_signal_algorithms/tools/build_m3_core_evidence.py` | 13627 | `9b0869817631ebc4b83094f30fca7d7553d6c21e93cb05df11d513999f6420bd` | 带中英文说明的 final_v0 工具；构建 M3 核心机器证据 / Build core M3 machine evidence.；主要入口：sha256_file, write_json, historical_crosswalk, filter_response, synthetic_comparison, m2_integrity_binding, main |
| `M3_unified_preprocessing_and_signal_algorithms/tools/build_m3_frailty_imu_proxy.py` | 8012 | `ea92a8804f0ef764cf94a06a42ce53a24b38c266666e2db59ed67a188a648180` | 带中英文说明的 final_v0 工具；构建 Frailty3 EKF/LPF 配对任务代理 / Build paired Frailty3 IMU proxies.；主要入口：sha256_file, write_json, read_prefix, route_metrics, median_or_none, main |
| `M3_unified_preprocessing_and_signal_algorithms/tools/build_m3_reference_fixtures.py` | 6536 | `aed347f484c2491a01de4327fa14bacf2bd6a36054842fa8d834420ca992ab14` | 带中英文说明的 final_v0 工具；构建确定性 M3 合成 reference fixtures / Build deterministic M3 fixtures.；主要入口：sha256_file, write_npy, write_json, build_ppg, build_imu, main |
| `M3_unified_preprocessing_and_signal_algorithms/tools/legacy_peak_parity.py` | 8806 | `ff18c4aaae29d370734ea2742d6a134b04b3cb28d9351d19fd16321bdf6a49d4` | 带中英文说明的 final_v0 工具；隔离执行历史 peak 函数并生成 parity 证据 / Isolated legacy peak parity.；主要入口：sha256_file, sha256_int64, _load_selected_functions, _arrays_equal, run_legacy_peak_parity, write_json, update_build_report, main |
| `M3_unified_preprocessing_and_signal_algorithms/tools/run_m3_reference_tests.py` | 4764 | `dd2b33aa14c32fd8c5082cd960238c6b88c80d33714aa92d18cf018004ff431d` | 带中英文说明的 final_v0 工具；运行 M3 unittest 并可写 strict JSON 报告 / Run M3 tests and write a report.；主要入口：sha256_file, build_input_snapshot, atomic_write_json, run_suite, main |
| `M3_unified_preprocessing_and_signal_algorithms/tools/validate_m3_contracts.py` | 76745 | `80d1afdb9e6e48dbb0619bd57a2e30a8740220da6ec82d09841edfdbbcbe6c98` | 带中英文说明的 final_v0 工具；验证 M3 合同、权威绑定、机器证据与正式测试报告。；主要入口：StrictJsonError, Audit, sha256_file, stable_json_bytes, compact_snapshot_bytes, _reject_constant, _unique_object, strict_load_json |
| `README.md` | 4272 | `debf64ad42b9766663d54ae421d76725fe209eeb8e371032195f8a923c46c23e` | 文档《`final_v0` 项目收尾工作区 / Project Finalization Workspace》；本目录是本次项目收尾会话唯一允许写入的位置。根目录代码、原始数据、历史结果、`AGENTS.md` 与 `_agent/` 均保持只读。；This directory is the only writable project location for the current finalization session. Root-level source code, original data, historical outputs, `AGENTS.md`, and `_agent/` remain read-only. |
| `algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md` | 2684 | `1ed28714c1f7f46a779f259a31f940dc26e18deaf4c08e3b20e5aea1caf5a9e4` | 算法图《项目历史信号处理总图 / Historical Signal-Processing Map》；含 1 个 Mermaid 图块；本图把 M0 审计到的历史输入、五类 motion/heartbeat 路线、实际证据和最终研究决策放在同一条可追溯链上。实线表示运行数据流；虚线表示监督、评价或审计引用，不表示部署时输入。；ECG 在图中均以虚线进入历史路线，表示它应当只作为监督/评价reference；v7 setup2 把它变成实质推理输入，是已登记的critical leakage。 |
| `algorithm_diagrams/01_PROJECT_END_TO_END_PIPELINE.md` | 2382 | `d33d7e46b14aed00a76bd75b25cc8e40c872d2b4d4471d71fe0dee4e08acb279` | 算法图《当前项目端到端算法总图 / Current End-to-End Project Pipeline》；含 1 个 Mermaid 图块；本图表示当前仓库中实际存在的研究路线与产物，而不是未来 M1–M10 的已实现状态。虚线表示监督、评价、分析或历史依赖；实线表示主要数据/产物流。；Motion/full-waveform 分支已经完成 M0 审计，结论是不恢复完整clean reconstruction。 |
| `algorithm_diagrams/README.md` | 9767 | `b73257bec7c22d0bb53e0464bb28466c7e766896feb31a425edba90e9e75aaf4` | 算法图《Algorithm Diagram Registry / 算法图索引》；含 0 个 Mermaid 图块；Diagram documents / 图文档数量：22；Format / 格式：Markdown + Mermaid |
| `algorithm_diagrams/baseline/00_ARCHIVED_CODE_LINEAGE.md` | 1585 | `6f2e339706863827ea78ca1bec20fe4e8032c4e90ee9c9df1a1be284890e98f8` | 算法图《归档代码版本关系总图 / Archived-code Lineage Map》；含 1 个 Mermaid 图块；箭头表示代码演化或直接替代关系，不表示后代自动修复前代的所有方法学问题。归档输出应归因到其实际生产版本；当前根文件只在有明确schema/路径证据时继承结果。 |
| `algorithm_diagrams/baseline/01_NON_M0_ROOT_SCRIPT_ATLAS.md` | 5755 | `70499233e6f32ada38ca59f8dcd0fef255cf1fa301f6d030d28f5c4ae7aa8a68` | 算法图《非 M0 根脚本与 Notebook 图册 / Non-M0 Root Script and Notebook Atlas》；含 13 个 Mermaid 图块；本图册覆盖根目录中不属于 M0 的8个Python入口和5个Notebook；每节描述当前保存代码的直接职责、输入和输出，不代表未来TODO已完成。；13个图块与 `ROOT_FILE_IO_INVENTORY.md` 的8个非M0 Python脚本、5个Notebook一一对应。M0的16个根代码入口由 `m0/05_SCRIPT_ALGORITHM_ATLAS.md` 覆盖。 |
| `algorithm_diagrams/baseline/02_ARCHIVED_SCRIPT_ATLAS.md` | 5283 | `26c6a8d5454e5b8d3856c5368c36b6b8eb070eea3ba9ddae1e5011b8f2bbe22f` | 算法图《23个归档代码/Notebook结构图册 / Archived Script Atlas》；含 23 个 Mermaid 图块；23个图块与 `CODE_FILES.jsonl` 的23个非根路径一一对应。它们只用于历史溯源和输出归因；当前算法候选仍以根代码为准。 |
| `algorithm_diagrams/m0/01_FOUNDATION_FUNCS_PPG.md` | 1807 | `c8586eb04bfe58fc3ff56f09e1d0221da600aa853312144ad3cc03638018f431` | 算法图《M0 基础函数与 Dash 算法图 / Foundation Functions and Dash Flow》；含 2 个 Mermaid 图块；基础算法透明且可作为 M3 候选，但当前双实现、参数错位、边界coverage和runtime默认路径使其处于 `implemented_unverified`；不得直接当作统一公共实现。 |
| `algorithm_diagrams/m0/02_V7_TO_STAGE2_EVOLUTION.md` | 3068 | `53d16384e83b8ce98ffafd2dc3107be803f5dee76a19a5a2f0175d5e981efb88` | 算法图《v7 至 Stage-2 演化图 / v7-to-Stage-2 Evolution》；含 4 个 Mermaid 图块 |
| `algorithm_diagrams/m0/03_HYBRID_SUITE.md` | 2256 | `de5bb69fc9fe0d2513986380f2c9d87b9ef88aa53b631af56a6311a1b671944f` | 算法图《Hybrid 去噪、导出与运行图 / Hybrid Denoiser Suite》；含 3 个 Mermaid 图块 |
| `algorithm_diagrams/m0/04_HEARTBEAT_AND_MOTION_AB.md` | 2515 | `1d9a75b196e673209d6867468a90e96bb6c99c4893deb62e6690822582cae835` | 算法图《Heartbeat 与 PPG+IMU Motion A/B 图 / Heartbeat and Motion A/B》；含 3 个 Mermaid 图块；P01 与 P02 虽在同一训练脚本中，但输入、target和证据不同：P01 是 PPG-only 的历史 heartbeat尝试；P02 才是 PPG+IMU motion-state benchmark。后续不得把 P02 的外部性能解释为 P01 gate 或 peak 的性能。 |
| `algorithm_diagrams/m0/05_SCRIPT_ALGORITHM_ATLAS.md` | 6934 | `d6681b04fc2c0235be1669ef026001774ddee3f681056e47b77e3629963685d4` | 算法图《M0 逐脚本算法结构图册 / Per-script Algorithm Atlas》；含 16 个 Mermaid 图块；本图册覆盖 M0 范围内承担算法、训练、评价、导出或运行职责的每个脚本。每节只画该文件的直接职责；跨脚本关系见上级总图。；以上16个入口覆盖 M0 中实际承担算法或运行职责的脚本。`funcs.py` 与 `ppg.py` 的重复实现被分别绘制；同一大脚本中的 PPG-only heartbeat 与 PPG+IMU A/B 被画成两条分支，防止证据混用。 |
| `algorithm_diagrams/m0/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md` | 5558 | `dec93e8d4901361ec78c512658ca08ce9bc17657851a7a0d9d3036a40aac1a12` | 算法图《M0 五类方法、统一实现与 Benchmark 算法图》；含 6 个 Mermaid 图块；本图把本轮五类扩展审计连接到三个工程问题：motion detector、denoising 和动态 HR。实线表示未来可执行数据流；虚线表示监督、评价、安全门或失败历史。所有“未实现”节点只是已定义的实现路线，不是现有结果。；已存在且可复用：STFT/ISTFT utilities、部分 IMU preprocessing、P02 detector benchmark、hybrid工程链、当前SQI消融框架。 |
| `algorithm_diagrams/m0/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md` | 3042 | `1cb90ecec9db32867f424c3db82aac91bac53281e075067a26807119690c5905` | 算法图《MAdenoiser 已确认路线到 Frailty 特征选择算法图》；含 4 个 Mermaid 图块；状态：路线已确认；实现、训练与 benchmark 尚未开始。；实线表示用户确认的未来执行顺序，不表示已完成。 |
| `algorithm_diagrams/m0/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md` | 2776 | `3dafee689821990a7318cb2b4c5510a99623e8c4d2855107324bef59da2b7e3a` | 算法图《Activity/Motion 迁移重训、SQI 与恢复特征流程图》；含 3 个 Mermaid 图块；`Rk` 与它前面的实际活动配对，不按 S/W 编号硬编码。；`p_active` 是 activity probability，不是 optical-artifact probability。 |
| `algorithm_diagrams/m1/00_END_TO_END_MOBILE_PIPELINE.md` | 2062 | `35eb6e96a8f38985c39a5cb7df12b86cdc64089fec9cfd3a0ffb0fbbbc9f3d53` | 算法图《M1 端到端移动处理中心架构图》；含 4 个 Mermaid 图块 |
| `algorithm_diagrams/m1/01_END_TO_END_MOBILE_PIPELINE_V2.md` | 1908 | `83019dfe4625ef344940a93d3369df35570e474f7687c5ccc4469fb6f420aea5` | 算法图《M1 V2 端到端移动处理中心架构图》；含 3 个 Mermaid 图块 |
| `algorithm_diagrams/m1/02_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md` | 3484 | `519779862b88b259b039afec7e5dee86449ad28f9425409508d0c1bed7d8a7a6` | 算法图《M1 V3 顺序 SQI–Motion–Denoiser 路由图》；含 4 个 Mermaid 图块；图注：；SQI 与启用后的 Motion detector 可在第一级并行计算；denoiser 不在该并行域中。 |
| `algorithm_diagrams/m2/00_DATA_MANIFEST_DUAL_FOLD_AND_PROTOCOL.md` | 1692 | `b74f94efa6ef47e927dffd851b9656057b9e61a7a675660291d078aa49b35151` | 算法图《M2 数据 Manifest、双 Fold 注册表与评估协议图》；含 1 个 Mermaid 图块；Group 的最小单位始终是 subject；文件和窗口不得独立重新分折。；未来注册表逐折三类齐全且每类 fold count 差不超过 1。 |
| `algorithm_diagrams/m2/01_EXTERNAL_SYNCHRONIZED_REFERENCE_MANIFEST.md` | 1373 | `3fd614053d8d7a404a6b88cf5a924ec85b9533db9114ff82afe17de294bad7f1` | 算法图《M2 外部同步 ECG/PPG/IMU 证据与资格图》；含 1 个 Mermaid 图块；任何 pseudo-ECG peak 数据不得与人工复核 annotation 使用相同 reference-strength 标签；没有 IMU/activity supervision 的源不得报告 motion detector accuracy。 |
| `algorithm_diagrams/m3/00_UNIFIED_PREPROCESSING_AND_SIGNAL_API.md` | 1635 | `b6e8911d3b6b4c10b78fa16d3908954f40820295b00199ae3186565044a2be75` | 算法图《M3 统一预处理与信号 API / Unified Preprocessing and Signal API》；含 1 个 Mermaid 图块；本图固定数据先经过可追溯质量门，再进入任务 profile；EKF 是 IMU 主路线，LPF；只作为输入完全一致的对照。任何 invalid/insufficient 状态都不得伪造特征。 |
| `algorithm_diagrams/m3/01_IMU_EKF_PRIMARY_AND_LPF_COMPARATOR.md` | 1412 | `b5e92b02b41dc325fe463e08efe6425ccbb47bde9b353698cde208bbcc79c936` | 算法图《M3 IMU：无预校准 ESKF 主路线与 LPF 对照》；含 2 个 Mermaid 图块；两条路线共享原始六轴、显式单位、质量 mask、20/40 Hz 前端和 jerk；禁止 EKF；失败后自动输出 LPF，确保比较只改变重力估计方法。 |
| `algorithm_diagrams/m3/02_PEAK_PPI_HR_PRV_COMMON_BACKEND.md` | 1007 | `c65ca1668c401305edb61613958be106e33394f377310c44d217af92e007b25e` | 算法图《M3 Peak、PPI、HR 与 PPG-derived PRV 公共后端》；含 1 个 Mermaid 图块；corrected_v1 不再让异常 PPI 删除峰，也不生成 RED/IR 共识峰；同一公共后端服务；high-quality raw 与 denoised feature 路线。 |
| `algorithm_diagrams/m3/03_REFERENCE_TEST_AND_PARITY_MATRIX.md` | 1068 | `34c79d831c9d3fe02c4cdf9901b9f5e769b68f6f30932ca6feec62c53054ba18` | 算法图《M3 固定 Reference Test 与 Parity 矩阵》；含 1 个 Mermaid 图块；测试从 deterministic fixtures 覆盖质量门、滤波、单位、ESKF/LPF、physiology 和；fold-only scaling；合成真值只作工程验收，不冒充临床验证。 |
| `final_pipeline_v1/MIGRATION.md` | 3532 | `9ce950984045234a0a7f8f3d073c37cac223c489d3ada7433c358c2792171968` | 文档《Legacy-to-V1 migration map / 历史到 V1 迁移映射》；本文件只定义迁移边界，不把任何历史脚本升级为活动实现。所有活动代码位于；`final_pipeline_v1/src/ppg_frailty`，历史路径只读。 |
| `final_pipeline_v1/PROJECT_TREE.md` | 39955 | `f6c341804f6074ab33ba0e796b989fe6c956dd37ff4899f129657a178b8f6b74` | 文档《Final Pipeline V1 detailed tree / 详细文件树》；说明 / Note: PROJECT_TREE.md omits its own hash to avoid recursive self-reference. |
| `final_pipeline_v1/README.md` | 5610 | `57146c3e8a571101d7f014b4f881c8f1d746adcd3f0c1d430570ea1a70dff563` | 文档《PPG Frailty Final Pipeline V1》；状态 / Status: **engineering acceptance checkpoint passed; scientific benchmark not run**；本目录是依据 `CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md` 建立的独立、 |
| `final_pipeline_v1/RUNBOOK.md` | 12165 | `64e915718fc478b4a972dba83c89fdfe532bb9b16654f1d784a23059fc2699df` | 文档《Final Pipeline V1 runbook / V1 运行手册》；The package reads frozen authorities outside V1 only through declared paths. Commands that write results must target a new path below `final_pipeline_v1`; historical root scripts/results are never overwritten.；本包只通过声明路径读取 V1 外的冻结权威文件；所有输出必须写入 `final_pipeline_v1` 内的新路径，禁止覆盖根目录历史脚本和结果。 |
| `final_pipeline_v1/STATUS.md` | 9678 | `8be367fc8042a6b1394409c00725c7235a07e0c258412cb440de11dd5ba988d3` | 文档《Final Pipeline V1 status / V1 状态总览》；1. **Authority and provenance / 权威与溯源** — byte-locked merged specification, versioned configuration, source hashes, exact environment/protocol identity, and fail-closed path boundaries.；2. **Data and folds / 数据与折** — 261 internal records, 29 participants, nine roles, unchanged three-class labels, and the frozen balanced subject-level 5×5 registry with seeds 42/10042/20042/30042… |
| `final_pipeline_v1/WORK_LOG.md` | 85633 | `49a62de9be57e67c698a794077df85d3781380178185274b10bedaa08f5a443d` | 文档《Final Pipeline V1 work log / 工作日志》；状态 / Status: completed；流程 / Process: 完整读取 766 行合并规范，核对 bytes/hash/commit/branch，停止旧 TODO 子任务，并建立 V1 独立边界。 |
| `final_pipeline_v1/artifacts/acceptance/cpu_ci_current.json` | 7309 | `0f2fcdee5096f96a65658fbe607d1f79acc2413ec6a55026ed21938f74a241f3` | 机器可读 JSON 证据 `cpu_ci_current.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/cpu_ci_tests_current.json` | 41497 | `819392c2d027802fcb2ff68ed19f92c727bfe26bdde46688145bef2c85a3122f` | 机器可读 JSON 证据 `cpu_ci_tests_current.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/artifact_parallel_20260815T154022Z_492045.json` | 6045 | `fcc7c2267bb1742a58fe53cf876c0cdb6ab2b4790a05586bdab0bae3c5cec642` | 机器可读 JSON 证据 `artifact_parallel_20260815T154022Z_492045.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/artifact_parallel_20260815T154410Z_493798.json` | 6044 | `96abd3d7886ddf211e3c0cae20c788ac425fe147ea901f241a1a26060abc6ce1` | 机器可读 JSON 证据 `artifact_parallel_20260815T154410Z_493798.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/artifact_parallel_20260815T165715Z_517223.json` | 6042 | `84c1da7ac284fda8d5aad06a5681413b1adb9f744a9074c529ff68a35179e59d` | 机器可读 JSON 证据 `artifact_parallel_20260815T165715Z_517223.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/cli_smoke_20260815T154022Z_492045.json` | 2843 | `b75b5611862311faaa7c9a338f41d1b13c1d44804aca1eb977135ac485a9bbf4` | 机器可读 JSON 证据 `cli_smoke_20260815T154022Z_492045.json`；status=smoke_passed; mode=smoke |
| `final_pipeline_v1/artifacts/acceptance/runs/cli_smoke_20260815T154410Z_493798.json` | 2843 | `b75b5611862311faaa7c9a338f41d1b13c1d44804aca1eb977135ac485a9bbf4` | 机器可读 JSON 证据 `cli_smoke_20260815T154410Z_493798.json`；status=smoke_passed; mode=smoke |
| `final_pipeline_v1/artifacts/acceptance/runs/cli_smoke_20260815T165715Z_517223.json` | 2843 | `b75b5611862311faaa7c9a338f41d1b13c1d44804aca1eb977135ac485a9bbf4` | 机器可读 JSON 证据 `cli_smoke_20260815T165715Z_517223.json`；status=smoke_passed; mode=smoke |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/confusion_matrices.json` | 380 | `6b8b4e42fd8c1d64c3c5a78d1e8aa9c1f7a50accb2a052f50a2313e2a3d8ae33` | 机器可读 JSON 证据 `confusion_matrices.json`；keys=cells,schema_version |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/experiment_result.json` | 6458 | `27ca402202d96357558e29dc783e97a4090054c4077629200341511c2eebfe47` | 机器可读 JSON 证据 `experiment_result.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/metrics_per_fold_seed.json` | 4461 | `71a40f1eefe7b6f745564641cb4811c834a7972a72670be87189be381d6ea373` | 机器可读 JSON 证据 `metrics_per_fold_seed.json`；keys=cells,schema_version |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/oof_file_predictions.parquet` | 17386 | `f280772e95885ebb1d0e3f4162509985d9490c0adbab488828339cee2425d036` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/oof_member_predictions.parquet` | 783 | `af105051d1fbc24315c82060b3bdfc4ce48e519f28706f9d30a996c3da9425ed` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/oof_subject_predictions.parquet` | 17547 | `a426f84ba8aa1d173e7ef2c55086e6b268cf01012f284f7543806e95f115e405` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/oof_window_predictions.parquet` | 840 | `2f2c3586704736c0dc1f562570e47e953705d4197864916d7b6a9c78b8403cac` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/acceptance/runs/experiment_reduced_r0_f0_20260815T165715Z_517223/run_manifest.json` | 4484 | `dfdf2d9ce9c68e2f9927b9bfc215b535653dcceeb23179c3c6dc8832143d0e5d` | 机器可读 JSON 证据 `run_manifest.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/model_parallel_20260815T154022Z_492045.json` | 5384 | `f3efd1f95d6d6a7c066331f85c6727e08e92132c6637b082024f7543a7c00b07` | 机器可读 JSON 证据 `model_parallel_20260815T154022Z_492045.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/model_parallel_20260815T154410Z_493798.json` | 8302 | `b0d5b15b71b06bba8bbc5d89a13286c518ceb88cff00d61cc4541d5471da1e7c` | 机器可读 JSON 证据 `model_parallel_20260815T154410Z_493798.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/model_parallel_20260815T165715Z_517223.json` | 8872 | `3db02ef912f90db861bd1294d97f7765b558a2adc12b8a690130031d696faa9b` | 机器可读 JSON 证据 `model_parallel_20260815T165715Z_517223.json`；status=passed |
| `final_pipeline_v1/artifacts/acceptance/runs/raw_window_ablation_20260815T154022Z_492045.json` | 365 | `247f89e9f0d52bcd5b822491e0cc11b5bc5b6c49f53f84ae127fb349ca52c521` | 机器可读 JSON 证据 `raw_window_ablation_20260815T154022Z_492045.json`；keys=factor,fixed_fields,one_factor_only,results |
| `final_pipeline_v1/artifacts/acceptance/runs/raw_window_ablation_20260815T154410Z_493798.json` | 365 | `247f89e9f0d52bcd5b822491e0cc11b5bc5b6c49f53f84ae127fb349ca52c521` | 机器可读 JSON 证据 `raw_window_ablation_20260815T154410Z_493798.json`；keys=factor,fixed_fields,one_factor_only,results |
| `final_pipeline_v1/artifacts/acceptance/runs/raw_window_ablation_20260815T165715Z_517223.json` | 365 | `247f89e9f0d52bcd5b822491e0cc11b5bc5b6c49f53f84ae127fb349ca52c521` | 机器可读 JSON 证据 `raw_window_ablation_20260815T165715Z_517223.json`；keys=factor,fixed_fields,one_factor_only,results |
| `final_pipeline_v1/artifacts/acceptance/source_snapshot_current.json` | 27121 | `c0472928effe9ca4973dad59ba21dd0283eb5848c55a51742cc00222d91e5875` | 机器可读 JSON 证据 `source_snapshot_current.json`；file_count=159 |
| `final_pipeline_v1/artifacts/acceptance/strict_acceptance_current.json` | 20286 | `1bab31fce649eda48eeddf25c8cd07fc1bd6ad94202689cb03e9dafce5210464` | 机器可读 JSON 证据 `strict_acceptance_current.json`；status=passed; mode=strict |
| `final_pipeline_v1/artifacts/acceptance/strict_acceptance_pending.json` | 9740 | `7c3bd85116e7f18b6ab447741eeba0ba382b4efab8a51bec1a62e5edf77f44c8` | 机器可读 JSON 证据 `strict_acceptance_pending.json`；status=passed_with_pending; mode=allow_pending |
| `final_pipeline_v1/artifacts/audit/baseline_inventory.json` | 2946 | `e6e40848bb94056ec5321acc51062faf4e12b94c564d3656121dd397f24df4a2` | 机器可读 JSON 证据 `baseline_inventory.json`；status=frozen_characterization_only |
| `final_pipeline_v1/artifacts/audit/legacy_characterization.json` | 1647 | `d06ce9bf66494ccdc2881e26c5eea6f75fb2fabe2ea6570af208270545c5c52d` | 机器可读 JSON 证据 `legacy_characterization.json`；status=historical_non_strict_not_eligible_for_v1_ranking |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/confusion_matrices.json` | 70 | `243e5ce4403b5a1d9d0e38251be6e4b01a39dd8107d6b54ba1e7eedef0cff6ca` | 机器可读 JSON 证据 `confusion_matrices.json`；status=failed_closed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json` | 1026 | `e3896dcd553454f7b732b2ea1a0a2c53c86e7f60ed19410eeb6efa8b3cb79a73` | 机器可读 JSON 证据 `experiment_result.json`；status=failed_closed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/metrics_per_fold_seed.json` | 70 | `243e5ce4403b5a1d9d0e38251be6e4b01a39dd8107d6b54ba1e7eedef0cff6ca` | 机器可读 JSON 证据 `metrics_per_fold_seed.json`；status=failed_closed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/oof_file_predictions.parquet` | 1307 | `2b887c5a8c6e06188b82f1a8ae89d550fc6eb4d3179c694a03763bd6b897217a` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/oof_member_predictions.parquet` | 1307 | `2b887c5a8c6e06188b82f1a8ae89d550fc6eb4d3179c694a03763bd6b897217a` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/oof_subject_predictions.parquet` | 1307 | `2b887c5a8c6e06188b82f1a8ae89d550fc6eb4d3179c694a03763bd6b897217a` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/oof_window_predictions.parquet` | 1307 | `2b887c5a8c6e06188b82f1a8ae89d550fc6eb4d3179c694a03763bd6b897217a` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/run_manifest.json` | 1026 | `e3896dcd553454f7b732b2ea1a0a2c53c86e7f60ed19410eeb6efa8b3cb79a73` | 机器可读 JSON 证据 `run_manifest.json`；status=failed_closed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/confusion_matrices.json` | 380 | `6b8b4e42fd8c1d64c3c5a78d1e8aa9c1f7a50accb2a052f50a2313e2a3d8ae33` | 机器可读 JSON 证据 `confusion_matrices.json`；keys=cells,schema_version |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/experiment_result.json` | 6434 | `a05533ac92d855c39fb45800c95610303aaf510381be1fc1fdc918f1958ef6c8` | 机器可读 JSON 证据 `experiment_result.json`；status=passed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/metrics_per_fold_seed.json` | 4461 | `53d10d0a373e0a0d75b034dc82288c8f1ada781bc09561d76f217621c1bd6732` | 机器可读 JSON 证据 `metrics_per_fold_seed.json`；keys=cells,schema_version |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/oof_file_predictions.parquet` | 17389 | `0595f4ce491aefd05f4a1631ec8cadb8981975c9aca2a43e897214ef860e865d` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/oof_member_predictions.parquet` | 783 | `af105051d1fbc24315c82060b3bdfc4ce48e519f28706f9d30a996c3da9425ed` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/oof_subject_predictions.parquet` | 17550 | `c648a63ebbae5cdcccf5dfef4192a15989561d3107adb722d447ba740103f82a` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/oof_window_predictions.parquet` | 840 | `2f2c3586704736c0dc1f562570e47e953705d4197864916d7b6a9c78b8403cac` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference/run_manifest.json` | 4484 | `d97d16f02e710ed4408640be8519fc517806d84366594831a5e7949f9328f386` | 机器可读 JSON 证据 `run_manifest.json`；status=passed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/confusion_matrices.json` | 380 | `6b8b4e42fd8c1d64c3c5a78d1e8aa9c1f7a50accb2a052f50a2313e2a3d8ae33` | 机器可读 JSON 证据 `confusion_matrices.json`；keys=cells,schema_version |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json` | 6453 | `c28674cf626627459b93447f2d4af47c749b126b0d78997e93f64356732d75ef` | 机器可读 JSON 证据 `experiment_result.json`；status=passed |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/metrics_per_fold_seed.json` | 4461 | `909fa94bc91452e1583c8b45653a99fe64e6a6039a1cd3c79cf2367c31015e40` | 机器可读 JSON 证据 `metrics_per_fold_seed.json`；keys=cells,schema_version |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/oof_file_predictions.parquet` | 17386 | `f280772e95885ebb1d0e3f4162509985d9490c0adbab488828339cee2425d036` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/oof_member_predictions.parquet` | 783 | `af105051d1fbc24315c82060b3bdfc4ce48e519f28706f9d30a996c3da9425ed` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/oof_subject_predictions.parquet` | 17547 | `a426f84ba8aa1d173e7ef2c55086e6b268cf01012f284f7543806e95f115e405` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/oof_window_predictions.parquet` | 840 | `2f2c3586704736c0dc1f562570e47e953705d4197864916d7b6a9c78b8403cac` | PARQUET 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/run_manifest.json` | 4484 | `3a1119fab7bdabc24394570a91882d7dc132d2d8c46626294d05705acebb3e74` | 机器可读 JSON 证据 `run_manifest.json`；status=passed |
| `final_pipeline_v1/artifacts/experiments/reference_registry.json` | 995 | `54f61d92e6ed92e9cf31c8babdf637fdcb8d2eeeb97b4e18c5a05fe513b4216d` | 机器可读 JSON 证据 `reference_registry.json`；keys=current_passing_reference,failed_gate_evidence,registry_id,schema_version,superseded_references |
| `final_pipeline_v1/artifacts/test_reports/artifact_comparison_canonical_manual.json` | 4943 | `4787eb45ee8f3c6a74b15ef4a4b53fe3ee66945757ad0524c5bbbe4590f471a7` | 机器可读 JSON 证据 `artifact_comparison_canonical_manual.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/artifact_comparison_manual.json` | 2895 | `e77cdeb5b691142190f523c74e7854d13c9f8b698a52845cfd6864f80e83d79c` | 机器可读 JSON 证据 `artifact_comparison_manual.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/audit_current.json` | 1201 | `7f17c49b3a2e6f9d77f0be062365411c8202e3ce5734debe58b970bb3f541ad6` | 机器可读 JSON 证据 `audit_current.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/cli_via_public_command.json` | 7538 | `7e7b399a28ed0d6b5fabf2395f80b41244d7e51fa930ef1a8af0a09fa43c57b2` | 机器可读 JSON 证据 `cli_via_public_command.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/contracts_current.json` | 1965 | `55d672a30b1e2864865eeee7162af43826c38b2820e925184f89a4c8b4753530` | 机器可读 JSON 证据 `contracts_current.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/data_current.json` | 3714 | `acd70008e38c205dfedb42ae1eaa4857cc1ea02478c864c498c8b70aa303cb2c` | 机器可读 JSON 证据 `data_current.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/dl_fs_ablation_manual.json` | 665 | `e39eb1bfe20dc3751398779d285b78a896c78413766f069a36f12ba21c392aa3` | 机器可读 JSON 证据 `dl_fs_ablation_manual.json`；keys=factor,fixed_fields,one_factor_only,results |
| `final_pipeline_v1/artifacts/test_reports/imu_gravity_comparison_manual.json` | 1170 | `58bfb49165d33a01e0e8e7e5bd08a4216051625c48900717b59fbbf0314fe122` | 机器可读 JSON 证据 `imu_gravity_comparison_manual.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/integration_smoke_canonical_manual.json` | 2843 | `b75b5611862311faaa7c9a338f41d1b13c1d44804aca1eb977135ac485a9bbf4` | 机器可读 JSON 证据 `integration_smoke_canonical_manual.json`；status=smoke_passed; mode=smoke |
| `final_pipeline_v1/artifacts/test_reports/integration_smoke_manual.json` | 2843 | `97fc8e17a5a3ea111a9e6d36913a8ceaee3d8d84ad661f8363b535bd7a2b49b4` | 机器可读 JSON 证据 `integration_smoke_manual.json`；status=smoke_passed; mode=smoke |
| `final_pipeline_v1/artifacts/test_reports/model_comparison_all13_manual.json` | 8886 | `c165f8ce3292ee6eed3ffcd55cfd23de09e18301d460d2b97bb3dd679f2fe9ce` | 机器可读 JSON 证据 `model_comparison_all13_manual.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/model_comparison_manual.json` | 1586 | `c8a8c9f68c760ecbab8347aa2af6eb9d44ab395834f4d6cf3ca93eaf307fbc71` | 机器可读 JSON 证据 `model_comparison_manual.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/models_current.json` | 3708 | `381b651589ea25fe1ed589516d13801aed0fdb7e045a7e723db80aa1afdff273` | 机器可读 JSON 证据 `models_current.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/phase08_cli_all_tests.json` | 30398 | `aebee4959c4184f62317f131a78e9250b9e9624a12d808b297ecaf3d4b0e123a` | 机器可读 JSON 证据 `phase08_cli_all_tests.json`；status=passed |
| `final_pipeline_v1/artifacts/test_reports/physical_time_contract_manual.json` | 148141 | `dbd7f7e029d7eadf01eff8c9cf70caf5acb48162e8c14976b8b492fdcfe123e7` | 机器可读 JSON 证据 `physical_time_contract_manual.json`；keys=case_count,constructed_case_count,design,execution_gate,factor,formal_training_status,forward_case_count,frozen_fold_seed_requirement,not_applicable_case_count,one_factor_only |
| `final_pipeline_v1/artifacts/test_reports/training_current.json` | 3214 | `adc9c69616d3231aee2e1f1260e8215e4205c3a746dfe68218df5f57d77445ff` | 机器可读 JSON 证据 `training_current.json`；status=passed |
| `final_pipeline_v1/configs/feature_matrix_v1.yaml` | 6268 | `879e72d08a50d3b3658e9751d2be4ba6239b69423711968337ae175fbc68d89c` | YAML 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/configs/motion_benchmark_v1.yaml` | 6425 | `ad8f2795eea64ea07a08fa4e68f930a7e09448aa0a0589e0faa74a7aa0a3a7bf` | YAML 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/configs/reference_all_roles_v1.yaml` | 6236 | `2f1569a4b2cd0ecc99e5efa2f146220888fda418a9d15fea5686fc290e1dca00` | YAML 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/configs/reference_static_v1.yaml` | 6224 | `7b348d8f40b840cc3c760d8df3dd937478689b74d35fc5e6b4e45c61292a7a9f` | YAML 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/docs/adr/ADR-001-canonical-experiment-entrypoint.md` | 1013 | `1a97b5051882e5345dd160dadc53be2639b0c42add5f4961bfb91f72dd699e7a` | 文档《ADR-001: Canonical experiment entry point / 规范实验入口》；状态 / Status: accepted_for_v1；依据 / Source: merged dev0 contract §§1, 4, 5.12, 8.2, 10 |
| `final_pipeline_v1/docs/adr/ADR-002-record-manifest-and-fold-freeze.md` | 1291 | `709ace3aa609b57e15273e57559aa43613facbc95a125536478093929567e8fd` | 文档《ADR-002: Record manifest and fold freeze / 记录清单与分折冻结》；状态 / Status: accepted_for_v1；依据 / Source: contract §§2, 5.2, 5.12, 8.3 |
| `final_pipeline_v1/docs/adr/ADR-003-signal-views-and-units.md` | 1069 | `0444f37a469f2bdb85b6dd1c82397d68af09f3fb4ba86c6c7f232988ae3bdc7a` | 文档《ADR-003: Signal views and units / 信号视图与单位》；状态 / Status: accepted_for_v1；依据 / Source: contract §§2, 5.3, 5.4, 6.3, 7 |
| `final_pipeline_v1/docs/adr/ADR-004-window-planning-padding-and-masks.md` | 1069 | `68f03f9373a800816de2b5b1d8b1522c3765d2960654e8a64d61d1372bdbab3d` | 文档《ADR-004: Window planning, padding, and masks / 窗口规划、填充与掩码》；状态 / Status: accepted_for_v1；依据 / Source: contract §§5.3, 5.8, 6.2, 7.4, 8.1 |
| `final_pipeline_v1/docs/adr/ADR-005-prv-eligibility-and-time-axis.md` | 1007 | `80042832276629bfe74997bbaf0eda8e40b323b3570ca45c37027a514bcff99b` | 文档《ADR-005: PRV eligibility and time axis / PRV 资格与时间轴》；状态 / Status: accepted_for_v1；依据 / Source: contract §§3, 5.4, 7.1, 8.1 |
| `final_pipeline_v1/docs/adr/ADR-006-window-file-subject-aggregation.md` | 1186 | `5ad6f43c05e21cf7376a42ee3f6a348e57f8c703cce75d9f07274891389ceeca` | 文档《ADR-006: Window→file→role-aware participant aggregation / 分层聚合》；状态 / Status: accepted_for_v1；依据 / Source: contract §§3, 5.7, 5.11, 5.13, 8.3 |
| `final_pipeline_v1/docs/adr/ADR-007-epoch-selection-and-outer-fold-isolation.md` | 1062 | `ef2327b4fd3fb1de833c28f2897537cfba14afca8a1fd1730b3652f7797092be` | 文档《ADR-007: Epoch selection and outer-fold isolation / Epoch 选择与外层隔离》；状态 / Status: accepted_for_v1；依据 / Source: contract §§2, 5.12, 8.3 |
| `final_pipeline_v1/docs/adr/ADR-008-model-naming-and-original-paper-deviations.md` | 920 | `27825218a164e71d2e9b1ce4f961a9b75362b985477d025f596095fb69e9e6b0` | 文档《ADR-008: Model naming and paper deviations / 模型命名与论文偏差》；状态 / Status: accepted_for_v1；依据 / Source: contract §§5.6, 5.10, 6.1, 9 |
| `final_pipeline_v1/docs/adr/ADR-009-dl-sampling-rate-and-kernel-time-scales.md` | 985 | `990ae0db0969a2fa7245144795fbd689eca8dde8559757dd90e29b0e1715ae2b` | 文档《ADR-009: DL sampling rate and kernel time scales / DL 采样率与卷积时间尺度》；状态 / Status: accepted_for_v1；依据 / Source: contract §§2, 5.3, 6.1, 6.2, 8.3 |
| `final_pipeline_v1/docs/adr/ADR-010-motion-branch-status-and-primary-experiment-boundary.md` | 1073 | `047925905150f68b606778b10581b81e8fa11a8b8aadecaa7c51f864c8889fe8` | 文档《ADR-010: Motion branch and primary experiment boundary / 运动分支与主实验边界》；状态 / Status: accepted_for_v1；依据 / Source: contract §§2, 5.5, 6.3, 7.7 |
| `final_pipeline_v1/docs/adr/ADR-011-representation-modes-and-feature-matrix-contract.md` | 1050 | `57159cebefc792942c1467ba3ee9a46e6670a23f650483198ed81f4effa95576` | 文档《ADR-011: Representation modes and feature matrix / 表征模式与特征矩阵》；状态 / Status: accepted_for_v1；依据 / Source: contract §§5.7, 5.8, 5.9, 5.11 |
| `final_pipeline_v1/docs/adr/ADR-012-post-artifact-rate-only-feature-contract.md` | 1084 | `015eed4b9508d7471fc906db4e3ae7699c3761ec0b05f3862a1c7dc48cea52db` | 文档《ADR-012: Post-artifact rate-only feature contract / 去伪影后仅 Rate 合同》；状态 / Status: accepted_for_v1；依据 / Source: contract §§2, 3, 6.3, 7.2–7.7, 9 |
| `final_pipeline_v1/docs/algorithms/00_END_TO_END_PIPELINE.md` | 4870 | `7e21d37475f00fefcc3d5354ed12e9c1e34608667574b60a7096c1368f61a593` | 文档《End-to-end V1 workflow / V1 端到端流程》；`run` is the real-input/protocol audit and emits no trained metric.；`run-experiment` is the real frozen outer-fold training/evaluation entry. |
| `final_pipeline_v1/docs/algorithms/01_DATA_MANIFEST_FOLDS_AND_LEAKAGE.md` | 1115 | `f83f3bde36caad5349bec77204ea82fe830b723a0ed3adb6c2155406a905994d` | 文档《Data, frozen folds, and fit boundary / 数据、冻结折与拟合边界》；Membership is imported, not regenerated. / 折成员从权威注册表导入，不重新生成。；Every fitted artifact records training participant IDs and rejects OOF IDs. |
| `final_pipeline_v1/docs/algorithms/02_SIGNAL_QUALITY_ARTIFACT_FEATURES.md` | 1212 | `1acc7aa0bd417c4c54cc7c7099a7c3e2f533c5e683faf94ae1bea4bd5c34264f` | 文档《Signal, SQI, artifact, and feature routes / 信号、SQI、伪影与特征路线》；The EKF and LPF branches share units, filtering, masks, timestamps, and output schema;；only the gravity estimator changes. / EKF 与 LPF 只改变重力估计器，其余上游与输出合同一致。 |
| `final_pipeline_v1/docs/algorithms/03_REPRESENTATIONS_AND_PARALLEL_MODELS.md` | 1148 | `6ddc274f0a09786290ce0ba260e4cf7941057a60f89944887bd8b3b383be4408` | 文档《Representations and parallel model families / 表征与并行模型族》；All branches share labels, memberships, role definitions, aggregation, OOF writer, and；participant metrics. / 所有分支共享标签、折、role、聚合、OOF 和 participant 指标。 |
| `final_pipeline_v1/docs/algorithms/04_TRAIN_OOF_BUNDLE.md` | 5334 | `39a1483dcf5395ddd615175663c40f9b9f1bd86ab855808dfbe0f328327b4a35` | 文档《Unified training, OOF, and bundle / 统一训练、OOF 与模型包》；The passing public example is:；`reference_static_v1.yaml` is raw and is not a passing current-runner example. |
| `final_pipeline_v1/docs/algorithms/05_ABLATION_AND_COMPARISON_EXECUTION.md` | 943 | `216ae36ec410da147dd10e8201903a9da2d70d9e1befa62b24d9d0ca7b1d58bd` | 文档《Ablation and comparison execution / 消融与对照执行》；Registered factor families include preprocessing, EKF-vs-LPF, quality/drop policy,；artifact reducers, feature families, sampling rate/kernel duration, representation, |
| `final_pipeline_v1/docs/algorithms/README.md` | 889 | `6a4fba4158859f193db296ee62d7cd9c193fe4bb9146cd0d7291fb820bd29226` | 文档《Algorithm diagrams / 算法图》 |
| `final_pipeline_v1/docs/comparisons/01_SPEC_VS_TODO_OVERLAP_AND_DIFFERENCES.md` | 12722 | `3374bc299cb53bb47d16397dd9989e3e4a1b8e14dbc009bb576c2d2989643c52` | 文档《Specification vs TODO / 实施规范与 TODO 的重合和差异》；本报告比较的是：；1. 用户直接指定的产品合同文件 |
| `final_pipeline_v1/docs/comparisons/02_SPEC_VS_COMPLETED_TODO.md` | 8349 | `1e2a7898a87ddef6dbf13b9330e3c11463b66dd60e2a68c9410359112361a1b0` | 文档《Specification vs completed M0–M3 / 规范与本会话已完成 TODO 的重合和矛盾》；This report uses the current status artifacts for M0–M3 and live source/test audits. It does；not treat a milestone document as proof of an unrun benchmark. In particular, M2 explicitly |
| `final_pipeline_v1/docs/comparisons/03_SPEC_VS_LOCAL_FROZEN_WORKFLOW.md` | 11380 | `24aa55bd55216a5553ffeb5ff338a4eeb6943187eefe0a64afac6272c0fc78d7` | 文档《Specification vs local/frozen implementation by workflow / 按 workflow 对照本地与冻结实现》；[Strict acceptance](../../artifacts/acceptance/strict_acceptance_current.json) and；[CPU CI](../../artifacts/acceptance/cpu_ci_current.json) are the current engineering |
| `final_pipeline_v1/docs/comparisons/04_ALGORITHM_REASONABLENESS_AND_TRADEOFFS.md` | 11274 | `da99be95a46dcf9c2ea65429195a1de3e0ed01b9f263b711ba493477d4855258` | 文档《Algorithm reasonableness, benefits, and limitations / 算法合理性、优点与缺点》；The attached specification is scientifically reasonable and substantially safer than the；historical pipeline because its first objective is **valid comparison**, not guaranteed score |
| `final_pipeline_v1/docs/comparisons/05_V1_TO_V2_CONFIRMATION_SUMMARY.md` | 4934 | `e9536ad863aa93be4b43e339eafa580450a224b5cb6a9aaed127c84c50e0f0eb` | 文档《V1→V2 confirmation summary / V1→V2 逐项确认摘要》；The authoritative detailed list is；[HUMAN_CONFIRMATION_POINTS.md](../../records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md). |
| `final_pipeline_v1/docs/spec/SPEC_LOCK.json` | 543 | `bd8737098657ad5fc10f970867b89a1141a4a10988c9235cd8ee1dd266bca2fa` | 机器可读 JSON 证据 `SPEC_LOCK.json`；status=byte_verified |
| `final_pipeline_v1/manifests/external_records_v1.csv` | 70664 | `e6be12bf1578553dccbcc8fa76c2c1e7be47e38b54e3581b6b03dbe9fc4cb7ee` | CSV 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/manifests/internal_records_v1.csv` | 183942 | `5b5788fff09910e6c224e2548869f4085fd2bbb480adcc92e0f11b09ee0387ee` | CSV 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/model_cards/README.md` | 1169 | `c086b4857d457d8f5b0a5cdd7724efb18377a2fbade47b7d8cbfb7078048c74a` | 文档《Generated model cards / 自动生成模型卡》；Generated by `tools/generate_model_cards.py`; do not hand-edit individual cards.；由唯一注册表生成；单份卡片不应手工修改。 |
| `final_pipeline_v1/model_cards/compact_cnn.md` | 1282 | `92348170bb590d8cb75f9f1021f630a9af9c08da30853a2adf9f1c3a139ffe4e` | 文档《CompactCNN1D》；Machine ID / 机器 ID：`compact_cnn`；Scientific status / 科学状态：`reference_single_network` |
| `final_pipeline_v1/model_cards/extra_trees.md` | 1288 | `f9e5389ee4f498e8a69068022dbd9e1811c69f5c155a13395fde8cf442eaad2c` | 文档《ExtraTrees》；Machine ID / 机器 ID：`extra_trees`；Scientific status / 科学状态：`reference_feature_baseline` |
| `final_pipeline_v1/model_cards/fusion_compact.md` | 1278 | `ff03974542bcdf1ec0b6e548aca83f512624b4ee449ba4cdbc82f415bdd34900` | 文档《FileBagFusionCompact》；Machine ID / 机器 ID：`fusion_compact`；Scientific status / 科学状态：`reference_file_level_fusion` |
| `final_pipeline_v1/model_cards/fusion_inception.md` | 1285 | `af419d8b44b5d0abe84caed0209109632524f8ad84f7a65ccf64177b778a24e1` | 文档《FileBagFusionInception》；Machine ID / 机器 ID：`fusion_inception`；Scientific status / 科学状态：`reference_file_level_fusion` |
| `final_pipeline_v1/model_cards/inception_five_member_ensemble.md` | 1356 | `be56c54f6f2bfb5cefd4b3858c312fe2bcc0925e11bb9efaa67f052fb9c89583` | 文档《InceptionTimeFiveMemberEnsemble》；Machine ID / 机器 ID：`inception_five_member_ensemble`；Scientific status / 科学状态：`optional_five_member_probability_ensemble` |
| `final_pipeline_v1/model_cards/inception_full.md` | 1262 | `772527a6f5e4aa7dc96d1671a58251066231d1a74a4e334b491d6542a879f89f` | 文档《InceptionTimeFull》；Machine ID / 机器 ID：`inception_full`；Scientific status / 科学状态：`reference_single_network` |
| `final_pipeline_v1/model_cards/inception_matrix.md` | 1311 | `7cd0699a4771bdb1f827f28e52117b7b0eb765e5b283643e120213d2537b7d18` | 文档《InceptionTimeMatrix》；Machine ID / 机器 ID：`inception_matrix`；Scientific status / 科学状态：`reference_single_network_mask_aware` |
| `final_pipeline_v1/model_cards/inception_small.md` | 1246 | `5c517d8bd0a8ec90da4abff20413ea839d4c7fb019e260098f3a969b5d58c61d` | 文档《InceptionTimeSmall》；Machine ID / 机器 ID：`inception_small`；Scientific status / 科学状态：`reference_single_network` |
| `final_pipeline_v1/model_cards/logistic_regression.md` | 1312 | `44efc6ebd6d47fa4c778b0546d2985d864d88c36a68bf48f371716d8e9df3676` | 文档《LogisticRegressionL2》；Machine ID / 机器 ID：`logistic_regression`；Scientific status / 科学状态：`reference_feature_baseline` |
| `final_pipeline_v1/model_cards/minirocket_ablation.md` | 1205 | `4caffe822157fc8056172fad0e8bebd5a53cd3d5f5cf65e7f19cc9d6b997ff00` | 文档《MiniROCKET》；Machine ID / 机器 ID：`minirocket_ablation`；Scientific status / 科学状态：`named_engineering_ablation` |
| `final_pipeline_v1/model_cards/rbf_svm.md` | 1265 | `8ab9a4e1f52b9fe13b7864ae8edb5aec9ec32a822de5b8e86f8b357327b72385` | 文档《RBFSVM》；Machine ID / 机器 ID：`rbf_svm`；Scientific status / 科学状态：`reference_feature_baseline` |
| `final_pipeline_v1/model_cards/rocket_numpy.md` | 1294 | `0c811fd1e0aeb3c588ef519ced50977844624ca684b7cd0e9bc4e756e5751e2c` | 文档《ROCKET》；Machine ID / 机器 ID：`rocket_numpy`；Scientific status / 科学状态：`self_contained_project_rocket` |
| `final_pipeline_v1/model_cards/shapeformer_effect_size.md` | 1634 | `0bd527ece3677146e6c4eb269f0900d231fac80ad4b6e832956d1ec667495e35` | 文档《ShapeFormerEffectSize》；Machine ID / 机器 ID：`shapeformer_effect_size`；Scientific status / 科学状态：`experimental_ineligible_for_parity_claim` |
| `final_pipeline_v1/pyproject.toml` | 668 | `648c682ba8b76dde150177ab1b9d1f18cbc7f7cb88c2b8f559ed002813c3f2ae` | TOML 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/records/log_entries/20260815_phase01_spec_lock_and_adr_001_004.md` | 596 | `363de75fe26acd420d0eacfbad307fc58135a0d7d53f04dd51baa9d0b9651a11` | 文档《Phase 01 — Spec lock and ADR 001–004 / 规格锁与前四项 ADR》；状态 / Status: completed；流程 / Process: 完整读取 766 行合并规范，核对 bytes/hash/commit/branch，停止旧 TODO 子任务，并建立 V1 独立边界。 |
| `final_pipeline_v1/records/log_entries/20260815_phase02_adr_005_008.md` | 559 | `0d0f979043c8d143758515d428012844c6e132baa6013c725a73f6c61e9bd206` | 文档《Phase 02 — ADR 005–008 / PRV、聚合、训练隔离与模型命名》；状态 / Status: completed；流程 / Process: 按规范冻结 PRV 时间资格、分层聚合、outer-fold 隔离和模型命名。 |
| `final_pipeline_v1/records/log_entries/20260815_phase03_adr_009_012.md` | 569 | `df7a8cd3d3f3f286ac7bc6cee9135b007b0172568bc1721c00b2b132385fa96d` | 文档《Phase 03 — ADR 009–012 / 时间尺度、运动边界、表征与 Rate-only 合同》；状态 / Status: completed；流程 / Process: 冻结 DL-only resampling、motion primary boundary、四种表征与非恒等 artifact rate-only 强约束。 |
| `final_pipeline_v1/records/log_entries/20260815_phase04_config_contracts_provenance.md` | 664 | `07d2306b0a38443c01dd01017c983632f6624ea1df78cc41f985f1293953db9a` | 文档《Phase 04 — Configuration, typed contracts, and provenance / 配置、类型与溯源》；状态 / Status: implemented_pending_tests；流程 / Process: 在 12 ADR 完成后建立 canonical package、strict YAML/JSON config、跨模块 typed containers 与 training-only provenance guard。 |
| `final_pipeline_v1/records/log_entries/20260815_phase04b_reference_configs_and_baseline.md` | 621 | `301eb9efda48048008a7c64af62b22ecd31335ae9315a0ce7b0019087be422ef` | 文档《Phase 04b — Reference configs and baseline / 参考配置与基线》；Re-scanned the 766-line locked specification before this logical write batch.；Added a generator for four fully resolved YAML configurations; runtime has no config inheritance or hidden behavior defaults. |
| `final_pipeline_v1/records/log_entries/20260815_phase04c1_validator_spec_lock_field_fix.md` | 364 | `ab07c96abf487bccb8b76ec210645e2085386bb639a0be0c812084b50c42ca50` | 文档《Phase 04c.1 — Validator lock-field correction / 验证器锁字段修正》；Re-read `SPEC_LOCK.json` and recomputed the attached specification SHA-256 before editing.；Corrected the validator from nonexistent `sha256` to the authoritative `source_sha256` field. |
| `final_pipeline_v1/records/log_entries/20260815_phase04c_standard_library_test_and_validator.md` | 746 | `64a0e0e7f8b8b3ee20e892e82bb920826ea5f7c13440e497f55169ae39afbf20` | 文档《Phase 04c — Test and validation harness / 测试与验证框架》；Re-scanned the V1 tree and local dependency inventory before writing.；Confirmed pytest, ruff, mypy, and coverage are not installed; no undeclared test dependency was introduced. |
| `final_pipeline_v1/records/log_entries/20260815_phase04d_core_contract_tests.md` | 441 | `bf6afaa3e88d1e79f4f0186d43291db20525cdb0effc1e69b8a31892ce96c4fd` | 文档《Phase 04d — Core contract tests / 核心合同测试》；Re-scanned the V1 source tree and the locked specification before writing.；Added standard-library tests for all four resolved configuration files, exact top-level keys, frozen memberships, and hidden outer labels. |
| `final_pipeline_v1/records/log_entries/20260815_phase04e_algorithm_diagrams.md` | 486 | `0479391ed597793d85bd260645d7c1a9ef1eeedad60d89a560a45e08f91e9498` | 文档《Phase 04e — V1 algorithm diagrams / V1 算法图》；Re-scanned the locked specification sections 4–8 and the active V1 tree before writing.；Added six professional Mermaid diagrams for the end-to-end route, frozen-fold boundary, signal/SQI/artifact features, representation/model families, Trainer/OOF/bundle, and paired comparisons. |
| `final_pipeline_v1/records/log_entries/20260815_phase04f_v2_decision_registry.md` | 496 | `529fad2a80bf269bd7c7ffc47266d6a52a13597141e3ce896ccbd3088ffbfef9` | 文档《Phase 04f — V2 decision registry / V2 决策注册表》；Re-scanned the specification, TODO, M0–M3 current status, and three implementation audits before writing.；Consolidated 27 human-confirmation points with the conservative V1 choice, alternatives, and rerun/deployment impact. |
| `final_pipeline_v1/records/log_entries/20260815_phase04g_matrix_input_dimension_contract.md` | 536 | `12a1795f2fde9181ea1a02adfc4e309ce41177f25bde6688d29a80963a955493` | 文档《Phase 04g — Matrix input-dimension contract / Matrix 输入维度合同》；Re-read the resolved configs and model factory before editing.；Replaced the feature-matrix sentinel `input_channels=-1` with an explicit schema-derived resolution rule; negative dimensions remain invalid. |
| `final_pipeline_v1/records/log_entries/20260815_phase04h_analysis_view_correction.md` | 485 | `f6bcaaf1bb1ed64abd7c90c5d9f6304227867c5e0c1de915c46dd64915a24ad9` | 文档《Phase 04h — Analysis-view correction / Analysis 视图修正》；Re-read contract §5.3, ADR-003, the resolved YAML generator, diagram, and current signal facade.；Removed an unused 0.4–8 Hz secondary direct filter from the configuration. |
| `final_pipeline_v1/records/log_entries/20260815_phase04i_raw_window_padding_alignment.md` | 561 | `6f16d740d75307c1ecb5c927c36a953058beec435ba9d7e5a818c6c1457adb0f` | 文档《Phase 04i — Raw-window padding alignment / Raw 窗口 padding 对齐》；Re-read contract §5.3, the resolved window config, `CompactCNN1D.forward_features`, and feature-matrix padding rules.；The V1 reference raw route now emits complete 5-second windows only; it does not ask a model that rejects non-trivial masks to consume right padding. |
| `final_pipeline_v1/records/log_entries/20260815_phase06_signal_artifact_features.md` | 15022 | `1fc0fe0982a3af6703b8f272758d812475992568211d16e791d289bbcdfd9ab0` | 文档《Phase 06 — Signal, artifact, quality, and features / 信号、伪影、质量与特征层》；Date / 日期: 2026-08-15；Status / 状态: implemented_and_verified_on_synthetic_contract_tests |
| `final_pipeline_v1/records/log_entries/20260815_phase07_models_training.md` | 11884 | `39ead8b8a18abdac8a74c523f77b08db63dbb883632c61b66d5caacbf6f21047` | 文档《Phase 07 — Models, frozen-fold training, OOF evaluation, and bundles》；This phase re-scanned the locked implementation specification, the V1 contracts,；ADR-006/007/008/011, the resolved configuration names, and the reviewed historical |
| `final_pipeline_v1/records/log_entries/20260815_phase08a_spec_todo_comparison_reports.md` | 487 | `bacb11a5569621f84bcc1c13dc28bf120ac6cf66fa4c81ccbcf2d0a4159b3b8e` | 文档《Phase 08a — Specification/TODO comparison reports / 规范与 TODO 对照报告》；Re-read the full TODO, M0–M3 current-status documents, locked specification, and live implementation audits before writing.；Added a requirement-level spec-vs-TODO overlap/difference/contradiction report. |
| `final_pipeline_v1/records/log_entries/20260815_phase08b_remaining_user_reports.md` | 460 | `3ad241e2ac51c272b396dfb0453707597bad1d9fe7f3c6affb6a9e1e74775789` | 文档《Phase 08b — Remaining user-requested reports / 其余用户要求报告》；Re-scanned relevant local source, frozen M0–M3 status, audits, and current V1 interfaces before writing.；Added the workflow-ordered local/frozen-code reuse/change matrix. |
| `final_pipeline_v1/records/log_entries/20260815_phase09a_baseline_regression_gate.md` | 1059 | `9e5166dc3fc4602c8e10bb9fe18445341a4c995301f632007a4482e80f734965` | 文档《Phase 09a — executable baseline regression gate / 可执行基线回归门》；Status / 状态：implemented; validation pending the next full-suite run.；Scope / 范围：only `final_pipeline_v1/tests/audit/`; no historical source was changed. |
| `final_pipeline_v1/records/log_entries/20260815_phase09b_generated_model_cards.md` | 1069 | `0c7696f39f124293d4569335611aec43bd189ba8ee9adbff21627af0a1c9d657` | 文档《Phase 09b — generated model cards / 自动生成模型卡》；Status / 状态：implemented; model-suite validation follows in the final gate.；Scope / 范围：one generator, thirteen model cards, one index, and two registry tests. |
| `final_pipeline_v1/records/log_entries/20260815_phase09c_physical_time_ablation.md` | 1235 | `a71f1f081b444d430bb3a70734f6b59db50d91ea438cf72c1d702579179b16c6` | 文档《Phase 09c — executable physical-time ablation / 可执行物理时间消融》；Status / 状态：implemented and tested.；Scope / 范围：`models/time_scale.py`, optional CompactCNN/Inception constructor |
| `final_pipeline_v1/records/log_entries/20260815_phase09c_training_evaluation_bundle_protocol.md` | 2059 | `baa6f8da2f41946e63d9eaa8511497b1541964a71c170d21387c018092830ea7` | 文档《Phase 09c — training, evaluation, OOF and bundle protocol / 训练评估与部署协议》；Status / 状态：implemented; the isolated training suite passes 28/28 tests.；Scope / 范围：training implementation, training tests, and this immutable phase entry; |
| `final_pipeline_v1/records/log_entries/20260815_phase09d_training_canonical_facade_parity.md` | 850 | `de7daa64616e207e4fb555a2bb61040fb402935cd5a43bcc38c03bfb0282de30` | 文档《Phase 09d — canonical training facade parity / canonical 训练门面一致性》；Status / 状态：implemented; the expanded training suite passes 31/31 tests.；Scope / 范围：three parity tests were added after the phase09c protocol gate. |
| `final_pipeline_v1/records/log_entries/20260815_phase10_strict_acceptance_cpu_ci.md` | 4323 | `31489b93d1cea3a2d7f6108c16a789315edf9ce8a327c79ca1d7152a7b46df55` | 文档《Phase 10 — Strict acceptance and CPU CI / 严格验收与纯 CPU CI》；Date / 日期：2026-08-15；Status / 状态：`complete`, strict gate `16/16`, CPU tests `146/146` |
| `final_pipeline_v1/records/log_entries/20260815_phase11_shapeformer_spec61_repair.md` | 4973 | `82ef0dc23681fce6488a6e9247d1839afdc9e91b3cdcbfbff16a214f4ec9663b` | 文档《Phase 11 — ShapeFormer §6.1 strict repair / ShapeFormer §6.1 严格修复》；Status / 状态：implemented and CPU-regression tested; experimental status retained.；Scope / 范围：`models/shapeformer.py`, strict model factory, model tests, |
| `final_pipeline_v1/records/log_entries/20260815_phase12_documentation_acceptance_handoff.md` | 6392 | `3c52cc5eb32d95f25f50f5e2948b1e6974ed3623ed84562d3baee7bd7d5dab97` | 文档《Phase 12 — Documentation acceptance handoff / 文档验收交接》；Date / 日期: 2026-08-15；Status / 状态: documentation frozen for final machine acceptance / 文档已冻结，等待最终机器验收 |
| `final_pipeline_v1/records/log_entries/20260815_phase12_real_reduced_current_acceptance.md` | 5193 | `5aa06a0ee1ce01abf3946c42bdf39c46afcb895d951077431fa3a38f3bba85d8` | 文档《Phase 12 — Real reduced current acceptance / 真实 reduced current 验收》；Date / 日期：2026-08-15；Status / 状态：`complete` |
| `final_pipeline_v1/records/log_entries/phase05_data_protocol.md` | 11857 | `866d002657b61a3e87eb07b9b073debfa43f84094fe4e2cfb048f1515911505c` | 文档《Phase 05 — Data and protocol layer / 数据与协议层》；Date / 日期: 2026-08-15；Status / 状态: completed_and_verified |
| `final_pipeline_v1/records/log_entries/phase10_experiment_runner.md` | 11267 | `93037f6a871107bb61fab261ac679abd318aead9941d5499f382751356c1196d` | 文档《Phase 10 — Frozen experiment runner / 冻结实验执行器》；Date / 日期: 2026-08-15；Scientific status / 科学状态: implementation verified; reduced results are smoke only / 实现已验证；reduced 结果仅为 smoke |
| `final_pipeline_v1/records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md` | 13227 | `c5843917a07fee9bf2e972a688c9158e6e07d9a7bfd67c5c77d617e9fd8de2a8` | 文档《V2 human-confirmation points / V2 人工确认点》；状态 / Status: **mixed: confirmed, partially confirmed, and pending**. This registry；contains 28 decision points. A V1 conservative default is not automatically a user |
| `final_pipeline_v1/records/v2_decision_points/INITIAL_CONSERVATIVE_DEFAULTS.md` | 1286 | `1839e4de56a70f7b6ce6edee7e6c11aae0c287fbccbcfa40adc51749ea6991a8` | 文档《Superseded initial V2 defaults / 已替代的 V2 初始默认》；状态 / Status: **superseded — do not use as the confirmation authority / 已替代，不再作为确认权威**。；This early seven-item note was created before the complete implementation audit and before |
| `final_pipeline_v1/reports/data_contract_report.json` | 7212 | `8b58a84d400e4749b474ebbd37e5952a4c647331d5b048ef39a3ad4aafa9df5c` | 机器可读 JSON 证据 `data_contract_report.json`；status=pass |
| `final_pipeline_v1/reports/external_data_contract_report.json` | 4035 | `3a424374c727d54bc84061b917cfcfb8e2cc2c07c75cdbbd52803e2d1d45dab2` | 机器可读 JSON 证据 `external_data_contract_report.json`；status=pass_with_provisional_split_pending_confirmation |
| `final_pipeline_v1/splits/sgkf5_repeats_v1.csv` | 34693 | `1693fc71b79411d166b79b56f840bce89b2bfca3d51ab755c85d567ad560b702` | CSV 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/splits/sgkf5_v1.csv` | 7013 | `130b2887eb29a5a534397b4ce4dc7032f9de30ae46533fa0b2c41559ff4a1284` | CSV 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/splits/v1_provisional_external_grouped_split_seed42.csv` | 34297 | `d37926011b61184742d819951329e96f7f87bd34108733fca182a8e08469ec6b` | CSV 文件；用途由路径和相邻审计记录定义 |
| `final_pipeline_v1/src/ppg_frailty/__init__.py` | 1013 | `f256d3662d42333d0ea663f12f249f5dd8227bdff7a45d2569bc5a3b548e5275` | 带中英文说明的 final_v0 工具；PPG frailty final pipeline V1 / PPG 衰弱度最终流程 V1.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/__init__.py` | 1042 | `6e1176cc2c83debff3b29d3a457e096ffdb8b6fbfb5258965ab2590ca2fcf1a7` | 带中英文说明的 final_v0 工具；规范 singular artifact 门面 / Canonical singular artifact facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/base.py` | 296 | `ff4e98e831c06caf6bdc8745d2b13e2a0158b365a9d0a5bf7cd378a0c240d1e5` | 带中英文说明的 final_v0 工具；ArtifactReducer 规范门面 / Canonical ArtifactReducer facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/bss.py` | 327 | `e5e7f1fdb58b7866c22a5e005217893ff9ac174db9e1b6e8f6dc374cedb8a48f` | 带中英文说明的 final_v0 工具；双波长 BSS 门面 / Dual-wavelength BSS facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/decomposition.py` | 262 | `c6481b069e039c6c071cb80a7446765ffdc37897a9b03f4b45dff0a38b270652` | 带中英文说明的 final_v0 工具；非平稳 SSA 门面 / Non-stationary SSA facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/identity.py` | 239 | `85311e7c7cb7b18661b3e4ab32e35ef560ce65167fedf55bd9b5a96fbc9b66d8` | 带中英文说明的 final_v0 工具；恒等对照门面 / Identity-control facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/nlms.py` | 266 | `43da8840868338299f840fb30fff77d09d3a5ad573357b249879ac350b54f30a` | 带中英文说明的 final_v0 工具；IMU-NLMS rate-only 门面 / IMU-NLMS rate-only facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/router.py` | 645 | `1deeb8922e933e23ec0054de51508dc202585d4f9576e87aa5720495975225fa` | 带中英文说明的 final_v0 工具；规范 singular artifact 路由门面 / Canonical singular artifact router.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifact/spectral.py` | 303 | `81e2b879205be7896371b496884445dc3f4c2ee27d2b669f9116d79dd5f98c92` | 带中英文说明的 final_v0 工具；STFT-IMU 谱抑制门面 / STFT-IMU spectral suppression facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifacts/__init__.py` | 876 | `ed597cca391702f5eeba07c203901c67cebbde663d8dd98c36491a07c3cb6dbb` | 带中英文说明的 final_v0 工具；V1 伪影削减公共入口。；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/artifacts/base.py` | 8264 | `c440f05b13b2f3cfbf233c7e203eb1b84525d2b6cbbbc3ded8f61ab689b67b95` | 带中英文说明的 final_v0 工具；ArtifactReducer 公共接口与失败闭合 / Common reducer interface and fail-closed rules.；主要入口：ArtifactReducer, validate_ppg, imu_reference_matrix, parameters_dict, failure_result, success_result, validate_result |
| `final_pipeline_v1/src/ppg_frailty/artifacts/bss.py` | 11603 | `fe4d0cc35fe118eb649564386f34bf4e07253d6a8b61c39602187f81e9d13169` | 带中英文说明的 final_v0 工具；双波长 PCA/FastICA/NMF 盲源分离 / Dual-wavelength blind source separation.；主要入口：BssConfig, _motion_magnitude, _cardiac_fraction, _select_source, _LinearBssReducer, PcaBssReducer, FastIcaBssReducer, NmfBssReducer |
| `final_pipeline_v1/src/ppg_frailty/artifacts/decomposition.py` | 5868 | `83a0d0d020bfea35b79fe9f0495899d843787da4ce26d175da8c8441fe7e9e7f` | 带中英文说明的 final_v0 工具；SSA 主非平稳分解 reducer / Primary singular-spectrum decomposition reducer.；主要入口：SsaConfig, _diagonal_average, _cardiac_concentration, _ssa_channel, SsaReducer |
| `final_pipeline_v1/src/ppg_frailty/artifacts/identity.py` | 1272 | `f65a64b4e2ed41fe16e68610dfc4b0dc9c7b09af5881d38b92c8246a2446fde5` | 带中英文说明的 final_v0 工具；恒等 reducer：direct control / Identity reducer for the direct control.；主要入口：IdentityReducer |
| `final_pipeline_v1/src/ppg_frailty/artifacts/nlms.py` | 5504 | `93421abed9beb78e8bb30f41f9ec0ca95991e1b2828ab55ba19e06665837bd97` | 带中英文说明的 final_v0 工具；带显式 delay taps 的 IMU-reference NLMS ANC / IMU-referenced tapped NLMS ANC.；主要入口：NlmsConfig, NlmsReducer |
| `final_pipeline_v1/src/ppg_frailty/artifacts/router.py` | 5663 | `9b6f2aceccfb27ca2c05d084d13f824ed3f51f2afa070bd95023288e826245af` | 带中英文说明的 final_v0 工具；Reducer registry 与无 fallback 路由 / Reducer registry and no-fallback routing.；主要入口：UnsupportedReducer, _config, get_reducer, ArtifactRouteOutcome, run_artifact_route |
| `final_pipeline_v1/src/ppg_frailty/artifacts/spectral.py` | 10219 | `02e25bf600a8013405a60b1b215ebbff27f3979e266d73230a99eb9704303649` | 带中英文说明的 final_v0 工具；STFT + IMU 谱掩蔽 reducer / STFT suppression using an IMU spectral mask.；主要入口：SpectralMaskConfig, _stft, SpectralMaskReducer |
| `final_pipeline_v1/src/ppg_frailty/bundle/__init__.py` | 691 | `788e96a52ba999cd051fced6bba6ad59278b82033f74879c3e172872b8554786` | 带中英文说明的 final_v0 工具；规范可部署 bundle 门面 / Canonical deployable-bundle facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/bundle/infer.py` | 1595 | `f1254e348dd60fd88e06ed9ff5385b7f59b74b6c9d774fd0776bea07950b1ee4` | 带中英文说明的 final_v0 工具；原始 recording 到 bundle 概率 / Raw recording to bundle probabilities.；主要入口：infer_raw_record |
| `final_pipeline_v1/src/ppg_frailty/bundle/load.py` | 317 | `6daa857eb87bbdeb96e007aa509ccaaa5cc3def6539759453daa938249a4c57a` | 带中英文说明的 final_v0 工具；完整性校验 bundle 加载 / Integrity-checked bundle loading.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/bundle/save.py` | 2166 | `845927c6805ab25e469bcf051a2ba9618ab88059db19bad979b41c99e962ec88` | 带中英文说明的 final_v0 工具；§5.14 完整 metadata 的 bundle 保存 / Bundle saving with complete metadata.；主要入口：validate_bundle_metadata, save_bundle_strict |
| `final_pipeline_v1/src/ppg_frailty/bundle/schema.py` | 268 | `b29e818224fd95ce208a80783969ecdc827a54edd7f339b8e5483cd4d1fddd2e` | 带中英文说明的 final_v0 工具；Bundle schema 身份 / Bundle schema identity.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/cli.py` | 13062 | `b34450430980b9a70442cdd3ff9122c26407ddf8135fbe9672cd828a036c4f70` | 带中英文说明的 final_v0 工具；V1 非交互命令行 / V1 non-interactive command line.；主要入口：_print, _registered_config, build_parser, _run_tests, _build_data, _validate_all_configs, main |
| `final_pipeline_v1/src/ppg_frailty/config.py` | 5262 | `115d44f87ae47c01057cef987154d0829045eee9c62e53c5bf2e34652ff42a24` | 带中英文说明的 final_v0 工具；严格、无隐藏默认的配置合同 / Strict configuration with no hidden defaults.；主要入口：_strict_mapping, _require_exact_keys, canonical_json_bytes, PipelineConfig, validate_config_payload, load_config |
| `final_pipeline_v1/src/ppg_frailty/contracts.py` | 7096 | `c3156d34d0aeccff0d8404fda573d67bc2b8374dcf8964e3ac8a0efe49146265` | 带中英文说明的 final_v0 工具；跨模块类型与科学不变量 / Cross-module types and scientific invariants.；主要入口：RepresentationMode, SignalRoute, QualityState, ManifestRow, SignalViews, QualityComponent, QualityEndpoint, QualityResult |
| `final_pipeline_v1/src/ppg_frailty/data/__init__.py` | 2268 | `b142d9101ac8ba6cd9c2dfd9e70a94b148718c88d2ab0268a66179da67f306bc` | 带中英文说明的 final_v0 工具；数据身份、QC、冻结分折、窗口与缓存 / Data contracts and safeguards.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/data/cache.py` | 7024 | `b0afc0179661fff6855fd22ed355bdff4e397ddb8a148a795d3ba9f5afdb127a` | 带中英文说明的 final_v0 工具；内容寻址、来源绑定的安全缓存 / Provenance-bound content-addressed cache.；主要入口：CacheMissError, StaleCacheError, CacheIdentity, ContentAddressedCache |
| `final_pipeline_v1/src/ppg_frailty/data/external_manifest.py` | 27775 | `159769f566e61b825b7ac8f8d209302f3b9dfc9df551853c5cf417a743b40223` | 带中英文说明的 final_v0 工具；外部 heartbeat/motion 数据合同 / External heartbeat-motion data contract.；主要入口：ExternalManifestError, ExternalRecord, _strict_json, _parse_source_files, _evaluation_role, _adapt_m2_row, _validate_external_record, _record_to_csv |
| `final_pipeline_v1/src/ppg_frailty/data/folds.py` | 15965 | `1d4156337786f06c18291fb3311a33750f75423147425b37251135c1dfa6fb91` | 带中英文说明的 final_v0 工具；只读导入并物化 outer-fold membership / Frozen outer-fold memberships.；主要入口：FrozenFoldAudit, FrozenFoldRegistry, _canonical_registry_payload, load_frozen_memberships, load_m2_frozen_registry, _participant_class_map, validate_frozen_memberships, materialize_assignments |
| `final_pipeline_v1/src/ppg_frailty/data/manifest.py` | 14016 | `8b0038ed5ae2f0f8a49b5ede598681a6795bb7b7d61ad0350bb2e176fc0b0c8a` | 带中英文说明的 final_v0 工具；从 M2 权威快照导入规范 manifest / Import the canonical M2 snapshot.；主要入口：ManifestImportError, _parse_m2_json, convert_m2_row, _validate_manifest_set, load_m2_internal_manifest, _checked_target, write_manifest_csv, load_internal_manifest |
| `final_pipeline_v1/src/ppg_frailty/data/qc.py` | 7981 | `148eb97b0e71d7555217bbe430694afeab7aab2354398e8665620d6faffc40d6` | 带中英文说明的 final_v0 工具；Recording 级 fail-closed QC / Fail-closed recording-level quality control.；主要入口：QCThresholds, QCAssessment, _longest_true_run, parse_failure_assessment, assess_numeric_record |
| `final_pipeline_v1/src/ppg_frailty/data/schema.py` | 10946 | `5e550633b6cc4c1a780e8b810c63d20d8bdf02a0cb566dc9015c6ab486d6dcc0` | 带中英文说明的 final_v0 工具；数据层机器合同 / Machine contracts for the data layer.；主要入口：QCStatus, QCReason, FoldAssignment, _strict_json, _parse_json_field, validate_manifest_row, manifest_row_to_csv, manifest_row_from_csv |
| `final_pipeline_v1/src/ppg_frailty/data/windows.py` | 7332 | `432f1809d249c9fe16de5b6bc8af4206bf5e673fbeabcfdf1dac08401d2656da` | 带中英文说明的 final_v0 工具；工程与 DL 共用 WindowPlan / Unified engineering and DL window planning.；主要入口：ShortRecordError, WindowSlice, WindowPlan, _uniform_indices, extract_window |
| `final_pipeline_v1/src/ppg_frailty/evaluate/__init__.py` | 851 | `ab55f888ae84aae06c622bcf72df6a6df4d4e2d7dbbb76f358f23b6e30cb4672` | 带中英文说明的 final_v0 工具；规范评价门面 / Canonical evaluation facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/evaluate/aggregate.py` | 1423 | `cbb59fc9afd18685d4104acd31aa342a48aeb9addf2f56c46cd607c9c2c4a39d` | 带中英文说明的 final_v0 工具；严格层级聚合的 canonical facade / Canonical strict-aggregation facade.；主要入口：aggregate_hierarchy_strict |
| `final_pipeline_v1/src/ppg_frailty/evaluate/benchmark.py` | 2422 | `b1e5656598b20f89483c8808bfb73d2ba09fe5c72417524a9272a27d8be4551a` | 带中英文说明的 final_v0 工具；Repeat 汇总与配对差值 / Repeat summaries and paired deltas.；主要入口：PairedMetricDelta, _summary, summarize_repeats, paired_metric_delta |
| `final_pipeline_v1/src/ppg_frailty/evaluate/calibration.py` | 1993 | `062bba6b04e0384225364cf909a81f2d6e36196cc4dfb9c88456b32f296e040a` | 带中英文说明的 final_v0 工具；仅 outer-train 可拟合的温度校准 / Outer-train-only temperature calibration.；主要入口：TemperatureCalibrator, fit_temperature |
| `final_pipeline_v1/src/ppg_frailty/evaluate/metrics.py` | 1749 | `2f8365a0300051f65d4c742920543fafbb2168c05d49fd026ca46394e0edf939` | 带中英文说明的 final_v0 工具；参与者指标的 canonical facade / Canonical participant-metrics facade.；主要入口：evaluate_participant_probabilities |
| `final_pipeline_v1/src/ppg_frailty/evaluate/oof.py` | 2603 | `08ff9b339d81150dd0ed74e1f3aa0e0195c74aed9402399e70acb57d1173ce4b` | 带中英文说明的 final_v0 工具；OOF 完整性 canonical facade / Canonical OOF-integrity facade.；主要入口：OofContractAudit, validate_oof_contract |
| `final_pipeline_v1/src/ppg_frailty/experiment.py` | 43948 | `1bc4108f7c5328e36096d1e4cefae7a691d4aa5facd7296b45f1de85b05d4090` | 带中英文说明的 final_v0 工具；冻结 outer-fold 实验执行入口 / Frozen outer-fold experiment entry points.；主要入口：ExperimentResult, _RuntimeRecord, _ExperimentProtocolError, _CellResult, _runtime_imports, _choose_records, _preprocess_records, _fit_quality_calibrator |
| `final_pipeline_v1/src/ppg_frailty/features/__init__.py` | 884 | `ad0453f2ff5a85184132a35e6e86eaa1b20475750193dac851169f3c3d77b7c4` | 带中英文说明的 final_v0 工具；V1 特征层公共入口。；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/features/dual_wavelength.py` | 307 | `737663206cd01391b106a6c198c1d37a13d182c0fad53183445bc004e3a2b463` | 带中英文说明的 final_v0 工具；双波长特征规范门面 / Canonical dual-wavelength feature facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/features/engineering.py` | 11264 | `f0650ef753b8311f0ee633c062685792f84426561952ecad7517d17f74c1cfa4` | 带中英文说明的 final_v0 工具；10 s/5 s 工程特征与 fold-local 变换 / Engineering features and fold-local transform.；主要入口：EngineeringExtraction, FoldFeatureTransform, _entropy, _band_power, _one_channel_features, engineering_feature_names, _imu_columns, extract_engineering_features |
| `final_pipeline_v1/src/ppg_frailty/features/file_vector.py` | 462 | `bf0467233413ef6058b96fff160f422c822b2469ff716df84c43643ac82de7e8` | 带中英文说明的 final_v0 工具；文件级向量的规范路径 / Canonical file-vector path.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/features/morphology.py` | 351 | `0922f32554c8224abeddcf73668cd03cbc452efba8efc92a2e835c4697eee6b7` | 带中英文说明的 final_v0 工具；形态学特征规范门面 / Canonical morphology-feature facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/features/ordered_matrix.py` | 496 | `72a9b73fa39f4cb8a20ff1c3bac1fcf1774c59c04e647a553219143bd95fa86b` | 带中英文说明的 final_v0 工具；有序特征矩阵的规范路径 / Canonical ordered-matrix path.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/features/prv.py` | 268 | `8b4ab350949abcf3f8a99baaf1aebaa275ee5bde06e4899161b19dac8cba254e` | 带中英文说明的 final_v0 工具；PRV 特征规范门面 / Canonical PRV-feature facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/features/registry.py` | 19777 | `8af704b840cbaa462aa9e6cc804ebc1c7d548450931e234b3ec779fb5c49064d` | 带中英文说明的 final_v0 工具；冻结十字段注册表与显式有效性模型编码 / Frozen registry and mask encoding.；主要入口：FeatureDefinition, FeatureRegistry, _hash_definitions, _definition, _prv_unit, default_registry, summarize_engineering, build_feature_vector |
| `final_pipeline_v1/src/ppg_frailty/features/spectral.py` | 511 | `1cfd3b4f951e71af22055c1079dd1e4b7b9bd813fa22c4e54b1afd30c4bf0941` | 带中英文说明的 final_v0 工具；谱特征的规范组合门面 / Canonical spectral-feature composition facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/__init__.py` | 1391 | `28c6e5c39b78c6fdad44e963e52d07160a2d4d53298ecaf73c729e20be4d5373` | 带中英文说明的 final_v0 工具；Public model construction facade for the final pipeline.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/compact_cnn.py` | 5255 | `25c97fbf53c7e93753d65c38d9b68a64e109b18828f737f6bc11af1fc2727783` | 带中英文说明的 final_v0 工具；Compact one-dimensional CNN reference model.；主要入口：CompactCNN1D, trainable_parameter_count |
| `final_pipeline_v1/src/ppg_frailty/models/factory.py` | 17395 | `e1dc95105bc44ccafb07363cbfbe438a0d62749aa321e830679adfd23b3a37fe` | 带中英文说明的 final_v0 工具；Strict model factory spanning all four representation modes.；主要入口：normalize_model_id, normalize_model_config, ModelInputSpec, _torch_seed, create_model |
| `final_pipeline_v1/src/ppg_frailty/models/feature_baselines.py` | 5737 | `9b7c54b19ee4a244fedfe737787cbe6c2f7bf9d77166d866aa83b1edcb948123` | 带中英文说明的 final_v0 工具；Leakage-safe feature-vector baseline models / 防泄漏特征向量基线模型。；主要入口：FeatureVectorBaseline |
| `final_pipeline_v1/src/ppg_frailty/models/feature_models.py` | 273 | `1f201a504d0072b1bc7073041e4e78b564f1dd9e6c9a6897e6b8d83b75e3f82d` | 带中英文说明的 final_v0 工具；文件向量基线模型规范门面 / Canonical feature-model facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/file_fusion.py` | 271 | `2e62b8b65e342880fd78500f67a590468b47669e7b362e4a371829ec8492680b` | 带中英文说明的 final_v0 工具；文件袋 fusion 模型规范门面 / Canonical file-bag fusion facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/fusion.py` | 5308 | `a767f0674711a40ca0618bd52b4f17b7ffe5f170108800611c227cd580140bae` | 带中英文说明的 final_v0 工具；File-bag signal/feature fusion without per-window feature duplication.；主要入口：FileBagFusionClassifier |
| `final_pipeline_v1/src/ppg_frailty/models/inception.py` | 11918 | `1b65ce8ddfc63b12e7816671b0963cfca3ba618d2e5447dcfdda09cdc16bf6f7` | 带中英文说明的 final_v0 工具；Mask-aware InceptionTime single networks and probability ensemble.；主要入口：masked_global_average, InceptionModule, InceptionBlock, InceptionTimeSingleNetwork, FullInceptionTimeSingleNetwork, SmallInceptionTimeSingleNetwork, InceptionTimeFiveMemberProbabilityEnsemble |
| `final_pipeline_v1/src/ppg_frailty/models/inception_ensemble.py` | 319 | `fecdfcbde479996e50cf6c0011b5b9398efd436c0d9e0e044aa1425270536bac` | 带中英文说明的 final_v0 工具；五成员 Inception 集成规范门面 / Canonical five-member Inception facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/inception_time_port.py` | 604 | `71e75e3bbf05e2c9576da63d0df3123acfaeec864daab5ae132b1b67a544fb34` | 带中英文说明的 final_v0 工具；InceptionTime 单网络规范门面 / Canonical single-network InceptionTime facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/rocket.py` | 12414 | `89af46863282b29f01a641065011a8a5d00a0805138dadc7167a50dd4a5f5d0c` | 带中英文说明的 final_v0 工具；Self-contained NumPy ROCKET and explicitly separate MiniROCKET ablation.；主要入口：_normalise_mask, RocketKernel, MaskedChannelRobustScaler, RocketTransformer, RocketRidgeClassifier, MiniRocketAblation |
| `final_pipeline_v1/src/ppg_frailty/models/rocket_ridge.py` | 486 | `1b927f2b76e3fd4dabd0bd0e05c580bda2b47b899ed7e7ac4a7635f228841f46` | 带中英文说明的 final_v0 工具；ROCKET + Ridge 规范门面 / Canonical ROCKET-plus-Ridge facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/shapeformer.py` | 20154 | `eaeac8e414e9e8ed826540cbe945a24c70f79a5b611ea5514e164c33eca0f911` | 带中英文说明的 final_v0 工具；Self-contained experimental effect-size ShapeFormer.；主要入口：EffectSizeShapelets, _participant_roster_hash, _z_normalise_window, _minimum_distance, discover_effect_size_shapelets, ExperimentalShapeFormer |
| `final_pipeline_v1/src/ppg_frailty/models/shapeformer_port.py` | 609 | `c6772438225af70ecd81887b11904da0e2cc6806eae8892eb255e4a322fce91d` | 带中英文说明的 final_v0 工具；Effect-size ShapeFormer 实验门面 / Canonical experimental ShapeFormer facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/models/time_scale.py` | 8370 | `6701660a7f8c390b4fa39ab8f407ac6d6f8818435adf445fd4522f98e57e3e54` | 带中英文说明的 final_v0 工具；Physical-time audit helpers for CNN/Inception ablations.；主要入口：RealizedKernelSet, PhysicalTimeAblationCase, _nearest_odd, realize_kernel_durations, inception_local_receptive_field, build_physical_time_cases, create_time_scaled_model |
| `final_pipeline_v1/src/ppg_frailty/module_registry.py` | 15936 | `0da924b64a31928e43ef92f8343515ec072dcc8235e692ee1e238875c6aec3f4` | 带中英文说明的 final_v0 工具；V1 模块注册表与严格配置适配 / V1 module registry and strict adapters.；主要入口：ModuleDescriptor, list_modules, registry_sha256, resolve_artifact_module_id, resolve_artifact_config, validate_model_config, resolve_window_config |
| `final_pipeline_v1/src/ppg_frailty/peaks/__init__.py` | 355 | `9b6263c78629d09453358e52cb3ead12a6775f454b7cab3715fe2604d2808ac0` | 带中英文说明的 final_v0 工具；脉搏事件、间期和匹配门面 / Pulse event, interval, and matching facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/peaks/aboy_project.py` | 440 | `9aecb5ec9c2c06bd286c5d781958d0bf96f4a063da9ec2d7d5d4ae3e1f23f22a` | 带中英文说明的 final_v0 工具；项目 pulse detector 规范入口 / Canonical project pulse-detector entry.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/peaks/intervals.py` | 249 | `ae816e06ba827d8ecf268554b0cd25cb76f4f73b09a9a25f9ed3a3fb33e66933` | 带中英文说明的 final_v0 工具；PPI/PRV 规范门面 / Canonical PPI/PRV facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/peaks/pairing.py` | 1900 | `a6d658fdc36b403c71115816c683d7cfb15b17fde04143705cbf0dd1df3cd0fc` | 带中英文说明的 final_v0 工具；一对一事件匹配 / One-to-one event matching for heartbeat benchmarks.；主要入口：EventMatchMetrics, match_events |
| `final_pipeline_v1/src/ppg_frailty/pipeline.py` | 45993 | `dd1d0b8d384fa087a810900d34c33eed3d6dcb8cc57eee1fdd33113fd68b25fd` | 带中英文说明的 final_v0 工具；V1 主流水线、预运行与量化比较 / V1 pipeline, preflight, and comparisons.；主要入口：PipelinePaths, PreflightReport, PipelineRunResult, _atomic_json, _config_path, preflight_pipeline, _load_record, _run_real_smoke |
| `final_pipeline_v1/src/ppg_frailty/provenance.py` | 3565 | `22a4dcde470cd5468f5c0cabf1cf4571dfb54bb15e50d45308e6c6bea90f36ee` | 带中英文说明的 final_v0 工具；哈希、原子写入与 fold-local 拟合证明 / Provenance and leakage guards.；主要入口：sha256_file, stable_payload_sha256, atomic_write_json, assert_training_only, FittedArtifactProvenance, runtime_environment |
| `final_pipeline_v1/src/ppg_frailty/quality/__init__.py` | 729 | `e3a6c6d4ebf3872f90e38853534a65e89376b17f52dbbb09a60bc79448867fd9` | 带中英文说明的 final_v0 工具；规范 SQI 门面 / Canonical endpoint-SQI facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/quality/components.py` | 808 | `1318b5a615e1a459bc1eaf0a21315ce53025cec6bb0b4de27cbb2d9500f3bdaf` | 带中英文说明的 final_v0 工具；SQI component 类型与表格化 / SQI component types and tabulation.；主要入口：component_rows |
| `final_pipeline_v1/src/ppg_frailty/quality/endpoint_sqi.py` | 494 | `9c204941865f0e31294b778924751594bd2ece0df8d74415a1f663c98297bfde` | 带中英文说明的 final_v0 工具；端点 SQI 稳定重导出 / Stable endpoint-SQI re-exports.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/quality/routing.py` | 578 | `9c14e71523cb0d0d9c1206405d1fb03330c3a69a45e885805c41b0472aa56e53` | 带中英文说明的 final_v0 工具；质量与 signal route 约束 / Quality-to-signal-route constraints.；主要入口：assert_quality_route |
| `final_pipeline_v1/src/ppg_frailty/representations/__init__.py` | 606 | `5a5ccdc835af57889a7b81e51de9124d3c5f98f810c2d33a0ef3192b77670e98` | 带中英文说明的 final_v0 工具；四种规范 representation 门面 / Four canonical representation facades.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/representations/feature_matrix.py` | 1112 | `9ae7651a8c79f0bcdf767e722cbf1d3a4801f951535a8c21a079477220cb79d4` | 带中英文说明的 final_v0 工具；OrderedFeatureMatrixV1 合同检查 / OrderedFeatureMatrixV1 validation.；主要入口：validate_feature_matrix |
| `final_pipeline_v1/src/ppg_frailty/representations/feature_vector.py` | 1030 | `c2db37588a2a7ca11b806a1f016d1cedd60816b59f953a993737fbc000e8154a` | 带中英文说明的 final_v0 工具；FeatureVectorV1 合同检查 / FeatureVectorV1 contract validation.；主要入口：validate_feature_vector |
| `final_pipeline_v1/src/ppg_frailty/representations/fusion.py` | 705 | `87df132a08d480477f9b93c1e096f2123f23328bdf29841333a294e17936d4f8` | 带中英文说明的 final_v0 工具；文件级融合池化 / File-level fusion pooling.；主要入口：masked_file_mean |
| `final_pipeline_v1/src/ppg_frailty/representations/modes.py` | 396 | `1475f404cb0685fc16e1fb8305f277d303a1f126415de2c1d23af368004ab68c` | 带中英文说明的 final_v0 工具；Representation 枚举检查 / Representation enum validation.；主要入口：assert_mode |
| `final_pipeline_v1/src/ppg_frailty/representations/raw.py` | 3670 | `16236a96a973d3c0c4d6c026d48f8bc6c4d6e274061b5e1b231c13d0b750bb6f` | 带中英文说明的 final_v0 工具；Raw 多通道窗口构建 / Raw multichannel window construction.；主要入口：RawWindows, _robust_scale, build_raw_windows |
| `final_pipeline_v1/src/ppg_frailty/signal/__init__.py` | 2639 | `3cfa57f77451e4763c82c0d0f249977c29eb204dd2f75c1e67cbc2b9ac46ddd2` | 带中英文说明的 final_v0 工具；V1 信号层公共 facade / Public facade for the V1 signal layer.；主要入口：extract_direct_features |
| `final_pipeline_v1/src/ppg_frailty/signal/imu.py` | 38096 | `fc0f3ca865b676e10fd650d109df7465026e39b35215d3fa7b5965139134544a` | 带中英文说明的 final_v0 工具；Stateful IMU preprocessing with the frozen M3 quaternion MEKF.；主要入口：EskfConfiguration, ImuProfile, ImuPreprocessResult, convert_acceleration, convert_gyro, skew, quat_normalize, quat_multiply |
| `final_pipeline_v1/src/ppg_frailty/signal/imu_preprocess.py` | 723 | `80d73449d29b514b8a1f9671c7ba2d69b0b544274f1da14b79661d2fd8a77810` | 带中英文说明的 final_v0 工具；规范 IMU 预处理门面 / Canonical IMU-preprocessing facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/signal/morphology.py` | 6840 | `abc2bd83cf85415a5c031951d23a293fb92f01714562934aea8c31b3d84e5906` | 带中英文说明的 final_v0 工具；仅 direct/identity 可用的逐搏形态 / Beat morphology for direct/identity only.；主要入口：MorphologyResult, require_direct_route, _polarity, _crossing_time, extract_morphology |
| `final_pipeline_v1/src/ppg_frailty/signal/optical.py` | 7532 | `f030c202ae99e6f1b11465d5a0ca23c1cbf542f28b399d5ac72e486b6d8d5950` | 带中英文说明的 final_v0 工具；仅 direct 的双波长 AC/DC、PI 与一致性 / Direct-only dual optical features.；主要入口：OpticalFeatureResult, _safe_ratio, _normalized_xcorr, extract_dual_optical |
| `final_pipeline_v1/src/ppg_frailty/signal/peaks.py` | 6691 | `009c69948fa83edd1b88efa616bb1f2ca1a0f40ec9f0fc4232dbb19635cf9b30` | 带中英文说明的 final_v0 工具；双极性脉搏峰、PPI 与邻接合同 / Dual-polarity pulse, PPI, and adjacency.；主要入口：_Candidate, _robust_scale, _candidate, detect_pulses |
| `final_pipeline_v1/src/ppg_frailty/signal/ppg_preprocess.py` | 553 | `a22b2a4d46ae39ffeb555db51c3a318587e3ab0cd79bd9833d30d65ed156e921` | 带中英文说明的 final_v0 工具；规范 PPG 预处理门面 / Canonical PPG-preprocessing facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/signal/preprocess.py` | 18721 | `69b9f6ae6d6d5f5dd006af278d40fb6c92417f20cdb403996d67cdf550612179` | 带中英文说明的 final_v0 工具；400 Hz PPG 质量检查与零相位预处理 / PPG QC and zero-phase preprocessing.；主要入口：InputQC, validate_timestamp_grid, _true_runs, _longest_constant_run, inspect_and_repair, design_ppg_sos, preprocess_ppg_pair, build_signal_views |
| `final_pipeline_v1/src/ppg_frailty/signal/prv.py` | 12608 | `8b2714e93ae3595319feea8469df2648f705ea3e045480fc4860524263a8bb9b` | 带中英文说明的 final_v0 工具；保留真实时间和邻接关系的 HR/PPI/PRV / Time- and adjacency-preserving PRV.；主要入口：PrvResult, _nan_payload, _sample_entropy, _band_integral, _true_runs, compute_prv |
| `final_pipeline_v1/src/ppg_frailty/signal/resample.py` | 2783 | `fdf6533e6b89f0f35bf108d1d6ddf57f5ca8ff08b5d8a2cf27537a7a48de0e7e` | 带中英文说明的 final_v0 工具；DL-only 抗混叠重采样 / Anti-aliased DL-only resampling.；主要入口：DlResampleResult, resample_dl_view |
| `final_pipeline_v1/src/ppg_frailty/signal/sqi.py` | 26080 | `4a600b875bdbfbf9d0e69d1cfbacc8bfb3e901782d0daf0b77fd7ce255673043` | 带中英文说明的 final_v0 工具；Endpoint-aware SQI：Q_rate 与 Q_morph / Endpoint-aware signal quality.；主要入口：SqiConfig, SqiCalibrator, fit_sqi_calibrator, _component, _welch_metrics, _autocorrelation_periodicity, _template_correlation, _endpoint |
| `final_pipeline_v1/src/ppg_frailty/signal/views.py` | 8428 | `96f8f2c8f45e32a4cb31acdf7c0dad43f4ac17c98b02db8962edc231ed4cccc2` | 带中英文说明的 final_v0 工具；显式信号视图与唯一窗口合同 / Explicit signal views and sole window contract.；主要入口：_matrix, CanonicalSignalViews |
| `final_pipeline_v1/src/ppg_frailty/signal/window_plan.py` | 409 | `eefb61c3ce7b3739e2371cd2fb13991eedb1bd975824ddb704c926837f2b75a1` | 带中英文说明的 final_v0 工具；唯一窗口计划的规范路径 / Canonical path for the sole window planner.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/train/__init__.py` | 801 | `80fae87ca37bba681416ee8b6b36b7e6c680603cd5f9876f232c7783263a9bfa` | 带中英文说明的 final_v0 工具；规范 train 门面 / Canonical training facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/train/datasets.py` | 469 | `97ac0df504af6af50ba7c0a51dae9ea35ee52d7b3c4dfbf46fff2ca235c7511c` | 带中英文说明的 final_v0 工具；训练数据集规范重导出 / Canonical training dataset re-exports.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/train/losses.py` | 799 | `ecca3ab4a0ec6e9f934ed0c6ab4a6cfde79612a360b5974c0a35c3a17c8c396e` | 带中英文说明的 final_v0 工具；训练标签权重工具 / Training-label weight utilities.；主要入口：inverse_frequency_class_weights |
| `final_pipeline_v1/src/ppg_frailty/train/sampling.py` | 324 | `8bcedf14352b2529d5a285be4b2b202e79ba4052e56e56010f6736a551f03bfb` | 带中英文说明的 final_v0 工具；Participant/file/window 平衡采样 / Participant-file-window sampling facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/train/selection.py` | 1052 | `36bf2b7969ec008c2c9d11117dc3df5aafabaea872641af5114a3108728ba4e8` | 带中英文说明的 final_v0 工具；Epoch 选择协议检查 / Epoch-selection protocol checks.；主要入口：validate_epoch_selection |
| `final_pipeline_v1/src/ppg_frailty/train/trainer.py` | 840 | `8b7353bffb622da4f95b5ce26783fd55ed610a36e6fce2b35eae832235fa66cd` | 带中英文说明的 final_v0 工具；统一 Trainer 规范重导出 / Canonical UnifiedTrainer re-exports.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/training/__init__.py` | 3104 | `a58ba47c8f587392bb1d3119620e48171345d93d96ef995f6469d675a7d713c1` | 带中英文说明的 final_v0 工具；Public training, evaluation, OOF and bundle facade.；主要入口：script entry only |
| `final_pipeline_v1/src/ppg_frailty/training/ablation.py` | 3238 | `892e26e57ed8a3be7c166ed14a297db07f6060dbd5cd21f9b93c919f72c70f3f` | 带中英文说明的 final_v0 工具；One-factor ablation and paired-comparison API / 单因素消融与配对比较 API。；主要入口：AblationCase, PairedComparison, _set_dotted, run_ablation_matrix, paired_subject_deltas |
| `final_pipeline_v1/src/ppg_frailty/training/aggregation.py` | 11229 | `44f5d67d3d28eb09b1b4b6f998672ef3c2f0f84ac408181a0119258a235e0e3d` | 带中英文说明的 final_v0 工具；Frozen window-to-file-to-role-to-participant aggregation.；主要入口：HierarchyAggregation, ExperimentIdentity, CoverageSummary, experiment_identity, _identity_key, _coverage_summaries, _mean_probabilities, _group_aggregate |
| `final_pipeline_v1/src/ppg_frailty/training/bundle.py` | 21847 | `c1a21a6906fc6987f1dd71f39a68a219365b175e18c8bbe7903a68afa7379ac2` | 带中英文说明的 final_v0 工具；Integrity-checked model bundles with golden prediction parity.；主要入口：LoadedBundle, _jsonable, validate_bundle_metadata, _atomic_json, _is_torch_model, _predict_model, _apply_transforms, save_bundle |
| `final_pipeline_v1/src/ppg_frailty/training/datasets.py` | 12551 | `3d20634098ade62abec95df2dcc2e5f8dc7825eddaebb67ca4cd4ff26a051f59` | 带中英文说明的 final_v0 工具；Typed datasets for the four frozen representation modes.；主要入口：_require_torch, SampleIdentity, _IdentityDataset, _validate_identities, RawWindowDataset, FeatureVectorDataset, FeatureMatrixDataset, FileBagDataset |
| `final_pipeline_v1/src/ppg_frailty/training/evaluator.py` | 11621 | `757de1846f8fabd2f40e566943c65f62da7940eff2715c49aceaf5f065481a10` | 带中英文说明的 final_v0 工具；Prediction-only evaluation utilities / 只预测评估工具。；主要入口：PerClassMetrics, EvaluationMetrics, RepeatMetricSummary, PairedDeltaSummary, _multiclass_brier, _expected_calibration_error, evaluate_predictions, summarize_repeat_metric |
| `final_pipeline_v1/src/ppg_frailty/training/oof.py` | 12460 | `0676fbdffb1ce7242fe8b447257690b65abcfb04eb02fef26522ae231177e1d9` | 带中英文说明的 final_v0 工具；Strict out-of-fold prediction rows and Parquet writer.；主要入口：OofPredictionRow, validate_unique_subject_oof, validate_expected_oof_roster, OofWriter, write_oof_parquet |
| `final_pipeline_v1/src/ppg_frailty/training/trainer.py` | 37005 | `42a6b6c45ff40e4c1d173bf888e06eb6d7c499da8528910e6b21c951898d277f` | 带中英文说明的 final_v0 工具；Unified frozen-membership trainer with outer-label isolation.；主要入口：_require_torch, FrozenOuterSplit, InnerGroupedSplit, TrainingConfig, FittedObjectProvenance, TrainingResult, dataset_participant_ids, dataset_identities |
| `final_pipeline_v1/tests/__init__.py` | 55 | `0e156266efb46fc50e46be018447b824a6847f5ee61c5414efca4c66356c7d80` | 带中英文说明的 final_v0 工具；V1 验收测试包 / V1 acceptance test package.；主要入口：script entry only |
| `final_pipeline_v1/tests/acceptance/__init__.py` | 61 | `99bab300be044c49a86bad699de4b988d963f6f26b7743f4826f23d65af7b321` | 带中英文说明的 final_v0 工具；验收门禁测试包 / Acceptance-gate test package.；主要入口：script entry only |
| `final_pipeline_v1/tests/acceptance/test_acceptance_gate.py` | 9704 | `4c7c1d7d83005f95b4b7a0ec3e2b72d0a8951cc12e5d2104145a32231625a03f` | 带中英文说明的 final_v0 工具；严格验收器的正例与负例 / Positive and negative tests for the strict gate.；主要入口：AcceptanceGateNegativeTests, AcceptanceGateLiveContractTests |
| `final_pipeline_v1/tests/acceptance/test_external_ecg_and_regression_guards.py` | 7940 | `ee089ff0170877972cbbe6c337581eabe054e037e1162f5bdfe3833a3735be7f` | 带中英文说明的 final_v0 工具；§8.2/§8.3 ECG fixture 与统计守卫 / ECG fixture and regression guards.；主要入口：_matched_pairs, _policy_metrics, ExternalEcgFixtureContractTest, LabelShuffleRegressionTest |
| `final_pipeline_v1/tests/acceptance/test_rocket_10000_serialization.py` | 4190 | `e02ae6ab83196fed90d6a3e6740347e0d3f78494b635a8fd5932e0b7e7878639` | 带中英文说明的 final_v0 工具；ROCKET 10,000-kernel 正式配置序列化 / Full ROCKET serialization test.；主要入口：RocketTenThousandKernelSerializationTest |
| `final_pipeline_v1/tests/artifacts/__init__.py` | 68 | `2fa164b8d4305e52918001fdd66db1dc0ed20444658cd6d9a22a9e5f79f01075` | 带中英文说明的 final_v0 工具；Artifact-reducer unit tests / 伪影 reducer 单元测试。；主要入口：script entry only |
| `final_pipeline_v1/tests/artifacts/test_reducers.py` | 6935 | `da12b774746c9124bd9eb27c548cfc85336133dea8f3e8ad1c560176df5af950` | 带中英文说明的 final_v0 工具；Identity/NLMS/SSA/STFT/BSS contract tests / 所有 reducer 合同测试。；主要入口：contaminated_fixture, ReducerTest |
| `final_pipeline_v1/tests/artifacts/test_router_rate_only.py` | 4787 | `6c88d111c01001f53b40d430b8f9c22f041f9d5809cd4de2f9864466f4ac059b` | 带中英文说明的 final_v0 工具；Router no-fallback and rate-only integration tests / 路由集成测试。；主要入口：views_fixture, RouterTest |
| `final_pipeline_v1/tests/audit/__init__.py` | 75 | `4f3cc349d277be38ac512d0fcfda696fd827310ab8cfba44d721fbdbdebc84b8` | 带中英文说明的 final_v0 工具；Baseline audit regression tests / 历史基线审计回归测试。；主要入口：script entry only |
| `final_pipeline_v1/tests/audit/test_baseline_characterization.py` | 4505 | `28e080303c7627a1e89451e6370de56f7e0328c762b735f5554a0555e752bac6` | 带中英文说明的 final_v0 工具；Executable guards for the frozen dev0 baseline and historical evidence.；主要入口：_sha256, BaselineCharacterizationTests |
| `final_pipeline_v1/tests/cli/__init__.py` | 54 | `37ae208a4cdec2093a1240a0012d4c27bb72f7e97a94b9c8d94b2af52c532029` | 带中英文说明的 final_v0 工具；V1 CLI black-box tests / V1 CLI 黑盒测试。；主要入口：script entry only |
| `final_pipeline_v1/tests/cli/test_cli_commands.py` | 11968 | `4976d91baebf6bada80cd0ac030e0eaff688dbfa55d7e77af32d55eec822337a` | 带中英文说明的 final_v0 工具；非交互 CLI 的 strict-JSON 黑盒验收 / Strict-JSON CLI black-box acceptance.；主要入口：run_cli, CliCommandTests |
| `final_pipeline_v1/tests/contracts/__init__.py` | 48 | `bdf505802c6cfd972148f764a3a386dd4f42164a45aea07df48bc6f72bcc485f` | 带中英文说明的 final_v0 工具；核心合同测试 / Core-contract tests.；主要入口：script entry only |
| `final_pipeline_v1/tests/contracts/test_core_contracts.py` | 4972 | `2a20352c606d2a635b1b05c81643cf8771e365a5d2fbb7f972bebaa7bbf41d8a` | 带中英文说明的 final_v0 工具；配置、类型和规格锁回归 / Configuration, type, and specification-lock regressions.；主要入口：ConfigContractTests, SignalAndQualityContractTests, SpecificationLockTests |
| `final_pipeline_v1/tests/data/__init__.py` | 79 | `1a2280705fbb68acad931e1a8e7fa892ef4a21d2be49c308d890db15c990b011` | 带中英文说明的 final_v0 工具；数据合同标准库测试 / Standard-library tests for data contracts.；主要入口：script entry only |
| `final_pipeline_v1/tests/data/test_folds.py` | 2863 | `13158854f7500df8d89b1ba8144bcf335b1266002e4e9d199ac6d71ff69af089` | 带中英文说明的 final_v0 工具；冻结 internal fold 合同测试 / Frozen internal fold contract tests.；主要入口：FrozenFoldTests |
| `final_pipeline_v1/tests/data/test_manifest_qc.py` | 5541 | `5ccdfb93944c97f8dc4ab8ae095322e19ed04c1b21caf57e13b8539bc2257a06` | 带中英文说明的 final_v0 工具；内部/外部 manifest 与 QC 测试 / Manifest and QC contract tests.；主要入口：ManifestContractTests, QualityControlTests |
| `final_pipeline_v1/tests/data/test_materialized_outputs.py` | 2918 | `d333bf17350d0ca1dc19da0307916192699b7cdddc2250ef33af5668083e397a` | 带中英文说明的 final_v0 工具；已物化 CSV/报告回读测试 / Materialized CSV and report read-back tests.；主要入口：MaterializedOutputTests |
| `final_pipeline_v1/tests/data/test_windows_cache.py` | 4843 | `9adb2e33559227c47811f69caec4212039077382b7723c2f9078dd9fccba7657` | 带中英文说明的 final_v0 工具；统一切窗与内容寻址缓存测试 / Window and cache contract tests.；主要入口：WindowPlanTests, ContentAddressedCacheTests |
| `final_pipeline_v1/tests/features/__init__.py` | 48 | `fac2baf8e27a22f68c3b8cabf62fdb696cd07fc180745be117762dc89c613d21` | 带中英文说明的 final_v0 工具；Feature-layer tests / 特征层测试。；主要入口：script entry only |
| `final_pipeline_v1/tests/features/test_engineering_registry.py` | 11426 | `885e9365b9ed5915aba738fe4a44929ce4c39075d6248836905f7aa8d5311352` | 带中英文说明的 final_v0 工具；Engineering, ten-field registry, vector, and matrix tests / 特征合同测试。；主要入口：resolved_config, views_fixture, engineering_plan, direct_extraction, EngineeringRegistryTest |
| `final_pipeline_v1/tests/integration/__init__.py` | 65 | `877a1ca4dff7f92d571ce74e9faabbbe5ec986bbe26df9e46ea2a00d79684be6` | 带中英文说明的 final_v0 工具；V1 integration acceptance tests / V1 集成验收测试。；主要入口：script entry only |
| `final_pipeline_v1/tests/integration/test_experiment_runner.py` | 9671 | `eda74b5839e2fb4993f5de0824c4a84cc094cda773be4a6be36238363acc1d89` | 带中英文说明的 final_v0 工具；真实 outer-fold runner 集成测试 / Frozen outer-fold runner integration tests.；主要入口：_SyntheticRegistry, _synthetic_record, _synthetic_contract, ExperimentRunnerTest |
| `final_pipeline_v1/tests/integration/test_pipeline_facades.py` | 9445 | `11dbda049073163d9170f45a6968fb984a29f8d26d1db38fe8176b34e9a8c100` | 带中英文说明的 final_v0 工具；Canonical facade、strict config 与 coverage 集成测试。；主要入口：CanonicalFacadeTests, AuthorityContractTests |
| `final_pipeline_v1/tests/models/__init__.py` | 113 | `e8321afd82cd87d38659fcfe4eadce8414e3013772b8c5546a7bfe8b62b9b9e1` | 带中英文说明的 final_v0 工具；English: Standard-library tests for model implementations.；主要入口：script entry only |
| `final_pipeline_v1/tests/models/test_architectures.py` | 11869 | `36d3917db39e684ebb9f9b0b1b1b0732386e85ee83d6d1160084adc71a6259a8` | 带中英文说明的 final_v0 工具；Architecture, masking and factory contract tests.；主要入口：ReviewedArchitectureTests |
| `final_pipeline_v1/tests/models/test_model_cards.py` | 2265 | `2f806a2277ca5278018e8e141aaa541919a764ccf0e0053de614003ed1aaf8e0` | 带中英文说明的 final_v0 工具；Generated model-card identity tests / 自动生成模型卡身份测试。；主要入口：ModelCardTests |
| `final_pipeline_v1/tests/models/test_rocket_and_fusion.py` | 4310 | `c9c4a145c72a231872ee7be386b9538819b346ac1d5ab14c84855280c0a68d53` | 带中英文说明的 final_v0 工具；ROCKET, feature baseline and FileBag fusion tests.；主要入口：_MeanSignalEncoder, RocketAndFusionTests |
| `final_pipeline_v1/tests/models/test_time_scale_ablation.py` | 3603 | `abab30293a3acf925ea5cab1cd4f1febe7e04939f50353b30db199dafca5dfe5` | 带中英文说明的 final_v0 工具；Physical-time CNN/Inception ablation tests / 物理时间消融测试。；主要入口：PhysicalTimeAblationTests |
| `final_pipeline_v1/tests/signal/__init__.py` | 58 | `9013e4d6c48cbf0ba14bdd102ad659ce128705f8edba20b9092a468b2e3c803d` | 带中英文说明的 final_v0 工具；Signal-layer unit tests / 信号层单元测试。；主要入口：script entry only |
| `final_pipeline_v1/tests/signal/test_morphology_optical_sqi.py` | 7154 | `7307b71894798006eb4d3c2b02c28b33f5129d35e81867382fdadfde1f61f722` | 带中英文说明的 final_v0 工具；Direct-only morphology/optical and endpoint SQI tests / 形态、双波长与SQI测试。；主要入口：synthetic_signals, DirectFeatureTest |
| `final_pipeline_v1/tests/signal/test_peaks_prv.py` | 7491 | `654efe14c53bd4528b053405c657b71203752a233b58c3ece2c789967ab70fd3` | 带中英文说明的 final_v0 工具；Peak/PPI/PRV eligibility tests / 峰、间期与 PRV 准入测试。；主要入口：regular_pulse_result, PeakPrvTest |
| `final_pipeline_v1/tests/signal/test_views_preprocess_imu.py` | 11244 | `0a541df5f97decf5f0f9b4a05bcd8f8533995f8a628a1af1b3a696f6ac9a0468` | 带中英文说明的 final_v0 工具；视图、窗计划和 IMU contract 测试 / Signal-view, window, and IMU tests.；主要入口：synthetic_record, signal_config, SignalViewsTest, ImuTest |
| `final_pipeline_v1/tests/training/__init__.py` | 134 | `8ad250bc75e38f634f5f2d068c8c00895fb8e2430e117ddad3d8164ce41ebf2f` | 带中英文说明的 final_v0 工具；English: Standard-library tests for training and evaluation contracts.；主要入口：script entry only |
| `final_pipeline_v1/tests/training/test_formal_protocol_guards.py` | 22410 | `5fb35e1e92641eac63e22d864ef05a465339d605982c653b5bc1e197adcf7697` | 带中英文说明的 final_v0 工具；Formal §5.12–§5.14 protocol regression tests.；主要入口：_TinyEstimator, _RawAdapter, _BrokenAdapter, _TinyTorchClassifier, _identity, _window_row, _formal_participant_row, _formal_metadata |
| `final_pipeline_v1/tests/training/test_oof_aggregation_ablation_bundle.py` | 7102 | `43eedb21a174ff947a347364c5d5f197bd48b5924f5995146e56057efb8a0fff` | 带中英文说明的 final_v0 工具；OOF hierarchy, ablation and deployable bundle tests.；主要入口：_row, OofAggregationTests, AblationAndBundleTests |
| `final_pipeline_v1/tests/training/test_training_isolation.py` | 11141 | `18c4efddbe554d625e4a0da3d496a39ff362ef2993922a8aaa51fb45d68a00bb` | 带中英文说明的 final_v0 工具；Frozen membership, epoch selection and evaluation isolation tests.；主要入口：_TinyClassifier, _identity, TrainingIsolationTests |
| `final_pipeline_v1/tools/acceptance_gate.py` | 66899 | `7a1affa9df8badc3f69026d789e19167978f32b7d981a438fcf91502b5f0806e` | 带中英文说明的 final_v0 工具；V1 严格验收门禁 / Strict V1 acceptance gate.；主要入口：AcceptanceFailure, CheckResult, sha256_file, canonical_json_bytes, python_tree_snapshot, active_source_snapshot, atomic_write_json, load_strict_json |
| `final_pipeline_v1/tools/build_baseline_audit.py` | 6505 | `5e751f96c9444e7c048d643d595387518006c517c8ff42a1f3412db0a960aff2` | 带中英文说明的 final_v0 工具；冻结 dev0 基线库存 / Freeze the dev0 baseline inventory.；主要入口：_sha, _version, _git_text, _strict_write, main |
| `final_pipeline_v1/tools/generate_model_cards.py` | 9084 | `ce5c62aca7fec80a57f0e3de6b928378bba8bde1e3f62670cfbc7a4ca6f50aa3` | 带中英文说明的 final_v0 工具；Generate auditable model cards from the frozen V1 model registry.；主要入口：CardDefinition, _render, generate, main |
| `final_pipeline_v1/tools/materialize_data_contracts.py` | 16204 | `87e42967c1e3b3b044a6d702105acfed2efc7a4c2d92efdc7dbf3ef9105bb502` | 带中英文说明的 final_v0 工具；物化 V1 数据与协议合同 / Materialize V1 data and protocol contracts.；主要入口：_verified_sha, _load_active_protocol, _artifact, _producer_sources, main |
| `final_pipeline_v1/tools/materialize_reference_configs.py` | 12593 | `99dc91426203404a600259aeb5280fe276bef2030ce24b1aebdeac10571cf356` | 带中英文说明的 final_v0 工具；生成完全展开的参考配置 / Materialize fully resolved reference configs.；主要入口：_base_config, _resolved_variants, main |
| `final_pipeline_v1/tools/run_cpu_ci.py` | 18971 | `c9c3689df413d5f6857bd8d77b9214e8dd61ce0f0a5aec381d9320f5b595da2d` | 带中英文说明的 final_v0 工具；V1 CPU-only continuous-integration gate / V1 纯 CPU 连续集成门禁。；主要入口：StageResult, _atomic_json, _parse_last_json, _summarize_stdout, _run_stage, _environment, _package_versions, _active_source_snapshot |
| `final_pipeline_v1/tools/run_test_suite.py` | 7504 | `1f5a4c551c428f103c70453e4137fff8943d8a6b573e150e207d97e5c7e611ac` | 带中英文说明的 final_v0 工具；标准库模块化测试入口 / Standard-library modular test entry point.；主要入口：RecordingResult, RecordingRunner, _discover, _write_report, _test_source_snapshot, main |
| `final_pipeline_v1/tools/sync_tracking.py` | 4206 | `506ff9e897c7997e7fabd9ab896260140c9840b7de531fab158825adb467479e` | 带中英文说明的 final_v0 工具；同步 V1 工作记录、算法索引与详细文件树 / Sync V1 tracking artifacts.；主要入口：_sha256, _atomic_write, _sorted_files, build_work_log, build_algorithm_index, build_tree, main |
| `final_pipeline_v1/tools/validate_v1.py` | 7140 | `4b90abafaadf362dd7815f67be19ba9c737a4cc41e56f0b19d17aed39d9859b5` | 带中英文说明的 final_v0 工具；V1 结构、合同和自审验证器 / V1 structural, contract, and review validator.；主要入口：Check, _sha, _required_paths, _spec_lock, _python_ast_and_bilingual, _no_legacy_runtime_imports, _strict_json, _configs |
| `records/ARCHIVED_CODE_IO_INVENTORY.md` | 6837 | `cf81c62086fa40812905f28ec7c27e52ad67409e69e699b390e7183345ded08a` | 文档《非根归档代码逐文件 I/O 与版本关系 / Archived Code I/O and Lineage Inventory》；状态 / Status：`complete; historical_only`；覆盖 / Coverage：`CODE_FILES.jsonl` 中全部23个非根代码/Notebook，逐字节复扫并核对SHA；所有 `.py` 静态编译通过。 |
| `records/CODE_IO_MASTER_INDEX.md` | 3761 | `78e60dda2d6d166b5e90e0e861ee47a2e1137f066adbc192bfcbd867fce95938` | 文档《52份代码/Notebook I/O 总索引 / Master Code and Notebook I/O Index》；状态 / Status：`complete`；证据 / Evidence：`CODE_FILES.jsonl` 52行，全部逐字节至EOF并记录SHA；根目录29份、非根归档23份。 |
| `records/HUMAN_DECISION_GATES.md` | 3913 | `2c7adbe6d9e8e8fc9a81a9789708dc79bcade842d83166020cdfe83eab65ea7d` | 文档《人工决策门 / Human Decision Gates》；状态 / Status：`no_current_M3_blocker_future_gates_remain`；规则 / Rule：发现会改变研究主线、论文口径、数据cohort或依赖范围的选择时停止；这里只记录选项与影响，不代替用户决定。 |
| `records/M0_ARCHIVED_LINEAGE_EVIDENCE.md` | 3106 | `17e5f4b33e2d225dc72a2e9a42ae5c7e592030023debbcce2ced7182ac6cc331` | 文档《M0 归档版本与实际输出生产关系 / Archived Lineage and Output Provenance》；状态 / Status：`complete_supplement`；目的 / Purpose：解释中间修复版本与现存目录的精确关系；不把修复链重复登记为独立科学方法。 |
| `records/M0_CODE_OUTPUT_CROSSWALK.md` | 9419 | `432718ff0f46c140e3a0ca6729d6e9548fb822be2520d00c0232f65e4d89a159` | 文档《M0 代码—输入—输出对应表 / Code–Input–Output Crosswalk》；状态 / Status：`complete`；对应范围 / Scope：M0 Motion Artifact、动态降噪、Heartbeat 与其公共基础函数。 |
| `records/M0_EXECUTIVE_REPORT.md` | 9014 | `cc297c4e12f68d0858f0b53591d299708993f402085f8a5fba809c00d4bda0d6` | 文档《M0 执行、算法与结果总报告 / M0 Execution, Algorithm, and Results Report》；TODO：M0 完整审计历史 Motion Artifact、动态降噪和 Heartbeat 路线；状态 / Status：`complete` |
| `records/M0_METHOD_REGISTRY.md` | 17119 | `7b4e4f13bebc964fcf2ebe87a258d08629c2922c9ad7895756712d63944f3cd6` | 文档《M0 Motion Processing Method Registry》；状态 / Status：`complete`；证据来源 / Evidence：逐字节代码读取、AST/逐行审计、输入头部 manifests、输出 EOF manifests、实际 JSON/CSV/Markdown、历史项目记录。 |
| `records/M0_PAPER_EVIDENCE.md` | 5977 | `c2e6c988bd827a500dbdca97b1dd6be13cc036088def9d45167200b13b5a70f1` | 文档《M0 论文证据、结果评价与表述边界 / Paper Evidence and Claim Boundaries》；状态 / Status：`confirmed_by_code_and_outputs`；目的 / Purpose：区分可写入论文的事实、只能作为探索性结果的证据，以及禁止作出的性能声明。 |
| `records/M0_RISK_REGISTER.md` | 7541 | `14d212d20bff712802b55696918cdb34fb7d37da94366055f645ec83c192e540` | 文档《M0 风险登记 / Risk Register》；状态 / Status：`complete`；分级 / Severity：`critical` 会使结论无效或运行阻断；`high` 会显著偏置结果/部署；`medium` 限制泛化、可复现性或解释；`low` 为文档/工程质量风险。 |
| `records/PENDING_AGENT_UPDATES.md` | 11604 | `b922cae10fb3101b000d71957c4118af5cac3ba760370f4562c9cee724e47d0f` | 文档《待录入 `_agent` 内容草稿簿 / Pending `_agent` Update Drafts》；候选内容默认为 `draft`，必须注明目标文档、来源、证据和待确认项。；用户明确要求后才整理成逐文档可审核正文。 |
| `records/PROJECT_WIDE_SCAN_FINDINGS.md` | 4315 | `a38e5401967d3b56034ea283dc9bc069f70fa2afc9fd3ad3dfbfe3b090dcf2d6` | 文档《Workspace 全项目扫描发现 / Project-wide Scan Findings》；状态 / Status：`baseline_complete; future_TODO_items_not_executed`；目的 / Purpose：保存 M0 前置全量扫描中发现、但属于 M1–M10 的事实，避免后续重复扫描后遗失上下文。 |
| `records/ROOT_FILE_IO_INVENTORY.md` | 11314 | `9a1550d161a251839187d41f00ccfe0d2022be19f3f0f6e79c872655effdae2e` | 文档《根目录逐文件 I/O 与内容清单 / Root-file I/O and Content Inventory》；状态 / Status：`complete`；覆盖 / Coverage：workspace 根目录 45 个文件，逐份完整读取或按非文本规则登记；其中29个代码/Notebook也在 `CODE_FILES.jsonl` 中逐字节校验。 |
| `records/SCAN_PROTOCOL.md` | 2143 | `82444b736ceedac4293f62ebee2d2e59c01394986a367faef0b8fca13e749180` | 文档《扫描协议与证据要求 / Scan Protocol and Evidence Requirements》；1. 根目录代码和文本文件逐文件、逐字节完整读取。；2. 从代码中提取并记录全部可识别的输入路径、输出路径、输出内容及输出结构。 |
| `records/WORK_LOG.md` | 56324 | `b5301375b804ba9f8d8f0c86045a2056659a0a780fcc33e807ad19719aea7d94` | 文档《工作日志 / Work Log》；Mermaid静态验证：`pass`；10份图文档、67个Mermaid图块，结构/fence无失败。；代码图覆盖：`pass`；52/52真实manifest路径均有逐脚本图入口。 |
| `records/decisions/20260803_m0_activity_motion_supervision.md` | 1905 | `f4f10640e4926a70b9e470bcf33f65aafa22b2ca3534f0b499abbc1495e75580` | 文档《M0-MOT-001 — 29-subject Activity/Motion 监督与时序特征决定》；日期 / Date：2026-08-03；状态 / Status：`confirmed` |
| `records/decisions/20260803_m0_madenoiser_route.md` | 1834 | `ca93169b3a0167598ddebbaa7c8d0d95ee8ff50f30703c2ae6791d0653ac57a3` | 文档《M0-MAD-001 — MAdenoiser 后续路线决定》；日期 / Date：2026-08-03；状态 / Status：`confirmed` |
| `records/decisions/20260814_m1_architecture_mobile_profiles.md` | 1724 | `4a1cdec6f4664f2b1195e7e19c7faac5ffd171a89e2f2396c6bdf6b3b0edd9c9` | 文档《M1-ARCH-001 — 端到端 API、质量动作与中心处理平台决定》；日期 / Date：2026-08-14；状态 / Status：`confirmed_user_constraints_contract_defined` |
| `records/decisions/20260814_m1_architecture_mobile_profiles_v2.md` | 1410 | `245d286115c8f4feb0d7d80ff669c20c58d3d94198d49d40bf72864157b90666` | 文档《M1-ARCH-002 — 最终审计后的 V2 流式、artifact 与回退合同》；日期 / Date：2026-08-14；状态 / Status：`contract_defined_waiting_user_acceptance` |
| `records/decisions/20260815_m1_quality_routing_v3.md` | 2259 | `8c4ff901364349e6c9d7019ced3fa9e806a605a9b695279caeb3d10594f15107` | 文档《M1-ARCH-003 — SQI-first、可选 Motion 与 low/motion 手动路由》；日期 / Date：2026-08-15；状态 / Status：`user_direction_recorded_contract_defined_waiting_m1_reacceptance` |
| `records/decisions/20260815_m2_dual_registry_and_stage_mapping.md` | 1257 | `adcd99610946758484c4573156478aa35b8c58412ddd21296c4a74b0f56271ed` | 文档《M2-DATA-001 — 双 Fold 注册表、5×5 主协议与阶段语义》；日期 / Date：2026-08-15；状态 / Status：`user_confirmed_contract_defined_candidates_not_rerun` |
| `records/decisions/20260815_m3_unified_preprocessing_contract.md` | 3328 | `63d61017510e0d02818a43b7ea38102ad7e5c12717d5e92e643ede41897a4a7b` | 文档《M3-PREPROCESS-001 — 统一预处理、无预校准 EKF 主路线与公共生理后端》；日期 / Date：2026-08-15；状态 / Status：user_confirmed_contract_frozen |
| `records/generated/ALGORITHM_DIAGRAM_VERIFICATION.json` | 4623 | `83336b1e187d97b62a303fd401424200d605e4bee9adc7c0301346c68a82ed45` | 机器可读 JSON 证据 `ALGORITHM_DIAGRAM_VERIFICATION.json`；status=pass; diagram_file_count=18; mermaid_block_count=93; failures=list[0] |
| `records/generated/BASELINE_SUMMARY.json` | 375 | `8cb2a6ead19c107cefb2dd1f4e1f65fc8533576a1d87749e360a58aa252af881` | 机器可读 JSON 证据 `BASELINE_SUMMARY.json`；error_count=0 |
| `records/generated/CODE_DIAGRAM_COVERAGE.json` | 6904 | `44f522b54d717a099543298506094b2fb4c02713678a3bfc44faacd0f3b965ff` | 机器可读 JSON 证据 `CODE_DIAGRAM_COVERAGE.json`；status=pass; failures=list[0] |
| `records/generated/CODE_FILES.jsonl` | 57269 | `92f9b0328fced5a72922cfc8eea8c0bd8fcc44f97090d5b0ac4f0a09d50ec2d0` | 全部代码/notebook逐字节读取与结构 manifest；52 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/CODE_PATH_REFERENCES.jsonl` | 767183 | `af487aca29c73018988af74e0805ff4e84c15142cfb61775b8d6d76e714b8132` | 代码静态输入/输出路径字符串引用清单；2387 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/FINAL_V0_VERIFICATION.json` | 6187 | `7edcd545f546700ef2bb51284597bbd7c67f0491a99360752587b8dbeb540e2a` | 机器可读 JSON 证据 `FINAL_V0_VERIFICATION.json`；status=pass; failures=list[0] |
| `records/generated/ROOT_FILES.jsonl` | 37555 | `e2843bac57e526da14c8cd7ed9fce604ceb48591997008ba732e58d1e1956bd1` | 根目录逐文件完整读取 manifest；45 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/SCAN_RUNS.jsonl` | 8377 | `c43da31b5de360796888bbc61b51539664490000d62bc98ab8202b8af66c9fbf` | baseline、输入和输出扫描事务账本；25 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/SCAN_VERIFICATION.json` | 4259 | `e562b99a17854050defefef45c4caacc52470420706fc27429d697fe0c52e0df` | 机器可读 JSON 证据 `SCAN_VERIFICATION.json`；status=pass; failures=list[0] |
| `records/generated/TOP_LEVEL_DIRECTORIES.json` | 7081 | `6b79fd6b4f78a71e8cb4ed7a67292e3917162c530c5e3e2e7157e492c4c72ebf` | 机器可读 JSON 证据 `TOP_LEVEL_DIRECTORIES.json`；top-level=list, items=32 |
| `records/generated/WORKSPACE_FILES.jsonl` | 7928412 | `8cf503c87fe3fb155c40cc35958001c910361956b7a148d118dad66bd5fb8b04` | workspace全文件树元数据 manifest；35214 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/PPG_Testing_05_01_2026.jsonl` | 930699 | `008f5e441f1228985e4ba4d28ca69cf4b03683e9ddc71003ebdbfc5b40cc7425` | 输入目录 `PPG_Testing_05_01_2026` 的逐文件头部/schema manifest；1134 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/PPG_Testing_05_01_2026.summary.json` | 277 | `7f16b8e6c23016576fe547476813cd34980255caf3388527a4e89d225415e135` | 机器可读 JSON 证据 `PPG_Testing_05_01_2026.summary.json`；file_count=1134; error_count=0 |
| `records/generated/inputs/datasets.jsonl` | 219497 | `465408894615304f1ccc08fd239586c32f3251a026c1396182ef8b45fccd633e` | 输入目录 `datasets` 的逐文件头部/schema manifest；327 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/datasets.summary.json` | 233 | `ae57e2d56c2b627a1a3e45694cc79a39b177786d9385eb46d3d9de7a7d1f485d` | 机器可读 JSON 证据 `datasets.summary.json`；file_count=327; error_count=0 |
| `records/generated/inputs/physionet.org.jsonl` | 3937308 | `873c1b76020e54366362bb8806d0815ef01c8e4301aab360b3d487f67865dcdb` | 输入目录 `physionet.org` 的逐文件头部/schema manifest；4920 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/physionet.org.summary.json` | 288 | `1875d00e52c9329d04d81b8819b366ae5a5a1716b234876118b25c9ea570c151` | 机器可读 JSON 证据 `physionet.org.summary.json`；file_count=4920; error_count=0 |
| `records/generated/inputs/train_labeled.jsonl` | 3496 | `5f89ce96d37f776fc06f326cf0151da2b36f1d1a7e731d07bce4df164bfafb65` | 输入目录 `train_labeled` 的逐文件头部/schema manifest；4 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/train_labeled.summary.json` | 214 | `40e3e7ffb14abc6265da293be78cac451540e91d5daf0d26f5e84a78ca951ed8` | 机器可读 JSON 证据 `train_labeled.summary.json`；file_count=4; error_count=0 |
| `records/generated/inputs/train_raw.jsonl` | 6638 | `49da93ae7f223a579d499266923c5dec1348ed0f0114d0ea0da65ce7fb632310` | 输入目录 `train_raw` 的逐文件头部/schema manifest；7 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/train_raw.summary.json` | 215 | `532d191fe3d0dea7028a852fc3f7bfda4841a3d1ab6d578b7bb4bfcdb87ed06f` | 机器可读 JSON 证据 `train_raw.summary.json`；file_count=7; error_count=0 |
| `records/generated/inputs/train_val.jsonl` | 4725 | `224d27c2857e589425dddbc743620b3c375379b4ed862ce7e155fb5fdddb97cc` | 输入目录 `train_val` 的逐文件头部/schema manifest；1 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/train_val.summary.json` | 213 | `dee2f6f9dcc82c487be16549a68d6be5b076634a6c06b653cc169cc10c632a43` | 机器可读 JSON 证据 `train_val.summary.json`；file_count=1; error_count=0 |
| `records/generated/inputs/train_window.jsonl` | 47000 | `28c6780663cc04adc173023c4141d8e0ab32aa57c0c1db090d5cfabf2c627c5c` | 输入目录 `train_window` 的逐文件头部/schema manifest；12 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/inputs/train_window.summary.json` | 218 | `cb58074aabf7c1088e04322d349be50b931444abc8111d19bc0e4c863b39c0c0` | 机器可读 JSON 证据 `train_window.summary.json`；file_count=12; error_count=0 |
| `records/generated/outputs/.CNN_results.jsonl` | 456692 | `3f76b67590d5b17e55a0b7dac9c45d0899b81492bf0256f680d465d0213ec7ff` | 输出目录 `.CNN_results` 的文本EOF/二进制元数据 manifest；687 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/.CNN_results.summary.json` | 352 | `e2e864632044318a4ceb8773dcf136528192e5fc8d1602970ba1180d3f6a63ba` | 机器可读 JSON 证据 `.CNN_results.summary.json`；file_count=687; error_count=0 |
| `records/generated/outputs/denoiser_preview_output.jsonl` | 1334 | `aaf64a5a9d4c2a11095e580ac10dcc6e53c7d16c8478e4f62f79f727ddf5071c` | 输出目录 `denoiser_preview_output` 的文本EOF/二进制元数据 manifest；8 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/denoiser_preview_output.summary.json` | 258 | `f2ca400b7ca021c7df49f046cf4a71978da2f0a7eadff6aba9454dfba78e58cb` | 机器可读 JSON 证据 `denoiser_preview_output.summary.json`；file_count=8; error_count=0 |
| `records/generated/outputs/models.jsonl` | 104592 | `a310e1ab10dcc60dcb69eb6e88c14d8bb882615f97d381de2d81e816bc2c8214` | 输出目录 `models` 的文本EOF/二进制元数据 manifest；653 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/models.summary.json` | 280 | `818257059cc3f148b7a682ddb8ee86df5087bd8693706c02e5ac0f6af384c243` | 机器可读 JSON 证据 `models.summary.json`；file_count=653; error_count=0 |
| `records/generated/outputs/results.jsonl` | 6574 | `df1aed3aa2e284840e93498ae1d3300c710554a4c24c603f4a8f72f9f4bd3c6b` | 输出目录 `results` 的文本EOF/二进制元数据 manifest；5 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results.summary.json` | 240 | `d642178e58439226705345a0bf88044dc90c395d65f8d487ae48c52f8e1c323f` | 机器可读 JSON 证据 `results.summary.json`；file_count=5; error_count=0 |
| `records/generated/outputs/results_denoiser_v8.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 输出目录 `results_denoiser_v8` 的文本EOF/二进制元数据 manifest；0 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_denoiser_v8.summary.json` | 231 | `ebf1f8d3f80e47c68f689c58251d9b03833330ad45c88e8f6e0454f061ca9e59` | 机器可读 JSON 证据 `results_denoiser_v8.summary.json`；file_count=0; error_count=0 |
| `records/generated/outputs/results_detector_v8.jsonl` | 1946 | `6431dcb6e02e5be0c071693b93acfadc51c94d3f8042fe5ec46884aa554913d0` | 输出目录 `results_detector_v8` 的文本EOF/二进制元数据 manifest；6 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_detector_v8.summary.json` | 284 | `e34fae91fc71b3892a603503fad49535207348c0b72ebccb04b9178d504fd087` | 机器可读 JSON 证据 `results_detector_v8.summary.json`；file_count=6; error_count=0 |
| `records/generated/outputs/results_frailty3.jsonl` | 56446291 | `4b49b24368b7b6605a8a52174e3d30d60e1f943fd67f299aff2b58b2852c3ae1` | 输出目录 `results_frailty3` 的文本EOF/二进制元数据 manifest；14496 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_frailty3.summary.json` | 319 | `e4432c91530a63009313e819cbdb0dabc40e9fa3af8468ee039b93b02fbec241` | 机器可读 JSON 证据 `results_frailty3.summary.json`；file_count=14496; error_count=0 |
| `records/generated/outputs/results_hybrid_denoiser.jsonl` | 3626 | `1d753e819b0c94d75900925dfb38e452dffd9fac53b3f77fefc3fb345dfd670c` | 输出目录 `results_hybrid_denoiser` 的文本EOF/二进制元数据 manifest；6 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_hybrid_denoiser.summary.json` | 273 | `0d5f98bcf7072f0fa7ce3cf0323b6fe502f2a5941858ae03b3183c0c17f6a7c9` | 机器可读 JSON 证据 `results_hybrid_denoiser.summary.json`；file_count=6; error_count=0 |
| `records/generated/outputs/results_hybrid_denoiser_raw_imu.jsonl` | 3948 | `9368bde239facc34c04dc51145e2c2b1ac094c76222cc6d984e82c7a0731c478` | 输出目录 `results_hybrid_denoiser_raw_imu` 的文本EOF/二进制元数据 manifest；8 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_hybrid_denoiser_raw_imu.summary.json` | 313 | `ef876dc94e26df82cf3fd0eadaa715092220143e97c42b5ac123720698d83f4d` | 机器可读 JSON 证据 `results_hybrid_denoiser_raw_imu.summary.json`；file_count=8; error_count=0 |
| `records/generated/outputs/results_hybrid_denoiser_raw_imu_baseline.jsonl` | 3906 | `5f64548aa7cb031709db78b41a249ba758b536305afee43fac2ee44a4c9cbde9` | 输出目录 `results_hybrid_denoiser_raw_imu_baseline` 的文本EOF/二进制元数据 manifest；8 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_hybrid_denoiser_raw_imu_baseline.summary.json` | 322 | `5649e32d09cbd3502540065e95a8f9ee03e55dfe44186fce5016f9be93112cab` | 机器可读 JSON 证据 `results_hybrid_denoiser_raw_imu_baseline.summary.json`；file_count=8; error_count=0 |
| `records/generated/outputs/results_stage1.jsonl` | 4240 | `34aa71e46ac802d4984fc281752694f9877c9ffd7470d63b5fcf63a322176e76` | 输出目录 `results_stage1` 的文本EOF/二进制元数据 manifest；17 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_stage1.summary.json` | 282 | `490692f73a7ec03885bfb020efe3c2917643c365b2009286302f3a78209f92be` | 机器可读 JSON 证据 `results_stage1.summary.json`；file_count=17; error_count=0 |
| `records/generated/outputs/results_stage2.jsonl` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | 输出目录 `results_stage2` 的文本EOF/二进制元数据 manifest；0 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_stage2.summary.json` | 226 | `869828aa6fe291ab74b98ec13fc54e630c04318850d1b6914eb20db606e8e6bd` | 机器可读 JSON 证据 `results_stage2.summary.json`；file_count=0; error_count=0 |
| `records/generated/outputs/results_v72_noleak.jsonl` | 24894 | `b768afbee485b8690fae8e23eda90224e3f44b38e93c13b6c52c624a97856aa5` | 输出目录 `results_v72_noleak` 的文本EOF/二进制元数据 manifest；16 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_v72_noleak.summary.json` | 285 | `ee4a88a2c718e76533f189d81e02340490e6c1917f81fca311e818578674134f` | 机器可读 JSON 证据 `results_v72_noleak.summary.json`；file_count=16; error_count=0 |
| `records/generated/outputs/results_v7_3.jsonl` | 10361 | `f73d4c98a2b3f7b3803300cc95a9073f0ac90375f37ebc5f6fc35f6894db9539` | 输出目录 `results_v7_3` 的文本EOF/二进制元数据 manifest；33 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_v7_3.summary.json` | 312 | `99c87886e3798a2da3cdd51e4b8c789535c2e4c8599c2d06bb5f88939098ceca` | 机器可读 JSON 证据 `results_v7_3.summary.json`；file_count=33; error_count=0 |
| `records/generated/outputs/results_v7_4.jsonl` | 15816 | `29428e241558b9e90fc4a2a7d98926b0c5465dea9ac4262e82db1c1140ee6d79` | 输出目录 `results_v7_4` 的文本EOF/二进制元数据 manifest；55 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_v7_4.summary.json` | 327 | `72e65eceddad71d9089aff63758a7d1f5b562cfe10e8ab878f47c640e5c57d6f` | 机器可读 JSON 证据 `results_v7_4.summary.json`；file_count=55; error_count=0 |
| `records/generated/outputs/results_v8_audit.jsonl` | 11682 | `7d189f1ca9d203ed122853aa3cde93af8625d99f159d6ed1231cdf09d8fcb23b` | 输出目录 `results_v8_audit` 的文本EOF/二进制元数据 manifest；30 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/results_v8_audit.summary.json` | 284 | `f032ac9f5d97d53d2c5c2c6bf5ff20d07fee8b2e30e377176f612dafa4d6a994` | 机器可读 JSON 证据 `results_v8_audit.summary.json`；file_count=30; error_count=0 |
| `records/generated/outputs/test_asa_classifier.jsonl` | 2920530 | `f80e3c0c64f2b0e9017093d274c95d7731cb832ebb7087d4d3d6e68a78b0062d` | 输出目录 `test_asa_classifier` 的文本EOF/二进制元数据 manifest；12642 条记录；每行保留路径、读取模式、结构和完整性字段 |
| `records/generated/outputs/test_asa_classifier.summary.json` | 352 | `5ade14ba66ae1d7c8fced5d0abb9375a26c08242a45bd42f1bfbc7a6a9d4622d` | 机器可读 JSON 证据 `test_asa_classifier.summary.json`；file_count=12642; error_count=0 |
| `records/log_entries/20260802_all_code_diagrams_verified.md` | 417 | `edcb6b1e283fe70abfe65a896db981e0a09d477f62ad6485d139d1d486efa2b0` | 不可变工作日志《Markdown document》；Mermaid静态验证：`pass`；10份图文档、67个Mermaid图块，结构/fence无失败。；代码图覆盖：`pass`；52/52真实manifest路径均有逐脚本图入口。 |
| `records/log_entries/20260802_archived_code_inventory_and_diagrams.md` | 575 | `48c2603933a3535b6491da6cb99a3491b95b3818b9f6b464c6cf21ae21948cc1` | 不可变工作日志《Markdown document》；扫描：23份代码/Notebook逐字节复扫、SHA核对；所有Python静态编译通过。；写入：`ARCHIVED_CODE_IO_INVENTORY.md`、归档lineage总图、23入口逐文件结构图册。 |
| `records/log_entries/20260802_code_master_index_and_coverage_tool.md` | 491 | `00f108310102c02f2376709385c2788b0f1238e31c8435a7b492f60ca9f3afca` | 不可变工作日志《Markdown document》；写入：`CODE_IO_MASTER_INDEX.md`与`verify_code_diagram_coverage.py`。；分组：16个M0根入口、13个非M0根入口、23个非根归档入口；总计52，与代码manifest一致。 |
| `records/log_entries/20260802_delivery_verifier_added.md` | 458 | `c53b9c65efc88629098094562c3dca1213e89875814559cadf8bfeaec2ed904b` | 不可变工作日志《Markdown document》；写入：`tools/verify_final_v0_delivery.py`。；检查：所有final_v0 Python的AST和双语说明、路径不越界、必需文档、详细树覆盖、扫描/算法图验证状态。 |
| `records/log_entries/20260802_detailed_tree_indexer_added.md` | 540 | `f56c3b7cde55d93ac13a8b8a13eee8530a389a6e67c647230b9d78fe916da7b9` | 不可变工作日志《Markdown document》；原因：旧索引虽有完整树、字节数和SHA，但对部分报告/manifest的说明过于通用。；写入：`tools/update_final_v0_index_detailed.py`；Markdown提取标题/实质段落，JSON/JSONL提取范围/记录数，Python提取模块职责/入口。 |
| `records/log_entries/20260802_m0_algorithm_atlas_added.md` | 616 | `dc47333ed976c96cad6c64381b42d1ff50dda35ce0861bff224bb22954e5e53a` | 不可变工作日志《Markdown document》；写入：项目历史信号总图、基础函数图、v7→Stage-2演化图、Hybrid套件图、Heartbeat/Motion A-B图、16入口逐脚本图册。；图示约定：实线为运行数据流；虚线为监督、评价、风险或审计引用；阻断和空结果直接落在相应算法节点旁。 |
| `records/log_entries/20260802_m0_algorithm_diagrams_verified.md` | 415 | `e3b1222ae76776d0a12ba79508329f5f789114601ae008d045d7504bb91015a1` | 不可变工作日志《Markdown document》；验证器：`tools/verify_algorithm_diagrams.py`。；结果：`pass`；6份算法图文档、29个 Mermaid 图块、16个预期 M0 脚本入口全部覆盖。 |
| `records/log_entries/20260802_m0_archived_lineage_evidence.md` | 440 | `13db5f66035448c92229197c0b157f300a116c3b4e3a4d313cf9e7583fa9d828` | 不可变工作日志《Markdown document》；写入：`M0_ARCHIVED_LINEAGE_EVIDENCE.md`。；核对：Stage1 holdout OR/AND、legacy detector v8、v7.3、v8 audit fix链和v7.2目录归因。 |
| `records/log_entries/20260802_m0_crosswalk_counts_corrected.md` | 581 | `3f879ee83a0603bca5fee7c15436115c57104b2591ab1c0d98d6f7ffd3e8d5c0` | 不可变工作日志《Markdown document》；发现：人工整理表中的若干目录文件数按预期结构估算，与输出扫描manifest不一致。；更正：以 `records/generated/outputs/*.summary.json` 的 `file_count` 为唯一依据，更新 `results=5`、`v72=16`、`v7_4=55`、`v7_3=33`、`v8_audit=30`、两个hybrid variant各8、legacy hybrid=6。 |
| `records/log_entries/20260802_m0_crosswalk_risks_gates.md` | 674 | `04c381dae8c8feda48ca1edc657bf53df90ca5456a14bf2ea2eebfa95952421c` | 不可变工作日志《Markdown document》；写入：`M0_CODE_OUTPUT_CROSSWALK.md`、`M0_RISK_REGISTER.md`、`PROJECT_WIDE_SCAN_FINDINGS.md`、`HUMAN_DECISION_GATES.md`。；流程：用已验证manifest与逐脚本审计结果建立代码—输入—输出关系；将确定性错误、泄漏、代理目标、部署契约和论文表述风险分级。 |
| `records/log_entries/20260802_m0_dash_filename_corrected.md` | 539 | `6145447b8686d63497ad36df232881eab3483d3d2d01ab4e6942f2af0abc94a9` | 不可变工作日志《Markdown document》；发现：三处审计引用写成 `dash_denoiser_utils.py`，实际根文件为 `ppg_denoiser_dash_utils.py`。；更正：使用只允许每目标命中一次的 `correct_m0_dash_filename_reference.py`，精确修改 crosswalk、逐脚本图册与图覆盖校验器。 |
| `records/log_entries/20260802_m0_final_verification.md` | 531 | `86eb55ba3d89a32519213ed077eebec51c418bd2f3ea72eae1bea7f124aae3cc` | 不可变工作日志《Markdown document》；扫描证据：`pass`，失败0；baseline + 7输入 + 17输出共25笔事务有效。；算法图：`pass`，10份图文档、67个Mermaid图块；52/52代码/Notebook路径均有逐脚本图，缺失0。 |
| `records/log_entries/20260802_m0_full_scan_verified.md` | 946 | `ca9746b9aece46dc34284628e5f790c3d4c536d5dd8821b2c2aa2009328667e5` | 不可变工作日志《Markdown document》；状态 / Status：`reporting_in_progress`；代码 / Code：52 个代码文件逐字节完整读取，均保留 SHA-256 或明确错误记录；错误数 0。 |
| `records/log_entries/20260802_m0_method_registry.md` | 496 | `2c3b09498bd01b16a8f64dfc5c75203c5166c578e13b09a0d20cbe6376ed80e5` | 不可变工作日志《Markdown document》；状态 / Status：`complete`；新增 / Added：`records/M0_METHOD_REGISTRY.md`。 |
| `records/log_entries/20260802_m0_reports_batch1.md` | 559 | `f24695e14f5cb303728861864c6355b37ddce833b1128419a0f62427e638fb93` | 不可变工作日志《Markdown document》；状态 / Status：`complete`；新增 / Added：`records/M0_EXECUTIVE_REPORT.md`、`records/M0_PAPER_EVIDENCE.md`。 |
| `records/log_entries/20260802_project_and_non_m0_root_diagrams.md` | 583 | `d7bdf67d6c5b6a1bfc7ed3cb740cc67dcd98848947414fd656bc1a78ad9563cf` | 不可变工作日志《Markdown document》；写入：`01_PROJECT_END_TO_END_PIPELINE.md`与`baseline/01_NON_M0_ROOT_SCRIPT_ATLAS.md`。；覆盖：当前数据→信号处理→motion/heartbeat→Frailty3/ASA/SVM→评价/论文的总流；8个非M0 Python入口和5个Notebook逐一图示。 |
| `records/log_entries/20260802_root_file_io_inventory.md` | 521 | `fc0e01269392edff9b304f1157d6519250e5de7ae889209fbf4e4190921c2485` | 不可变工作日志《Markdown document》；覆盖：16个M0代码入口、8个非M0主脚本、5个Notebook、16个配置/文本/二进制/来源附件。；依据：逐字节根文件manifest、全代码manifest、静态路径引用、逐脚本算法审计和实际输出文本。 |
| `records/log_entries/20260802_scan_and_delivery_preverification.md` | 725 | `ed21ac7f173182d804dde7c9b552943011f35f7fb575de8171d7d8e85c0e3863` | 不可变工作日志《Markdown document》；扫描重验：`verify_scan_evidence.py` 返回 `status=pass`、失败0；源代码SHA、输入头部、输出文本EOF和25笔事务证据未发现漂移。；首次交付预验收：仅发现 `verify_scan_evidence.py` 缺少独立中英文行内注释；模块双语docstring本身有效。 |
| `records/log_entries/20260802_scan_verifier_added.md` | 539 | `efc3bc686436b6c35618fecf986f1d7516ac0e2ddb303de1d048d9b14db0f307` | 不可变工作日志《扫描证据校验器建立 / Scan-evidence verifier added》；日期 / Date：2026-08-02；状态 / Status：`implemented_unverified` |
| `records/log_entries/20260802_scanner_added.md` | 751 | `32ca92a92f7553106adfa6ac8bb8aeeeb0578a14544708e1549a273957fd2d13` | 不可变工作日志《分段扫描工具建立 / Sectioned scanner added》；日期 / Date：2026-08-02；状态 / Status：`implemented_unverified` |
| `records/log_entries/20260802_session_baseline.md` | 787 | `f7ee59edffdf82d2ac0ad8f448fcb7033521f6c9ac154165ec4de2004f9140c2` | 不可变工作日志《Markdown document》；状态 / Status：`complete`；来源 / Source：用户指令、`AGENTS.md`、`_agent/WRITE_RULES.md`、`_agent/TODO.md`、只读命令结果。 |
| `records/log_entries/20260803_activity_motion_supervision_and_history.md` | 1349 | `c31818d832a11075ddc2739c325c27b4557c11594a18c96a3179cc6975e9b451` | 不可变工作日志《Markdown document》；操作 / Action：把用户确认的 B/R 静态、S/W 动态监督语义写入 M0，并追溯早期多类模型、结果与混淆矩阵。；数据核验 / Data audit：逐字节读取29人261份CSV；确认两个数据目录、统一8列结构、每角色29份、角色持续时间与全部活动后恢复顺序。 |
| `records/log_entries/20260803_m0_candidate_future_direction_heading.md` | 764 | `3272cbb1b6c733de56dfdd416ccd7acd6f600ae4cc03f14f127a8ea91fe30eb5` | 不可变工作日志《Markdown document》；操作 / Action：根据最终要求矩阵自审，将候选脚本文档末章标题由 [路线选择建议] 规范为 [路线选择与未来方向建议]。；范围 / Scope：仅修改 `M0_history_MA_denoising_detector_HR_feature/02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md` 的标题；既有算法、路径、结果、状态与建议正文不变。 |
| `records/log_entries/20260803_m0_candidate_routes_catalog.md` | 1076 | `650426d8adc3e95e30186f005407d60bb1ffd0a50d1fd699f4ef07cfe559c903` | 不可变工作日志《Markdown document》；操作 / Action：新增用户指定的 motion detector、denoising、动态 HR 候选脚本文档。；写入 / Written：`M0_history_MA_denoising_detector_HR_feature/02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md`。 |
| `records/log_entries/20260803_m0_evidence_provenance_index.md` | 1014 | `20a7a418e36576111d741c781e324a04263de9c0b9a3fee2daa60ca165ed5a19` | 不可变工作日志《Markdown document》；操作 / Action：新增 M0 源码、输入、输出、机器验证和快照来源链索引。；写入 / Written：`M0_history_MA_denoising_detector_HR_feature/05_EVIDENCE_INDEX_AND_PROVENANCE.md`。 |
| `records/log_entries/20260803_m0_five_family_diagrams.md` | 774 | `843e1e9f06bd5c7312d7f3798ad2d784fd389d1b6bd40a54a3d66be54d808a68` | 不可变工作日志《Markdown document》；操作 / Action：在专业算法图目录新增五类方法、三个问题和统一 benchmark 的 Mermaid 图。；写入 / Written：`algorithm_diagrams/m0/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md`。 |
| `records/log_entries/20260803_m0_five_method_families_audit.md` | 1356 | `39851e9d3e11bfc5891c3abc2b96af89bbf576a8a1194d5e7d04cdfdd70fa39a` | 不可变工作日志《Markdown document》；操作 / Action：新增五类方法的代码、理论、应用、测试、缺口与实现可行性总审计。；写入 / Written：`M0_history_MA_denoising_detector_HR_feature/03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md`。 |
| `records/log_entries/20260803_m0_history_package_core_results.md` | 1159 | `8025ef11cac2556c13eec456871db2eaee21acd0dffbdd1b74b6b869a37f612d` | 不可变工作日志《Markdown document》；操作 / Action：在 `final_v0/M0_history_MA_denoising_detector_HR_feature/` 建立归档入口和完整 M0 结果/决定文档。；写入 / Written：`README.md`、`01_M0_COMPLETE_RESULTS_AND_DECISIONS.md`。 |
| `records/log_entries/20260803_m0_history_package_snapshots.md` | 984 | `186e90f2515c868b3f5fca679deb1fc591f844394d1a87b64de7e197464bd414` | 不可变工作日志《Markdown document》；操作 / Action：将 M0 完整历史结论、三类候选路线、五类方法审计、统一测试合同、算法图与关键机器证据组织为独立专题归档。；写入 / Written：`M0_history_MA_denoising_detector_HR_feature/` 下的 `snapshots/`、`M0_SOURCE_SNAPSHOT_MANIFEST.json`、`M0_PACKAGE_VERIFICATION.json` 与 `06_M0_PACKAGE_TREE.md`。 |
| `records/log_entries/20260803_m0_unified_benchmark_contract.md` | 1219 | `020a54bb18697b3ce8066bed691ba8d4f2f43aa48297eb42e63739011055b96e` | 不可变工作日志《Markdown document》；操作 / Action：把五类方法的“可实现”要求固化为公共数据合同、接口、测试先决条件、指标、输出 schema 和验收门。；写入 / Written：`M0_history_MA_denoising_detector_HR_feature/04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md`。 |
| `records/log_entries/20260803_madenoiser_confirmed_route.md` | 1134 | `55deb937b5c46ce3c622e96276eec1ec2913100921891b358d8b04862c7a4ea0` | 不可变工作日志《Markdown document》；操作 / Action：把用户确认的 SQI-v2、Motion-29、四条 MA/HR/PPI 路线、PTT 监督 benchmark 和 Frailty feature/CV 选择规则固化为可执行合同。；新增 / Added：专题 `07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md`、M0 算法图 07 和不可变决策片段 `M0-MAD-001`。 |
| `records/log_entries/20260814_m1_architecture_contract.md` | 1001 | `fb2cd773f76bfcf56e60e68585ddb82ca4cc6fcf890897d65b6c482dabc1bd88` | 不可变工作日志《Markdown document》；操作 / Action：定义 SignalBatch→PipelineResult 模块顺序、机器 schema、可替换 registries、训练/推理隔离和三档中心处理平台。；用户确认 / User-confirmed：血压仪大小中心屏显处理设备；可穿戴 PPG+IMU；允许 NumPy/SciPy/ONNX Runtime/scikit-learn；需要高性能和性价比方案。 |
| `records/log_entries/20260814_m1_architecture_contract_v2.md` | 784 | `cb62eacb3745a8915a15f7e12ace72ec2411c74c0ebe6c8be7cb610beaf68873` | 不可变工作日志《Markdown document》；操作 / Action：以追加式 V2 固化有界流式、窗口坐标/coverage、单一 action owner、完整 artifact hash 与 accelerator→CPU fallback。；原因 / Reason：现有文件补丁入口受沙箱读取故障；为避免绕过 `apply_patch` 或静默覆盖，保留 V1 历史并新建权威 V2。 |
| `records/log_entries/20260814_m1_architecture_contract_v2_verification_correction.md` | 642 | `ab38e94d90d1a043c6a305a43691d4075d5e6d00205f139ea897b11c46ce4741` | 不可变工作日志《Markdown document》；修正 / Correction：较早日志中的“JSON Schema 校验”仅指 schema 结构和 registry/config 交叉校验；本机无第三方 Draft 2020-12 引擎，因此该完整项未运行。；补充 / Added：新增零第三方依赖语义验证器，覆盖 ok/no-result 状态机、概率和、唯一 action owner、CPU fallback、locked artifacts、threshold 与 bundle path containment。 |
| `records/log_entries/20260815_m1_quality_routing_v3.md` | 1207 | `f1fae596fd151993071104acdad9155b06107fbfbd9800d96744ce88a455cf6e` | 不可变工作日志《Markdown document》；操作 / Action：按用户修正，以追加式 V3 取代 V1/V2 的 SQI/coarse-denoise action-owner 路由；V2 输入、流式、bundle、平台和 provider fallback 继续有效。；算法 / Algorithm：必做 SQI + 可选 Motion → join；high/non-motion 绕过 denoiser；low 或 motion 按 run/session 级手动配置互斥执行 drop 或 denoise→FeatureBlock；invalid/unrecoverable 强制 drop，module failure fail-closed。 |
| `records/log_entries/20260815_m2_manifest_dual_fold_protocol.md` | 1034 | `fad683ea98aa4d0ba62ffc9071a3b8d47b28f02bbd30357331dc4bbd240a1f9b` | 不可变工作日志《Markdown document》；操作 / Action：只读审计 Frailty3 和五类外部数据源；在 `final_v0/` 新增 M2 数据/阶段/协议包、生成器、验证器入口、机器 schemas、双注册表图和溯源合同。；算法 / Algorithm：完整字节/数值扫描 → file/subject manifests；保留 sklearn 1.4.2 历史 SGKF defect membership；同步置换 group 与 class-count rows 生成 corrected SGKF future membership；以固定 5×5、fixed epoch/no early stopping 输出 OOF。 |
| `records/log_entries/20260815_m3_contract_edge_tests_phase7.md` | 694 | `e14160a47481c3ef32fc09e3dc66426745fdd3401133d65883f358bf731e23c1` | 不可变工作日志《M3 contract edge tests phase 7 / M3 合同边界测试第 7 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_execution |
| `records/log_entries/20260815_m3_core_evidence_builder_phase8.md` | 624 | `a6f59b9d5dd172d9b481393419e91bf3810b9dc0cd6c68992c86c441374a6d91` | 不可变工作日志《M3 core evidence builder phase 8 / M3 核心证据构建第 8 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_execution |
| `records/log_entries/20260815_m3_core_phase1.md` | 654 | `a6919fe43dd9ce641c7d3a5fdff18dde9fbf99bcc4c3f2af3aea1740096a4c37` | 不可变工作日志《2026-08-15 M3 公共预处理核心第一阶段》；范围：仅在 `final_v0/M3_unified_preprocessing_and_signal_algorithms/` 新建公共合同、；异常门控、PPG 滤波、fold-only scaling、profile registry 和 schema。 |
| `records/log_entries/20260815_m3_d8_symmetric_scorecard_phase21.md` | 911 | `4f15aaed8e59ea576a1531f753cf8d717755fcb251911a24c3a792a33b44f1c9` | 不可变工作日志《M3 D8 symmetric scorecard phase 21 / M3 D8 对称评价第 21 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_d8_training_split_identity_phase24.md` | 1128 | `8f6136d93602833382d7db770453e6cf4f8ea2fc1b1b425203807718529e2037` | 不可变工作日志《M3 D8 training-split identity phase 24 / M3 D8 训练分割身份第 24 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_decision_contract_phase16.md` | 769 | `424e7c2b6280340fc634ada8f010e7aba2502bba19d571ddcbd20d742c8d47d2` | 不可变工作日志《M3 decision contract phase 16 / M3 决策合同第 16 阶段》；时间 / Date：2026-08-15；状态 / Status：user_decisions_recorded |
| `records/log_entries/20260815_m3_deprecated_profile_fail_closed_phase29.md` | 939 | `97872a899a1a12b22e1f90a4f5376ab3dc9fd576b45c29e2f06d7f0e79f517e8` | 不可变工作日志《M3 deprecated-profile fail-closed phase 29 / M3 弃用 Profile 关闭失败第 29 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_reference_test |
| `records/log_entries/20260815_m3_deprecated_profile_tests_phase30.md` | 881 | `088332d1aeb64f46bb6bd0fc7911bdb7efe620e3b14abc969819bc05a47693aa` | 不可变工作日志《M3 deprecated-profile tests phase 30 / M3 弃用 Profile 测试第 30 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_evidence_authority_phase17.md` | 860 | `8333fc86d675cb9805f256ff371a73fd6153d547c16a39bbc2fbba282310a86f` | 不可变工作日志《M3 evidence authority phase 17 / M3 证据权威边界第 17 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_evidence_rebuilt |
| `records/log_entries/20260815_m3_fixture_manifest_contract_phase25.md` | 977 | `2bd435af900410a3d58152354b770d544475ad443d38d83501bd7344fc648f7e` | 不可变工作日志《M3 fixture manifest contract phase 25 / M3 Fixture 清单合同第 25 阶段》；时间 / Date：2026-08-15；状态 / Status：generator_strengthened_pending_regeneration |
| `records/log_entries/20260815_m3_fixture_manifest_regeneration_phase26.md` | 1139 | `6e89c5cda2273e6392b0b9b678aac6959eeab624d95b5c1104f127a5bd059a28` | 不可变工作日志《M3 fixture manifest regeneration phase 26 / M3 Fixture 清单重建第 26 阶段》；时间 / Date：2026-08-15；状态 / Status：regenerated_and_integrity_tested |
| `records/log_entries/20260815_m3_fold_and_reference_tests_phase12.md` | 618 | `a88d5da9544001bb404d2b4efeaf4b02f15d77b180ea8e98789cd7691d0f9040` | 不可变工作日志《M3 fold/reference tests phase 12 / M3 训练折与参考评价测试第 12 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_execution |
| `records/log_entries/20260815_m3_fold_artifact_envelope_phase22.md` | 922 | `fc408a1c4f43ce7b024a9114346cdfffcbbbecf934943dc553e5e7bedf5b5375` | 不可变工作日志《M3 fold artifact envelope phase 22 / M3 训练折 artifact 完整封装第 22 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_fold_registry_field_correction.md` | 503 | `264c0adfc501fc308aa38d8bcf83c6b19ef411e703588e785ad37ec38bca51d0` | 不可变工作日志《M3 fold registry field correction / M3 fold registry 字段修正》；时间 / Date：2026-08-15；状态 / Status：corrected_pending_retest |
| `records/log_entries/20260815_m3_fold_robust_scaling_phase20.md` | 816 | `62c17a6c5bcc81860ded388cd21cce523316a53dc787721ec898376242751bf9` | 不可变工作日志《M3 future fold scaling phase 20 / M3 未来训练折缩放第 20 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_fold_schema_alignment_phase23.md` | 897 | `07e8abe610107ec90df014378c0018707fb4af66aad71e3b6091a18afb3097d9` | 不可变工作日志《M3 fold schema alignment phase 23 / M3 训练折 Schema 对齐第 23 阶段》；时间 / Date：2026-08-15；状态 / Status：runtime_aligned_pending_schema_regeneration |
| `records/log_entries/20260815_m3_frailty_imu_proxy_builder_phase9.md` | 636 | `4983ba1102d4453cb8ea6f6b74bd19bb6fa6b8ec0a0014f45589d75b367f2429` | 不可变工作日志《M3 Frailty3 IMU proxy builder phase 9 / M3 Frailty3 IMU 代理构建第 9 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_execution |
| `records/log_entries/20260815_m3_historical_discovery_phase19.md` | 948 | `3b06c1682190a00fb046ec5628c41df3e0d2aeb474b4fc96f82d466a7610641f` | 不可变工作日志《M3 historical preprocessing discovery phase 19 / M3 历史预处理全量发现第 19 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_evidence_rebuilt |
| `records/log_entries/20260815_m3_imu_core_phase2a.md` | 615 | `89152e520e8ccfafe015d366ba0c81cfacffce4d0fbaa36a49345aa102532b7b` | 不可变工作日志《2026-08-15 M3 无预校准 ESKF 与 LPF 对照实现》；新增 quaternion multiplicative error-state Kalman filter 主路线。；新增共享单位、质量门、20/40 Hz 前端和 jerk 的 0.3 Hz LPF 重力对照。 |
| `records/log_entries/20260815_m3_legacy_peak_parity_phase15.md` | 1039 | `19bfe2f4614d3c95121de01428eaaf453b627e6fabb39af160d30e13f2ec04ff` | 不可变工作日志《M3 legacy peak parity phase 15 / M3 历史峰算法一致性第 15 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_evidence_persisted |
| `records/log_entries/20260815_m3_m2_fold_artifact_binding_phase10.md` | 666 | `4871a3c4b7517e38650fb07ac8d560705d5557fde235855a45d02b402e76a4ff` | 不可变工作日志《M3 M2 fold-artifact binding phase 10 / M3–M2 训练折 artifact 绑定第 10 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_tests |
| `records/log_entries/20260815_m3_physiology_core_phase2b.md` | 667 | `f03ee2019a4f93dc706eeb4a04e57567c42eeb6b2a3b4e222e5fd45b2bb15d8f` | 不可变工作日志《2026-08-15 M3 Peak、PPI、HR 与 PRV 公共实现》；新增 corrected_v1 双极性 peak detector，固定 10 秒窗口、5 秒 hop 和 0.15 秒事件合并。；PPI 固定为 0.30–2.00 秒；无效 PPI 不删除源峰，raw/valid/corrected NNI 分列。 |
| `records/log_entries/20260815_m3_physiology_provenance_phase13.md` | 805 | `c21e46fd7e74501fd6ee0d854b1261a11179d245eb19895a3cd1a68bccd81c33` | 不可变工作日志《M3 physiology provenance phase 13 / M3 生理结果溯源第 13 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_ppg_raw_repaired_views_phase18.md` | 835 | `bdeb20ae5760c29b222ee8faa19c2618f6a06006fdf85684c536a28db5e9a168` | 不可变工作日志《M3 PPG source/repaired views phase 18 / M3 PPG 原始与修复视图第 18 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_profile_and_physiology_corrections_phase5.md` | 975 | `47b118031f512ca015faefc9dda72aa2460eba0effd3350206f4d9464b22fec9` | 不可变工作日志《M3 profile/physiology corrections phase 5 / M3 profile 与生理算法修正第 5 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_reference_tests |
| `records/log_entries/20260815_m3_profile_locked_peak_and_resampling_phase27.md` | 1258 | `7afcecff07432f3564cae69376e298af02a8dd56bc5603c5c10a7b128638f16a` | 不可变工作日志《M3 profile-locked peak and resampling phase 27 / M3 Profile 锁定峰检测与重采样第 27 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_reference_test |
| `records/log_entries/20260815_m3_profile_locked_peak_and_resampling_tests_phase28.md` | 1121 | `b0d30f601971808b32340eb1fc6055df586cdba49b3cef6333de8585184eae1e` | 不可变工作日志《M3 profile-locked peak and resampling tests phase 28 / M3 Profile 锁定峰检测与重采样测试第 28 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_ptt_ecg_reference_evaluator_phase11.md` | 687 | `154627bc67e1379f305fe3c5916052820e4c802cca0826a141f9817da6613ef2` | 不可变工作日志《M3 PTT ECG reference evaluator phase 11 / M3 PTT ECG 参考评价器第 11 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_tests |
| `records/log_entries/20260815_m3_reference_report_snapshot_phase14.md` | 837 | `45975da6953d8e9114bc131fa8bca4c19b2488d33db33b6c923d42ff3271ad22` | 不可变工作日志《M3 reference report snapshot phase 14 / M3 测试报告快照第 14 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_reference_test_corrections.md` | 572 | `026c2551d79533fd12823002c9fe5be3300149229e1d4006cc90d43d2e2a0069` | 不可变工作日志《2026-08-15 M3 Reference Test 首轮修正》；首轮结果：22 项中 20 通过、1 failure、1 error。；failure 原因：原 fixture 长度使 100 点 gap 占总样本 2.5%，正确触发了 >1% fatal； |
| `records/log_entries/20260815_m3_reference_tests_phase3.md` | 597 | `34494dab156b716b547278f261c3533bd343892ed81bba8878c64c7416f19c38` | 不可变工作日志《2026-08-15 M3 固定 Fixtures 与 Reference Tests》；新增固定 seed 20260815 的 PPG/IMU 合成真值生成器，使用稳定 NPY 和 SHA manifest。；新增异常 gap/flatline、PPG 频响、重采样、fold-only scaler 泄漏哨兵测试。 |
| `records/log_entries/20260815_m3_sim_256_resampling_fixture_phase31.md` | 804 | `922418e852c49bc6de8f001ad404ef5f71715a41ee1838bdf9325547aed05dfd` | 不可变工作日志《M3 Sim 256 Hz resampling fixture phase 31 / M3 Sim 256 Hz 重采样 Fixture 第 31 阶段》；时间 / Date：2026-08-15；状态 / Status：test_added_pending_full_run |
| `records/log_entries/20260815_m3_sim_256_resampling_tests_phase32.md` | 827 | `20dc024fd665ca29a1c00bd9b042f1b46f42b78e18f30a9d9d41bdbe9304dd72` | 不可变工作日志《M3 Sim 256 Hz resampling tests phase 32 / M3 Sim 256 Hz 重采样测试第 32 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_and_reference_tested |
| `records/log_entries/20260815_m3_stateful_imu_runtime_corrections_phase6.md` | 828 | `ec6edf676d4b87e5c40b35e8662a203fa3c9806129e42f7e0f999bab8ddc6c41` | 不可变工作日志《M3 stateful IMU runtime corrections phase 6 / M3 有状态 IMU runtime 修正第 6 阶段》；时间 / Date：2026-08-15；状态 / Status：implemented_pending_reference_tests |
| `records/pending_agent_updates/BASELINE_AND_M0_FINAL_TOPICS.md` | 774 | `d881c11d3ad03a19446098634e3c297dda6a33bd4776536c00e5b949099233fb` | 待用户要求后处理的 `_agent` 候选《Baseline + M0 待录入主题（内部草稿）》；状态：`draft; do_not_write_without_user_request`；候选目标：`_agent/MODULES.md`、`PROJECT_STRUCTURE.md`、`NOTES.md`、`PROJECT_HANDOFF.md`、`TODO.md`、`CHANGELOG.md`。 |
| `records/pending_agent_updates/M0_ACTIVITY_MOTION_SUPERVISION_DRAFT.md` | 2084 | `38a92186d7fae461d41267d41155d95a356b09f06968fa8b9a69493fc805dd2b` | 待用户要求后处理的 `_agent` 候选《M0 Activity/Motion supervision private draft》；Decision `M0-MOT-001` confirms the 29-subject target as activity/motion state.；Map B and R1–R4 to static; map S1–S2 stand-and-sit and W1–W2 walking to motion. |
| `records/pending_agent_updates/M0_DRAFT_TOPICS.md` | 838 | `c3966fc57a3ab558b9edbc2c35e80a2bf3d182bf6edd8ce90841b4b2cf466ba3` | 待用户要求后处理的 `_agent` 候选《Markdown document》；状态 / Status：`draft`；来源 / Source：完整代码读取、输入头部扫描、输出文本 EOF 扫描、历史记录及结果交叉核验。 |
| `records/pending_agent_updates/M0_FIVE_FAMILY_EXTENSION_DRAFT.md` | 742 | `b23c2a85ddbad9b289855d3a8ff402c4857ea24aead2999068c53f30b39f029e` | 待用户要求后处理的 `_agent` 候选《Markdown document》；状态：`draft_not_for_agent_write`；来源：2026-08-03 本地代码/输出扩展审计 |
| `records/pending_agent_updates/M0_MADENOISER_CONFIRMED_ROUTE_DRAFT.md` | 2415 | `cbea42b52ca1b835e926e154e9b4d6a3fdc8a3069c276bedc5390bfbbc57fa01` | 待用户要求后处理的 `_agent` 候选《M0 MAdenoiser confirmed-route private draft》；Decision ID: M0-MAD-001；Status: confirmed_route_implementation_not_started |
| `records/pending_agent_updates/M1_ARCHITECTURE_CONTRACT_DRAFT.md` | 694 | `4df247f5de40124d17ba108ac1ea349dd5fbf37b0114e8c1ada2218f2e3ef207` | 待用户要求后处理的 `_agent` 候选《待录入 `_agent` 草稿：M1 架构与移动平台合同》；建议目标文档：`TODO.md`、`MODULES.md`、`ROADMAP.md`、`docs/decision-log.md`、`CHANGELOG.md`、`PROJECT_STRUCTURE.md`。；待记录主题：M1 端到端模块顺序、SignalBatch/PipelineResult 合同、SQI诊断与质量动作互斥、训练/部署隔离、允许依赖、三档中心平台、provisional资源预算、M2/M3/M4/M9后续门。 |
| `records/pending_agent_updates/M1_ARCHITECTURE_CONTRACT_V2_DRAFT.md` | 368 | `4393f79d14b3d3f81dadf3d50307de0db0df129d14655620a25146df09421514` | 待用户要求后处理的 `_agent` 候选《私有待录入草稿：M1 V2》；仅供未来用户明确要求草拟 `_agent` 更新时使用；当前不得写入 `_agent`，最终报告也不展示正文。；待录入主题：M1 V2 有界流式合同、preprocessing execution mode、窗口坐标/coverage、单一 action owner、完整 artifact hashes、CPU fallback、当前未实现/未 benchmark 边界。 |
| `records/pending_agent_updates/M1_QUALITY_ROUTING_V3_DRAFT.md` | 1448 | `435c40f16553b17ddd58affe2e963f8efd205ff8680449e2b21068978a94eada` | 待用户要求后处理的 `_agent` 候选《私有待审草稿：M1 V3 质量路由 / Private pending draft》；本文件只保存未来可能写入 `_agent` 的候选内容。；未收到用户明确要求时，不展示本文件内容，不写入 `_agent`。 |
| `records/pending_agent_updates/M2_DATA_MANIFEST_PROTOCOL_DRAFT.md` | 704 | `10342d6e7552e1e9ea9fcd8bf8f0ceb91bcc098f39021b37e61c73d279f4f695` | 待用户要求后处理的 `_agent` 候选《Markdown document》；状态：`draft_not_authorized_for_agent_write`；目标文档候选：`MODULES.md`、`TODO.md`、`PROJECT_STRUCTURE.md`、`docs/decision-log.md`、`CHANGELOG.md`、`NOTES.md` |
| `records/pending_agent_updates/M3_UNIFIED_PREPROCESSING_DRAFT.md` | 894 | `56d9ee74ba8a04b3fe5aa906aa8e801f56e3bf522ebdccc105549c10b7510a8e` | 待用户要求后处理的 `_agent` 候选《M3 统一预处理与信号算法待录入主题》；状态：draft；仅在用户要求草拟 `_agent` 更新时整理，不直接写入 `_agent`。；候选主题：M3 冻结的 400 Hz profiles、corrected/legacy 边界、无预校准 EKF 主路线、 |
| `tools/add_bilingual_inline_comments_to_scan_verifier.py` | 1912 | `bd53d372f2cef306fc1ccc713ea93cbed6a7a3e3b82f836af1466b6aa7a0a769` | 带中英文说明的 final_v0 工具；Add one audited bilingual inline-comment block to the scan verifier.；主要入口：main |
| `tools/build_m0_history_package.py` | 22564 | `17b8cb9d5f8bf91877689dd022e418e710f00c26c2992cc29548a9eca033b5fb` | 带中英文说明的 final_v0 工具；构建并验证 M0 历史归档包；build and verify the M0 history package.；主要入口：sha256_bytes, relative_to_checked, write_bytes_atomic, write_text_atomic, write_json_atomic, snapshot_one, build_snapshot_manifest, load_manifest |
| `tools/build_m0_history_package_v3.py` | 15663 | `73819c76ad6f7a7f0a443f3fec8eb9be7bdff9f1c7cb11de58e7375e8b81ba64` | 带中英文说明的 final_v0 工具；追加构建并验证 M0 v3 归档；additively build and verify the M0 v3 archive.；主要入口：sha256_bytes, relative_to_checked, write_bytes_atomic, write_text_atomic, write_json_atomic, load_base_specs, snapshot_one, build_manifest |
| `tools/correct_archived_inventory_details.py` | 2636 | `0187f30188c036bf70fd6a9d8ff856bc7a68ea7a6b458f5241e95183dbf71d89` | 带中英文说明的 final_v0 工具；Apply four evidence-backed corrections to the archived-code inventory.；主要入口：main |
| `tools/correct_m0_crosswalk_manifest_counts.py` | 2619 | `4472e3054e24fde85da29e845af8e085a3039c511af481dc5ae612bfc0a314ff` | 带中英文说明的 final_v0 工具；Correct M0 crosswalk file counts from verified output manifests.；主要入口：main |
| `tools/correct_m0_dash_filename_reference.py` | 1902 | `50c82d9a0dedef2524a75f541a0f917729d28caaa66d2cb634a83c9638646ef1` | 带中英文说明的 final_v0 工具；Apply one audited filename-reference correction inside final_v0.；主要入口：main |
| `tools/sync_algorithm_index.py` | 4102 | `f23e7173f1ef0f1a3852d5508e9fc4267bbec33ef4eae86c98e9b1ab222f38db` | 带中英文说明的 final_v0 工具；Rebuild the algorithm-diagram index from Markdown sources.；主要入口：sha256_bytes, extract_title_and_summary, main |
| `tools/sync_tracking_docs.py` | 2893 | `1ba89e07f7c69f01bb4cca0640c82d5a8061ce496ab435c494b878dd4ae277b5` | 带中英文说明的 final_v0 工具；从不可变片段重建 final_v0 的两份追踪主文档。；主要入口：read_fragments, write_document, main |
| `tools/update_final_v0_index.py` | 6294 | `633c09f6001cb8632b7bbf8c0622786b657a4cbb1eb8b74cbca12022c05e89ac` | 带中英文说明的 final_v0 工具；更新 ``final_v0`` 文件树索引。；主要入口：sha256_file, generic_description, build_tree, render_index, main |
| `tools/update_final_v0_index_detailed.py` | 10700 | `5d9170cb5a671d0b31d37c769dc9cff8d60dbd0ede6d3fa2b3e7525dd03b0213` | 带中英文说明的 final_v0 工具；Build a content-aware file tree and per-file description for final_v0.；主要入口：sha256_bytes, clean_table_text, first_markdown_content, describe_markdown, compact_json_fields, describe_json, describe_jsonl, describe_python |
| `tools/verify_algorithm_diagrams.py` | 5657 | `4b21ade13146e5f201b94ea4abc40467b1acc6ad9951040370fcd8057abb15a1` | 带中英文说明的 final_v0 工具；Verify Markdown/Mermaid diagram completeness and script coverage.；主要入口：digest, inspect_markdown, main |
| `tools/verify_code_diagram_coverage.py` | 4463 | `422992edbc35fe797a308745e6265a78dfa58e0899cb2d6332ffb1ca843bba10` | 带中英文说明的 final_v0 工具；Verify that every scanned code/Notebook path has a per-script diagram entry.；主要入口：read_manifest_paths, main |
| `tools/verify_final_v0_delivery.py` | 8065 | `51d9430761028fc717423c1242d3afc265380627f54097ca1cda370c3c66fbb3` | 带中英文说明的 final_v0 工具；Verify the final_v0 delivery boundary, documentation, and generated evidence.；主要入口：sha256_bytes, load_json, inspect_python, tree_indexed_paths, main |
| `tools/verify_scan_evidence.py` | 9495 | `7e32b9fa49a9eb41abc2aaada718b422b6ad149c4a742c5ba2db8f27c5011172` | 带中英文说明的 final_v0 工具；验证 workspace 扫描证据的完整性和内部一致性。；主要入口：read_json, read_jsonl, check, verify_baseline, verify_inputs, verify_outputs, verify_ledger, main |
| `tools/workspace_audit.py` | 29707 | `4c58dc30303e4b1ce1b39488f748de092b3da186f696d3fe04b960ac7d3dbd58` | 带中英文说明的 final_v0 工具；对只读项目执行可追溯扫描，并将全部证据写入 ``final_v0``。；主要入口：now_utc, rel, iter_files, iter_source_files, safe_top, sha256, decode, sanitize |
| `FINAL_V0_TREE.md` | self | intentionally omitted | 自动生成的本文件树、内容说明和完整性索引自身。 |

## 完整性与更新规则 / Integrity and update rules

- 永久文件总数（含本索引）：**745**。
- 每次逻辑写入后运行本工具；索引自身的更新不递归产生日志。
- 所有非索引文件必须同时具有字节数、SHA-256和内容感知说明；缺一项即视为未验证。
