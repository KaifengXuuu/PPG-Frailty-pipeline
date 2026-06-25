# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260427-01_smoke_detector_ab
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 30
- internal_holdout_subject_count: 8
- extra_holdout_subject_count: 14
- cv_folds: 2
- final_train_epochs: 1
- norm_type: instance
- balanced_sampling: True
- augmentation_enabled: True
- domain_adversarial_lambda: 1.0000
- worst_domain_weight: 0.2500
- group_scorecards: True
- delay_analysis: True
- lodo_validation: False
- detector_benchmark: True
- detector_epochs: 1
- detector_model_base_channels: 24
- gate_input: shared PPG encoder mid-level features
- gate_uses_imu: False
- mimic_subsets: ['mimic_perform_train_all_csv', 'mimic_perform_test_all_csv']
- mimic_max_records: 1
- include_mimic_special_extra_holdout: True
- mimic_extra_holdout_subsets: ['mimic_perform_af_csv', 'mimic_perform_non_af_csv']
- mimic_extra_holdout_mat_files: ['MIMIC_PERform_1_min_noisy.mat', 'MIMIC_PERform_1_min_neonate.mat']
- enable_vitaldb: False
- vitaldb_max_cases: 20
- ecg_detector_preflight: False
- ecg_preflight_peak_f1: NA
- ecg_preflight_ibi_accuracy: NA

## Thresholds
- peak_threshold: 0.7250
- gate_threshold: 0.4500
- ppg_main_event_tolerance_ms: 20
- ppg_layered_event_tolerances_ms: [10, 20, 30, 40]

## ECG Peak Detector Preflight
- status: disabled

## Cross-Validation Aggregate
### cross_validation
- windows: 10887
- peak_samples: 2787072
- rr_samples: 2787072
- gate_windows_labeled: 6249
- peak_sequence:
  n: 2787072
  threshold: 0.7250
  positive_rate: 0.1268
  accuracy: 0.8894
  balanced_accuracy: 0.7804
  precision: 0.5561
  recall: 0.6343
  f1: 0.5926
  confusion_matrix: [[2254560, 179009], [129259, 224244]]
  roc_auc: 0.8895
  pr_auc: 0.6207
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 32806
  fp: 28257
  fn: 39013
  precision: 0.5372
  recall: 0.4568
  f1: 0.4938
  timing_n: 32806
  timing_bias: -0.0008
  timing_mae: 0.0099
  timing_median_ae: 0.0156
  timing_std: 0.0124
  timing_variance: 0.0002
- hr_interval_sequence:
  n: 2787072
  mae: 0.2168
  rmse: 0.9354
  median_ae: 0.1394
  bias: 0.0192
  pearson_r: -0.0349
  r2: -0.0197
- rri_event_sequence:
  n: 24087
  bias: -0.0000
  mae: 0.0080
  median_ae: 0.0000
  std: 0.0124
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 6249
  threshold: 0.4500
  positive_rate: 0.6662
  accuracy: 0.6209
  balanced_accuracy: 0.5662
  precision: 0.7091
  recall: 0.7307
  f1: 0.7197
  confusion_matrix: [[838, 1248], [1121, 3042]]
  roc_auc: 0.6753
  pr_auc: 0.8109
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 32806 | 28257 | 39013 | 0.5372 | 0.4568 | 0.4938 | 0.0099 | 0.0156 |
  | 20ms | 32806 | 28257 | 39013 | 0.5372 | 0.4568 | 0.4938 | 0.0099 | 0.0156 |
  | 30ms | 45421 | 15642 | 26398 | 0.7438 | 0.6324 | 0.6836 | 0.0158 | 0.0156 |
  | 40ms | 49989 | 11074 | 21830 | 0.8186 | 0.6960 | 0.7524 | 0.0187 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 24087 | -0.0000 | 0.0080 | 0.0000 | 0.0124 | 0.0002 |
  | 20ms | 24087 | -0.0000 | 0.0080 | 0.0000 | 0.0124 | 0.0002 |
  | 30ms | 35510 | 0.0000 | 0.0108 | 0.0156 | 0.0162 | 0.0003 |
  | 40ms | 39791 | 0.0001 | 0.0127 | 0.0156 | 0.0190 | 0.0004 |

## Holdout
### holdout
- windows: 2782
- peak_samples: 712192
- rr_samples: 712192
- gate_windows_labeled: 1828
- peak_sequence:
  n: 712192
  threshold: 0.7250
  positive_rate: 0.1238
  accuracy: 0.8903
  balanced_accuracy: 0.7892
  precision: 0.5475
  recall: 0.6549
  f1: 0.5964
  confusion_matrix: [[576314, 47724], [30420, 57734]]
  roc_auc: 0.9091
  pr_auc: 0.6048
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 6965
  fp: 8816
  fn: 10945
  precision: 0.4414
  recall: 0.3889
  f1: 0.4135
  timing_n: 6965
  timing_bias: 0.0018
  timing_mae: 0.0105
  timing_median_ae: 0.0156
  timing_std: 0.0127
  timing_variance: 0.0002
- hr_interval_sequence:
  n: 712192
  mae: 0.1441
  rmse: 0.1889
  median_ae: 0.1110
  bias: -0.0097
  pearson_r: -0.1095
  r2: -0.4684
- rri_event_sequence:
  n: 4921
  bias: -0.0002
  mae: 0.0074
  median_ae: 0.0000
  std: 0.0119
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 1828
  threshold: 0.4500
  positive_rate: 0.6668
  accuracy: 0.6668
  balanced_accuracy: 0.5000
  precision: 0.6668
  recall: 1.0000
  f1: 0.8001
  confusion_matrix: [[0, 609], [0, 1219]]
  roc_auc: 0.9247
  pr_auc: 0.9593
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 6965 | 8816 | 10945 | 0.4414 | 0.3889 | 0.4135 | 0.0105 | 0.0156 |
  | 20ms | 6965 | 8816 | 10945 | 0.4414 | 0.3889 | 0.4135 | 0.0105 | 0.0156 |
  | 30ms | 11076 | 4705 | 6834 | 0.7019 | 0.6184 | 0.6575 | 0.0182 | 0.0156 |
  | 40ms | 13007 | 2774 | 4903 | 0.8242 | 0.7262 | 0.7721 | 0.0225 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 4921 | -0.0002 | 0.0074 | 0.0000 | 0.0119 | 0.0001 |
  | 20ms | 4921 | -0.0002 | 0.0074 | 0.0000 | 0.0119 | 0.0001 |
  | 30ms | 8529 | -0.0005 | 0.0097 | 0.0000 | 0.0150 | 0.0002 |
  | 40ms | 10326 | -0.0007 | 0.0116 | 0.0156 | 0.0178 | 0.0003 |

## Extra Holdout
### extra_holdout
- windows: 6710
- peak_samples: 1717760
- rr_samples: 1717760
- gate_windows_labeled: 6052
- peak_sequence:
  n: 1717760
  threshold: 0.7250
  positive_rate: 0.1286
  accuracy: 0.8006
  balanced_accuracy: 0.5192
  precision: 0.1689
  recall: 0.1404
  f1: 0.1533
  confusion_matrix: [[1344285, 152553], [189912, 31010]]
  roc_auc: 0.6185
  pr_auc: 0.1638
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 1570
  fp: 26471
  fn: 43305
  precision: 0.0560
  recall: 0.0350
  f1: 0.0431
  timing_n: 1570
  timing_bias: -0.0088
  timing_mae: 0.0119
  timing_median_ae: 0.0156
  timing_std: 0.0105
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 1717760
  mae: 0.1752
  rmse: 0.2201
  median_ae: 0.1458
  bias: 0.0757
  pearson_r: -0.1121
  r2: -1.0542
- rri_event_sequence:
  n: 697
  bias: 0.0004
  mae: 0.0061
  median_ae: 0.0000
  std: 0.0107
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 6052
  threshold: 0.4500
  positive_rate: 0.5423
  accuracy: 0.5423
  balanced_accuracy: 0.5000
  precision: 0.5423
  recall: 1.0000
  f1: 0.7032
  confusion_matrix: [[0, 2770], [0, 3282]]
  roc_auc: 0.2656
  pr_auc: 0.4106
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 1570 | 26471 | 43305 | 0.0560 | 0.0350 | 0.0431 | 0.0119 | 0.0156 |
  | 20ms | 1570 | 26471 | 43305 | 0.0560 | 0.0350 | 0.0431 | 0.0119 | 0.0156 |
  | 30ms | 3967 | 24074 | 40908 | 0.1415 | 0.0884 | 0.1088 | 0.0236 | 0.0312 |
  | 40ms | 7103 | 20938 | 37772 | 0.2533 | 0.1583 | 0.1948 | 0.0339 | 0.0312 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 697 | 0.0004 | 0.0061 | 0.0000 | 0.0107 | 0.0001 |
  | 20ms | 697 | 0.0004 | 0.0061 | 0.0000 | 0.0107 | 0.0001 |
  | 30ms | 2159 | 0.0007 | 0.0088 | 0.0000 | 0.0147 | 0.0002 |
  | 40ms | 4414 | 0.0009 | 0.0107 | 0.0156 | 0.0174 | 0.0003 |

## Gate Diagnostics
### cross_validation gate diagnostics
- true_motion_rate: 0.6662
- pred_motion_rate: 0.6865
- near_threshold_frac_0p05: 1.0000
- threshold_percentile: 0.3135
- rest_score_mean: 0.4616
- motion_score_mean: 0.4686
- class_score_gap: 0.0070
- class_separation_cohen_d: 0.3205
- rest_false_motion_rate: 0.5983
- motion_false_rest_rate: 0.2693
### holdout gate diagnostics
- true_motion_rate: 0.6668
- pred_motion_rate: 1.0000
- near_threshold_frac_0p05: 0.0000
- threshold_percentile: 0.0000
- rest_score_mean: 0.5641
- motion_score_mean: 0.5725
- class_score_gap: 0.0084
- class_separation_cohen_d: 1.9632
- rest_false_motion_rate: 1.0000
- motion_false_rest_rate: 0.0000
### extra_holdout gate diagnostics
- true_motion_rate: 0.5423
- pred_motion_rate: 1.0000
- near_threshold_frac_0p05: 0.0000
- threshold_percentile: 0.0000
- rest_score_mean: 0.5756
- motion_score_mean: 0.5633
- class_score_gap: -0.0123
- class_separation_cohen_d: -0.8726
- rest_false_motion_rate: 1.0000
- motion_false_rest_rate: 0.0000

## Grouped Scorecards
### Cross-validation by dataset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| iamwell | 4488 | 0.4103 | 0.5211 | 0.3387 | NA | NA | NA |
| mimic_perform | 150 | 0.0033 | 0.0066 | 0.1340 | NA | NA | NA |
| pulse_transit_time_ppg | 6249 | 0.5789 | 0.6656 | 0.1313 | 0.7197 | 0.6753 | 0.6865 |
### Holdout by dataset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| iamwell | 954 | 0.3943 | 0.5473 | 0.2116 | NA | NA | NA |
| pulse_transit_time_ppg | 1828 | 0.4260 | 0.6271 | 0.1088 | 0.8001 | 0.9247 | 1.0000 |
### Extra holdout by dataset/subset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| mimic_perform::mimic_perform_af_csv | 300 | 0.0027 | 0.0073 | 0.1067 | NA | NA | NA |
| simultaneous_measurements::simultaneous_measurements | 6410 | 0.0452 | 0.1624 | 0.1784 | 0.7032 | 0.2656 | 1.0000 |

## Gate Input / IMU Audit
- model_input_channels: 1
- model_input_names: ['ppg']
- gate_head_input: shared PPG encoder mid-level features
- uses_imu_in_gate_head: False
- note: This script trains a PPG-only gate. It should not be interpreted as the IMU-based static/dynamic detector.
| dataset | raw_IMU_columns_available | loaded_into_model | unit_status | gravity_removal_status |
|---|---:|---:|---|---|
| iamwell | False | False | not_available | not_applicable |
| mimic_perform | False | False | not_available | not_applicable |
| pulse_transit_time_ppg | True | False | PTT a_x/a_y/a_z appear near m/s^2 scale; raw gravity component is present unless removed upstream | not standardized in this PPG-only training script |
| simultaneous_measurements | True | False | likely g for HEXOSKIN acceleration channels | not standardized in this PPG-only training script |
| vitaldb_open | False | False | not_available | not_applicable |

## PPG-ECG Delay Analysis
- analyzed_segments: 65
| dataset | segments | matched_count | median_delay_s | delay_IQR_across_segments_s | mean_coverage |
|---|---:|---:|---:|---:|---:|
| iamwell | 15 | 37311 | 0.2879 | 0.1719 | 0.9049 |
| mimic_perform | 2 | 2750 | 0.4794 | 0.0547 | 0.9676 |
| pulse_transit_time_ppg | 22 | 11229 | 0.2357 | 0.1406 | 0.8094 |
| simultaneous_measurements | 26 | 16092 | 0.2924 | 0.0312 | 0.8520 |

## Leave-One-Dataset-Out Validation
- status: disabled or unavailable

## Motion Detector Benchmark
- status: completed
- reason: None
- uses_imu: True
- input_channels: ['ppg', 'acc_dyn_x', 'acc_dyn_y', 'acc_dyn_z', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_mag', 'gyro_mag', 'jerk_mag']
- unit_handling: acceleration inferred per record and gravity removed; missing gyro zero-filled
- detector_A: denoiser-style encoder reused as a classifier; artifact decoder removed; motion head added
- detector_B: directly trained lightweight CNN motion detector
- split_counts: train_records=42, validation_records=9, holdout_records=15, extra_holdout_records=13
| model | split | windows | threshold | precision | recall | F1 | balanced_acc | ROC-AUC | PR-AUC | confusion_matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_denoiser_encoder | validation | 1111 | 0.9500 | 0.9725 | 1.0000 | 0.9861 | 0.9715 | 1.0000 | 1.0000 | [[347, 21], [0, 743]] |
| A_denoiser_encoder | holdout | 1828 | 0.9500 | 0.9975 | 1.0000 | 0.9988 | 0.9975 | 1.0000 | 1.0000 | [[606, 3], [0, 1219]] |
| A_denoiser_encoder | extra_holdout | 6052 | 0.9500 | 0.6974 | 0.7895 | 0.7406 | 0.6918 | 0.7775 | 0.8120 | [[1646, 1124], [691, 2591]] |
| B_light_cnn | validation | 1111 | 0.9000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [[368, 0], [0, 743]] |
| B_light_cnn | holdout | 1828 | 0.9000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [[609, 0], [0, 1219]] |
| B_light_cnn | extra_holdout | 6052 | 0.9000 | 0.9905 | 0.2538 | 0.4041 | 0.6255 | 0.8511 | 0.8929 | [[2762, 8], [2449, 833]] |

### Detector grouped by dataset
#### A_denoiser_encoder
| split | dataset | windows | F1 | balanced_acc | ROC-AUC | pred_motion_rate | motion_false_rest_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | pulse_transit_time_ppg | 1111 | 0.9861 | 0.9715 | 1.0000 | 0.6877 | 0.0000 |
| holdout | pulse_transit_time_ppg | 1828 | 0.9988 | 0.9975 | 1.0000 | 0.6685 | 0.0000 |
| extra_holdout | simultaneous_measurements | 6052 | 0.7406 | 0.6918 | 0.7775 | 0.6138 | 0.2105 |
#### B_light_cnn
| split | dataset | windows | F1 | balanced_acc | ROC-AUC | pred_motion_rate | motion_false_rest_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | pulse_transit_time_ppg | 1111 | 1.0000 | 1.0000 | 1.0000 | 0.6688 | 0.0000 |
| holdout | pulse_transit_time_ppg | 1828 | 1.0000 | 1.0000 | 1.0000 | 0.6668 | 0.0000 |
| extra_holdout | simultaneous_measurements | 6052 | 0.4041 | 0.6255 | 0.8511 | 0.1390 | 0.7462 |

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 15
- peak_sequence:
  n: 1360128
  threshold: 0.7250
  positive_rate: 0.1312
  accuracy: 0.8859
  balanced_accuracy: 0.7988
  precision: 0.5530
  recall: 0.6806
  f1: 0.6102
  confusion_matrix: [[1083489, 98189], [56994, 121456]]
  roc_auc: 0.8919
  pr_auc: 0.6347
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 17169
  fp: 15894
  fn: 19086
  precision: 0.5193
  recall: 0.4736
  f1: 0.4954
  timing_n: 17169
  timing_bias: -0.0048
  timing_mae: 0.0102
  timing_median_ae: 0.0156
  timing_std: 0.0117
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 1360128
  mae: 0.1855
  rmse: 0.2401
  median_ae: 0.1428
  bias: 0.1015
  pearson_r: -0.1427
  r2: -0.8486
- rri_event_sequence:
  n: 12880
  bias: 0.0001
  mae: 0.0077
  median_ae: 0.0000
  std: 0.0121
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 2945
  threshold: 0.4500
  positive_rate: 0.6669
  accuracy: 0.5708
  balanced_accuracy: 0.6417
  precision: 0.8550
  recall: 0.4292
  f1: 0.5715
  confusion_matrix: [[838, 143], [1121, 843]]
  roc_auc: 0.8417
  pr_auc: 0.8916

### fold_2
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 15
- peak_sequence:
  n: 1426944
  threshold: 0.7250
  positive_rate: 0.1227
  accuracy: 0.8927
  balanced_accuracy: 0.7613
  precision: 0.5598
  recall: 0.5872
  f1: 0.5732
  confusion_matrix: [[1171071, 80820], [72265, 102788]]
  roc_auc: 0.8930
  pr_auc: 0.6091
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 15637
  fp: 12363
  fn: 19927
  precision: 0.5585
  recall: 0.4397
  f1: 0.4920
  timing_n: 15637
  timing_bias: 0.0036
  timing_mae: 0.0095
  timing_median_ae: 0.0156
  timing_std: 0.0117
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 1426944
  mae: 0.2467
  rmse: 1.2861
  median_ae: 0.1363
  bias: -0.0592
  pearson_r: -0.0138
  r2: -0.0094
- rri_event_sequence:
  n: 11207
  bias: -0.0002
  mae: 0.0082
  median_ae: 0.0000
  std: 0.0126
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 3304
  threshold: 0.4500
  positive_rate: 0.6656
  accuracy: 0.6656
  balanced_accuracy: 0.5000
  precision: 0.6656
  recall: 1.0000
  f1: 0.7992
  confusion_matrix: [[0, 1105], [0, 2199]]
  roc_auc: 0.8582
  pr_auc: 0.8979

## Notes
- simultaneous_measurements uses .atr consensus beat annotations when available and .aux phase markers for gate supervision.
- iamwell currently contributes peak/IBI supervision only; its gate labels remain unavailable in this script.
- mimic_perform contributes PPG/ECG-derived pseudo peak/IBI supervision from local CSV files; CSV and WFDB mirrors should not be loaded together.
- mimic_perform_af_csv, mimic_perform_non_af_csv, MIMIC_PERform_1_min_noisy.mat, and MIMIC_PERform_1_min_neonate.mat are reserved for extra-holdout by default.
- vitaldb_open is optional and uses the vitaldb Python API with SNUADC/PLETH and SNUADC/ECG_II when --enable_vitaldb is set.
- ECG pseudo-label training is blocked unless the PTT ECG detector preflight passes aggregate peak F1 and IBI accuracy thresholds.
- peak_sequence is point-wise over the dense peak target; peak_events is the main beat-level metric at +/-20 ms.
- peak_events_by_tolerance reports layered PPG beat matching at +/-10, +/-20, +/-30, and +/-40 ms.
- rri_event_sequence is the main matched-beat interval metric at +/-20 ms; rri_event_sequence_by_tolerance reports the same layers.
- gate_logit remains an auxiliary PPG-only state head inside the peak/IBI model; the Motion Detector Benchmark is the dedicated IMU-aware rest/motion detector module.