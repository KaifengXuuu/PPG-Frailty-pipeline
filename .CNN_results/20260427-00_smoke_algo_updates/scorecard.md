# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260427-00_smoke_algo_updates
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 31
- internal_holdout_subject_count: 8
- extra_holdout_subject_count: 15
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
- gate_input: shared PPG encoder mid-level features
- gate_uses_imu: False
- mimic_subsets: ['mimic_perform_train_all_csv', 'mimic_perform_test_all_csv']
- mimic_max_records: 2
- include_mimic_special_extra_holdout: True
- mimic_extra_holdout_subsets: ['mimic_perform_af_csv', 'mimic_perform_non_af_csv']
- mimic_extra_holdout_mat_files: ['MIMIC_PERform_1_min_noisy.mat', 'MIMIC_PERform_1_min_neonate.mat']
- enable_vitaldb: False
- vitaldb_max_cases: 20
- ecg_detector_preflight: False
- ecg_preflight_peak_f1: NA
- ecg_preflight_ibi_accuracy: NA

## Thresholds
- peak_threshold: 0.7500
- gate_threshold: 0.5000
- ppg_main_event_tolerance_ms: 20
- ppg_layered_event_tolerances_ms: [10, 20, 30, 40]

## ECG Peak Detector Preflight
- status: disabled

## Cross-Validation Aggregate
### cross_validation
- windows: 11040
- peak_samples: 2826240
- rr_samples: 2826240
- gate_windows_labeled: 6252
- peak_sequence:
  n: 2826240
  threshold: 0.7500
  positive_rate: 0.1256
  accuracy: 0.8780
  balanced_accuracy: 0.7585
  precision: 0.5122
  recall: 0.5990
  f1: 0.5522
  confusion_matrix: [[2268693, 202519], [142375, 212653]]
  roc_auc: 0.8684
  pr_auc: 0.5513
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 29730
  fp: 36002
  fn: 42397
  precision: 0.4523
  recall: 0.4122
  f1: 0.4313
  timing_n: 29730
  timing_bias: 0.0011
  timing_mae: 0.0094
  timing_median_ae: 0.0156
  timing_std: 0.0120
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 2826240
  mae: 0.2349
  rmse: 0.9346
  median_ae: 0.1579
  bias: 0.0732
  pearson_r: -0.0251
  r2: -0.0375
- rri_event_sequence:
  n: 21850
  bias: -0.0001
  mae: 0.0080
  median_ae: 0.0000
  std: 0.0123
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 6252
  threshold: 0.5000
  positive_rate: 0.6665
  accuracy: 0.7374
  balanced_accuracy: 0.6126
  precision: 0.7214
  recall: 0.9873
  f1: 0.8336
  confusion_matrix: [[496, 1589], [53, 4114]]
  roc_auc: 0.6838
  pr_auc: 0.8258
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 29730 | 36002 | 42397 | 0.4523 | 0.4122 | 0.4313 | 0.0094 | 0.0156 |
  | 20ms | 29730 | 36002 | 42397 | 0.4523 | 0.4122 | 0.4313 | 0.0094 | 0.0156 |
  | 30ms | 41152 | 24580 | 30975 | 0.6261 | 0.5705 | 0.5970 | 0.0154 | 0.0156 |
  | 40ms | 48242 | 17490 | 23885 | 0.7339 | 0.6688 | 0.6999 | 0.0201 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 21850 | -0.0001 | 0.0080 | 0.0000 | 0.0123 | 0.0002 |
  | 20ms | 21850 | -0.0001 | 0.0080 | 0.0000 | 0.0123 | 0.0002 |
  | 30ms | 31624 | -0.0003 | 0.0107 | 0.0156 | 0.0164 | 0.0003 |
  | 40ms | 38138 | -0.0006 | 0.0128 | 0.0156 | 0.0197 | 0.0004 |

## Holdout
### holdout
- windows: 2779
- peak_samples: 711424
- rr_samples: 711424
- gate_windows_labeled: 1825
- peak_sequence:
  n: 711424
  threshold: 0.7500
  positive_rate: 0.1266
  accuracy: 0.8915
  balanced_accuracy: 0.7580
  precision: 0.5706
  recall: 0.5791
  f1: 0.5748
  confusion_matrix: [[582069, 39261], [37920, 52174]]
  roc_auc: 0.8866
  pr_auc: 0.6162
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 8621
  fp: 6677
  fn: 9683
  precision: 0.5635
  recall: 0.4710
  f1: 0.5131
  timing_n: 8621
  timing_bias: 0.0004
  timing_mae: 0.0101
  timing_median_ae: 0.0156
  timing_std: 0.0126
  timing_variance: 0.0002
- hr_interval_sequence:
  n: 711424
  mae: 0.1998
  rmse: 0.2499
  median_ae: 0.1658
  bias: 0.0678
  pearson_r: -0.1026
  r2: -0.4491
- rri_event_sequence:
  n: 6248
  bias: -0.0002
  mae: 0.0085
  median_ae: 0.0000
  std: 0.0129
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 1825
  threshold: 0.5000
  positive_rate: 0.6658
  accuracy: 0.3342
  balanced_accuracy: 0.5000
  precision: 0.0000
  recall: 0.0000
  f1: 0.0000
  confusion_matrix: [[610, 0], [1215, 0]]
  roc_auc: 0.8746
  pr_auc: 0.9154
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 8621 | 6677 | 9683 | 0.5635 | 0.4710 | 0.5131 | 0.0101 | 0.0156 |
  | 20ms | 8621 | 6677 | 9683 | 0.5635 | 0.4710 | 0.5131 | 0.0101 | 0.0156 |
  | 30ms | 11182 | 4116 | 7122 | 0.7309 | 0.6109 | 0.6656 | 0.0149 | 0.0156 |
  | 40ms | 12061 | 3237 | 6243 | 0.7884 | 0.6589 | 0.7179 | 0.0173 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 6248 | -0.0002 | 0.0085 | 0.0000 | 0.0129 | 0.0002 |
  | 20ms | 6248 | -0.0002 | 0.0085 | 0.0000 | 0.0129 | 0.0002 |
  | 30ms | 8648 | -0.0002 | 0.0124 | 0.0156 | 0.0178 | 0.0003 |
  | 40ms | 9464 | -0.0001 | 0.0144 | 0.0156 | 0.0206 | 0.0004 |

## Extra Holdout
### extra_holdout
- windows: 7010
- peak_samples: 1794560
- rr_samples: 1794560
- gate_windows_labeled: 6052
- peak_sequence:
  n: 1794560
  threshold: 0.7500
  positive_rate: 0.1269
  accuracy: 0.8169
  balanced_accuracy: 0.5116
  precision: 0.1581
  recall: 0.1024
  f1: 0.1243
  confusion_matrix: [[1442671, 124157], [204420, 23312]]
  roc_auc: 0.6123
  pr_auc: 0.1653
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 1901
  fp: 24422
  fn: 44352
  precision: 0.0722
  recall: 0.0411
  f1: 0.0524
  timing_n: 1901
  timing_bias: -0.0049
  timing_mae: 0.0105
  timing_median_ae: 0.0156
  timing_std: 0.0118
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 1794560
  mae: 0.1966
  rmse: 0.2420
  median_ae: 0.1698
  bias: 0.1185
  pearson_r: -0.1273
  r2: -1.1983
- rri_event_sequence:
  n: 697
  bias: 0.0019
  mae: 0.0106
  median_ae: 0.0156
  std: 0.0148
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 6052
  threshold: 0.5000
  positive_rate: 0.5423
  accuracy: 0.4577
  balanced_accuracy: 0.5000
  precision: 0.0000
  recall: 0.0000
  f1: 0.0000
  confusion_matrix: [[2770, 0], [3282, 0]]
  roc_auc: 0.6557
  pr_auc: 0.6905
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 1901 | 24422 | 44352 | 0.0722 | 0.0411 | 0.0524 | 0.0105 | 0.0156 |
  | 20ms | 1901 | 24422 | 44352 | 0.0722 | 0.0411 | 0.0524 | 0.0105 | 0.0156 |
  | 30ms | 3553 | 22770 | 42700 | 0.1350 | 0.0768 | 0.0979 | 0.0201 | 0.0156 |
  | 40ms | 5490 | 20833 | 40763 | 0.2086 | 0.1187 | 0.1513 | 0.0296 | 0.0312 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 697 | 0.0019 | 0.0106 | 0.0156 | 0.0148 | 0.0002 |
  | 20ms | 697 | 0.0019 | 0.0106 | 0.0156 | 0.0148 | 0.0002 |
  | 30ms | 1600 | 0.0020 | 0.0133 | 0.0156 | 0.0189 | 0.0004 |
  | 40ms | 2846 | 0.0020 | 0.0156 | 0.0156 | 0.0225 | 0.0005 |

## Gate Diagnostics
### cross_validation gate diagnostics
- true_motion_rate: 0.6665
- pred_motion_rate: 0.9122
- near_threshold_frac_0p05: 1.0000
- threshold_percentile: 0.0878
- rest_score_mean: 0.5150
- motion_score_mean: 0.5172
- class_score_gap: 0.0022
- class_separation_cohen_d: 0.1408
- rest_false_motion_rate: 0.7621
- motion_false_rest_rate: 0.0127
### holdout gate diagnostics
- true_motion_rate: 0.6658
- pred_motion_rate: 0.0000
- near_threshold_frac_0p05: 1.0000
- threshold_percentile: 1.0000
- rest_score_mean: 0.4862
- motion_score_mean: 0.4890
- class_score_gap: 0.0029
- class_separation_cohen_d: 1.5071
- rest_false_motion_rate: 0.0000
- motion_false_rest_rate: 1.0000
### extra_holdout gate diagnostics
- true_motion_rate: 0.5423
- pred_motion_rate: 0.0000
- near_threshold_frac_0p05: 1.0000
- threshold_percentile: 1.0000
- rest_score_mean: 0.4902
- motion_score_mean: 0.4907
- class_score_gap: 0.0005
- class_separation_cohen_d: 0.4132
- rest_false_motion_rate: 0.0000
- motion_false_rest_rate: 1.0000

## Grouped Scorecards
### Cross-validation by dataset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| iamwell | 4488 | 0.3064 | 0.4530 | 0.3724 | NA | NA | NA |
| mimic_perform | 300 | 0.0203 | 0.0261 | 0.1914 | NA | NA | NA |
| pulse_transit_time_ppg | 6252 | 0.5650 | 0.6581 | 0.1383 | 0.8336 | 0.6838 | 0.9122 |
### Holdout by dataset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| iamwell | 954 | 0.4325 | 0.5127 | 0.2526 | NA | NA | NA |
| pulse_transit_time_ppg | 1825 | 0.5654 | 0.6123 | 0.1721 | 0.0000 | 0.8746 | 0.0000 |
### Extra holdout by dataset/subset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| mimic_perform::mimic_perform_af_csv | 600 | 0.1330 | 0.1853 | 0.1661 | NA | NA | NA |
| simultaneous_measurements::simultaneous_measurements | 6410 | 0.0447 | 0.1180 | 0.1995 | 0.0000 | 0.6557 | 0.0000 |

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
| pulse_transit_time_ppg | False | False | not_available | not_applicable |
| simultaneous_measurements | True | False | likely g for HEXOSKIN acceleration channels | not standardized in this PPG-only training script |
| vitaldb_open | False | False | not_available | not_applicable |

## PPG-ECG Delay Analysis
- analyzed_segments: 67
| dataset | segments | matched_count | median_delay_s | delay_IQR_across_segments_s | mean_coverage |
|---|---:|---:|---:|---:|---:|
| iamwell | 15 | 37311 | 0.2879 | 0.1719 | 0.9049 |
| mimic_perform | 4 | 4784 | 0.4837 | 0.0391 | 0.9784 |
| pulse_transit_time_ppg | 22 | 11229 | 0.2357 | 0.1406 | 0.8094 |
| simultaneous_measurements | 26 | 16092 | 0.2924 | 0.0312 | 0.8520 |

## Leave-One-Dataset-Out Validation
- status: disabled or unavailable

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 16
- peak_sequence:
  n: 1439232
  threshold: 0.7500
  positive_rate: 0.1255
  accuracy: 0.8834
  balanced_accuracy: 0.7804
  precision: 0.5294
  recall: 0.6429
  f1: 0.5807
  confusion_matrix: [[1155355, 103242], [64502, 116133]]
  roc_auc: 0.8737
  pr_auc: 0.5647
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 16128
  fp: 17980
  fn: 20586
  precision: 0.4729
  recall: 0.4393
  f1: 0.4555
  timing_n: 16128
  timing_bias: 0.0013
  timing_mae: 0.0094
  timing_median_ae: 0.0156
  timing_std: 0.0121
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 1439232
  mae: 0.2229
  rmse: 0.2811
  median_ae: 0.1838
  bias: 0.1597
  pearson_r: -0.0392
  r2: -1.3498
- rri_event_sequence:
  n: 11900
  bias: -0.0002
  mae: 0.0081
  median_ae: 0.0000
  std: 0.0125
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 2958
  threshold: 0.5000
  positive_rate: 0.6670
  accuracy: 0.6670
  balanced_accuracy: 0.5000
  precision: 0.6670
  recall: 1.0000
  f1: 0.8002
  confusion_matrix: [[0, 985], [0, 1973]]
  roc_auc: 0.9077
  pr_auc: 0.9418

### fold_2
- best_epoch: 1
- train_subject_count: 16
- val_subject_count: 15
- peak_sequence:
  n: 1387008
  threshold: 0.7500
  positive_rate: 0.1257
  accuracy: 0.8723
  balanced_accuracy: 0.7358
  precision: 0.4930
  recall: 0.5535
  f1: 0.5215
  confusion_matrix: [[1113338, 99277], [77873, 96520]]
  roc_auc: 0.8633
  pr_auc: 0.5349
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 13602
  fp: 18022
  fn: 21811
  precision: 0.4301
  recall: 0.3841
  f1: 0.4058
  timing_n: 13602
  timing_bias: 0.0009
  timing_mae: 0.0093
  timing_median_ae: 0.0156
  timing_std: 0.0120
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 1387008
  mae: 0.2474
  rmse: 1.3030
  median_ae: 0.1332
  bias: -0.0165
  pearson_r: -0.0117
  r2: -0.0116
- rri_event_sequence:
  n: 9950
  bias: 0.0001
  mae: 0.0079
  median_ae: 0.0000
  std: 0.0121
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 3294
  threshold: 0.5000
  positive_rate: 0.6661
  accuracy: 0.8005
  balanced_accuracy: 0.7134
  precision: 0.7800
  recall: 0.9758
  f1: 0.8670
  confusion_matrix: [[496, 604], [53, 2141]]
  roc_auc: 0.8315
  pr_auc: 0.8905

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