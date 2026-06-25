# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260427-01_peak_hr_gate_balanced_v2
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 254
- internal_holdout_subject_count: 63
- extra_holdout_subject_count: 50
- cv_folds: 5
- final_train_epochs: 6
- norm_type: instance
- balanced_sampling: True
- augmentation_enabled: True
- domain_adversarial_lambda: 1.0000
- worst_domain_weight: 0.2500
- group_scorecards: True
- delay_analysis: True
- lodo_validation: False
- detector_benchmark: True
- detector_epochs: 8
- detector_model_base_channels: 24
- gate_input: shared PPG encoder mid-level features
- gate_uses_imu: False
- mimic_subsets: ['mimic_perform_train_all_csv', 'mimic_perform_test_all_csv']
- mimic_max_records: 200
- include_mimic_special_extra_holdout: True
- mimic_extra_holdout_subsets: ['mimic_perform_af_csv', 'mimic_perform_non_af_csv']
- mimic_extra_holdout_mat_files: ['MIMIC_PERform_1_min_noisy.mat', 'MIMIC_PERform_1_min_neonate.mat']
- enable_vitaldb: True
- vitaldb_max_cases: 80
- ecg_detector_preflight: True
- ecg_preflight_peak_f1: 0.9973
- ecg_preflight_ibi_accuracy: 1.0000

## Thresholds
- peak_threshold: 0.7500
- gate_threshold: 0.4750
- ppg_main_event_tolerance_ms: 20
- ppg_layered_event_tolerances_ms: [10, 20, 30, 40]

## ECG Peak Detector Preflight
- status: passed
- dataset: pulse_transit_time_ppg
- records: 66
- match_tolerance_sec: 0.0040
- required_peak_f1: 0.9500
- required_ibi_accuracy: 0.9500
- peak_tp: 46288
- peak_fp: 134
- peak_fn: 117
- peak_precision: 0.9971
- peak_recall: 0.9975
- peak_f1: 0.9973
- peak_timing_bias_sec: 0.0000
- peak_timing_mae_sec: 0.0000
- peak_timing_median_ae_sec: 0.0000
- peak_timing_std_sec: 0.0002
- peak_timing_variance_sec2: 0.0000
- ibi_pairs: 46222
- ibi_accuracy: 1.0000
- ibi_mae_sec: 0.0000
- ibi_median_ae_sec: 0.0000
- ibi_std_sec: 0.0003
- ibi_variance_sec2: 0.0000
- low_peak_f1_record_count: 0
- low_ibi_accuracy_record_count: 0

### Worst ECG detector records by peak F1
| record | true_peaks | pred_peaks | F1 | timing_MAE_s | IBI_acc |
|---|---:|---:|---:|---:|---:|
| s12_run.csv | 728 | 726 | 0.9546 | 0.0000 | 1.0000 |
| s13_sit.csv | 603 | 612 | 0.9547 | 0.0000 | 1.0000 |
| s12_walk.csv | 710 | 714 | 0.9831 | 0.0001 | 1.0000 |
| s2_run.csv | 1148 | 1147 | 0.9839 | 0.0000 | 1.0000 |
| s9_walk.csv | 620 | 629 | 0.9912 | 0.0001 | 1.0000 |
| s1_walk.csv | 767 | 767 | 0.9922 | 0.0001 | 1.0000 |
| s7_run.csv | 688 | 693 | 0.9949 | 0.0000 | 1.0000 |
| s16_run.csv | 606 | 606 | 0.9950 | 0.0002 | 1.0000 |
| s5_sit.csv | 638 | 636 | 0.9953 | 0.0003 | 1.0000 |
| s20_run.csv | 751 | 750 | 0.9953 | 0.0001 | 1.0000 |

## Cross-Validation Aggregate
### cross_validation
- windows: 89244
- peak_samples: 182771712
- rr_samples: 182771712
- gate_windows_labeled: 13819
- peak_sequence:
  n: 182771712
  threshold: 0.7500
  positive_rate: 0.1189
  accuracy: 0.8749
  balanced_accuracy: 0.7554
  precision: 0.4791
  recall: 0.5988
  f1: 0.5323
  confusion_matrix: [[146888996, 14149908], [8719971, 13012837]]
  roc_auc: 0.8809
  pr_auc: 0.5309
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 424551
  fp: 615872
  fn: 729347
  precision: 0.4081
  recall: 0.3679
  f1: 0.3870
  timing_n: 424551
  timing_bias: 0.0015
  timing_mae: 0.0100
  timing_median_ae: 0.0117
  timing_std: 0.0117
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 182771712
  mae: 4.9752
  rmse: 33.8696
  median_ae: 0.1000
  bias: -4.8382
  pearson_r: 0.1301
  r2: -0.0194
- rri_event_sequence:
  n: 362106
  bias: -0.0002
  mae: 0.0090
  median_ae: 0.0078
  std: 0.0120
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 13819
  threshold: 0.4750
  positive_rate: 0.6664
  accuracy: 0.8763
  balanced_accuracy: 0.8640
  precision: 0.9124
  recall: 0.9009
  f1: 0.9066
  confusion_matrix: [[3813, 797], [913, 8296]]
  roc_auc: 0.9005
  pr_auc: 0.9268
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 287737 | 752686 | 866161 | 0.2766 | 0.2494 | 0.2623 | 0.0065 | 0.0078 |
  | 20ms | 424551 | 615872 | 729347 | 0.4081 | 0.3679 | 0.3870 | 0.0100 | 0.0117 |
  | 30ms | 585490 | 454933 | 568408 | 0.5627 | 0.5074 | 0.5336 | 0.0147 | 0.0156 |
  | 40ms | 664933 | 375490 | 488965 | 0.6391 | 0.5762 | 0.6060 | 0.0174 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 232169 | -0.0001 | 0.0067 | 0.0039 | 0.0088 | 0.0001 |
  | 20ms | 362106 | -0.0002 | 0.0090 | 0.0078 | 0.0120 | 0.0001 |
  | 30ms | 517377 | -0.0002 | 0.0113 | 0.0078 | 0.0156 | 0.0002 |
  | 40ms | 594612 | -0.0002 | 0.0125 | 0.0078 | 0.0176 | 0.0003 |

## Holdout
### holdout
- windows: 20742
- peak_samples: 42479616
- rr_samples: 42479616
- gate_windows_labeled: 2164
- peak_sequence:
  n: 42479616
  threshold: 0.7500
  positive_rate: 0.1169
  accuracy: 0.8693
  balanced_accuracy: 0.7460
  precision: 0.4540
  recall: 0.5851
  f1: 0.5113
  confusion_matrix: [[34021661, 3493599], [2059766, 2904590]]
  roc_auc: 0.8610
  pr_auc: 0.4768
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 95200
  fp: 144841
  fn: 168483
  precision: 0.3966
  recall: 0.3610
  f1: 0.3780
  timing_n: 95200
  timing_bias: 0.0013
  timing_mae: 0.0102
  timing_median_ae: 0.0117
  timing_std: 0.0119
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 42479616
  mae: 4.8271
  rmse: 27.0166
  median_ae: 0.1288
  bias: -4.6863
  pearson_r: 0.0334
  r2: -0.0306
- rri_event_sequence:
  n: 81491
  bias: -0.0002
  mae: 0.0095
  median_ae: 0.0078
  std: 0.0127
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 2164
  threshold: 0.4750
  positive_rate: 0.6668
  accuracy: 0.8226
  balanced_accuracy: 0.7906
  precision: 0.8532
  recall: 0.8863
  f1: 0.8695
  confusion_matrix: [[501, 220], [164, 1279]]
  roc_auc: 0.8499
  pr_auc: 0.8944
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 63672 | 176369 | 200011 | 0.2653 | 0.2415 | 0.2528 | 0.0066 | 0.0078 |
  | 20ms | 95200 | 144841 | 168483 | 0.3966 | 0.3610 | 0.3780 | 0.0102 | 0.0117 |
  | 30ms | 131560 | 108481 | 132123 | 0.5481 | 0.4989 | 0.5223 | 0.0148 | 0.0156 |
  | 40ms | 148634 | 91407 | 115049 | 0.6192 | 0.5637 | 0.5901 | 0.0174 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 51446 | -0.0001 | 0.0069 | 0.0039 | 0.0090 | 0.0001 |
  | 20ms | 81491 | -0.0002 | 0.0095 | 0.0078 | 0.0127 | 0.0002 |
  | 30ms | 116675 | -0.0002 | 0.0123 | 0.0078 | 0.0168 | 0.0003 |
  | 40ms | 133257 | -0.0003 | 0.0136 | 0.0078 | 0.0188 | 0.0004 |

## Extra Holdout
### extra_holdout
- windows: 33738
- peak_samples: 69095424
- rr_samples: 69095424
- gate_windows_labeled: 12032
- peak_sequence:
  n: 69095424
  threshold: 0.7500
  positive_rate: 0.1147
  accuracy: 0.8403
  balanced_accuracy: 0.5818
  precision: 0.2783
  recall: 0.2464
  f1: 0.2614
  confusion_matrix: [[56106975, 5063619], [5972116, 1952714]]
  roc_auc: 0.6782
  pr_auc: 0.2371
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 54733
  fp: 235247
  fn: 366058
  precision: 0.1887
  recall: 0.1301
  f1: 0.1540
  timing_n: 54733
  timing_bias: 0.0007
  timing_mae: 0.0106
  timing_median_ae: 0.0117
  timing_std: 0.0123
  timing_variance: 0.0002
- hr_interval_sequence:
  n: 69095424
  mae: 0.1364
  rmse: 0.1809
  median_ae: 0.1110
  bias: -0.0058
  pearson_r: 0.4659
  r2: 0.1157
- rri_event_sequence:
  n: 38000
  bias: -0.0001
  mae: 0.0099
  median_ae: 0.0078
  std: 0.0133
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 12032
  threshold: 0.4750
  positive_rate: 0.5419
  accuracy: 0.4031
  balanced_accuracy: 0.3955
  precision: 0.4528
  recall: 0.4865
  f1: 0.4690
  confusion_matrix: [[1678, 3834], [3348, 3172]]
  roc_auc: 0.4088
  pr_auc: 0.5455
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 35066 | 254914 | 385725 | 0.1209 | 0.0833 | 0.0987 | 0.0067 | 0.0078 |
  | 20ms | 54733 | 235247 | 366058 | 0.1887 | 0.1301 | 0.1540 | 0.0106 | 0.0117 |
  | 30ms | 81089 | 208891 | 339702 | 0.2796 | 0.1927 | 0.2282 | 0.0160 | 0.0156 |
  | 40ms | 95700 | 194280 | 325091 | 0.3300 | 0.2274 | 0.2693 | 0.0192 | 0.0195 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 21572 | -0.0001 | 0.0069 | 0.0039 | 0.0091 | 0.0001 |
  | 20ms | 38000 | -0.0001 | 0.0099 | 0.0078 | 0.0133 | 0.0002 |
  | 30ms | 61449 | -0.0001 | 0.0137 | 0.0078 | 0.0189 | 0.0004 |
  | 40ms | 74623 | -0.0002 | 0.0161 | 0.0117 | 0.0225 | 0.0005 |

## Gate Diagnostics
### cross_validation gate diagnostics
- true_motion_rate: 0.6664
- pred_motion_rate: 0.6580
- near_threshold_frac_0p05: 0.0096
- threshold_percentile: 0.3420
- rest_score_mean: 0.1811
- motion_score_mean: 0.8874
- class_score_gap: 0.7064
- class_separation_cohen_d: 2.2501
- rest_false_motion_rate: 0.1729
- motion_false_rest_rate: 0.0991
### holdout gate diagnostics
- true_motion_rate: 0.6668
- pred_motion_rate: 0.6927
- near_threshold_frac_0p05: 0.0231
- threshold_percentile: 0.3073
- rest_score_mean: 0.3105
- motion_score_mean: 0.8654
- class_score_gap: 0.5549
- class_separation_cohen_d: 1.5740
- rest_false_motion_rate: 0.3051
- motion_false_rest_rate: 0.1137
### extra_holdout gate diagnostics
- true_motion_rate: 0.5419
- pred_motion_rate: 0.5823
- near_threshold_frac_0p05: 0.0135
- threshold_percentile: 0.4177
- rest_score_mean: 0.7015
- motion_score_mean: 0.5141
- class_score_gap: -0.1874
- class_separation_cohen_d: -0.4557
- rest_false_motion_rate: 0.6956
- motion_false_rest_rate: 0.5135

## Grouped Scorecards
### Cross-validation by dataset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| iamwell | 9491 | 0.3656 | 0.4877 | 0.2160 | NA | NA | NA |
| mimic_perform | 47520 | 0.3441 | 0.5189 | 0.3839 | NA | NA | NA |
| pulse_transit_time_ppg | 13819 | 0.6509 | 0.7047 | 0.0904 | 0.9066 | 0.9005 | 0.6580 |
| vitaldb_open | 18414 | 0.3183 | 0.4563 | 22.9428 | NA | NA | NA |
### Holdout by dataset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| iamwell | 1352 | 0.5690 | 0.6538 | 0.1257 | NA | NA | NA |
| mimic_perform | 11880 | 0.3969 | 0.5414 | 0.1705 | NA | NA | NA |
| pulse_transit_time_ppg | 2164 | 0.3007 | 0.3909 | 0.1189 | 0.8695 | 0.8499 | 0.6927 |
| vitaldb_open | 5346 | 0.2361 | 0.3519 | 18.2699 | NA | NA | NA |
### Extra holdout by dataset/subset
| group | windows | peak_event_F1_20ms | peak_seq_F1 | IBI_MAE_s | gate_F1 | gate_AUC | gate_pred_motion_rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| mimic_perform::MIMIC_PERform_1_min_neonate | 27 | 0.8205 | 0.7519 | 0.1439 | NA | NA | NA |
| mimic_perform::MIMIC_PERform_1_min_noisy | 27 | 0.2181 | 0.2753 | 0.1618 | NA | NA | NA |
| mimic_perform::mimic_perform_af_csv | 11343 | 0.1093 | 0.2174 | 0.1441 | NA | NA | NA |
| mimic_perform::mimic_perform_non_af_csv | 9552 | 0.2855 | 0.4776 | 0.1326 | NA | NA | NA |
| simultaneous_measurements::simultaneous_measurements | 12789 | 0.0948 | 0.0934 | 0.1323 | 0.4690 | 0.4088 | 0.5823 |

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
- analyzed_segments: 378
| dataset | segments | matched_count | median_delay_s | delay_IQR_across_segments_s | mean_coverage |
|---|---:|---:|---:|---:|---:|
| iamwell | 15 | 36997 | 0.2796 | 0.1738 | 0.8958 |
| mimic_perform | 237 | 269198 | 0.3228 | 0.2930 | 0.9409 |
| pulse_transit_time_ppg | 21 | 11141 | 0.2305 | 0.1523 | 0.8430 |
| simultaneous_measurements | 26 | 16061 | 0.2923 | 0.0361 | 0.8503 |
| vitaldb_open | 79 | 37297 | 0.4626 | 0.2344 | 0.8336 |

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
- split_counts: train_records=45, validation_records=12, holdout_records=9, extra_holdout_records=13
| model | split | windows | threshold | precision | recall | F1 | balanced_acc | ROC-AUC | PR-AUC | confusion_matrix |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_denoiser_encoder | validation | 2896 | 0.0500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [[965, 0], [0, 1931]] |
| A_denoiser_encoder | holdout | 2164 | 0.0500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [[721, 0], [0, 1443]] |
| A_denoiser_encoder | extra_holdout | 12032 | 0.0500 | 0.8559 | 0.6741 | 0.7542 | 0.7699 | 0.8269 | 0.8588 | [[4772, 740], [2125, 4395]] |
| B_light_cnn | validation | 2896 | 0.0500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [[965, 0], [0, 1931]] |
| B_light_cnn | holdout | 2164 | 0.0500 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [[721, 0], [0, 1443]] |
| B_light_cnn | extra_holdout | 12032 | 0.0500 | 0.8710 | 0.6794 | 0.7634 | 0.7802 | 0.8642 | 0.9055 | [[4856, 656], [2090, 4430]] |

### Detector grouped by dataset
#### A_denoiser_encoder
| split | dataset | windows | F1 | balanced_acc | ROC-AUC | pred_motion_rate | motion_false_rest_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | pulse_transit_time_ppg | 2896 | 1.0000 | 1.0000 | 1.0000 | 0.6668 | 0.0000 |
| holdout | pulse_transit_time_ppg | 2164 | 1.0000 | 1.0000 | 1.0000 | 0.6668 | 0.0000 |
| extra_holdout | simultaneous_measurements | 12032 | 0.7542 | 0.7699 | 0.8269 | 0.4268 | 0.3259 |
#### B_light_cnn
| split | dataset | windows | F1 | balanced_acc | ROC-AUC | pred_motion_rate | motion_false_rest_rate |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | pulse_transit_time_ppg | 2896 | 1.0000 | 1.0000 | 1.0000 | 0.6668 | 0.0000 |
| holdout | pulse_transit_time_ppg | 2164 | 1.0000 | 1.0000 | 1.0000 | 0.6668 | 0.0000 |
| extra_holdout | simultaneous_measurements | 12032 | 0.7634 | 0.7802 | 0.8642 | 0.4227 | 0.3206 |

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 7
- train_subject_count: 203
- val_subject_count: 51
- peak_sequence:
  n: 40060928
  threshold: 0.7500
  positive_rate: 0.1210
  accuracy: 0.8803
  balanced_accuracy: 0.7679
  precision: 0.5045
  recall: 0.6195
  f1: 0.5561
  confusion_matrix: [[32264513, 2949258], [1844128, 3003029]]
  roc_auc: 0.8907
  pr_auc: 0.5643
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 98020
  fp: 130579
  fn: 159395
  precision: 0.4288
  recall: 0.3808
  f1: 0.4034
  timing_n: 98020
  timing_bias: -0.0000
  timing_mae: 0.0099
  timing_median_ae: 0.0078
  timing_std: 0.0117
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 40060928
  mae: 0.5376
  rmse: 3.3747
  median_ae: 0.0811
  bias: -0.4170
  pearson_r: 0.2587
  r2: 0.0115
- rri_event_sequence:
  n: 83201
  bias: -0.0001
  mae: 0.0083
  median_ae: 0.0078
  std: 0.0113
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 5078
  threshold: 0.4750
  positive_rate: 0.6658
  accuracy: 0.9027
  balanced_accuracy: 0.8876
  precision: 0.9217
  recall: 0.9332
  f1: 0.9274
  confusion_matrix: [[1429, 268], [226, 3155]]
  roc_auc: 0.9155
  pr_auc: 0.9383

### fold_2
- best_epoch: 6
- train_subject_count: 203
- val_subject_count: 51
- peak_sequence:
  n: 36194304
  threshold: 0.7500
  positive_rate: 0.1174
  accuracy: 0.8643
  balanced_accuracy: 0.7356
  precision: 0.4396
  recall: 0.5673
  f1: 0.4954
  confusion_matrix: [[28870416, 3073904], [1838841, 2411143]]
  roc_auc: 0.8619
  pr_auc: 0.4478
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 66606
  fp: 136327
  fn: 158964
  precision: 0.3282
  recall: 0.2953
  f1: 0.3109
  timing_n: 66606
  timing_bias: 0.0018
  timing_mae: 0.0107
  timing_median_ae: 0.0117
  timing_std: 0.0122
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 36194304
  mae: 1.7641
  rmse: 15.0431
  median_ae: 0.1010
  bias: -1.6293
  pearson_r: 0.0094
  r2: -0.0118
- rri_event_sequence:
  n: 55330
  bias: -0.0002
  mae: 0.0095
  median_ae: 0.0078
  std: 0.0127
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 1457
  threshold: 0.4750
  positive_rate: 0.6637
  accuracy: 0.8312
  balanced_accuracy: 0.8617
  precision: 0.9712
  recall: 0.7684
  f1: 0.8580
  confusion_matrix: [[468, 22], [224, 743]]
  roc_auc: 0.9269
  pr_auc: 0.9655

### fold_3
- best_epoch: 4
- train_subject_count: 203
- val_subject_count: 51
- peak_sequence:
  n: 37269504
  threshold: 0.7500
  positive_rate: 0.1135
  accuracy: 0.8768
  balanced_accuracy: 0.7620
  precision: 0.4675
  recall: 0.6135
  f1: 0.5307
  confusion_matrix: [[30081009, 2956599], [1635693, 2596203]]
  roc_auc: 0.8881
  pr_auc: 0.5628
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 92826
  fp: 122536
  fn: 131831
  precision: 0.4310
  recall: 0.4132
  f1: 0.4219
  timing_n: 92826
  timing_bias: 0.0031
  timing_mae: 0.0096
  timing_median_ae: 0.0078
  timing_std: 0.0110
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 37269504
  mae: 9.4455
  rmse: 49.1156
  median_ae: 0.0923
  bias: -9.3390
  pearson_r: 0.1938
  r2: -0.0361
- rri_event_sequence:
  n: 79855
  bias: -0.0001
  mae: 0.0092
  median_ae: 0.0078
  std: 0.0120
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 4391
  threshold: 0.4750
  positive_rate: 0.6684
  accuracy: 0.8700
  balanced_accuracy: 0.8835
  precision: 0.9571
  recall: 0.8433
  f1: 0.8966
  confusion_matrix: [[1345, 111], [460, 2475]]
  roc_auc: 0.9543
  pr_auc: 0.9761

### fold_4
- best_epoch: 6
- train_subject_count: 203
- val_subject_count: 51
- peak_sequence:
  n: 36497408
  threshold: 0.7500
  positive_rate: 0.1204
  accuracy: 0.8786
  balanced_accuracy: 0.7586
  precision: 0.4965
  recall: 0.6007
  f1: 0.5436
  confusion_matrix: [[29425643, 2677295], [1754910, 2639560]]
  roc_auc: 0.8848
  pr_auc: 0.5495
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 92329
  fp: 120222
  fn: 141006
  precision: 0.4344
  recall: 0.3957
  f1: 0.4141
  timing_n: 92329
  timing_bias: 0.0010
  timing_mae: 0.0100
  timing_median_ae: 0.0117
  timing_std: 0.0117
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 36497408
  mae: 9.0616
  rmse: 49.0722
  median_ae: 0.1205
  bias: -8.8848
  pearson_r: 0.1548
  r2: -0.0326
- rri_event_sequence:
  n: 79326
  bias: -0.0001
  mae: 0.0092
  median_ae: 0.0078
  std: 0.0123
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 1446
  threshold: 0.4750
  positive_rate: 0.6667
  accuracy: 0.8285
  balanced_accuracy: 0.7443
  precision: 0.7968
  recall: 0.9969
  f1: 0.8857
  confusion_matrix: [[237, 245], [3, 961]]
  roc_auc: 0.7540
  pr_auc: 0.8336

### fold_5
- best_epoch: 9
- train_subject_count: 204
- val_subject_count: 50
- peak_sequence:
  n: 32749568
  threshold: 0.7500
  positive_rate: 0.1224
  accuracy: 0.8736
  balanced_accuracy: 0.7513
  precision: 0.4866
  recall: 0.5894
  f1: 0.5331
  confusion_matrix: [[26247415, 2492852], [1646399, 2362902]]
  roc_auc: 0.8778
  pr_auc: 0.5366
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 74770
  fp: 106208
  fn: 138151
  precision: 0.4131
  recall: 0.3512
  f1: 0.3796
  timing_n: 74770
  timing_bias: 0.0021
  timing_mae: 0.0102
  timing_median_ae: 0.0117
  timing_std: 0.0118
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 32749568
  mae: 4.3111
  rmse: 26.6300
  median_ae: 0.1090
  bias: -4.1610
  pearson_r: 0.1690
  r2: -0.0226
- rri_event_sequence:
  n: 64394
  bias: -0.0002
  mae: 0.0087
  median_ae: 0.0078
  std: 0.0118
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 1447
  threshold: 0.4750
  positive_rate: 0.6648
  accuracy: 0.8956
  balanced_accuracy: 0.8443
  precision: 0.8643
  recall: 1.0000
  f1: 0.9272
  confusion_matrix: [[334, 151], [0, 962]]
  roc_auc: 0.9774
  pr_auc: 0.9848

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