# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260420-04_peak_hr_gate_v1
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 365
- internal_holdout_subject_count: 91
- extra_holdout_subject_count: 50
- cv_folds: 5
- final_train_epochs: 6
- norm_type: instance
- balanced_sampling: True
- augmentation_enabled: True
- domain_adversarial_lambda: 1.0000
- mimic_subsets: ['mimic_perform_train_all_csv', 'mimic_perform_test_all_csv']
- mimic_max_records: 0
- include_mimic_special_extra_holdout: True
- mimic_extra_holdout_subsets: ['mimic_perform_af_csv', 'mimic_perform_non_af_csv']
- mimic_extra_holdout_mat_files: ['MIMIC_PERform_1_min_noisy.mat', 'MIMIC_PERform_1_min_neonate.mat']
- enable_vitaldb: True
- vitaldb_max_cases: 20
- ecg_detector_preflight: True
- ecg_preflight_peak_f1: 0.9973
- ecg_preflight_ibi_accuracy: 1.0000

## Thresholds
- peak_threshold: 0.7750
- gate_threshold: 0.9500
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
- windows: 122053
- peak_samples: 249964544
- rr_samples: 249964544
- gate_windows_labeled: 13067
- peak_sequence:
  n: 249964544
  threshold: 0.7750
  positive_rate: 0.1320
  accuracy: 0.8678
  balanced_accuracy: 0.7428
  precision: 0.4991
  recall: 0.5729
  f1: 0.5335
  confusion_matrix: [[198008904, 18967791], [14088002, 18899847]]
  roc_auc: 0.8637
  pr_auc: 0.5406
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 666124
  fp: 888087
  fn: 1085314
  precision: 0.4286
  recall: 0.3803
  f1: 0.4030
  timing_n: 666124
  timing_bias: -0.0004
  timing_mae: 0.0098
  timing_median_ae: 0.0078
  timing_std: 0.0116
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 249964544
  mae: 1.1221
  rmse: 12.6543
  median_ae: 0.1210
  bias: -0.9537
  pearson_r: 0.0559
  r2: -0.0047
- rri_event_sequence:
  n: 576562
  bias: -0.0001
  mae: 0.0086
  median_ae: 0.0078
  std: 0.0116
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 13067
  threshold: 0.9500
  positive_rate: 0.6657
  accuracy: 0.8160
  balanced_accuracy: 0.7618
  precision: 0.8210
  recall: 0.9254
  f1: 0.8701
  confusion_matrix: [[2613, 1755], [649, 8050]]
  roc_auc: 0.8401
  pr_auc: 0.8758
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 463929 | 1090282 | 1287509 | 0.2985 | 0.2649 | 0.2807 | 0.0064 | 0.0078 |
  | 20ms | 666124 | 888087 | 1085314 | 0.4286 | 0.3803 | 0.4030 | 0.0098 | 0.0078 |
  | 30ms | 885952 | 668259 | 865486 | 0.5700 | 0.5058 | 0.5360 | 0.0140 | 0.0117 |
  | 40ms | 988432 | 565779 | 763006 | 0.6360 | 0.5644 | 0.5980 | 0.0164 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 384019 | -0.0000 | 0.0064 | 0.0039 | 0.0085 | 0.0001 |
  | 20ms | 576562 | -0.0001 | 0.0086 | 0.0078 | 0.0116 | 0.0001 |
  | 30ms | 787968 | -0.0001 | 0.0110 | 0.0078 | 0.0154 | 0.0002 |
  | 40ms | 886793 | -0.0001 | 0.0124 | 0.0078 | 0.0176 | 0.0003 |

## Holdout
### holdout
- windows: 29513
- peak_samples: 60442624
- rr_samples: 60442624
- gate_windows_labeled: 2916
- peak_sequence:
  n: 60442624
  threshold: 0.7750
  positive_rate: 0.1360
  accuracy: 0.8593
  balanced_accuracy: 0.7593
  precision: 0.4866
  recall: 0.6218
  f1: 0.5460
  confusion_matrix: [[46827989, 5393325], [3109194, 5112116]]
  roc_auc: 0.8704
  pr_auc: 0.5545
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 168964
  fp: 234309
  fn: 267635
  precision: 0.4190
  recall: 0.3870
  f1: 0.4024
  timing_n: 168964
  timing_bias: -0.0012
  timing_mae: 0.0102
  timing_median_ae: 0.0117
  timing_std: 0.0118
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 60442624
  mae: 0.6288
  rmse: 6.5671
  median_ae: 0.1295
  bias: -0.4124
  pearson_r: 0.0412
  r2: -0.0029
- rri_event_sequence:
  n: 145301
  bias: -0.0001
  mae: 0.0092
  median_ae: 0.0078
  std: 0.0122
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 2916
  threshold: 0.9500
  positive_rate: 0.6698
  accuracy: 0.8697
  balanced_accuracy: 0.8861
  precision: 0.9629
  recall: 0.8377
  f1: 0.8959
  confusion_matrix: [[900, 63], [317, 1636]]
  roc_auc: 0.9613
  pr_auc: 0.9810
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 113278 | 289995 | 323321 | 0.2809 | 0.2595 | 0.2698 | 0.0066 | 0.0078 |
  | 20ms | 168964 | 234309 | 267635 | 0.4190 | 0.3870 | 0.4024 | 0.0102 | 0.0117 |
  | 30ms | 232164 | 171109 | 204435 | 0.5757 | 0.5318 | 0.5529 | 0.0148 | 0.0156 |
  | 40ms | 262031 | 141242 | 174568 | 0.6498 | 0.6002 | 0.6240 | 0.0173 | 0.0156 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 92013 | -0.0000 | 0.0068 | 0.0039 | 0.0089 | 0.0001 |
  | 20ms | 145301 | -0.0001 | 0.0092 | 0.0078 | 0.0122 | 0.0001 |
  | 30ms | 206527 | -0.0001 | 0.0118 | 0.0078 | 0.0162 | 0.0003 |
  | 40ms | 235707 | -0.0001 | 0.0132 | 0.0078 | 0.0184 | 0.0003 |

## Extra Holdout
### extra_holdout
- windows: 33738
- peak_samples: 69095424
- rr_samples: 69095424
- gate_windows_labeled: 12032
- peak_sequence:
  n: 69095424
  threshold: 0.7750
  positive_rate: 0.1147
  accuracy: 0.8348
  balanced_accuracy: 0.5705
  precision: 0.2541
  recall: 0.2277
  f1: 0.2402
  confusion_matrix: [[55873076, 5297518], [6120364, 1804466]]
  roc_auc: 0.6453
  pr_auc: 0.2070
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 50009
  fp: 261251
  fn: 370782
  precision: 0.1607
  recall: 0.1188
  f1: 0.1366
  timing_n: 50009
  timing_bias: -0.0006
  timing_mae: 0.0104
  timing_median_ae: 0.0117
  timing_std: 0.0121
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 69095424
  mae: 0.1356
  rmse: 0.1787
  median_ae: 0.1069
  bias: 0.0040
  pearson_r: 0.3853
  r2: 0.1371
- rri_event_sequence:
  n: 33154
  bias: 0.0001
  mae: 0.0110
  median_ae: 0.0078
  std: 0.0143
  variance: 0.0002
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 12032
  threshold: 0.9500
  positive_rate: 0.5419
  accuracy: 0.3354
  balanced_accuracy: 0.3243
  precision: 0.4006
  recall: 0.4566
  f1: 0.4268
  confusion_matrix: [[1058, 4454], [3543, 2977]]
  roc_auc: 0.3083
  pr_auc: 0.4509
- peak_events_by_tolerance:
  | tolerance | TP | FP | FN | precision | recall | F1 | timing_MAE_s | timing_median_AE_s |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | 10ms | 32627 | 278633 | 388164 | 0.1048 | 0.0775 | 0.0891 | 0.0067 | 0.0078 |
  | 20ms | 50009 | 261251 | 370782 | 0.1607 | 0.1188 | 0.1366 | 0.0104 | 0.0117 |
  | 30ms | 72420 | 238840 | 348371 | 0.2327 | 0.1721 | 0.1979 | 0.0156 | 0.0156 |
  | 40ms | 85830 | 225430 | 334961 | 0.2758 | 0.2040 | 0.2345 | 0.0190 | 0.0195 |
- rri_event_sequence_by_tolerance:
  | tolerance | n | bias_s | MAE_s | median_AE_s | std_s | variance_s2 |
  |---|---:|---:|---:|---:|---:|---:|
  | 10ms | 19214 | -0.0001 | 0.0074 | 0.0078 | 0.0096 | 0.0001 |
  | 20ms | 33154 | 0.0001 | 0.0110 | 0.0078 | 0.0143 | 0.0002 |
  | 30ms | 52340 | 0.0000 | 0.0155 | 0.0117 | 0.0205 | 0.0004 |
  | 40ms | 64305 | -0.0000 | 0.0184 | 0.0156 | 0.0247 | 0.0006 |

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 10
- train_subject_count: 292
- val_subject_count: 73
- peak_sequence:
  n: 48250880
  threshold: 0.7750
  positive_rate: 0.1375
  accuracy: 0.8688
  balanced_accuracy: 0.7647
  precision: 0.5189
  recall: 0.6212
  f1: 0.5655
  confusion_matrix: [[37798095, 3819896], [2512526, 4120363]]
  roc_auc: 0.8837
  pr_auc: 0.5945
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 152027
  fp: 174905
  fn: 200121
  precision: 0.4650
  recall: 0.4317
  f1: 0.4477
  timing_n: 152027
  timing_bias: -0.0001
  timing_mae: 0.0096
  timing_median_ae: 0.0078
  timing_std: 0.0114
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 48250880
  mae: 0.5913
  rmse: 4.0998
  median_ae: 0.0960
  bias: -0.4600
  pearson_r: -0.0271
  r2: -0.0155
- rri_event_sequence:
  n: 133876
  bias: -0.0001
  mae: 0.0082
  median_ae: 0.0078
  std: 0.0111
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 2166
  threshold: 0.9500
  positive_rate: 0.6676
  accuracy: 0.7433
  balanced_accuracy: 0.6202
  precision: 0.7263
  recall: 0.9876
  f1: 0.8370
  confusion_matrix: [[182, 538], [18, 1428]]
  roc_auc: 0.7497
  pr_auc: 0.8026

### fold_2
- best_epoch: 2
- train_subject_count: 292
- val_subject_count: 73
- peak_sequence:
  n: 53884928
  threshold: 0.7750
  positive_rate: 0.1287
  accuracy: 0.8643
  balanced_accuracy: 0.7287
  precision: 0.4764
  recall: 0.5460
  f1: 0.5088
  confusion_matrix: [[42783778, 4163869], [3149508, 3787773]]
  roc_auc: 0.8500
  pr_auc: 0.5194
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 130149
  fp: 192180
  fn: 238201
  precision: 0.4038
  recall: 0.3533
  f1: 0.3769
  timing_n: 130149
  timing_bias: -0.0002
  timing_mae: 0.0098
  timing_median_ae: 0.0078
  timing_std: 0.0115
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 53884928
  mae: 0.1879
  rmse: 0.6242
  median_ae: 0.1345
  bias: 0.0058
  pearson_r: 0.1186
  r2: 0.0135
- rri_event_sequence:
  n: 110583
  bias: -0.0001
  mae: 0.0085
  median_ae: 0.0078
  std: 0.0115
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 5079
  threshold: 0.9500
  positive_rate: 0.6661
  accuracy: 0.8017
  balanced_accuracy: 0.7437
  precision: 0.8095
  recall: 0.9184
  f1: 0.8605
  confusion_matrix: [[965, 731], [276, 3107]]
  roc_auc: 0.8373
  pr_auc: 0.8808

### fold_3
- best_epoch: 2
- train_subject_count: 292
- val_subject_count: 73
- peak_sequence:
  n: 48635904
  threshold: 0.7750
  positive_rate: 0.1324
  accuracy: 0.8704
  balanced_accuracy: 0.7469
  precision: 0.5094
  recall: 0.5790
  f1: 0.5420
  confusion_matrix: [[38607221, 3590128], [2710783, 3727772]]
  roc_auc: 0.8594
  pr_auc: 0.5279
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 133339
  fp: 167602
  fn: 208518
  precision: 0.4431
  recall: 0.3900
  f1: 0.4149
  timing_n: 133339
  timing_bias: -0.0012
  timing_mae: 0.0099
  timing_median_ae: 0.0078
  timing_std: 0.0116
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 48635904
  mae: 2.8501
  rmse: 22.8682
  median_ae: 0.1181
  bias: -2.6879
  pearson_r: 0.1565
  r2: -0.0122
- rri_event_sequence:
  n: 116227
  bias: -0.0001
  mae: 0.0086
  median_ae: 0.0078
  std: 0.0116
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 717
  threshold: 0.9500
  positive_rate: 0.6667
  accuracy: 0.9944
  balanced_accuracy: 0.9937
  precision: 0.9958
  recall: 0.9958
  f1: 0.9958
  confusion_matrix: [[237, 2], [2, 476]]
  roc_auc: 0.9999
  pr_auc: 0.9999

### fold_4
- best_epoch: 8
- train_subject_count: 292
- val_subject_count: 73
- peak_sequence:
  n: 51300352
  threshold: 0.7750
  positive_rate: 0.1335
  accuracy: 0.8757
  balanced_accuracy: 0.7567
  precision: 0.5307
  recall: 0.5944
  f1: 0.5607
  confusion_matrix: [[40850582, 3600568], [2777986, 4071216]]
  roc_auc: 0.8820
  pr_auc: 0.5820
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 147891
  fp: 169896
  fn: 215717
  precision: 0.4654
  recall: 0.4067
  f1: 0.4341
  timing_n: 147891
  timing_bias: -0.0015
  timing_mae: 0.0097
  timing_median_ae: 0.0078
  timing_std: 0.0114
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 51300352
  mae: 1.7247
  rmse: 16.2345
  median_ae: 0.1539
  bias: -1.5071
  pearson_r: -0.0135
  r2: -0.0089
- rri_event_sequence:
  n: 129289
  bias: -0.0001
  mae: 0.0088
  median_ae: 0.0078
  std: 0.0118
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 3663
  threshold: 0.9500
  positive_rate: 0.6645
  accuracy: 0.8630
  balanced_accuracy: 0.7994
  precision: 0.8331
  recall: 0.9926
  f1: 0.9059
  confusion_matrix: [[745, 484], [18, 2416]]
  roc_auc: 0.8753
  pr_auc: 0.8985

### fold_5
- best_epoch: 6
- train_subject_count: 292
- val_subject_count: 73
- peak_sequence:
  n: 47892480
  threshold: 0.7750
  positive_rate: 0.1280
  accuracy: 0.8595
  balanced_accuracy: 0.7150
  precision: 0.4570
  recall: 0.5208
  f1: 0.4868
  confusion_matrix: [[37969228, 3793330], [2937199, 3192723]]
  roc_auc: 0.8425
  pr_auc: 0.4727
- peak_events:
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
  tp: 102718
  fp: 183504
  fn: 222757
  precision: 0.3589
  recall: 0.3156
  f1: 0.3358
  timing_n: 102718
  timing_bias: 0.0013
  timing_mae: 0.0100
  timing_median_ae: 0.0078
  timing_std: 0.0117
  timing_variance: 0.0001
- hr_interval_sequence:
  n: 47892480
  mae: 0.3079
  rmse: 2.2373
  median_ae: 0.1144
  bias: -0.1766
  pearson_r: 0.0287
  r2: -0.0068
- rri_event_sequence:
  n: 86587
  bias: 0.0000
  mae: 0.0089
  median_ae: 0.0078
  std: 0.0120
  variance: 0.0001
  tolerance_sec: 0.0200
  tolerance_ms: 20.0000
- gate_logit:
  n: 1442
  threshold: 0.9500
  positive_rate: 0.6644
  accuracy: 0.7677
  balanced_accuracy: 0.8252
  precision: 1.0000
  recall: 0.6503
  f1: 0.7881
  confusion_matrix: [[484, 0], [335, 623]]
  roc_auc: 0.9866
  pr_auc: 0.9936

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