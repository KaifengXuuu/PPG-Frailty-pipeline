# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260420-03_smoke_vital_mimic_extra
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 34
- internal_holdout_subject_count: 8
- extra_holdout_subject_count: 19
- cv_folds: 2
- final_train_epochs: 1
- norm_type: instance
- balanced_sampling: True
- augmentation_enabled: True
- domain_adversarial_lambda: 1.0000
- mimic_subsets: ['mimic_perform_train_all_csv', 'mimic_perform_test_all_csv']
- mimic_max_records: 4
- include_mimic_special_extra_holdout: True
- mimic_extra_holdout_subsets: ['mimic_perform_af_csv', 'mimic_perform_non_af_csv']
- mimic_extra_holdout_mat_files: ['MIMIC_PERform_1_min_noisy.mat', 'MIMIC_PERform_1_min_neonate.mat']
- enable_vitaldb: True
- vitaldb_max_cases: 1
- ecg_detector_preflight: True
- ecg_preflight_peak_f1: 0.9943
- ecg_preflight_ibi_accuracy: 0.9999

## Thresholds
- peak_threshold: 0.7750
- gate_threshold: 0.5250

## Cross-Validation Aggregate
### cross_validation
- windows: 11561
- peak_samples: 2959616
- rr_samples: 2959616
- gate_windows_labeled: 6616
- peak_sequence:
  n: 2959616
  threshold: 0.7750
  positive_rate: 0.1272
  accuracy: 0.8911
  balanced_accuracy: 0.7721
  precision: 0.5662
  recall: 0.6127
  f1: 0.5885
  confusion_matrix: [[2406635, 176653], [145769, 230559]]
  roc_auc: 0.8722
  pr_auc: 0.6071
- peak_events:
  tp: 57736
  fp: 11549
  fn: 18730
  precision: 0.8333
  recall: 0.7551
  f1: 0.7923
  timing_n: 57736
  timing_bias: -0.0024
  timing_mae: 0.0215
  timing_median_ae: 0.0156
  timing_std: 0.0288
  timing_variance: 0.0008
- hr_interval_sequence:
  n: 2959616
  mae: 0.2081
  rmse: 0.4005
  median_ae: 0.1507
  bias: 0.0639
  pearson_r: -0.0337
  r2: -0.2123
- rri_event_sequence:
  n: 46835
  bias: -0.0009
  mae: 0.0166
  median_ae: 0.0156
  std: 0.0274
  variance: 0.0007
- gate_logit:
  n: 6616
  threshold: 0.5250
  positive_rate: 0.6666
  accuracy: 0.7293
  balanced_accuracy: 0.6508
  precision: 0.7519
  recall: 0.8864
  f1: 0.8136
  confusion_matrix: [[916, 1290], [501, 3909]]
  roc_auc: 0.6737
  pr_auc: 0.7637

## Holdout
### holdout
- windows: 2565
- peak_samples: 656640
- rr_samples: 656640
- gate_windows_labeled: 1461
- peak_sequence:
  n: 656640
  threshold: 0.7750
  positive_rate: 0.1227
  accuracy: 0.9085
  balanced_accuracy: 0.7997
  precision: 0.6199
  recall: 0.6557
  f1: 0.6373
  confusion_matrix: [[543718, 32378], [27734, 52810]]
  roc_auc: 0.9170
  pr_auc: 0.6767
- peak_events:
  tp: 13080
  fp: 1866
  fn: 3273
  precision: 0.8752
  recall: 0.7999
  f1: 0.8358
  timing_n: 13080
  timing_bias: 0.0022
  timing_mae: 0.0201
  timing_median_ae: 0.0156
  timing_std: 0.0270
  timing_variance: 0.0007
- hr_interval_sequence:
  n: 656640
  mae: 0.1651
  rmse: 0.2127
  median_ae: 0.1310
  bias: 0.0591
  pearson_r: -0.1396
  r2: -0.5485
- rri_event_sequence:
  n: 10623
  bias: -0.0005
  mae: 0.0153
  median_ae: 0.0156
  std: 0.0246
  variance: 0.0006
- gate_logit:
  n: 1461
  threshold: 0.5250
  positive_rate: 0.6653
  accuracy: 0.8412
  balanced_accuracy: 0.8796
  precision: 0.9973
  recall: 0.7634
  f1: 0.8648
  confusion_matrix: [[487, 2], [230, 742]]
  roc_auc: 0.9916
  pr_auc: 0.9957

## Extra Holdout
### extra_holdout
- windows: 8210
- peak_samples: 2101760
- rr_samples: 2101760
- gate_windows_labeled: 6052
- peak_sequence:
  n: 2101760
  threshold: 0.7750
  positive_rate: 0.1269
  accuracy: 0.8175
  balanced_accuracy: 0.4908
  precision: 0.0975
  recall: 0.0530
  f1: 0.0686
  confusion_matrix: [[1704139, 130834], [252654, 14133]]
  roc_auc: 0.5051
  pr_auc: 0.1237
- peak_events:
  tp: 8911
  fp: 21929
  fn: 45282
  precision: 0.2889
  recall: 0.1644
  f1: 0.2096
  timing_n: 8911
  timing_bias: -0.0249
  timing_mae: 0.0525
  timing_median_ae: 0.0469
  timing_std: 0.0541
  timing_variance: 0.0029
- hr_interval_sequence:
  n: 2101760
  mae: 0.1972
  rmse: 0.2423
  median_ae: 0.1716
  bias: 0.1057
  pearson_r: -0.0455
  r2: -0.7070
- rri_event_sequence:
  n: 4969
  bias: -0.0001
  mae: 0.0364
  median_ae: 0.0156
  std: 0.0565
  variance: 0.0032
- gate_logit:
  n: 6052
  threshold: 0.5250
  positive_rate: 0.5423
  accuracy: 0.5415
  balanced_accuracy: 0.4994
  precision: 0.5420
  recall: 0.9963
  f1: 0.7021
  confusion_matrix: [[7, 2763], [12, 3270]]
  roc_auc: 0.3436
  pr_auc: 0.4752

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 1
- train_subject_count: 17
- val_subject_count: 17
- peak_sequence:
  n: 1600512
  threshold: 0.7750
  positive_rate: 0.1303
  accuracy: 0.8883
  balanced_accuracy: 0.7539
  precision: 0.5714
  recall: 0.5720
  f1: 0.5717
  confusion_matrix: [[1302539, 89469], [89242, 119262]]
  roc_auc: 0.8621
  pr_auc: 0.5957
- peak_events:
  tp: 30901
  fp: 6300
  fn: 11478
  precision: 0.8306
  recall: 0.7292
  f1: 0.7766
  timing_n: 30901
  timing_bias: 0.0008
  timing_mae: 0.0217
  timing_median_ae: 0.0156
  timing_std: 0.0297
  timing_variance: 0.0009
- hr_interval_sequence:
  n: 1600512
  mae: 0.2403
  rmse: 0.4864
  median_ae: 0.1748
  bias: 0.1116
  pearson_r: -0.0343
  r2: -0.1735
- rri_event_sequence:
  n: 24946
  bias: -0.0016
  mae: 0.0177
  median_ae: 0.0156
  std: 0.0287
  variance: 0.0008
- gate_logit:
  n: 3297
  threshold: 0.5250
  positive_rate: 0.6652
  accuracy: 0.6652
  balanced_accuracy: 0.5000
  precision: 0.6652
  recall: 1.0000
  f1: 0.7989
  confusion_matrix: [[0, 1104], [0, 2193]]
  roc_auc: 0.8128
  pr_auc: 0.8023

### fold_2
- best_epoch: 1
- train_subject_count: 17
- val_subject_count: 17
- peak_sequence:
  n: 1359104
  threshold: 0.7750
  positive_rate: 0.1235
  accuracy: 0.8943
  balanced_accuracy: 0.7950
  precision: 0.5607
  recall: 0.6632
  f1: 0.6077
  confusion_matrix: [[1104096, 87184], [56527, 111297]]
  roc_auc: 0.8861
  pr_auc: 0.6313
- peak_events:
  tp: 26835
  fp: 5249
  fn: 7252
  precision: 0.8364
  recall: 0.7873
  f1: 0.8111
  timing_n: 26835
  timing_bias: -0.0061
  timing_mae: 0.0213
  timing_median_ae: 0.0156
  timing_std: 0.0273
  timing_variance: 0.0007
- hr_interval_sequence:
  n: 1359104
  mae: 0.1703
  rmse: 0.2658
  median_ae: 0.1316
  bias: 0.0077
  pearson_r: -0.0432
  r2: -0.3939
- rri_event_sequence:
  n: 21889
  bias: -0.0001
  mae: 0.0153
  median_ae: 0.0156
  std: 0.0257
  variance: 0.0007
- gate_logit:
  n: 3319
  threshold: 0.5250
  positive_rate: 0.6680
  accuracy: 0.7930
  balanced_accuracy: 0.8026
  precision: 0.9022
  recall: 0.7740
  f1: 0.8332
  confusion_matrix: [[916, 186], [501, 1716]]
  roc_auc: 0.8880
  pr_auc: 0.9258

## Notes
- simultaneous_measurements uses .atr consensus beat annotations when available and .aux phase markers for gate supervision.
- iamwell currently contributes peak/IBI supervision only; its gate labels remain unavailable in this script.
- mimic_perform contributes PPG/ECG-derived pseudo peak/IBI supervision from local CSV files; CSV and WFDB mirrors should not be loaded together.
- mimic_perform_af_csv, mimic_perform_non_af_csv, MIMIC_PERform_1_min_noisy.mat, and MIMIC_PERform_1_min_neonate.mat are reserved for extra-holdout by default.
- vitaldb_open is optional and uses the vitaldb Python API with SNUADC/PLETH and SNUADC/ECG_II when --enable_vitaldb is set.
- ECG pseudo-label training is blocked unless the PTT ECG detector preflight passes aggregate peak F1 and IBI accuracy thresholds.
- peak_sequence is point-wise over the dense peak target; peak_events is beat-level matching with timing error statistics.
- rri_event_sequence compares matched predicted beat intervals against matched reference beat intervals.