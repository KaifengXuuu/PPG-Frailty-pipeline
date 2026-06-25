# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260420-02_smoke_event_eval
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 30
- internal_holdout_subject_count: 7
- extra_holdout_subject_count: 13
- cv_folds: 2
- final_train_epochs: 1
- norm_type: instance
- balanced_sampling: True
- augmentation_enabled: True
- domain_adversarial_lambda: 1.0000

## Thresholds
- peak_threshold: 0.7750
- gate_threshold: 0.4750

## Cross-Validation Aggregate
### cross_validation
- windows: 11106
- peak_samples: 2843136
- rr_samples: 2843136
- gate_windows_labeled: 6618
- peak_sequence:
  n: 2843136
  threshold: 0.7750
  positive_rate: 0.1287
  accuracy: 0.9020
  balanced_accuracy: 0.8045
  precision: 0.6079
  recall: 0.6732
  f1: 0.6389
  confusion_matrix: [[2318219, 158909], [119613, 246395]]
  roc_auc: 0.9111
  pr_auc: 0.6799
- peak_events:
  tp: 61075
  fp: 7318
  fn: 13283
  precision: 0.8930
  recall: 0.8214
  f1: 0.8557
  timing_n: 61075
  timing_bias: -0.0021
  timing_mae: 0.0209
  timing_median_ae: 0.0156
  timing_std: 0.0284
  timing_variance: 0.0008
- hr_interval_sequence:
  n: 2843136
  mae: 0.1781
  rmse: 0.3759
  median_ae: 0.1148
  bias: 0.0533
  pearson_r: -0.0650
  r2: -0.1468
- rri_event_sequence:
  n: 50147
  bias: -0.0002
  mae: 0.0162
  median_ae: 0.0156
  std: 0.0270
  variance: 0.0007
- gate_logit:
  n: 6618
  threshold: 0.4750
  positive_rate: 0.6667
  accuracy: 0.7600
  balanced_accuracy: 0.6803
  precision: 0.7669
  recall: 0.9195
  f1: 0.8363
  confusion_matrix: [[973, 1233], [355, 4057]]
  roc_auc: 0.6958
  pr_auc: 0.7932

## Holdout
### holdout
- windows: 2413
- peak_samples: 617728
- rr_samples: 617728
- gate_windows_labeled: 1459
- peak_sequence:
  n: 617728
  threshold: 0.7750
  positive_rate: 0.1189
  accuracy: 0.9146
  balanced_accuracy: 0.8506
  precision: 0.6124
  recall: 0.7667
  f1: 0.6809
  confusion_matrix: [[508636, 35638], [17139, 56315]]
  roc_auc: 0.9437
  pr_auc: 0.7633
- peak_events:
  tp: 13017
  fp: 1624
  fn: 1904
  precision: 0.8891
  recall: 0.8724
  f1: 0.8807
  timing_n: 13017
  timing_bias: -0.0063
  timing_mae: 0.0185
  timing_median_ae: 0.0156
  timing_std: 0.0253
  timing_variance: 0.0006
- hr_interval_sequence:
  n: 617728
  mae: 0.1720
  rmse: 0.2199
  median_ae: 0.1365
  bias: -0.0270
  pearson_r: -0.0310
  r2: -0.1848
- rri_event_sequence:
  n: 10640
  bias: 0.0001
  mae: 0.0145
  median_ae: 0.0156
  std: 0.0237
  variance: 0.0006
- gate_logit:
  n: 1459
  threshold: 0.4750
  positive_rate: 0.6648
  accuracy: 0.6984
  balanced_accuracy: 0.7636
  precision: 0.9665
  recall: 0.5660
  f1: 0.7139
  confusion_matrix: [[470, 19], [421, 549]]
  roc_auc: 0.9210
  pr_auc: 0.9583

## Extra Holdout
### extra_holdout
- windows: 6410
- peak_samples: 1640960
- rr_samples: 1640960
- gate_windows_labeled: 6052
- peak_sequence:
  n: 1640960
  threshold: 0.7750
  positive_rate: 0.1290
  accuracy: 0.8066
  balanced_accuracy: 0.5141
  precision: 0.1623
  recall: 0.1199
  f1: 0.1379
  confusion_matrix: [[1298258, 130980], [186342, 25380]]
  roc_auc: 0.6368
  pr_auc: 0.1684
- peak_events:
  tp: 16085
  fp: 9383
  fn: 26924
  precision: 0.6316
  recall: 0.3740
  f1: 0.4698
  timing_n: 16085
  timing_bias: -0.0602
  timing_mae: 0.0660
  timing_median_ae: 0.0625
  timing_std: 0.0365
  timing_variance: 0.0013
- hr_interval_sequence:
  n: 1640960
  mae: 0.1932
  rmse: 0.2434
  median_ae: 0.1570
  bias: 0.1152
  pearson_r: -0.1419
  r2: -1.4557
- rri_event_sequence:
  n: 12207
  bias: 0.0021
  mae: 0.0187
  median_ae: 0.0156
  std: 0.0353
  variance: 0.0012
- gate_logit:
  n: 6052
  threshold: 0.4750
  positive_rate: 0.5423
  accuracy: 0.5630
  balanced_accuracy: 0.5234
  precision: 0.5542
  recall: 0.9915
  f1: 0.7110
  confusion_matrix: [[153, 2617], [28, 3254]]
  roc_auc: 0.5472
  pr_auc: 0.5747

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 15
- peak_sequence:
  n: 1418752
  threshold: 0.7750
  positive_rate: 0.1294
  accuracy: 0.8947
  balanced_accuracy: 0.7906
  precision: 0.5834
  recall: 0.6503
  f1: 0.6150
  confusion_matrix: [[1149966, 85244], [64189, 119353]]
  roc_auc: 0.8923
  pr_auc: 0.6474
- peak_events:
  tp: 29940
  fp: 4762
  fn: 7357
  precision: 0.8628
  recall: 0.8027
  f1: 0.8317
  timing_n: 29940
  timing_bias: -0.0075
  timing_mae: 0.0216
  timing_median_ae: 0.0156
  timing_std: 0.0286
  timing_variance: 0.0008
- hr_interval_sequence:
  n: 1418752
  mae: 0.1648
  rmse: 0.2241
  median_ae: 0.1142
  bias: 0.0677
  pearson_r: -0.1251
  r2: -0.6188
- rri_event_sequence:
  n: 24478
  bias: -0.0003
  mae: 0.0161
  median_ae: 0.0156
  std: 0.0273
  variance: 0.0007
- gate_logit:
  n: 3324
  threshold: 0.4750
  positive_rate: 0.6655
  accuracy: 0.6655
  balanced_accuracy: 0.5000
  precision: 0.6655
  recall: 1.0000
  f1: 0.7991
  confusion_matrix: [[0, 1112], [0, 2212]]
  roc_auc: 0.8311
  pr_auc: 0.8480

### fold_2
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 15
- peak_sequence:
  n: 1424384
  threshold: 0.7750
  positive_rate: 0.1281
  accuracy: 0.9094
  balanced_accuracy: 0.8185
  precision: 0.6330
  recall: 0.6963
  f1: 0.6631
  confusion_matrix: [[1168253, 73665], [55424, 127042]]
  roc_auc: 0.9328
  pr_auc: 0.7127
- peak_events:
  tp: 31135
  fp: 2556
  fn: 5926
  precision: 0.9241
  recall: 0.8401
  f1: 0.8801
  timing_n: 31135
  timing_bias: 0.0032
  timing_mae: 0.0202
  timing_median_ae: 0.0156
  timing_std: 0.0272
  timing_variance: 0.0007
- hr_interval_sequence:
  n: 1424384
  mae: 0.1914
  rmse: 0.4817
  median_ae: 0.1154
  bias: 0.0389
  pearson_r: -0.0512
  r2: -0.0813
- rri_event_sequence:
  n: 25669
  bias: -0.0001
  mae: 0.0163
  median_ae: 0.0156
  std: 0.0267
  variance: 0.0007
- gate_logit:
  n: 3294
  threshold: 0.4750
  positive_rate: 0.6679
  accuracy: 0.8555
  balanced_accuracy: 0.8640
  precision: 0.9385
  recall: 0.8386
  f1: 0.8857
  confusion_matrix: [[973, 121], [355, 1845]]
  roc_auc: 0.9379
  pr_auc: 0.9628

## Notes
- simultaneous_measurements uses .atr consensus beat annotations when available and .aux phase markers for gate supervision.
- iamwell currently contributes peak/IBI supervision only; its gate labels remain unavailable in this script.
- peak_sequence is point-wise over the dense peak target; peak_events is beat-level matching with timing error statistics.
- rri_event_sequence compares matched predicted beat intervals against matched reference beat intervals.