# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: smoke_eval
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 30
- internal_holdout_subject_count: 7
- extra_holdout_subject_count: 13
- cv_folds: 2
- final_train_epochs: 1

## Thresholds
- peak_threshold: 0.7750
- gate_threshold: 0.3750

## Cross-Validation Aggregate
### cross_validation
- windows: 22040
- peak_samples: 11284480
- rr_samples: 11284480
- gate_windows_labeled: 13097
- peak_sequence:
  n: 11284480
  threshold: 0.7750
  positive_rate: 0.1289
  accuracy: 0.9090
  balanced_accuracy: 0.8161
  precision: 0.6351
  recall: 0.6910
  f1: 0.6618
  confusion_matrix: [[9252884, 577429], [449342, 1004825]]
  roc_auc: 0.9372
  pr_auc: 0.7183
- hr_interval_sequence:
  n: 11284480
  mae: 0.0712
  rmse: 0.3157
  median_ae: 0.0339
  bias: -0.0040
  pearson_r: 0.4434
  r2: 0.1962
- gate_logit:
  n: 13097
  threshold: 0.3750
  positive_rate: 0.6666
  accuracy: 0.8257
  balanced_accuracy: 0.7916
  precision: 0.8520
  recall: 0.8938
  f1: 0.8724
  confusion_matrix: [[3010, 1356], [927, 7804]]
  roc_auc: 0.8617
  pr_auc: 0.8995

## Holdout
### holdout
- windows: 4787
- peak_samples: 2450944
- rr_samples: 2450944
- gate_windows_labeled: 2887
- peak_sequence:
  n: 2450944
  threshold: 0.7750
  positive_rate: 0.1191
  accuracy: 0.9259
  balanced_accuracy: 0.8591
  precision: 0.6621
  recall: 0.7715
  f1: 0.7126
  confusion_matrix: [[2044236, 114916], [66668, 225124]]
  roc_auc: 0.9575
  pr_auc: 0.7962
- hr_interval_sequence:
  n: 2450944
  mae: 0.0679
  rmse: 0.1160
  median_ae: 0.0336
  bias: -0.0383
  pearson_r: 0.8418
  r2: 0.6704
- gate_logit:
  n: 2887
  threshold: 0.3750
  positive_rate: 0.6654
  accuracy: 0.8379
  balanced_accuracy: 0.8658
  precision: 0.9690
  recall: 0.7814
  f1: 0.8651
  confusion_matrix: [[918, 48], [420, 1501]]
  roc_auc: 0.9685
  pr_auc: 0.9772

## Extra Holdout
### extra_holdout
- windows: 12789
- peak_samples: 6547968
- rr_samples: 6547968
- gate_windows_labeled: 7683
- peak_sequence:
  n: 6547968
  threshold: 0.7750
  positive_rate: 0.1129
  accuracy: 0.8697
  balanced_accuracy: 0.5524
  precision: 0.3249
  recall: 0.1424
  f1: 0.1980
  confusion_matrix: [[5589568, 218845], [634245, 105310]]
  roc_auc: 0.6578
  pr_auc: 0.2220
- hr_interval_sequence:
  n: 6547968
  mae: 14.8569
  rmse: 62.2973
  median_ae: 0.1157
  bias: -14.6027
  pearson_r: -0.0521
  r2: -0.0585
- gate_logit:
  n: 7683
  threshold: 0.3750
  positive_rate: 0.5025
  accuracy: 0.5297
  balanced_accuracy: 0.5298
  precision: 0.5332
  recall: 0.5154
  f1: 0.5242
  confusion_matrix: [[2080, 1742], [1871, 1990]]
  roc_auc: 0.5343
  pr_auc: 0.5190

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 15
- peak_sequence:
  n: 5630976
  threshold: 0.7750
  positive_rate: 0.1295
  accuracy: 0.9034
  balanced_accuracy: 0.7991
  precision: 0.6195
  recall: 0.6583
  f1: 0.6383
  confusion_matrix: [[4607008, 294857], [249117, 479994]]
  roc_auc: 0.9252
  pr_auc: 0.6939
- hr_interval_sequence:
  n: 5630976
  mae: 0.0624
  rmse: 0.1022
  median_ae: 0.0343
  bias: 0.0297
  pearson_r: 0.8340
  r2: 0.6644
- gate_logit:
  n: 6578
  threshold: 0.3750
  positive_rate: 0.6654
  accuracy: 0.8170
  balanced_accuracy: 0.8002
  precision: 0.8711
  recall: 0.8508
  f1: 0.8608
  confusion_matrix: [[1650, 551], [653, 3724]]
  roc_auc: 0.8646
  pr_auc: 0.9089

### fold_2
- best_epoch: 1
- train_subject_count: 15
- val_subject_count: 15
- peak_sequence:
  n: 5653504
  threshold: 0.7750
  positive_rate: 0.1282
  accuracy: 0.9146
  balanced_accuracy: 0.8333
  precision: 0.6500
  recall: 0.7238
  f1: 0.6850
  confusion_matrix: [[4645876, 282572], [200225, 524831]]
  roc_auc: 0.9492
  pr_auc: 0.7429
- hr_interval_sequence:
  n: 5653504
  mae: 0.0799
  rmse: 0.4342
  median_ae: 0.0335
  bias: -0.0375
  pearson_r: 0.3712
  r2: 0.1275
- gate_logit:
  n: 6519
  threshold: 0.3750
  positive_rate: 0.6679
  accuracy: 0.8345
  balanced_accuracy: 0.7826
  precision: 0.8352
  recall: 0.9371
  f1: 0.8832
  confusion_matrix: [[1360, 805], [274, 4080]]
  roc_auc: 0.8759
  pr_auc: 0.9214

## Notes
- simultaneous_measurements gate labels are protocol-based approximations using four consecutive 5-minute task blocks: rest, walking, standing 2-back, uphill walking/running.
- iamwell currently contributes peak/IBI supervision only; its gate labels remain unavailable in this script.
- peak metrics are sequence-level metrics over the peak target sequence, not beat-matching metrics.