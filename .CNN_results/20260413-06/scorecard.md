# Peak / HR Interval / Gate Scorecard

## Configuration
- results_root: .CNN_results
- run_name: 20260413-06
- external_holdout_dataset: simultaneous_measurements
- internal_train_subject_count: 30
- internal_holdout_subject_count: 7
- extra_holdout_subject_count: 13
- cv_folds: 5
- final_train_epochs: 3

## Thresholds
- peak_threshold: 0.8000
- gate_threshold: 0.7500

## Cross-Validation Aggregate
### cross_validation
- windows: 22040
- peak_samples: 45137920
- rr_samples: 45137920
- gate_windows_labeled: 13097
- peak_sequence:
  n: 45137920
  threshold: 0.8000
  positive_rate: 0.1223
  accuracy: 0.9105
  balanced_accuracy: 0.8007
  precision: 0.6290
  recall: 0.6552
  f1: 0.6418
  confusion_matrix: [[37481056, 2134363], [1904033, 3618468]]
  roc_auc: 0.9270
  pr_auc: 0.6857
- hr_interval_sequence:
  n: 45137920
  mae: 0.0757
  rmse: 0.3168
  median_ae: 0.0452
  bias: -0.0133
  pearson_r: 0.4363
  r2: 0.1889
- gate_logit:
  n: 13097
  threshold: 0.7500
  positive_rate: 0.6666
  accuracy: 0.7677
  balanced_accuracy: 0.7574
  precision: 0.8521
  recall: 0.7885
  f1: 0.8190
  confusion_matrix: [[3171, 1195], [1847, 6884]]
  roc_auc: 0.8064
  pr_auc: 0.8391

## Holdout
### holdout
- windows: 4786
- peak_samples: 9801728
- rr_samples: 9801728
- gate_windows_labeled: 2886
- peak_sequence:
  n: 9801728
  threshold: 0.8000
  positive_rate: 0.1130
  accuracy: 0.9291
  balanced_accuracy: 0.8167
  precision: 0.6919
  recall: 0.6714
  f1: 0.6815
  confusion_matrix: [[8362977, 331158], [363908, 743685]]
  roc_auc: 0.9461
  pr_auc: 0.7612
- hr_interval_sequence:
  n: 9801728
  mae: 0.0902
  rmse: 0.1254
  median_ae: 0.0628
  bias: -0.0470
  pearson_r: 0.8275
  r2: 0.6135
- gate_logit:
  n: 2886
  threshold: 0.7500
  positive_rate: 0.6656
  accuracy: 0.8600
  balanced_accuracy: 0.8642
  precision: 0.9322
  recall: 0.8516
  f1: 0.8901
  confusion_matrix: [[846, 119], [285, 1636]]
  roc_auc: 0.9339
  pr_auc: 0.9469

## Extra Holdout
### extra_holdout
- windows: 12789
- peak_samples: 26191872
- rr_samples: 26191872
- gate_windows_labeled: 7683
- peak_sequence:
  n: 26191872
  threshold: 0.8000
  positive_rate: 0.1070
  accuracy: 0.8900
  balanced_accuracy: 0.5021
  precision: 0.1915
  recall: 0.0086
  f1: 0.0164
  confusion_matrix: [[23287972, 101246], [2778677, 23977]]
  roc_auc: 0.5592
  pr_auc: 0.1274
- hr_interval_sequence:
  n: 26191872
  mae: 13.8988
  rmse: 59.8321
  median_ae: 0.1357
  bias: -13.7009
  pearson_r: -0.0143
  r2: -0.0554
- gate_logit:
  n: 7683
  threshold: 0.7500
  positive_rate: 0.5025
  accuracy: 0.4812
  balanced_accuracy: 0.4825
  precision: 0.4656
  recall: 0.2191
  f1: 0.2980
  confusion_matrix: [[2851, 971], [3015, 846]]
  roc_auc: 0.4671
  pr_auc: 0.4898

## Cross-Validation Fold Summary
### fold_1
- best_epoch: 10
- train_subject_count: 24
- val_subject_count: 6
- peak_sequence:
  n: 8986624
  threshold: 0.8000
  positive_rate: 0.1243
  accuracy: 0.9223
  balanced_accuracy: 0.8531
  precision: 0.6636
  recall: 0.7611
  f1: 0.7090
  confusion_matrix: [[7438455, 431055], [266915, 850199]]
  roc_auc: 0.9542
  pr_auc: 0.7662
- hr_interval_sequence:
  n: 8986624
  mae: 0.0559
  rmse: 0.0790
  median_ae: 0.0434
  bias: -0.0017
  pearson_r: 0.8999
  r2: 0.8097
- gate_logit:
  n: 2184
  threshold: 0.7500
  positive_rate: 0.6644
  accuracy: 0.8965
  balanced_accuracy: 0.8928
  precision: 0.9378
  recall: 0.9042
  f1: 0.9207
  confusion_matrix: [[646, 87], [139, 1312]]
  roc_auc: 0.9621
  pr_auc: 0.9793

### fold_2
- best_epoch: 3
- train_subject_count: 24
- val_subject_count: 6
- peak_sequence:
  n: 8577024
  threshold: 0.8000
  positive_rate: 0.1174
  accuracy: 0.9120
  balanced_accuracy: 0.7922
  precision: 0.6226
  recall: 0.6357
  f1: 0.6291
  confusion_matrix: [[7182199, 388003], [366810, 640012]]
  roc_auc: 0.9130
  pr_auc: 0.6398
- hr_interval_sequence:
  n: 8577024
  mae: 0.0663
  rmse: 0.1134
  median_ae: 0.0461
  bias: -0.0037
  pearson_r: 0.6617
  r2: 0.3665
- gate_logit:
  n: 2907
  threshold: 0.7500
  positive_rate: 0.6677
  accuracy: 0.8074
  balanced_accuracy: 0.7481
  precision: 0.8126
  recall: 0.9248
  f1: 0.8651
  confusion_matrix: [[552, 414], [146, 1795]]
  roc_auc: 0.8404
  pr_auc: 0.8742

### fold_3
- best_epoch: 8
- train_subject_count: 24
- val_subject_count: 6
- peak_sequence:
  n: 9832448
  threshold: 0.8000
  positive_rate: 0.1184
  accuracy: 0.9060
  balanced_accuracy: 0.7986
  precision: 0.5929
  recall: 0.6579
  f1: 0.6237
  confusion_matrix: [[8141780, 526114], [398421, 766133]]
  roc_auc: 0.9212
  pr_auc: 0.6821
- hr_interval_sequence:
  n: 9832448
  mae: 0.0590
  rmse: 0.1048
  median_ae: 0.0367
  bias: -0.0047
  pearson_r: 0.8665
  r2: 0.7499
- gate_logit:
  n: 2901
  threshold: 0.7500
  positive_rate: 0.6674
  accuracy: 0.5826
  balanced_accuracy: 0.6872
  precision: 1.0000
  recall: 0.3745
  f1: 0.5449
  confusion_matrix: [[965, 0], [1211, 725]]
  roc_auc: 0.9425
  pr_auc: 0.9716

### fold_4
- best_epoch: 2
- train_subject_count: 24
- val_subject_count: 6
- peak_sequence:
  n: 8962048
  threshold: 0.8000
  positive_rate: 0.1063
  accuracy: 0.9376
  balanced_accuracy: 0.8086
  precision: 0.7353
  recall: 0.6448
  f1: 0.6871
  confusion_matrix: [[7788557, 221075], [338274, 614142]]
  roc_auc: 0.9469
  pr_auc: 0.7490
- hr_interval_sequence:
  n: 8962048
  mae: 0.0617
  rmse: 0.0833
  median_ae: 0.0476
  bias: -0.0063
  pearson_r: 0.5876
  r2: 0.3413
- gate_logit:
  n: 4376
  threshold: 0.7500
  positive_rate: 0.6666
  accuracy: 0.8167
  balanced_accuracy: 0.7853
  precision: 0.8505
  recall: 0.8797
  f1: 0.8648
  confusion_matrix: [[1008, 451], [351, 2566]]
  roc_auc: 0.8703
  pr_auc: 0.9246

### fold_5
- best_epoch: 3
- train_subject_count: 24
- val_subject_count: 6
- peak_sequence:
  n: 8779776
  threshold: 0.8000
  positive_rate: 0.1460
  accuracy: 0.8745
  balanced_accuracy: 0.7539
  precision: 0.5683
  recall: 0.5836
  f1: 0.5759
  confusion_matrix: [[6930065, 568116], [533613, 747982]]
  roc_auc: 0.8935
  pr_auc: 0.6059
- hr_interval_sequence:
  n: 8779776
  mae: 0.1381
  rmse: 0.6910
  median_ae: 0.0581
  bias: -0.0512
  pearson_r: 0.2910
  r2: 0.0776
- gate_logit:
  n: 729
  threshold: 0.7500
  positive_rate: 0.6667
  accuracy: 0.6667
  balanced_accuracy: 0.5000
  precision: 0.6667
  recall: 1.0000
  f1: 0.8000
  confusion_matrix: [[0, 243], [0, 486]]
  roc_auc: 0.2593
  pr_auc: 0.5604

## Notes
- simultaneous_measurements gate labels are protocol-based approximations using four consecutive 5-minute task blocks: rest, walking, standing 2-back, uphill walking/running.
- iamwell currently contributes peak/IBI supervision only; its gate labels remain unavailable in this script.
- peak metrics are sequence-level metrics over the peak target sequence, not beat-matching metrics.