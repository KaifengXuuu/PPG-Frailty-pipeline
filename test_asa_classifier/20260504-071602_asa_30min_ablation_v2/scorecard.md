# ASA Classifier Scorecard

## Dataset
- source: VitalDB cases with ASA, PLETH, and ECG_II
- ASA classes kept: [1, 2, 3]
- ASA classes removed before training: 4, 6, NaN
- total_cases_after_signal_loading: 5946
- train_cases: 4750
- test_cases: 1196
- train_subjects: 4558
- test_subjects: 1152

### ASA Distribution
| split | ASA | cases | subjectids | case_percent | subject_percent |
|---|---:|---:|---:|---:|---:|
| all | 1 | 1755 | 1743 | 29.5200 | 30.5300 |
| all | 2 | 3565 | 3462 | 59.9600 | 60.6300 |
| all | 3 | 626 | 564 | 10.5300 | 9.8800 |
| train | 1 | 1402 | 1393 | 29.5200 | 30.5600 |
| train | 2 | 2848 | 2766 | 59.9600 | 60.6800 |
| train | 3 | 500 | 453 | 10.5300 | 9.9400 |
| test | 1 | 353 | 350 | 29.5200 | 30.3800 |
| test | 2 | 717 | 696 | 59.9500 | 60.4200 |
| test | 3 | 126 | 111 | 10.5400 | 9.6400 |

## ECG Peak Detector Preflight
- status: passed
- tolerance_sec: 0.0100
- min_f1: 0.9500
- precision: 0.9975
- recall: 0.9978
- f1: 0.9977
- timing_mae_sec: 0.0000

## Model Comparison
| input_mode | CV macro F1 | test macro F1 | test bal acc | test QWK | threshold macro F1 | threshold bal acc |
|---|---:|---:|---:|---:|---:|---:|
| ppg | 0.2068 | 0.2096 | 0.4302 | 0.1391 | 0.4619 | 0.4569 |
| ppg_spec | 0.2187 | 0.2250 | 0.4510 | 0.1680 | 0.4550 | 0.4523 |
| ppg_rr | 0.2258 | 0.2334 | 0.4595 | 0.1811 | 0.4384 | 0.4389 |
| full | 0.2150 | 0.2265 | 0.4538 | 0.1710 | 0.4501 | 0.4471 |

## Model: ppg
- signal span: first 1800.0000 sec
- window/hop: 120.0000 sec / 60.0000 sec
- loss: focal + ordinal_weight=0.3500
- balanced_sampler: True
- mean ordinal thresholds: [1.0799999237060547, 1.7700001001358032]

### Cross-validation Folds
| fold | train_cases | val_cases | best_epoch | val macro F1 | val bal acc | val QWK |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3800 | 950 | 10 | 0.1989 | 0.4212 | 0.1254 |
| 2 | 3800 | 950 | 12 | 0.1979 | 0.4199 | 0.1209 |
| 3 | 3800 | 950 | 6 | 0.2132 | 0.4214 | 0.1374 |
| 4 | 3800 | 950 | 2 | 0.2243 | 0.4330 | 0.1569 |
| 5 | 3800 | 950 | 12 | 0.1997 | 0.4124 | 0.1108 |

- CV macro F1 mean/std: 0.2068 / 0.0104
- CV balanced accuracy mean/std: 0.4216 / 0.0066

### Test Argmax
- n: 1196
- accuracy: 0.2032
- balanced_accuracy: 0.4302
- macro_f1: 0.2096
- weighted_f1: 0.1433
- mae_asa_grade: 0.9941
- within_1_accuracy: 0.8027
- quadratic_weighted_kappa: 0.1391
- roc_auc_ovr_macro: 0.6092
- pr_auc_macro: 0.4150
- confusion_matrix: [[125, 0, 228], [130, 0, 587], [8, 0, 118]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4753 | 0.3541 | 0.4058 | 353.0000 |
| ASA2 | 0.0000 | 0.0000 | 0.0000 | 717.0000 |
| ASA3 | 0.1265 | 0.9365 | 0.2229 | 126.0000 |

### Test Thresholded Ordinal Score
- n: 1196
- accuracy: 0.5661
- balanced_accuracy: 0.4569
- macro_f1: 0.4619
- weighted_f1: 0.5606
- mae_asa_grade: 0.4574
- within_1_accuracy: 0.9766
- quadratic_weighted_kappa: 0.2744
- roc_auc_ovr_macro: 0.6092
- pr_auc_macro: 0.4150
- confusion_matrix: [[151, 185, 17], [163, 494, 60], [11, 83, 32]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4646 | 0.4278 | 0.4454 | 353.0000 |
| ASA2 | 0.6483 | 0.6890 | 0.6680 | 717.0000 |
| ASA3 | 0.2936 | 0.2540 | 0.2723 | 126.0000 |

## Model: ppg_spec
- signal span: first 1800.0000 sec
- window/hop: 120.0000 sec / 60.0000 sec
- loss: focal + ordinal_weight=0.3500
- balanced_sampler: True
- mean ordinal thresholds: [1.0200001001358032, 1.7299998998641968]

### Cross-validation Folds
| fold | train_cases | val_cases | best_epoch | val macro F1 | val bal acc | val QWK |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3800 | 950 | 12 | 0.2319 | 0.4595 | 0.1787 |
| 2 | 3800 | 950 | 8 | 0.2241 | 0.4442 | 0.1560 |
| 3 | 3800 | 950 | 4 | 0.2116 | 0.4107 | 0.1346 |
| 4 | 3800 | 950 | 10 | 0.2019 | 0.4179 | 0.1257 |
| 5 | 3800 | 950 | 11 | 0.2242 | 0.4474 | 0.1513 |

- CV macro F1 mean/std: 0.2187 / 0.0107
- CV balanced accuracy mean/std: 0.4360 / 0.0185

### Test Argmax
- n: 1196
- accuracy: 0.2216
- balanced_accuracy: 0.4510
- macro_f1: 0.2250
- weighted_f1: 0.1548
- mae_asa_grade: 0.9574
- within_1_accuracy: 0.8211
- quadratic_weighted_kappa: 0.1680
- roc_auc_ovr_macro: 0.6170
- pr_auc_macro: 0.4222
- confusion_matrix: [[147, 0, 206], [159, 0, 558], [8, 0, 118]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4682 | 0.4164 | 0.4408 | 353.0000 |
| ASA2 | 0.0000 | 0.0000 | 0.0000 | 717.0000 |
| ASA3 | 0.1338 | 0.9365 | 0.2341 | 126.0000 |

### Test Thresholded Ordinal Score
- n: 1196
- accuracy: 0.5543
- balanced_accuracy: 0.4523
- macro_f1: 0.4550
- weighted_f1: 0.5515
- mae_asa_grade: 0.4657
- within_1_accuracy: 0.9799
- quadratic_weighted_kappa: 0.2874
- roc_auc_ovr_macro: 0.6170
- pr_auc_macro: 0.4222
- confusion_matrix: [[155, 183, 15], [170, 476, 71], [9, 85, 32]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4641 | 0.4391 | 0.4512 | 353.0000 |
| ASA2 | 0.6398 | 0.6639 | 0.6516 | 717.0000 |
| ASA3 | 0.2712 | 0.2540 | 0.2623 | 126.0000 |

## Model: ppg_rr
- signal span: first 1800.0000 sec
- window/hop: 120.0000 sec / 60.0000 sec
- loss: focal + ordinal_weight=0.3500
- balanced_sampler: True
- mean ordinal thresholds: [1.0099999904632568, 1.7400000095367432]

### Cross-validation Folds
| fold | train_cases | val_cases | best_epoch | val macro F1 | val bal acc | val QWK |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3800 | 950 | 7 | 0.2312 | 0.4588 | 0.1804 |
| 2 | 3800 | 950 | 10 | 0.2241 | 0.4416 | 0.1595 |
| 3 | 3800 | 950 | 7 | 0.2358 | 0.4436 | 0.1791 |
| 4 | 3800 | 950 | 8 | 0.2256 | 0.4395 | 0.1626 |
| 5 | 3800 | 950 | 10 | 0.2126 | 0.4248 | 0.1325 |

- CV macro F1 mean/std: 0.2258 / 0.0079
- CV balanced accuracy mean/std: 0.4416 / 0.0108

### Test Argmax
- n: 1196
- accuracy: 0.2366
- balanced_accuracy: 0.4595
- macro_f1: 0.2334
- weighted_f1: 0.1606
- mae_asa_grade: 0.9273
- within_1_accuracy: 0.8361
- quadratic_weighted_kappa: 0.1811
- roc_auc_ovr_macro: 0.6082
- pr_auc_macro: 0.4051
- confusion_matrix: [[170, 0, 183], [207, 0, 510], [13, 0, 113]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4359 | 0.4816 | 0.4576 | 353.0000 |
| ASA2 | 0.0000 | 0.0000 | 0.0000 | 717.0000 |
| ASA3 | 0.1402 | 0.8968 | 0.2425 | 126.0000 |

### Test Thresholded Ordinal Score
- n: 1196
- accuracy: 0.5334
- balanced_accuracy: 0.4389
- macro_f1: 0.4384
- weighted_f1: 0.5335
- mae_asa_grade: 0.4875
- within_1_accuracy: 0.9791
- quadratic_weighted_kappa: 0.2765
- roc_auc_ovr_macro: 0.6082
- pr_auc_macro: 0.4051
- confusion_matrix: [[174, 167, 12], [211, 437, 69], [13, 86, 27]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4372 | 0.4929 | 0.4634 | 353.0000 |
| ASA2 | 0.6333 | 0.6095 | 0.6212 | 717.0000 |
| ASA3 | 0.2500 | 0.2143 | 0.2308 | 126.0000 |

## Model: full
- signal span: first 1800.0000 sec
- window/hop: 120.0000 sec / 60.0000 sec
- loss: focal + ordinal_weight=0.3500
- balanced_sampler: True
- mean ordinal thresholds: [0.9599999189376831, 1.649999976158142]

### Cross-validation Folds
| fold | train_cases | val_cases | best_epoch | val macro F1 | val bal acc | val QWK |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3800 | 950 | 1 | 0.2071 | 0.4202 | 0.1299 |
| 2 | 3800 | 950 | 1 | 0.2119 | 0.4362 | 0.1342 |
| 3 | 3800 | 950 | 12 | 0.2148 | 0.4245 | 0.1407 |
| 4 | 3800 | 950 | 12 | 0.2378 | 0.4596 | 0.1897 |
| 5 | 3800 | 950 | 12 | 0.2036 | 0.4116 | 0.1093 |

- CV macro F1 mean/std: 0.2150 / 0.0120
- CV balanced accuracy mean/std: 0.4304 / 0.0166

### Test Argmax
- n: 1196
- accuracy: 0.2241
- balanced_accuracy: 0.4538
- macro_f1: 0.2265
- weighted_f1: 0.1557
- mae_asa_grade: 0.9523
- within_1_accuracy: 0.8236
- quadratic_weighted_kappa: 0.1710
- roc_auc_ovr_macro: 0.6173
- pr_auc_macro: 0.4160
- confusion_matrix: [[150, 0, 203], [166, 0, 551], [8, 0, 118]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4630 | 0.4249 | 0.4431 | 353.0000 |
| ASA2 | 0.0000 | 0.0000 | 0.0000 | 717.0000 |
| ASA3 | 0.1353 | 0.9365 | 0.2365 | 126.0000 |

### Test Thresholded Ordinal Score
- n: 1196
- accuracy: 0.5543
- balanced_accuracy: 0.4471
- macro_f1: 0.4501
- weighted_f1: 0.5458
- mae_asa_grade: 0.4649
- within_1_accuracy: 0.9808
- quadratic_weighted_kappa: 0.2682
- roc_auc_ovr_macro: 0.6173
- pr_auc_macro: 0.4160
- confusion_matrix: [[126, 211, 16], [143, 501, 73], [7, 83, 36]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4565 | 0.3569 | 0.4006 | 353.0000 |
| ASA2 | 0.6302 | 0.6987 | 0.6627 | 717.0000 |
| ASA3 | 0.2880 | 0.2857 | 0.2869 | 126.0000 |

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Default input now uses the first 30 minutes, split into 120s windows with 60s hop unless overridden.
- RR is derived from ECG R-peak detection; HRV features are computed from RR intervals.
- PPG branch includes detrended/bandpassed raw PPG, log spectrogram, SQI/shape/pulse-interval features depending on input_mode.
- Training supports branch ablation, balanced sampling, focal loss, logit adjustment, and ordinal EMD penalty.