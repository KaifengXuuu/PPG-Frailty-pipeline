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