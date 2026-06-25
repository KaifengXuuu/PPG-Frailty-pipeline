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

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Default input now uses the first 30 minutes, split into 120s windows with 60s hop unless overridden.
- RR is derived from ECG R-peak detection; HRV features are computed from RR intervals.
- PPG branch includes detrended/bandpassed raw PPG, log spectrogram, SQI/shape/pulse-interval features depending on input_mode.
- Training supports branch ablation, balanced sampling, focal loss, logit adjustment, and ordinal EMD penalty.