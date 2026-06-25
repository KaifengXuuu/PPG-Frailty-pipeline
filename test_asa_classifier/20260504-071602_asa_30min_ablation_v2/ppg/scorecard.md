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

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Default input now uses the first 30 minutes, split into 120s windows with 60s hop unless overridden.
- RR is derived from ECG R-peak detection; HRV features are computed from RR intervals.
- PPG branch includes detrended/bandpassed raw PPG, log spectrogram, SQI/shape/pulse-interval features depending on input_mode.
- Training supports branch ablation, balanced sampling, focal loss, logit adjustment, and ordinal EMD penalty.