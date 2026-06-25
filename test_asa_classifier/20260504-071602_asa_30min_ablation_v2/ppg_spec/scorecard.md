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

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Default input now uses the first 30 minutes, split into 120s windows with 60s hop unless overridden.
- RR is derived from ECG R-peak detection; HRV features are computed from RR intervals.
- PPG branch includes detrended/bandpassed raw PPG, log spectrogram, SQI/shape/pulse-interval features depending on input_mode.
- Training supports branch ablation, balanced sampling, focal loss, logit adjustment, and ordinal EMD penalty.