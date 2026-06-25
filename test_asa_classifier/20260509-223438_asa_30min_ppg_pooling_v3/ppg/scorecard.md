# ASA Classifier Scorecard

## Dataset
- source: VitalDB cases with ASA, PLETH, and ECG_II
- ASA classes kept: [1, 2, 3]
- ASA classes removed before training: 4, 6, NaN
- total_cases_after_signal_loading: 5955
- train_cases: 4759
- test_cases: 1196
- train_subjects: 4566
- test_subjects: 1152

### ASA Distribution
| split | ASA | cases | subjectids | case_percent | subject_percent |
|---|---:|---:|---:|---:|---:|
| all | 1 | 1758 | 1746 | 29.5200 | 30.5400 |
| all | 2 | 3570 | 3467 | 59.9500 | 60.6300 |
| all | 3 | 627 | 564 | 10.5300 | 9.8600 |
| train | 1 | 1405 | 1396 | 29.5200 | 30.5700 |
| train | 2 | 2853 | 2771 | 59.9500 | 60.6900 |
| train | 3 | 501 | 453 | 10.5300 | 9.9200 |
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
- window/hop: 30.0000 sec / 15.0000 sec
- pooling: mean_std_topk top_k_windows=10
- loss: focal + ordinal_weight=0.3500
- balanced_sampler: True
- OOF ordinal thresholds: [1.05, 1.6500000000000001]

### Cross-validation Folds
| fold | train_cases | val_cases | best_epoch | val macro F1 | val bal acc | val QWK |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3806 | 953 | 10 | 0.2318 | 0.4515 | 0.1698 |
| 2 | 3807 | 952 | 7 | 0.1922 | 0.3977 | 0.1020 |
| 3 | 3808 | 951 | 2 | 0.2216 | 0.4337 | 0.1418 |
| 4 | 3807 | 952 | 12 | 0.2343 | 0.4543 | 0.1789 |
| 5 | 3808 | 951 | 9 | 0.2399 | 0.4539 | 0.1918 |

- CV macro F1 mean/std: 0.2240 / 0.0169
- CV balanced accuracy mean/std: 0.4382 / 0.0216

### OOF Thresholded Validation
- n: 4759
- accuracy: 0.5079
- balanced_accuracy: 0.4564
- macro_f1: 0.4415
- weighted_f1: 0.5167
- mae_asa_grade: 0.5270
- within_1_accuracy: 0.9651
- quadratic_weighted_kappa: 0.2646
- roc_auc_ovr_macro: 0.6057
- pr_auc_macro: 0.4043
- confusion_matrix: [[716, 604, 85], [904, 1541, 408], [81, 260, 160]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4209 | 0.5096 | 0.4610 | 1405.0000 |
| ASA2 | 0.6407 | 0.5401 | 0.5862 | 2853.0000 |
| ASA3 | 0.2450 | 0.3194 | 0.2773 | 501.0000 |

### Test Final Ordinal Thresholded
- n: 1196
- accuracy: 0.5426
- balanced_accuracy: 0.4728
- macro_f1: 0.4667
- weighted_f1: 0.5471
- mae_asa_grade: 0.4866
- within_1_accuracy: 0.9707
- quadratic_weighted_kappa: 0.2890
- roc_auc_ovr_macro: 0.6137
- pr_auc_macro: 0.4118
- confusion_matrix: [[184, 151, 18], [216, 427, 74], [17, 71, 38]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4412 | 0.5212 | 0.4779 | 353.0000 |
| ASA2 | 0.6579 | 0.5955 | 0.6252 | 717.0000 |
| ASA3 | 0.2923 | 0.3016 | 0.2969 | 126.0000 |

### Test Argmax Diagnostic
- n: 1196
- accuracy: 0.2349
- balanced_accuracy: 0.4576
- macro_f1: 0.2339
- weighted_f1: 0.1620
- mae_asa_grade: 0.9306
- within_1_accuracy: 0.8344
- quadratic_weighted_kappa: 0.1831
- roc_auc_ovr_macro: 0.6137
- pr_auc_macro: 0.4118
- confusion_matrix: [[168, 0, 185], [190, 0, 527], [13, 0, 113]]

| class | precision | recall | f1 | support |
|---|---:|---:|---:|---:|
| ASA1 | 0.4528 | 0.4759 | 0.4641 | 353.0000 |
| ASA2 | 0.0000 | 0.0000 | 0.0000 | 717.0000 |
| ASA3 | 0.1370 | 0.8968 | 0.2376 | 126.0000 |

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Default input now uses the first 30 minutes, split into 30s windows with 15s hop unless overridden.
- RR is derived from ECG R-peak detection; HRV features are computed from RR intervals.
- PPG branch includes detrended/bandpassed raw PPG, log spectrogram, SQI/shape/pulse-interval features depending on input_mode.
- Final predictions use OOF-tuned ordinal thresholds; argmax is reported only as a diagnostic.
- Case aggregation uses mean + std pooling over all windows plus top-k mean pooling over highest-risk windows.