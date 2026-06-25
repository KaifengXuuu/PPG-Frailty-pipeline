# ASA Classifier Scorecard

## Dataset
- source: VitalDB cases with ASA, PLETH, and ECG_II
- ASA classes kept: [1, 2, 3]
- ASA classes removed: 4, 6, NaN
- total_cases_after_signal_loading: 5987
- train_cases: 4789
- test_cases: 1198
- train_subjects: 4590
- test_subjects: 1154

### ASA Distribution
| split | ASA | cases | subjectids | case_percent | subject_percent |
|---|---:|---:|---:|---:|---:|
| all | 1 | 1770 | 1758 | 29.5600 | 30.6100 |
| all | 2 | 3586 | 3479 | 59.9000 | 60.5700 |
| all | 3 | 631 | 567 | 10.5400 | 9.8700 |
| train | 1 | 1416 | 1407 | 29.5700 | 30.6500 |
| train | 2 | 2868 | 2782 | 59.8900 | 60.6100 |
| train | 3 | 505 | 456 | 10.5400 | 9.9300 |
| test | 1 | 354 | 351 | 29.5500 | 30.4200 |
| test | 2 | 718 | 697 | 59.9300 | 60.4000 |
| test | 3 | 126 | 111 | 10.5200 | 9.6200 |

## ECG Peak Detector Preflight
- status: passed
- tolerance_sec: 0.0100
- min_f1: 0.9500
- precision: 0.9975
- recall: 0.9978
- f1: 0.9977
- timing_mae_sec: 0.0000

## Model Comparison
| input_mode | CV case macro F1 mean | CV case macro F1 std | test case accuracy | test case balanced accuracy | test case macro F1 | test ROC-AUC OVR macro | test PR-AUC macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppg | 0.3629 | 0.0728 | 0.4232 | 0.5099 | 0.4100 | 0.6726 | 0.4601 |
| ecg | 0.3704 | 0.0710 | 0.3114 | 0.4034 | 0.2767 | 0.5995 | 0.3962 |
| ecg_peaks | 0.3290 | 0.0151 | 0.2947 | 0.4101 | 0.2852 | 0.5854 | 0.3894 |

## ppg
### Cross-validation folds
| fold | train_cases | val_cases | best_epoch | val case accuracy | val case balanced accuracy | val case macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3831 | 958 | 12 | 0.4708 | 0.5103 | 0.4380 |
| 2 | 3831 | 958 | 8 | 0.4468 | 0.5079 | 0.4262 |
| 3 | 3831 | 958 | 1 | 0.5992 | 0.3333 | 0.2498 |
| 4 | 3831 | 958 | 11 | 0.3184 | 0.4314 | 0.3070 |
| 5 | 3832 | 957 | 10 | 0.3992 | 0.5147 | 0.3936 |

### Test case metrics
- n: 1198
- accuracy: 0.4232
- balanced_accuracy: 0.5099
- macro_f1: 0.4100
- weighted_f1: 0.4289
- roc_auc_ovr_macro: 0.6726
- pr_auc_macro: 0.4601
- confusion_matrix: [[214, 83, 57], [249, 214, 255], [10, 37, 79]]

## ecg
### Cross-validation folds
| fold | train_cases | val_cases | best_epoch | val case accuracy | val case balanced accuracy | val case macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3831 | 958 | 6 | 0.4457 | 0.4611 | 0.4069 |
| 2 | 3831 | 958 | 1 | 0.3205 | 0.4584 | 0.2768 |
| 3 | 3831 | 958 | 12 | 0.5282 | 0.4919 | 0.4626 |
| 4 | 3831 | 958 | 5 | 0.3351 | 0.4429 | 0.2980 |
| 5 | 3832 | 957 | 10 | 0.4347 | 0.4866 | 0.4075 |

### Test case metrics
- n: 1198
- accuracy: 0.3114
- balanced_accuracy: 0.4034
- macro_f1: 0.2767
- weighted_f1: 0.2531
- roc_auc_ovr_macro: 0.5995
- pr_auc_macro: 0.3962
- confusion_matrix: [[263, 20, 71], [425, 62, 231], [62, 16, 48]]

## ecg_peaks
### Cross-validation folds
| fold | train_cases | val_cases | best_epoch | val case accuracy | val case balanced accuracy | val case macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3831 | 958 | 4 | 0.3372 | 0.4540 | 0.3338 |
| 2 | 3831 | 958 | 5 | 0.3225 | 0.4264 | 0.3115 |
| 3 | 3831 | 958 | 9 | 0.3382 | 0.4696 | 0.3342 |
| 4 | 3831 | 958 | 4 | 0.3267 | 0.4629 | 0.3134 |
| 5 | 3832 | 957 | 5 | 0.3563 | 0.4667 | 0.3521 |

### Test case metrics
- n: 1198
- accuracy: 0.2947
- balanced_accuracy: 0.4101
- macro_f1: 0.2852
- weighted_f1: 0.2728
- roc_auc_ovr_macro: 0.5854
- pr_auc_macro: 0.3894
- confusion_matrix: [[196, 26, 132], [326, 87, 305], [36, 20, 70]]

## Notes
- Splits are subject-level, using subject_group derived from VitalDB subjectid.
- Holdout is created by StratifiedGroupKFold with 5 splits; the first fold is the 20% test set.
- Five-fold CV is run only on the 80% training set with StratifiedGroupKFold.
- Fold normalization statistics are fitted only on each fold's training cases and reused for validation/test.
- ecg_peaks input is a dense Gaussian peak-train sequence generated from ECG_II R-peak coordinates detected by the ECG detector validated on pulse-transit-time-ppg.
- StratifiedGroupKFold preserves class proportions as much as possible while keeping groups non-overlapping; see scikit-learn documentation.