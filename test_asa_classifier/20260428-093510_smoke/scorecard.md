# ASA Classifier Scorecard

## Dataset
- source: VitalDB cases with ASA, PLETH, and ECG_II
- ASA classes kept: [1, 2, 3]
- ASA classes removed: 4, 6, NaN
- total_cases_after_signal_loading: 24
- train_cases: 19
- test_cases: 5
- train_subjects: 19
- test_subjects: 5

### ASA Distribution
| split | ASA | cases | subjectids | case_percent | subject_percent |
|---|---:|---:|---:|---:|---:|
| all | 1 | 8 | 8 | 33.3300 | 33.3300 |
| all | 2 | 8 | 8 | 33.3300 | 33.3300 |
| all | 3 | 8 | 8 | 33.3300 | 33.3300 |
| train | 1 | 6 | 6 | 31.5800 | 31.5800 |
| train | 2 | 6 | 6 | 31.5800 | 31.5800 |
| train | 3 | 7 | 7 | 36.8400 | 36.8400 |
| test | 1 | 2 | 2 | 40.0000 | 40.0000 |
| test | 2 | 2 | 2 | 40.0000 | 40.0000 |
| test | 3 | 1 | 1 | 20.0000 | 20.0000 |

## ECG Peak Detector Preflight
- status: disabled

## Model Comparison
| input_mode | CV case macro F1 mean | CV case macro F1 std | test case accuracy | test case balanced accuracy | test case macro F1 | test ROC-AUC OVR macro | test PR-AUC macro |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppg | 0.1311 | 0.0732 | 0.8000 | 0.8333 | 0.8222 | 0.8889 | 0.8611 |
| ecg | 0.1578 | 0.0347 | 0.2000 | 0.3333 | 0.1111 | 0.4722 | 0.4556 |
| ecg_peaks | 0.1578 | 0.0347 | 0.2000 | 0.1667 | 0.1111 | 0.1111 | 0.3139 |

## ppg
### Cross-validation folds
| fold | train_cases | val_cases | best_epoch | val case accuracy | val case balanced accuracy | val case macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 2 | 15 | 4 | 1 | 0.5000 | 0.3333 | 0.2222 |
| 3 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 4 | 15 | 4 | 1 | 0.2500 | 0.1667 | 0.1667 |
| 5 | 16 | 3 | 1 | 0.0000 | 0.0000 | 0.0000 |

### Test case metrics
- n: 5
- accuracy: 0.8000
- balanced_accuracy: 0.8333
- macro_f1: 0.8222
- weighted_f1: 0.7867
- roc_auc_ovr_macro: 0.8889
- pr_auc_macro: 0.8611
- confusion_matrix: [[2, 0, 0], [1, 1, 0], [0, 0, 1]]

## ecg
### Cross-validation folds
| fold | train_cases | val_cases | best_epoch | val case accuracy | val case balanced accuracy | val case macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 2 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 3 | 15 | 4 | 1 | 0.5000 | 0.3333 | 0.2222 |
| 4 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 5 | 16 | 3 | 1 | 0.3333 | 0.3333 | 0.1667 |

### Test case metrics
- n: 5
- accuracy: 0.2000
- balanced_accuracy: 0.3333
- macro_f1: 0.1111
- weighted_f1: 0.0667
- roc_auc_ovr_macro: 0.4722
- pr_auc_macro: 0.4556
- confusion_matrix: [[0, 0, 2], [0, 0, 2], [0, 0, 1]]

## ecg_peaks
### Cross-validation folds
| fold | train_cases | val_cases | best_epoch | val case accuracy | val case balanced accuracy | val case macro F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 2 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 3 | 15 | 4 | 1 | 0.2500 | 0.3333 | 0.1333 |
| 4 | 15 | 4 | 1 | 0.2500 | 0.1667 | 0.2222 |
| 5 | 16 | 3 | 1 | 0.3333 | 0.3333 | 0.1667 |

### Test case metrics
- n: 5
- accuracy: 0.2000
- balanced_accuracy: 0.1667
- macro_f1: 0.1111
- weighted_f1: 0.1333
- roc_auc_ovr_macro: 0.1111
- pr_auc_macro: 0.3139
- confusion_matrix: [[0, 2, 0], [1, 1, 0], [0, 1, 0]]

## Notes
- Splits are subject-level, using subject_group derived from VitalDB subjectid.
- Holdout is created by StratifiedGroupKFold with 5 splits; the first fold is the 20% test set.
- Five-fold CV is run only on the 80% training set with StratifiedGroupKFold.
- Fold normalization statistics are fitted only on each fold's training cases and reused for validation/test.
- ecg_peaks input is a dense Gaussian peak-train sequence generated from ECG_II R-peak coordinates detected by the ECG detector validated on pulse-transit-time-ppg.
- StratifiedGroupKFold preserves class proportions as much as possible while keeping groups non-overlapping; see scikit-learn documentation.