# ASA Classifier Scorecard (multi-branch, ensemble)

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
| train | 1 | 7 | 7 | 36.8400 | 36.8400 |
| train | 2 | 7 | 7 | 36.8400 | 36.8400 |
| train | 3 | 5 | 5 | 26.3200 | 26.3200 |
| test | 1 | 1 | 1 | 20.0000 | 20.0000 |
| test | 2 | 1 | 1 | 20.0000 | 20.0000 |
| test | 3 | 3 | 3 | 60.0000 | 60.0000 |

## ECG Peak Detector Preflight
- status: disabled

## Cross-validation folds (per-fold preprocessing)
| fold | train_cases | val_cases | best_epoch | val case macro F1 | val case balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 15 | 4 | 1 | 0.3889 | 0.5000 |
| 2 | 15 | 4 | 1 | 0.2222 | 0.2500 |
| 3 | 15 | 4 | 2 | 0.1333 | 0.2500 |
| 4 | 16 | 3 | 1 | 0.0000 | 0.0000 |
| 5 | 15 | 4 | 2 | 0.1333 | 0.2500 |

- CV macro F1 mean: 0.1756
- CV macro F1 std: 0.1281

## Test (5-fold ensemble, equal-weight soft vote)
- n: 5
- accuracy: 0.6000
- balanced_accuracy: 0.3333
- macro_f1: 0.2500
- weighted_f1: 0.4500
- roc_auc_ovr_macro: 0.2778
- pr_auc_macro: 0.4111
- confusion_matrix: [[0, 0, 1], [0, 0, 1], [0, 0, 3]]

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Per-fold preprocessing: normalizers (PPG raw, PPG log-spec, RR resampled, HRV) are fit ONLY on each fold's train cases; class weights also fit per fold.
- Multi-branch model: PPG raw 1D-CNN + PPG STFT 2D-CNN + RR resampled 1D-CNN + HRV scalar MLP, late fusion + per-window attention pooling for case aggregation.
- Ensemble: 5 best per-fold checkpoints, equal-weight soft vote on test cases.
- ECG R-peaks reused for RR series; ECG detector validated on pulse-transit-time-ppg.