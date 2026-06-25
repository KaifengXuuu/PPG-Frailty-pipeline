# ASA Classifier Scorecard (multi-branch, ensemble)

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

## Cross-validation folds (per-fold preprocessing)
| fold | train_cases | val_cases | best_epoch | val case macro F1 | val case balanced accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 3831 | 958 | 10 | 0.4001 | 0.4130 |
| 2 | 3831 | 958 | 7 | 0.4160 | 0.4285 |
| 3 | 3831 | 958 | 11 | 0.4114 | 0.4480 |
| 4 | 3831 | 958 | 5 | 0.4083 | 0.4102 |
| 5 | 3832 | 957 | 12 | 0.4109 | 0.4660 |

- CV macro F1 mean: 0.4094
- CV macro F1 std: 0.0052

## Test (5-fold ensemble, equal-weight soft vote)
- n: 1198
- accuracy: 0.4591
- balanced_accuracy: 0.4169
- macro_f1: 0.3967
- weighted_f1: 0.4736
- roc_auc_ovr_macro: 0.5926
- pr_auc_macro: 0.3921
- confusion_matrix: [[174, 137, 43], [256, 340, 122], [24, 66, 36]]

## Notes
- 80/20 subject-level holdout via StratifiedGroupKFold (first fold = test).
- 5-fold StratifiedGroupKFold CV on the 80% training set; subjects do not leak across folds.
- Per-fold preprocessing: normalizers (PPG raw, PPG log-spec, RR resampled, HRV) are fit ONLY on each fold's train cases; class weights also fit per fold.
- Multi-branch model: PPG raw 1D-CNN + PPG STFT 2D-CNN + RR resampled 1D-CNN + HRV scalar MLP, late fusion + per-window attention pooling for case aggregation.
- Ensemble: 5 best per-fold checkpoints, equal-weight soft vote on test cases.
- ECG R-peaks reused for RR series; ECG detector validated on pulse-transit-time-ppg.