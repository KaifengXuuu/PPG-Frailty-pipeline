# Generalization Sweep Analysis Report

- Source directory: `results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2`
- Generated at: 2026-07-03T12:04
- Runs: 1160 single runs
- Summary configs: 232 configs = 224 generalization configs + 8 fixed references
- Repeats per config: min 5, max 5
- Protocol: 5-fold StratifiedGroupKFold, no early stopping, fs=400 Hz according to manifest/config.

## Executive Summary

1. Best overall config: `ref_20260625_top1_s1_122` (fixed_reference), subject BA mean = 0.623, macro F1 = 0.615, worst-class F1 = 0.521.
2. Best fixed reference: `ref_20260625_top1_s1_122`, BA = 0.623.
3. Best new generalization-grid config: `gen_212`, BA = 0.581; it does not exceed the best fixed reference.
4. No generalization-grid config reaches BA >= 0.73. Generalization configs with BA >= 0.60: 0; configs matching/exceeding best fixed reference: 0.
5. The main overfitting signal remains large: top configs still show high train BA and much lower validation/subject BA.

## Top Overall Configs

| overfit_config_id | overfit_stage | reference_source | reference_source_config_id | model | cnn_epochs | stage1_regularization_value | sqi_mode | aggregation | window_sampler | windows_per_subject_per_epoch | train_overlap_pct | subject_balanced_accuracy_mean | subject_balanced_accuracy_std | subject_balanced_accuracy_ci95_low | subject_macro_f1_mean | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ref_20260625_top1_s1_122 | fixed_reference | 20260625 | s1_122 | inceptiontime | 15 | s1_122 | top50_quality | quality_weighted_mean | none | all | 50.000 | 0.623 | 0.014 | 0.606 | 0.615 | 0.483 | 0.521 | 0.461 |
| ref_20260625_top2_s1_102 | fixed_reference | 20260625 | s1_102 | inceptiontime | 15 | s1_102 | top50_quality | mean_prob | none | all | 50.000 | 0.618 | 0.031 | 0.579 | 0.608 | 0.483 | 0.529 | 0.464 |
| gen_212 | generalization |  |  | small_inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.581 | 0.056 | 0.512 | 0.570 | 0.417 | 0.479 | 0.460 |
| gen_080 | generalization |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.581 | 0.037 | 0.534 | 0.569 | 0.425 | 0.476 | 0.491 |
| gen_030 | generalization |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | none | mean_prob | none | all | 30.000 | 0.577 | 0.057 | 0.506 | 0.560 | 0.389 | 0.460 | 0.475 |
| gen_066 | generalization |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.577 | 0.089 | 0.466 | 0.561 | 0.453 | 0.477 | 0.470 |
| gen_074 | generalization |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | subject_balanced | 50% | 30.000 | 0.576 | 0.106 | 0.444 | 0.558 | 0.350 | 0.400 | 0.480 |
| gen_136 | generalization |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.567 | 0.081 | 0.466 | 0.556 | 0.392 | 0.436 | 0.432 |
| gen_044 | generalization |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.567 | 0.077 | 0.471 | 0.555 | 0.436 | 0.490 | 0.479 |
| gen_093 | generalization |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.567 | 0.075 | 0.473 | 0.535 | 0.325 | 0.384 | 0.454 |
| gen_164 | generalization |  |  | small_inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.566 | 0.059 | 0.492 | 0.552 | 0.408 | 0.451 | 0.433 |
| gen_121 | generalization |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.565 | 0.043 | 0.511 | 0.556 | 0.392 | 0.454 | 0.435 |

## Top Generalization-Only Configs

| overfit_config_id | overfit_stage | reference_source | reference_source_config_id | model | cnn_epochs | stage1_regularization_value | sqi_mode | aggregation | window_sampler | windows_per_subject_per_epoch | train_overlap_pct | subject_balanced_accuracy_mean | subject_balanced_accuracy_std | subject_balanced_accuracy_ci95_low | subject_macro_f1_mean | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen_212 | generalization |  |  | small_inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.581 | 0.056 | 0.512 | 0.570 | 0.417 | 0.479 | 0.460 |
| gen_080 | generalization |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.581 | 0.037 | 0.534 | 0.569 | 0.425 | 0.476 | 0.491 |
| gen_030 | generalization |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | none | mean_prob | none | all | 30.000 | 0.577 | 0.057 | 0.506 | 0.560 | 0.389 | 0.460 | 0.475 |
| gen_066 | generalization |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.577 | 0.089 | 0.466 | 0.561 | 0.453 | 0.477 | 0.470 |
| gen_074 | generalization |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | subject_balanced | 50% | 30.000 | 0.576 | 0.106 | 0.444 | 0.558 | 0.350 | 0.400 | 0.480 |
| gen_136 | generalization |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.567 | 0.081 | 0.466 | 0.556 | 0.392 | 0.436 | 0.432 |
| gen_044 | generalization |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.567 | 0.077 | 0.471 | 0.555 | 0.436 | 0.490 | 0.479 |
| gen_093 | generalization |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.567 | 0.075 | 0.473 | 0.535 | 0.325 | 0.384 | 0.454 |
| gen_164 | generalization |  |  | small_inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.566 | 0.059 | 0.492 | 0.552 | 0.408 | 0.451 | 0.433 |
| gen_121 | generalization |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.565 | 0.043 | 0.511 | 0.556 | 0.392 | 0.454 | 0.435 |
| gen_024 | generalization |  |  | inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.565 | 0.053 | 0.499 | 0.548 | 0.350 | 0.413 | 0.474 |
| gen_094 | generalization |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.563 | 0.080 | 0.464 | 0.546 | 0.408 | 0.450 | 0.470 |

## Fixed Reference Configs

| overfit_config_id | overfit_stage | reference_source | reference_source_config_id | model | cnn_epochs | stage1_regularization_value | sqi_mode | aggregation | window_sampler | windows_per_subject_per_epoch | train_overlap_pct | subject_balanced_accuracy_mean | subject_balanced_accuracy_std | subject_balanced_accuracy_ci95_low | subject_macro_f1_mean | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ref_20260625_top1_s1_122 | fixed_reference | 20260625 | s1_122 | inceptiontime | 15 | s1_122 | top50_quality | quality_weighted_mean | none | all | 50.000 | 0.623 | 0.014 | 0.606 | 0.615 | 0.483 | 0.521 | 0.461 |
| ref_20260625_top2_s1_102 | fixed_reference | 20260625 | s1_102 | inceptiontime | 15 | s1_102 | top50_quality | mean_prob | none | all | 50.000 | 0.618 | 0.031 | 0.579 | 0.608 | 0.483 | 0.529 | 0.464 |
| ref_20260608_top2_s1_091 | fixed_reference | 20260608 | s1_091 | inceptiontime | 10 | s1_091 | none | mean_prob | none | all | 50.000 | 0.565 | 0.056 | 0.496 | 0.547 | 0.367 | 0.429 | 0.472 |
| ref_20260608_top3_s1_105 | fixed_reference | 20260608 | s1_105 | inceptiontime | 10 | s1_105 | none | mean_prob | none | all | 50.000 | 0.560 | 0.043 | 0.507 | 0.551 | 0.408 | 0.458 | 0.473 |
| ref_20260527_top1_g0068 | fixed_reference | 20260527 | g0068 | inceptiontime | 50 | g0068 | none | mean_prob | none | all | 30.000 | 0.553 | 0.055 | 0.485 | 0.546 | 0.442 | 0.473 | 0.495 |
| ref_20260527_top2_g0056 | fixed_reference | 20260527 | g0056 | inceptiontime | 50 | g0056 | none | mean_prob | none | all | 50.000 | 0.551 | 0.098 | 0.430 | 0.554 | 0.458 | 0.501 | 0.490 |
| ref_20260608_top4_s1_163 | fixed_reference | 20260608 | s1_163 | inceptiontime | 15 | s1_163 | none | mean_prob | none | all | 50.000 | 0.548 | 0.056 | 0.479 | 0.545 | 0.450 | 0.498 | 0.475 |
| ref_20260608_top1_s1_085 | fixed_reference | 20260608 | s1_085 | inceptiontime | 10 | s1_085 | none | mean_prob | none | all | 50.000 | 0.543 | 0.057 | 0.472 | 0.538 | 0.442 | 0.491 | 0.491 |

## Main Effects on Generalization Grid

| factor | level | n_configs | mean_ba | best_ba | best_ci_low | mean_worst_f1 | best_worst_f1 | mean_gap | mean_samples_per_epoch | mean_duration_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_epochs | 15 | 112 | 0.505 | 0.581 | 0.535 | 0.390 | 0.511 | 0.448 | 10545.907 | 116.177 |
| cnn_epochs | 10 | 112 | 0.504 | 0.577 | 0.511 | 0.386 | 0.490 | 0.431 | 10545.907 | 80.509 |
| model | small_inceptiontime | 112 | 0.507 | 0.581 | 0.523 | 0.396 | 0.479 | 0.409 | 10545.907 | 46.820 |
| model | inceptiontime | 112 | 0.502 | 0.581 | 0.535 | 0.379 | 0.511 | 0.470 | 10545.907 | 149.865 |
| sqi_mode | top50_quality | 112 | 0.511 | 0.581 | 0.535 | 0.390 | 0.511 | 0.439 | 7652.000 | 81.396 |
| sqi_mode | none | 112 | 0.498 | 0.577 | 0.523 | 0.386 | 0.478 | 0.440 | 13439.814 | 115.289 |
| stage1_regularization_value | wd0.005_do0.5_ls0.2 | 112 | 0.505 | 0.581 | 0.534 | 0.384 | 0.477 | 0.442 | 10545.907 | 98.262 |
| stage1_regularization_value | wd0.01_do0.5_ls0.3 | 112 | 0.504 | 0.581 | 0.535 | 0.391 | 0.511 | 0.437 | 10545.907 | 98.423 |
| train_overlap_pct | 30.000 | 112 | 0.507 | 0.581 | 0.535 | 0.388 | 0.490 | 0.442 | 12066.729 | 107.352 |
| train_overlap_pct | 0.000 | 112 | 0.502 | 0.567 | 0.511 | 0.387 | 0.511 | 0.437 | 9025.086 | 89.333 |
| window_sampler | none | 32 | 0.535 | 0.581 | 0.512 | 0.428 | 0.511 | 0.478 | 28890.000 | 205.337 |
| window_sampler | class_subject_balanced | 96 | 0.505 | 0.581 | 0.535 | 0.385 | 0.477 | 0.430 | 8295.783 | 85.254 |
| window_sampler | subject_balanced | 96 | 0.493 | 0.576 | 0.523 | 0.377 | 0.474 | 0.436 | 6681.333 | 75.766 |
| windows_per_subject_per_epoch | all | 32 | 0.535 | 0.581 | 0.512 | 0.428 | 0.511 | 0.478 | 28890.000 | 205.337 |
| windows_per_subject_per_epoch | 50% | 64 | 0.528 | 0.581 | 0.535 | 0.411 | 0.477 | 0.463 | 16225.675 | 131.526 |
| windows_per_subject_per_epoch | 32 | 64 | 0.496 | 0.556 | 0.523 | 0.380 | 0.465 | 0.437 | 4160.000 | 61.086 |
| windows_per_subject_per_epoch | 16 | 64 | 0.474 | 0.532 | 0.480 | 0.353 | 0.436 | 0.399 | 2080.000 | 48.918 |

## Paired Effects

Positive `mean_diff_ba` means the right side of the comparison is better.

| comparison | n_pairs | mean_diff_ba | median_diff_ba | positive_pairs | negative_pairs |
| --- | --- | --- | --- | --- | --- |
| small_inceptiontime - inceptiontime | 112 | 0.005 | 0.006 | 61 | 50 |
| 15 - 10 | 112 | 0.000 | -0.000 | 54 | 56 |
| top50_quality - none | 112 | 0.013 | 0.014 | 71 | 41 |
| wd0.01_do0.5_ls0.3 - wd0.005_do0.5_ls0.2 | 112 | -0.000 | -0.001 | 49 | 61 |
| 30.0 - 0.0 | 112 | 0.005 | 0.002 | 58 | 53 |
| class_subject_balanced - subject_balanced | 96 | 0.012 | 0.012 | 61 | 34 |

## Interpretation Notes

- `small_inceptiontime` improves mean generalization relative to full InceptionTime in this sweep. This supports the hypothesis that the full model is too large for the current subject count.
- SQI `top50_quality` is still useful. The best fixed references are top50 SQI configs, and the best new generalization-grid config also uses top50 quality.
- The sampler changes did not solve the problem in this run. The best new config uses `window_sampler=none`; subject-balanced and class-subject-balanced sampling reduce samples per epoch but do not clearly increase BA.
- `train_overlap_pct=30` tends to be better than 0 in this implementation. This suggests some overlapped windows are useful, despite the redundancy risk.
- Worst-class F1/recall remain low relative to overall BA, so class balance remains a bottleneck.

## Figures

- `figures/01_top20_subject_ba.png`
- `figures/02_ba_distribution_stage.png`
- `figures/03_main_effects_generalization.png`
- `figures/04_heatmap_sampler_model_ba.png`
- `figures/04_heatmap_sampler_model_train_val_gap.png`
- `figures/04_heatmap_sampler_model_worst_f1.png`
- `figures/05_ba_vs_train_val_gap.png`
- `figures/06_top_class_recalls.png`
- `figures/07_confusion_gen_080.png`
- `figures/07_confusion_gen_212.png`
- `figures/07_confusion_ref_20260625_top1_s1_122.png`
- `figures/07_confusion_ref_20260625_top2_s1_102.png`
- `figures/08_runtime_model_sampler.png`
- `figures/09_samples_per_epoch_vs_ba.png`

## Suggested Next Steps

1. Keep `small_inceptiontime + top50_quality + train_overlap=30` as the strongest new direction.
2. Do not expand sampler search blindly; first inspect whether subject-balanced sampling reduces useful variability too aggressively.
3. Try a middle-capacity InceptionTime variant between full and small.
4. Add train-only augmentation to small InceptionTime: amplitude scaling, mild Gaussian noise, channel dropout, time shift.
5. Consider subject-level/MIL training objective next, because window-level CE still overfits strongly.
