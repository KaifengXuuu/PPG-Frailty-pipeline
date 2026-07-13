# 20260630_0630 Generalization Sweep 完整图表分析报告

- 源目录：`results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2`
- 报告目录：`results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2/analysis_report_20260703_1217_complete_refs`
- 生成时间：2026-07-03T12:18
- 单次 runs：1160；参数组 configs：232 = generalization 224 + fixed references 8
- repeats 完整性：每组 min=5, max=5
- 协议：5-fold StratifiedGroupKFold；no early stopping；采样率 400 Hz；主指标为 subject-level balanced accuracy。

## 1. 核心结论

1. 最好总体配置是 `ref_20260625_top1_s1_122`，来源 `fixed_reference`，subject BA=0.623，macro F1=0.615，worst-class F1=0.521。
2. 最好的新 generalization-grid 配置是 `gen_212`，subject BA=0.581；仍低于最好 fixed reference `ref_20260625_top1_s1_122` 的 0.623。
3. 新 grid 中 BA >= 0.60 的配置数为 0，BA >= 0.73 的配置数为 0。
4. 这轮新策略没有达到 0.73，也没有超过 20260625 的 top50 SQI 参照；目前最有效的仍是 SQI top50 + full InceptionTime + 15 epoch 的 fixed reference。
5. 过拟合仍然存在：高分组 train-val BA gap 仍约 0.46-0.49，说明 window-level 训练目标仍不能很好转化为 subject-level 泛化。

## 2. Top 25 总榜

| overfit_rank | overfit_stage | overfit_config_id | reference_source | reference_source_config_id | model | cnn_epochs | stage1_regularization_value | sqi_mode | aggregation | window_sampler | windows_per_subject_per_epoch | train_overlap_pct | subject_balanced_accuracy_mean | subject_balanced_accuracy_std | subject_balanced_accuracy_ci95_low | subject_macro_f1_mean | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean | best_epoch_train_balanced_accuracy_mean | best_epoch_val_balanced_accuracy_mean | n_train_samples_per_epoch_mean | duration_sec_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fixed_reference | ref_20260625_top1_s1_122 | 20260625.000 | s1_122 | inceptiontime | 15 | s1_122 | top50_quality | quality_weighted_mean | none | all | 50.000 | 0.623 | 0.014 | 0.606 | 0.615 | 0.483 | 0.521 | 0.461 | 0.995 | 0.534 | 31328.000 | 433.602 |
| 2 | fixed_reference | ref_20260625_top2_s1_102 | 20260625.000 | s1_102 | inceptiontime | 15 | s1_102 | top50_quality | mean_prob | none | all | 50.000 | 0.618 | 0.031 | 0.579 | 0.608 | 0.483 | 0.529 | 0.464 | 0.995 | 0.531 | 31328.000 | 433.045 |
| 3 | generalization | gen_212 |  |  | small_inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.581 | 0.056 | 0.512 | 0.570 | 0.417 | 0.479 | 0.460 | 0.967 | 0.507 | 22632.000 | 94.570 |
| 4 | generalization | gen_080 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.581 | 0.037 | 0.534 | 0.569 | 0.425 | 0.476 | 0.491 | 0.981 | 0.489 | 14111.800 | 216.172 |
| 5 | generalization | gen_030 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | none | mean_prob | none | all | 30.000 | 0.577 | 0.057 | 0.506 | 0.560 | 0.389 | 0.460 | 0.475 | 0.989 | 0.513 | 45180.000 | 371.350 |
| 6 | generalization | gen_066 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.577 | 0.089 | 0.466 | 0.561 | 0.453 | 0.477 | 0.470 | 0.988 | 0.518 | 28078.400 | 367.019 |
| 7 | generalization | gen_074 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | subject_balanced | 50% | 30.000 | 0.576 | 0.106 | 0.444 | 0.558 | 0.350 | 0.400 | 0.480 | 0.982 | 0.502 | 11364.000 | 186.306 |
| 8 | generalization | gen_136 |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.567 | 0.081 | 0.466 | 0.556 | 0.392 | 0.436 | 0.432 | 0.927 | 0.495 | 14111.800 | 46.728 |
| 9 | generalization | gen_044 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.567 | 0.077 | 0.471 | 0.555 | 0.436 | 0.490 | 0.479 | 0.983 | 0.504 | 22632.000 | 208.958 |
| 10 | generalization | gen_093 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.567 | 0.075 | 0.473 | 0.535 | 0.325 | 0.384 | 0.454 | 0.980 | 0.526 | 19735.000 | 276.616 |
| 11 | generalization | gen_164 |  |  | small_inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.566 | 0.059 | 0.492 | 0.552 | 0.408 | 0.451 | 0.433 | 0.923 | 0.490 | 14111.800 | 46.708 |
| 12 | generalization | gen_121 |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.565 | 0.043 | 0.511 | 0.556 | 0.392 | 0.454 | 0.435 | 0.940 | 0.505 | 19735.000 | 58.366 |
| 13 | fixed_reference | ref_20260608_top2_s1_091 | 20260608.000 | s1_091 | inceptiontime | 10 | s1_091 | none | mean_prob | none | all | 50.000 | 0.565 | 0.056 | 0.496 | 0.547 | 0.367 | 0.429 | 0.472 | 0.989 | 0.518 | 62628.000 | 521.434 |
| 14 | generalization | gen_024 |  |  | inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.565 | 0.053 | 0.499 | 0.548 | 0.350 | 0.413 | 0.474 | 0.972 | 0.499 | 14111.800 | 147.711 |
| 15 | generalization | gen_094 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.563 | 0.080 | 0.464 | 0.546 | 0.408 | 0.450 | 0.470 | 0.987 | 0.517 | 28078.400 | 366.663 |
| 16 | generalization | gen_116 |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | subject_balanced | 50% | 30.000 | 0.563 | 0.069 | 0.478 | 0.558 | 0.392 | 0.440 | 0.445 | 0.951 | 0.506 | 22608.000 | 64.989 |
| 17 | generalization | gen_065 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.563 | 0.095 | 0.445 | 0.552 | 0.392 | 0.446 | 0.460 | 0.982 | 0.522 | 19735.000 | 276.108 |
| 18 | fixed_reference | ref_20260608_top3_s1_105 | 20260608.000 | s1_105 | inceptiontime | 10 | s1_105 | none | mean_prob | none | all | 50.000 | 0.560 | 0.043 | 0.507 | 0.551 | 0.408 | 0.458 | 0.473 | 0.992 | 0.519 | 62628.000 | 543.436 |
| 19 | generalization | gen_001 |  |  | inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | none | all | 0.000 | 0.559 | 0.055 | 0.491 | 0.549 | 0.400 | 0.451 | 0.467 | 0.986 | 0.519 | 31704.000 | 274.580 |
| 20 | generalization | gen_177 |  |  | small_inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.558 | 0.106 | 0.427 | 0.552 | 0.433 | 0.463 | 0.448 | 0.954 | 0.506 | 19735.000 | 83.753 |
| 21 | generalization | gen_198 |  |  | small_inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | none | all | 30.000 | 0.558 | 0.080 | 0.459 | 0.546 | 0.425 | 0.478 | 0.481 | 0.981 | 0.500 | 45180.000 | 168.294 |
| 22 | generalization | gen_108 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.557 | 0.018 | 0.535 | 0.544 | 0.408 | 0.454 | 0.484 | 0.978 | 0.495 | 14111.800 | 216.268 |
| 23 | generalization | gen_099 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 0.000 | 0.556 | 0.098 | 0.435 | 0.555 | 0.469 | 0.511 | 0.486 | 0.987 | 0.501 | 16044.000 | 236.606 |
| 24 | generalization | gen_051 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 0.000 | 0.556 | 0.087 | 0.449 | 0.545 | 0.375 | 0.429 | 0.463 | 0.962 | 0.499 | 9976.200 | 117.824 |
| 25 | generalization | gen_056 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 32 | 30.000 | 0.556 | 0.081 | 0.456 | 0.539 | 0.367 | 0.444 | 0.438 | 0.927 | 0.489 | 4608.000 | 79.509 |

## 3. 新 generalization grid Top 25

| overfit_rank | overfit_stage | overfit_config_id | reference_source | reference_source_config_id | model | cnn_epochs | stage1_regularization_value | sqi_mode | aggregation | window_sampler | windows_per_subject_per_epoch | train_overlap_pct | subject_balanced_accuracy_mean | subject_balanced_accuracy_std | subject_balanced_accuracy_ci95_low | subject_macro_f1_mean | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean | best_epoch_train_balanced_accuracy_mean | best_epoch_val_balanced_accuracy_mean | n_train_samples_per_epoch_mean | duration_sec_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | generalization | gen_212 |  |  | small_inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.581 | 0.056 | 0.512 | 0.570 | 0.417 | 0.479 | 0.460 | 0.967 | 0.507 | 22632.000 | 94.570 |
| 4 | generalization | gen_080 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.581 | 0.037 | 0.534 | 0.569 | 0.425 | 0.476 | 0.491 | 0.981 | 0.489 | 14111.800 | 216.172 |
| 5 | generalization | gen_030 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | none | mean_prob | none | all | 30.000 | 0.577 | 0.057 | 0.506 | 0.560 | 0.389 | 0.460 | 0.475 | 0.989 | 0.513 | 45180.000 | 371.350 |
| 6 | generalization | gen_066 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.577 | 0.089 | 0.466 | 0.561 | 0.453 | 0.477 | 0.470 | 0.988 | 0.518 | 28078.400 | 367.019 |
| 7 | generalization | gen_074 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | subject_balanced | 50% | 30.000 | 0.576 | 0.106 | 0.444 | 0.558 | 0.350 | 0.400 | 0.480 | 0.982 | 0.502 | 11364.000 | 186.306 |
| 8 | generalization | gen_136 |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.567 | 0.081 | 0.466 | 0.556 | 0.392 | 0.436 | 0.432 | 0.927 | 0.495 | 14111.800 | 46.728 |
| 9 | generalization | gen_044 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.567 | 0.077 | 0.471 | 0.555 | 0.436 | 0.490 | 0.479 | 0.983 | 0.504 | 22632.000 | 208.958 |
| 10 | generalization | gen_093 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.567 | 0.075 | 0.473 | 0.535 | 0.325 | 0.384 | 0.454 | 0.980 | 0.526 | 19735.000 | 276.616 |
| 11 | generalization | gen_164 |  |  | small_inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.566 | 0.059 | 0.492 | 0.552 | 0.408 | 0.451 | 0.433 | 0.923 | 0.490 | 14111.800 | 46.708 |
| 12 | generalization | gen_121 |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.565 | 0.043 | 0.511 | 0.556 | 0.392 | 0.454 | 0.435 | 0.940 | 0.505 | 19735.000 | 58.366 |
| 14 | generalization | gen_024 |  |  | inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.565 | 0.053 | 0.499 | 0.548 | 0.350 | 0.413 | 0.474 | 0.972 | 0.499 | 14111.800 | 147.711 |
| 15 | generalization | gen_094 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | class_subject_balanced | 50% | 30.000 | 0.563 | 0.080 | 0.464 | 0.546 | 0.408 | 0.450 | 0.470 | 0.987 | 0.517 | 28078.400 | 366.663 |
| 16 | generalization | gen_116 |  |  | small_inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | subject_balanced | 50% | 30.000 | 0.563 | 0.069 | 0.478 | 0.558 | 0.392 | 0.440 | 0.445 | 0.951 | 0.506 | 22608.000 | 64.989 |
| 17 | generalization | gen_065 |  |  | inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.563 | 0.095 | 0.445 | 0.552 | 0.392 | 0.446 | 0.460 | 0.982 | 0.522 | 19735.000 | 276.108 |
| 19 | generalization | gen_001 |  |  | inceptiontime | 10 | wd0.005_do0.5_ls0.2 | none | mean_prob | none | all | 0.000 | 0.559 | 0.055 | 0.491 | 0.549 | 0.400 | 0.451 | 0.467 | 0.986 | 0.519 | 31704.000 | 274.580 |
| 20 | generalization | gen_177 |  |  | small_inceptiontime | 15 | wd0.005_do0.5_ls0.2 | none | mean_prob | class_subject_balanced | 50% | 0.000 | 0.558 | 0.106 | 0.427 | 0.552 | 0.433 | 0.463 | 0.448 | 0.954 | 0.506 | 19735.000 | 83.753 |
| 21 | generalization | gen_198 |  |  | small_inceptiontime | 15 | wd0.01_do0.5_ls0.3 | none | mean_prob | none | all | 30.000 | 0.558 | 0.080 | 0.459 | 0.546 | 0.425 | 0.478 | 0.481 | 0.981 | 0.500 | 45180.000 | 168.294 |
| 22 | generalization | gen_108 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 30.000 | 0.557 | 0.018 | 0.535 | 0.544 | 0.408 | 0.454 | 0.484 | 0.978 | 0.495 | 14111.800 | 216.268 |
| 23 | generalization | gen_099 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 0.000 | 0.556 | 0.098 | 0.435 | 0.555 | 0.469 | 0.511 | 0.486 | 0.987 | 0.501 | 16044.000 | 236.606 |
| 24 | generalization | gen_051 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 50% | 0.000 | 0.556 | 0.087 | 0.449 | 0.545 | 0.375 | 0.429 | 0.463 | 0.962 | 0.499 | 9976.200 | 117.824 |
| 25 | generalization | gen_056 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | class_subject_balanced | 32 | 30.000 | 0.556 | 0.081 | 0.456 | 0.539 | 0.367 | 0.444 | 0.438 | 0.927 | 0.489 | 4608.000 | 79.509 |
| 26 | generalization | gen_016 |  |  | inceptiontime | 10 | wd0.005_do0.5_ls0.2 | top50_quality | mean_prob | none | all | 30.000 | 0.554 | 0.108 | 0.420 | 0.554 | 0.467 | 0.467 | 0.490 | 0.984 | 0.494 | 22632.000 | 208.988 |
| 29 | generalization | gen_155 |  |  | small_inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 0.000 | 0.550 | 0.080 | 0.451 | 0.547 | 0.400 | 0.439 | 0.443 | 0.935 | 0.492 | 16044.000 | 50.380 |
| 30 | generalization | gen_100 |  |  | inceptiontime | 15 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | none | all | 30.000 | 0.550 | 0.049 | 0.489 | 0.546 | 0.431 | 0.472 | 0.487 | 0.989 | 0.503 | 22632.000 | 307.922 |
| 31 | generalization | gen_045 |  |  | inceptiontime | 10 | wd0.01_do0.5_ls0.3 | top50_quality | mean_prob | subject_balanced | 50% | 0.000 | 0.550 | 0.037 | 0.504 | 0.541 | 0.450 | 0.474 | 0.478 | 0.962 | 0.484 | 8040.000 | 103.744 |

## 4. 所有 fixed reference 参数组成绩

| overfit_rank | overfit_stage | overfit_config_id | reference_source | reference_source_config_id | model | cnn_epochs | stage1_regularization_value | sqi_mode | aggregation | window_sampler | windows_per_subject_per_epoch | train_overlap_pct | subject_balanced_accuracy_mean | subject_balanced_accuracy_std | subject_balanced_accuracy_ci95_low | subject_macro_f1_mean | worst_class_recall_mean | worst_class_f1_mean | train_val_balanced_accuracy_gap_mean | best_epoch_train_balanced_accuracy_mean | best_epoch_val_balanced_accuracy_mean | n_train_samples_per_epoch_mean | duration_sec_mean | delta_vs_best_ref_ba |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | fixed_reference | ref_20260625_top1_s1_122 | 20260625.000 | s1_122 | inceptiontime | 15 | s1_122 | top50_quality | quality_weighted_mean | none | all | 50.000 | 0.623 | 0.014 | 0.606 | 0.615 | 0.483 | 0.521 | 0.461 | 0.995 | 0.534 | 31328.000 | 433.602 | 0.000 |
| 2 | fixed_reference | ref_20260625_top2_s1_102 | 20260625.000 | s1_102 | inceptiontime | 15 | s1_102 | top50_quality | mean_prob | none | all | 50.000 | 0.618 | 0.031 | 0.579 | 0.608 | 0.483 | 0.529 | 0.464 | 0.995 | 0.531 | 31328.000 | 433.045 | -0.006 |
| 13 | fixed_reference | ref_20260608_top2_s1_091 | 20260608.000 | s1_091 | inceptiontime | 10 | s1_091 | none | mean_prob | none | all | 50.000 | 0.565 | 0.056 | 0.496 | 0.547 | 0.367 | 0.429 | 0.472 | 0.989 | 0.518 | 62628.000 | 521.434 | -0.058 |
| 18 | fixed_reference | ref_20260608_top3_s1_105 | 20260608.000 | s1_105 | inceptiontime | 10 | s1_105 | none | mean_prob | none | all | 50.000 | 0.560 | 0.043 | 0.507 | 0.551 | 0.408 | 0.458 | 0.473 | 0.992 | 0.519 | 62628.000 | 543.436 | -0.063 |
| 27 | fixed_reference | ref_20260527_top1_g0068 | 20260527.000 | g0068 | inceptiontime | 50 | g0068 | none | mean_prob | none | all | 30.000 | 0.553 | 0.055 | 0.485 | 0.546 | 0.442 | 0.473 | 0.495 | 0.997 | 0.502 | 45180.000 | 1771.366 | -0.070 |
| 28 | fixed_reference | ref_20260527_top2_g0056 | 20260527.000 | g0056 | inceptiontime | 50 | g0056 | none | mean_prob | none | all | 50.000 | 0.551 | 0.098 | 0.430 | 0.554 | 0.458 | 0.501 | 0.490 | 0.998 | 0.508 | 62628.000 | 2438.677 | -0.072 |
| 34 | fixed_reference | ref_20260608_top4_s1_163 | 20260608.000 | s1_163 | inceptiontime | 15 | s1_163 | none | mean_prob | none | all | 50.000 | 0.548 | 0.056 | 0.479 | 0.545 | 0.450 | 0.498 | 0.475 | 0.996 | 0.521 | 62628.000 | 800.777 | -0.075 |
| 40 | fixed_reference | ref_20260608_top1_s1_085 | 20260608.000 | s1_085 | inceptiontime | 10 | s1_085 | none | mean_prob | none | all | 50.000 | 0.543 | 0.057 | 0.472 | 0.538 | 0.442 | 0.491 | 0.491 | 0.997 | 0.506 | 62628.000 | 507.089 | -0.081 |

## 5. 主效应分析：generalization grid

| factor | level | n_configs | mean_ba | median_ba | best_ba | best_ci95_low | mean_worst_f1 | best_worst_f1 | mean_gap | mean_samples_per_epoch | mean_duration_sec |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_epochs | 15 | 112 | 0.505 | 0.505 | 0.581 | 0.535 | 0.390 | 0.511 | 0.448 | 10545.907 | 116.177 |
| cnn_epochs | 10 | 112 | 0.504 | 0.506 | 0.577 | 0.511 | 0.386 | 0.490 | 0.431 | 10545.907 | 80.509 |
| model | small_inceptiontime | 112 | 0.507 | 0.506 | 0.581 | 0.523 | 0.396 | 0.479 | 0.409 | 10545.907 | 46.820 |
| model | inceptiontime | 112 | 0.502 | 0.505 | 0.581 | 0.535 | 0.379 | 0.511 | 0.470 | 10545.907 | 149.865 |
| sqi_mode | top50_quality | 112 | 0.511 | 0.511 | 0.581 | 0.535 | 0.390 | 0.511 | 0.439 | 7652.000 | 81.396 |
| sqi_mode | none | 112 | 0.498 | 0.500 | 0.577 | 0.523 | 0.386 | 0.478 | 0.440 | 13439.814 | 115.289 |
| stage1_regularization_value | wd0.005_do0.5_ls0.2 | 112 | 0.505 | 0.507 | 0.581 | 0.534 | 0.384 | 0.477 | 0.442 | 10545.907 | 98.262 |
| stage1_regularization_value | wd0.01_do0.5_ls0.3 | 112 | 0.504 | 0.505 | 0.581 | 0.535 | 0.391 | 0.511 | 0.437 | 10545.907 | 98.423 |
| train_overlap_pct | 30.0 | 112 | 0.507 | 0.506 | 0.581 | 0.535 | 0.388 | 0.490 | 0.442 | 12066.729 | 107.352 |
| train_overlap_pct | 0.0 | 112 | 0.502 | 0.505 | 0.567 | 0.511 | 0.387 | 0.511 | 0.437 | 9025.086 | 89.333 |
| window_sampler | none | 32 | 0.535 | 0.536 | 0.581 | 0.512 | 0.428 | 0.511 | 0.478 | 28890.000 | 205.337 |
| window_sampler | class_subject_balanced | 96 | 0.505 | 0.505 | 0.581 | 0.535 | 0.385 | 0.477 | 0.430 | 8295.783 | 85.254 |
| window_sampler | subject_balanced | 96 | 0.493 | 0.496 | 0.576 | 0.523 | 0.377 | 0.474 | 0.436 | 6681.333 | 75.766 |
| windows_per_subject_per_epoch | all | 32 | 0.535 | 0.536 | 0.581 | 0.512 | 0.428 | 0.511 | 0.478 | 28890.000 | 205.337 |
| windows_per_subject_per_epoch | 50% | 64 | 0.528 | 0.528 | 0.581 | 0.535 | 0.411 | 0.477 | 0.463 | 16225.675 | 131.526 |
| windows_per_subject_per_epoch | 32 | 64 | 0.496 | 0.496 | 0.556 | 0.523 | 0.380 | 0.465 | 0.437 | 4160.000 | 61.086 |
| windows_per_subject_per_epoch | 16 | 64 | 0.474 | 0.474 | 0.532 | 0.480 | 0.353 | 0.436 | 0.399 | 2080.000 | 48.918 |

## 6. 成对对照分析

| comparison | n_pairs | mean_diff_ba | median_diff_ba | positive_pairs | negative_pairs | zero_pairs | mean_diff_macro_f1 | mean_diff_worst_f1 | mean_diff_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_inceptiontime - inceptiontime | 112 | 0.005 | 0.006 | 61 | 50 | 1 | 0.009 | 0.017 | -0.061 |
| 15 - 10 epochs | 112 | 0.000 | -0.000 | 54 | 56 | 2 | -0.001 | 0.004 | 0.017 |
| top50_quality - none | 112 | 0.013 | 0.014 | 71 | 41 | 0 | 0.013 | 0.004 | -0.001 |
| wd0.01_do0.5_ls0.3 - wd0.005_do0.5_ls0.2 | 112 | -0.000 | -0.001 | 49 | 61 | 2 | -0.002 | 0.007 | -0.005 |
| train overlap 30 - 0 | 112 | 0.005 | 0.002 | 58 | 53 | 1 | 0.004 | 0.001 | 0.005 |
| class_subject_balanced - subject_balanced | 96 | 0.012 | 0.012 | 61 | 34 | 1 | 0.011 | 0.008 | -0.006 |

## 7. 图表解读

- `01_top25_overall_subject_ba.png`：前两名均为 fixed reference，都是 20260625 的 top50 SQI 配置；新 grid 最好只能到约 0.581。
- `02_all_fixed_reference_scores.png`：8 个 ref 中，20260625 两组明显领先；20260527 的旧高 BA 配置在 no early stopping/当前协议下回落到约 0.55。
- `03_fixed_reference_repeat_boxplot.png`：ref 的 repeat 变异不小，尤其 20260527 top2 的 std 接近 0.098，说明单次 repeat 最高分不能作为最终选择依据。
- `05_main_effects_generalization.png`：平均主效应看，small_inceptiontime、top50_quality、train_overlap=30、windows_per_subject=all/50% 相对较好；16/32 的强限制较弱。
- `06_heatmap_sampler_model_*.png` 与 `07_heatmap_windows_overlap_ba.png`：sampler 没有带来预期提升；class_subject_balanced 比 subject_balanced 稍好，但均不如不采样或 50% cap。
- `08_ba_vs_overfit_gap.png`：BA 较高的点并没有明显低 gap，说明目前高分并不等于过拟合问题已解决。
- `09_class_recalls_top_and_refs.png`：三类 recall 不平衡仍明显，worst-class 仍限制 BA。
- `11_all_fixed_reference_confusion_matrices.png`：包含全部 8 个 ref 的 subject-level 混淆矩阵，可以直接查看 Pre-Frail vs Robust/Non-Frail 的混淆。

## 8. 文字解释与判断

### 8.1 fixed reference 为什么领先
20260625 的两个 ref 都使用 top50 SQI，并且基于 full InceptionTime、15 epoch、50% overlap。它们在当前统一协议下仍是最好两组，说明 SQI gating 仍是最稳定的有效因素之一。
### 8.2 新 sampler 为什么没有提高
本轮 sampler 通过减少每个 subject 每个 epoch 的训练 windows 来抑制 subject/window 冗余。但结果显示，`none/all` 和 `50%` 比 16/32 更好。这说明强行限制 windows 可能丢掉了同一 subject 内的有效状态差异，导致欠拟合或信号不足。
### 8.3 small_inceptiontime 的意义
small_inceptiontime 的平均 BA 略高，训练时间显著更低，说明降低容量有一定价值；但 best BA 仍停在 0.581，说明单纯缩小模型不是突破 0.73 的充分条件。
### 8.4 下一步建议
下一步不建议继续扩大 sampler 网格。更有价值的是：保留 top50 SQI 和适度 overlap，尝试中等容量 InceptionTime、train-only 数据增强、以及 subject/file-level 或 MIL training objective。当前瓶颈更像训练目标错位，而不是常规超参数还没扫够。

## 9. 图表文件

- `figures/01_top25_overall_subject_ba.png`
- `figures/02_all_fixed_reference_scores.png`
- `figures/03_fixed_reference_repeat_boxplot.png`
- `figures/04_stage_model_distribution.png`
- `figures/05_main_effects_generalization.png`
- `figures/06_heatmap_sampler_model_ba.png`
- `figures/06_heatmap_sampler_model_gap.png`
- `figures/06_heatmap_sampler_model_worst_f1.png`
- `figures/07_heatmap_windows_overlap_ba.png`
- `figures/08_ba_vs_overfit_gap.png`
- `figures/09_class_recalls_top_and_refs.png`
- `figures/10_samples_per_epoch_vs_ba.png`
- `figures/11_all_fixed_reference_confusion_matrices.png`
- `figures/12_confusion_gen_080.png`
- `figures/12_confusion_gen_212.png`
- `figures/12_confusion_ref_20260625_top1_s1_122.png`
- `figures/12_confusion_ref_20260625_top2_s1_102.png`

## 10. CSV 表格文件

- `tables/all_fixed_reference_confusion_long.csv`
- `tables/all_fixed_references_repeat_runs.csv`
- `tables/all_fixed_references_scores.csv`
- `tables/all_fixed_references_scores_with_delta.csv`
- `tables/completeness_by_stage.csv`
- `tables/confusion_gen_080_counts.csv`
- `tables/confusion_gen_080_row_normalized.csv`
- `tables/confusion_gen_212_counts.csv`
- `tables/confusion_gen_212_row_normalized.csv`
- `tables/confusion_ref_20260527_top1_g0068_counts.csv`
- `tables/confusion_ref_20260527_top1_g0068_row_normalized.csv`
- `tables/confusion_ref_20260527_top2_g0056_counts.csv`
- `tables/confusion_ref_20260527_top2_g0056_row_normalized.csv`
- `tables/confusion_ref_20260608_top1_s1_085_counts.csv`
- `tables/confusion_ref_20260608_top1_s1_085_row_normalized.csv`
- `tables/confusion_ref_20260608_top2_s1_091_counts.csv`
- `tables/confusion_ref_20260608_top2_s1_091_row_normalized.csv`
- `tables/confusion_ref_20260608_top3_s1_105_counts.csv`
- `tables/confusion_ref_20260608_top3_s1_105_row_normalized.csv`
- `tables/confusion_ref_20260608_top4_s1_163_counts.csv`
- `tables/confusion_ref_20260608_top4_s1_163_row_normalized.csv`
- `tables/confusion_ref_20260625_top1_s1_122_counts.csv`
- `tables/confusion_ref_20260625_top1_s1_122_row_normalized.csv`
- `tables/confusion_ref_20260625_top2_s1_102_counts.csv`
- `tables/confusion_ref_20260625_top2_s1_102_row_normalized.csv`
- `tables/fixed_reference_class_metrics.csv`
- `tables/main_effects_generalization.csv`
- `tables/paired_effects_generalization.csv`
- `tables/top40_class_metrics.csv`
- `tables/top40_generalization_only.csv`
- `tables/top40_overall.csv`
