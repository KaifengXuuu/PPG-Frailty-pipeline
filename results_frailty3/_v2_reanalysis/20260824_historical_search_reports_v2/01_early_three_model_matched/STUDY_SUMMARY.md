# Matched historical CNN1D–InceptionTime–ShapeFormer report

> Historical post-hoc candidate evidence; not a confirmatory V2 test.

## V2-style leaderboard

| rank | config_or_model | model | subject_BA_mean_sd_percent | subject_BA_repeat_t_CI95_percent | subject_macro_F1_mean_sd_percent | subject_macro_F1_repeat_t_CI95_percent | subject_macro_ROC_AUC | ROC_AUC_applicability | scientific_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | InceptionTime | inception_time | 72.7 ± 6.0 | [65.2, 80.2] | 72.3 ± 5.9 | [65.1, 79.6] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 2 | CNN1D | cnn1d | 70.3 ± 4.5 | [64.7, 75.9] | 70.8 ± 4.8 | [64.7, 76.8] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 3 | ShapeFormer-PISD | shapeformer | 61.6 ± 4.5 | [55.9, 67.2] | 60.5 ± 5.1 | [54.2, 66.9] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |

<details><summary>Column definitions and formulas</summary>

- **rank** (`rank`): Ordinal position after applying the table's declared sorting rule. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **config_or_model** (`config_or_model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **model** (`model`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **subject_BA_mean_sd_percent** (`subject_BA_mean_sd_percent`): Compact mean and sample-standard-deviation display for Macro-average recall across the K declared classes. Formula: `display = 100 * mean +/- 100 * sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_BA_repeat_t_CI95_percent** (`subject_BA_repeat_t_CI95_percent`): Reported 95% confidence bound or interval for Macro-average recall across the K declared classes. Formula: `repeat CI95 = 100 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_F1_mean_sd_percent** (`subject_macro_F1_mean_sd_percent`): Compact mean and sample-standard-deviation display for Unweighted mean of the K class-specific F1 scores. Formula: `display = 100 * mean +/- 100 * sqrt[sum_i (x_i - mean)^2 / (n - 1)]`
- **subject_macro_F1_repeat_t_CI95_percent** (`subject_macro_F1_repeat_t_CI95_percent`): Reported 95% confidence bound or interval for Unweighted mean of the K class-specific F1 scores. Formula: `repeat CI95 = 100 * (mean +/- t_(0.975,n-1) * sample_SD / sqrt(n))`
- **subject_macro_ROC_AUC** (`subject_macro_ROC_AUC`): Area under the empirical receiver-operating-characteristic curve. Formula: `ROC-AUC = integral_0^1 TPR(FPR) dFPR (empirical trapezoidal area)`
- **ROC_AUC_applicability** (`ROC_AUC_applicability`): Direct method, applicability, reason, metric-roster, or cluster-unit provenance for the reported statistic. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`
- **scientific_role** (`scientific_role`): Direct identifier, provenance, configuration, grouping, or status value. Formula: `N/A — identifier, provenance, configuration, status, or other non-arithmetic field.`

</details>

## Conclusions

- Descriptive leader: InceptionTime with subject BA 72.7 ± 6.0% and macro-F1 72.3 ± 5.9%.
- ShapeFormer-PISD trails the leader by 11.1 BA points and 11.8 macro-F1 points.
- ShapeFormer is below both comparators on BA in all 10 matched pair/repeat rows=True and on macro-F1 in all 10 rows=True; its mean runtime is 12.3× the leader.
- This supports excluding this historical ShapeFormer-PISD implementation from the ordinary mega-study on utility/cost grounds; it does not establish that every ShapeFormer implementation is inferior.
- The exact five-repeat sign-flip P values are exploratory; only 32 sign patterns exist, so the minimum attainable two-sided P is 0.0625.
- The held-out fold supplied the historical best-epoch/early-stopping trajectory and the reported score, creating selection contamination; absolute scores are therefore candidate-generation evidence, not selection-unbiased OOF confirmation.

## Missing V2 calculations

ROC-AUC/ROC curves, PR-AUC, participant-cluster CI and formal participant-exchange P are N/A because participant-keyed OOF class probabilities were not archived. They are not reconstructed from aggregate confusion matrices.

See `STUDY_SUMMARY.html` for all paired plots and `tables/report_tables.xlsx` for one worksheet per table.
