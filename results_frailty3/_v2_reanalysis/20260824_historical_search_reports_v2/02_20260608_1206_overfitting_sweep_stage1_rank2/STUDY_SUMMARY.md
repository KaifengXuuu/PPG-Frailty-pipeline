# Historical V2-style report: 20260608_1206_overfitting_sweep_stage1_rank2

> Historical post-hoc search evidence; not a confirmatory V2 test.

## Top-15 V2-style leaderboard

| rank | config_or_model | model | subject_BA_mean_sd_percent | subject_BA_repeat_t_CI95_percent | subject_macro_F1_mean_sd_percent | subject_macro_F1_repeat_t_CI95_percent | subject_macro_ROC_AUC | ROC_AUC_applicability | scientific_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | s1_085 | inception_time | 62.3 ± 7.0 | [53.6, 71.0] | 62.6 ± 7.7 | [53.0, 72.1] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 2 | s1_091 | inception_time | 62.1 ± 5.1 | [55.8, 68.5] | 62.2 ± 4.1 | [57.1, 67.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 3 | s1_105 | inception_time | 61.6 ± 3.8 | [56.9, 66.3] | 62.5 ± 3.9 | [57.7, 67.4] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 4 | s1_163 | inception_time | 61.2 ± 3.1 | [57.4, 65.0] | 61.7 ± 2.9 | [58.1, 65.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 5 | s1_079 | inception_time | 60.6 ± 4.0 | [55.6, 65.7] | 61.3 ± 3.7 | [56.7, 65.8] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 6 | s1_075 | inception_time | 60.5 ± 5.3 | [53.8, 67.1] | 61.2 ± 5.7 | [54.2, 68.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 7 | s1_077 | inception_time | 60.4 ± 6.1 | [52.8, 67.9] | 61.2 ± 6.7 | [52.8, 69.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 8 | s1_165 | inception_time | 60.4 ± 3.9 | [55.5, 65.3] | 60.0 ± 3.5 | [55.7, 64.4] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 9 | s1_099 | inception_time | 60.3 ± 8.4 | [49.8, 70.8] | 61.0 ± 8.7 | [50.2, 71.8] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 10 | s1_183 | inception_time | 60.3 ± 4.6 | [54.6, 66.0] | 59.9 ± 4.0 | [55.0, 64.8] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 11 | s1_181 | inception_time | 60.3 ± 6.2 | [52.6, 68.0] | 59.3 ± 7.4 | [50.1, 68.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 12 | s1_160 | inception_time | 60.1 ± 4.2 | [54.9, 65.3] | 60.7 ± 3.8 | [56.0, 65.4] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 13 | s1_178 | inception_time | 60.1 ± 4.1 | [55.0, 65.2] | 59.9 ± 5.3 | [53.3, 66.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 14 | s1_148 | inception_time | 60.1 ± 8.0 | [50.1, 70.0] | 58.6 ± 8.0 | [48.7, 68.6] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 15 | s1_158 | inception_time | 60.0 ± 7.0 | [51.3, 68.7] | 60.8 ± 6.9 | [52.2, 69.4] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |

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

## Conclusions and writing role

- Point leader: s1_085, subject BA 62.3 ± 7.0% and macro-F1 62.6 ± 7.7%.
- Runner-up: s1_091, subject BA 62.1 ± 5.1%; the same archive searched 186 configurations, so neither row is an unbiased final estimate.
- Writing role: motivate fixed-epoch and regularization hypotheses for later matched V2 tests
- Parameter-value boxplots are descriptive marginal views. A parameter can appear favorable because other settings differ; only later matched V2 ablations can support a component-effect claim.
- Repeat Student-t CI95 is available; participant-cluster CI, formal V2 P, ROC-AUC and ROC curves are N/A because participant-keyed OOF probabilities were not archived.
- Design dependency — multi_parameter_compositions: composition/search comparison; not a single-factor ablation.
- Stage-specific point leaders — stage1: s1_085 (62.3% BA); reference: ref_rank2_fixed_epoch (58.5% BA).

## Parameter comparison caution

Every varied parameter and factor group is recorded. Boxplots use archived config-repeat summaries and are intentionally labelled descriptive because participants recur and many settings vary jointly.

## Missing V2 calculations

ROC-AUC/ROC curves, PR-AUC, participant-cluster CI and formal participant-exchange P are N/A. The required participant-keyed OOF probability rows were not archived; aggregate confusion matrices are not a valid replacement.

See `STUDY_SUMMARY.html` for all plots and `tables/report_tables.xlsx` for one worksheet per table.
