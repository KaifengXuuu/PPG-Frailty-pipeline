# Historical V2-style report: 20260625_2320_overfitting_sweep_stage1_rank2

> Historical post-hoc search evidence; not a confirmatory V2 test.

## Top-15 V2-style leaderboard

| rank | config_or_model | model | subject_BA_mean_sd_percent | subject_BA_repeat_t_CI95_percent | subject_macro_F1_mean_sd_percent | subject_macro_F1_repeat_t_CI95_percent | subject_macro_ROC_AUC | ROC_AUC_applicability | scientific_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | s1_122 | inception_time | 61.0 ± 6.1 | [53.4, 68.6] | 60.4 ± 6.1 | [52.9, 67.9] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 2 | s1_102 | inception_time | 61.0 ± 2.1 | [58.4, 63.6] | 60.1 ± 2.0 | [57.6, 62.6] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 3 | s1_121 | inception_time | 59.3 ± 8.2 | [49.0, 69.5] | 58.7 ± 8.4 | [48.3, 69.2] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 4 | s1_055 | inception_time | 58.0 ± 5.4 | [51.3, 64.6] | 56.7 ± 6.4 | [48.7, 64.7] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 5 | s1_075 | inception_time | 58.0 ± 5.6 | [51.0, 64.9] | 56.9 ± 6.8 | [48.5, 65.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 6 | s1_061 | inception_time | 57.8 ± 5.7 | [50.7, 64.9] | 57.6 ± 6.5 | [49.5, 65.6] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 7 | s1_054 | inception_time | 57.5 ± 6.0 | [50.0, 65.0] | 56.3 ± 7.4 | [47.1, 65.6] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 8 | s1_033 | inception_time | 57.3 ± 4.3 | [51.9, 62.7] | 55.6 ± 4.9 | [49.5, 61.6] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 9 | s1_099 | inception_time | 57.3 ± 5.6 | [50.3, 64.3] | 56.3 ± 5.2 | [49.8, 62.7] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 10 | s1_067 | inception_time | 57.2 ± 4.6 | [51.6, 62.9] | 55.5 ± 5.1 | [49.2, 61.8] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 11 | s1_065 | inception_time | 57.0 ± 2.1 | [54.4, 59.7] | 57.0 ± 2.5 | [53.9, 60.2] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 12 | s1_003 | inception_time | 57.0 ± 8.0 | [47.2, 66.9] | 56.1 ± 8.0 | [46.1, 66.0] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 13 | s1_090 | inception_time | 56.9 ± 4.8 | [51.0, 62.8] | 54.7 ± 5.4 | [48.0, 61.4] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 14 | s1_074 | inception_time | 56.8 ± 7.4 | [47.5, 66.0] | 55.8 ± 8.1 | [45.7, 65.8] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 15 | s1_079 | inception_time | 56.7 ± 9.0 | [45.5, 67.8] | 55.8 ± 10.1 | [43.3, 68.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |

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

- Point leader: s1_122, subject BA 61.0 ± 6.1% and macro-F1 60.4 ± 6.1%.
- Runner-up: s1_102, subject BA 61.0 ± 2.1%; the same archive searched 129 configurations, so neither row is an unbiased final estimate.
- Writing role: motivate SQI, loss, aggregation and engineered-feature hypotheses
- Parameter-value boxplots are descriptive marginal views. A parameter can appear favorable because other settings differ; only later matched V2 ablations can support a component-effect claim.
- Repeat Student-t CI95 is available; participant-cluster CI, formal V2 P, ROC-AUC and ROC curves are N/A because participant-keyed OOF probabilities were not archived.
- Design dependency — quality_route_bundle: interpret as SQI route/composition, not threshold-only effect.
- Design dependency — multi_parameter_compositions: composition/search comparison; not a single-factor ablation.
- Stage-specific point leaders — stage1: s1_122 (61.0% BA); reference: ref_rank2_fixed_epoch_ep15 (53.1% BA).

## Parameter comparison caution

Every varied parameter and factor group is recorded. Boxplots use archived config-repeat summaries and are intentionally labelled descriptive because participants recur and many settings vary jointly.

## Missing V2 calculations

ROC-AUC/ROC curves, PR-AUC, participant-cluster CI and formal participant-exchange P are N/A. The required participant-keyed OOF probability rows were not archived; aggregate confusion matrices are not a valid replacement.

See `STUDY_SUMMARY.html` for all plots and `tables/report_tables.xlsx` for one worksheet per table.
