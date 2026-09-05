# Historical V2-style report: 20260630_0630_overfitting_sweep_generalization_rank2

> Historical post-hoc search evidence; not a confirmatory V2 test.

## Top-15 V2-style leaderboard

| rank | config_or_model | model | subject_BA_mean_sd_percent | subject_BA_repeat_t_CI95_percent | subject_macro_F1_mean_sd_percent | subject_macro_F1_repeat_t_CI95_percent | subject_macro_ROC_AUC | ROC_AUC_applicability | scientific_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ref_20260625_top1_s1_122 | inception_time | 62.3 ± 1.4 | [60.6, 64.0] | 61.5 ± 0.5 | [60.9, 62.0] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 2 | ref_20260625_top2_s1_102 | inception_time | 61.8 ± 3.1 | [57.9, 65.6] | 60.8 ± 3.3 | [56.7, 64.9] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 3 | gen_212 | small_inception_time | 58.1 ± 5.6 | [51.2, 65.1] | 57.0 ± 6.0 | [49.6, 64.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 4 | gen_080 | inception_time | 58.1 ± 3.7 | [53.4, 62.7] | 56.9 ± 3.6 | [52.4, 61.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 5 | gen_030 | inception_time | 57.7 ± 5.7 | [50.6, 64.8] | 56.0 ± 7.4 | [46.9, 65.2] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 6 | gen_066 | inception_time | 57.7 ± 8.9 | [46.6, 68.7] | 56.1 ± 11.4 | [41.9, 70.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 7 | gen_074 | inception_time | 57.6 ± 10.6 | [44.4, 70.7] | 55.8 ± 11.1 | [42.0, 69.7] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 8 | gen_136 | small_inception_time | 56.7 ± 8.1 | [46.6, 66.7] | 55.6 ± 8.7 | [44.7, 66.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 9 | gen_044 | inception_time | 56.7 ± 7.7 | [47.1, 66.3] | 55.5 ± 7.1 | [46.7, 64.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 10 | gen_093 | inception_time | 56.7 ± 7.5 | [47.3, 66.0] | 53.5 ± 10.5 | [40.6, 66.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 11 | gen_164 | small_inception_time | 56.6 ± 5.9 | [49.2, 64.0] | 55.2 ± 6.5 | [47.1, 63.3] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 12 | gen_121 | small_inception_time | 56.5 ± 4.3 | [51.1, 61.8] | 55.6 ± 4.1 | [50.5, 60.7] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 13 | ref_20260608_top2_s1_091 | inception_time | 56.5 ± 5.6 | [49.6, 63.4] | 54.7 ± 5.2 | [48.3, 61.1] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 14 | gen_024 | inception_time | 56.5 ± 5.3 | [49.9, 63.0] | 54.8 ± 6.2 | [47.0, 62.5] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |
| 15 | gen_094 | inception_time | 56.3 ± 8.0 | [46.4, 66.2] | 54.6 ± 10.7 | [41.4, 67.9] | N/A | continuous participant-level OOF probabilities not archived | historical_hypothesis_generation_only |

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

- Point leader: ref_20260625_top1_s1_122, subject BA 62.3 ± 1.4% and macro-F1 61.5 ± 0.5%.
- Runner-up: ref_20260625_top2_s1_102, subject BA 61.8 ± 3.1%; the same archive searched 232 configurations, so neither row is an unbiased final estimate.
- Writing role: stress-test generalization/sampling hypotheses and compare frozen historical references
- Parameter-value boxplots are descriptive marginal views. A parameter can appear favorable because other settings differ; only later matched V2 ablations can support a component-effect claim.
- Repeat Student-t CI95 is available; participant-cluster CI, formal V2 P, ROC-AUC and ROC curves are N/A because participant-keyed OOF probabilities were not archived.
- Fixed historical references remain in the overall leaderboard as anchors but are excluded from generalization-factor boxplots; those plots use the new generalization grid only.
- Design dependency — regularization_bundle: report bundle; do not attribute separate WD or LS effect.
- Design dependency — sampling_policy: compare observed joint policies; do not infer independent sampler/quota effects.
- Design dependency — quality_route_bundle: interpret as SQI route/composition, not threshold-only effect.
- Stage-specific point leaders — fixed_reference: ref_20260625_top1_s1_122 (62.3% BA); generalization: gen_212 (58.1% BA).

## Parameter comparison caution

Every varied parameter and factor group is recorded. Boxplots use archived config-repeat summaries and are intentionally labelled descriptive because participants recur and many settings vary jointly.

## Missing V2 calculations

ROC-AUC/ROC curves, PR-AUC, participant-cluster CI and formal participant-exchange P are N/A. The required participant-keyed OOF probability rows were not archived; aggregate confusion matrices are not a valid replacement.

See `STUDY_SUMMARY.html` for all plots and `tables/report_tables.xlsx` for one worksheet per table.
