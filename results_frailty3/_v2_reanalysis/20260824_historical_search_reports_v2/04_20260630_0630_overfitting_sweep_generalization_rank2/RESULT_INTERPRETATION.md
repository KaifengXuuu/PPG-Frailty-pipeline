# Result interpretation

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
