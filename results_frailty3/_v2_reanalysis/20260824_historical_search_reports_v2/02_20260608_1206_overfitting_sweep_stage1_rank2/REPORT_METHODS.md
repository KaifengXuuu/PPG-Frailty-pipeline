# Historical reporter methods

- No model was retrained and no archived source was modified.
- Unit for repeat summaries: one archived complete participant-grouped 5-fold repeat (n=5).
- Display: arithmetic mean ± sample SD across repeat seeds.
- Descriptive CI95: mean ± t(0.975, n−1) × sample SD / sqrt(n). Repeated CV estimates are correlated, so this interval is descriptive rather than an unbiased generalization-error interval.
- Ranking: subject balanced accuracy, then subject macro-F1; all rankings are post-hoc and selection-contaminated.
- Parameter boxplots use archived config-repeat summaries. They share participants and often differ in multiple parameters; they show marginal associations, not causal single-factor effects.
- Formal V2 participant-cluster CI would resample participant IDs within true-class strata, carry all rows for each sampled participant across repeats, recompute each repeat metric, equally average repeats, and take the 2.5th/97.5th percentiles. Required participant-keyed OOF rows are absent, so these cells remain N/A.
- ROC-AUC requires continuous per-class participant OOF probabilities. Those probabilities were not archived; the ROC-AUC panel is intentionally marked N/A and no hard-label surrogate is invented.
- Every displayed table has a CSV, every root CSV becomes one workbook sheet, every plot has a CSV data partner, and every table column has a generated definition/formula catalog.

## References

- Student (1908), *The Probable Error of a Mean*, Biometrika 6:1–25.
- Brodersen et al. (2010), *The Balanced Accuracy and Its Posterior Distribution*, ICPR.
- Sokolova & Lapalme (2009), *A systematic analysis of performance measures for classification tasks*, Information Processing & Management 45:427–437.
- Fawcett (2006), *An introduction to ROC analysis*, Pattern Recognition Letters 27:861–874.
- Bengio & Grandvalet (2004), *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, JMLR 5:1089–1105.
- Varma & Simon (2006), *Bias in error estimation when using cross-validation for model selection*, BMC Bioinformatics 7:91.
- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, JMLR 11:2079–2107.
