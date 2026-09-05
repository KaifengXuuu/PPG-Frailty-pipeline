# Historical major-search V2-oriented reanalysis

## Scope and evidential status

This is a read-only reanalysis of four archived searches. It does not retrain a model, alter an archived run, or upgrade a post-hoc search into a confirmatory V2 test.

The matched architecture comparison is restricted to `5 s`, `50%` overlap, patience `20`, and no extra PPI/HRV input. All four source studies share seeds `42, 10042, 20042, 30042, 40042` and the exact participant-fold roster for every seed; this improves pairing but does not make the searches independent evidence.

## Matched CNN1D–InceptionTime–ShapeFormer comparison

| rank | model | subject BA mean ± SD | subject BA repeat t-CI95 | subject macro-F1 mean ± SD | runtime mean (s) |
| --- | --- | --- | --- | --- | --- |
| 1 | InceptionTime | 72.7 ± 6.0 | [65.2, 80.2] | 72.3 ± 5.9 | 396.8974 |
| 2 | CNN1D | 70.3 ± 4.5 | [64.7, 75.9] | 70.8 ± 4.8 | 173.1708 |
| 3 | ShapeFormer-PISD | 61.6 ± 4.5 | [55.9, 67.2] | 60.5 ± 5.1 | 4864.1006 |

InceptionTime is the descriptive leader. ShapeFormer-PISD is lower by `11.1` BA percentage points and `11.8` macro-F1 points, while requiring about `12.3×` the mean archived run time. ShapeFormer was below both alternatives in all five matched repeats. This supports not advancing this historical ShapeFormer-PISD implementation in the ordinary shared search on utility/cost grounds, but does not prove general ShapeFormer inferiority.

The exact aggregate-repeat sign-flip tests are retained in `tables/early_three_model_exploratory_paired_tests.csv`. With only five sign pairs there are 32 possible sign patterns, so the smallest attainable two-sided P is 0.0625; these P values are exploratory and are not V2 participant-exchange inference.

## Fixed-epoch regularization search (20260608)

The archive contains `186` complete five-repeat configurations. Its top observed configuration is `s1_085` with subject BA `62.3 ± 7.0`. Because this winner was chosen from the same 186-config evidence, it is hypothesis-generating. The defensible manuscript use is to motivate fixed-epoch and regularization hypotheses later retested in V2, not to report the winner as an unbiased final estimate.

## SQI/loss/feature extension (20260625)

The archive contains `129` complete five-repeat configurations. Its top observed configuration is `s1_122` with subject BA `61.0 ± 6.1`. The strongest descriptive signal came from the historical SQI/aggregation family, which justifies a matched V2 Stage5 composition study; it does not freeze an SQI route by itself.

## Statistical compatibility with V2

Available: config-level means, sample SD, two-sided Student-t 95% intervals across the five archived repeat summaries, class-level hard-label metrics, confusion matrices, learning curves, and run duration.

Unavailable: participant-cluster bootstrap intervals, formal participant-exchange paired permutation P values, ROC/PR curves, probability calibration and t-SNE. See `tables/missing_v2_statistics.csv`. A P value is a null-tail probability, not posterior confidence.

The early architecture archive also carries a selection-contamination risk: the persisted fold histories and archived generator structure indicate that the held-out fold supplied the validation trajectory used for best-epoch selection. It is therefore described as legacy fold-held-out CV with selection contamination, not untouched OOF confirmation. The fixed-epoch June searches disable best-epoch selection, but still have no CV-external test set.

## Recommended writing order

1. Use the matched three-model historical comparison to motivate CNN/InceptionTime as ordinary candidates and to separate ShapeFormer into a non-blocking diagnostic route.
2. Use the 20260608 archive only to motivate fixed-epoch and regularization hypotheses.
3. Use the 20260625 archive only to motivate SQI, aggregation, loss and engineered-feature hypotheses.
4. Move every effectiveness claim to the later matched V2 ablations, hyperparameter studies and locked confirmation.

## Method references

- Student (1908), *The Probable Error of a Mean*, https://doi.org/10.1093/biomet/6.1.1
- Bengio & Grandvalet (2004), *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, https://www.jmlr.org/papers/v5/grandvalet04a.html
- Varma & Simon (2006), *Bias in error estimation when using cross-validation for model selection*, https://doi.org/10.1186/1471-2105-7-91
- Cawley & Talbot (2010), *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*, https://www.jmlr.org/papers/v11/cawley10a.html
- Ojala & Garriga (2010), *Permutation Tests for Studying Classifier Performance*, https://www.jmlr.org/papers/v11/ojala10a.html
- Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*, https://doi.org/10.2307/4615733
