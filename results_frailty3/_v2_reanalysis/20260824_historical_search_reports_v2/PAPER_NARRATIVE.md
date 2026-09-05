# Paper narrative and cross-study interpretation

## Evidence order

1. **Historical architecture candidate generation.** The matched early archive selects `InceptionTime` descriptively (BA 72.7%) and supplies implementation-specific evidence for moving ShapeFormer-PISD to a separate diagnostic route.
2. **Fixed-epoch/regularization hypothesis generation.** `s1_085` is the 20260608 point leader (BA 62.3%) among 186 searched configurations; use it to motivate later fixed-epoch and regularization tests, not as an unbiased final estimate.
3. **SQI/loss/feature hypothesis generation.** `s1_122` is the 20260625 point leader (BA 61.0%) among 129 configurations; treat SQI/aggregation/manual-feature/loss compositions as route hypotheses.
4. **Generalization-grid stress test.** The overall point leader `ref_20260625_top1_s1_122` (BA 62.3%) is a fixed historical reference. The new-grid leader is reported separately inside report 04; fixed references are excluded from factor boxplots.
5. **Move confirmation to V2.** Only later matched V2 ablations, frozen hyperparameter studies, representation selection, SQI–motion–denoiser composition, and a final locked 5×5 run should support confirmatory claims.

## Cross-study boundaries

- All report units reuse the same 29 participants and the same participant-fold roster for each split seed. This supports matched descriptive contrasts but means the studies are not independent replications.
- The historical preprocessing contract changes: 20260608 uses a 64 Hz DL target, whereas 20260625 and 20260630 use 400 Hz. Baselines and available modules also change. Do not interpret their point leaders as a one-factor cross-study ablation.
- The early archive uses held-out-fold epoch selection and is selection-contaminated. The June fixed-epoch searches remove best-epoch selection but still use the same pooled participant-grouped CV evidence for search and ranking.
- ROC-AUC/ROC curves, PR-AUC, calibration, participant-cluster CI and formal participant-exchange P cannot be recovered because participant-keyed OOF probability rows and model checkpoints were not archived.
- Repeat Student-t CI95 quantifies dispersion across five correlated repeated-CV summaries. It must not be renamed participant-cluster CI or treated as model-selection-adjusted uncertainty.

## Recommended manuscript tables and plots

- Main historical model table: report 01 leaderboard plus matched repeat-delta and runtime plots.
- Fixed-epoch evidence: report 02 top configurations and parameter/factor boxplots, explicitly labelled hypothesis-generating.
- SQI/loss/feature evidence: report 03 route-composition plots and per-class tables.
- Generalization evidence: report 04 new-grid-only factor plots, seven-level joint sampling-policy comparison and matched marginal-delta table; fixed references shown only as anchors.
- Use `cross_study_contract_summary.csv` beside these results so sampling-rate and selection-protocol changes remain visible.
