# Historical V2 report suite

One matched early-model report plus three separately rendered historical-search reports. No source archive was modified and no model was retrained.

| report | sources | configs | runs | point leader | BA |
| --- | --- | ---: | ---: | --- | ---: |
| [01_early_three_model_matched](01_early_three_model_matched/STUDY_SUMMARY.md) | 20260527_1320_cnn_inceptionTime;20260528_1045_shapeformer_0extra | 3 | 15 | InceptionTime | 72.7% |
| [02_20260608_1206_overfitting_sweep_stage1_rank2](02_20260608_1206_overfitting_sweep_stage1_rank2/STUDY_SUMMARY.md) | 20260608_1206_overfitting_sweep_stage1_rank2 | 186 | 930 | s1_085 | 62.3% |
| [03_20260625_2320_overfitting_sweep_stage1_rank2](03_20260625_2320_overfitting_sweep_stage1_rank2/STUDY_SUMMARY.md) | 20260625_2320_overfitting_sweep_stage1_rank2 | 129 | 645 | s1_122 | 61.0% |
| [04_20260630_0630_overfitting_sweep_generalization_rank2](04_20260630_0630_overfitting_sweep_generalization_rank2/STUDY_SUMMARY.md) | 20260630_0630_overfitting_sweep_generalization_rank2 | 232 | 1160 | ref_20260625_top1_s1_122 | 62.3% |

Open `STUDY_INDEX.html` for the plot-rich index.

Cross-study writing order and evidential boundaries: [PAPER_NARRATIVE.md](PAPER_NARRATIVE.md).
