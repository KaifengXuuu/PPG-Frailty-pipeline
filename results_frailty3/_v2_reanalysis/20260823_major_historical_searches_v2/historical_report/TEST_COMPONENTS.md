# Test components and fixed inputs

## Historical sources

- `/home/trinker/Code/github/PPG-Frailty-pipeline/results_frailty3/20260527_1320_cnn_inceptionTime`
- `/home/trinker/Code/github/PPG-Frailty-pipeline/results_frailty3/20260528_1045_shapeformer_0extra`
- `/home/trinker/Code/github/PPG-Frailty-pipeline/results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2`
- `/home/trinker/Code/github/PPG-Frailty-pipeline/results_frailty3/_overfitting_sweep/20260625_2320_overfitting_sweep_stage1_rank2`

## Matched architecture contract

- Window: 5 s
- Overlap: 50%
- Patience: 20
- Extra input: 0
- Split seeds: 42, 10042, 20042, 30042, 40042
- Split: exact participant-grouped five-fold rosters audited across all four sources
- Consumed report JSON files: 1590
- Scientific role: historical candidate/hypothesis generation only
