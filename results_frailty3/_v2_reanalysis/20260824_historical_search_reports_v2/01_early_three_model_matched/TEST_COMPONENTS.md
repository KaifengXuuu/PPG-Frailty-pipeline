# Test components and fixed inputs

- Sources: /home/trinker/Code/github/PPG-Frailty-pipeline/results_frailty3/_overfitting_sweep/20260527_1320_cnn_inceptionTime; /home/trinker/Code/github/PPG-Frailty-pipeline/results_frailty3/_overfitting_sweep/20260528_1045_shapeformer_0extra
- Models: CNN1D, InceptionTime, ShapeFormer-PISD
- Matched filter: window=5s; overlap=50%; patience=20; extra_input=0
- Input columns: RED, IR, AX, AY, AZ, GX, GY, GZ; historical DL view includes filtering, 64 Hz resampling and per-window robust scaling
- Files/participants/windows per selected run: 145 / 29 / 15,657
- Split seeds: 42, 10042, 20042, 30042, 40042; participant-grouped 5-fold rosters match exactly
- Epoch budget: up to 50 with historical held-out-fold validation trajectory and patience 20
- No PPI/HRV extra features
