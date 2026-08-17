# Heartbeat 与 PPG+IMU Motion A/B 图 / Heartbeat and Motion A/B

## A. PPG-only peak / IBI / auxiliary gate

```mermaid
flowchart TD
    DATA["PTT + iAMwell + MIMIC + VitalDB; SIM external"] --> LOAD["Dataset-specific loader and resampling"]
    LOAD --> Z["Per-record PPG z-score"]
    Z --> WIN["Overlapping PPG windows, one input channel"]
    WIN --> ENC["1D U-Net encoder-decoder"]
    ENC --> PEAK["Dense peak probability head"]
    ENC --> IBI["Dense IBI head, bounded 0.3–2.0 s"]
    ENC --> GATE["Window gate head"]

    ECG["ECG R timing / annotations"] -. "Gaussian target without ECG→PPG delay correction" .-> PEAK
    ECG -. "RR dense track" .-> IBI
    ACT["Partial activity labels"] -. "auxiliary target" .-> GATE

    PEAK --> THR["OOF peak threshold"]
    GATE --> GTHR["OOF gate threshold"]
    THR --> SCORE["CV / internal / extra event scorecards"]
    GTHR --> SCORE
    IBI --> SCORE
    SCORE --> NEG["External SIM peak F1@20 ms .0948; extra gate AUC .4088"]
```

## B. 数据切分与关键偏置

```mermaid
flowchart LR
    SUBJECTS["Subject registry"] --> TRAIN["254 train subjects"]
    SUBJECTS --> HOLD["63 internal holdout subjects"]
    SUBJECTS --> EXTRA["50 extra holdout subjects"]
    TRAIN --> CV["5-fold subject CV → OOF thresholds"]
    CV --> FINAL["Final model"]
    FINAL --> HOLD
    FINAL --> EXTRA

    DUP["Overlapping windows duplicate the same beat"] -.-> CV
    POOL["Pooled sample/window metrics, not subject-balanced"] -.-> HOLD
    DELAY["Dataset delay: .2305 to .4626 s, analyzed only post hoc"] -.-> FINAL
```

## C. 独立 PPG+IMU motion detector A/B

```mermaid
flowchart TD
    PPG["PPG"] --> C10["10 channels"]
    ACC["Gravity-removed accelerometer xyz"] --> C10
    GYR["Gyroscope xyz"] --> C10
    MAG["Acceleration, gyro, jerk magnitudes"] --> C10
    C10 --> A["Model A: denoiser-style encoder, random initialization"]
    C10 --> B["Model B: light CNN"]
    A --> VAL["PTT validation selects threshold by BA"]
    B --> VAL
    VAL --> PH["PTT holdout: both reported 1.0"]
    VAL --> SIM["External SIM"]
    SIM --> RA["A: F1 .7542; BA .7699; AUC .8269"]
    SIM --> RB["B: F1 .7634; BA .7802; AUC .8642"]
    LIMIT["Pooled overlapping windows; no subject-level CI"] -.-> SIM
```

## 路线分界

P01 与 P02 虽在同一训练脚本中，但输入、target和证据不同：P01 是 PPG-only 的历史 heartbeat尝试；P02 才是 PPG+IMU motion-state benchmark。后续不得把 P02 的外部性能解释为 P01 gate 或 peak 的性能。

