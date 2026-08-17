# M0 基础函数与 Dash 算法图 / Foundation Functions and Dash Flow

## A. `funcs.py` 公共算法族

```mermaid
flowchart TD
    X["PPG + optional IMU + sampling rate"] --> QC["Finite-value and shape assumptions"]
    QC --> FILT["Butterworth high-pass / band-pass / notch"]
    FILT --> PK["Aboy++-style adaptive peak detection"]
    PK --> RR["Peak rejection → PPI / HR / HRV"]

    QC --> IMU["Bias estimate → EKF orientation → gravity removal"]
    IMU --> MS["Motion score / static-motion classification"]
    IMU --> ANC["Multi-reference normalized LMS ANC"]
    FILT --> ANC
    ANC --> RES["Artifact estimate + residual PPG"]

    FILT --> EMD["CEEMD-lite / frequency component selection"]
    IMU --> EMD
    EMD --> RES2["Selected component residual"]

    ERR1["API mismatch: notch f0 vs notch_freq"] -.-> FILT
    ERR2["Argument mismatch: fs used as lower_bpm"] -.-> RR
    ERR3["EKF switch / init and IMU cutoff issues"] -.-> IMU
```

## B. `ppg.py` 交互运行链

```mermaid
flowchart LR
    ENV[".env keys and user-selected paths"] --> LOAD["Load CSV and map signal columns"]
    LOAD --> VIEW["Dash callback / selected processing branch"]
    VIEW --> FN["Duplicated or imported filtering, IMU, ANC, peak routines"]
    FN --> MET["Peaks → PPI / HR / HRV"]
    FN --> PLOT["Interactive waveform and motion traces"]
    MET --> CSV["Optional user-path HRV CSV"]

    BUNDLE["Legacy v8 NPZ bundle"] -. "optional detector" .-> VIEW
    PATHERR["Default bundle filename is absent"] -.-> BUNDLE
    CONTRACT["Training/runtime feature-contract drift"] -.-> VIEW
```

## 审计结论

基础算法透明且可作为 M3 候选，但当前双实现、参数错位、边界coverage和runtime默认路径使其处于 `implemented_unverified`；不得直接当作统一公共实现。

