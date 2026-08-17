# 项目历史信号处理总图 / Historical Signal-Processing Map

本图把 M0 审计到的历史输入、五类 motion/heartbeat 路线、实际证据和最终研究决策放在同一条可追溯链上。实线表示运行数据流；虚线表示监督、评价或审计引用，不表示部署时输入。

```mermaid
flowchart LR
    subgraph DATA["Local physiological data / 本地生理数据"]
        PPG["PPG channels"]
        IMU["Accelerometer + gyroscope"]
        ECG["ECG / R-peak reference"]
        ACT["Sit, walk, run labels"]
    end

    subgraph ROUTES["Historical method families / 历史方法族"]
        F["F01–F05 Transparent filters, Aboy++, EKF, ANC, CEEMD-lite"]
        V["V01–V06 v7 → v7.2 → v7.4 → v8 / Stage-2"]
        D["D01 Legacy handcrafted detector v8"]
        H["H01–H03 Hybrid pseudo-supervised denoiser + ONNX"]
        P["P01 PPG-only peak / IBI / gate"]
        AB["P02 PPG + IMU motion A/B"]
    end

    subgraph EVIDENCE["Observed evidence / 实际证据"]
        NEG["Negative or invalid waveform evidence"]
        PROXY["Proxy-only validation / visual smoke tests"]
        BLOCK["Runtime blockers or empty output directories"]
        PROM["Promising external motion-state benchmark"]
        DEPLOY["Partial PT / ONNX engineering artifacts"]
    end

    subgraph DECISION["M0 conclusion / M0 结论"]
        STOP["Do not restore full clean-waveform reconstruction"]
        BASE["Retain raw and high-quality-only baselines"]
        NEXT["Next: unique beat, IBI, HR/HRV, confidence, failure state"]
    end

    PPG --> F
    PPG --> V
    IMU --> V
    PPG --> D
    IMU --> D
    PPG --> H
    IMU --> H
    PPG --> P
    PPG --> AB
    IMU --> AB
    ECG -. "training / evaluation reference" .-> V
    ECG -. "training / evaluation reference" .-> H
    ECG -. "uncorrected timing target" .-> P
    ACT -. "activity proxy labels" .-> D
    ACT -. "activity proxy labels" .-> AB

    F --> PROXY
    V --> NEG
    V --> BLOCK
    D --> PROXY
    H --> PROXY
    H --> DEPLOY
    P --> NEG
    P --> DEPLOY
    AB --> PROM

    NEG --> STOP
    PROXY --> STOP
    BLOCK --> STOP
    DEPLOY --> STOP
    PROM --> BASE
    STOP --> BASE
    BASE --> NEXT
```

## 图示判读 / Reading guide

- ECG 在图中均以虚线进入历史路线，表示它应当只作为监督/评价reference；v7 setup2 把它变成实质推理输入，是已登记的critical leakage。
- P02 的 external SIM 指标是 motion-state 候选证据，不是 clean-waveform reconstruction证据。
- PT/ONNX artifact只证明工程写出路径存在；没有完整 preprocessing/parity/holdout 时，不进入“strictly validated”。

