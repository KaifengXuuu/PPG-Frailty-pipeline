# M0 五类方法、统一实现与 Benchmark 算法图

本图把本轮五类扩展审计连接到三个工程问题：motion detector、denoising 和动态 HR。实线表示未来可执行数据流；虚线表示监督、评价、安全门或失败历史。所有“未实现”节点只是已定义的实现路线，不是现有结果。

## 1. 当前证据到路线选择

```mermaid
flowchart LR
    A["只读事实源<br/>52 code/notebooks + inputs + outputs"] --> B["M0 方法登记<br/>17 historical routes"]
    B --> C1["Motion detector<br/>P02 Light CNN is promising"]
    B --> C2["Waveform denoising<br/>no strictly validated route"]
    B --> C3["Dynamic HR<br/>current direct route is negative"]

    C1 --> D1["Priority<br/>PPG+IMU detector + real artifact labels"]
    C2 --> D2["Conditional<br/>benchmark hybrid before further training"]
    C3 --> D3["Priority new route<br/>spectral candidates + temporal tracking"]

    E1["NLMS / DWT-A2 / CEEMD-lite"] -. "risk or baseline only" .-> D2
    E2["v7/v7.2/v8/Stage2"] -. "negative or blocked history" .-> D2
    E3["SQI + reject state"] --> D1
    E3 --> D2
    E3 --> D3
```

## 2. 五类方法与公共数据合同

```mermaid
flowchart TB
    R["SignalRecord<br/>subject + record boundaries<br/>PPG + IMU + timestamps"] --> Q["Shared quality layer<br/>interpretable SQI + flags"]
    R --> A["Adaptive ANC suite<br/>Wiener / LMS / NLMS / RLS"]
    R --> D["Nonstationary decomposition<br/>DWT / WPT / CWT / EMD / VMD / SSA"]
    R --> S["Spectral HR evidence<br/>joint TFR + IMU suppression"]
    R --> B["Dual-channel BSS<br/>PCA / FastICA / STFT-NMF"]

    B --> S
    Q --> A
    Q --> D
    Q --> S
    Q --> B

    A --> U["Unified benchmark adapter"]
    D --> U
    S --> U
    B --> U
    Q --> U

    T["ECG / manual labels<br/>evaluation only"] -.-> U
    U --> O["Per-record and per-subject metrics<br/>coverage + failures + confidence"]
```

## 3. Adaptive ANC 及真实 HR 保护门

```mermaid
flowchart LR
    P["Observed PPG d = s + v"] --> F["Adaptive filter"]
    I["Calibrated IMU references"] --> L["Per-channel FIR delay bank"]
    L --> F
    G["Train-only normalization<br/>record-local lag"] --> F

    F --> V["Artifact estimate v_hat"]
    P --> E["Residual e = d - v_hat"]
    V --> E
    E --> H["Peak / PPI / HR evaluation"]

    X["Assumption-break test<br/>motion causes real HR rise"] -.-> SAFE{"ΔHR and pulse energy retained?"}
    H --> SAFE
    SAFE -- "yes" --> PASS["Eligible as benchmark candidate"]
    SAFE -- "no" --> FAIL["Risk baseline only<br/>do not call clean PPG"]
```

## 4. 优先谱域动态 HR 路线

```mermaid
flowchart LR
    P1["IR / RED PPG"] --> TFR["Aligned multi-channel STFT"]
    I1["ACC / GYRO / jerk"] --> TFRI["IMU STFT + coherence"]
    TFR --> M["Motion contamination probability"]
    TFRI --> M
    M --> SM["Soft spectral evidence mask<br/>no clean-waveform claim"]
    TFR --> SM

    SM --> C["Top-K HR candidates per frame"]
    Q1["SQI components"] --> C
    C --> V["Viterbi offline path"]
    C --> K["Kalman causal path"]
    C --> PF["Particle multi-modal ablation"]

    V --> SEL["Selected HR + confidence"]
    K --> SEL
    PF --> SEL
    SEL --> R{"Evidence sufficient?"}
    R -- "yes" --> HR["HR track + PPI diagnostics"]
    R -- "no" --> MISS["Missing / reject + reason code"]

    ECG["ECG interval reference"] -. "subject-holdout evaluation" .-> HR
    ECG -. "coverage-aware error" .-> MISS
```

## 5. 双波长 BSS 与 SQI

```mermaid
flowchart TB
    X["Two synchronized PPG channels"] --> PRE["Robust centering + band contract"]
    PRE --> PCA["PCA baseline"]
    PRE --> ICA["FastICA multi-seed"]
    PRE --> NMF["Multichannel STFT-NMF"]

    PCA --> SCORE["Cardiac component scores"]
    ICA --> SCORE
    NMF --> SCORE
    IMU["IMU coherence"] --> SCORE
    SQI["Periodicity / template / entropy / IBI / channel agreement"] --> SCORE

    SCORE --> STABLE{"Stable and identifiable?"}
    STABLE -- "yes" --> OUT["Selected component or spectral evidence"]
    STABLE -- "no" --> FALL["Fallback to best-SQI raw channel"]
    OUT --> TRACK["Shared spectral candidate tracker"]
    FALL --> TRACK
```

## 6. 测试、证据和路线淘汰门

```mermaid
flowchart LR
    G0["G0 Contract<br/>shape units boundaries"] --> G1["G1 Unit<br/>edge cases determinism"]
    G1 --> G2["G2 Synthetic safety<br/>identifiability + HR retention"]
    G2 --> G3["G3 Leakage<br/>train-only calibration"]
    G3 --> G4["G4 Subject holdout<br/>paired improvement"]
    G4 --> G5["G5 External<br/>subject-level CI"]
    G5 --> G6["G6 Runtime<br/>streaming parity"]

    G2 -. "fail" .-> B0["Keep only as risk baseline"]
    G3 -. "fail" .-> VOID["Invalidate result"]
    G4 -. "no benefit over raw/SQI" .-> STOP["Stop route"]
    G5 -. "not run" .-> LIMIT["Internal exploratory claim only"]
    G6 -. "fail" .-> NODEP["Do not claim deployable"]

    G6 --> FINAL["Eligible final candidate<br/>HR + coverage + confidence + failures"]
```

## 7. 图中状态说明

- 已存在且可复用：STFT/ISTFT utilities、部分 IMU preprocessing、P02 detector benchmark、hybrid工程链、当前SQI消融框架。
- 已存在但只作基线：IMU-NLMS、DWT-A2、CEEMD-lite、v7.4 MaskNet、legacy detector、SoftHRFromFFT。
- 尚未实现：Wiener/LMS/RLS统一suite、真正wavelet threshold/CWT/WPT/EEMD/VMD/SSA、联合IMU谱抑制、候选格与Viterbi/Kalman/Particle、raw PPG BSS、新SQI。
- 所有未来输出均须位于 `final_v0/`，并在每次写入后同步日志、算法索引和详细文件树。
