# M2 外部同步 ECG/PPG/IMU 证据与资格图

> 本图区分强人工 ECG reference、pseudo reference、motion 资格与 BSS 元数据门；虚线表示限制，不是模型运行输入。

```mermaid
flowchart TD
    E["External source snapshots"] --> PTT["PTT-PPG 1.1.0<br/>66 sit/walk/run records"]
    E --> SIM["Simultaneous 1.0.0<br/>13 usable multi-stage records"]
    E --> IAM["iAMwell local 15"]
    E --> MIM["MIMIC-PERform local"]
    E --> VIT["VitalDB online/unfrozen"]
    PTT --> R1["Manual-verified ECG R peaks<br/>PPG + accel + gyro<br/>heartbeat and motion eligible"]
    SIM --> R2["Manual consensus ECG R peaks<br/>manual stage markers + accel<br/>heartbeat and motion eligible"]
    PTT -. "red/IR mapping conflict" .-> HOLD["BSS HOLD"]
    SIM -. "single unspecified Pleth wavelength" .-> HOLD
    IAM --> C1["Conditional heartbeat only<br/>pseudo ECG peaks · no IMU"]
    MIM --> C2["Conditional domain data<br/>deduplicate + freeze partitions first"]
    VIT --> C3["HOLD<br/>freeze case IDs, tracks, package and hashes"]
    R1 --> M["External record manifest"]
    R2 --> M
    C1 -.-> D["Dataset-level limitation registry"]
    C2 -.-> D
    C3 -.-> D
```

任何 pseudo-ECG peak 数据不得与人工复核 annotation 使用相同 reference-strength 标签；没有 IMU/activity supervision 的源不得报告 motion detector accuracy。
