# M3 统一预处理与信号 API / Unified Preprocessing and Signal API

本图固定数据先经过可追溯质量门，再进入任务 profile；EKF 是 IMU 主路线，LPF
只作为输入完全一致的对照。任何 invalid/insufficient 状态都不得伪造特征。

```mermaid
flowchart TD
    A["Raw window + M2 manifest metadata"] --> B["Channel/order/unit contract"]
    B --> C["Pre-filter quality gate<br/>finite, gaps, flatline, clipping heuristic"]
    C -->|invalid or insufficient| X["Explicit no-result + reason codes"]
    C -->|valid or repaired| D{"Versioned modality profile"}
    D --> P1["Static PPG 400 Hz<br/>0.2–8 Hz SOS"]
    D --> P2["Motion/peak PPG 400 Hz<br/>0.4–8 Hz SOS"]
    D --> I1["Primary IMU<br/>no-precal quaternion ESKF"]
    D --> I2["Comparator IMU<br/>0.3 Hz gravity LPF"]
    P1 --> R["Raw DC/AC context + filtered shape"]
    P2 --> R
    I1 --> M["SI gravity, dynamic acceleration,<br/>gyro, vector jerk, diagnostics"]
    I2 --> M
    R --> S["Training-fold-only scaling<br/>or reversible window scaling"]
    M --> S
    S --> Q["M1 SQI + optional Motion join"]
    Q --> H["High quality bypass denoiser"]
    Q --> L["Low/motion manual policy:<br/>drop XOR denoise"]
    H --> F["Shared FeatureBlock / physiology API"]
    L --> F
```

## 强制不变量 / Mandatory invariants

- 主路径不得进行单位猜测、缺失 gyro 无 mask 补零或 PPG 波长推断。
- 离线零相位与移动端因果 profile ID 不得复用。
- EKF 和 LPF 对照只允许重力估计方法不同。
- M4 以后新模块只能调用本包公共实现；根目录脚本标为历史只读。

