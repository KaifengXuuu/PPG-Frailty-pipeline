# M1 V3 顺序 SQI–Motion–Denoiser 路由图

> 权威范围：取代 V1/V2 中 quality action owner 的路由图；V2 的输入、流式、bundle 和 provider fallback 图继续有效。

## 1. 运行时主流程 / Runtime flow

```mermaid
flowchart LR
    I["Validated window<br/>same ID / bounds / timestamps"] --> P["Versioned common preprocessing"]
    P --> SQ["SQI<br/>required, exactly once"]
    P --> ME{"Motion detector<br/>enabled?"}
    ME -->|"yes"| MD["Motion inference<br/>STATIC / MOTION / UNKNOWN"]
    ME -->|"no"| NE["NOT_EVALUATED"]
    SQ --> J["Evidence join"]
    MD --> J
    NE --> J
    J --> IV{"Invalid or<br/>unrecoverable?"}
    IV -->|"yes"| FD["FORCED DROP<br/>explicit abstention"]
    IV -->|"no"| HQ{"SQI HIGH and<br/>STATIC / NOT_EVALUATED?"}
    HQ -->|"yes"| RAW["Return unchanged<br/>preprocessed raw"]
    RAW --> FE["Shared FeatureBlock adapter"]
    HQ -->|"no: LOW or MOTION"| MP{"Run/session config<br/>manual exclusive policy"}
    MP -->|"drop"| DR["POLICY DROP<br/>features = null"]
    MP -->|"denoise_then_extract_features"| DN["One registered denoiser/frontend"]
    DN -->|"success"| FE
    DN -->|"failure"| NR["Explicit no-result<br/>no raw fallback"]
    FE --> CL["Classifier + coverage-aware aggregation"]
    CL --> OUT["Frailty3 result or explicit no-result"]
```

图注：

- SQI 与启用后的 Motion detector 可在第一级并行计算；denoiser 不在该并行域中。
- join 之前不得路由；Motion-positive 覆盖 high-SQI 的直返资格。
- `drop` 与 `denoise_then_extract_features` 是配置时互斥分支，不是同窗双算后择优。

## 2. 两个正交判定轴 / Orthogonal axes

```mermaid
flowchart TB
    Q["Quality axis<br/>HIGH / LOW / UNRECOVERABLE / UNKNOWN"]
    M["Activity axis<br/>STATIC / MOTION / NOT_EVALUATED / UNKNOWN"]
    Q --> T{"Deterministic truth table"}
    M --> T
    T --> A1["HIGH + non-motion<br/>RAW → FEATURES"]
    T --> A2["LOW or MOTION<br/>configured DROP or DENOISE"]
    T --> A3["INVALID / UNRECOVERABLE<br/>FORCED DROP"]
    T --> A4["UNKNOWN / module failure<br/>EXPLICIT NO-RESULT"]
```

29-subject 的 B/R 与 S/W 标签只监督 activity 轴；不能把 S/W 自动改写成 low SQI。

## 3. FeatureBlock 合流合同 / Feature convergence

```mermaid
flowchart LR
    H["High-quality path<br/>preprocessed_raw"] --> HA["Raw heartbeat/feature adapter"]
    L["Low-quality or motion path"] --> D["Selected denoiser/frontend"]
    D --> DA["Denoiser feature adapter"]
    HA --> C{"Same FeatureBlock contract?"}
    DA --> C
    C -->|"schema / dtype / unit / mask / time align"| F["Classifier-compatible features"]
    C -->|"mismatch"| R["Reject config or explicit failure"]
```

## 4. M8 最小实验矩阵 / Minimum factorial benchmark

```mermaid
flowchart TB
    FIX["Freeze subject folds, seeds,<br/>candidate-window hash and SQI"]
    FIX --> D0["Motion disabled"]
    FIX --> D1["Motion enabled"]
    D0 --> D0A["drop"]
    D0 --> D0B["denoise → features"]
    D1 --> D1A["drop"]
    D1 --> D1B["denoise → features"]
    D0B --> C["Within denoise arms:<br/>spectral / BSS / non-stationary / adaptive"]
    D1B --> C
    D0A --> E["Coverage-aware paired evaluation"]
    D1A --> E
    C --> E
    E --> G["Coverage/no-result gates<br/>→ BA → resource tie-break"]
```

每个 arm 必须保留路由前分母，分别报告 B/R/S/W、dynamic/recovery coverage、HR/PPI error、Frailty BA、risk–coverage、失败率和资源开销。

