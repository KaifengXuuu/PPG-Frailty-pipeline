# Signal, SQI, artifact, and feature routes / 信号、SQI、伪影与特征路线

```mermaid
flowchart TD
    A["RED/IR + six-axis IMU"] --> B["Repair finite internal gaps only"]
    B --> C["x_native: acquisition scale"]
    B --> D["x_filter: zero-phase 0.2–8 Hz"]
    D --> E["direct x_analysis = x_filter"]
    A --> F["SI units + causal sensor filters"]
    F --> G["EKF gravity primary"]
    F --> H["0.3 Hz LPF comparator"]
    G --> I["a_dyn, A, Ω, J"]
    H --> I
    C --> Q["Direct endpoint SQI"]
    D --> Q
    I --> Q
    Q --> R{"direct or non-identity?"}
    R -->|"direct/identity"| S["Q_rate + Q_morph"]
    S --> T["peaks + PPI + PRV"]
    S --> U["morphology + AC/DC + PI + R + coherence"]
    R -->|"NLMS / SSA / spectral / BSS"| V["aligned x_ar = rate-only x_analysis"]
    V --> W["Q_rate_post only"]
    W --> T
    W --> X["Morphology slots = null<br/>validity=false / not_applicable"]
    T --> Y["Versioned rate feature block"]
    U --> Z["Versioned direct feature block"]
```

The EKF and LPF branches share units, filtering, masks, timestamps, and output schema;
only the gravity estimator changes. / EKF 与 LPF 只改变重力估计器，其余上游与输出合同一致。
