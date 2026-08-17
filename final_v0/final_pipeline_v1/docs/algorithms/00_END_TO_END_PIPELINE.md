# End-to-end V1 workflow / V1 端到端流程

```mermaid
flowchart TD
    CLI["Public CLI: run-experiment<br/>公共真实训练/评估入口"] --> CFG["Versioned config + frozen repeat/fold<br/>版本化配置 + 冻结 participant 折"]
    CFG --> A["Frozen manifest rows for exact roster<br/>精确名单的冻结 recording 清单"]
    A --> B["QC + typed synchronized load<br/>质量检查与强类型同步读取"]
    B -->|"invalid"| Z["Explicit dropped/no-result row<br/>显式丢弃/无结果及原因"]
    B -->|"eligible"| C["400 Hz SignalViews<br/>x_native / x_filter / IMU"]

    C --> D0["Direct fixed-formula SQI components<br/>先计算 direct SQI 基础分量"]
    D0 --> D1["Fit empirical calibrator on outer-train IDs only<br/>仅 outer-train participant 拟合经验校准"]
    D1 --> D["Apply endpoint SQI without outer-label fitting<br/>Q_rate_pre + Q_morph_pre"]
    D --> E{"Run-locked policy<br/>SQI → optional motion → drop XOR reducer"}
    E -->|"high quality / identity"| F["Direct route<br/>rate + morphology + optical"]
    E -->|"low quality: drop"| Z
    E -->|"non-identity reducer"| G["ArtifactReducer<br/>aligned x_ar"]
    G -->|"failure"| Z
    G -->|"success"| H["Recompute Q_rate_post only<br/>Q_morph=not_applicable"]
    H -->|"post Q_rate fail"| Z
    H -->|"post Q_rate pass"| I["Pulse/PPI/eligible PRV<br/>仅 rate-qualified 生理量"]

    F --> J["Frozen feature registry<br/>values + validity"]
    I --> J
    C --> K["Robust-normalized raw windows<br/>原始八通道窗口"]
    J --> L{"representation_mode / 表征"}

    L -->|"feature_vector<br/>current formal executor"| N["L2 LR / RBF SVM / ExtraTrees"]
    N --> R["File probabilities + explicit dropped rows<br/>文件概率 + 显式丢弃行"]
    R --> S["file → role → participant<br/>equal-weight OOF aggregation"]
    Z --> S
    S --> T["Fixed experiment artifacts<br/>run_manifest + metrics + confusion<br/>4 OOF parquet + experiment_result"]

    L -->|"raw"| M["CompactCNN / Inception / ShapeFormer-exp"]
    L -->|"feature_matrix"| O["Mask-aware Inception / ROCKET+ridge"]
    L -->|"fusion"| P["Window encoder → file pool<br/>+ file feature once"]
    M --> U["Comparison/construction/training tests available<br/>current formal runner: failed_closed"]
    O --> U
    P --> U

    T -.-> V["Independent bundle subsystem<br/>save/load/golden parity tested separately<br/>current experiment runner does not export bundle"]
```

## Runtime meaning / 运行含义

- `run` is the real-input/protocol audit and emits no trained metric.
- `run-experiment` is the real frozen outer-fold training/evaluation entry.
- The passing reduced command uses `configs/motion_benchmark_v1.yaml`, preserves the
  complete participant roster, and fixes 60 seconds/one record/one epoch-equivalent.
- The current formal cell executor accepts `feature_vector` only. Raw, feature-matrix
  and fusion modules are available for named comparisons and contract tests, but their
  formal scientific-runner requests return `failed_closed`; no implicit conversion occurs.
- Full mode without repeat/fold requests all 25 frozen cells. A supplied repeat/fold pair
  runs one full-length cell with incomplete-5×5 scope.

`run` 只做真实输入/协议审计；`run-experiment` 才执行真实 outer-fold 训练。
当前 passing reduced 使用 motion feature-vector 配置并保留完整 participant 名单；
raw、matrix、fusion 虽已有模块与合同测试，但进入正式 runner 会明确关闭失败，不会
静默改成 feature-vector。

## Fixed evidence / 固定证据

- Authority pointer / 权威指针：
  [reference_registry.json](../../artifacts/experiments/reference_registry.json)
- Current 60 s r0/f0 smoke / 当前 60 秒单折 smoke：
  [experiment_result.json](../../artifacts/experiments/reduced_real_r0_f0_reference_width_preserved_v2/experiment_result.json)
- 12 s fail-closed gate / 12 秒关闭失败门禁：
  [experiment_result.json](../../artifacts/experiments/reduced_real_r0_f0_12s_failed_closed/experiment_result.json)

The 60-second values (5/6 OOF retained, coverage 0.8333, BA 0.5) are implementation
and abstention evidence with `scientific_scope=smoke_not_scientific_benchmark`. They
are not a completed 5×5 candidate result.

## Core invariants / 核心不变量

1. Non-identity `x_ar` never enters morphology or amplitude-dependent features.
2. Reducer or quality failure never silently falls back to identity.
3. SQI calibration, imputation, scaling and model fit use outer-train participants only.
4. Every selected OOF participant remains visible as retained or dropped/no-result.
5. The experiment runner writes fixed OOF artifacts atomically and refuses overwrite.
6. Bundle save/load/golden parity exists as a separately tested subsystem; current
   runner output must not be described as containing a bundle.
