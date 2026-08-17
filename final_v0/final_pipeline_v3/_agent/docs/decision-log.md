# V2 decision log / V2 决策日志

## 2026-08-16 — ShapeFormer、motion、统计、final refit 与 V2 证据边界

- 状态：`confirmed`
- 来源：用户在当前 V2 实现会话中的明确确认。

### V2-029a — Literature-reference ShapeFormer discovery

- 选择：`D`。
- canonical literature-reference route 不使用固定 `shapelet_length_samples`，也不使用
  candidate stride。
- discovery 必须 fold-local、participant/file-balanced、channel-specific OSD/PISD。
- `num_pip_ratio=0.20`，从每条实际 discovery sequence 的长度 `T` 计算；禁止硬编码
  64。64 仅是 `0.20 × (5 s × 64 Hz)` 的派生示例。
- variable-length candidate 由三个连续 PIP 限定；每个 shapelet 必须保存
  `source_channel`、start/end samples、start/end seconds 和 candidate length。
- 每个 class 必须选择相同数量的 shapelets。
- 项目特定 capacity controls：3 shapelets/class；participant/file-balanced 最多180个
  discovery windows。这些不是 literature defaults。
- 历史 `PISD window_size=128` 是 position-search neighbourhood，不得解释为 shapelet
  length。
- `effect_size_fixed_v1` 单独保留 fixed 128 samples / stride 64。
- fixed 400/800 sample routes 只能是具名 fixed-length ablations，均不是 PISD reference。
- PISD 失败必须显式 fail closed，禁止静默回退到 effect-size。

### V2-029b — PISD channel semantics

- 选择：`A`。
- canonical reference ID：`channel_specific_osd`。
- 每个 PISD/OSD candidate 只有一个 `source_channel`；discovery 与 best-fit distance
  search 都只在该通道进行。
- ShapeFormer 整体仍是 multivariate：shapelets 可来自八个不同通道，generic branch
  仍接收完整八通道输入。
- 当前 joint-eight-channel candidate 仅可作为
  `multichannel_pip_centered_ig` 具名 ablation；不得标为 PISDPort，也不得成为 fallback。

### V2-030 — Formal motion input and preprocessing

- 选择：`A`。
- window：8 s @ 400 Hz；hop：2 s。
- ordered model tensor：
  `RED, IR, A_dyn_x, A_dyn_y, A_dyn_z, GX, GY, GZ, A_dyn_magnitude, Omega, J`。
- 先按设备已知单位转换：acceleration `g → m/s²`；gyro `degree/s → rad/s`。
- reference gravity removal：calibrated roll–pitch EKF；随后计算 `A_dyn`、`Omega`、`J`。
- 下列全部 IMU channels 使用 outer-training-participant-only robust scaler：
  `A_dyn_x/y/z, GX/GY/GZ, A_dyn_magnitude, Omega, J`。
- 禁止只缩放 gyro/derived；禁止逐 window 将 IMU 归一为相同幅度，因为 motion
  intensity 是目标信息。
- EKF 启用前必须通过 unit tests，并保存 process covariance、observation covariance、
  unit conversion 和 profile identity。
- EKF 失败不得静默回退 low-pass。Profile A low-pass gravity separation 保留为独立
  ablation。

### V2-031 — Explicit architecture/training identity

- 选择：`A`；所有 formal model config 必须显式冻结，不依赖隐藏代码默认值。
- 必须纳入 config/provenance/hash 的字段：architecture parameters、input channels/order、
  sampling rate、window/hop、normalization、padding/mask、feature schema hash、SQI/routing、
  loss、class weighting、sampler、epoch rule、optimizer、learning rate、weight decay、
  dropout、label smoothing、gradient clipping、random seeds、fold hash、aggregation、
  calibration。

### V2-032 — Five-member InceptionTime comparisons

- 选择：`C`。
- raw `InceptionTimeFull × 5` 与 feature-matrix `InceptionTimeMatrix × 5` 均实现为两个
  独立、显式、默认不运行的 comparison ablations。

### V2-024e2 — Statistical metric scope

- 选择：`A*`。
- participant bootstrap CI 同时覆盖 balanced accuracy 与 macro-F1。
- 报告必须同时保留 BA LCB95 和 macro-F1 LCB95 两列。
- paired permutation 分别用于 BA 和 macro-F1。
- Holm correction 在同一具名 comparison group × 同一 metric 内分别执行。
- ECE、worst-class、confusion matrix 和成本等仍完整报告，但不参与自动选模。

### V2-033 — Final refit and ensemble seeds

- 选择：`A*`。
- 每个由人工按用途选出的 final configuration，在全29名内部 participants 上从头
  refit；内部性能证据仍来自 OOF，full-data refit 不自报内部泛化性能。
- 单模型 final seed：`42`。
- 五成员 InceptionTime ensemble member seeds：
  `[42, 10042, 20042, 30042, 40042]`。
- 禁止将五成员 final model 描述成“seed 42 的一个模型”；必须保存五个成员身份、
  state hash 与 golden probabilities。
- winner ONNX gate 当前覆盖 model-input tensor → probabilities；严格 Python bundle
  负责完整预处理，硬件确定后再讨论端到端设备 gate。

### V2-034 — V1/V2 evidence boundary

- 选择：`A`。
- V2 是严密审核、作为论文基础的 final version；正式科学运行只允许来自可追溯的
  V2 source identity。
- V1 仅为过渡/历史证据，不得以 `current` 身份混入 V2 acceptance、报告、模型卡或
  tracking。
- V2 内继承的 V1-only 资产必须放入明确 historical namespace 并保存 inventory/hash；
  原始 `final_pipeline_v1` 不作为 V2 active source。

### V2-035 — Aura/nolds compatibility research

- 状态：`research_requested`，尚未选择修复方案。
- 要求：查找是否存在同时满足最新版 `hrv-analysis` 与 Python 3.11/main comparison
  environment 的兼容 `nolds` 版本；必须依据官方 package metadata/source 与真实
  isolated import smoke，不得修改 conda `ml` 保护栈，不得静默降低 Aura 版本。

## Explicitly deferred / 明确继续搁置且不重复询问

- V2-006：设备 ADC rail、绝对 scale、设备特定 QC。
- V2-009a/b/c：SQI 权重、阈值、监督 route。
- V2-010：motion override 激活，等待内部监督证据和 PTT。
- V2-012：最终 reducer winner。
- V2-026：部署硬件、功耗和端到端延迟门槛。
- V2-027：todo-only scope。
- 正式 ablation、完整 5×5、PTT benchmark：当前实现批次不得运行。

