# M1 端到端模块架构与统一 Python API

## 1. 冻结的模块顺序

```text
input/manifest
→ validation/anomaly detection
→ preprocessing
→ IMU activity/motion detection
→ data-state classification
→ quality action: exactly one of {sqi_gate, coarse_denoise}
→ feature extraction
→ classifier adapter
→ subject/stage aggregation
→ probability calibration
→ output
```

`SQI` 还有一个不改变波形的共同诊断出口。部署动作互斥规则是：

- `sqi_gate`：SQI 决定接受、降权或拒绝原始/基础预处理窗口；signal frontend 必须是 `identity`。
- `coarse_denoise`：选定一个粗处理 frontend；SQI 只提供诊断、coverage 和置信度，不再对同一窗口施加第二套隐式波形处理。

这同时满足旧 TODO 的“二选一”要求和后续四条路线共用 SQI 的新决定。

## 2. 公共运行时对象

### 2.1 `SignalBatch`

```python
SignalBatch(
    values: np.ndarray,          # [n_samples, 8], float32
    timestamps_s: np.ndarray,    # [n_samples], float64, monotonic
    valid_mask: np.ndarray,      # [n_samples, 8], bool
    channel_order: tuple[str],   # exact canonical order
    sampling_rate_hz: float,
    units: dict[str, str],
    metadata: dict[str, object],
)
```

规范通道顺序固定为：

```text
RED, IR, AX, AY, AZ, GX, GY, GZ
```

原始边界允许设备单位：PPG `adc_counts/raw_counts`，ACC `g` 或 `m/s2`，GYRO `deg/s` 或 `rad/s`。单位不得靠数值范围静默猜测；`units` 缺失时 validation 返回 `unit_unknown`，由显式 preprocessing profile 决定是否允许继续。

缺失语义：

- 原始 `values` 可含 NaN/Inf，但必须同步给出 `valid_mask`。
- validation 产生插值/拒绝决定和 reason code。
- preprocessing 后的有效窗口不得含 NaN/Inf。
- 无法修复时返回 `invalid_input` 或 `insufficient_quality`，不得用零填充后伪装为有效结果。

### 2.2 中间结果

所有模块返回带版本和 reason code 的显式对象：

```python
ModuleResult(
    status: str,                 # ok / partial / invalid / no_estimate
    values: object,
    valid_mask: np.ndarray | None,
    confidence: np.ndarray | float | None,
    reason_codes: tuple[str, ...],
    module_id: str,
    module_version: str,
    diagnostics: dict[str, object],
)
```

任何模块失败都不得隐式回退。允许回退时必须写入 `fallback_used`、原失败原因和 fallback 版本。

### 2.3 `PipelineResult`

```python
PipelineResult(
    status: str,                    # ok / partial / invalid_input / insufficient_quality
    probabilities: dict[str, float] | None,
    predicted_label: str | None,
    confidence: float | None,
    coverage: float,
    quality_summary: dict[str, object],
    feature_summary: dict[str, object],
    reason_codes: tuple[str, ...],
    versions: dict[str, str],
    timing_ms: dict[str, float],
)
```

当 `status != ok/partial` 时，`probabilities` 和 `predicted_label` 必须为空；不可强制输出 Frailty3 数字。

## 3. 模块合同

| 模块 | 主要输入 | 主要输出 | 必须保持的不变量 |
|---|---|---|---|
| Input adapter | device packet/file | `SignalBatch` | 8通道顺序、时间戳、单位、设备版本显式 |
| Validation | raw `SignalBatch` | anomaly mask + status | 不学习 fold/test 统计；所有修复可追踪 |
| Preprocessing | validated batch + profile | canonical float32 batch | 参数版本化；单位转换显式；无非有限值 |
| Motion detector | canonical PPG/IMU windows | OOF/deploy `p_active` | activity probability 不是 artifact truth |
| Data-state classifier | validation+motion+SQI | invalid/static/motion-usable/unrecoverable | reason code 与 coverage 必须输出 |
| SQI monitor | canonical/processed window | component scores + composite | 组件、权重、版本和缺失策略显式 |
| Quality action | state + SQI + selected frontend | accepted/denoised signal | `sqi_gate`/`coarse_denoise` 恰好一个 |
| Feature extractor | accepted signal + masks | named feature block | schema ID 和 availability time 显式 |
| Classifier adapter | feature block | `[batch,3]` probabilities | 固定 label order，不暴露模型特有接口 |
| Aggregator | window/file/stage probabilities | subject probabilities | coverage 和被拒窗口数保留 |
| Calibrator | uncalibrated probabilities | calibrated probabilities | 仅使用训练数据拟合的 artifact |
| Output adapter | all summaries | `PipelineResult` | 无可靠输入时不预测 |

## 4. 时间尺度与 shape

- 原始记录：`[N,8]`。
- 通用窗口接口：`[B,C,T]`，其中 `C` 由 registry 定义，`T=round(fs×window_sec)`。
- tabular feature：`[B,F]`，列顺序由 `feature_schema_id` 冻结。
- probability：`[B,3]`，label order 固定为 `pre_frail, robust_non_frail, young`。
- window、file、stage、subject 四个 level 必须作为字段保存，不允许仅靠表名推断。
- feature 必须带 `available_at_sec`；禁止使用预测时刻之后的整文件信息。

窗口长度不在 M1 全局写死：Motion、SQI、heartbeat 和 Frailty classifier 各自通过版本化 module config 声明 window/hop。跨模块对齐使用绝对 `timestamps_s`，不使用数组位置猜测。

## 5. 统一 Pipeline API

部署入口冻结为：

```python
pipeline = FrailtyPipeline.from_bundle(bundle_dir, config_path)
result = pipeline.process(signal_batch)
```

训练/评估入口独立：

```python
run_training(train_manifest, split_manifest, experiment_config, output_dir)
run_evaluation(bundle_dir, evaluation_manifest, protocol_config, output_dir)
```

移动 bundle 不得包含 `train_manifest`、outer/inner fold IDs、early-stopping history、validation labels、Notebook state 或 PyTorch optimizer。

## 6. 可替换注册表

配置只引用 registry ID：

- `quality_strategy.policy_id`
- `quality_strategy.signal_frontend_id`
- `feature_extractor.id`
- `classifier.id`
- `aggregation.id`
- `calibration.id`

每个 feature extractor 声明 `output_adapter`；每个 classifier 声明 `accepted_input_adapters`。只有 adapter 相容时配置才有效。所有 classifier adapter 最终统一返回同一 Frailty3 probability contract，因此替换 classifier 不改变 output API。

## 7. 配置与版本

每个 deploy config 必须包含：

- config/schema version；
- platform profile；
- input/output contract ref；
- preprocessing profile/version；
- quality action 与 SQI version；
- motion detector、feature schema、classifier、aggregation、calibration 版本；
- threshold 和 label map；
- artifact 相对路径及 SHA-256（实际 bundle 阶段）；
- allowed runtime dependencies。

配置中禁止出现训练 fold、测试指标或运行时自动调参字段。阈值在部署前冻结，运行时只读取。

## 8. M1 验收状态

| 要求 | 状态 |
|---|---|
| 完整模块顺序 | `defined` |
| SQI/coarse-denoise 动作互斥 | `defined_machine_checked` |
| shape/dtype/channel/fs/unit/timestamp/missing | `defined` |
| training/mobile 边界 | `defined` |
| 统一配置 schema | `defined_machine_checked` |
| 多分类器 registry | `defined` |
| 实际模块实现与 smoke | `not_started_M4` |
| 实际硬件 latency/memory | `not_run_M9` |

