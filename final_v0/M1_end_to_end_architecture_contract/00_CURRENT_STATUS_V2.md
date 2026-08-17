# M1 当前权威合同 V2 / Current Authoritative Contract V2

## 1. 版本地位

- 当前权威版本：`m1.architecture.v2`。
- `README.md` 与 V1 schema/registry/example 保留为初始合同历史，不删除、不覆盖。
- 若 V1 与本文件或 `*_v2` 机器合同冲突，以 V2 为准。
- V2 是最终只读代码审计后的合同修订，不表示 M2–M4 实现、ONNX smoke 或 M9 硬件 benchmark 已完成。

## 2. 冻结的端到端顺序

```text
timestamped input + manifest
→ validation/anomaly detection
→ versioned preprocessing
→ activity/motion detector
→ data-state classification
→ diagnostic candidates
→ exactly-one quality-action arbiter
→ feature extractor adapter
→ classifier adapter
→ stage/subject aggregation
→ frozen calibration
→ explicit result or explicit no-result
```

SQI、motion probability 与候选粗处理可以并行计算供诊断，但每个窗口只有一个 action owner：

- `sqi_gate`：`KEEP_RAW / SQI_DROP / SQI_WEIGHT`；
- `coarse_denoise`：`KEEP_RAW / COARSE_REPLACE`，SQI 仅观察/置信度。

禁止同一分支同时做粗处理替换、SQI top-k 丢弃和 SQI 加权聚合。

## 3. V2 数据合同增量

1. Canonical channel order 固定为 `RED,IR,AX,AY,AZ,GX,GY,GZ`。
2. 输入新增 `channel_present[8]`；缺失传感器不得用“真实零”伪装。
3. 每个窗口保存 `record_id/start_sample/end_sample/start_time_s/end_time_s/coverage`。
4. 运行时使用有界 ring buffer；三个示例为 40 s，禁止 whole-record cache/ridge 与 transductive normalization。
5. preprocessing profile 声明 `streaming_causal` 或 `buffered_zero_phase` 及算法延迟；二者不能共用同一版本声称 parity。
6. 输出显式包含 motion、SQI/quality action、HR/PPI、route/feature/classifier、provider/fallback/backlog 与 Frailty3 结果。
7. `invalid_input/insufficient_quality/processing_lag/runtime_error` 必须返回 no-result；不得复用旧 HR 或旧分类。

## 4. V2 bundle 与 runtime 增量

- ONNX 模型与每个 `*.onnx.data` 分别保存字节数和 SHA-256。
- imputer、scaler、calibrator、shapelet、ROCKET kernel、系数与 feature schema 都是独立 artifact。
- 加速器 profile 必须以 CPU FP32 为共同参考并保留 CPU fallback。
- provider、fallback、线程数、量化模式、latency 与 backlog 必须进入 runtime summary。
- 模型更新先在临时位置完整校验，再原子切换；保留上一已验证版本回滚。
- M1 示例是 `contract_example/pending_export`；只有 artifacts locked、thresholds 冻结且 parity 通过后才可成为 `deploy_locked`。

## 5. 三档中心处理平台

| Profile | 角色 | 代表性类别 | 结论 |
|---|---|---|---|
| High-performance x86-64 | golden/parity、高余量研究与产品候选 | Core Ultra / Ryzen Embedded 类 | 候选，未实测 |
| Accelerated ARM64 | 首要低功耗加速候选 | Jetson Orin / 成熟 ARM NPU 类 | 候选，必须验证整链与 CPU fallback |
| Value ARM64 SBC | 成本/功耗下限候选 | RK3588 / Raspberry Pi 5 类 | 候选，优先轻量路线 |

处理器名称只表示架构类别，不是价格、库存或采购建议。V1 表中的 latency/RAM/bundle/power 全部仍是 provisional engineering gates。

## 6. V2 文件导航

| 路径 | 作用 |
|---|---|
| `04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md` | 已有实现证据、风险与合同响应 |
| `schemas_v2/` | 输入、配置与输出 JSON Schema |
| `registries_v2/` | 平台、quality、feature、classifier 注册表 |
| `examples_v2/` | 三档可替换配置 |
| `tools/validate_m1_contracts_v2.py` | V2 机器校验和 V2 包树生成 |
| `M1_CONTRACT_VERIFICATION_V2.json` | V2 校验结果 |
| `M1_PACKAGE_TREE_V2.md` | V2 权威文件的字节数与 SHA-256 |

## 7. M1 完成边界

| 项目 | 状态 |
|---|---|
| 模块顺序与统一接口 | `defined_v2` |
| quality policy / feature / classifier 可替换 | `defined_machine_checked_v2` |
| shape/dtype/channel/unit/time/missing | `defined_machine_checked_v2` |
| 有界流式、bundle 完整性、CPU fallback | `defined_machine_checked_v2` |
| 实际 SQI/Motion/denoiser/classifier adapter | `not_started_M4` |
| ONNX Runtime smoke | `not_run_runtime_absent` |
| 真实设备性能/功耗 | `not_run_M9` |

