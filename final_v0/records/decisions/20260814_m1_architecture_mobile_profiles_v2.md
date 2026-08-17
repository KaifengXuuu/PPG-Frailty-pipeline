# M1-ARCH-002 — 最终审计后的 V2 流式、artifact 与回退合同

- 日期 / Date：2026-08-14
- 状态 / Status：`contract_defined_waiting_user_acceptance`
- 来源 / Source：用户确认、M1 TODO、三路只读代码/部署审计
- 关系 / Relation：补充 `M1-ARCH-001`；冲突处以本决定为准

## 决定 / Decision

1. 移动 runtime 采用有界 ring buffer；三个 M1 示例使用 40 s，禁止 whole-record 推理与 transductive normalization。
2. preprocessing profile 必须声明 causal/buffered 模式与算法延迟。
3. SQI/粗处理可作为诊断候选并行计算，但每窗只有一个 action owner 与一个 action code。
4. 窗口必须有右开 sample 坐标、绝对时间和真实 coverage；未覆盖位置不得生成伪零波形。
5. ONNX external data、imputer、scaler、calibrator、shapelet/kernel/coefficients 都是独立 artifact，逐文件校验。
6. 加速平台必须保留 CPU FP32 reference/fallback，并输出 provider、fallback、backlog 与 timing。
7. M1 config 分为 `contract_example` 与 `deploy_locked`；当前三例均不是可部署模型。

## 暂定项 / Provisional

- 40 s buffer 是覆盖当前 30 s Frailty 窗口的 M1 示例，不是 M3 最终值。
- 三档 latency/RAM/bundle/power 仍是 M9 工程门槛，不是实测。
- `TARGET_VENDOR_EP_PENDING_M9` 是 schema 示例占位，不表示已选择 NPU provider。

