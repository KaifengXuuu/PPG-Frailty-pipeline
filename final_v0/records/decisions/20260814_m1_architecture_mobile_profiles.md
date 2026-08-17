# M1-ARCH-001 — 端到端 API、质量动作与中心处理平台决定

- 日期 / Date：2026-08-14
- 状态 / Status：`confirmed_user_constraints_contract_defined`
- 来源 / Source：用户确认、M1 TODO、现有 ONNX/runtime 代码只读审计
- 范围 / Scope：M1 架构合同；不表示 M4 模块实现或 M9 硬件验收完成

## 决定 / Decision

1. 中心设备采用血压仪大小的屏显处理中心，接收可穿戴 RED/IR PPG 与六轴 IMU。
2. 允许 NumPy、SciPy、ONNX Runtime、scikit-learn；部署端禁止 Notebook 和 PyTorch。
3. 平台不锁定单一型号，定义高性能 x86-64、加速 ARM64 和高性价比 ARM64 SBC 三档合同，M9 以真实设备 benchmark 选择。
4. 公共 API 固定为 `SignalBatch → PipelineResult`；通道顺序固定 `RED,IR,AX,AY,AZ,GX,GY,GZ`。
5. SQI 始终可以输出诊断，但波形动作只能选择 `sqi_gate` 或 `coarse_denoise`，不得同时隐式作用。
6. classifier、feature extractor、quality policy 通过 registry/config 切换，保持输入输出合同不变。
7. 训练 fold、标签、early stopping、optimizer 和 Notebook state 不进入 deploy bundle。

## 暂定但需实测 / Provisional

- 三档 latency、RAM、bundle 和功耗门槛是工程目标，不是已测结果。
- 处理器类别名称仅作架构示例，不是采购结论。
- 连接方式尚未在 BLE/USB/serial 中锁定，但包序号、传感器时间戳、单位和 CRC/status 是共同最低字段。

## 后续验证 / Follow-up

- M2/M3 冻结 manifest、单位、采样与 preprocessing profiles。
- M4 实现模块与统一 adapter。
- M9 在真实三档候选设备进行 latency、RAM、parity、温度/长稳和断连测试。

