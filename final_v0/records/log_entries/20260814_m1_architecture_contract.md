## 2026-08-14 — M1 端到端架构、统一合同与移动处理中心分档

- 操作 / Action：定义 SignalBatch→PipelineResult 模块顺序、机器 schema、可替换 registries、训练/推理隔离和三档中心处理平台。
- 用户确认 / User-confirmed：血压仪大小中心屏显处理设备；可穿戴 PPG+IMU；允许 NumPy/SciPy/ONNX Runtime/scikit-learn；需要高性能和性价比方案。
- 关键设计 / Key design：SQI 保留共同诊断出口，波形动作在 sqi_gate/coarse_denoise 中恰好选一；具体处理器留待 M9 实测锁定。
- 新增 / Added：M1 文档包、3 schemas、4 registries、3 example configs、双语验证器、4幅 Mermaid 流程图和决策 M1-ARCH-001。
- 边界 / Boundary：未联网、未安装依赖、未训练/推理、未修改 final_v0 外文件、未写入 `_agent`。
- 验证 / Validation：随后运行合同验证、算法图/覆盖验证、总树和 delivery verification；这些维护更新不递归记日志。

