# M1 端到端架构、数据契约与移动处理中心约束

## 当前状态

- 里程碑：`M1`
- 状态：`contract_defined_implementation_not_started`
- 日期：2026-08-14
- 写入范围：仅 `final_v0/`
- 结论性质：架构与接口合同，不是模型性能结果、硬件实测或最终采购决定

本包冻结项目从原始 PPG/IMU 到 Frailty3 输出的模块顺序、跨模块数据契约、训练/移动推理边界、配置格式、候选分类器注册表，以及血压仪大小中心屏显处理设备的三档平台约束。

## 已确认事实

1. 中心设备接收可穿戴采集端的 PPG 与 IMU 信号，并承担屏显和本地处理。
2. 允许依赖：`NumPy`、`SciPy`、`ONNX Runtime`、`scikit-learn`。
3. 部署端不得依赖 Notebook 或 PyTorch；深度模型通过 ONNX/CPU-only adapter 使用。
4. 保留高性能与高性价比多档平台，不在 M1 锁定具体采购型号。
5. SQI 始终可以输出诊断/置信度；对波形采取的动作只能在 `sqi_gate` 与 `coarse_denoise` 中二选一。

## 文件导航

| 文件/目录 | 用途 |
|---|---|
| `01_END_TO_END_ARCHITECTURE_AND_API.md` | 模块顺序、公共 Python API、shape/dtype/channel/unit/missing 合同 |
| `02_MOBILE_PLATFORM_PROFILES.md` | 中心处理设备拓扑、三档平台和 provisional 资源门槛 |
| `03_TRAINING_VS_MOBILE_INFERENCE_BOUNDARY.md` | 训练与部署隔离、bundle、ONNX parity、安全边界 |
| `schemas/` | 输入、pipeline config、输出的机器可读 JSON Schema |
| `registries/` | 平台、质量策略、特征提取器和分类器候选注册表 |
| `examples/` | 三档平台配置实例 |
| `tools/validate_m1_contracts.py` | 只读验证或生成验证/树索引的双语工具 |
| `M1_CONTRACT_VERIFICATION.json` | 自动生成的合同完整性结果 |
| `M1_PACKAGE_TREE.md` | 自动生成的包树、字节、SHA-256 和逐文件说明 |

## 关键边界

- M1 不实现 SQI-v2、Motion-29、降噪器、heartbeat extractor 或 Frailty3 新模型。
- 平台预算是 M9 真实设备 benchmark 的工程门槛，不是已测性能。
- 代表性处理器名称仅表示兼容性类别，不构成当前价格、库存或采购建议。
- 当前 WSL 环境已检测到 NumPy、SciPy、scikit-learn；未检测到 ONNX Runtime。M1 不安装依赖。

