# M1 训练/评估 Pipeline 与移动推理 Runtime 边界

## 1. 强制隔离

| 内容 | Training/Evaluation | Mobile inference |
|---|---|---|
| PyTorch、optimizer、backprop | 允许 | 禁止 |
| Notebook | 可作探索 adapter | 禁止作为核心依赖 |
| fold/seed/label/OOF 统计 | 允许且必须记录 | 禁止进入 bundle |
| scaler/imputer/calibrator 拟合 | 仅 training fold | 只读已冻结参数 |
| threshold 选择 | inner training data | 只读已冻结阈值 |
| ONNX Runtime CPU | parity/evaluation | 允许 |
| NumPy/SciPy | 允许 | 允许 |
| scikit-learn | 允许 | 允许；artifact 必须可信且版本锁定 |
| 自动下载或在线服务 | 禁止默认依赖 | 禁止 |

## 2. Deploy bundle 最小结构

```text
bundle/
├── bundle_manifest.json
├── pipeline_config.json
├── preprocessing_profile.json
├── input_contract.json
├── output_contract.json
├── feature_schema.json
├── label_map.json
├── thresholds.json
├── artifacts/
│   ├── motion_detector.*
│   ├── heartbeat_or_frontend.*
│   └── classifier.*
└── hashes.sha256
```

所有路径必须相对 bundle 根；manifest 保存每个 artifact 的 SHA-256、字节数、producer code version、训练数据 manifest ID 和 protocol ID。移动端启动时先校验 schema、hash、依赖与 provider，再加载模型。

## 3. 分类器部署 adapter

| 候选 | 训练端 | 移动 adapter | M1 状态 |
|---|---|---|---|
| Flat InceptionTime | PyTorch | ONNX → `[B,3]` | contract defined, export/parity pending |
| Hierarchical InceptionTime | 两个 PyTorch 模型 | 两个 ONNX + 概率组合 adapter | contract defined, pending |
| 1D-CNN / Small InceptionTime | PyTorch | ONNX | contract defined, pending |
| ShapeFormer/PISD | PyTorch + discovery artifact | ONNX 或冻结 embedding adapter | research candidate; export not assumed |
| ROCKET/MiniROCKET + ridge | training Python | frozen kernels + NumPy transform + coefficient arrays | preferred lightweight contract |
| Tabular physiology | scikit-learn | trusted versioned sklearn artifact或显式系数/tree arrays | contract defined |

模型不能导出或 parity 不通过时标记 `training_only`，不得以 Python 中可运行作为移动可部署证据。

## 4. ONNX parity 门

每个深度 artifact 至少通过：

1. 固定输入 fixture 的 Python/PyTorch 与 ONNX CPU 输出最大绝对差、平均绝对差；
2. 完整 preprocessing + postprocessing 端到端 parity；
3. dynamic/static shape 审计；
4. CPU provider 明确指定，非 CPU provider 单独报告；
5. 低质量、全零、NaN 已拒绝、短窗 padding 和 batch>1 测试；
6. 模型加载失败时返回失败状态，不用随机初始化模型继续。

现有 `pttppg_denoiser_hybrid_export_onnx.py` 已提供 max-absolute-difference 思路；`pttppg_denoiser_onnx_runtime.py` 已提供 CPU provider、metadata 和 overlap-add 经验，但旧 hybrid 路线本身不是新架构的部署结论。

## 5. scikit-learn artifact 边界

- pickle/joblib 只加载本项目产生、hash 匹配且版本兼容的可信 artifact。
- 轻量线性/ridge 路线优先保存 scaler、系数、截距和 label order 为 JSON/NPZ，移动端用 NumPy 重建。
- tree/SVM 若使用 sklearn runtime，manifest 必须保存 sklearn 版本和安全来源。
- 不从用户上传路径或网络加载任意 pickle。

## 6. Runtime dependency 状态

2026-08-14 本地原生 WSL 只读检查：

| 依赖 | 用户允许 | 当前 WSL 可导入 |
|---|---|---|
| NumPy | 是 | 是 |
| SciPy | 是 | 是 |
| ONNX Runtime | 是 | **否** |
| scikit-learn | 是 | 是 |

因此 M1 只能验证 schema/合同，不能声称完成 ONNX runtime smoke。安装或下载依赖需要后续单独授权。

## 7. 运行时禁止项

- 访问 Frailty 标签或 fold registry；
- 按当前病人数据重新拟合 scaler、imputer、threshold 或 calibration；
- 读取整个未来阶段后回填早期预测；
- 捕获异常后输出默认类别；
- 隐式切换 `sqi_gate` 与 `coarse_denoise`；
- 未记录的 provider、线程数或量化模式；
- 将 OOF validation 指标展示为独立 test 性能。

