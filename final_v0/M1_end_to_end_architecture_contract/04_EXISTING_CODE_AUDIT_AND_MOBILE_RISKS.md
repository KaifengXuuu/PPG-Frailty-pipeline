# M1 现有实现接口审计与移动部署风险

## 1. 审计结论

本文件记录对 Motion detector、hybrid denoiser、peak/IBI gate、Frailty3 classifier 与 ONNX runtime 的只读接口审计。已有算法部件可复用，但当前没有任何一条路线形成“原始 8 通道输入 → 统一预处理 → Motion/SQI/HR/PPI → Frailty3 → 屏显输出”的完整移动 bundle。

M1 只冻结新合同；旧根目录文件保持只读。具体修复、adapter、单元测试与 smoke test 留到 M3/M4。

## 2. 可复用资产

| 部件 | 已有能力 | M1 登记状态 |
|---|---|---|
| Dedicated Motion detector | 256 Hz、8 s/2 s hop、binary logit、已有 ONNX 导出路径 | 网络候选；窗口坐标和 preprocessing metadata 待补 |
| Hybrid denoiser | 500 Hz、6 s/1 s hop、11/15 channel、PyTorch/ONNX runtime | network-only candidate；整链 parity 未完成 |
| Peak/IBI gate | peak logit、IBI seconds、gate logit、ONNX + external data | candidate；dynamic batch 与 bundle 完整性待验 |
| Frailty CNN/InceptionTime | 8-channel sequence encoder、`forward_features` 接口 | exporter/runtime/bundle 未统一 |
| Frailty fusion/tabular | file features、imputer/scaler、sklearn 历史实现 | feature pipeline 与 artifact 保存待统一 |
| SQI/physiology features | skew、kurtosis、Welch、IBI、RED/IR consistency 等散落实现 | 重构证据；不视为 SQI-v2 已实现 |

## 3. 已确认风险及合同响应

1. **Overlap-add 未覆盖位置可能成为零**：Hann 首尾权重为零且不完整尾窗可能无 coverage。V2 强制真实 coverage，并要求 raw fallback 或 no-estimate。
2. **Fusion artifact 不完整**：部分历史保存路径缺 imputer/scaler。V2 把全部 transform 设为独立、带 hash 的 bundle 文件。
3. **历史 CV 存在 outer-test early stopping**：相关成绩只能标为 historical/non-strict，不得用于最终部署模型选择。
4. **Motion window 缺 start/end**：V2 强制右开 sample 坐标、绝对时间戳与 coverage。
5. **单位与缺失策略冲突**：固定假设、幅值猜测、插值和零填充并存。V2 要求单位只在 canonical ingestion 显式转换，并保留 `valid_mask/channel_present`。
6. **SQI 原始信息与校准风险**：已有实现可能在独立标准化后计算，percentile calibration 也可能跨 fold。V2 要求 amplitude-preserving 输入与 training-fold-only calibration。
7. **研究 runtime 不是有界流式**：whole-record ridge、SciPy/Pandas 处理和重复拷贝可能比小 ONNX 网络更耗资源。V2 禁止 whole-record runtime，M9 测整链。
8. **分类器 profile 漂移**：历史模型 sampling/window 与当前默认值可能不同。每个 artifact 必须绑定 preprocessing profile、shape、dtype、channel/class order。
9. **现有 ONNX 不是完整 pipeline**：hybrid ONNX 只封装 network；peak gate 还依赖独立 `.onnx.data`。V2 对每个文件分别校验。
10. **加速器并不覆盖预处理**：NPU/GPU 不会自动加速 SciPy、lag-ridge、窗口与 overlap-add，因此必须保留 CPU reference 并做端到端 benchmark。

## 4. 分类器移动状态

| 家族 | 合同结论 |
|---|---|
| 1D-CNN / InceptionTime / Small InceptionTime | ONNX 候选；完整 preprocessing、aggregation、export/parity 未完成 |
| Hierarchical InceptionTime | 两个模型与概率组合 adapter 均须冻结 |
| ShapeFormer/PISD | experimental/training-only；shapelets、索引与算子 parity 未完成 |
| ROCKET/MiniROCKET + ridge | 尚未实现；预期冻结 kernel、scaler 和系数数组 |
| Logistic/RBF-SVM/ExtraTrees tabular | classifier 历史路线存在；feature extractor、class order、版本兼容与安全加载待统一 |

## 5. 最低后续测试

- raw input 到每个 preprocessing stage 的 golden tensor parity；
- NaN/Inf、flatline、缺失 gyro、channel swap、单位错误与短窗语义；
- 400↔500↔256 Hz 时间轴、窗口坐标与 tail coverage；
- PyTorch/sklearn reference 对 ONNX/native 的 batch 1/2/odd parity；
- SQI train-only calibration、唯一 action owner 与 branch attribution；
- classifier + imputer + scaler + calibrator round-trip；
- CPU FP32 reference、accelerator provider parity 与 accelerator→CPU fallback；
- 2 小时 worst-motion soak、8 小时内存漂移、功耗/温度/backlog；
- bundle 缺文件、`.onnx.data` 缺失、hash 不符、升级失败与回滚。

## 6. 状态判定

- 现有部件：`historical_candidate_or_partial`。
- M1 V2：`architecture_and_contract_defined`。
- 完整移动 pipeline：`not_implemented`。
- ONNX smoke：当前 WSL 未安装 ONNX Runtime，`not_run`。
- 硬件性能：`not_benchmarked_M9`。

