# M1 血压仪大小中心屏显处理设备：平台分档与工程预算

## 1. 用户确认的产品形态

```text
可穿戴采集端（RED/IR PPG + 3-axis ACC + 3-axis GYRO）
        │  BLE / USB / 有线串行（具体链路后续冻结）
        ▼
血压仪大小中心处理设备
        ├─ 本地缓存与数据完整性检查
        ├─ CPU/可选加速器推理
        ├─ 中心屏显
        ├─ 结果与失败原因输出
        └─ 可选加密导出
```

允许软件依赖：NumPy、SciPy、ONNX Runtime、scikit-learn。中心设备是主要计算节点；穿戴端初期只负责采集、时间戳、缓存和传输，不承担 Frailty3 模型推理。

## 2. 共同最低合同

- 64-bit Linux 作为 M1/M9 的参考操作系统；其他系统必须通过相同 bundle parity。
- CPU-only 路径必须可运行；NPU/GPU 仅是可选 acceleration provider。
- 支持至少 8 通道、400 Hz 连续输入和环形缓冲。
- 不依赖 Notebook、PyTorch、显示器交互状态或联网服务。
- 断连时采集端/中心端至少一侧有带序号缓存；重连不得静默重复或丢包。
- 屏幕必须显示：输入状态、运动/质量状态、coverage、预测/无可靠结果和 reason code。
- 所有本地文件使用明确 schema/version；个人数据的加密与权限在部署阶段另行审查。

## 3. 三档候选平台

以下处理器名称是架构级示例，不是已核验的当前价格、库存或采购推荐。

| Profile | 代表性处理器类别 | 适用目的 | 参考系统内存 | 加速器 | 主要取舍 |
|---|---|---|---:|---|---|
| `high_performance_x86_64` | Intel Core Ultra / AMD Ryzen Embedded 类 x86-64 | 开发、研究、复杂深度路线、快速屏显 | 16–32 GB | 可选 NPU/GPU | 余量最大，功耗和成本较高 |
| `accelerated_arm64_edge` | NVIDIA Jetson Orin 类或带成熟 NPU 的 ARM64 SoC | 高性能与体积/功耗折中 | 8–16 GB | NPU/GPU | provider/算子兼容需逐模型验证 |
| `value_arm64_sbc` | RK3588 / Raspberry Pi 5 类 ARM64 SBC | 高性价比 CPU-only、SQI/ROCKET/tabular | 4–8 GB | 无或可选 NPU | 成本低，但大型 ShapeFormer/深网余量较小 |

## 4. Provisional 工程门槛

这些数值是未来硬件验收门槛，不是当前测量结果。`per-hop latency` 从完整新数据窗到屏幕结果，不包含下一窗等待时间。

| Profile | 最大 per-hop pipeline latency | 最大 pipeline 峰值 RAM | 最大 deploy bundle | 目标屏显刷新 | 参考计算功耗包络 |
|---|---:|---:|---:|---:|---:|
| High-performance x86-64 | 500 ms | 4096 MB | 500 MB | ≤2 s | 15–45 W |
| Accelerated ARM64 | 750 ms | 2048 MB | 250 MB | ≤2 s | 10–25 W |
| Value ARM64 SBC | 2000 ms | 1024 MB | 100 MB | ≤5 s | 5–15 W |

共同硬门槛：不得因延迟超限而复用旧窗口结果冒充当前结果；若 backlog 超限，输出 `processing_lag` 并降级到已注册的 deterministic fallback。

## 5. 推荐的职责分配

### 高性能 x86-64

- 可运行所有候选算法的研究参考实现。
- 作为 ONNX 与 NumPy/SciPy 结果的 parity oracle。
- 允许 flat/hierarchical InceptionTime、ShapeFormer 候选和多路线批处理。
- 不因性能充足而放宽 bundle/schema/无泄漏要求。

### Accelerated ARM64

- 优先 Small InceptionTime、1D-CNN 或已验证算子集合的 ONNX 模型。
- NPU/GPU provider 失败时必须回退 CPU provider，并记录 provider。
- 需要真实设备上比较 provider 数值差、首次加载和持续推理延迟。

### Value ARM64 SBC

- 首选 SQI gate、MiniROCKET/ROCKET + ridge、tabular physiology 或小型 ONNX。
- 避免运行时生成大 kernel bank；训练端预生成并序列化只读数组。
- SciPy/ONNX Runtime 若在目标发行版不可用，M9 必须选择已验证镜像或改用等价轻量实现，不能静默删算法。

## 6. 连接、缓存与时钟

M1 不锁定 BLE 或 USB，但冻结传输包最低字段：

```text
device_id, firmware_version, packet_sequence, sensor_timestamp,
sampling_rate_hz, channel_order, units, payload, crc/status
```

中心端根据序号和时间戳识别丢包、重复、漂移和乱序。模型输入使用重建后的单调 sample time；墙钟时间只用于记录，不用于心搏间期。

## 7. M9 真实设备选择门

最终硬件必须用同一保存 bundle、同一固定 fixture 和同一线程设置报告：

- warm/cold latency 的 median、P95、P99；
- 峰值 RSS、长期运行内存漂移；
- bundle/model 大小和加载时间；
- Python/ONNX/provider 数值 parity；
- 1 小时连续流 backlog、温度降频和丢包行为；
- 断连、低 SQI、极短输入、缺失通道时的失败输出；
- 屏显刷新和操作响应。

在这些结果产生前，三档 profile 都保持 `candidate_profile`，不声称某个具体处理器已满足要求。

