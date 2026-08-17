# M3 统一预处理与信号算法 / Unified Preprocessing and Signal Algorithms

## 当前状态 / Current status

本包是 M3 的唯一未来活动实现入口。根目录历史脚本保持只读，只用于历史复现和
parity 审计；M4 之后的新模块必须调用本包公共 API 或读取本包版本化注册表。

This package is the sole future-active implementation boundary for M3. Root-level
historical scripts remain read-only and are retained only for historical reproduction
and parity audits. New M4+ modules must call this package or consume its versioned
registry.

## 已冻结原则 / Frozen principles

- 新主路线统一使用 400 Hz；256/500 Hz 仅作明确命名的 legacy comparator。
- static PPG 使用三阶 SOS 0.2–8 Hz；motion/peak/denoiser 使用三阶 SOS 0.4–8 Hz。
- 离线零相位与移动端因果滤波使用不同 profile ID。
- RED/IR 原始 DC、AC 和比例信息在归一化前保留。
- 所有 scaler、imputer、SQI 阈值和校准量只能由 training fold 拟合。
- 无预校准 quaternion/error-state EKF 是 IMU 主路线；0.3 Hz LPF 是受控对照。
- corrected_v1 是未来主协议；legacy_bug_compatible 只用于复现。

## 目录 / Layout

- src/m3_signal_core：NumPy/SciPy 公共实现。
- registries：固定参数和活动/历史 profile 映射。
- schemas：机器可校验合同。
- tests：确定性 reference fixtures、parity、泄漏和异常测试。
- evidence：源代码审计、数据质量和算法测试的机器证据。
- tools：只写本 M3 包的构建与验证工具。

