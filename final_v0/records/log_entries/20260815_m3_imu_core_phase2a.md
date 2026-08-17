# 2026-08-15 M3 无预校准 ESKF 与 LPF 对照实现

- 新增 quaternion multiplicative error-state Kalman filter 主路线。
- 新增共享单位、质量门、20/40 Hz 前端和 jerk 的 0.3 Hz LPF 重力对照。
- ESKF 显式输出 initialization_pending、tracking、prediction_only 和 no_estimate；
  无静态预校准、yaw 不可观与 bias 部分可观限制随输出保留。
- 禁止 ESKF 失败时静默回退 LPF。
- 修正 raw8 scaling：零 IQR 明确 no-estimate，取消未由训练折拟合的固定 clip。
- 状态：IMU 公共实现已落盘，尚待固定 fixtures 和机器验收。

