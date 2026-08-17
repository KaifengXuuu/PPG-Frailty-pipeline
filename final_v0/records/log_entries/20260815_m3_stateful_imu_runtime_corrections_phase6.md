# M3 stateful IMU runtime corrections phase 6 / M3 有状态 IMU runtime 修正第 6 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_tests
- 流程 / Process：新增注册表驱动的 causal IMU processor，持久保存 20/40 Hz SOS、ESKF、0.3 Hz LPF、跨块 jerk 和 timestamp 边界状态。
- 算法 / Algorithm：EKF 终止 no-estimate 锁存到显式 session reset；bias random-walk 离散噪声含 attitude 与 cross-covariance；公共 sample mask 同时要求 gravity、dynamic acceleration 和 jerk 有限。
- 结果 / Result：新增 chunk parity、profile mismatch、M1 m/s2 单位兼容、合成真值和 no-estimate latch 测试；执行结果待本批次复验。
- 边界 / Boundary：旧 root EKF 仍为 historical reproduction only；未修改 final_v0 外文件。
