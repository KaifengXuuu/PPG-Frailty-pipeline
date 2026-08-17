# M3 Frailty3 IMU proxy builder phase 9 / M3 Frailty3 IMU 代理构建第 9 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_execution
- 流程 / Process：为 M2 冻结的 261 个文件逐一读取起始六秒，运行同上游 EKF 与 LPF，并按 B/R/S/W role family 汇总。
- 算法 / Algorithm：比较 coverage、dynamic-acceleration RMS 与 gravity-norm error proxy；Frailty3 无姿态真值，因此不计算或宣称 gravity RMSE。
- 结果 / Result：构建器已保存；机器结果待执行后写入 evidence。
- 边界 / Boundary：原始 CSV 只读且不复制；结果仅写 final_v0。
