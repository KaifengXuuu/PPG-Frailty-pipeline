# M3 deprecated-profile fail-closed phase 29 / M3 弃用 Profile 关闭失败第 29 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_reference_test
- 流程 / Process：重新扫描 PPG registry 与运行时入口后，发现旧 `mobile_ppg_400_causal_v1` 已在机器合同中降为 deprecated alias，但 `preprocess_ppg` 仍只检查 modality 和采样率，可能被新实验直接运行。
- 算法 / Algorithm：PPG facade 现在同时要求 `status=future_active`、`modality=ppg`、用途属于 static/motion/peak/denoiser input，且 `resampling=no_resample`；任一不符立即拒绝。
- 结果 / Result：新增 deprecated mobile alias 负例；下一阶段运行完整参考测试确认无回归。
- 边界 / Boundary：旧 alias 仍保留在 registry 供历史配置解析与显式迁移，但不允许进入 corrected benchmark；移动端必须明确选择 static、motion 或 peak profile。
