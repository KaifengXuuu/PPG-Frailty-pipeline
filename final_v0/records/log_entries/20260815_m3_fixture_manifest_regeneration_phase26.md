# M3 fixture manifest regeneration phase 26 / M3 Fixture 清单重建第 26 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：regenerated_and_integrity_tested
- 流程 / Process：使用固定 seed `20260815` 重建三份 NPY fixture 和 strict JSON manifest，并逐项复算 generator SHA、文件 SHA 与文件字节数；随后重新运行完整 M3 参考测试。
- 算法 / Algorithm：PPG fixture 冻结 30 秒原始波形和 37 个真值峰；IMU fixture 冻结 4,800×12 数组，列序为三轴加速度、三轴角速度、三轴重力真值、三轴动态加速度真值。所有含义均写入 manifest，不再依赖测试代码中的隐式切片。
- 结果 / Result：三个 fixture 的原 SHA 保持不变；generator/文件哈希与字节数逐项一致；M3 参考测试 42/42 通过。当前系统 `jsonschema` 版本不提供 Draft 2020-12 validator，因此本阶段不伪报第三方 schema 验证通过，最终由 M3 自有合同 validator 重新执行结构门。
- 边界 / Boundary：合成真值只证明确定性实现和误差计算口径；Frailty3 仍无姿态重力真值或人工 PPG 峰真值。
