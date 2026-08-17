# 2026-08-15 M3 固定 Fixtures 与 Reference Tests

- 新增固定 seed 20260815 的 PPG/IMU 合成真值生成器，使用稳定 NPY 和 SHA manifest。
- 新增异常 gap/flatline、PPG 频响、重采样、fold-only scaler 泄漏哨兵测试。
- 新增 SI 单位等价、causal 分块、无预校准 ESKF、LPF 隔离和 vector jerk 测试。
- 新增双极性、峰事件 recall、PPI 边界、PRV 公式/分层及 RED/IR 语义测试。
- 新增 unittest JSON runner 和 reference-test 算法矩阵图。
- 状态：测试代码已保存；机器结果由本阶段紧接运行并写入。

