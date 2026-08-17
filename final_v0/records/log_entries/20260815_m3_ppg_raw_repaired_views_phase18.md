# M3 PPG source/repaired views phase 18 / M3 PPG 原始与修复视图第 18 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫 PPG facade 后发现 raw metrics 实际取自插值后的 quality signal；现将 source raw 与 repaired raw 分开保存并分别计算描述量。
- 算法 / Algorithm：source metrics 仅在原始有限样本上计算并记录 nonfinite fraction；repaired metrics 来自显式修复视图；filtered AC/pulse amplitude 继续单独记录，三层语义不再混淆。
- 结果 / Result：单点 NaN fixture 保留 source NaN、repaired view 全有限、repair 状态和比例正确；全量 reference tests 41/41 通过。
- 边界 / Boundary：未知 ADC rail 仍只能报告 amplitude/clipping proxy，不宣称硬件饱和真值。
