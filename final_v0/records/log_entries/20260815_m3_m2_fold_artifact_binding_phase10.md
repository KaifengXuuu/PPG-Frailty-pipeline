# M3 M2 fold-artifact binding phase 10 / M3–M2 训练折 artifact 绑定第 10 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_tests
- 流程 / Process：新增 corrected M2 materialized fold resolver 与 scaler facade，拒绝自报 training role 但 roster 不符的拟合。
- 算法 / Algorithm：fit 前要求 observed subjects 精确等于 train roster、与 OOF 零交集；artifact 固化 dataset/fold/protocol/hash/seed/feature order/统计量。
- 结果 / Result：公共实现已保存，正负 membership 测试待加入并执行。
- 边界 / Boundary：M2 registry 只读，artifact 未来输出仅允许在 final_v0。
