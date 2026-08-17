# M3 future fold scaling phase 20 / M3 未来训练折缩放第 20 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫 raw8 view、FoldScaler 与 M2 exact-roster facade，确认低层 StandardScaler/clip 仍可绕过 D4 冻结路线后收紧 future-active 边界。
- 算法 / Algorithm：M2-bound fit_fold_scaler 只接受 RobustScaler、no clip；artifact 固定 m3_raw8_dynamic_sequence.v1。model view 还要求 scaler 来自 training role，并显式输出 RED/IR + dynamic-acc XYZ + gyro XYZ 语义。
- 结果 / Result：新增 standard 与 clip 负例；全量 reference tests 42/42 通过。
- 边界 / Boundary：StandardScaler 仍可在低层用于历史对照，但不得进入 corrected future leaderboard 或伪装为 D4 hybrid view。
