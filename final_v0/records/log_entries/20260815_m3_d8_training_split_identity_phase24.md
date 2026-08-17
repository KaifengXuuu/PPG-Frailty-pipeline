# M3 D8 training-split identity phase 24 / M3 D8 训练分割身份第 24 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：在写入前重新扫描 M3 的 reference evaluator、D8 测试和 machine schemas；随后把 `training_split_id` 从可省略的空字符串升级为拟合 transit-delay artifact 时的必填身份，并重新运行完整参考测试。
- 算法 / Algorithm：ECG→PPG transit delay 只能在 training subjects 上估计；artifact 同时绑定 dataset、fold registry、training split、preprocessing profile 与 algorithm 身份。空白 `training_split_id` 立即 fail closed，防止无法追溯的延迟参数进入 OOF 或 external evaluation。
- 结果 / Result：M3 参考测试 42/42 通过，0 failures、0 errors、0 skipped；D8 的 raw 与 delay-corrected scorecard 保持对称，训练/评价 subject overlap 负例继续被拒绝。
- 边界 / Boundary：本阶段只强化身份与泄漏防线，不生成新的 PTT 性能结论；正式数值仍必须在冻结的 M2 fold registry 或显式 external holdout 上重跑。
