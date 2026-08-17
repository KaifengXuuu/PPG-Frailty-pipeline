# M3 统一预处理与信号算法待录入主题

- 状态：draft；仅在用户要求草拟 `_agent` 更新时整理，不直接写入 `_agent`。
- 候选主题：M3 冻结的 400 Hz profiles、corrected/legacy 边界、无预校准 EKF 主路线、
  LPF 对照、异常门控、fold-only scaling、peak/PPI/HRV 公共 API 与验收结果。
- 证据位置：`final_v0/M3_unified_preprocessing_and_signal_algorithms/`。
- 当前进展：M3 公共实现已完成 profile-bound PPG、stateful causal IMU、train-fold scaler、
  corrected peak/PPI/HR/PRV、PTT train-only delay evaluator；38 项 reference tests 暂时全部通过。
- 已形成证据：合成真值 EKF/LPF 对照、261 文件 Frailty3 完整性审计和 B/R/S/W 角色级代理统计。
- 待完成：机器 schemas/registries、正式 M3 validator、完整文档/算法图和全局回归均通过后再冻结结论。
