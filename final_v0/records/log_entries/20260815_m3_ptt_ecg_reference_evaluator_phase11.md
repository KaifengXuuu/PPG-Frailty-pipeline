# M3 PTT ECG reference evaluator phase 11 / M3 PTT ECG 参考评价器第 11 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_pending_tests
- 流程 / Process：新增 training-only PPG transit-delay artifact 与 disjoint evaluation scorecard。
- 算法 / Algorithm：ECG R peak 后 0.05–0.60 s 内一对一匹配 PPG，训练折取 median delay；评价同时报告 raw/corrected F1、timing、PPI/HR error、coverage/failure。
- 结果 / Result：D8 evaluator 实现已保存；固定合成正例和泄漏负例待执行。
- 边界 / Boundary：detector 对 ECG annotations 的成绩不冒充 PPG peak 成绩；正式 PTT 全数据结果留后续 benchmark。
