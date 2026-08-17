# M3 fold registry field correction / M3 fold registry 字段修正

- 时间 / Date：2026-08-15
- 状态 / Status：corrected_pending_retest
- 原因 / Cause：M2 的 subject_input_order 值是 stable_utf8_bytewise 排序规则名称，不是 subject roster。
- 修正 / Correction：fold union 不变量改为 train/OOF 零交集且 union 数量精确等于 n_subjects=29；实际成员仍来自物化 fold。
- 影响 / Impact：只修正 validator 字段解释，不改变任何 M2 fold 成员。
