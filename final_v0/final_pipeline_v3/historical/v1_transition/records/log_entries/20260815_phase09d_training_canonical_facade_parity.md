# Phase 09d — canonical training facade parity / canonical 训练门面一致性

- Status / 状态：implemented; the expanded training suite passes 31/31 tests.
- Scope / 范围：three parity tests were added after the phase09c protocol gate.
- Process / 流程：canonical singular paths for aggregation, metrics, OOF and bundle were
  imported through their public APIs and compared with the plural training authorities.
- Result / 结果：drop coverage, metric formulas, exact OOF roster validation and the
  §5.14 required metadata set now have one implementation authority. Formal bundle export
  additionally requires the caller to state strict_metadata=True.
- 中文说明：新增三项门面一致性测试；canonical 单数路径不再维护第二套公式或
  metadata schema。当前 training 相关测试总计 31/31 通过。
