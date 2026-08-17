# ADR-002: Record manifest and fold freeze / 记录清单与分折冻结

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§2, 5.2, 5.12, 8.3
- 上游权威 / Upstream authority: `final_v0/M2_data_manifest_and_evaluation_protocol`

## Decision / 决策

V1 复制并校验 M2 已完成全字节/全数值审计的 261-record、29-participant manifest，
并读取校正后的均衡 subject-level 5×5 registry。主结果使用 seeds
`42,10042,20042,30042,40042`；所有分支读取同一物化成员表，不在运行时重新
调用 splitter。

V1 consumes the byte-verified 261-record/29-participant M2 manifest and the corrected
balanced subject-level 5×5 registry. Every branch reads the same materialized
membership rather than regenerating folds at runtime.

## Frozen invariants / 冻结不变量

- participant 只属于一个 outer fold；其全部文件继承该 fold。
- class/role 标签不由文件内容重新推断。
- B=baseline、R=relax/recovery、S=stand-and-sit、W=walk；仅确认 S/W 在 R 前。
- 任何 fit artifact 都保存 train/oof participant IDs、manifest/fold file SHA 和
  canonical payload SHA。
- `sgkf5_v1.csv` 表示 seed 42 的主可读表；完整重复由
  `sgkf5_repeats_v1.csv` 保存，二者不得混称独立 test。

