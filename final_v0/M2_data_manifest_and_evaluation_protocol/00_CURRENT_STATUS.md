# M2 当前状态 / Current Status

- 里程碑：`M2`
- 状态：`contract_and_registries_defined_no_model_rerun_yet`
- 日期：2026-08-15
- 写入范围：仅 `final_v0/`
- 原始数据：只读

## 已冻结

1. Frailty3 29 名受试者、261 个八通道 CSV 的文件级与受试者级清单。
2. B=baseline、R=relax/recovery、S=stand-and-sit、W=walk；B/R 为静态 activity 范围，S/W 为 motion；仅确认 S/W 先于 Relax。
3. 历史 SGKF 注册表只用于复现；修正 shuffle-group 映射后的、每折三类齐全且类计数差不超过 1 的 subject-level SGKF 为未来唯一主注册表。
4. 5 repeats、5 folds、seeds `42,10042,20042,30042,40042`。
5. fixed epoch、no early stopping；OOF 数据不进入训练循环，训练完成后只评估一次。
6. 非独立测试结果统一命名 `oof_validation_*`。

## 尚未完成

- 所有候选尚未在未来主注册表上重跑；历史绝对分数不能直接并入未来 leaderboard。
- preprocessing version 将在 M3 冻结；M2 只要求每个结果显式携带该引用。
- R/S/W 编号含义、完整动作顺序、每次动作时点与重复定义仍未确认。
- PTT 双波长映射冲突、若干外部数据缺少波长/放置/冻结快照等问题仍是使用门。
