# M2-DATA-001 — 双 Fold 注册表、5×5 主协议与阶段语义

- 日期 / Date：2026-08-15
- 状态 / Status：`user_confirmed_contract_defined_candidates_not_rerun`
- 来源 / Source：用户本回话明确确认；代码、数据和历史结果只读审计

## 决定 / Decision

1. 保留历史 scikit-learn 1.4.2 SGKF membership 仅作复现；它不是未来主协议。
2. 建立修正 shuffle-group 映射且类别均衡的 subject-level 5-fold SGKF，作为今后全部路线与 benchmark 的唯一主协议。
3. 固定 5 repeats、seeds `42,10042,20042,30042,40042`；所有候选统一重跑。
4. B=baseline、R=relax/recovery、S=stand-and-sit、W=walk；只确认 S/W 在 Relax 前，不补全编号含义或总时序。
5. 主训练为 fixed epoch、no early stopping；outer OOF 不进入训练循环，只在训练完成后评估。
6. 无独立 test 时统一 `oof_validation_*`，历史命名错误必须在协议注册表中显式标记。

## 影响 / Consequence

历史分数只能在历史 registry 内复现和讨论，不能与未来 corrected registry 的绝对分数直接排名。M4–M8 所有候选、benchmark、消融和 Top 5 必须引用未来物化 membership、M2 dataset version 与 M3 preprocessing version。
