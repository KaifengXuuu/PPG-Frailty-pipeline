# 结果溯源与命名合同

## 每个正式结果的最小字段

1. `dataset_version_id`、manifest SHA-256、纳入/排除 filter。
2. `fold_registry_id`、registry payload SHA-256、repeat、split seed、fold、train/OOF subject IDs。
3. `protocol_id`、protocol family、fixed epoch、early-stopping source、training seed。
4. `preprocessing_version`、feature schema、model config hash、代码版本。
5. `evaluation_role`、输出层级、coverage、failure/no-result 计数。
6. 指标前缀与独立性声明。

## 命名规则

| 数据角色 | 允许前缀 | 禁止表述 |
|---|---|---|
| 当前 5-fold outer validation | `oof_validation_*` | `test_*`, independent test |
| 严格 holdout 且训练/选择未使用 | `holdout_*` | 自动称 external/independent，除非协议明确 |
| 真正外部数据集 | `external_holdout_*` | 与内部 OOF 混为一个 leaderboard |
| debug/smoke | `debug_*` / `smoke_*` | 正式性能 |

未来 fixed-epoch OOF 的训练循环不得接收 OOF labels/features。即使旧代码在 `select_best_epoch=False` 时最终权重理论上不依赖 OOF，未来实现也必须传 `x_val=None,y_val=None`，训练结束后再单次评估，以消除人为观察和误用面。
