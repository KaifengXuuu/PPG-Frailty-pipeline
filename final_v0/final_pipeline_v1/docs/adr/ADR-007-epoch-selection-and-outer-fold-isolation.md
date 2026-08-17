# ADR-007: Epoch selection and outer-fold isolation / Epoch 选择与外层隔离

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§2, 5.12, 8.3

## Decision / 决策

Trainer 只接收 outer-train 数据和不可变 outer-OFF evaluation handle；训练过程中不向
模型、scheduler、early stopper 或日志回调暴露 outer labels。

允许两种规则：

1. `fixed_epoch`: 配置在运行前冻结；
2. `inner_grouped_selection`: 只在 outer-train participants 内按 participant 分组
   选择 epoch，随后用选定 epoch 在全部 outer-train 上从头 refit。

Reference V1 使用 `fixed_epoch`；smoke 配置为 1，正式参考配置暂定 50。任何
outer-early-stopped 历史结果标为 protocol-incompatible。所有 scaler、SQI 阈值、
shapelets、ROCKET kernels、ridge alpha、calibrator 与 feature selection 同样只用
outer-train。

正式 epoch=50 是否保留，或改用训练折内 grouped selection，列入 V2 人工确认点；
无论选择哪一项，都不得查看 outer 指标后再决定。

