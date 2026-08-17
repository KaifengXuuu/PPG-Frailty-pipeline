# ADR-008: Model naming and paper deviations / 模型命名与论文偏差

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§5.6, 5.10, 6.1, 9

## Decision / 决策

- 当前 32/64/128 filters、kernels 9/9/7、两次 pool-4、GAP 模型命名
  `CompactCNN1D`，不是 Wang-FCN；
- full/small Inception ports 均命名 `InceptionTimeSingleNetwork`，不是原论文
  五网络 ensemble；
- 只有五个独立训练成员、不同确定性 seeds、无共享权重且逐概率精确算术平均的 wrapper
  才可命名 `InceptionTimeFiveMemberProbabilityEnsemble`；
- ShapeFormer 使用显式 `discovery_method`，本地非 PISD 方法必须在模型卡中标
  `experimental deviation`，禁止宣称原实现 parity；
- ROCKET reference 为 10,000 random kernels + ridge；MiniROCKET 是独立消融。

Architecture snapshot 和 model card 是发布门；名称不能随结果表现改变。

