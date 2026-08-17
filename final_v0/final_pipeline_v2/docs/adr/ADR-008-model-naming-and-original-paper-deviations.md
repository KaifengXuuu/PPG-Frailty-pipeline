# ADR-008: Model naming and paper deviations / 模型命名与论文偏差

- 状态 / Status: accepted_for_v2
- 依据 / Source: contract §§5.6, 5.10, 6.1, 9

## Decision / 决策

- 当前 32/64/128 filters、kernels 9/9/7、两次 pool-4、GAP 模型命名
  `CompactCNN1D`，不是 Wang-FCN；
- full/small Inception ports 均命名 `InceptionTimeSingleNetwork`，不是原论文
  五网络 ensemble；
- 五成员raw与matrix ensemble必须是两个显式comparison ID；每个都使用exact seeds
  `[42,10042,20042,30042,40042]`、无共享权重和逐概率算术平均；
- ShapeFormer literature-reference只有通过faithful ShapeBlock/IG/OSD gate后才可运行；
  其他发现方法必须使用独立ablation ID，禁止宣称PISD parity；
- ROCKET reference 为 10,000 random kernels + ridge；MiniROCKET 是独立消融。

Architecture snapshot 和 model card 是发布门；名称不能随结果表现改变。
