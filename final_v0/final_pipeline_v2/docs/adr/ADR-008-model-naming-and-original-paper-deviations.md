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
- ShapeFormer literature reference 由可执行的faithful ShapeBlock/IG/OSD模块实现，
  不设运行门禁；其他发现或downstream方法必须使用独立ablation ID，禁止宣称
  PISD parity。可选 `FileBagFusion` 只组合已注册raw encoder，并保持fold-local
  discovery与file-feature transform隔离；
- ROCKET/Ridge 与 MiniROCKET 已从当前可执行 pipeline、catalog 和 study 中退役；历史名称不得被复用为其他 matrix 模型。

Architecture snapshot 和 model card 是必需追溯材料；名称不能随结果表现改变。
