# Phase 09b — generated model cards / 自动生成模型卡

- Status / 状态：implemented; model-suite validation follows in the final gate.
- Scope / 范围：one generator, thirteen model cards, one index, and two registry tests.
- Process / 流程：the generator maps the sole canonical model registry to stable machine
  IDs, eligible representation/signal routes, scientific status, deviations, and limits.
- Algorithm / 算法：coverage is exact set equality between registered machine IDs and card
  filenames; each card is checked for participant-level evaluation and the explicit absence
  of an independent frailty test.
- Result / 结果：all 13 registered routes now have a machine-traceable card.  The cards make
  no performance claim and distinguish single networks, ensembles, project deviations,
  experimental ShapeFormer, and the named MiniROCKET ablation.
- 中文说明：13 个模型路线均有生成式模型卡；模型卡明确 OOF 而非独立测试，并
  明确原论文偏离与尚未运行的计算预算，防止命名越界。
