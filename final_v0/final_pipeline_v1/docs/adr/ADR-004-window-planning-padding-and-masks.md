# ADR-004: Window planning, padding, and masks / 窗口规划、填充与掩码

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§5.3, 5.8, 6.2, 7.4, 8.1

## Decision / 决策

所有 engineering 和 DL 窗口由一个 `WindowPlan` 生成，显式保存 `fs`、
`start_sample`、`end_sample`、`valid_length`、`padding_mask` 和
`source_record_id`。不得在 dataset/model 内复制隐藏切窗逻辑。

- engineering reference: complete 10 s windows, 5 s hop, no padding；
- raw-DL reference: 5 s windows；10 s 为命名 ablation；
- end alignment、overlap、window cap、short-record action 和 padding 均为配置；
- feature matrix: K=32；长记录按进度均匀取样，短记录在 fold-local transform 后
  右侧零填，且 `row_mask=false`；
- zero 仅代表标准化中性填充值，mask 必须阻止其被解释为观察到的生理数据。

Offline zero-phase 处理绝不静默切换 causal；短输入只可 reject、显式 pad，或使用
另行登记且经过 impulse/peak-location 测试的 reduced-padlen profile。

