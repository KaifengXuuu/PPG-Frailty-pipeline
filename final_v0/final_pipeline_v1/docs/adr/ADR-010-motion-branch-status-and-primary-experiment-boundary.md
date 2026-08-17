# ADR-010: Motion branch and primary experiment boundary / 运动分支与主实验边界

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§2, 5.5, 6.3, 7.7

## Decision / 决策

主 frailty reference 先对 direct `x_filter` 计算 `Q_rate_pre` 与 `Q_morph_pre`。
高质量 direct 段可提取 rate 与形态学；低质量/motion 段按运行前配置在
`drop` 与某个非恒等 `ArtifactReducer` 之间互斥选择，禁止逐窗看结果后择优。

非恒等 reducer 成功后只生成 `x_ar -> Q_rate_post -> pulse/HR/PPI/eligible PRV`；
`Q_morph_post=not_applicable`。失败不回退成“看似有效”的 raw morphology。

外部 PTT-PPG 仅开发/评价 motion 与 heartbeat recovery，不携带 frailty labels。
内部 S/W 无 ECG 真值时只报告 coverage、rate plausibility、RED/IR rate agreement 与
route stability，禁止称为 ground-truth accuracy。

PTT 波长映射在 M2 仍冲突，因此 V1 不把其 pleth 通道强行命名 RED/IR；需双波长语义的
BSS/光学特征 fail closed，列入 V2 人工确认。

