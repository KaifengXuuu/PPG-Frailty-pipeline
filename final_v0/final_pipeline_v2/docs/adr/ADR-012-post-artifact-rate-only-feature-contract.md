# ADR-012: Post-artifact rate-only feature contract / 去伪影后仅 Rate 合同

- 状态 / Status: accepted_for_v2
- 依据 / Source: contract §§2, 3, 6.3, 7.2–7.7, 9

## Decision / 决策

所有非恒等 `ArtifactReducer` 返回与原时间网格对齐的 RED/IR `x_ar`、版本参数、
diagnostics、confidence、status 与 channel/alignment metadata。下游强制：

```text
x_ar -> Q_rate_post -> common pulse detector
     -> HR/PPI/eligible PRV + coverage/confidence + reducer/IMU provenance
```

以下字段必须是 NaN/null 且 validity=false/not_applicable：`Q_morph_post`、pulse
amplitude/width/rise/decay/slope/area、DC、AC、PI、ratio-of-ratios、morphology
coherence、direct optical power/context。不得从 `x_filter` 复制值冒充 `x_ar`
输出，也不得用 waveform smoothness 声称 clean morphology recovery。

`identity` 不改变波形，按 direct branch 处理，可保留 `Q_morph`。未来任何
morphology-preserving reducer 必须新 ADR、paired-clean ground truth 与 landmark/
area/amplitude preservation tests；不属于当前V2合同。
