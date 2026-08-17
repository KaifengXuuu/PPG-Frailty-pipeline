# ADR-005: PRV eligibility and time axis / PRV 资格与时间轴

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§3, 5.4, 7.1, 8.1

## Decision / 决策

Pulse 事件始终保留原始 sample index、timestamp、interval endpoints、
`valid_interval_mask` 与 `adjacency_mask`。拒绝 PPI 不压缩时间，不允许 RMSSD、
SDSD、NN50 跨越缺失相邻间期。

- HR/PPI：观察时长至少 8 s、至少 5 peaks；
- time-domain PRV：至少 60 s、有效 PPI coverage 至少 0.80；
- frequency-domain PRV：direct static/reference、至少 300 s 且至少 200 accepted
  intervals，使用真实时间、4 Hz 插值 tachogram 与线性 detrend；
- bands：VLF 0.003–0.04、LF 0.04–0.15、HF 0.15–0.40 Hz；
- SampEn：accepted intervals≥200，m=2，r=0.2×sample SD；
- 不可用值：NaN/null + validity=false，绝不使用生理有效零。

V1 将 300 s 作为“approximately five-minute”的确定性实现；是否放宽至 240 s
列入 V2 人工确认点。

