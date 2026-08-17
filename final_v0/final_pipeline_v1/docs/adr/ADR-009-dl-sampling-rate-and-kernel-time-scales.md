# ADR-009: DL sampling rate and kernel time scales / DL 采样率与卷积时间尺度

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§2, 5.3, 6.1, 6.2, 8.3

## Decision / 决策

`x_native`、`x_filter`、morphology、peak timing 与 audit 始终保留 400 Hz。
DL-only 路线可在切窗后用 anti-aliased polyphase 重采样到 100/160/200/400 Hz，
必须保存 source/target fs、up/down、时间映射和 profile hash，且不得覆盖原视图。

Reference raw model 暂以 400 Hz 和 5 s context 保留历史可比性；10 s context 为命名
对照。Kernel 配置同时保存 odd sample length 与 duration seconds。Inception 当前
39-sample 最长 kernel 在 400 Hz 只等于 97.5 ms，模型卡必须说明它是局部形态 baseline，
而非完整 pulse-cycle receptive-field 证明。

Matched ablation 固定 `dl_fs={100,160,200,400}`、物理 kernel duration 集、可选
dilation 与 `window_sec={5,10}`，使用相同 folds/seeds/budget。

