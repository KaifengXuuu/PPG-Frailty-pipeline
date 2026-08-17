# ADR-003: Signal views and units / 信号视图与单位

- 状态 / Status: accepted_for_v2
- 依据 / Source: contract §§2, 5.3, 5.4, 6.3, 7

## Decision / 决策

`SignalViews` 显式分离：

- `x_native`: 400 Hz、保留 acquisition-scale RED/IR baseline 的已许可短缺口修复视图；
- `x_filter`: 400 Hz、线性 detrend、三阶 Butterworth SOS、0.2–8 Hz、
  offline zero-phase 的 direct analysis 视图；
- `x_analysis`: direct 路线等于 `x_filter`；非恒等 reducer 路线等于对齐的
  `x_ar`，并标记 `rate_only=true`；
- `imu_processed`: 加速度 m/s²、角速度 rad/s、动态加速度、模长和 jerk。

RED/IR 原始单位保留为 counts/unknown，不以幅值猜单位。ACC 只接受 `g`、`m/s2`
或 `m/s^2`；gyro 只接受 `deg/s` 或 `rad/s`。未知 SI 单位 fail closed。

## Scientific boundary / 科学边界

DC、AC、PI、ratio-of-ratios 和形态学只从 direct `x_native+x_filter` 产生。
非恒等 `x_ar` 只重新计算 `Q_rate` 与 rate/PPI/eligible PRV，永不输出或复制
形态学值。
