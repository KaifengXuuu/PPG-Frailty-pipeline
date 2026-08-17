# Profile 与移动平台绑定 / Profile and Mobile-Platform Binding

## 1. 产品边界 / Product boundary

用户确认的形态为：RED/IR PPG + 3-axis ACC + 3-axis GYRO 穿戴采集端，连接血压仪大小、带中心屏显的处理中心。穿戴端初期负责采集、timestamp、packet sequence、缓存和传输；中心端执行 M3 preprocessing、M1 routing、feature/model inference 与屏显。

The wearable is a sensor/transport endpoint; the central unit is the primary compute node.

## 2. Profile 解析门 / Profile-resolution gate

运行参数只能来自 versioned profile ID。公共入口必须验证：

- modality；
- exact sampling rate；
- channel order；
- phase mode；
- algorithm key；
- registry ID/hash。

调用点不得覆盖 cutoff、order、gravity method 或 initialization threshold。profile mismatch 抛出显式错误；它不是可自动修复的数据问题。

## 3. Offline training 与 mobile inference / Offline training vs mobile inference

| 域 / Domain | Offline/research | Mobile/streaming |
|---|---|---|
| PPG | static/motion/peak/denoiser registered zero-phase profiles | `mobile_ppg_400_causal_v1` |
| IMU | 当前没有 active offline IMU profile | EKF primary 或 LPF comparator causal profile |
| detrend | offline PPG linear | mobile PPG none |
| state | record/window processing | persistent SOS, ESKF/LPF, jerk, timestamp state |
| parity target | feature/statistical envelope | fixed-fixture deterministic output |

离线 PPG 的 filtfilt 会平方单向 magnitude；移动 causal 输出存在 phase delay/transient。M9 parity 应比较事件、统计量、coverage 和容差包络，不要求 waveform sample equality。

## 4. Stateful runtime 合同 / Stateful runtime contract

移动端每个 physical session 创建一次 `CausalImuProcessor`：

1. 绑定 profile、fs 与显式 units；
2. 连续调用 `process_chunk`；
3. failed chunk 不提交 filter/estimator/timestamp state；
4. successful chunk 提交 acc/gyro/gravity SOS、last dynamic、global sample index；
5. `no_estimate` 后只允许 `reset_for_new_session`；
6. session reset 必须写入审计日志，不能作为逐窗自动 fallback。

For PPG, the causal profile similarly requires retained filter state. Recreating processors per chunk creates boundary transients and invalidates latency/feature parity.

## 5. 三档候选平台绑定 / Binding to the three candidate platforms

| M1 profile | 候选职责 / Candidate role | M3 当前绑定 / Current M3 binding | 尚需 M9 证明 / M9 evidence needed |
|---|---|---|---|
| `high_performance_x86_64` | golden parity、研究与复杂路线 | NumPy/SciPy float64 reference；所有候选 routes | 500 ms/hop、≤4096 MB、bundle≤500 MB、15–45 W |
| `accelerated_arm64_edge` | 产品性能/功耗折中 | M3 CPU path；后续 ONNX denoiser/classifier 可用 accelerator + CPU fallback | 750 ms/hop、≤2048 MB、bundle≤250 MB、10–25 W；provider parity |
| `value_arm64_sbc` | 高性价比 CPU-only | SQI、Motion、M3 signal core、tabular/ROCKET 类候选 | 2000 ms/hop、≤1024 MB、bundle≤100 MB、5–15 W |

以上均是 provisional budgets，不是实测结果或处理器采购推荐。尚不能声称三档任一已满足 M3 real-time requirement。

## 6. EKF/LPF 部署选择 / EKF/LPF deployment selection

D5 冻结 EKF 为主路线，LPF 为受控 comparator。deployment config 可以在 session 开始前选择已注册 route，并计算 config hash；但：

- 同一 session 不按 sample/window 动态切换；
- EKF no_estimate 不自动改用 LPF 并伪装连续输出；
- 如产品未来需要用户可见的 LPF emergency mode，必须作为新 session/action，显示方法变化和 coverage discontinuity；
- 路线选择最终仍需 M8 predictive utility 与 M9 resource benchmark。

## 7. 传输与时间 / Transport and time

最低 packet fields：device_id、firmware_version、packet_sequence、sensor_timestamp、sampling_rate_hz、channel_order、units、payload、CRC/status。中心端：

- 以 sequence/timestamp 识别缺包、重复、乱序和漂移；
- 生理间期使用重建的 monotonic sample time，不使用 wall clock；
- gap 只按 M3 bounded repair 规则处理；
- backlog 超限返回 `processing_lag`，不得复用旧结果。

## 8. 屏显与输出 / Display and output

屏幕至少显示 input state、SQI、Motion state（若启用）、coverage、route、当前结果或“无可靠结果”、reason code。不得把：

- `initialization_pending` 显示为 HR=0；
- `no_estimate` 显示为上一窗口值；
- LPF route 显示为 EKF；
- Frailty proxy 显示为姿态/重力 accuracy。

## 9. 依赖与 bundle / Dependencies and bundle

允许 NumPy、SciPy、ONNX Runtime、scikit-learn。M3 signal core 当前主数学路径依赖 NumPy/SciPy；ONNX Runtime 用于后续模型，不应携带第二套 preprocessing。部署 bundle 必须包含 registries、schemas、reason codes、fixtures、source/model hashes 和 parity report，并支持 CPU-only 路径。

