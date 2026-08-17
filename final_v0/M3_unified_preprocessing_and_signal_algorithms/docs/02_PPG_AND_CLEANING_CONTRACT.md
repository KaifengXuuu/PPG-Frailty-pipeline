# PPG 与清洗合同 / PPG and Cleaning Contract

## 1. 输入边界 / Input boundary

PPG 处理在任何滤波、归一化、SQI、denoising 或 feature extraction 之前执行输入合同。

- canonical 双波长顺序：RED, IR；
- 未来 Frailty 主采样率：400 Hz；
- samples 使用 float64 内部计算；
- 显式 timestamps 必须与样本等长、有限、严格递增；相邻间隔相对 1/fs 的 P99 误差不得超过 5%；
- stateful chunk 的首时间戳还必须与上一 chunk 末时间戳保持 5% 内的连续性；
- timestamps 为 null 只表示调用者使用另行登记的均匀网格，不表示“未检查采样率”。

The implementation rejects profile/fs mismatch instead of pretending that data sampled at another rate are 400 Hz.

## 2. 滤波前质量与有限修复 / Pre-filter quality and bounded repair

| 检查 / Check | 冻结规则 / Frozen rule | 结果 / Result |
|---|---|---|
| shape/channel | samples×channels；通道名数量、缺失、顺序和额外数量均检查 | mismatch → invalid |
| non-finite total | 每通道 >1% | invalid |
| internal gap | 长度 ≤0.25 s，且不触及边界 | 线性插值；status repaired；source valid mask 保持 false |
| internal gap | >0.25 s | invalid |
| boundary gap | 不外推 | invalid |
| all non-finite / empty | 无可用样本 | invalid |
| minimum duration | 由调用 profile/算法设置 | insufficient |
| PPG flatline | exact run ≥1.0 s | invalid |
| clipping proxy | min/max occupancy ≥2% 或极值 plateau ≥0.25 s | warning only |

clipping 规则只是 heuristic，因为 M2 没有权威 ADC rails。它不得写成 ADC saturation truth。插值仅修复允许的内部短 gap；`valid_mask` 表示修复前 source finite，`repair_mask` 表示被重建的位置，两者不得合并成一个“都有效”mask。

The clipping rule is a warning, not an ADC-saturation claim. Source validity and repair provenance remain separate.

## 3. 活动 PPG profiles / Active PPG profiles

| Profile ID | 用途 / Use | Bandpass | Phase / detrend |
|---|---|---:|---|
| `frailty3_static_ppg_400_offline_v1` | B/R 静态分析 | 0.2–8.0 Hz, order 3 | offline zero-phase; linear detrend |
| `frailty3_motion_ppg_400_offline_v1` | S/W、motion 路线 | 0.4–8.0 Hz, order 3 | offline zero-phase; linear detrend |
| `frailty3_peak_ppg_400_offline_v1` | corrected physiology backend | 0.4–8.0 Hz, order 3 | offline zero-phase; linear detrend |
| `frailty3_denoiser_ppg_400_offline_v1` | denoiser adapter | 0.4–8.0 Hz, order 3 | offline zero-phase; linear detrend |
| `mobile_ppg_400_causal_v1` | 中心设备流式处理 | 0.4–8.0 Hz, order 3 | causal stateful; no detrend |

所有 profile 的 notch 当前为 disabled。启用 notch 需要新 profile 和实际电源频率/设备证据，不能在调用点暗中打开。

Offline zero-phase applies the forward/backward SOS response; its effective magnitude is the square of the causal magnitude. Therefore causal and offline profiles share design intent but are not sample-by-sample parity targets.

## 4. 原始幅值上下文 / Raw amplitude context

在 detrend、bandpass 和 normalization 前保存：

- DC median；
- source standard deviation；
- robust q95−q05 span；
- raw perfusion/amplitude proxy；
- RED 与 IR 各自的 DC/AC；
- RED/IR DC ratio、AC ratio 与 AC/DC context。

滤波后另存 filtered AC/std/amplitude 和 AC/DC proxy。raw 与 filtered 字段必须用不同名称，不得把标准化振幅解释成生理灌注。训练折幅值风险模型只能产生 `sqi_risk` heuristic，不能证明 gain mismatch 或 sensor failure。

Raw and filtered amplitude descriptors remain distinct. Normalized waveform magnitude must not be interpreted as perfusion.

## 5. 重采样与 scaling / Resampling and scaling

- 非 400 Hz 输入必须通过显式 polyphase resampling；结果保存 source rate、target rate、up 与 down。
- 不能先把 metadata 改成 400 Hz 再跳过重采样。
- waveform view 的 robust scaling 为每窗口 median 与 IQR/1.349、默认不 clip；IQR=0 时 no-estimate/fail closed。
- raw amplitude context 在 scaling 前保存，因此可审计 normalization 是否掩盖 DC/gain 信息。
- classifier/aggregate feature scaler 与 imputer 只在 M2 training subjects 拟合；不能借 window scaling 绕过 fold isolation。

Window scaling and fold scaling solve different problems: the former creates a reversible numerical view, while the latter is a learned training-only artifact.

## 6. 输出与状态 / Output and status

PPG 结果必须包含 profile ID、quality assessment、filtered waveform（若可计算）、raw metrics、filter/resampling metadata、source-valid mask、repair mask 和 status。状态映射原则：

- valid：完整可用；
- repaired：按冻结规则修复，必须保留 repair provenance；
- partial：仅部分事件/特征可用；
- invalid：输入或合同致命失败；
- insufficient：长度或 coverage 不足；
- initialization_pending/no_estimate：主要用于 stateful IMU 或无可靠估计，不得改成零。

M1 输出层映射由公共函数统一处理；特别是 `initialization_pending` 在流中为 `processing_lag`，在 end-of-stream 为 `insufficient_quality`。

## 7. 验收门 / Acceptance gates

1. 400 Hz 下 100-sample 内部 gap 可修复；101 samples 拒绝。
2. 399-sample flat run 不触发 exact 1 s 门；400 samples 触发。
3. 500→400 Hz reference 的长度与 4/5 polyphase metadata 正确。
4. 35 bpm 在静态 zero-phase profile 的有效增益 >0.99；motion profile >0.90；8 Hz zero-phase 有效增益为 0.5。
5. profile/fs mismatch、重复 timestamp、通道换序和 zero-IQR 均 fail closed。

These are engineering contract tests, not a claim that the selected passband is clinically optimal for every subject or motion condition.

