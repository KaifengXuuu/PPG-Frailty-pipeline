# Peak、PPI、HR 与 PRV 合同 / Peak, PPI, HR, and PRV Contract

## 1. 单一公共后端 / Single shared backend

high-quality bypass 路线和 denoised 路线必须调用同一个 `m3_physiology_corrected_v1` 后端。算法输入 profile 固定为 `frailty3_peak_ppg_400_offline_v1`、400 Hz、0.4–8 Hz。若 denoiser 输出不是该 profile 的 canonical waveform/feature adapter，必须显式转换并保存 provenance，不能在 denoiser 内另写 peak helper。

The shared backend prevents a route from winning merely because it used a more favorable peak or PPI definition.

## 2. Corrected peak detection / Corrected peak detection

每个波长独立处理：

1. 最短 observation 8 s；不足时返回 insufficient，不补零伪造长度。
2. 同时计算 polarity +1 与 −1。
3. 使用 10 s local windows、5 s hop。
4. 最小 peak distance 0.30 s。
5. prominence 为窗口 robust scale 的 0.25 倍。
6. overlapping windows 内，0.15 s merge radius 只保留最高 confidence 的**现有 peak location**；不把两峰平均或移动到共识位置。
7. 选择更可信的 polarity；输出 polarity、peak indices、bounded confidence、score、profile 与 reason codes。

Peak confidence is constrained to [0,1]. It is an algorithm confidence descriptor, not a calibrated clinical probability.

## 3. PPI 与 corrected NNI / PPI and corrected NNI

```text
raw_ppi[i] = (peak[i+1] - peak[i]) / fs
valid_ppi[i] = 0.30 s <= raw_ppi[i] <= 2.00 s
```

两个边界均包含。invalid PPI 只把对应 interval mask 置 false，**不删除任一源峰**。当前 `corrected_nni_v1` 的精确定义是 `hard_valid_ppi_without_ectopic_imputation`：

- corrected NNI view = hard-valid PPI 的拷贝；
- 未实现 ectopic interpolation；
- 不按数值去重相同 PPI；
- 不跨 invalid interval 连接相邻差分。

This design preserves event provenance. It does not claim that every hard-valid interval is physiologically normal.

## 4. HR 合同 / HR contract

HR 输出至少要求 8 s、5 peaks 和 4 valid PPI：

```text
primary_hr_bpm = 60 / median(valid_ppi)
secondary_hr_bpm = mean(60 / each_valid_ppi)
```

primary 用于稳健轨迹/窗口汇总；secondary 只作并列诊断。缺少足够事件时返回 no estimate/insufficient，而不是复用上一窗口 HR。

## 5. PPG-derived PRV，而非 ECG-HRV / PPG-derived PRV, not ECG-HRV

本包输出命名为 PRV，因为事件来自 PPG pulse，而非直接 ECG R–R intervals。

### 5.1 Time domain

- observation duration ≥60 s；
- valid PPI time coverage ≥0.80；
- SDNN 使用 sample standard deviation, ddof=1；
- RMSSD、SDSD、NN50 与 pNN50 只使用源索引连续的相邻 valid PPI；
- pNN50 单位为 fraction 0–1，不是 percent。

### 5.2 Frequency domain

- 只使用最长 contiguous valid run；
- ≥120 s 为 exploratory tier；
- ≥300 s 为 confirmatory tier；
- 4 Hz linear tachogram；
- linear detrend；
- LF 0.04–0.15 Hz，HF 0.15–0.40 Hz。

在 60 s duration 但实际只有少量 PPI 时，结果必须 partial 且 PRV 为 null；名义 record duration 不能掩盖低 event coverage。

Frequency-domain PRV below the stated contiguous-duration tier must not be reported as confirmatory.

## 6. RED/IR 双通道语义 / RED/IR dual-channel semantics

- RED 与 IR 独立检测，保存各自 peaks/PPI/status/SQI。
- primary selection 先比较有效状态，再比较 SQI。
- exact SQI tie 选择 RED，保证 deterministic。
- SQI 非有限或两通道均不可用时，selected channel = null。
- 报告 20/50/100 ms one-to-one agreement；50 ms 为 primary agreement。
- `generate_consensus_peaks=false`：不得移动、平均或合并两个波长的 peak indices。

Agreement is a cross-channel consistency measure, not ECG-referenced correctness.

## 7. D8 ECG-reference evaluator / D8 ECG-reference evaluator

ECG R peaks 是心动事件 reference，但 PPG pulse 存在生理 transit delay。因此 D8 分两步：

### 7.1 Training-only delay fit

- 仅 exact training subjects；
- fit role 必须为 training；
- 对每个 ECG peak 寻找其后第一个 0.05–0.60 s 内的 PPG peak；
- 汇总 training matches 的 median delay，并四舍五入为 samples；
- artifact 保存 fs、delay、training roster/hash、matched count、范围与 reference source；
- 没有合法 delay 或 roster 不一致时 fail closed。

### 7.2 Disjoint evaluation

- evaluation subject 不得出现在 delay training roster；
- evaluation fs 必须与 artifact 相同；
- 同时报告 raw PPG peaks 和 delay-corrected peaks；
- 默认 50 ms monotonic one-to-one event tolerance；
- precision、recall、F1、timing median/MAE；
- 仅对 reference/candidate index 均连续的 matched pairs 计算 PPI MAE；
- HR error 使用各自 median PPI；
- 保存 coverage 与 `NO_MATCHED_CORRECTED_PEAKS`。

The evaluator measures the PPG detector against an ECG event reference after a training-only transit-delay correction. It must never be described as “ECG detector accuracy.”

## 8. 路线 benchmark 输出 / Route benchmark output

每个 high-quality/denoiser arm 至少保存：

- event precision/recall/F1 与 timing error；
- PPI MAE、HR error；
- raw 与 delay-corrected scorecard；
- valid event/window/subject/stage coverage；
- no-result/drop/failure counts；
- peak/physiology profile IDs and hashes；
- delay artifact、M2 fold artifact、candidate-window hash；
- paired repeat/fold/seed identifiers。

最终 Frailty 路线选择不能只看 peak F1，也不能只看 BA；必须同时满足 HR/PPI error、coverage、risk 和无泄漏门。

