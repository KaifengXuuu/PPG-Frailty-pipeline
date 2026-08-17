# 五类方法：代码、算法、应用、测试与可实现性审计

## 0. 审计方法

本报告对 52 份现行/归档代码与 notebook 做完整字节校验，并以关键函数逐行复核、全项目符号检索、实际输入 header、输出 JSON/CSV/Markdown 和既有 M0 证据交叉确认。三项并行只读审计分别覆盖：

1. 自适应滤波 + 非平稳分解；
2. 谱域抑制/轨迹 + 双通道 BSS；
3. SQI + 三问题候选路线。

三项审计重新计算的代码总字节为 4,190,267 bytes，与既有 52 文件 SHA-256 manifest 一致，`mismatches=0`。本报告没有执行训练或修改源代码。

## 0.1 总覆盖判定

| 用户指定方法 | 代码存在性 | 实际用途 | 有无有效测试 | 可实现性结论 |
|---|---|---|---|---|
| Wiener | 不存在 | 无 | 无 | 可新实现；作自适应基准 |
| 标准 LMS | 无独立实现 | 部分函数名把 NLMS 简称 LMS | 无 | 可新实现 |
| 多输入 IMU-NLMS | 存在 | Dash ANC/HR 可视化 | 只有失败观察，无数值 benchmark | 仅负对照/风险基线 |
| RLS | 不存在；notebook 仅文字 | 无 | 无 | 可新实现；需数值稳定保护 |
| 真正 DWT threshold denoise | 不存在 | 当前只有 DWT-A2 压缩 | 无 on/off 对照 | 可新实现 |
| CWT/WPT | 不存在 | 无 | 无 | 可新实现 |
| EMD | 存在手写 sifting | CEEMD-lite 组成部分 | 仅图形入口 | 可作探索基线 |
| EEMD/CEEMDAN | 不存在标准实现 | 当前仅“CEEMD-lite” | 无 | 可新实现；必须准确命名 |
| VMD | 不存在 | 无 | 无 | 可新实现 |
| SSA | notebook 仅待办文字 | 无 | 无 | 可新实现 |
| STFT + IMU 谱抑制 | STFT 存在，联合抑制不存在 | MaskNet/损失 | 历史 proxy/负结果 | 优先新实现 |
| 候选谱峰 + Viterbi | 不存在 | 无 | 无 | 优先新实现 |
| HR Kalman/Particle | 不存在 | 现有 Kalman 只做姿态 | 无 | 可实现；Viterbi先行 |
| TROIKA/JOSS 复现 | 不存在 | `JOSS` 字样仅为 HRV 库注释 | 无 | 需另行文献授权后精确复现 |
| 双 PPG PCA/ICA/NMF BSS | 不存在 | SVM PCA 只是特征降维 | 无 | 数据已具备，可新实现 |
| SQI | 部分存在 | frailty CNN 选窗/聚合 | 有 generalization 消融 | 必须扩充分量和防泄漏 |

---

# 1. 自适应滤波：Wiener、LMS/NLMS/RLS 与 IMU-ANC

## 1.1 理论前提与步长/收敛权衡

设观测 PPG 为：

```text
d[n] = s[n] + v[n]
```

其中 `s[n]` 是真实心脏信号，`v[n]` 是运动伪影；IMU 构成参考向量 `x[n]`。ANC 学习 `v_hat[n]=wᵀx[n]`，输出 `e[n]=d[n]-v_hat[n]`。

理想前提是：

1. `x` 与 `v` 足够相关；
2. `x` 与 `s` 无关；
3. 短时耦合可由所选 FIR/模型表示；
4. 对齐、单位和统计量稳定。

本项目第 2 条只部分成立：运动开始后真实 HR 通常上升，脉搏幅值/形态也会改变，因此 IMU 强度与真实生理状态相关。最小均方误差算法没有“伪影/真实心率”语义，只会删除参考可预测的 PPG 成分。即使 IMU 中没有直接脉搏波形，只要有限样本内 `x` 与 `s` 存在相关，Wiener/LMS/NLMS/RLS 都可能吞掉真实信号。

| 方法 | 核心更新 | 收敛/代价 | 本项目风险 |
|---|---|---|---|
| Wiener | `w*=Rxx⁻¹ pxd` | batch 最优；矩阵求逆/正则 | 非平稳、相关参考、训练统计漂移；会把运动相关 HR 当可预测项 |
| LMS | `w←w+μe x` | `μ` 小：慢但低稳态误差；大：快但 misadjustment/发散增加；稳定受 `λmax(Rxx)` 限制 | IMU通道尺度/相关性强时难调；活动切换时跟踪慢/吞信号权衡 |
| NLMS | `w←w+μe x/(ε+||x||²)` | 理想条件常用 `0<μ<2`；对参考能量缩放更稳 | 仍不能解决目标/参考相关；大 `μ` 更容易追随真实 HR 变化 |
| RLS | 基于逆协方差递推与遗忘因子 `λ` | 收敛快；参数维度平方计算/存储；`λ` 小跟踪快但更敏感 | 更可能快速吸收运动诱发真实 HR；协方差病态/数值爆炸 |

## 1.2 现有多输入 IMU-NLMS

### 代码位置

- `funcs.py:959-974`：`build_xref_from_motion`
- `funcs.py:976-981`：`standardize_cols`
- `funcs.py:983-991`：`motion_mask_from_labels`
- `funcs.py:993-1030`：`nlms_multi`
- `funcs.py:1033-1066`：`hr_from_anc_pipeline`
- `funcs.py:1070-1130`：整段 `imu_preprocess_and_anc` 位于三引号字符串内，实际不可调用
- `ppg.py:1468-1575`：当前运行副本
- `ppg.py:2871-2876`：Dash 实际调用
- `ppg.py:3240-3260`：时/频图
- 三个 `Arc/ppg_with_detector_v8*.py` 与四个分析 notebook 是同源副本，不应计作独立算法。

### 当前算法

1. PPG 高通；应用分支当前 `notch_hz=None`。
2. IMU 姿态/重力处理。
3. 构造 6 维参考：`a_dyn_x/y/z, AccMag, GyroMag, JerkMag`。
4. 对完整记录逐列 z-score。
5. motion label 形成逐样本门控并扩展 0.25 s。
6. 每样本仅使用当前 6 维 `x[n]`，默认 `mu=.02, eps=1e-6`。
7. residual 再做 0.5–8 Hz 带通、峰检测、HR/HRV。

重要结构缺口：这不是完整多通道 FIR ANC。6 路 IMU 没有 tapped delay line，无法表示传感器/皮肤/机械传播的延迟、相位和频率响应。

### 输入/输出

- 当前 Dash 路径由 `.env` 的 `folderpath1/folderpath2` 和交互选择决定；值不写入归档。
- CSV 要求所选 PPG/IR/RED 和 `AX,AY,AZ,GX,GY,GZ`，可选 `Time`。
- 历史 notebook 保存过外部 D 盘 `base.csv` 与 bias CSV、`FS=400`，只能证明一次本地尝试，不是可复现项目默认。
- 输出仅内存 `y_min,y_clean,y_band,W,gate,peaks,hr,hrv` 与 Dash 图；没有固定 scorecard 目录。

### 已有测试和结果

- `PPG_Analy_Visual_test.ipynb` 保存的项目日志记录：moving/static 误差大、假峰多、PPG–IMU 延迟、会去掉心搏、参数不稳，已决定放弃时域自适应消噪。
- `Arc/ppg_analy2.ipynb` 中 `anc_on=['off']` 因非空列表仍为真；该版本随后 `UnboundLocalError`，无可用指标。
- 当前 `ppg.py` 的 checklist 默认仍是 `['off']`，而运行用 `if anc_on`，默认误启用风险仍在。
- 没有 ΔSNR、HR/PPI MAE、subject holdout、clean reference 或收敛曲线结果。

### 其他确定性风险

- 只按数组下标对齐，无时钟/重采样/lag 校验。
- 全记录 z-score 借用未来统计，不是 streaming contract。
- `notch_filter` 参数名与调用 `f0` 不一致。
- `reject_artifacts(..., rr0, fs)` 把 `fs` 传到 `lower_bpm` 位置。
- motion gate 本身没有 artifact 真值。

判定：`conditional_reuse_only_as_negative_control`。

## 1.3 CEEMD-lite reference + leaky NLMS

### 代码位置与默认参数

- `funcs.py:1139-1159`：Welch/粗 HR
- `funcs.py:1164-1229`：局部极值、样条包络、EMD sifting
- `funcs.py:1231-1320`：`ceemd_reference`
- `funcs.py:1325-1354`：32-tap leaky `nlms_anc`
- `funcs.py:1359-1387`：`remove_ma_cemd_lms`
- `ppg.py:1703-1900`：当前副本
- `ppg.py:3226-3232`：每次分析无条件执行并画图

默认：6 组互补噪声对、噪声 `.2*std(PPG)`、最多 6 IMF、每 IMF 10 次 sifting、停止 `.2`；先从污染 PPG 的 `0.6–3.5 Hz` Welch 主峰估 HR，保护 HR 与二次谐波 ±`.25 Hz`；其余 IMF 按主频/相关阈值分到 motion；随后 32-tap NLMS `mu=.1, leak=1e-4`。

### 理论和实现问题

1. `u_ref` 直接来自被去噪 PPG，不可能与真实心搏独立，吞信号风险高于 IMU reference。
2. 污染主峰占优时可能保护运动、删除心搏。
3. 运动频率与 HR 重合时无法凭主频/相关阈值辨识。
4. 不是标准 CEEMDAN；只是互补噪声 EMD 后按 IMF 序号平均。
5. 不同 realization 的同序号 IMF 不保证物理对应，mode mixing 未处理。
6. 注释称 endpoint mirroring，但实现只加入原始端点。
7. 分母固定 `2*pairs`，某次 EMD/IMF 缺失时仍照除，分量被衰减。
8. 无耗时/内存门；Dash 每次约执行 `12×6×10` 次 sifting 上限。

输出仅内存 `x,u_ref,y_ma,e_clean,W,imfs,residual,motion_idx,cardiac_idx,f_hr` 和图。没有 JSON/CSV 数值测试。判定：EMD sifting 可作探索基线；整条 self-reference clean route 为 `deprecated_for_clean_waveform`。

## 1.4 缺失实现与统一接口

Wiener、标准 LMS、RLS 在代码中均为 0；notebook 的 “LMS/RLS” 只是规划文字。后续获准实施时建议统一：

```text
AdaptiveANC.fit_or_adapt(
    ppg,                       # observed PPG / 观测PPG
    imu_refs,                  # calibrated IMU references / 校准IMU参考
    timestamps,
    record_boundaries,
    config                     # method, taps, mu/lambda, gate, normalization
) -> {
    artifact_hat,
    residual,
    weights_or_filter,
    convergence_diagnostics,
    physiological_preservation,
    valid,
    reason_code
}
```

必须实现 ridge-Wiener、LMS、NLMS、RLS；将每个 IMU 通道扩展为 `M×L` delay taps；每记录独立 lag/标准化；参数只在 train subjects 选择；输出权重范数、瞬态误差和拒绝原因。

## 1.5 必须实现的测试

1. **合成可辨识性**：clean PPG + IMU 经已知 FIR 生成 artifact；扫描 SNR、lag、taps、mu、RLS λ，评价 filter recovery、ΔSNR、pulse amplitude retention。
2. **假设破坏/吞信号安全测试**：运动后真实 HR ramp 与 IMU 强度相关；评价 ΔHR 保留率、真实心搏能量删除量和错误高置信输出。
3. **负对照**：IMU 打乱、相位随机、错位 ±0.1–2 s；不相关参考不得产生虚假改善。
4. **真实 subject holdout**：PTT sit/walk/run，ECG 仅作 reference；按 subject 报 HR MAE、PPI MAE、peak F1、coverage、ΔHR 保留和 bootstrap CI。
5. **在线收敛**：活动切换前后 transient MSE、权重范数、settling time、发散率、CPU/内存。
6. **一致性**：batch 与 streaming normalization/lag contract、记录边界、缺失 IMU 和不同单位。

---

# 2. 非平稳分解：Wavelet、EMD/EEMD/VMD、SSA

## 2.1 `wavelet_denoise` 是误名的 Savitzky–Golay

- `funcs.py:45-50`
- `ppg.py:541-546`
- 应用：`ppg.py:2623` 批量 HRV；`ppg.py:2904-2906` 交互路径。

实现只根据 `level` 选择 Savitzky–Golay 窗长并做三阶多项式平滑，没有 wavelet basis、分解系数、threshold 或逆变换。正式登记名必须是 `Savitzky–Golay smoothing (misnamed wavelet)`。

## 2.2 真正存在的 DWT-A2 压缩

### 位置

- `pttppg_pipeline_v7.py:97-106`，调用 `:182`
- `cnnppg_v7.py:146-164`，调用 `:319`
- `pttppg_pipeline_v7_4_noleak_viz_ae.py:306-322`，AE 调用 `:171-172`

### 算法边界

```text
coeffs = pywt.wavedec(x, 'db4', level=2)
approx = coeffs[0]
output = linear_interpolate(approx, original_length)
```

它丢弃全部 detail，不 `waverec`，没有 hard/soft threshold、SURE、BayesShrink、universal threshold 或 noise variance estimation。v7/v7.2 用作 CNN-BiLSTM AE 输入；v7.4 只用于 detector AE，不是 STFT MaskNet denoiser 输入。

### 已有结果

- v7 setup1/setup2 的 ΔSNR 大多为负，且协议泄漏；不能归因于 DWT。
- v7.2 DWT-AE 因 `empty_inner_train_or_val_sit_clean` 全部跳过；denoiser holdout SNR/HR 明确为负，但也不是 DWT 开关实验。
- v7.4 `use_dwt=true` AE holdout：BA `.64946`、F1 `.68407`、ROC-AUC `.66927`、PR-AUC `.72431`；标签是 activity，且没有 DWT-off 对照。

判定：只保留准确命名的 `DWT_A2_interpolated` baseline。

## 2.3 EMD/CEEMD-lite

基础 EMD 与 CEEMD-lite 见 1.3。它能输出 IMF/residual，但没有精确重构门、mode-mixing 指标、seed sensitivity、边界误差、clean reference 或 subject-heldout HR/PPI 结果。

代码中的 CEEMD-lite 不是独立 EEMD，也不是标准 CEEMDAN。EEMD/CEEMDAN 必须新实现并记录 ensemble seed、噪声幅度、IMF 对齐和失败次数。

## 2.4 完全缺失的方法

| 方法 | 检索结果 | 可实现建议 |
|---|---|---|
| DWT threshold | 0 | `wavedec → train-only noise estimate → hard/soft threshold details → waverec` |
| CWT | 0 | 合法逆 CWT/尺度重构；mask 参数序列化 |
| WPT | 0 | 小波包节点能量、周期性、IMU coherence 选择 |
| EEMD/CEEMDAN | 0 标准实现 | 受控 ensemble、IMF 对齐、确定 seed |
| VMD | 0 | train-only 选择 `K, alpha`，记录 convergence |
| SSA | 仅 notebook “SSA + spectral subtraction”文字 | trajectory matrix、SVD、分组、对角平均 |

## 2.5 统一可实现接口

```text
decompose(x, fs, config)
    -> components, residual, component_metadata, convergence, valid

select_components(components, imu_context, sqi_context, train_fitted_rules)
    -> selected_cardiac, selected_artifact, uncertain

reconstruct(components, selection)
    -> reconstructed_signal_or_spectral_evidence, reconstruction_error
```

所有实现必须保留记录边界、原长输出、确定 seed、参数 JSON、边缘误差、失败码和运行成本。分解只用于 HR 证据时，应允许不重建时域波形。

## 2.6 必须实现的测试

1. **精确重构与边界**：关闭抑制时长度/原信号误差；短窗、常数、NaN、边缘 10% 误差。
2. **合成非平稳组件**：HR ramp/chirp + AM/FM motion + drift + impulse；评价 mode attribution、心搏能量保留、ΔSNR、edge error。
3. **频率穿越安全**：artifact 主频穿越 HR 与二次谐波，检查 HR track/幅值/形态是否被吞。
4. **真实 subject holdout**：raw、bandpass、DWT-A2、DWT-soft、WPT、CWT、EEMD/CEEMDAN、VMD、SSA；统一 HR/PPI/peak/coverage。
5. **参数稳定性**：wavelet/level/threshold、ensemble/noise/seed、VMD `K/alpha`、SSA `L/rank` 全曲面，而非只报最优点。
6. **运行代价/负对照**：无 artifact、纯 artifact、IMU 打乱、极低 SNR；每分钟 CPU、内存、失败率。

---

# 3. 谱域抑制 + 候选峰 + HR 轨迹追踪

## 3.1 仓库已有的相关模块

| 脚本/位置 | 实际算法 | 应用边界 | 结果 |
|---|---|---|---|
| `pttppg_pipeline_v7_4_noleak_viz_ae.py:887-917` | Hann STFT/ISTFT | 默认 `n_fft=256,hop=64,win=256` | 可复用变换 |
| 同文件 `:1004-1058` | 双 PPG magnitude + 37 广播特征的 2D mask | 无 IMU 频谱局部输入；保留 noisy phase | walk/run inner-val `.9878/.7024` |
| `pttppg_denoiser_v8_masknet.py:214-255` | time mask 沿频率复制 | `F` 覆盖导致运行阻断；频率平滑恒 0 | 目录 0 文件 |
| `pttppg_stage2_denoiser.py:135-149,628-650` | time mask 插值并沿频率复制 | 不是频率选择性 suppressor | 目录 0 文件 |
| `pttppg_pipeline_v7.py:68-88`、`cnnppg_v7.py:111-143` | multi-resolution STFT loss、HR-band regularizer | 训练损失，不生成候选峰 | v7/v7.2 负结果 |
| `cnnppg_v7.py:416-443` | `SoftHRFromFFT` | 每 6 s 窗独立 soft-argmax | holdout HR MAE 33–38 bpm |
| `funcs.py:664-761`、`ppg.py:1161-1250` | 5-state EKF | roll/pitch + gyro bias | 不是 HR Kalman |

## 3.2 明确不存在的模块

- TROIKA 代码：0。
- 经典运动 HR 意义的 JOSS：0。
- IMU spectrum subtraction/mask：0。
- candidate HR spectrum graph：0。
- Viterbi HR tracking：0。
- Particle filter：0。
- HR Kalman filter：0。

`ppg.py:1981-2017` 的 `hrv(JOSS)` 只是 `hrv.rri.RRi`、`quotient`、`threshold_filter` 与 HRV统计的库说明，不是 JOSS 运动 HR 算法。不得误引用。

## 3.3 频率分辨率问题

现有常用 `fs=500,n_fft=256`：

```text
Δf = 500 / 256 = 1.953125 Hz
ΔHR = 60 * Δf = 117.1875 bpm
```

它可以服务神经 mask 的局部时频张量，但不能直接作为精细 HR 候选格。新路线应降采样、增加窗长/FFT、使用插值或多分辨率估计，并把算法延迟显式纳入 benchmark。

## 3.4 推荐完整算法

### 步骤 1：对齐与工作采样

- 每记录独立检查 PPG/IMU timestamp、采样率、缺样和 lag。
- 统一工作采样率建议 100 Hz；保留原始索引映射。
- 初始透明配置：8 s Hann window、1 s hop、`n_fft=2048`，以后由 train/validation 固定。

### 步骤 2：联合时频证据

- 分别计算 IR、RED、ACC xyz/magnitude、GYRO xyz/magnitude、jerk 的 STFT。
- 从 IMU energy、PPG–IMU coherence、跨波长一致性形成污染概率 `q_ma(t,f)∈[0,1]`。
- soft mask 只降低候选证据，不宣称得到 clean waveform。

### 步骤 3：候选峰

在 40–210 bpm 内每帧提取 top-K；每候选记录：

- raw/masked spectrum amplitude 与 prominence；
- IR/RED 频率一致性；
- 基频、二次谐波、次谐波支持；
- IMU overlap/coherence penalty；
- SQI components；
- 前一帧生理可达性；
- `missing` 候选及拒绝原因。

候选分数可写成可审计加权和，权重只由 train/validation 选择：

```text
score = w1*ppg_peak + w2*dual_channel_agreement + w3*harmonic_support
        - w4*imu_overlap - w5*spectral_entropy_penalty + w6*sqi
```

### 步骤 4：轨迹

Viterbi 首选作为透明离线基准，最小化：

```text
J = Σt [-candidate_score(t,k_t)]
    + λ1*|HR_t-HR_(t-1)|
    + λ2*|ΔHR_t-ΔHR_(t-1)|
    + λmiss*missing_state
```

Kalman 在线版使用例如 `[HR, dHR/dt]` 状态；candidate/SoftHR 是带 SQI 决定方差的 measurement。Particle 只在多峰、谐波跳转和快速 HR 变化导致非高斯/多模态时作为消融。

### 步骤 5：输出

```text
hr_track.csv:
time_center,hr_bpm,confidence,sqi,motion_score,
raw_peak_bpm,selected_rank,is_missing,reason_code

summary.json:
subject/activity MAE,RMSE,±5/±10bpm,gross_error,
coverage,jump_rate,recovery_latency,runtime

diagnostics.npz:
PPG spectra,IMU spectra,mask,candidate lattice,path costs
```

## 3.5 TROIKA/JOSS 原理边界

本地代码没有这两类方法。当前报告只把“稀疏/高分辨谱估计、运动谱识别/抑制、谱峰跟踪”作为设计关键词，不声明已经精确复现原论文步骤。若需要论文级原理、公式、作者参数和可引用对照，必须另行获得外部联网确认，并只读原始论文/权威来源。

## 3.6 必须实现的测试

1. 已知 HR pulse/chirp + IMU 相关 artifact，覆盖无重叠、谐波重叠、artifact 穿越 HR。
2. 运动开始后真实 HR 上升且 IMU 主频接近 HR；要求保留生理上升或拒绝，不得压平。
3. PTT subject holdout；ECG peaks 生成 reference HR；sit/walk/run 分开报告。
4. 消融：raw argmax、mask-only、candidate-only、Viterbi、Kalman、Particle、无IMU、无双波长、无SQI。
5. IMU 时间偏移 ±.25/.5/1 s；评价性能和 lag recovery。
6. online-only 因果测试：延迟、每秒 CPU/内存、丢帧与恢复。

判定：`highest_priority_promising_but_not_implemented`。

---

# 4. 双通道/双波长盲源分离：PCA、ICA、NMF

## 4.1 代码存在性

全项目 Python 检索只发现：

- `svm2_dataset_train.py:20,1160-1175` 导入并使用 `sklearn.decomposition.PCA`；流程是手工窗口特征 → StandardScaler → PCA 保留 0.80–0.99 方差 → SVC motion classification。

这不是 raw PPG source separation。其输出 `models/svm_motion_*.pkl` 有大量 pickle，但未反序列化，不能作为 BSS 证据。

raw-waveform PCA/BSS=0；FastICA/ICA=0；NMF=0；BSS 测试=0。

容易误判的现有代码：

- `ppg.py:2572-2578`：`IR_RED=(IR+RED)/2`，只是平均。
- `funcs.py:66-70` / `ppg.py:564-569`：检测 IR/RED 峰但最终只返回 IR 峰。
- `frailty_3class_classifier.py`：分别提取 IR/RED PPI/HRV/形态特征。
- v7.4 MaskNet：两路 magnitude 都输入神经网络，但不输出独立源。

## 4.2 可用数据

- `PPG_Testing_05_01_2026/tradeali.csv`：`IrFinger,RedFinger,GreenWrist,RedWrist,SensorTemp,AmbientTemp`。
- 同目录 `tradeprof.csv`、`alifinger.csv` 和其他测试记录。
- PTT：`pleth_1...pleth_6`；当前代码把 `pleth_1/2` 当两路 PPG，但正式实施前必须查数据字典确认波长/位置。

## 4.3 三种可实现方法

### PCA baseline

1. 对 `[IR,RED]` 作 robust centering/scaling。
2. 协方差特征分解，保留两个分量。
3. 以 HR-band energy、autocorrelation、pulse morphology、低 IMU coherence 选择 cardiac component。
4. PCA 只去相关、不保证统计独立；主要价值是最低复杂度对照。

### FastICA

1. 白化、两个分量、`logcosh` 非线性。
2. 多随机 seed 检查 component 稳定性。
3. 处理符号/排列不确定性；每次输出 mixing/unmixing matrix。
4. 通道近共线、饱和或 component 不稳定时回退最佳单通道。

### STFT-NMF

1. 不对去基线后的有符号时域波形直接 NMF。
2. 对双通道 STFT magnitude 做 multichannel NMF，rank 2–4，比较 KL/IS divergence。
3. 以周期性、轨迹连续性、跨通道 loading 和 IMU overlap 选 cardiac basis。
4. 若只服务 HR candidates，不需不可靠的相位重建；若重建波形，必须声明相位来源。

## 4.4 统一接口

```text
separate_dual_ppg(ppg_channels[T,2], fs, method_config, imu=None)
    -> components,
       mixing_unmixing,
       component_scores,
       selected_component,
       stability,
       fallback,
       valid,
       reason_code
```

分量选择必须包含：HR带周期性、自相关、模板相关、峰可解释性、IR/RED一致性、IMU coherence、SQI 和跨 seed 稳定性。

## 4.5 必须实现的测试

1. 已知 cardiac/motion 源与多种 `2×2` mixing matrix；评价 component correlation、SI-SDR、HR MAE、选择正确率。
2. sit 不劣化：IR、RED、均值、best-SQI channel、PCA、ICA、NMF 与 ECG reference 比较。
3. walk/run subject holdout：HR/PPI/peak/coverage/gross error。
4. 退化输入：通道相同、常数、削顶、饱和、极端增益、时间错位；必须稳定回退。
5. 运动频率与 HR 重合；检查是否把真实运动诱发 HR 错分。
6. 无 ECG 的本地双波长数据只报 SQI、一致性、重复性、coverage，不冒充 accuracy。

判定：`data_available_implementation_missing`；作为 spectral tracker 前端与消融，不预设优于单通道。

---

# 5. SQI：偏度/峰度、自相关、模板、谱熵、IBI 合理性

## 5.1 当前准确实现

核心：`frailty_3class_classifier.py:1671-1766`。

窗口来自 `extract_cnn_windows_from_file`，shape `[N,8,T]`：RED、IR、去重力 ACC xyz、低通 GYRO xyz。随后每窗每通道 robust 标准化并 clip `[-8,8]`。

每窗当前算法：

1. RED/IR 中选标准差更大者。
2. Welch 计算 `0.5–3.0 Hz` 功率占总功率比。
3. `find_peaks(distance≈0.28fs,prominence=.3)`；至少 3 峰时：
   - `PPI_stability=1/(1+std(interval)/mean(interval))`；
   - 峰密度 clip 到 1。
4. `motion=RMS(|ACC|)+.25*RMS(|GYRO|)`；`penalty=1/(1+max(0,motion-1))`。
5. 总分：

```text
SQI_raw = .40*pulse_band_ratio
        + .35*PPI_stability
        + .15*peak_density
        + .10*motion_penalty
```

6. 全窗口 P5/P95 映射 `[0,1]`。
7. `sqi_keep_mask` 每 key 保留 top70/top50、至少一窗；`aggregate_by_key_with_quality` 可质量加权平均概率。

## 5.2 用户点名分量的覆盖

| 分量 | 当前是否进入 SQI | 现有相关代码 | 缺口 |
|---|---|---|---|
| skew/kurtosis | 否 | `time_features:255-270` 只作传统特征 | 需对形态副本计算并校准 |
| autocorrelation periodicity | 否 | 分类器无 SQI 自相关 | 只在 35–210 bpm lag 区间取归一化峰 |
| template correlation | 否 | hybrid 有模板但作伪目标 | 需中位搏动模板、有效 beat 数 |
| spectral entropy | 否 | `spectral_features:273-302` 非归一化 entropy，只作传统特征 | 需除 `log(Nbins)` 并明确频带 |
| IBI plausibility | 部分相关，不在 SQI公式 | `clean_pp_intervals:331-345` 用 35–210 bpm + 4×MAD | 需有效率、异常率、漏/双峰诊断 |
| RED/IR agreement | 否 | morphology/per-window features 有 corr/lag/ratio | 需两路峰匹配、lag、相位与退化标志 |
| motion | 是 | 当前权重 `.10` | 逐窗标准化后物理幅度被抹弱 |

## 5.3 已有实验

来源：

`results_frailty3/_overfitting_sweep/20260630_0630_overfitting_sweep_generalization_rank2/analysis_report_20260703_1202/tables/main_effects_generalization.csv`

112 组成对配置：

- none：mean subject BA `.497900`，median `.500463`，best `.576852`，mean worst-class F1 `.385712`。
- top50：mean `.510789`，median `.511111`，best `.581481`，mean worst-class F1 `.389670`。
- 平均差 `+.012889`；71 对提升、41 对下降。
- 当前最佳 top50 配置：small-InceptionTime、mean-prob，BA `.58148±.05571`。

判定：有微弱正向信号，但提升不稳定；不是 SQI 单元测试、人工质量验证或外部 benchmark。

## 5.4 关键风险

1. ACC/GYRO 先逐窗单位方差标准化，再估 motion，物理强度信息被削弱。
2. RED/IR 同样先标准化，选标准差更大通道近似随机。
3. P5/P95 读取包括测试窗口，属于 transductive leakage。
4. CNN 外层 test fold 同时作 early-stop validation，消融仍混入模型选择偏差。
5. 最终全数据模型没有复用 SQI 过滤；CV/部署分布不一致。
6. ShapeFormer 和传统模型路径不使用 SQI。
7. 无版本、component 输出、flags、单位或失败码。
8. 固定权重无人工质量/ECG/downstream 校准。
9. 可能系统删除与年龄、frailty、疾病相关的真实形态，必须做 fairness/coverage 审计。

## 5.5 新统一 SQI 接口

```text
compute_ppg_sqi(
    red_raw, ir_raw,
    acc_raw, gyro_raw,
    fs,
    peaks=None,
    reference_template=None,
    calibration=None
) -> {
    score,
    components,
    valid,
    flags,
    peak_indices,
    version,
    units,
    parameters
}
```

实现规则：

- 物理运动量在校准但未逐窗尺度归一化副本上计算；形态在独立标准化副本上计算。
- 自相关只在 35–210 bpm lag 范围。
- 模板相关输出有效 beat 数与离散度。
- 谱熵归一化至 `[0,1]`。
- IBI 输出 valid fraction、CV、MAD outlier、missing/double beat flags。
- RED/IR 输出 zero-lag/max-lag corr、lag、phase/peak match、single-channel fallback。
- 加入 PPG–IMU coherence；固定规则与可学习校准分开。
- 所有 percentile、权重、阈值只在 train fold 拟合并保存；CV/final/holdout/runtime 调同一 API。

## 5.6 必须实现的测试

1. 空/短/常数/NaN/Inf/缺通道/补零/不同 fs 与长度的数值边界。
2. clean pulse 逐级加入白噪声、漂移、削顶、丢样、motion；SQI 总体单调下降。
3. 35/210 bpm、漏峰、双峰、异位、随机 IBI 的生理周期测试。
4. RED/IR 增益/DC/极性不变性、可控 lag、一路损坏与不一致。
5. 静止重力、去重力动态、不同单位、gyro 缺失和同频运动。
6. top50/top70 数量、至少一窗、并列分和 key 不串组。
7. 改变 test fold 分布不得影响 train SQI/calibration/mask。
8. CV、final、holdout、保存/加载后分数与选窗一致。
9. 盲法人工质量标签 + ECG peak 真值；按 subject/activity/dataset 分层。
10. frailty class、年龄、数据集与 subject 的保留率公平性。

判定：`shared_high_value_layer_requires_reimplementation_and_validation`。

---

# 6. 五类方法的统一实现可行性

| 方法族 | 数据是否具备 | 依赖是否常规可得 | 核心难点 | 是否能实现 |
|---|---|---|---|---|
| Adaptive ANC suite | PPG+IMU+ECG reference具备 | NumPy/SciPy | 假设破坏、安全门、对齐 | 是；仅作对照优先 |
| Decomposition suite | PPG+IMU具备 | PyWavelets/SciPy；VMD/EEMD可自实现或后续授权依赖 | 参数曲面、mode mixing、边界 | 是 |
| Spectral tracker | 双PPG+IMU+ECG具备 | NumPy/SciPy | 候选评分、时序路径、延迟 | 是；最高优先 |
| BSS | 双通道数据具备 | scikit-learn/NumPy | 可识别性、分量选择、回退 | 是 |
| SQI | PPG/IMU/peaks具备 | NumPy/SciPy | 标签、校准、防泄漏、公平性 | 是；所有路线前置 |

“可实现”只代表接口、数据和测试前提明确，不代表本轮已经实现或验证。实际编码必须在用户确认后作为新的串行步骤执行，并继续遵守只写 `final_v0/`。

# 7. 用户确认路线后的审计增量

## 7.1 状态不被路线决定改写

用户已确认实施顺序，但本文件前述存在性和测试判定保持不变：

- 谱域轨迹仍为 not_implemented。
- 双波长 BSS 仍为 not_implemented。
- 非平稳分解只有部分、非等价基线，尚无统一测试。
- 自适应滤波只有存在已知信号吞噬风险的局部 NLMS，尚无 Wiener/LMS/RLS 统一套件。
- SQI 仍缺用户点名分量、fold-local calibration 和独立质量标签验证。

因此，路线决定只能写成“可实现并计划测试”，不能写成“算法已完成”或“结果有效”。

## 7.2 实施前置关系

1. 先冻结 SignalRecord、subject split、窗口索引、单位、事件合并和 reference 语义。
2. 完成 SQI-v2 的确定性 unit/synthetic/leakage 测试。
3. 明确 29-subject Motion 标签后，才实现 nested threshold/CV；输出的 fold-local motion score 再进入 SQI。
4. 四条路线只更换前端，候选生成后的轨迹、SQI、拒绝和 HR/PPI 评价后端保持一致。
5. PTT 初测通过后，才允许生成 frailty route feature blocks；所有 OOF 特征必须只由对应训练折拟合的处理器产生。

## 7.3 新增必测项

- Motion：subject-grouped outer CV、inner threshold selection、阈值跨 fold 稳定性、活动/伪影语义混淆审计、29 人全覆盖。
- SQI：分量单调性、缺通道回退、fold-local 校准、Motion 概率融合前后质量–覆盖曲线。
- PTT：ECG R-peak reference、delay-aware PPG 事件、record-level event merge、HR/PPI 与 coverage 成对比较。
- Frailty：相同 split/seeds、13 个预注册候选、nested inner selection、outer test 一次性评价、subject-level bootstrap。
