# M0 完整结果、算法结论与路线决定

## 1. 范围与完成状态

M0 对 workspace 的历史 motion artifact、动态降噪、heartbeat/IBI/HR 和相关 detector 路线进行了代码—输入—输出—历史记录四方审计。本归档是在既有 M0 完成报告基础上的扩展，补充五类用户指定方向及其可实现性；它不重跑训练、不修复根目录代码，也不提前执行 M1–M10。

当前状态：`M0_local_audit_complete_pending_user_acceptance`。

## 2. 完整扫描基线

| 对象 | 数量/大小 | 实际读取边界 | 结果 |
|---|---:|---|---|
| workspace 文件（排除 `.git`、`final_v0`） | 35,214 / 42,794,025,593 bytes | 全树元数据 | 0 错误 |
| 根目录文件 | 45 | 41 个文本/代码完整字节；4 个非文本元数据 | 0 错误 |
| 代码与 notebook | 52 | 逐字节 EOF + SHA-256 + AST/notebook 结构 | 0 错误 |
| 静态路径引用 | 2,387 | 代码字符串、存在性、输入/输出角色 | 已登记 |
| 输入目录 | 7 / 6,405 文件 / 34,581,621,834 bytes | 每文件最多 65,536 bytes 头部与 schema | 0 错误 |
| 输出目录 | 17 / 28,670 文件 / 8,165,668,577 bytes | 文本完整 EOF；非文本文件名/类型/大小 | 0 错误 |
| 输出文本 | 9,314 | 全字节、SHA-256、行数、schema/指标行 | 完整读取 |
| 输出非文本 | 19,356 | 名称、后缀、大小、角色元数据 | 全部登记 |
| 扫描事务 | 25 | baseline + 7 input + 17 output | 全部存在 |

机器复核：`SCAN_VERIFICATION.json`、`ALGORITHM_DIAGRAM_VERIFICATION.json`、`CODE_DIAGRAM_COVERAGE.json` 与 `FINAL_V0_VERIFICATION.json` 均为 `pass`。

## 3. 历史方法总登记

| ID | 路线 | 主要脚本 | M0 最终状态 | 当前用途 |
|---|---|---|---|---|
| F01 | Butterworth/high-pass/band-pass/notch | `funcs.py`, `ppg.py` | `implemented_unverified` | 透明滤波基线；当前 API/参数需修复 |
| F02 | Aboy++ peak/PPI/HRV | `funcs.py`, `ppg.py` | `implemented_unverified` | 动态 HR 的确定性基线候选 |
| F03 | IMU EKF/gravity/motion rules | `funcs.py`, `ppg.py` | `implemented_unverified` | 可解释 motion 基础候选 |
| F04 | Multi-reference IMU NLMS ANC | `funcs.py`, `ppg.py` | `baseline_only` | 已知会吞真实信号的风险/负对照 |
| F05 | CEEMD-lite + NLMS | `funcs.py`, `ppg.py` | `failed_or_deprecated`（clean-waveform 用途） | 仅保留探索性分解代码和失败经验 |
| V01 | v7 DWT-AE + 1D U-Net | `pttppg_pipeline_v7.py` | `failed_or_deprecated` | 泄漏与负结果历史基线 |
| V02 | v7.2 no-leak proxy denoiser | `cnnppg_v7.py` | `failed_or_deprecated` | subject holdout 负结果基线 |
| V03 | v7.4 rule/AE/fused detector | `pttppg_pipeline_v7_4_noleak_viz_ae.py` | `implemented_unverified` | activity/motion proxy 候选 |
| V04 | v7.4 STFT magnitude MaskNet | 同上 | `implemented_unverified` | STFT/ONNX 工程对照，无 holdout 效果 |
| V05 | v8 time-mask denoiser | `pttppg_denoiser_v8_masknet.py` | `failed_or_deprecated` | 空目录且确定性运行阻断 |
| V06 | Stage-2 mask denoiser | `pttppg_stage2_denoiser.py` | `failed_or_deprecated` | 空目录、训练/评价协议问题 |
| D01 | legacy v8 handcrafted detector | `pttppg_detector_v8_scores_audit_fix9.py` | `baseline_only` | 可解释 detector 对照；IMU/activity 主导 |
| H01 | Hybrid pseudo-supervised denoiser | hybrid core + train | `implemented_unverified` | 当前最完整的 waveform 工程套件，但仅 proxy |
| H02 | Hybrid preview/A-B | preview + ab_compare | `smoke_only` | 视觉故障观察，不作性能证据 |
| H03 | Hybrid ONNX/runtime/dashboard | export/runtime/dash utils | `implemented_unverified` | 部署合同原型，端到端 parity 未闭环 |
| P01 | PPG-only peak/IBI/gate | `ppg_peak_hr_gating_train.py` | `failed_or_deprecated`（当前模型） | 复用 scorecard/多数据 loader；重构 target |
| P02 | 10-channel PPG+IMU detector A/B | 同上 | `implemented_unverified` | 当前最有希望 motion 工程候选 |

严格结论：上述 17 路中 `strictly_validated = 0`。

## 4. 关键结果与评价

| 路线 | 数据与协议 | 真实结果 | 证据解释 |
|---|---|---|---|
| v7 detector | GroupKFold；阈值在被评价数据拟合 | F1≈`0.095`，BA≈`0.050` | 无效/指标组合退化 |
| v7 setup1 | 五折 noisy-self proxy | ΔSNR `-5.83,-6.74,+3.30,+2.50,-7.33 dB` | 所谓 holdout 只是最后一折 |
| v7 setup2 | ECG/peaks/p6 进入推理输入 | ΔSNR `-5.92,-6.66,-6.17,-8.60,+4.29 dB` | 输入泄漏，不可部署 |
| v7.2 setup1 | true subject holdout | walk `-7.39 dB/35.64 bpm`；run `-5.43/33.70` | 严格程度较高的明确负结果 |
| v7.2 setup2 | true subject holdout | walk `-6.40/33.44`；run `-6.68/37.80` | ECG-HR proxy 未改善泛化 |
| v7.4 detector | subject holdout | OR BA `.500`；AND `.573`；AE `.649`；fused AND `.670` | activity proxy，不是 artifact truth |
| v7.4 MaskNet | inner validation | walk loss `.987795`；run `.702436` | 无独立 HR/PPI/waveform holdout |
| v8/Stage-2 | 预期训练 | 两个结果目录均 0 文件 | 未产出且代码有阻断 |
| legacy v8 | PTT internal holdout | fused BA `.9880`/F1 `.9881`；IMU BA `.9992`；PPG BA `.7203` | 近满分由活动/IMU主导；协议有偏 |
| hybrid raw+IMU | train/val proxy | best val `.54578` | 无 clean reference、无 holdout scorecard |
| hybrid + baseline | 同 split/proxy | best val `.45273` | 约 17% objective 下降不等于真实去噪改善 |
| PPG peak model | OOF/internal/extra/SIM | event F1@20 ms `.3870/.3780/.1540/.0948` | target 是未校正 ECG timing，外部泛化失败 |
| PPG-only gate | CV/internal/extra | F1 `.9066/.8695/.4690`；extra AUC `.4088` | 外部 score 方向异常，不可部署 |
| dense IBI | internal/VitalDB | aggregate MAE `4.8271 s/18.2699 s` | aggregate 定义失真，不能作 HR 证据 |
| detector A | external SIM | F1 `.7542`，BA `.7699`，AUC `.8269` | 有希望，但 pooled overlap windows |
| Light CNN B | external SIM | F1 `.7634`，BA `.7802`，AUC `.8642` | 当前 detector 首选，仍缺 subject CI |
| frailty SQI top50 | 112 组 generalization sweep | mean BA `.51079` vs none `.49790`；71 提升/41 下降 | 微弱正信号，不是独立 SQI 质量验证 |

## 5. 输入结构结论

### 5.1 主 PTT-PPG 数据

路径：`physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv`。

实测字段：

```text
time, ecg, peaks,
pleth_1, pleth_2, pleth_3, pleth_4, pleth_5, pleth_6,
lc_1, lc_2, temp_1, temp_2, temp_3,
a_x, a_y, a_z, g_x, g_y, g_z
```

活动由 `sXX_sit.csv`、`sXX_walk.csv`、`sXX_run.csv` 文件名解析。sit/walk/run 是活动标签，不是每窗光学伪影真值。

### 5.2 多数据动态 HR 输入

`ppg_peak_hr_gating_train.py` 默认根为 `physionet.org/files`，并解析：

- `pulse-transit-time-ppg/1.1.0/csv/`
- `simultaneous-measurements/1.0.0/generated_data/`
- `iAMwell Dataset - Intelligent Athlete Monitoring for Cardiovascular Wellness/`
- `MIMIC/mimic_perform_*_csv/` 与指定 MAT extra holdout
- 可选 VitalDB API 的 `SNUADC/PLETH`、`SNUADC/ECG_II`

### 5.3 双波长输入

`PPG_Testing_05_01_2026/tradeali.csv` 实测包含 `IrFinger,RedFinger,GreenWrist,RedWrist,SensorTemp,AmbientTemp`；另有 `tradeprof.csv`、`alifinger.csv` 等。PTT 的 `pleth_1/pleth_2` 被现有代码注释解释为两路 PPG，但正式 BSS 前仍需数据字典确认其真实波长/位置语义。

### 5.4 SQI/frailty 输入

`frailty_3class_classifier.py` 默认 `--data-root PPG_Testing_05_01_2026`。CNN 窗口进入 SQI 时为 `[N,8,T]`：RED、IR、去重力 ACC xyz、低通 GYRO xyz。

## 6. 输出对应关系结论

| 脚本族 | 声明/实际输出 | M0 对应判定 |
|---|---|---|
| `funcs.py`, `ppg.py` | 内存数组、Dash 图、部分用户指定 CSV；无稳定模块目录 | 算法存在，但无固定 scorecard |
| v7 | `results/` | 产物存在；协议无效/泄漏 |
| v7.2 | `results_v72_noleak/` | 16 文件；完整负结果 |
| v7.4 | 实际 `results_v7_4/`，脚本默认仍写 `results_v7_3` | 55 文件；目录名有歧义 |
| v8 MaskNet | `results_denoiser_v8/` | 0 文件 |
| Stage-2 | `results_stage2/` | 0 文件 |
| legacy v8 | `results_v8_audit/` | 30 文件；bias protocol |
| hybrid | `results_hybrid_denoiser_raw_imu*` | PT/meta/history/split/ONNX 等存在；proxy only |
| hybrid preview | `denoiser_preview_output/` | 8 PNG；smoke only |
| heartbeat/motion A-B | `.CNN_results/<run>/` | 687 文件多 run；最新审计 run 产物齐全 |
| SQI/frailty sweep | `results_frailty3/_overfitting_sweep/...` | 有 top50 vs none 消融，但非独立 SQI benchmark |

## 7. 五类扩展审计结论

### 7.1 自适应滤波

- 已有：多输入 IMU-NLMS、CEEMD self-reference 的 32-tap leaky NLMS。
- 不存在：Wiener、独立标准 LMS、RLS。
- 多输入 NLMS 只有当前 6 维样本，没有每通道 FIR delay taps；不能拟合机械传递延迟。
- 运动会真实抬高 HR，参考与目标“无关”的 ANC 前提被部分破坏；当前 Notebook 历史已记录假峰、延迟、删心搏和不稳定。
- 决定：只保留作风险/负对照，不恢复为主路线。

### 7.2 非平稳分解

- `wavelet_denoise` 实际为 Savitzky–Golay 平滑，不能登记为小波。
- DWT 为 `db4 level=2` 只留近似系数并插值回原长；不是阈值去噪，且没有 on/off 隔离实验。
- 有手写 EMD 和 CEEMD-lite；没有真正 EEMD/CEEMDAN、VMD、SSA、CWT、WPT、wavelet threshold。
- 决定：DWT-A2 与 EMD 可作受控 baseline；其他方法需新实现后统一 benchmark。

### 7.3 谱域抑制与轨迹追踪

- 已有 STFT/ISTFT、MaskNet、谱损失和每窗 SoftHRFromFFT。
- 不存在 IMU 频谱局部抑制、候选 HR 峰图、Viterbi/HR-Kalman/Particle 路径。
- 现有 `n_fft=256, fs=500` 的频率格约 `1.953 Hz≈117 bpm`，不能直接作精细 HR 轨迹候选。
- 决定：这是后续优先实现主线；先 Viterbi 离线透明基准，再 Kalman 在线版，Particle 只作复杂多峰消融。

### 7.4 双通道/双波长盲源分离

- 数据具备；raw-waveform PCA/BSS、FastICA、NMF 全部为 0。
- `svm2_dataset_train.py` 的 PCA 只压缩手工特征后接 SVC，不能算 BSS。
- 决定：PCA 是最低复杂度 baseline；FastICA 加稳定性/回退；NMF 应优先在双通道 STFT magnitude 上做，不强行重建时域波形。

### 7.5 SQI

- 当前公式：`0.40*0.5–3Hz功率比 + 0.35*PPI稳定 + 0.15*峰密度 + 0.10*运动惩罚`。
- 偏度/峰度、自相关、模板相关、归一化谱熵、完整 IBI 合理性和 RED/IR 一致性未进入 SQI 主公式。
- ACC/GYRO 先逐窗标准化再算运动 penalty，物理运动幅度被削弱；P5/P95 使用全窗口造成 transductive 风险。
- 决定：SQI 必须成为所有路线的共同质量/拒绝层，但先重构为可解释分量、训练折校准、统一部署合同。

## 8. M0 路线决定

### 保留并优先

1. P02 10-channel PPG+IMU detector A/B 框架。
2. 透明 Aboy++/峰检测作为动态 HR baseline，但先修复确定性错误并做 parity。
3. 新建 spectral suppression + candidate peaks + Viterbi/Kalman + SQI route。
4. 双波长 PCA/ICA/STFT-NMF 作为前端增强与消融。
5. 统一 SQI/coverage/拒绝输出层。

### 仅作 baseline/工程复用

1. IMU-NLMS、DWT-A2、CEEMD-lite 组件。
2. v7.4 STFT/ONNX 工具与 hybrid loader/runtime/scorecard 结构。
3. legacy v8 与 Stage1 作为可解释 activity detector 对照。

### 不恢复为主线

1. 以完整 clean waveform reconstruction 为成功目标的 v7/v7.2/v8/Stage-2 历史路线。
2. CEEMD self-reference + NLMS 作为 clean waveform 方案。
3. 当前 PPG-only peak/IBI/gate 权重和 target 定义。

## 9. 论文表述边界

可写：历史上系统试验了透明滤波、IMU ANC、DWT/AE/U-Net、STFT mask、hybrid、direct peak/IBI 和 motion A/B；subject holdout v7.2 结果为负；external SIM 上 PPG+IMU Light CNN 是有希望但未最终验证的候选。

不可写：恢复了真实 clean PPG；v8 以近 100% 检测 PPG artifact；当前模型已完成 PPG pulse peak 或 HR accuracy；hybrid B 比 A 去噪提高 17%；目录或 ONNX 存在即证明模型成功。

## 10. 本归档未执行事项

- 未新增训练、推理或 benchmark 结果。
- 未修改根目录任何算法代码。
- 未反序列化未知 PT/Pickle/ONNX 权重。
- 未联网下载 TROIKA/JOSS 论文。
- 未写入 `AGENTS.md` 或 `_agent/`。
- 未进入 M1；用户已确认 MAdenoiser 后续路线，但没有批准跳过当前 M0 决策门或开始实现。

## 11. 用户确认的 MAdenoiser 后续路线

### 11.1 串行主线

1. 先完善 SQI：偏度、峰度、自相关、模板相关、归一化谱熵、完整 IBI 生理合理性和 RED/IR 一致性可分批或全部纳入，但必须输出可解释分量和失败码。
2. 在 29-subject frailty 队列上完善 Motion detector 的阈值与 subject-grouped CV，并把 fold-local 的 motion probability/状态作为 SQI 分量。
3. 按统一后端比较四条前端路线：spectral_track_sqi、dual_ppg_bss_sqi、nonstationary_sqi、adaptive_sqi。
4. 先在 pulse-transit-time-ppg 中用标注的 ECG R-peaks 构造 HR/RRI reference，测试动态 HR 与 PPI；若评价绝对 PPG pulse timing，必须先拟合并冻结 ECG→PPG 延迟。
5. 再把四条路线提取的 motion HR/PPI 写成显式、互斥的 frailty 特征块，在相同 subject folds 和相同 seeds 下比较。

### 11.2 选择与证据边界

- 初始候选空间固定为 13 组：baseline，以及四条路线各自的 HR-only、PPI-only、HR+PPI；首轮不搜索任意跨路线子集。
- 路线和特征选择必须在 inner CV 完成，outer fold 只做一次评价；若直接按同一 5-fold 的最高 BA 选最终版本，只能写成 development-selected CV BA，不能写成独立最终性能。
- 每条路线共用完全相同的窗口、subject split、SQI/拒绝策略、event merge、HR/PPI 聚合器与缺失值合同，避免把后端差异误归因于去噪路线。
- 29 人队列当前没有窗口级 optical-artifact 真值；B/R/S/W 只能是 activity/motion proxy，除非用户另行提供或定义监督标签。

状态：route_confirmed_implementation_not_started；当前首要人工决策是 29-subject Motion 的监督目标语义。
