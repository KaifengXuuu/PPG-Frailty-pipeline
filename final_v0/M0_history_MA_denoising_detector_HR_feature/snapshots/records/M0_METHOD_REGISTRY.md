# M0 Motion Processing Method Registry

- 状态 / Status：`complete`
- 证据来源 / Evidence：逐字节代码读取、AST/逐行审计、输入头部 manifests、输出 EOF manifests、实际 JSON/CSV/Markdown、历史项目记录。
- 统一状态 / Status vocabulary：`strictly_validated`、`implemented_unverified`、`smoke_only`、`failed_or_deprecated`、`not_implemented`、`unknown`。

## 1. 总览 / Registry overview

| ID | 方法 / Method | 主要脚本 | 最终状态 | 可复用价值 |
|---|---|---|---|---|
| F01 | Butterworth/high-pass/band-pass/notch | `funcs.py`, `ppg.py` | `implemented_unverified` | 透明滤波思路；实现需统一/修复 |
| F02 | Aboy++ peak/PPI/HRV | `funcs.py`, `ppg.py` | `implemented_unverified` | 峰/PPI 公共算法候选；需 parity/reference test |
| F03 | IMU EKF/gravity/motion classification | `funcs.py`, `ppg.py` | `implemented_unverified` | motion-state 基础候选；参数和单位需重审 |
| F04 | Multi-reference IMU NLMS ANC | `funcs.py`, `ppg.py` | `implemented_unverified` | 透明 coarse baseline；无独立结果 |
| F05 | CEEMD-lite + NLMS | `funcs.py`, `ppg.py` | `implemented_unverified` | 非平稳分解概念候选；不是完整 CEEMDAN |
| V01 | v7 DWT-AE + 1D U-Net | `pttppg_pipeline_v7.py` | `failed_or_deprecated` | 历史 baseline；泄漏/标签退化证据 |
| V02 | v7.2 no-leak proxy denoiser | `cnnppg_v7.py` | `failed_or_deprecated` | subject holdout 设计；负结果基线 |
| V03 | v7.4 rule/AE/fused motion detector | `pttppg_pipeline_v7_4_noleak_viz_ae.py` | `implemented_unverified` | holdout 边界较好；activity proxy |
| V04 | v7.4 STFT magnitude MaskNet | 同上 | `implemented_unverified` | STFT mask/ONNX 经验；无 holdout效果 |
| V05 | v8 time-mask denoiser | `pttppg_denoiser_v8_masknet.py` | `failed_or_deprecated` | 仅保留失败原因；当前不可运行 |
| V06 | Stage-2 mask denoiser | `pttppg_stage2_denoiser.py` | `failed_or_deprecated` | 仅保留失败原因；结果目录为空 |
| D01 | legacy v8 handcrafted detector | `pttppg_detector_v8_scores_audit_fix9.py` | `failed_or_deprecated` | feature/score audit 经验；不可作主证据 |
| H01 | Hybrid pseudo-supervised denoiser | `pttppg_denoiser_hybrid_core.py` + train | `implemented_unverified` | raw/IMU/baseline A-B、bundle/ONNX 工程经验 |
| H02 | Hybrid preview/A-B visual audit | preview + ab_compare | `smoke_only` | 定性故障观察；非性能证据 |
| H03 | Hybrid ONNX/runtime/dashboard | export/runtime/dash utils | `implemented_unverified` | CPU runtime 原型；contract/parity 未闭环 |
| P01 | PPG-only peak/IBI/auxiliary gate | `ppg_peak_hr_gating_train.py` | `failed_or_deprecated`（当前版本） | scorecard框架可复用；target/metric需重构 |
| P02 | 10-channel PPG+IMU detector A/B | 同上 | `implemented_unverified` | 当前最有希望 detector 候选；需严格 benchmark |

> 本轮没有任何 motion/denoising/heartbeat 方法达到 `strictly_validated`。

## 2. 基础方法 / Foundation methods

### F01 — PPG 基础滤波

- 输入：单通道 PPG，调用方采样率；主旧入口常用 400 Hz。
- 输出：高通、带通、notch 或平滑后的数组。
- 预处理/参数：多个不统一 profile；常见 0.5–5、0.5–8 Hz；`wavelet_denoise` 实际为 Savitzky–Golay。
- 监督/训练：无。
- 数据/split：无固定数据集或 split。
- 指标/结果：没有独立 parity、频响、边界或 reference fixture 结果。
- 确定性问题：notch 函数接收 `notch_freq`，调用却传 `f0`；启用会 `TypeError`。IMU band-pass 默认高截止 520 Hz，超过 400 Hz Nyquist。
- 部署依赖：NumPy/SciPy。
- 状态：`implemented_unverified`；保留透明算法概念，不保留当前函数为最终公共实现。

### F02 — Aboy++ 风格 peak/PPI/HRV

- 输入：预处理 PPG、采样率、BPM范围；可分窗处理。
- 输出：peak indices、HR series、PPI/RR 和 HRV。
- 固定参数：历史默认 40–180 bpm；10 s 分窗包装只提交末 4 s。
- 监督/训练：无模型训练；规则与自适应 cutoff。
- 指标/结果：没有与 ECG 或固定 fixture 的正式 parity/accuracy 结果。
- 确定性问题：
  - `reject_artifacts` 位置参数错位，把 `fs` 当 `lower_bpm`；
  - window wrapper 查找不存在的 `HRi`，又把 BPM 写回无量纲状态；
  - 首 0–6 s 与末尾不足整窗部分不提交；
  - `ppg.py` 的 RR “deduplicate” 比较 RR 数值而不是峰时刻，会删除正常稳定心搏。
- 部署依赖：NumPy/SciPy；`funcs.py` 与 `ppg.py` 双实现。
- 状态：`implemented_unverified`；M3 必须做 parity/reference test 后才能复用。

### F03 — IMU EKF、重力去除和 motion classification

- 输入：AX/AY/AZ、GX/GY/GZ、采样率、可选 bias/calibration。
- 输出：动态加速度、姿态/重力估计、motion/static label/score。
- 算法：静止段 bias、EKF 姿态、重力向量扣除、阈值分类。
- 确定性问题：已估计初始姿态却传 `init=None`；`use_ekf=False` 仍执行 EKF；旧 UI/代码单位与通道行为不一致。
- 结果：没有独立跨数据集、subject-level CI 的基础函数 benchmark。
- 状态：`implemented_unverified`；思想可保留，参数/单位必须在 M2/M3 固化。

### F04 — Multi-reference IMU NLMS ANC

- 输入：PPG + 多路 IMU reference；调用方定义 filter length、step size 与 reference normalization。
- 输出：estimated artifact、residual/coarse-clean PPG。
- 监督：自适应最小均方误差，无 clean label。
- 假设：IMU reference 与 PPG artifact 存在线性、短时近似稳定关系。
- 数据/结果：旧 Dash 中可视化运行；没有固定 subject split、scorecard 或 HR/PPI 改善指标。
- 风险：可能删除与运动相关但生理真实的 PPG；无 reference 时平滑 residual 不能证明 clean。
- 部署：NumPy，可轻量实现。
- 状态：`implemented_unverified`；只可作为 transparent coarse baseline。

### F05 — CEEMD-lite + NLMS

- 输入：PPG、IMU/motion reference。
- 输出：按 IMF/频带选择后再用 NLMS 处理的 residual。
- 监督：无；以分解频率/与 IMU 关系作启发式选择。
- 实现边界：项目实现是 CEEMD-lite/self-reference 近似，不应写成完整标准 CEEMDAN。
- 数据/结果：无固定 output、split、严格 metric 或 deployment artifact。
- 依赖：NumPy/SciPy；计算成本高于简单 ANC。
- 状态：`implemented_unverified`；若在 M4 复用，必须与真正 EMD/EEMD/CEEMD/VMD/SSA 分开命名。

## 3. v7 系列 / v7 family

### V01 — v7 DWT-AE detector + 1D U-Net denoiser

- 输入：PTT `pleth_4/5/6`、accelerometer、gyroscope；setup2 再输入 ECG、ECG peaks、p6。
- 输出：detector fold JSON；denoiser fold JSON；comparison JSON。
- 预处理：默认 500 Hz、6 s window、1 s hop；PPG 0.5–8 Hz；DWT db4 level 2。
- 监督目标：motion label=`raw accel RMS > 0.8`；denoise target=同一 p5 的带通 z-score。
- 模型：DWT压缩 PPG→CNN+BiLSTM AE；1D U-Net。
- split/protocol：GroupKFold/GroupShuffleSplit；5 folds、20 epochs。
- 泄漏：AE 阈值在被评估数据自身取 95% 分位；validation 同时选 epoch 和报告；setup2 将 ECG/peak label 作为推理输入；`holdout` 实为最后一折。
- 实际结果：detector F1≈.095、BA≈.050；setup1/2 多数 SNR 为负。
- 部署：PyTorch；setup2 不可部署。
- 状态：`failed_or_deprecated`。

### V02 — v7.2 no-leak proxy denoiser

- 输入：固定 8 通道；ECG peaks 只作窗口 HR supervision，不作为推理输入。
- 输出：`results_v72_noleak` 的 split、AE、CV/holdout CSV/JSON/PNG。
- split：先固定 external subject holdout；train 内 inner train/val 与 5-fold CV。
- 协议改进：holdout 不参与 threshold/训练；比 v7 严格。
- 残留偏差：fold validation 仍同时 early-stop/fit threshold/report；clean target 仍为 noisy PPG proxy。
- 实际结果：AE 因 `empty_inner_train_or_val_sit_clean` 跳过；walk/run 的四组 holdout SNR 均为 -5.43 到 -7.39 dB，ECG-HR MAE 33.44–37.80 bpm。
- 状态：`failed_or_deprecated`；可作为负结果和 split 改进参考。

### V03 — v7.4 rule/AE/fused activity detector

- 输入：`pleth_1/2` + IMU；sit=0、walk/run=1 activity label。
- 特征：10 PPG + 27 IMU；单特征 threshold；OR/AND；PPG-AE；fused rule。
- 参数：lag ±5 window steps，按同序号 PPG/IMU 特征绝对 Spearman 最大化。
- split：subject train/holdout；阈值、lag、AE threshold 均来自 train subjects。
- 风险：同序号特征语义不一致；拼接所有记录后 shift，跨文件/subject 边界；CV AE validation 兼作 epoch selection；主 OR 规则全 motion。
- 实际 holdout：OR BA .500；AND .573；AE .649；fused AND .670。
- 输出：`detector/cv_summary.json`、artifact JSON、NPZ/PT、confusion/lag PNG。
- 部署：规则 NPZ + PyTorch AE；无独立完整 runtime。
- 状态：`implemented_unverified`；只能称 activity/motion proxy detector。

### V04 — v7.4 STFT magnitude MaskNet

- 输入：pleth2/pleth1 STFT magnitude + 37 broadcast features = 39 channels；原始 8-channel 名义输入并未全部进入模型。
- 输出：0–1 magnitude mask；使用 noisy phase iSTFT；walk/run PT、ONNX、`.onnx.data`、meta、subject-a 表。
- target/loss：bandpass PPG proxy、sit template、soft peak/ECG delay、mask regularization。
- split：train/inner-val；hold records 未用于 denoiser evaluation。
- 错误：subject `a` 经 round/int 后无梯度；positive second derivative 用 ReLU 与真实峰顶曲率相反；sit template 未峰对齐；峰配对无合理上限/一对一。
- 结果：walk val loss .9878；run .7024；所有 `a=1.0`；无 holdout waveform/SNR/peak/IBI 指标。
- 状态：`implemented_unverified`；仅工程 artifact，不是有效性证据。

### V05 — v8 time-mask denoiser

- 输入：8 time channels + 37 broadcast features。
- 输出预期：walk/run model、a table、metrics/summary。
- 目标：同一 bandpass pleth2；第一 train subject 首段 sit template shape loss。
- 运行阻断：`B,F,Tt=mag_shape` 覆盖 `torch.nn.functional as F`，随后 `F.interpolate` 必然失败；变长 peak tensors 无法由默认 DataLoader collate。
- 其他错误：time mask复制到所有频率，frequency smooth loss恒0；`lam_shape`未应用；phase方向相反；holdout自身拟合 phase/a；SNR before error恒0。
- 实际输出：`results_denoiser_v8` 0 文件。
- 状态：`failed_or_deprecated`。

### V06 — Stage-2 denoiser

- 输入：47 channels（raw8 + 39 broadcast，其中2个固定0）。
- 输出预期：CV/final PT/ONNX、metrics、stage2 summary。
- 监督：稀疏 ECG impulse BCE、pseudo shape L1、frequency smooth；subject `a∈(0.5,1.5)`。
- split：subject holdout + train GroupKFold。
- 阻断/泄漏：变长 peaks collate；`a.detach().item()` 无梯度；phase符号错误；CV前使用全部 train subjects phase（含 fold validation）；最终用 external holdout 选 epoch并再次报告；validation loss不含ECG项。
- 实际输出：`results_stage2` 0 文件。
- 状态：`failed_or_deprecated`。

## 4. Detector、Hybrid 与 Heartbeat / Detector, hybrid, and heartbeat

### D01 — Legacy v8 handcrafted score detector

- 输入：PPG + IMU，500 Hz；window 1/2/6 s，hop .5/1 s。
- 算法：PPG 10 + IMU 27 handcrafted features；sit-clean AE anchor；Mahalanobis scores；global lag；logistic fusion。
- split：subject 80/20 holdout；train subjects GroupKFold。
- 固定参数：lag ±5 s、AE q=.2、10 epochs、covariance shrink .1。
- 泄漏/错误：全 train subjects fit mu/cov 后才做 CV；lag/shift跨记录和subject；图调用顺序异常被吞；unknown activity=motion；旧 NumPy 时 bandpower可能全0。
- 结果（2 s/.5 s）：fused holdout BA .9880/F1 .9881/AUC .9995；PPG-only BA .7203；IMU-only BA .9992；global lag corr≈.0535。
- 输出：summary JSON、bundle NPZ、audit JSON/PNG；旧 `results_detector_v8` schema 与当前 audit schema 不同。
- 部署：需手工复制 feature/score/lag/postprocess；非 ONNX runtime。
- 状态：`failed_or_deprecated`；仅历史参考。

### H01 — Hybrid pseudo-supervised artifact denoiser

- 输入/shape：
  - `raw_imu`：2 normalized PPG + 9 standardized IMU = `[B,11,3000]`；
  - `raw_imu_baseline`：再加2 baseline clean +2 artifact = `[B,15,3000]`；
  - 500 Hz、6 s、hop1 s。
- 预处理：PPG 0.5–8 Hz；dynamic acc/gyro/magnitudes/jerk；81维 lag bank ridge alpha=8。
- priors：sit beat template（IBI 5 bins）、ECG→PPG delay 80–450 ms/默认200 ms、linear baseline amplitude anchor、IR prominence fallback。
- 网络：1D U-Net residual artifact predictor；`clean_hat=raw_norm-artifact_hat`。
- loss：artifact1、clean.35、sit.75、peak.2、decorr.12、slope.18、anchor.12。
- 数学问题：artifact与clean L1完全等价，是同一代理目标重复权重；`base_art_norm` 形参未使用。
- split：15 train / 3 val / 4 holdout；holdout只写入 split，未推理/评分。
- 结果：raw_imu val .54578；baseline val .45273；不能解释为真实去噪改善。
- 其他风险：PTT IMU单位假设错误；whole-record ridge是batch/transductive；overlap-add未覆盖边界输出0；最早窗口截断采样。
- 输出：PT、meta/history/splits/delay；raw variants有ONNX与external data。
- 状态：`implemented_unverified`。

### H02 — Preview / visual A-B

- 输入：model PT + CSV；A/B使用两个bundle。
- 输出：8 个实际 preview PNG；raw/linear/hybrid曲线。
- 评价：无数值指标、CI或 blinded selection；所谓 raw 已经滤波两次。
- 状态：`smoke_only`；只能作定性补充和失败观察。

### H03 — Hybrid ONNX/runtime/dashboard

- contract：ONNX `model_input → artifact_hat`；dynamic batch、固定 time；preprocessing/ridge/normalization/OLA在Python外部。
- 依赖：NumPy/Pandas/SciPy/ONNX Runtime CPU；dashboard再依赖Plotly。
- parity：只在随机 Gaussian tensor 比 PyTorch/ONNX最大差；无 pass tolerance、无 CSV→输出端到端 parity。
- 风险：`.onnx.data`依赖未显式列入meta；runtime双实现漂移；motion mask截断/补False不对齐时戳；cache不感知文件更新。
- 状态：`implemented_unverified`；不是 production-ready。

### P01 — PPG-only peak/IBI/auxiliary gate

- 输入：重采样/标准化后的 PPG `[B,1,T]`；主 gate 不使用 IMU。
- 数据：PTT、simultaneous、iAMwell、MIMIC、VitalDB；external dataset=SIM；special MIMIC extra。
- target：ECG R-peak/annotation Gaussian peak；ECG RR dense track；部分 activity gate。
- 模型：1D U-Net + peak/IBI/gate heads；IBI bounded .3–2.0 s。
- loss：peak1、beat.1、IBI.35、gate.25、domain默认实际0、worst-domain.25。
- split：254 train subjects、63 internal holdout、50 extra；train内5-fold；OOF threshold；LODO默认关闭。
- 核心错误：未把 ECG→PPG delay反馈给 target；delay仅事后分析。HR(bpm)未计算；RR异常未clip；overlapping windows重复计event；dense IBI按duration/sample/window重复加权。
- 结果：event F1@20ms CV .3870、holdout .3780、extra .1540、SIM .0948；gate F1 CV .9066/holdout .8695/extra .4690；VitalDB holdout IBI MAE 18.2699 s。
- 输出：完整 scorecard/JSON/plots/PT/ONNX bundle。
- 部署风险：metadata声称已有bandpass但loader不一致；ONNX无dynamic axes；per-record zscore非streaming；无正式runtime；JSON允许NaN。
- 状态：当前版本 `failed_or_deprecated`；scorecard框架可复用，target/aggregation/runtime必须重构。

### P02 — PPG+IMU Motion Detector A/B

- 输入：PPG、dynamic acc xyz、gyro xyz、acc/gyro/jerk magnitude = 10 channels。
- 数据：PTT train/val/holdout；SIM external；SIM缺gyro时补0。
- 模型A：denoiser-style encoder classifier，随机初始化，未加载 hybrid weights。
- 模型B：直接训练 Light CNN。
- split：45 train records、12 validation、9 holdout、13 external；threshold按validation BA。
- 结果：PTT val/holdout两模型均1.0；SIM A F1 .7542/BA .7699/AUC .8269；B .7634/.7802/.8642。
- 局限：大量 overlapping-window pooled scores、无 subject-level CI、外部数据单一。
- 输出：每模型 PT/ONNX/meta/export status/curves + benchmark summary。
- 状态：`implemented_unverified`；M4.1 中优先作为候选，但必须与规则detector和subject-level协议比较。

## 5. 避免重复试验规则 / Non-duplication rules

后续任何新 motion/denoising/heartbeat 实验必须在 config/report 中声明：

1. 相对上述 registry 的方法 ID 和真实新增点；
2. 是否仍使用 noisy-self proxy、activity label 或 ECG timing；
3. 如何避免已记录的 split/threshold/delay/aggregation 泄漏；
4. raw/no-denoising 与 high-quality-only 基线；
5. HR/PPI/coverage/失败状态，而不是只报告平滑波形或 proxy SNR；
6. 可复现 preprocessing/runtime contract 和独立 subject/dataset 证据。

