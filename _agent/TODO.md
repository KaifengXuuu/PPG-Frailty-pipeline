# TODO

状态：confirmed
来源：用户 2026-08-01/2026-08-02 要求、`_agent/arc/PROJECT_HANDOFF.md`、现有 `_agent` 活动文档、相关代码与结果只读复核
最后手动更新时间：2026-08-02
用途：记录项目必须完成项、具体实现顺序、依赖关系、验收条件和可选任务。

## 总体实施顺序

1. 审计历史方法、代码、数据和结果。
2. 定义端到端模块架构、数据契约和移动端约束。
3. 固化数据 manifest、阶段映射和评估协议。
4. 统一并固定 PPG/IMU 预处理、数据清洗、归一化和特征算法。
5. 实现并跑通 motion/SQI、粗降噪、heartbeat、ROCKET、8-channel deep learning、hierarchical 和 Base/Motion/Relax 各模块。
6. 整合全部历史 Sweep/Grid Search，建立历史严格可比 Top 5。
7. 在所有候选代码跑通后，再运行统一 benchmark、评估和报告。
8. 完成统一消融、最终模型选择和独立性能陈述。
9. 完成最终训练、Python-based 移动端 pipeline 整合和导出。
10. 归档旧路线并维护可追溯文档。

以下 `M0–M10` 均为必须完成项。所有 benchmark、正式比较和报告必须位于项目实现及 smoke test 之后；在代码尚未跑通时，不得提前用不完整结果锁定最终路线。

---

## 必须完成项

### M0 完整审计历史 Motion Artifact、动态降噪和 Heartbeat 路线

- [ ] 重新阅读相关代码、结果、scorecard、preview 和历史记录，建立 method registry。
- [ ] 至少覆盖：
  - `funcs.py`、`ppg.py` 中的基础滤波、motion classification、IMU-NLMS ANC、EMD/CEEMD-lite+NLMS。
  - `pttppg_pipeline_v7.py`、`cnnppg_v7.py` 中的 DWT、AE、UNet、STFT/HR loss。
  - `pttppg_pipeline_v7_4_noleak_viz_ae.py` 中的 feature-threshold detector、STFT mask、sit template、ECG delay compensation。
  - `pttppg_denoiser_v8_masknet.py`、`pttppg_stage2_denoiser.py` 中的 STFT magnitude mask。
  - `pttppg_denoiser_hybrid_*` 中的 `raw PPG + IMU`、`raw PPG + IMU + linear baseline`、sit prior、peak prior、clean prior 和 artifact-relation learning。
  - `ppg_peak_hr_gating_train.py` 中的 motion detector、direct peak/IBI/HR extraction。
- [ ] 对每种方法记录输入、输出、监督目标、预处理、固定参数、数据集、subject split、训练协议、评价指标、已有结果、失败原因和部署依赖。
- [ ] 区分：已实现且已严格验证、已实现但未检验、仅 smoke test、已失败/deprecated、尚未实现。
- [ ] 检查历史结果是否存在 subject leakage、阈值在 validation/test 上拟合、错误 ECG–PPG 对齐或只依赖人工波形观察的问题。
- [ ] 保留旧结论：不恢复完整 clean waveform；新实验仅允许把粗处理作为 HR/PPI 提取或 Frailty3 分类输入候选。
- 验收：形成可追溯的 motion-processing registry，所有新实验都能指出相对于历史方法的新增点和避免重复试验的依据。

### M1 定义端到端模块架构、数据契约和移动端约束

- [ ] 确定完整模块顺序：
  `input/manifest → validation/anomaly detection → preprocessing → IMU motion detection → data classification → SQI 或 coarse denoising → feature extraction → classifier → aggregation/calibration → output`。
- [ ] 将 SQI 和 coarse denoising 定义为并行可替换策略；部署配置必须明确二选一，不得隐式同时启用。
- [ ] 为各模块定义统一 Python API、输入输出 shape、dtype、channel order、sampling rate、单位、时间戳和缺失值语义。
- [ ] 区分 training/evaluation pipeline 与 mobile inference runtime，禁止训练逻辑、fold 统计或 notebook 隐性状态进入部署端。
- [ ] 定义统一配置格式，记录 preprocessing version、固定参数、feature schema、classifier type、threshold、label map 和模型版本。
- [ ] 明确移动端目标平台、Python/runtime 方式、CPU/内存/延迟预算及允许依赖。
- [ ] 部署核心不得依赖 notebook；PyTorch 模型应转换为 ONNX 或其他 CPU-only runtime。
- [ ] 保留 NumPy/SciPy 或等价轻量实现的 deterministic fallback。
- [ ] 设计可选分类器 registry，至少支持：
  - flat/hierarchical InceptionTime。
  - 1D-CNN/Small InceptionTime。
  - ShapeFormer/ShapeFormer-PISD。
  - ROCKET/MiniROCKET + ridge classifier。
  - tabular physiological-feature classifier。
- 验收：单个配置可以替换 quality policy、feature extractor 和 classifier，而不改变其他模块接口。

### M2 固化数据 Manifest、阶段映射和评估协议

- [ ] 建立统一 dataset manifest，包含 subject、class、file、role、channel、原始采样率、单位、数据版本、纳入/排除原因和 reference 可用性。
- [ ] 确认文件 role `B/R/S/W` 与 Base/Motion/Relax 的真实实验阶段映射；未确认前不得按文件名推断。
- [ ] 为外部同步 ECG/PPG/IMU 数据建立独立 manifest，记录 placement、wave length、activity、sampling pipeline 和 ECG reference 来源。
- [ ] 固化 Frailty3 subject-level fold registry、seeds、repeats 和数据版本。
- [ ] 固化当前 Frailty3 主协议：原始 400 Hz、subject-level 5-fold `StratifiedGroupKFold`、fixed epoch、no early stopping、相同 folds/seeds/repeats。
- [ ] 建立 protocol registry，分开标记 strict holdout、early-stopping CV、fixed-epoch OOF CV 和存在 data leakage 的历史结果。
- [ ] 没有独立 test 时统一使用 `oof_validation_*`，不得继续把 OOF validation 命名为 `test_*`。
- [ ] 原始目录 `PPG_Testing_05_01_2026/` 和 `physionet.org/` 保持只读。
- 验收：任何结果都能追溯到数据版本、subject split、fold、seed、protocol 和 preprocessing version。

### M3 统一并固定 PPG/IMU 预处理、数据清洗、归一化和特征算法

- [ ] 审计所有活动模块中不一致的 preprocessing：
  - PPG `0.2–8 Hz`、`0.5–5 Hz`、`0.5–8 Hz` 等滤波范围。
  - per-record z-score、per-window median/IQR 和 fold-level scaling。
  - 外部数据 resampling 与 Frailty3 原始 400 Hz 流程。
  - IMU low-pass gravity removal 与 EKF/bias-corrected gravity removal。
- [ ] 检查 NaN、Inf、flatline、saturation、异常振幅、异常 PPI、过短文件、channel missing、时间戳错误和设备量纲。
- [ ] 明确 RED、IR、accelerometer、gyroscope 的单位转换、channel order 和 polarity。
- [ ] 比较 PPG interpolation、detrending/high-pass、band-pass/notch，以及是否保留 DC、pulse amplitude 和 IR/RED ratio。
- [ ] 比较 IMU bias correction、low-pass、LPF gravity removal、EKF/姿态辅助 gravity removal、dynamic acceleration、gyro magnitude 和 jerk。
- [ ] 比较当前 per-window median/IQR、fold-level per-channel scaling、`StandardScaler`、`RobustScaler` 和 amplitude-preserving hybrid scaling。
- [ ] 所有 scaler、imputer、normalization statistics 和阈值只能在 training fold 拟合。
- [ ] 统一 peak/PPI/HR/HRV 算法，完成 `aboypp_detect_peaks` 与 `ppg.py/funcs.py` 的 parity test。
- [ ] 固定并版本化 static PPG、motion PPG、IMU、raw 8-channel classifier 和 mobile runtime preprocessing profiles。
- [ ] 建立单元测试和固定 reference fixtures；相同输入在所有调用入口应得到一致输出。
- 验收：所有活动模块调用同一公共实现或同一参数注册表，固定参数和适用场景均可追溯。

### M4 实现并跑通所有项目候选模块

本阶段先完成代码、接口、单元测试和 smoke test，不提前运行最终 benchmark 或撰写性能结论。

#### M4.1 异常检测、IMU Motion Monitoring、数据分类与 SQI

- [ ] 实现 channel missing、NaN/Inf、flatline、saturation/clipping、时间戳异常、振幅/单位异常和过短片段检测。
- [ ] 整合并跑通规则/阈值 detector、denoiser encoder motion head 和 lightweight CNN detector。
- [ ] 实现统一数据状态：invalid、static/high-quality、motion but usable、low-quality/unrecoverable。
- [ ] 实现组合 SQI，至少包含偏度、峰度、自相关周期性强度、模板相关系数、谱熵、心率带能量比例、搏动间隔生理合理性、RED/IR 一致性和 motion penalty。
- [ ] SQI peak logic 与统一 peak detector 对齐，不继续依赖未经验证的独立 `find_peaks` 参数。
- [ ] 实现 high-quality-only 策略，允许主动丢弃不可救片段并返回“无可靠结果”。
- [ ] 输出每个 class、subject、stage 和 activity 的保留窗口数、拒绝率和 coverage 所需字段。
- 验收：全部策略可通过同一 API 调用，并完成代表性静态、运动、低质量和异常输入 smoke test。

#### M4.2 Motion PPG 粗降噪与 HR/PPI 候选方法

- [ ] 跑通 raw/no-denoising、high-quality-only 和基础 band-pass baseline。
- [ ] 复核并模块化现有 IMU multi-reference NLMS ANC、CEEMD-lite+NLMS、历史 STFT mask、AE/UNet 和 hybrid A/B。
- [ ] 实现或整理非平稳分解候选：CWT、DWT、wavelet packet、wavelet threshold denoising、EMD、EEMD/CEEMD、VMD 和 SSA。
- [ ] 区分真正的 wavelet implementation 与部分旧函数中的 Savitzky–Golay 近似。
- [ ] 重点实现：`STFT → IMU spectrum-guided suppression/masking → candidate spectral peaks → temporal HR tracking`。
- [ ] 实现 IMU 谱减、soft mask、harmonic-aware suppression，以及 Kalman、particle filter、Viterbi 路径追踪接口。
- [ ] 精读并整理 TROIKA、JOSS 等经典方法的原理、假设和可复用部分。
- [ ] 基于 RED/IR 双通道实现 PCA、ICA、NMF；单通道数据不得运行该分支。
- [ ] 所有方法输出 HR trajectory、PPI/IBI、置信度、coverage 和失败状态；不得只输出看似平滑的波形。
- [ ] 实现部署策略开关：`sqi_gate` 与 `coarse_denoise` 二选一。
- 验收：每种方法均可在统一输入 schema 下运行，输出格式一致，并通过最小 smoke dataset。

#### M4.3 Dynamic Heartbeat / Peak / IBI / HRV 模块

- [ ] 完成正式配置所需代码路径，不只支持 smoke mode。
- [ ] 使用同步 ECG 开放数据监督 peak timing；ECG 不作为部署输入。
- [ ] 跑通 ECG detector preflight、PPG–ECG physiological delay calibration、LODO、extra-holdout 和 per-domain scorecard 生成逻辑。
- [ ] 跑通 denoiser encoder motion head、lightweight CNN、规则 detector 及组合策略接口。
- [ ] 定义 HR、PPI/IBI、SDNN、RMSSD、其他有足够时长支持的 HRV、peak confidence、SQI 和 motion/activity feature schema。
- [ ] 定义异常 PPI、缺失 beat、短窗口和低 confidence 的处理策略。
- [ ] 外部 heartbeat extractor 不得读取 Frailty validation/test 标签或统计量。
- [ ] 将稳定输出接入公共 Python feature API。
- 验收：`ppg_peak_hr_gating_train.py` 的训练、导出、scorecard 和复用接口均能完成端到端 smoke run。

#### M4.4 独立 ROCKET 卷积岭回归 Frailty3 路线

- [ ] 建议新建 `frailty_3class_rocket.py`。
- [ ] 实现：`异常检测 → IMU motion monitoring → 数据状态分类 → SQI 或粗降噪 → 特征提取 → ROCKET/MiniROCKET transform → ridge classifier → subject aggregation`。
- [ ] 初版优先实现 ROCKET 与 MiniROCKET；MultiROCKET 仅在资源允许且实现透明时加入。
- [ ] 支持 PPG-only RED/IR、SQI-selected PPG 和 coarse-denoised PPG 输入。
- [ ] IMU 初版只用于 motion/SQI/gating，不默认作为 ROCKET 分类输入，以区别于 8-channel deep-learning 路线。
- [ ] 支持 ROCKET transform features 单独分类，以及 ROCKET features + HR/PPI/HRV tabular late fusion。
- [ ] ridge scaler、kernel parameters、feature normalization 和 classifier 只能在 training fold 拟合。
- [ ] 输出 window/file/subject metrics、三类 report、confusion matrix、coverage 和 route-specific config 所需字段。
- [ ] 保存可移动端复用的 kernels、normalization、ridge coefficients、label map 和 feature schema。
- 验收：形成可独立训练、推理、保存和加载的 Frailty3 三分类器，并完成 smoke run。

#### M4.5 8-Channel Raw Deep-Learning 与 Hierarchical 路线

- [ ] 保持 flat InceptionTime、1D-CNN 和 Small InceptionTime 作为 8-channel raw baseline。
- [ ] 新建 `frailty_3class_hierarchical.py`。
- [ ] 上层分类器：`Young` vs `Old`。
- [ ] 下层分类器：只在 outer-train 的 old subjects 中分类 `Pre-Frail` vs `Robust/Non-Frail`。
- [ ] 初版使用两个独立 InceptionTime，不共享参数。
- [ ] 概率组合为 `P(Young)`、`P(Old) × P(Pre-Frail | Old)`、`P(Old) × P(Robust | Old)`。
- [ ] 下层训练、预处理、scaler 和 feature statistics 只能使用 outer-train old subjects。
- [ ] 实现 top-level、old-only oracle bottom-level、end-to-end subject-level 和错误传播报告字段。
- 验收：flat 与 hierarchical 路线均能使用统一 manifest/config 完成 smoke run。

#### M4.6 Base/Motion/Relax 分阶段生理特征路线

- [ ] 使用 M2 确认后的真实阶段映射。
- [ ] 接入 M4.1 motion state/SQI 和 M4.3 heartbeat extractor。
- [ ] 分阶段提取 HR、PPI/HRV、SQI、IMU activity、pulse morphology、阶段间变化、recovery slope、recovery time 和 relative-to-baseline。
- [ ] 定义 feature availability time，避免使用预测时刻之后的信息。
- [ ] 跑通 regularized logistic regression、SVM、ExtraTrees 和小型 gradient boosting。
- [ ] feature imputer/scaler 必须 fold-only。
- [ ] 输出阶段级 feature schema 和解释字段。
- 验收：从多阶段原始文件到 tabular Frailty3 prediction 的整条路线能够完成 smoke run。

#### M4.7 异步、多时间尺度特征融合

- [ ] 标记每个特征属于 window、file、stage 还是 subject。
- [ ] 修正或隔离 file-level features 复制到每个 window 造成的隐式加权。
- [ ] 阻止使用当前窗口之后的整文件信息。
- [ ] 实现 window early fusion、raw embedding 后 file fusion、subject-level late fusion 和严格 OOF stacking 接口。
- [ ] stacking meta-model 只能使用 OOF predictions；imputer/scaler 必须 fold-only。
- [ ] 不得在单独路线完成严格评估前默认融合 ROCKET、raw deep 和 stage physiology。
- 验收：各融合方式可配置切换，并通过无跨 fold、跨 subject 或未来信息泄漏检查。

### M5 整合全部历史 Sweep/Grid Search，建立严格可比基线 Top 5

本阶段只整合历史结果并建立基线；所有新实现路线的统一 benchmark 位于 M6。

- [ ] 新建跨实验分析入口：`analyze_all_frailty_experiments.py`。
- [ ] 建立 parameter ontology 和 protocol registry，统一旧字段名、缺失字段和 reference 来源。
- [ ] 纳入至少：
  - `20260527_1320_cnn_inceptionTime`。
  - `20260528_1045_shapeformer_0extra`。
  - `overfitting_20260608_0752`。
  - `20260608_1206_overfitting_sweep_stage1_rank2`。
  - `20260625_2320_overfitting_sweep_stage1_rank2`。
  - `20260630_0630_overfitting_sweep_generalization_rank2`。
- [ ] 将 ShapeFormer/ShapeFormer-PISD 纳入历史 grid registry，恢复并记录其完整 config、discovery method、shapelet 参数、extra input、fold/seed、protocol、运行完整性和结果来源。
- [ ] 检查 `20260528_1045_shapeformer_0extra` 是否包含 ShapeFormer 与 ShapeFormer-PISD、是否完整、是否与当前协议严格可比；不兼容时保留但明确标记 protocol/comparability。
- [ ] 2026-05-27 data-leakage 结果只保留历史参考；只允许当前协议重跑的 reference 做绝对分数比较。
- [ ] 在相同 folds/seeds 下计算 paired metric differences、bootstrap CI、class-level effects 和 stability。
- [ ] 不把不平衡 grid 的回归系数解释为因果效应。
- [ ] 输出参数覆盖图、主效应/交互图、class recall/F1、confusion matrix、repeat/fold stability 和 reference drift。
- [ ] 在严格相同协议中按 config-level repeat mean 排名，不按最佳单次 repeat 排名。
- [ ] 不使用 runtime/cost/Pareto 指标影响性能排名，但可单独报告部署成本。
- [ ] 扩展 `analyze_sweep.py` config identity，使 SQI、aggregation、manual features、loss、class weight、sampler、window quota、train overlap，以及 ShapeFormer/ShapeFormer-PISD 参数成为显式 columns。
- [ ] 默认模型列表加入 `small_inceptiontime`、`shapeformer` 和 `shapeformer_pisd`。
- [ ] 不完整 config 不与完整 config 公平排名。
- 验收：输出历史严格可比 Top 5，包含完整参数、mean/std/CI、worst-class 指标、三类 confusion matrix 和可比性标签。

### M6 在所有代码跑通后建立统一 Benchmark、正式评估和报告

#### M6.1 Frailty3 Benchmark

- [ ] M0–M5 完成且所有候选路线 smoke test 通过后，新建 `frailty_3class_benchmark.py`。
- [ ] 为 flat InceptionTime、hierarchical InceptionTime、1D-CNN、Small InceptionTime、ShapeFormer、ShapeFormer-PISD、ROCKET/MiniROCKET + ridge 和 tabular baseline 提供统一 wrapper。
- [ ] 统一输出 window/file/stage/subject-level metrics、三类 report、confusion matrix、learning curves、fold subject IDs、完整 config、seed、数据版本和 completeness check。
- [ ] benchmark 必须显式传入训练脚本全部关键参数，避免旧 CLI/defaults 影响实验。
- [ ] 不同 protocol 不得进入同一 leaderboard。
- [ ] 与 M5 历史基线对照时，必须保留 protocol/comparability 标签。

#### M6.2 Motion-PPG/SQI/Heartbeat Benchmark

- [ ] M4.1–M4.3 全部跑通后，新建 `motion_ppg_benchmark.py`。
- [ ] 设置 raw/no-denoising 和 high-quality-only baseline。
- [ ] 统一 subject-level、dataset-level 和 activity-level split。
- [ ] 在有 ECG reference 的数据上报告 peak precision/recall/F1、peak timing error、HR MAE/RMSE/bias、PPI/IBI MAE、漏检率、额外搏动率、continuity，以及 SDNN/RMSSD 等 HRV 指标误差。
- [ ] 在无 ECG reference 的 Frailty 数据上报告 coverage、周期一致性、双波长一致性和运动分层稳定性，但不得把一致性当作 ground truth accuracy。
- [ ] 同时记录 CPU latency、peak memory、模型大小、失败样本和“看似合理但错误”的输出比例。
- [ ] 波形 SNR/相关系数只作辅助指标，不得代替 HR/PPI 指标。
- [ ] 比较 `sqi_gate` 与 `coarse_denoise`；如果粗降噪不能显著优于 high-quality-only，应选择 SQI 路线。

#### M6.3 报告要求

- [ ] 所有路线使用相同 manifest、fold registry、seeds、repeats、preprocessing version 和明确训练预算。
- [ ] 分别报告 raw 8-channel、hierarchical、ROCKET 和 Base/Motion/Relax 路线，不得先融合后只报告总结果。
- [ ] 同时报告 mean/std/bootstrap CI、worst-class 指标、coverage、稳定性和失败样本。
- 验收：同一 benchmark 可重复运行所有候选路线；任何报告都可追溯到代码版本、subject split、seed、protocol、preprocessing 和完整 config。

### M7 完成统一消融试验

- [ ] 使用相同 folds、seeds、epoch、window、训练预算和基础 config；每次只改变一个因素。
- [ ] 至少覆盖 RED/IR、PPG-only、IMU-only、gravity removal、preprocessing profile、scaler、异常检测、SQI、high-quality-only、coarse denoising、PPI/HRV、morphology、ROCKET、ShapeFormer/ShapeFormer-PISD、hierarchy、sampler、aggregation、Base/Motion/Relax、recovery features 和 fusion strategy。
- [ ] 报告 subject-level paired BA/macro-F1 delta、bootstrap CI、class recall/F1、confusion matrix、coverage 和失败样本。
- [ ] 区分平均性能提高、稳定性提高、仅特定 class 提高，以及因丢弃低质量窗口而改变 coverage。
- [ ] PPI/HRV/manual features 是否保留必须由本消融决定，不依据单次 run。
- [ ] 历史 grid 的关联性结果不得替代本阶段的单因素因果消融。
- 验收：每个最终保留模块都有独立、配对、可重复的贡献证据。

### M8 更新最终排名并选择最终配置

- [ ] 将新 preprocessing、SQI/coarse-denoising、ROCKET、ShapeFormer/ShapeFormer-PISD、hierarchy 和 stage-feature 实验纳入统一分析入口。
- [ ] 只在严格相同 protocol 中按 config-level repeat mean 排名。
- [ ] 输出最终 Top 5 完整参数组，包括 mean/std/bootstrap CI、worst-class recall/F1、三类 confusion matrix、stability、coverage 和完整模块版本。
- [ ] 不完整 config 不与完整 config 公平排名。
- [ ] 锁定 final selection rule。
- [ ] 如需发表级独立性能，预留 untouched test 或 external validation，并在查看结果前锁定配置。
- [ ] 当前 5-fold OOF CV 不得表述为额外独立 test。
- 验收：最终选择不依赖最佳单次 repeat，并能解释为何选定特定 quality policy、feature route 和 classifier。

### M9 最终训练、Python-Based 移动端 Pipeline 整合和导出

- [ ] 用锁定 config 重新训练最终候选，不从 CV fold 中挑选单个最好模型。
- [ ] 保存模型权重、preprocessing profile、scaler/imputer、SQI 或 coarse-denoising policy、motion detector、peak/PPI/HRV extractor、feature schema、classifier artifact、label map、manifest、fold registry、window 参数和评估摘要。
- [ ] 实现一条龙 Python API 和 CLI：
  `原始 PPG/IMU → 质量检查 → 预处理 → motion/data state → SQI 或粗降噪 → 特征 → 可选分类器 → Frailty3 概率与置信度`。
- [ ] 分类器通过统一 registry 切换，至少支持最终入选路线及可复现 baseline。
- [ ] 移动 runtime 不依赖 notebook，不要求 PyTorch。
- [ ] 对深度模型完成 ONNX/CPU-only export、runtime smoke test 和 Python/ONNX parity。
- [ ] 对 ROCKET 提供轻量 transform + ridge inference。
- [ ] `ppg_analyse4_calib.ipynb` 只作为校准、可视化和人工检查 adapter，不保存核心业务逻辑。
- [ ] 完成 schema validation、deterministic output、缺失 channel、低 SQI、motion、极短输入和多阶段输入端到端测试。
- [ ] 报告目标设备 latency、memory、model size 和失败行为。
- [ ] 信号不可用时返回 invalid/insufficient-quality，不强制输出 Frailty3 预测。
- 验收：在干净环境和目标移动端方案中，可仅通过保存的 bundle 和配置完成 CPU-only 推理。

### M10 归档旧路线并维护可追溯文档

- [ ] 在 M0 审计和 M6 benchmark 完成后，再归档旧 denoiser。
- [ ] 明确 `pttppg_denoiser_hybrid_*`、`pttppg_stage2_denoiser.py`、`pttppg_denoiser_v8_masknet.py` 和 `pttppg_pipeline_v7*` 的 deprecated/reference 状态。
- [ ] 不删除仍有参考价值的代码、结果或 ONNX 部署经验。
- [ ] 将最终模块状态、算法选择、失败路线原因、固定参数和部署接口同步到对应 `_agent` 文档。
- [ ] 将 preprocessing、SQI、motion processing、HR/PPI、ROCKET、ShapeFormer 历史比较和最终 pipeline 整理为 thesis-ready 算法说明。
- 验收：新接手者不会把旧 full-waveform denoiser 误认为活动主线，并能复现最终 pipeline。

---

## 可选或低优先级任务

### O1 W&B

- [ ] 当前继续使用本地 CSV/JSON/PNG/Markdown。
- [ ] 只有跨机器协作或实验规模继续扩大时再评估 W&B。

### O2 额外移动端优化

- [ ] 在功能和数值 parity 完成后，再考虑量化、剪枝、硬件加速或平台原生封装。
- [ ] 这些优化不得先于最终模型和 preprocessing profile 锁定。

---

## 已完成或被后续实验替代

- [x] 当前活动 Frailty3 流程保持原始 400 Hz，不做 resampling。
- [x] `PPG_Testing_05_01_2026/` 和 `physionet.org/` 已设为只读。
- [x] 已实现 PPG 基础预处理、IMU gravity removal、local Aboy++ morphology、file-level PPI/HRV/manual fusion、现有 SQI gating、weighted CE/balanced softmax/focal loss、class weights 和 subject-aware samplers。
- [x] 2026-06-08 stage1 Top4 后续任务已被 2026-06-25 和 2026-06-30 更完整 sweep 覆盖，不再直接作为当前 stage2 入口。
- [x] 2026-06-25 sweep：129 configs、645 runs，完整。
- [x] 2026-06-30 generalization sweep：232 configs、1160 runs，完整；新 config 未达到 BA 0.60，当前结果仍未达到 0.73。
- [x] 已完成 2026-06-16 和 2026-07-06 独立 sweep reports；后续由 M5/M8 统一整合。
- [x] 当前 Frailty3 采样率和时间基线已核准；外部数据时间轴、设备量纲和重采样仍由 M2/M3 审计。
