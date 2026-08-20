# MODULES

状态：confirmed
来源：用户项目总纲、用户当前进度说明、`_agent/arc/PROJECT_HANDOFF.md`、代码与结果目录复核。
最后手动更新时间：2026-08-19
用途：记录本项目各核心模块的用途、输入输出、脚本/函数对应关系、历史版本、当前状态、已实现内容、待实现内容、算法思路和改进方向。

## 模块总览

本项目目标是开发并评估一个 Python-based PPG signal processing pipeline，用于从 PPG 信号及 IMU 辅助信息中提取 frailty 相关特征，并完成 `frail / pre-frail / non-frail` 或当前数据定义下的 `Pre-Frail / Robust-Non-Frail / Young` 三分类。核心要求是透明、可复现，并且核心信号处理组件不依赖 opaque third-party toolkits。

当前主线分为：

1. PPG 静态/基础预处理与特征提取。
2. Aboy++ PPG peak detection、PPI、HRV。
3. IMU-led motion/static detector。
4. 动态 PPG heartbeat / IBI / HRV 提取。
5. frailty3 三分类模型。
6. overfitting/generalization sweep、跨实验 analysis、benchmark 与消融。
7. 已失败但保留历史价值的 dynamic PPG denoiser 路线。
8. ShapeFormer、ASA 等旁支模型试验。

---

### 1. PPG 预处理与基础特征模块

- 当前最新版本：frailty3 活动流程已核准为 400 Hz、无 resampling；其他入口仍需分别审计。
- 相关文件：
  - `funcs.py`
  - `ppg.py`
  - `frailty_3class_classifier.py`
- 关键函数：
  - `highpass_filter`
  - `bandpass_filter`
  - `notch_filter`
  - `wavelet_denoise`
  - `preprocess_ppg_min`
  - `time_features`
  - `spectral_features`
  - `extract_file_features`
- 模块用途：
  - 对原始 PPG 信号做基础滤波、去趋势、频域/时域基础特征提取。
  - 为静态 PPG 波形分析、peak detection、HRV 计算和 frailty classifier 提供基础输入。
- 输入：
  - 原始 PPG 通道，主要为 `RED`、`IR`。
  - 在 frailty3 主线中也会与 IMU 通道共同组成 window。
- 输出：
  - 预处理后的 PPG 序列。
  - 基础时域/频域特征。
  - 可供 peak detection 或 classifier 使用的 cleaned/minimal-preprocessed signal。
- 已实现内容：
  - 高通、带通、notch、wavelet denoise 等基础处理函数。
  - frailty3 raw window 为 8 通道、400 Hz；5 秒窗口形状为 `[N,8,2000]`。
  - frailty3 PPG 使用缺失值插值、线性 detrend、0.2--8 Hz Butterworth；
    accelerometer 使用 20 Hz low-pass 后减去 0.3 Hz low-pass gravity；
    gyroscope 使用 40 Hz low-pass。
  - 每个 window 使用 median/IQR robust scaling，并 clip 到 `[-8,8]`。
  - frailty3 脚本中已有文件级特征提取与窗口构建逻辑。
- 待实现内容：
  - 对 frailty3 之外的每批原始数据继续明确采样率、时间戳和设备量纲。
  - 整理哪些预处理步骤属于 thesis core pipeline，哪些只是 notebook/诊断辅助。
  - 对 `funcs.py` 与 `ppg.py` 中重复函数做归档或来源说明，避免后续 chat 误用旧入口。
  - 比较 per-window scaling 与 fold-level/channel-level scaling，检查当前方法是否删除
    绝对 pulse amplitude 和 IR/RED ratio 信息。
- 历史版本与更新目的：
  - `ppg.py` 是较早的交互式/可视化入口，用户已明确指出其过时。
  - `ppg_analyse4_calib.ipynb` 是当前主分析 notebook，应逐步替代旧 `ppg.py` 的入口作用。
- 算法思路：
  - 先用透明基础滤波和最小必要预处理保留 PPG waveform morphology。
  - 避免 opaque toolkit 直接输出最终特征，以满足 thesis 对透明性和可复现性的要求。
- 可能改进方向：
  - 建立统一 preprocessing API。
  - 为静态段和动态段区分不同处理策略。
  - 增加每一步处理前后可视化与质量指标。
- 验证状态：
  - 当前活动 frailty3 脚本不执行 resampling，`RunConfig.fs=400`。
  - archive 中仍可存在旧 resample 代码，不能据此推断活动流程。
- 后续建议：
  - 将最终采用的预处理流程单独整理成 thesis-ready 文档和代码注释。

---

### 2. Aboy++ Peak Detection、PPI、HRV 模块

- 当前最新版本：静态/file-level 实验接口已实现，但 `ppg.py` parity 尚未验证。
- 相关文件：
  - `funcs.py`
  - `ppg.py`
  - `frailty_3class_classifier.py`
  - `frailty_3class_overfitting_sweep.py`
- 关键函数：
  - `aboypp_peak_hr`
  - `aboypp_peak_hr_windowed`
  - `aboypp_detect_peaks`
  - `calculate_hrv`
  - `ppi_from_peaks`
  - `hrv_compare_from_aboy_ir`
  - `detect_ppg_peaks`
  - `detect_common_peaks`
  - `reject_artifacts`
  - `caculate_clean_peaks`
  - `ppi_hrv_features`
- 模块用途：
  - 从 PPG 中检测 pulse peaks。
  - 计算 PPI、beat-to-beat intervals、HRV/manual features。
  - 为静态 PPG 分析和 frailty3 classifier 提供可解释手工特征。
- 输入：
  - 400 Hz 预处理 PPG，主要为 `IR` 或 `RED/IR`。
  - file-level signal、采样率和可选 IMU motion magnitude。
- 输出：
  - peak indices。
  - PPI sequence。
  - HR/HRV features，例如 SDNN、RMSSD 等候选指标。
  - 在 frailty3 pipeline 中作为 fold 内标准化后的 tabular extra features。
- 已实现内容：
  - `ppg.py`/`funcs.py` 中的 Aboy++ peak/HR 和 windowed peak/HR 计算。
  - frailty3 内部 `aboypp_detect_peaks` 移植了 Aboy++ 的核心流程，用于
    file-level PPI/HRV 和 morphology。
  - file-level morphology 包括 pulse amplitude、rise/decay time、pulse width、
    systolic slope、pulse area、PPI stability、IR/Red AC/DC ratio、
    correlation/phase 和 motion-normalized PPG features。
  - frailty3 中的 manual features 经 training-fold scaler 后通过 MLP fusion，
    不是把 PPI/HRV 直接当作 raw 时序通道。
  - 当前 feature cache：
    `datasets/frailty3_features_v2_aboy_morph_gravity_B_R1_R2_R3_R4_fs400_w10_h5.csv`，
    包含 145 files、29 subjects。
- 待实现内容：
  - 对 `aboypp_detect_peaks` 与 `ppg.py::aboypp_peak_hr` 做同输入 parity test；
    当前不能称为逐行原样调用。
  - 当前 SQI 仍使用简化 `find_peaks` 逻辑，应决定是否统一到 Aboy++，
    并通过真实 peak/ECG reference 验证。
  - 将 window/file/stage/subject-level 特征范围写入 schema，避免整文件特征复制到
    window 后产生隐式加权或未来信息问题。
  - 明确哪些 HRV 指标进入最终 thesis evaluation，并分开评估静态、动态和恢复阶段。
  - 对 peak detection 在不同 frailty 状态、不同 role、不同运动状态下的失败模式做记录。
- 历史版本与更新目的：
  - 早期 peak detection 主要服务于 PPG 分析与可视化。
  - 后续加入 frailty3 classifier，PPI/HRV 作为额外手工特征参与 MLP fusion。
- 算法思路：
  - 使用自定义 peak detection 和 artifact rejection，而不是依赖 opaque third-party toolkit。
  - PPI/HRV 不作为 raw time-series channel 拼入 `[N,8,T]`，而是作为 fold 内标准化表格特征，经 MLP 与深度特征融合。
- 可能改进方向：
  - 加入 ECG reference 或高质量静态段人工标注对 peak detection 做 benchmark。
  - 增加 HRV 指标层评价，而不仅评价 peak timing。
  - 比较静态 HRV、动态 HRV、relax-stage heart-rate recovery speed 对 frailty 分类的贡献。
- 验证状态：
  - 静态/file-level 特征提取可运行。
  - 尚无证据证明 frailty3 本地 Aboy++ 与 `ppg.py` 输出完全一致。
  - 动态段 peak/HRV 尚未稳定成功。
- 后续建议：
  - 将 Aboy++ 算法流程、阈值、artifact rejection 规则整理为 thesis algorithm design 部分。

---

### 3. IMU-led Motion / Static Detection 模块

- 当前最新版本：
  - 部署端需要微调版：半可用。
  - 高度泛化机器学习模型复用版：待完成，尚未成功。
- 相关文件：
  - `funcs.py`
  - `ppg.py`
  - `pttppg_detector_v8_scores_audit_fix9.py`
  - `ppg_peak_hr_gating_train.py`
- 关键函数/类：
  - `imu_preprocess_with_kf`
  - `classify_motion_from_df`
  - `v8_detector_predict`
  - `extract_window_features`
  - `build_window_table`
  - `DenoiserEncoderMotionDetector`
  - `LightCnnMotionDetector`
  - `run_motion_detector_benchmark`
- 模块用途：
  - 根据 IMU 信号或 PPG+IMU 表征判断静止/运动状态。
  - 为动态段处理、静态段 waveform analysis 和最终 frailty fusion 提供状态信息。
- 输入：
  - IMU 6 通道：`AX, AY, AZ, GX, GY, GZ`。
  - 可选 PPG 信号或 denoiser encoder 特征。
- 输出：
  - motion/static labels 或概率。
  - detector benchmark scorecard。
  - 可用于后续动态 heartbeat extraction 的 gating/state 信息。
- 已实现内容：
  - 基于 IMU 的预处理与 motion classification 函数。
  - 旧 detector v8 分数审计脚本。
  - `ppg_peak_hr_gating_train.py` 中已包含 motion detector A/B benchmark：
    - A：denoiser encoder motion head。
    - B：lightweight CNN detector。
- 待实现内容：
  - 决定最终 motion detector 使用方案。
  - 基于 extra-holdout、per-dataset、per-subject、per-activity 结果比较 A/B。
  - 将最终 detector 部署到 `ppg_analyse4_calib.ipynb`。
- 历史版本与更新目的：
  - 旧 denoiser/gating 路线虽无法完成可靠动态去噪，但暴露出 gating/static-motion recognition 有价值。
  - 后续从 denoising 目标转向独立 motion detector 和 dynamic heartbeat extractor。
- 算法思路：
  - 不把 IMU 直接当作 artifact teacher。
  - 将 IMU 作为运动状态、信号质量、动态段分层判断依据。
  - 通过分层评估确认 detector 是否跨数据集泛化。
- 可能改进方向：
  - 使用 lightweight model 做部署端 detector。
  - 将 detector 输出作为 frailty classifier 的额外状态特征。
  - 将 detector 与动态 peak/IBI 模型联合或级联。
- 验证状态：
  - 半可用。
  - 需要正式 benchmark 和 extra-holdout 结果后定版。
- 后续建议：
  - 当前不要把旧 detector 或 denoiser gating 自动视为最终方案，应先比较并记录 A/B 结果。

---

### 4. Dynamic PPG Denoising 模块，已失败路线

- 当前最新版本：deprecated / experimental，不作为当前主线。
- 相关文件：
  - `pttppg_denoiser_hybrid_core.py`
  - `pttppg_denoiser_hybrid_train.py`
  - `pttppg_denoiser_hybrid_preview.py`
  - `pttppg_denoiser_hybrid_ab_compare.py`
  - `pttppg_denoiser_hybrid_export_onnx.py`
  - `pttppg_denoiser_onnx_runtime.py`
  - `pttppg_denoiser_v8_masknet.py`
  - `pttppg_stage2_denoiser.py`
  - `pttppg_pipeline_v7_4_noleak_viz_ae.py`
- 关键函数/类：
  - `HybridArtifactRefiner`
  - `CleanPriorAutoencoder`
  - `MaskNet`
  - `train_hybrid_model`
  - `denoise_record`
  - `denoise_record_onnx`
  - `eval_denoiser`
  - `train_one_activity`
  - `export_bundle_to_onnx`
  - `load_bundle`
- 模块用途：
  - 早期尝试从动态 PPG 中恢复 clean waveform。
  - 目前主要作为失败路线、诊断工具和 ONNX 部署经验保留。
- 输入：
  - Denoiser A：`raw PPG + IMU`。
  - Denoiser B：`raw PPG + IMU + linear baseline`。
  - 曾尝试 sit prior、peak prior、clean prior、artifact relation learning。
- 输出：
  - 去噪后的 PPG waveform。
  - denoiser preview plots。
  - old denoiser ONNX bundle/runtime。
- 已实现内容：
  - hybrid denoiser 训练、预览、A/B 对比、ONNX 导出和 ONNX runtime。
  - 输出目录包括：
    - `results_hybrid_denoiser_raw_imu/`
    - `results_hybrid_denoiser_raw_imu_baseline/`
    - `denoiser_preview_output/`
- 待实现内容：
  - 如果继续保留，应归档并明确 deprecated 状态。
  - 不建议继续作为动态 PPG 去噪主线。
  - 可作为 frailty classifier 输入做探索性实验，但必须标注高风险。
- 历史版本与更新目的：
  - 初始假设 IMU 与 motion artifact 存在线性/非线性关系，可学习 artifact removal。
  - 用户指出 IMU 与 artifact 的关系不能被预设为 teacher，关系本身才是要学习的目标。
  - 实验显示动态段泛化失败。
- 算法思路：
  - 旧路线试图恢复完整 clean waveform。
  - 当前结论是：在无真实 motion clean PPG、无可靠部署端 motion peaks、ECG-PPG 存在生理 delay 的条件下，该目标不稳健。
- 失败原因：
  - Denoiser A 输出接近复制 raw PPG，动态段无稳定周期恢复。
  - Denoiser B 出现双倍伪峰、峰谷错位、梯形/非生理形态。
  - static/sit 段看似很好，可能只是 identity mapping 或 gating 抑制修正，不能证明动态去噪有效。
  - motion artifact、motion clean PPG 和 IMU-artifact 数学关系都缺少可靠监督。
- 保留下来的价值：
  - gating/static-motion behavior 可能有价值。
  - ONNX/CPU-only 部署经验可迁移到新 detector 或 peak/IBI 模型。
- 可能改进方向：
  - 不再追求 full clean waveform reconstruction。
  - 转向 direct peak/IBI extraction 或信号质量分层。
- 验证状态：
  - 动态去噪路线失败。
  - 无 clean 对照数据时难以评价准确率。
- 后续建议：
  - 在文档中明确 deprecated，避免后续 chat 误认为这是当前主线。

---

### 5. Dynamic Heartbeat / Peak / IBI / HRV Extraction 模块

- 当前最新版本：待完成，尚未成功定版。
- 相关文件：
  - `ppg_peak_hr_gating_train.py`
- 关键函数/类：
  - `WindowConfig`
  - `LossConfig`
  - `ModelConfig`
  - `AugmentConfig`
  - `DetectorConfig`
  - `MultiDatasetPeakWindowDataset`
  - `PeakIntervalGateNet`
  - `PeakIntervalGateOnnxWrapper`
  - `detect_ecg_rpeaks`
  - `detect_ppg_pulse_peaks`
  - `build_peak_target`
  - `build_rr_track`
  - `fit_model_with_validation`
  - `compute_event_metrics`
  - `analyze_ppg_ecg_delay`
  - `run_cross_validation`
  - `run_leave_one_dataset_out`
  - `export_deploy_bundle`
  - `write_scorecard_markdown`
- 模块用途：
  - 从动态 PPG 中直接提取可靠 heartbeats、IBI、HR/HRV。
  - 替代失败的 dynamic denoising 路线。
  - 为最终 frailty classifier 提供动态生理参数。
- 输入：
  - 动态 PPG 原信号。
  - 同步 ECG reference 用于监督。
  - 可选 IMU/motion state。
  - 多数据集来源：PTT、MIMIC、iAMwell、simultaneous、VitalDB 等。
- 输出：
  - peak sequence。
  - HR interval / IBI sequence。
  - gate logits。
  - per-dataset/per-subject/per-activity scorecard。
  - cross-validation、holdout、extra-holdout、LODO 结果。
  - 可部署 bundle / ONNX runtime 方向。
- 已实现内容：
  - ECG detector preflight。
  - beat-level peak supervision。
  - IBI Huber loss 与生理范围约束。
  - dataset/activity-balanced training。
  - domain-aware augmentation。
  - instance norm / adversarial domain generalization。
  - delay analysis。
  - LODO。
  - GroupDRO / worst-domain 思路。
  - motion detector A/B benchmark。
  - PPG 主指标改为 ±20 ms，并分层报告 ±10/20/30/40 ms。
- 待实现内容：
  - 跑正式全量训练，不只 smoke。
  - 系统读取 scorecard。
  - 决定最终部署模型是 peak/IBI 主模型、IMU motion detector，还是二者组合。
  - 输出 ONNX/CPU-only 部署模块。
  - 与 `ppg_analyse4_calib.ipynb` 集成。
  - 进一步做 ECG detector 与 PPG delay 校准。
  - 做 HRV 指标层评估。
- 历史版本与更新目的：
  - 从“恢复 clean waveform”转向“直接提取 reliable beat timing / IBI”。
  - 原因是 frailty pipeline 真正需要的是 HR/HRV 等中间参数，而不是视觉上 clean 的动态 PPG。
- 算法思路：
  - 使用 ECG reference 监督 peak timing。
  - 不假设 motion clean PPG 可恢复。
  - 重点评价 beat timing、IBI continuity、漏检/误检，而不是 waveform reconstruction。
- 可能改进方向：
  - 加强 PPG-ECG delay 建模。
  - 针对 AF/noisy/neonate 等困难子集做分层失败分析。
  - 将输出 HRV 特征接入 frailty3 pipeline。
- 验证状态：
  - 代码结构已实现较多。
  - 正式训练和稳定评估尚未完成。
- 后续建议：
  - 这是动态段当前最重要路线，应优先推进。

---

### 6. Frailty3 三分类主模型模块

- 当前最新版本：主线进行中；严格可比结果约 0.62 BA，目标 0.73，尚未达到。
- 相关文件：
  - `frailty_3class_classifier.py`
  - `frailty_3class_cnn_fusion.py`
  - `frailty_3class_overfitting_sweep.py`
  - `frailty_3class_holdout_eval.py`
  - `shapeformer_port.py`
- 关键函数/类：
  - `RunConfig`
  - `build_manifest`
  - `build_cnn_window_table`
  - `InceptionTimeClassifier`
  - `FeatureFusionClassifier`
  - `train_cnn_model`
  - `evaluate_cnn`
  - `train_shapeformer_model`
  - `PortedShapeFormer`
  - `ShapeBlock`
  - `discover_shapelets`
  - `discover_shapelets_pisd`
- 模块用途：
  - Frailty三分类深度学习分类器，输出路径results_frailty3
  - 使用 RED/IR 双通道 PPG 与 IMU 6 维信号区分
    `Pre-Frail`、`Robust/Non-Frail`、`Young`。
  - ###！！！比较 different raw deep models、manual-feature fusion、SQI、loss、sampler，强正则，windows size，数据增强 和
    subject-level aggregation等等 对泛化的影响。由于算力和时间限制，同分类参数组（比如强正则，models）的比较在同分类中进行，而不进行全参数组的网格比较
- 数据来源：
  - `PPG_Testing_05_01_2026/StudyData_frailtyScored/StudyData_V7_standard.csv`
  - `PPG_Testing_05_01_2026/StudyData`
  - `PPG_Testing_05_01_2026/TestDataYoungers`
  - 上述原始数据目录和 `physionet.org/` 已设为只读；`datasets/` 是生成/读取的
    cache 目录，不是 raw input source。
- 标签定义：
  - `FRAILTY-STATUS=2` -> `Pre-Frail`
  - `FRAILTY-STATUS=3` -> `Robust/Non-Frail`
  - `TestDataYoungers` -> `Young`
- 数据审计：
  - raw 总计 29 subjects、261 files：StudyData 21 subjects/189 files，
    Young 8 subjects/72 files。
  - 静态纳入 `B,R1,R2,R3,R4` 后为 145 files：
    Pre-Frail 9 subjects/45 files、Robust 12/60、Young 8/40。
  - 每个静态 role 均有 29 files。
  - 若纳入 `B,R1-R4,S1,S2,W1,W2`，三类文件数为 81/108/72。
  - label CSV 中有 10 个 ID 不在 StudyData；其中 6 个属于 Young folder，
    真正缺失的是 `BAE28,NRE29,PSR16,PSS22`。
  - Young subjects `AB_01`、`EE_02` 不在 label CSV，但按文件夹正确标为 Young。
- 文件纳入规则：
  - `STE072` 已纠正，可以纳入。
  - 默认静态实验只采用 role/suffix 为 `B,R1,R2,R3,R4` 的文件。
  - `train_all_roles` 是可选动态扩展，不应与 static-only baseline 混为同一 config。
- 输入：
  - raw 8 channels：`RED, IR, AX, AY, AZ, GX, GY, GZ`。
  - 全程保持 400 Hz；5 秒 window 的形状为 `[N,8,2000]`。
  - PPG：interpolation、linear detrend、0.2--8 Hz Butterworth。
  - accelerometer：20 Hz low-pass 后减去 0.3 Hz low-pass gravity；
    gyroscope：40 Hz low-pass。
  - 每 window median/IQR scaling，clip `[-8,8]`。
  - 可选 file-level tabular features：PPI、HRV、morphology；
    training fold 内缩放后由 MLP 与 raw embedding 融合。
- 输出：
  - window/file/subject-level metrics。
  - confusion matrix。
  - learning curves。
  - per-run CSV/JSON/report、fold predictions 和 config summary。
  - model artifacts。
- 已实现内容：
  - 数据读取与 label mapping。
  - role filter：`B,R1,R2,R3,R4`。
  - `STE072` 纳入。
  - local Aboy++ peak detection、PPI、HRV 和 file-level morphology。
  - SQI modes：`none/top70_quality/top50_quality`。
  - subject aggregation：`mean_prob/quality_weighted_mean`。
  - losses：weighted CE、balanced softmax、focal loss。
  - class weights：inverse subject count、effective number。
  - samplers：none、subject-balanced、class-subject-balanced；
    per-subject window quota 支持 all、百分比和绝对数，并按 seed+epoch 随机采样。
  - 1D-CNN、full InceptionTime、Small InceptionTime。
  - Small InceptionTime 当前为 depth 3、filters 16、bottleneck 16；
    full 版本为 depth 6、filters 32、bottleneck 32。
  - ShapeFormer core port。
  - ShapeFormer-PISD wrapper。
  - subject-level 5-fold `StratifiedGroupKFold`。
  - learning curve plot。
  - auto sweep、incremental CSV/report、总/子进度条。
- 待实现内容：
  - 建立统一 Frailty3 benchmark 和跨 sweep protocol registry。
  - 尝试输入dynamic coarse-denoised signal （探索项）
  - 选定最终config后重新训练并保存参数（scaler，label map，window参数，feature schema）部署模型
  - 实现 hierarchical InceptionTime：Young/Old 后再分 Pre-Frail/Robust。
  - 建立 Base/Motion/Relax 的relax-stage HR recovery speed生理特征路线和弱模型 baseline。
  - 系统审计 scaler、异步 file-level feature fusion 和 SQI coverage。
  - 在相同 folds/seeds 下完成消融并选出严格可比 Top 5。
  - 最终选定 config 后重新训练部署模型，保存 scaler、label map、manifest、
    fold registry、window 参数和 feature schema。
- 历史版本与更新目的：
  - 从单一 CNN 分类脚本扩展为统一训练和 sweep 框架。
  - `frailty_3class_cnn_fusion.py` 是早期 raw window + handcrafted feature fusion 旁支，新主脚本已吸收其 `extra_input=PPI/HRV` 思路。
  - ShapeFormer 被保留为接口和实验记录，但当前不作为主要优化方向。
- 算法思路：
  - 使用 subject-level split，避免同一 subject windows 同时出现在 train/validation。
  - 用 config-level mean/std/CI 判断模型，不按单次最好 repeat 选择。
  - 当前不把 runtime/cost/Pareto efficiency 作为 leaderboard 排名依据。
  - 当前主协议为 5-fold `StratifiedGroupKFold`、fixed epoch、no early stopping；
    每个 fold 的 validation 仅用于 OOF evaluation 和 learning curve。
  - 当前 CV 不含额外独立 test set；报告中的历史 `test_*` 字段可能实际是
    OOF validation，需要在 benchmark 中改名。
- 可能改进方向：
  - hierarchy、stage-level feature engineering、amplitude-preserving scaler。
  - file/subject late fusion 或严格 OOF stacking。
  - paired ablation 和 subject-level calibration。
- 验证状态：
  - 当前所有严格可比候选仍低于 BA 0.73。
  - 2026-06-30 最佳 reference 的 aggregate confusion 中 Young recall 最低；
    因此不能继续沿用“Young 一定容易、只剩 Pre-vs-Robust”这一旧结论。
- 后续建议：
  - 先统一 benchmark 和跨实验分析，再新增模型；否则协议差异会继续掩盖真实改进。

---

### 7. Sweep Analysis 与 Strict Holdout Evaluation 模块

- 当前最新版本：单目录分析可用；跨协议整合和 config identity 仍需升级。
- 相关文件：
  - `analyze_sweep.py`
  - `frailty_3class_holdout_eval.py`
  - `frailty_3class_overfitting_sweep.py`
- 关键函数/逻辑：
  - `aggregate_config_summary`
  - `build_leaderboard`
  - `train_eval_holdout_once`
  - `stage1_grid`
  - `stage2_grid`
  - `train_eval_groupkfold_once`
- 模块用途：
  - 对 frailty3 sweep 结果进行 config-level 汇总。
  - 选择 top configs。
  - 区分 strict holdout、early-stopping CV、fixed-epoch CV 和泄漏历史结果。
  - 做 overfitting、regularization 和 generalization sweep。
- 输入：
  - `results_frailty3/` 下的 sweep run outputs。
  - `results_frailty3/_sweep_analyse/`
  - `results_frailty3/_holdout_eval/`
  - `results_frailty3/_overfitting_sweep/`
- 输出：
  - `clean_runs.csv`
  - `config_summary.csv`
  - `leaderboard_top_configs.csv`
  - `incomplete_configs.csv`
  - `class_level_summary.csv`
  - `top_config_confusion_matrices_long.csv`
  - `analysis_report.md`
  - holdout summary/report/plots。
  - overfitting sweep summary。
- 已实现内容：
  - `analyze_sweep.py` 可读取历史 artifacts、递归恢复 report 路径、按 config
    聚合 repeats，并按 reference-specific expected repeats 检查完整性。
  - strict holdout 支持 train/inner-val/test 三分法。
  - overfitting sweep 已支持 no-early-stopping fixed final epoch、5-fold StratifiedGroupKFold、stage1/stage2。
- 协议边界：
  - `20260527_1320_cnn_inceptionTime` 的原始绝对 BA 存在 data leakage，
    只能用于参数探索历史；其 current-protocol reference reruns 才能比较。
  - `overfitting_20260608_0752` 使用 holdout/early stopping，与当前 fixed-epoch
    5-fold CV 不可直接排名。
  - 当前主协议没有 CV 之外的独立 test set；5-fold 汇总是全数据的 OOF validation。
- 关键实验结果：
  - 2026-06-08 baseline：
    `20260608_1206_overfitting_sweep_stage1_rank2`，930 runs、186 configs。
    当时 top `s1_085` BA 0.623148、macro F1 0.625782、
    BA std 0.069952、CI low 0.536305、worst-class F1 0.539840，
    train-validation window BA gap 0.439868。
  - 2026-06-25：
    `20260625_2320_overfitting_sweep_stage1_rank2`，645 runs、129 configs，
    全部完整。Top `s1_122` BA 0.610185、macro F1 0.603988、
    std 0.061454、CI low 0.533892、worst-class F1 0.509158、
    train-validation gap 0.463349。配置为 top50 SQI、quality-weighted aggregation、
    epoch 15、lr 0.001、wd 0.005、dropout 0.5、label smoothing 0.2、
    weighted CE、inverse-subject-count、5 秒/50% overlap。
  - 同轮中 morphology-only 最大 BA 约 0.463，
    morphology+PPI/HRV 最大约 0.439；balanced softmax 最大约 0.558，
    focal 最大约 0.570，weighted CE 最好。以上是该 grid 内观察，
    不能直接解释为因果主效应。
  - 2026-06-30：
    `20260630_0630_overfitting_sweep_generalization_rank2`，
    1160 runs、232 configs，全部完整。最佳 overall 是固定 reference
    `ref_20260625_top1_s1_122`：BA 0.623148、macro F1 0.614629、
    std 0.013734、CI low 0.6061、worst-class F1 0.554286、
    train-validation gap 0.460922。
  - 同一 nominal `s1_122` 在 2026-06-25 为 0.610 +/- 0.061，在 2026-06-30
    reference 为 0.623 +/- 0.014；稳定性本身没有复现，不能称为确定提升。
  - 2026-06-30 最佳新配置 `gen_212` 为 Small InceptionTime：
    epoch 15、wd 0.01、dropout 0.5、label smoothing 0.3、top50 SQI、
    no sampler/all windows、train overlap 30%，BA 0.581481、macro F1 0.5702。
    最佳 full InceptionTime 新配置 `gen_080` BA 0.580556。
  - 该轮描述性均值：top50 SQI 0.5108 vs none 0.4979；
    Small InceptionTime 0.5071 vs full 0.5016；
    train overlap 30% 为 0.5067 vs 0% 为 0.5020；
    no sampler/all windows 0.5349，但与 quota 设计混杂。
    quota 50%/32/16 的均值约 0.528/0.496/0.474。
  - 当前最佳 reference 的 5-repeat aggregate confusion counts，行是真实类、列是预测类：
    `[[35,7,3],[14,34,12],[7,12,21]]`。
    行归一化分别为 Pre-Frail `77.78/15.56/6.67%`、
    Robust `23.33/56.67/20.00%`、Young `17.50/30.00/52.50%`。
    本结果中 Young recall 最低。
- 2026-06-30 current-protocol references：

| Reference | BA mean | Macro F1 mean | BA std |
|---|---:|---:|---:|
| `ref_20260625_top1_s1_122` | 0.6231 | 0.6146 | 0.0137 |
| `ref_20260625_top2_s1_102` | 0.6176 | 0.6081 | 0.0310 |
| `ref_20260608_s1_091` | 0.5648 | 0.5467 | 0.0556 |
| `ref_20260608_s1_105` | 0.5602 | 0.5511 | 0.0431 |
| `ref_20260527_g0068` | 0.5528 | 0.5461 | 0.0546 |
| `ref_20260527_g0056` | 0.5509 | 0.5535 | 0.0976 |
| `ref_20260608_s1_163` | 0.5481 | 0.5449 | 0.0555 |
| `ref_20260608_s1_085` | 0.5426 | 0.5376 | 0.0566 |

- 规范分析输出：
  - 2026-06-16：`results_frailty3/_sweep_analyse/20260616_1143_overfitting_inceptiontime`。
  - 2026-07-06：
    `20260706_0947_overfitting_inceptiontime_small_inceptiontime`、
    `20260706_0947_overfitting_inceptiontime_small_inceptiontime_02`、
    `20260706_0956_overfitting_inceptiontime`。
- 待实现内容：
  - 新建跨实验分析脚本和统一 benchmark。
  - 扩展 `analyze_sweep.py` config columns；当前依赖
    `overfit_config_id/name` 避免错误聚合，结构较脆弱。
  - 默认模型过滤加入 `small_inceptiontime`；当前需要显式 CLI 才会分析。
  - 同时报告 mean BA、macro F1、worst-class metrics、CI low、std、
    class confusion、coverage 和 train-validation gap。
- 算法思路：
  - config-level 聚合，避免单 run 偶然性。
  - strict holdout 中 test 不参与 early stopping。
  - final deployment model 不从 5 个 CV fold 里挑最高分，而是在锁定 config
    和 epoch 后用明确的 final-training protocol 重训。
- 验证状态：
  - 上述三个完整 sweep 的 run/config 数量和 canonical reports 已核对。
  - 明显 train-validation gap 仍存在；尚无 BA >= 0.73 的严格可比结果。
- 后续建议：
  - 先完成 benchmark、跨 sweep metadata normalization 和 ablation，
    再决定是否进行新的大规模 grid。

---

### 8. ShapeFormer 模块

- 当前最新版本：port 已实现，但不是当前主优先级。
- 相关文件：
  - `shapeformer_port.py`
  - `frailty_3class_classifier.py`
- 关键函数/类：
  - `ShapeletBundle`
  - `ShapeFormerAttention`
  - `LearnablePositionalEncoding`
  - `ShapeBlock`
  - `PortedShapeFormer`
  - `discover_shapelets`
  - `discover_shapelets_pisd`
  - `train_shapeformer_model`
- 模块用途：
  - 将 ShapeFormer/ShapeFormer-PISD 思路接入 frailty3 time-series classifier。
- 输入：
  - frailty3 raw windows。
  - 可选 PPI/HRV fusion features。
- 输出：
  - ShapeFormer model predictions。
  - 与 CNN/InceptionTime 可比的 CV/sweep 指标。
- 已实现内容：
  - 核心结构移植。
  - PISD discovery wrapper。
  - `forward_features()` 与 feature fusion 对接。
- 待实现内容：
  - 如果继续，应做小范围 ablation，不应放入超大 sweep。
  - 需要明确其 ranking metric 是否与 CNN/InceptionTime 完全一致。
- 历史版本与更新目的：
  - 用户关注 ShapeFormer 移植完整性和结果不提升原因。
  - 当前结论是原版不能无改动套用，需要适配数据接口、input dimension、device/batch、shapelet discovery、output head。
- 算法思路：
  - `shapeformer` 使用较快的 effect-size discovery。
  - `shapeformer_pisd` 使用原版 PISD discovery wrapper，但运行成本高。
- 可能改进方向：
  - 仅在 InceptionTime/CNN 达到稳定基线后，再用小规模实验验证 ShapeFormer 是否有必要。
- 验证状态：
  - 已实现，但提升不明确。
- 后续建议：
  - 保留接口和记录，不作为当前提分主路径。

---

### 9. ASA 旁支实验模块

- 当前最新版本：旁支实验，不纳入 frailty pipeline 主线。
- 相关文件：
  - `asa_classifier.py`
  - `test_asa_classifier/`
  - `test_asa_classifier/_vitaldb_signal_cache/`
- 关键函数/类：
  - `AsaConfig`
  - `PpgRawBranch`
  - `PpgSpecBranch`
  - `RrSeqBranch`
  - `HrvBranch`
  - `PpgFeatureBranch`
  - `MultiBranchAsaModel`
  - `train_fold`
- 模块用途：
  - VitalDB ASA 1/2/3 三分类实验。
  - 仅作为模型试验/方法验证。
- 输入：
  - VitalDB 中同时包含 ASA、PLETH、ECG_II 的数据。
  - 删除 ASA 4/6/NaN。
  - 支持 PPG-only、ECG-only、ECG-peaks-only 输入。
- 输出：
  - ASA classifier scorecard、summary、模型、图表、预测 CSV。
- 已实现内容：
  - subject-level split。
  - StratifiedGroupKFold。
  - class weighting。
  - fold 内 normalization 防泄漏。
- 待实现内容：
  - 无需纳入 frailty 主线。
  - 如果保留，应明确标注为 side experiment。
- 历史版本与更新目的：
  - 用于验证 PPG/ECG/peaks 在 ASA 分类中的可分类性。
- 算法思路：
  - 多分支模型融合 raw/spec/RR/HRV/features。
- 可能改进方向：
  - 仅作为方法参考，不迁移结论到 frailty。
- 验证状态：
  - 非主线。
- 后续建议：
  - 防止后续 chat 把 ASA 当作 frailty 结果。

---

### 10. 当前主分析 Notebook 与旧入口关系

- 当前最新版本：
  - `ppg_analyse4_calib.ipynb` 是当前主分析 notebook。
  - `ppg.py` 已过时。
- 相关文件：
  - `ppg_analyse4_calib.ipynb`
  - `ppg.py`
  - `funcs.py`
- 模块用途：
  - 主 notebook 应作为最终整合入口，连接预处理、motion detector、dynamic heartbeat extractor、静态 waveform analysis 和 frailty features。
- 输入：
  - 项目 PPG/IMU 数据。
  - detector / peak-IBI 模型输出。
- 输出：
  - 分析图、校准结果、pipeline 中间结果。
- 已实现内容：
  - 旧阶段曾整合 denoiser 复用、compare plot、ONNX runtime 方向。
- 待实现内容：
  - 将旧 denoiser 路线替换为 motion detector + dynamic heartbeat extractor。
  - 整合 `ppg_peak_hr_gating_train.py` 的输出。
  - 明确 notebook 与脚本模块之间的职责边界。
- 历史版本与更新目的：
  - `ppg.py` 曾是旧入口，但用户已明确其过时。
  - `ppg_analyse4_calib.ipynb` 应作为当前主分析 notebook。
- 算法思路：
  - notebook 负责集成、校准、可视化和人工检查。
  - 可复用逻辑应尽量沉淀回 `.py` 模块，避免 notebook-only 隐性状态。
- 可能改进方向：
  - 建立 notebook 到 script 的清晰接口。
  - 输出固定格式中间结果，便于 frailty classifier 使用。
- 验证状态：
  - notebook 主入口地位 confirmed。
  - 新 detector + peak/IBI 集成未完成。
- 后续建议：
  - 优先把动态心搏模块接入 notebook。

---

### 11. `final_pipeline_v2` 实验闭环、报告与完成 case 恢复

- 当前最新版本：`final_v0/final_pipeline_v2` 当前活动实现。
- 相关文件：
  - `final_v0/final_pipeline_v2/src/ppg_frailty/experiment.py`
  - `final_v0/final_pipeline_v2/src/ppg_frailty/study/`
  - `final_v0/final_pipeline_v2/frailty_3class_sweep_v2.py`
  - `final_v0/final_pipeline_v2/frailty_3class_final_refit_v2.py`
  - `final_v0/final_pipeline_v2/tools/recover_completed_cases_v2.py`
- 模块用途：
  - 固化 study/cell/resume/final-refit 的配置、源码和 artifact provenance。
  - 从同一份 held-out OOF 生成三种仅报告用 participant aggregation 视角。
  - 将多个旧 study 中已完成且可校验的 case 汇入独立恢复目录，供统一报告使用。
- 报告视角：
  - `W`：window equal-weight；先按 participant 汇总全部 window。
  - `A`：Line A equal-file；先得到 file 概率，再对同一 participant 的 file 等权。
  - `B`：Line B equal-role；先在 role 内汇总，再对 participant 的可用 role 等权。
  - 三个视角只重聚合同一 held-out OOF，不代表三次训练；只有 case 声明并保存了相应
    source evidence 时才输出该视角。缺少 window OOF 的 feature/matrix/fusion case 明确
    标为 N/A，不从 participant-level 结果反推或伪造 window 结果。
- 可复现性边界：
  - `features`/`aggregation` 配置使用 exact-key、exact-value fail-closed。
  - final-refit 要求 OOF 的源码 SHA-256 与当前完整 `ppg_frailty` 源码树一致；旧
    placeholder hash 不能绕过预检。
  - `measure_operational_costs` 属于 execution/resume contract，不改变科学算法默认值。
- diagnostics 边界：
  - 保存小型 detector、运行参数、polarity、pairing-rule、聚合值/有效性、计数和失败摘要。
  - 不保存逐搏 pairing rows 或 `beat_audit`；该压缩不得改变 predictor、route、
    retention、aggregation 或 prediction。
- 已实现内容：
  - 三视角表格、全类别指标图、confusion matrix 和 HTML 图像引用。
  - 13 completed cases / 65 cells 的独立恢复 bundle；原 study 和冗余 diagnostics 保持原地。
  - 仅归档一个已证明不可达的旧 `train/sampling.py` facade；活动源码仍全部静态可达。
- 验证状态：
  - safe suite 298/298；study/report 46/46；恢复 hash/index 完整性检查通过。
  - 未运行新训练；恢复结果不构成新的模型选择证据。
- 后续建议：
  - 代码稳定后再生成新鲜 formal 5×5 OOF，满足 source hash 和 operational contract。
  - 对旧文档中已陈旧的 Line A/Line B、seed 和 active-version 表述单独做文档对齐。
  - 测试耗时若需继续优化，优先考虑 package lazy import 与 fixture 复用，而非继续归档
    仍被 registry 或公共 facade 使用的源码。
