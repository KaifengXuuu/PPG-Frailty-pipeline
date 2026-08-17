# 历史预处理 Crosswalk / Historical Preprocessing Crosswalk

## 1. 使用规则 / Use rule

本 crosswalk 将根目录历史脚本映射到 M3 公共实现。所有列出的根目录脚本状态均为 `historical_reproduction_only`：允许读取、复现和解释旧结果，但不得成为 M4 以后第二个活动预处理入口。机器证据位于 `evidence/historical_preprocessing_crosswalk_v1.json`，其中保存逐文件 SHA-256、字节数和存在性。

This crosswalk maps root-level historical scripts to the M3 public implementation. Every listed root script is reproduction-only. The evidence JSON, not this prose table, is the checksum authority.

## 2. 逐脚本映射 / Script-by-script mapping

| 历史脚本 / Historical script | 已存在算法 / Existing algorithms | 已识别风险 / Identified risks | M3 迁移落点 / M3 destination |
|---|---|---|---|
| `frailty_3class_classifier.py` | 双极性 peaks；35–210 bpm PPI；record-wide scaling | 峰合同不同；整段 MAD 可能压低恢复转折 | corrected peak/PPI；per-window 或 training-fold-only scaling |
| `funcs.py` | legacy Aboy-like peaks；Euler roll/pitch EKF；gravity LPF | artifact-rejection 位置缺陷；Euler 奇异；计算出的 initializer 未被使用 | quaternion ESKF 主路线；0.3 Hz LPF 对照；corrected physiology |
| `ppg.py` | legacy peaks；PPI/HRV helpers；Euler EKF | 同类 artifact 缺陷；按数值去重 PPI；SDNN 定义不一致 | 保留源峰与 PPI mask；统一 ddof=1；输出命名为 PPG-derived PRV |
| `ppg_peak_hr_gating_train.py` | PPG/IMU motion gate；ECG-supervised peak/HR gate；resampling | legacy 256 Hz；dataset-specific unit heuristics | 400 Hz profile；显式 polyphase provenance；D8 ECG evaluator |
| `pttppg_denoiser_hybrid_core.py` | hybrid denoising；multimodal normalization；peak supervision | legacy rate 假设与历史单位 heuristic | M3 preprocessing/physiology 作为 denoiser 前后共享 adapter |
| `pttppg_denoiser_onnx_runtime.py` | ONNX denoiser runtime；runtime preprocessing | 可能形成第二套 runtime preprocessing | 只保留 denoiser 推理；输入/输出必须经过 M3 profile 与 FeatureBlock |
| `asa_classifier.py` | ASA frailty features；legacy HR/HRV ingestion | physiology 语义旧；M8 决策未冻结 | 使用同一 HR/PPI/PRV 后端后再做 classifier 消融 |
| `svm2_dataset_train.py` | tabular SVM；feature scaling | scaler 未绑定修正 M2 membership | `fit_fold_scaler` 与物化 fold artifact |
| `pttppg_denoiser_v8_masknet.py` | mask-network denoising；spectral/temporal loss | 仅历史候选，尚未 corrected-fold rerun | 作为 M8 denoise arm 候选，不自带另一 peak/scaling 语义 |
| `pttppg_stage2_denoiser.py` | stage-2 denoising；peak-aware objectives | 使用不同 peak backend | objective 可迁移；评价必须调用 D8/M3 公共后端 |
| `ppg_denoiser_dash_utils.py` | preview preprocessing；visual peak helpers | 可视化 helper 不是权威 runtime | 只用于展示；数值来自 M3 已保存结果 |
| `pttppg_pipeline_v7_4_noleak_viz_ae.py` | legacy end-to-end；autoencoder features | 历史 filter contract；需统一重跑 | autoencoder 可作为候选 arm；preprocessing/fold/output 改绑 M1–M3 |

## 3. 算法层迁移规则 / Algorithm-level migration rules

### 3.1 PPG 滤波与峰 / PPG filtering and peaks

- 历史 256/500 Hz 或隐式采样率不得直接改常量“升级”为 400 Hz；必须选择对应 registered profile，必要时显式 resample。
- 历史双极性思想保留，但窗口、prominence、merge、PPI mask、置信度和状态统一由 corrected_v1 定义。
- 历史 abnormal-PPI 处理不得通过删除源峰改变事件序列；M3 分离 raw PPI、valid mask 与 hard-valid NNI view。

Historical peak code may inform reproduction, but corrected_v1 alone defines future peak, PPI, HR, and PRV semantics.

### 3.2 IMU 重力与 motion 特征 / IMU gravity and motion features

- 历史 Euler roll/pitch EKF 不迁移为活动路线；原因是奇异性、初始化缺陷与状态/协方差合同不足。
- 历史 gravity LPF 只迁移为受控 comparator，不是 EKF 失败 fallback。
- 两路线都必须使用同一六轴、显式单位、20/40 Hz sensor frontend、窗口、mask 与 vector jerk 定义。

### 3.3 Scaling 与分类器 / Scaling and classifiers

- 整段或全数据拟合的 scaler 只用于复现旧结果。
- 新实验的 imputer/scaler/amplitude-risk 模型只能在 M2 exact training roster 上拟合，并保存 artifact。
- classifier、denoiser、SQI 与 Motion detector 可以作为不同候选，但不得各自复制滤波、单位或 physiology 定义。

## 4. 历史结果的论文使用 / Use of historical results in the paper

历史 confusion matrix、BA、HR/PPI 或 denoising 图可以作为“路线历史与问题发现”材料，前提是逐项标明：

1. historical script path 与 source SHA-256；
2. 原采样率、单位和旧 preprocessing/peak/scaling 语义；
3. 使用历史 SGKF 还是其他 split；
4. 是否存在 fold 缺类、leakage 风险或未验证的 heuristic；
5. 结论只能描述当时配置，不能并入未来 corrected-fold leaderboard。

Historical outputs may document route history, but they cannot be silently relabeled as M3-corrected results or pooled with the future benchmark.

## 5. 未来单一活动边界 / Single future-active boundary

未来所有新代码通过 `final_v0/M3_unified_preprocessing_and_signal_algorithms/src/m3_signal_core` 调用：

`raw/denoised signal → registered preprocessing → quality/status/masks → shared physiology/features → M1 route/output → M2 fold/provenance`.

任何需要改变 cutoff、相位、采样率、单位、峰规则、PPI 界限或 scaler fit 范围的实验，必须创建新 profile/algorithm ID；不得在调用点传入隐藏参数。

