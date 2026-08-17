# 29 人 Activity/Motion 监督合同、早期三分类历史与运动后恢复特征路线

## 1. 本文身份与已确认决定

- 决策 ID：`M0-MOT-001`
- 日期：2026-08-03
- 状态：`supervision_confirmed_implementation_not_started`
- 范围：M0 扩展收尾；本文件不表示已经进入 M1、完成重训或产生新 benchmark
- 写入边界：仅 `final_v0/`；根目录代码、原始数据、历史输出、`AGENTS.md` 与 `_agent/` 均保持只读
- 网络：未使用

用户已确认 29-subject detector 的监督目标是 **activity/motion state**，不是窗口级光学伪影真值：

| 原始角色 | 生理/协议含义 | 主二分类标签 `activity_binary` | 保留的阶段标签 |
|---|---|---:|---|
| `B` | baseline / 基线静息 | 0 | `B` |
| `R1–R4` | 每次活动后的 relax/recovery | 0 | `R1–R4`，并关联前一活动 |
| `S1–S2` | stand-and-sit 往复运动 | 1 | `S1–S2` |
| `W1–W2` | walking | 1 | `W1–W2` |

主任务是可靠地区分静态与动态。`B/R/S/W` 四种阶段仍必须保留，供角色分层误差、辅助多任务头、运动后恢复和 frailty 特征探索使用。任何论文与模型卡都必须使用 `activity/motion detector`，不得改写为 `PPG artifact ground-truth detector`。

## 2. 29 人输入数据的逐字节核验

### 2.1 路径、规模与结构

数据不是单一目录中的 29 人，而是两个目录合并：

| 路径 | 受试者 | 文件 | 每人角色 |
|---|---:|---:|---|
| `PPG_Testing_05_01_2026/StudyData/*.csv` | 21 | 189 | 9 |
| `PPG_Testing_05_01_2026/TestDataYoungers/*.csv` | 8 | 72 | 9 |
| 合计 | **29** | **261** | `B,R1,R2,R3,R4,S1,S2,W1,W2` |

本轮对 261 份 CSV 逐字节读取并核对首行。所有文件表头完全一致：

```text
RED,IR,AX,AY,AZ,GX,GY,GZ
```

数据没有显式时间列；现有项目代码按 `fs=400 Hz` 解释。抽查数值显示 RED/IR 为原始 PPG 计数，ACC 与 GYRO 为设备输出值；正式重训必须继续使用现有单位推断与转换，并将每个文件的单位判定写入 manifest。

### 2.2 按角色持续时间

下表由完整数据行数除以 400 Hz 得到；每个角色均有 29 份文件：

| 角色 | 平均秒数 | 中位秒数 | 最短–最长秒数 |
|---|---:|---:|---:|
| B | 302.337 | 300.565 | 298.715–361.690 |
| R1 | 298.296 | 300.517 | 238.860–302.322 |
| R2 | 298.718 | 300.473 | 239.070–310.527 |
| R3 | 300.434 | 300.502 | 298.993–303.728 |
| R4 | 301.328 | 300.695 | 298.995–314.137 |
| S1 | 21.614 | 21.328 | 10.682–35.500 |
| S2 | 19.454 | 18.435 | 12.170–38.203 |
| W1 | 11.869 | 12.470 | 6.883–17.990 |
| W2 | 10.799 | 10.070 | 6.888–19.918 |

静态数据时长远大于动态数据，若按所有重叠窗口直接训练会造成严重类别、角色与受试者权重失衡。训练采样必须同时平衡 `subject × activity_binary × role_family`，并对 B/R 每文件窗口数设上限；评价仍使用完整 OOF 窗口和 subject-macro 聚合。

## 3. 时序协议与恢复阶段配对

### 3.1 已核验的共同结构

29 人的文件修改时间均呈现：

```text
B → active → R1 → active → R2 → active → R3 → active → R4
```

因此 `Rk` 表示“第 k 次活动后的恢复”，不能硬编码为 `R1↔S1`、`R2↔S2` 等编号配对。每名受试者必须按实际顺序生成 `preceding_active_role`。

本轮识别出六种完整顺序：

| 顺序 | 人数 |
|---|---:|
| `B,W1,R1,S1,R2,W2,R3,S2,R4` | 8 |
| `B,S1,R1,W1,R2,S2,R3,W2,R4` | 7 |
| `B,S1,R1,W1,R2,W2,R3,S2,R4` | 6 |
| `B,W1,R1,S1,R2,S2,R3,W2,R4` | 6 |
| `B,S1,R1,S2,R2,W1,R3,W2,R4` | 1 |
| `B,W1,R1,W2,R2,S1,R3,S2,R4` | 1 |

顺序证据来自文件时间戳，用户同时确认了“先 S/W、后 Relax”的协议含义。实现时应把它冻结为 provisional `StageManifest`；若以后取得正式采集表，以正式表复核并升级 `order_confidence`，不得静默重排旧结果。

### 3.2 StageManifest 最小合同

```text
subject_id,dataset,option,role,stage_family,activity_binary,
bout_index,sequence_index,preceding_active_role,
source_path,fs,sample_count,duration_sec,
order_source,order_confidence,qc_notes
```

`stage_family` 固定为 `baseline / recovery / stand_sit / walk`。`activity_binary` 只用于主 detector；恢复特征、角色分层和论文流程图使用完整阶段信息。

## 4. “早期三分类 CNN”历史追溯结论

### 4.1 结论先行

在当前工作树、Git 历史、归档 Notebook、模型目录和全部混淆矩阵文件名中，**没有找到可复核的 Static/Rest、Sit&Stand、Walk 三分类 CNN 源码、三输出 head、CNN 权重、3×3 混淆矩阵或数值报告**。

找到的最接近资产是三分类/多分类 **SVM** 数据与权重，以及一条明确写着放弃 CNN+LSTM 的 Notebook 记录。用户对“早期三分类尝试、坐站与步行边界模糊”的记忆有直接证据支持；但可复核模型类型是 SVM，不能在论文中写成已验证的三分类 CNN。

### 4.2 三分类 SVM 数据

| 文件 | 文件名字样 | 实际行数 | `Rest` | `Sit/Stand` | `Walk` |
|---|---:|---:|---:|---:|---:|
| `train_window/motion_dataset_552samp_4s_2over.csv` | 552 | 552 | 365 | 145 | 42 |
| `train_window/motion_dataset_791samp_3s_1over.csv` | 791 | **731** | 292 | 327 | 112 |

第二个文件名与实际行数不一致，后续不得只依据文件名登记样本量。

`svm2_dataset_train.py` 与相应 Notebook 的完整标签表实际上是五类：

```text
0 Rest
1 Sit/Stand
2 Walk
3 Transition
4 StrongMotion
```

现存多数训练表只含 0/1/2，少数表另含 3；没有发现 label 4 样本。历史注释中的“合并”是 Sit 与 Stand 合成 `Sit/Stand`，不是把三分类 CNN 合并成二分类。

### 4.3 SVM 算法与已有测试结构

`svm2_dataset_train.py` 实现：

1. PPG、ACC magnitude、GYRO magnitude、jerk magnitude 的时域统计；
2. Welch 频带功率、谱熵和主频；
3. `StandardScaler`；
4. 可选 PCA；
5. `SVC(kernel,C,gamma,class_weight="balanced",probability=True)`；
6. 优先按 source file/segment 做 GroupKFold 与 group holdout；
7. 输出 accuracy、balanced accuracy、macro precision/recall/F1、weighted F1、classification report 与 row-normalized confusion matrix；
8. 保存 `models/svm_motion_<kernel>_<timestamp>.pkl`。

模型目录现有 **649** 个 `svm_motion_*.pkl`：645 个 RBF、4 个 linear，总计 49,537,847 bytes，时间戳范围 2025-09-15 至 2025-11-25。本轮遵守安全边界，未反序列化这些 pickle。

### 4.4 历史结果与混淆矩阵状态

`PPG_Analy_Visual_test.ipynb` 保存的唯一直接评价是：

- SVM 对 rest 较好；
- walking 与 sitting/standing 混淆；
- `DWT + deep learning: CNN+LSTM` 被标记为 `give up`，理由是不可解释、黑箱。

训练器确实有混淆矩阵绘制函数，但两份 SVM Notebook 没有保存 image/png 的 3×3 矩阵，也没有保存 BA/F1 数值账本。排除 frailty 三分类、ASA 三分类、peak/gate 二分类和 sit-vs-motion 二分类后，全库没有 Rest/Sit&Stand/Walk 的 3×3 confusion image。

证据状态：

| 项目 | 状态 |
|---|---|
| 三分类数据 | `verified` |
| 三分类 SVM 训练实现 | `verified` |
| 大量 SVM 权重文件存在 | `verified_without_deserialization` |
| “rest 好、walk 与 sit/stand 混淆”历史评价 | `verified_qualitative_only` |
| 三分类数值指标 | `not_persisted_not_recoverable_as_history` |
| 三分类 3×3 混淆矩阵 | `not_found` |
| 三分类 CNN 源码/权重/结果 | `not_found` |

不能事后重跑 SVM 或训练新 CNN 再把新矩阵冒充为历史矩阵。未来可把相同三类作为可复现 baseline 重建，但必须使用新的 run ID、split manifest 与时间戳，并明确标记 `reconstructed_baseline`。

## 5. 实际可复核的二分类演化

1. `funcs.py` / `ppg.py` 有六状态规则器：`Static, StandUp, SitDown, Walking, Resting, Transition`；其后把 Walking/StandUp/SitDown/Transition 折叠为 motion，把 Static/Resting 折叠为 non-motion。它是规则器，没有训练 CV 或混淆矩阵。
2. v7 的 CNN-BiLSTM autoencoder 从开始就是基于 ACC RMS 或重建误差的二值 motion 代理，不是三分类 CNN。
3. v7.4、Stage1 与 v8 都把 `sit=0, walk/run=1` 作为直接二分类；v8 的高分主要由 IMU 可分性驱动。
4. 当前 `ppg_peak_hr_gating_train.py` 的 A/B CNN 均只有一个 motion logit，并用 BCE 训练。

论文的准确表述应为：

> 早期多类 SVM 在静态类上表现较好，但坐站与步行存在明显混淆；项目随后采用静态/动态二分类代理。仓库未保存可复核的早期三分类 CNN 产物或 3×3 混淆矩阵。

## 6. 当前 PTT/SIM 二分类 CNN：可迁移结构与域偏移证据

### 6.1 模型与输入

`ppg_peak_hr_gating_train.py` 的 `LightCnnMotionDetector` 使用：

```text
PPG,
dynamic ACC x/y/z,
GYRO x/y/z,
ACC magnitude,
GYRO magnitude,
jerk magnitude
```

共 10 通道；256 Hz、8 s 窗、2 s hop。网络为 Conv1d kernel 9/7/5、GroupNorm、GELU、池化、global pooling 和单 logit。PTT 标签是 `sit=0`、`walk/run=1`。

### 6.2 完整 balanced_v2 运行

来源：`.CNN_results/20260427-01_peak_hr_gate_balanced_v2/detector_benchmark/`。

| 模型/分割 | 阈值 | 窗口 | BA | F1 | ROC-AUC | 混淆矩阵 `[[TN,FP],[FN,TP]]` |
|---|---:|---:|---:|---:|---:|---|
| A validation PTT | .05 | 2,896 | 1.0000 | 1.0000 | 1.0000 | `[[965,0],[0,1931]]` |
| A holdout PTT | .05 | 2,164 | 1.0000 | 1.0000 | 1.0000 | `[[721,0],[0,1443]]` |
| A external SIM | .05 | 12,032 | .7699 | .7542 | .8269 | `[[4772,740],[2125,4395]]` |
| B validation PTT | .05 | 2,896 | 1.0000 | 1.0000 | 1.0000 | `[[965,0],[0,1931]]` |
| B holdout PTT | .05 | 2,164 | 1.0000 | 1.0000 | 1.0000 | `[[721,0],[0,1443]]` |
| B external SIM | .05 | 12,032 | **.7802** | **.7634** | **.8642** | `[[4856,656],[2090,4430]]` |

完整运行中 B light CNN 是 external SIM 的较优候选，但仍是 pooled overlapping-window 结果，不是 detector 五折 CV，也没有 subject bootstrap CI。

### 6.3 Smoke 运行与稳定性警告

1-epoch、64 Hz、4 s/4 s 的 smoke 运行中，A external BA/F1 为 `.6918/.7406`，B 为 `.6255/.4041`。阈值分别为 `.95/.90`，与完整运行的 `.05` 差异巨大。

PTT 内部接近满分、external SIM 显著下降，且 smoke 与完整运行阈值不稳定，构成直接域偏移证据。它支持“复用结构/初始化、在本地29人设备域重训并重新校准”，不支持直接部署旧阈值或把 PTT 满分写成跨设备泛化。

## 7. 29 人重训与阈值/CV 实现合同

### 7.1 必做对照

1. `imu_rule`：可解释 ACC/GYRO/jerk 阈值规则；
2. `imu_only_light_cnn`；
3. `ppg_only_light_cnn`；
4. `ptt_pretrained_10ch_finetune`：沿用旧 Light CNN 结构和可兼容权重；
5. `local_10ch_from_scratch`；
6. `red_vs_ir`：RED 与 IR 分别替代单 PPG 通道；
7. `dual_ppg_11ch`：RED+IR 的新模型消融，新增输入通道时不得宣称完整继承旧权重；
8. 可选 `binary_main + B/R/S/W_aux_head`：主目标仍是二分类，四阶段仅作 fold-local 辅助任务和探索特征。

### 7.2 切分、训练与阈值

- group 固定为 subject，任何同一人的九个角色不得跨 fold。
- 使用同一五组 seeds：`42,10042,20042,30042,40042`。
- outer 5-fold 只评价；inner subject-grouped split 负责 early stopping、预训练/冻结选择、概率校准和阈值。
- 若使用 cohort/frailty/sex 平衡 fold，这些字段只能用于 subject-level split balancing，不进入 detector 监督或输入。
- 采样按 subject、主二类和角色族平衡；B/R 的长时记录不得淹没 S/W。
- 动静态段首尾加入预注册 guard interval，guard 窗标记 `transition_uncertain`，不强制错误二值标签。
- 每个 outer-test subject 只生成一次严格 OOF `p_active`；最终全量模型必须在选择结束后另行重训。
- 阈值目标优先 subject-macro BA；同时报告固定 specificity 下 sensitivity、F1、AUROC、AUPRC、Brier、ECE 与 quality–coverage 曲线。

### 7.3 必需结果

每个 run 至少保存：

```text
stage_manifest.csv
split_manifest.json
window_predictions_oof.csv
subject_metrics.csv
role_metrics.csv
thresholds_by_fold.json
calibration_by_fold.json
binary_confusion_matrix_oof.png
role_conditioned_confusion_matrix.png
probability_histograms_by_role.png
model_card.md
failure_table.csv
```

二分类混淆矩阵必须给原始计数与行归一化比例；另按 B、R、S、W 分层报告 false-motion/false-static。若做四阶段辅助头，单独保存 4×4 矩阵，不能与主二分类结果混榜。

## 8. Motion 融入 SQI 的合同

SQI 接收 fold-local、校准后的连续 `p_active`，而不是直接把所有 motion 窗硬删除：

```text
motion_quality = 1 - calibrated_p_active
```

推荐同时输出：`p_active`、`activity_binary_at_frozen_threshold`、`motion_quality`、`transition_uncertain`、`role` 与 `detector_version`。比较 `SQI-v2 without motion`、`detector only`、`SQI-v2 + p_active` 和 `SQI-v2 + hard reject`。阈值不得通过 outer-test frailty BA 反向调整。

## 9. S/W→Relax 的 frailty 时序特征路线

### 9.1 配对单位

每个活动 bout 与其紧随的 Rk 构成一对：

```text
(preceding_active_role, Rk, subject_id, bout_index)
```

所有 HR/PPI 必须来自同一路线的 OOF 输出，并附 SQI、coverage、有效窗口数和失败码。现有标签表含 `HRbaseline`、`R1–R4_HRrecovery` 与 maxHR 字段，但生成算法、延迟处理和来源未找到；这些字段暂不作监督真值，只能作辅助一致性检查。

### 9.2 Active 特征

- S/W 各 bout 的 HR peak、P95、median、min、range、robust slope；
- 相对 B 或前一恢复末段的 `ΔHR`；
- PPI minimum、median、CV、RMSSD；
- 活动时长；
- ACC/GYRO motion dose、AUC 与活动强度分位数；
- SQI 与有效覆盖率；
- S1/S2、W1/W2 重复性和次序效应。

### 9.3 Recovery 特征

- R 开始后 `0–30、30–60、60–120、120–300 s` 的 HR/PPI 稳健摘要；
- `HRR30/60/120 = HR_active_peak_or_end - HR_recovery_t`，定义必须预注册；
- robust recovery slope；
- 可辨识时的指数衰减时间常数 `tau`；
- 回到 `baseline+5 bpm`、`baseline+10 bpm` 的时间；
- baseline 以上 HR AUC；
- PPI rebound、PPI slope 与恢复 HRV；
- 按活动 `ΔHR` 或 motion dose 归一化的恢复能力；
- 低 SQI、coverage 不足、未回到基线和拟合失败的显式标志。

### 9.4 下游选择

现有 frailty classifier 默认只使用 B/R；`train_all_roles` 只是把动态窗口加入训练，不是显式恢复特征。新特征矩阵应单独增加：

```text
route__active__*
route__recovery__*
route__paired_response__*
route__coverage__*
```

四条 MA/HR/PPI 路线必须复用相同 StageManifest、outer folds、seeds、SQI、缺失合同和特征定义。最高 BA 按 nested subject CV 选择；非嵌套同一 5-fold 最高分只称 `development-selected CV BA`。

## 10. 论文材料登记

### 10.1 可写结论

- 早期多类 SVM 对静态识别较好，但 walking 与 sit/stand 混淆；这是改用静态/动态主任务的历史动机。
- 本地 29 人完整包含 baseline、四次活动后恢复、两次 stand-and-sit 和两次 walk，可建立活动响应与恢复速度特征。
- 当前 PTT/SIM Light CNN 跨域 BA 为 `.7802`，低于 PTT 内部满分，显示设备/环境域偏移并支持本地重训。

### 10.2 不可写结论

- 不得写“已完成三分类 CNN 并获得某 3×3 confusion matrix”。
- 不得把三分类 SVM 的未来重跑结果写成历史结果。
- 不得把 B/R/S/W activity 标签写成窗口级 optical artifact 真值。
- 不得把 PTT 内部满分写成29人或跨设备性能。
- 不得在尚未运行时声称恢复速度、运动 HR 上下限或 frailty BA 已有结果。

## 11. 当前状态与下一步

| 项目 | 状态 |
|---|---|
| Activity/motion 监督语义 | `confirmed_documented` |
| 29人数据与时序协议 | `verified` |
| 早期三分类历史追溯 | `verified_svm_assets_cnn_not_found` |
| 历史 3×3 混淆矩阵 | `not_found` |
| PTT/SIM 二分类 A/B 结果 | `verified_existing_output` |
| Motion-29 代码适配 | `not_implemented` |
| Motion-29 nested 5-fold/阈值 | `not_run` |
| Motion→SQI | `not_implemented` |
| 恢复时序特征 | `specified_not_implemented` |
| Frailty 路线比较 | `not_run` |

本小结只完成监督语义、历史证据和实现合同落盘；没有训练模型、修改根代码或生成新性能结果。下一 TODO 实施动作仍需用户确认后开始。
