# MAdenoiser 已确认后续路线：从 SQI、Motion-29 到 Frailty 特征选择

## 1. 决策身份与边界

- 决策 ID：`M0-MAD-001`
- 决策日期：2026-08-03
- 来源：用户在 M0 扩展验收后的明确路线指示
- 状态：`route_confirmed_implementation_not_started`
- TODO 影响：M1、M2、M3、M4.1、M4.2、M4.3、M5、M6.1、M6.2、M7、M8、M9
- 写入边界：后续代码、测试、缓存和结果仍只能写入 `final_v0/`
- 非授权事项：本决定不等于进入 M1，不授权联网、不授权新增依赖，也不把尚未实现的方法写成已有结果

本路线把 SQI 视为四条 MA/HR/PPI 路线的共同质量层，而不是第五条竞争路线。四条候选路线只允许改变信号增强或轨迹前端，必须共用 HR/PPI 后端、窗口、split、seed、SQI、coverage 和评价器，才能归因比较。

## 2. 用户确认的串行主线

1. 完善 SQI：补充部分或全部偏度、峰度、自相关、模板相关、归一化谱熵、完整 IBI 生理合理性和 RED/IR 一致性。
2. 完善 29-subject Motion detector 的阈值与 CV，把 OOF motion probability 融入 SQI。
3. 依次实现四条路线：
   - `R1 spectral_track_sqi`：谱域抑制、候选谱峰、时序轨迹、SQI；
   - `R2 dual_ppg_bss_sqi`：双波长 BSS 前端、公共 HR/PPI 后端、SQI；
   - `R3 nonstationary_sqi`：非平稳分解前端、公共 HR/PPI 后端、SQI；
   - `R4 adaptive_sqi`：自适应滤波前端、真实 HR 保护门、公共 HR/PPI 后端、SQI。
4. 在 `pulse-transit-time-ppg` 上用 `peaks` 参考做四路线初步 HR/PPI benchmark。
5. 在新的 frailty feature matrix 中分别加入四路线输出的 motion HR/PPI；固定相同 seeds 与 subject-level 5-fold，按预注册规则选择开发阶段最高 BA 组合。

## 3. 数据事实与必须分开的 cohort

### 3.1 Frailty Motion-29

- 原始路径：`PPG_Testing_05_01_2026/StudyData/*.csv` 与 `PPG_Testing_05_01_2026/TestDataYoungers/*.csv`。
- 完整缓存：`datasets/frailty3_features_ppi_hrv_all_roles_fs400_w10_h5.csv`。
- 规模：29 subjects、261 个原始文件、每人 9 个角色 `B/R1/R2/R3/R4/S1/S2/W1/W2`。
- 类别：12 robust non-frail、9 pre-frail、8 young。
- 通道：`RED,IR,AX,AY,AZ,GX,GY,GZ`。
- 当前 classifier 默认只用 `B/R1/R2/R3/R4`；`S1/S2/W1/W2` 只在 `all_roles` 加入。
- 当前没有窗口级人工 motion-artifact 真值；角色只能作为待确认的 activity proxy。

### 3.2 PTT 监督 benchmark

- 路径：`physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/s*_*.csv`。
- 规模：22 subjects、66 records；每人 sit/walk/run。
- 采样率：500 Hz。
- 关键列：`ecg,peaks,pleth_1..6,a_x,a_y,a_z,g_x,g_y,g_z`。
- `peaks` 是人工核验 ECG R-peaks，可生成 ECG-derived HR/RRI reference；不是未校正的 PPG pulse-peak 真值。
- RED/IR 候选：distal `pleth_1/pleth_2`，proximal `pleth_4/pleth_5`；波长/位置仍需数据字典最终确认。

### 3.3 现有 Motion A/B 不能冒充 Motion-29

现有 `ppg_peak_hr_gating_train.py` benchmark 使用 PTT 22 subjects 与 SIM 13 subjects。完整运行把 PTT 分成 15 train、4 validation、3 holdout；它不是 29-subject CV。现有阈值在单一 validation 的重叠窗口上搜索，完整 run 两模型阈值均为 `.05`，早期 smoke 却出现 `.90/.95`，因此当前阈值未稳定校准。

## 4. Stage 1：SQI-v2

### 4.1 分量合同

| 分量 | 当前复用点 | SQI-v2 要求 |
|---|---|---|
| 偏度/峰度 | `time_features` 已计算 | 在独立形态副本上计算，输出原值、质量映射与有效标志 |
| 自相关周期性 | 当前缺失 | 只在 35–210 bpm lag 范围搜索归一化峰，并输出峰值与 lag |
| 模板相关 | 当前缺失 | 训练折高质量 beat 建模板，输出中位相关、离散度、有效 beat 数 |
| 归一化谱熵 | `spectral_features` 有未归一化 entropy | 在预注册频带内除以 `log(Nbins)`，再转为质量方向 |
| IBI 合理性 | `clean_pp_intervals` 有 35–210 bpm 与 MAD | 输出 valid fraction、异常率、CV、MAD outlier、漏峰/双峰标志 |
| RED/IR 一致性 | morphology/per-window 有 corr/lag/ratio | 增加双通道峰匹配、HR/PPI差、最大相关与单通道回退 |
| motion | 当前是标准化后 RMS penalty | 改用物理单位副本并接收 OOF `motion_probability` |
| 心率带与峰密度 | 当前已有 | 修复 `peak_density` 硬编码 400 Hz，统一使用实际 `fs` 与公共 peak 后端 |

用户所说“部分或全部”解释为：实现层尽量提供全部分量；是否进入最终 composite 由训练折内消融和预注册规则决定，禁止看 outer-test 或 frailty test BA 后选择。

### 4.2 接口与输出

```text
compute_sqi_components(
    red_raw, ir_raw, acc_raw, gyro_raw, fs,
    motion_probability=None,
    template_state=None,
    calibration=None
) -> {
    components,
    overall_sqi,
    accepted,
    reason_codes,
    peak_indices,
    version,
    parameters
}
```

必须从标准化前的原始/校准信号计算 motion 与原始形态分量；classifier 的逐窗标准化副本只供模型输入。所有 percentile、模板、权重、阈值和校准器只允许在对应训练折拟合并序列化。

## 5. Stage 2：Motion-29 阈值、CV 与 SQI 融合

### 5.1 监督语义门

当前 29-subject 数据没有 artifact truth。实施前必须由用户确认以下之一：

1. 提供或制作窗口级 optical-artifact 人工标签；或
2. 明确把 `B/R/S/W` 映射为 activity proxy，并且输出只能称 `activity/motion state`；或
3. 定义基于独立生理失败的标签，例如公共 peak 后端相对可信 reference 的不可用性，但该标签不能由同一待测 SQI 自循环生成。

没有该决定时，可以完成接口、数据清单和无监督 motion score，但不能报告 29-subject detector BA/F1。

### 5.2 严格 CV

- seeds 固定为 `[42,10042,20042,30042,40042]`。
- outer：`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)`，group=`subject`。
- inner：只在 outer-train subjects 内训练 detector、校准 probability、选择 threshold。
- outer-test 只推理一次；不能用于 early stopping、threshold、SQI weight 或特征选择。
- threshold 主目标：subject-macro BA；同时报告 macro-F1、ROC-AUC、PR-AUC、Brier、ECE、固定 specificity 下 sensitivity。
- 重叠窗口必须做 record/subject 聚合或 block bootstrap，不能当独立样本计算窄 CI。
- 比较 `current SQI`、`SQI-v2 without detector`、`detector only`、`SQI-v2 + detector`。

### 5.3 融合语义

SQI 接收连续 `motion_probability`、校准后的 `motion_quality=1-motion_probability` 和阈值状态；硬标签只用于诊断与拒绝原因。RED 与 IR 可先分别进入既有 10-channel detector，再在 inner CV 比较 `min/mean/max` 融合；若改为 RED+IR 11-channel，则属于新模型，不能继承旧结果。

## 6. Stage 3：四条 MA/HR/PPI 路线

### 6.1 公共后端

四条路线必须共用：

- 相同重采样、窗口和时间戳；
- 相同 motion probability 与 SQI-v2；
- 相同 peak candidate/event merge；
- 相同 HR/PPI 生成、异常处理、coverage 与 `no_estimate`；
- 相同 subject split、seed、预算、metric 和失败码。

### 6.2 路线差异

| 路线 | 唯一允许变化的核心 | 必须的安全/回退 |
|---|---|---|
| R1 spectral | IMU谱污染概率、soft mask、谐波候选、Viterbi/Kalman/Particle | raw STFT argmax 与 SQI-only baseline |
| R2 BSS | RED/IR PCA、FastICA、STFT-NMF 与分量选择 | best single channel；单通道时拒绝 BSS |
| R3 nonstationary | DWT/WPT/threshold、EMD/EEMD/VMD/SSA 前端 | raw/bandpass；参数只在 train 拟合 |
| R4 adaptive | Wiener/LMS/NLMS/RLS/IMU-ANC 前端 | 真实 HR 上升与 pulse energy 保护门；失败即风险基线 |

## 7. Stage 4：PTT 初步 HR/PPI benchmark

### 7.1 Reference

- 用 `peaks` 的 ECG R-R interval 生成 reference HR/RRI。
- HR/PPI interval 比较可直接进行，但必须标记 PPI-vs-RRI 为 surrogate physiological reference。
- 若评价绝对 PPG pulse event timing，必须只在 train/reference 协议内拟合 ECG→PPG delay，再在 test 应用；不能把 ECG timing 直接称 PPG pulse truth。

### 7.2 指标

- HR：MAE、RMSE、bias、95% limits、coverage、连续性。
- PPI：MAE、MedAE、RMSE、有效率、漏检率、额外 beat 率。
- Event：delay-aware precision/recall/F1 与 timing error。
- 分层：subject、sit/walk/run、RED/IR、route、motion/SQI strata。
- 统计：subject bootstrap CI；不以 pooled overlapping windows 作为独立样本。

## 8. Stage 5：29-subject Frailty feature matrix 与最终选择

### 8.1 三层表

1. event 表：`path,subject,role,route,peak_time,ppi_ms,hr_bpm,confidence`。
2. window 表：`window_start/end,motion_probability,sqi components,accepted,coverage,reject_reason,HR/PPI摘要`。
3. file feature 表：以 `path+subject+role` 严格左连接现有 145-row static cohort，任何路线失败都保留该行并写 NaN、coverage 与 missing flag。

### 8.2 公共特征语义

- `motion_hr_mean/median/std/iqr/slope_bpm`
- `motion_ppi_mean/median/std/iqr/cv/rmssd_ms`
- `valid_window_fraction,accepted_beat_count,sqi_mean,sqi_p10,motion_fraction`
- `missing_hr,missing_ppi,no_estimate,reason_codes`

每条路线写独立 feature block；classifier 必须显式选择一个候选 block，不能把四路线同名列全部自动收入后再宣称是单路线结果。

### 8.3 公平候选与 split

- 初始候选固定为 13 项：baseline，以及 4 routes × `{HR-only,PPI-only,HR+PPI}`。
- 第一轮不搜索任意跨路线子集；若以后搜索，必须另行预注册并进入 inner CV。
- cohort 固定：29 subjects、145 static-role rows；`all_roles` 作为独立协议，不与 static 混榜。
- 所有候选逐字节复用同一 outer split manifest 和同一五组 seeds。
- outer-test subject-level BA 是主指标；macro-F1、worst-class recall/F1、coverage、拒绝率、稳定性为共同约束。

### 8.4 “最高 BA 组合”的严格解释

用户确认以相同 seeds 的 5-fold CV 最高 BA 组合锁定最终版本。为避免 winner's curse，执行含义固定为：

1. 每个 outer fold 只在 outer-train 的 inner group CV 选择路线、HR/PPI block、SQI/motion threshold、模型和超参数。
2. 用全 outer-train 重训后，outer-test 只预测一次。
3. 汇总 outer OOF subject predictions，报告 nested-CV BA 和选择频率。
4. 最终锁定稳定赢家并在 29 subjects 全量重训。
5. 若直接在同一非嵌套 5-fold 的 13 项中选最高 BA，该分数只能写成 `development-selected CV BA`；最终论文性能仍需 untouched/external cohort。

禁止用 outer-test frailty BA 反向调 motion threshold、SQI 权重或信号前端；这些参数只能由独立质量/生理监督或 inner folds 决定。

## 9. 计划输出路径

```text
final_v0/benchmarks/m0_signal_routes/<run>/
├── manifests/
├── sqi_v2/
├── motion_29/
├── ptt_route_benchmark/
├── route_features/
├── frailty_route_cv/
└── report/
```

关键文件至少包括：`split_manifest.json`、`route_config_manifest.json`、`event_predictions.csv`、`window_predictions.csv`、`per_record_metrics.csv`、`subject_oof_predictions.csv`、`candidate_summary.csv`、`selection_trace.csv`、`final_locked_config.json` 和 `report.md`。

## 10. 当前状态与下一阻塞门

| 阶段 | 状态 |
|---|---|
| 路线决定与合同 | `confirmed_documented` |
| SQI-v2 | `not_implemented` |
| Motion-29 threshold/CV | `blocked_on_target_semantics` |
| 四路线实现 | `not_implemented` |
| PTT HR/PPI benchmark | `not_run` |
| Frailty route feature/CV | `not_run` |
| 最终锁定版本 | `not_selected` |

下一项必须由用户决定：29-subject motion detector 的监督目标究竟是窗口级 optical artifact、B/R/S/W activity proxy，还是另一个明确定义的 peak/HR 不可用性标签。
