# V2 算法结构差距分析

审查日期：2026-08-17
审查对象：<code>final_v0/final_pipeline_v2</code> 当前工作区
审查方式：静态源码、配置、测试定义和文档证据复核；本次未运行训练、完整 5×5、PTT benchmark、ablation 或科学指标计算。ensemble focused tests 87/87 已通过；定向 thesis parity 修正合并后的 conda-ml safe suite 为 295/295 通过。这属于实现合同验证，不是正式科学运行。

## 1. 范围与判定边界

本文只回答“当前 V2 的算法结构与 thesis 要求之间还差什么”。命令行易用性、Dash 交互体验、进度条和输出目录产品设计只在它们直接影响科学产物时出现；完整产品形态由单独的产品审查文档覆盖。

“已实现”表示当前源码和配置中存在可追溯实现，不等于已经获得真实数据上的性能证据。凡未运行正式 5×5、外部 PTT 或完整 ablation 的项目，本文不会写成“已科学验证”。

要求优先级如下：

1. 用户后续明确确认的 V2 决策；
2. 新版 <code>CODEX_CANONICAL_PIPELINE_WORKFLOW_V1.md</code>；
3. <code>CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md</code>；
4. 当前 V2 的 ADR、决策配置和算法说明；它们只能解释实现，不能覆盖更高优先级要求。

权威输入快照：

| 输入 | SHA-256 |
|---|---|
| <code>AA_TODO/workflow/CODEX_CANONICAL_PIPELINE_WORKFLOW_V1.md</code> | <code>4bee984206587983821ec4b544a408ac0c5f38f52afb962aff80f813c844d87c</code> |
| <code>AA_TODO/3/CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md</code> | <code>cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000</code> |

状态定义：

| 状态 | 含义 |
|---|---|
| MATCHED | 当前算法、配置身份和主要接线与已裁决要求一致；仍可能缺正式运行证据。 |
| PARTIAL | 低层模块存在，但接线、粒度、产物或验证不完整。 |
| MISSING / DEFERRED | 尚未实现，或用户已明确允许搁置且当前不得启用。 |
| DELIBERATE DEVIATION / ABLATION | 与较早文档不同，但被后续人工裁决覆盖，或被明确保留为具名 ablation。 |
| CONFLICT | 当前源码、配置或文档内部仍有互相矛盾的有效声明。 |

优先级定义：P0 = 正式科学运行或对应 thesis claim 前必须解决；P1 = 正式发布或大规模运行前解决；P2 = 完整论文复现包前解决；DEFERRED = 按人工裁决保持关闭，不重新询问。

## 2. 当前结论

当前 V2 已经具备清晰的核心 frailty 算法骨架：冻结 manifest 和重复分组外层划分、400 Hz 信号视图、校准 roll–pitch EKF、最终确认的八通道 frailty tensor、四种表征、经典模型与深度模型、fold-local ROCKET、channel-specific OSD ShapeFormer、role-aware 聚合、分层 OOF、参与者级指标和统计模块均有明确实现。

当前最重要的未闭合点不是“再增加模型”，而是以下两类：

1. A1/A2 的七状态路由目前按整条 recording 构造一个 segment，而不是对异质 segment 分段；并且 <code>rate_only_direct</code> 在正式特征接线中仍会因为被折叠为 <code>SignalRoute.DIRECT</code> 而进入 morphology/optical 提取。默认 quality=off 时不影响当前静态 frailty reference，但它阻止 A1/A2 路线被严谨启用或用于 thesis claim。
2. A3 ECG-reference endpoint benchmark 和 A4 internal dynamic report 没有完整 runner/规定产物；现有 motion 命令评估的是 activity detector，不等价于 A3/A4。

## 3. 后续人工裁决对两份权威文档的覆盖

| 主题 | 最终采用的 V2 裁决 | 对较早要求的处理 | 当前证据 |
|---|---|---|---|
| Frailty 通道 | canonical raw、fusion、所有 frailty raw 模型和 ShapeFormer 均为 8 通道：RED、IR、A_dyn_x/y/z、GX/GY/GZ。 | 后续人工裁决明确禁止把 A_mag、Omega_mag、J_mag 混入 frailty。 | <code>configs/v2_decision_profile.yaml:113-127</code>；<code>representations/imu_transform.py::RAW_CHANNEL_SCHEMA</code>；<code>representations/raw.py::build_raw_windows</code>。 |
| Motion 通道 | motion reference 同样为上述 8 通道；11 通道仅为 <code>motion_11ch_derived_augmentation_ablation_v2</code>。 | 11 通道不是 frailty reference，也不得静默替代 8 通道。 | <code>configs/v2_decision_profile.yaml:128-140</code>；<code>configs/motion_detector_contract_v2.yaml:113-120</code>。 |
| IMU 重力分离 | calibrated roll–pitch EKF 是 reference；0.3 Hz low-pass 是独立 ablation；失败不得回退。 | 覆盖 workflow 中较早的 Profile A reference 文字。 | <code>configs/v2_decision_profile.yaml:58-64,111-112</code>；<code>signal/motion_imu.py::preprocess_imu_reference</code> 和 <code>::preprocess_imu_lowpass_ablation</code>。 |
| IMU 单位 | internal acceleration 为 g→m/s²；PTT acceleration 依据 V2-036 为 m/s² identity；gyro 为 °/s→rad/s。 | 数据源分别处理，禁止把 PTT acceleration 再乘 9.80665。 | <code>configs/v2_decision_profile.yaml:65-77</code>；<code>signal/motion_imu.py</code>；<code>manifests/ptt_imu_unit_evidence_v2_036.json</code>。 |
| IMU scaling | Frailty 和 motion 8ch reference 只缩放六个 IMU 轴；11ch motion augmentation 才缩放九个 IMU/derived channels。均只用 outer-training participants，median center、IQR/1.349 scale；IQR 退化时用 population SD，再退化为 1；禁止逐窗幅度归一。 | 覆盖更早“全部九通道进入 frailty scaler”的决策。 | <code>configs/v2_decision_profile.yaml:78-110</code>；<code>representations/imu_transform.py::fit_fold_imu_channel_transform</code>。 |
| 聚合 | canonical 为 window→file→role→participant，角色等权；旧 Line A equal-files 只保留具名 ablation。 | workflow 的 role-aware 解析为最终 reference。 | <code>configs/v2_decision_profile.yaml:46-57</code>；<code>training/aggregation.py::aggregate_hierarchy</code>。 |
| ShapeFormer | reference 为 8ch channel-specific OSD/PISD；每个 candidate 只在一个 source channel discovery 和 best-fit；PIP 比例 0.20，三连续 PIP 变长候选，无固定长度/stride，3/class，最多 180 个 participant/file-balanced discovery windows，失败显式。 | 128 仅为 position-search neighbourhood；固定 128/64 是 <code>effect_size_fixed_v1</code> ablation。 | <code>models/pisd_port.py:19-31,46-90,692-942</code>；<code>configs/formal_experiment_catalog_v2.yaml:263-400</code>。 |
| 单模型 CV seed | 五个 repeats 分别使用 split seed 42、10042、20042、30042、40042；同一 repeat 的五个 folds 使用该 repeat seed。最终全 29 人单模型 refit 才固定 seed=42。 | 明确否定“全部 25 cells 固定 42”。 | <code>data/folds.py:48-58</code>；<code>configs/formal_experiment_catalog_v2.yaml</code> 中 <code>outer_cv_repeat_seed_equals_split_seed</code>；<code>configs/v2_decision_profile.yaml:323-330</code>。 |
| Ensemble seed | split seeds 为 42/10042/20042/30042/40042；五成员固定为 50042/60042/70042/80042/90042，跨 repeat/fold 重用；member0 comparator=50042；每 fold 先平均五成员概率，再拼完整 repeat OOF 并每 repeat 计算一次指标；最终 refit 用同一五成员 seed roster。 | 禁止成员挑选、成员指标平均充当 ensemble、25-member pooling、把 outer-fold 模型当部署 ensemble。 | <code>configs/v2_decision_profile.yaml:246-264,323-330</code>；已验证实现见 <code>models/inception.py::CANONICAL_ENSEMBLE_MEMBER_SEEDS</code>、<code>experiment.py</code>、catalog；实现验证状态见 T4。 |
| SQI/route | 默认 quality=off；diagnostics_only 不影响保留、聚合或预测；route 必须等待已注册监督 artifact/hash。 | 不是缺省启用项，不得通过 YAML 布尔值绕过监督证据。 | <code>configs/v2_decision_profile.yaml:39-45,412-418</code>；<code>experiment.py::_quality_mode</code>。 |
| Aura/nolds | Aura 仅做固定 PPI 函数比较，使用 <code>hrv-analysis==1.0.2</code>、<code>nolds==0.6.2</code> 的隔离环境；formal backend 仍为 local manual。 | 不把 Aura 清洗或可选环境引入主 classifier。 | <code>requirements/requirements-prv-aura-compare.txt</code>；<code>features/prv_backend_compare.py</code>；<code>configs/v2_decision_profile.yaml:141-183</code>。 |

## 4. 数据、manifest、QC 与 folds

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| D1 | 统一 typed contracts：manifest、signal views、quality、route、pulse、features、matrix、artifact 和 prediction。 | MATCHED | <code>src/ppg_frailty/contracts.py:12-365</code>，包括 <code>ManifestRow</code>、<code>SignalViews</code>、<code>QualityResult</code>、<code>RouteResult</code>、<code>PulseResult</code>、<code>FeatureVectorV1</code>、<code>OrderedFeatureMatrixV1</code>。 | 降低跨模块字段漂移；为 bundle/OOF 可追溯提供基础。 | 保持接口冻结；新增字段须同步 schema/hash tests。 |
| D2 | 标签、角色、文件身份必须来自版本化 manifest，不从文件名临时推断；重复记录和集合完整性 fail closed。 | MATCHED | <code>data/manifest.py::convert_m2_source_row</code>、<code>::validate_manifest_set</code>、<code>::load_internal_manifest</code>、<code>::audit_manifest</code>；M2 manifest/fold 均带 hash。 | 参与者标签与角色不随脚本路径或命名规则变化。 | 正式运行时把 manifest/fold hash 写进每个 run summary。 |
| D3 | 物理 QC 要有原因码；设备 ADC rail、绝对 scale、设备特定 clipping/saturation 只能在设备证据存在时启用。 | PARTIAL / DEFERRED | <code>data/qc.py::QCThresholds</code>、<code>::assess_numeric_record</code>、<code>::physical_recording_qc_thresholds_v2</code> 已覆盖解析、非有限、时长、flatline、时间轴等；<code>configs/v2_decision_profile.yaml:406-411</code> 明确 V2-006 deferred。 | 通用坏记录可以拒绝；但尚不能严谨声明设备 rail/saturation 质量门槛。 | DEFERRED：取得设备 ADC rail/absolute scale 后再新增具证据阈值；当前不得猜测。 |
| D4 | 外层评估必须 participant-grouped、stratified、冻结并跨模型共享。 | MATCHED + DELIBERATE EXTENSION | <code>data/folds.py::M2_SEEDS</code>、<code>::FrozenFoldRegistry</code>、<code>::validate_frozen_memberships</code>；当前是 5 repeats × 5 folds 的冻结 corrected SGKF，而不是 workflow 示例中的单一 5-fold CSV。 | 匹配比较可配对到相同 held-out participants；重复划分提供 repeat-level 不确定性。 | 不在 runtime 重建 folds；报告中同时保存 split seed 和 fold hash。 |
| D5 | 外部 PTT manifest 要锁定 roster、activity、通道、ECG reference、单位和同步关系。 | MATCHED AT ADAPTER LEVEL | <code>data/external_manifest.py::PttManifestRow</code>、<code>::audit_external_manifests</code>、PTT unit evidence；activity 覆盖检查要求 sit/walk/run。 | 外部数据身份可追溯；但这不等于 A3 endpoint runner 已完成。 | 与 E1–E3 一起闭合 benchmark grid 和规定输出。 |
| D6 | ADR 和模型卡必须明确原论文偏离、聚合、epoch、信号视图和表征。 | MATCHED | <code>docs/adr/ADR-001</code> 至 <code>ADR-012</code>；<code>model_cards/</code> 含 single-network、five-member、ROCKET、ShapeFormer 等卡片。 | 能区分实现合同和性能证据。 | 正式运行后只由 generator 更新结果字段，避免手工改卡片。 |

## 5. 信号预处理、IMU、窗口与归一化

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| S1 | 构造 <code>x_native</code> 与 amplitude-preserving <code>x_filter</code>；linear detrend、third-order 0.2–8 Hz SOS、zero-phase；短序列显式拒绝。 | MATCHED | <code>signal/preprocess.py::REFERENCE_PPG_PROFILE</code>、<code>::design_ppg_sos</code>、<code>::preprocess_ppg</code>、<code>::build_signal_views</code>。 | raw/fusion 和 direct morphology 使用一致、可复现的 PPG reference。 | 0.5–5 Hz 只保留具名滤波 ablation。 |
| S2 | 重力分离需保存协方差和单位转换，EKF 失败不得回退 LPF。 | MATCHED + DELIBERATE OVERRIDE | <code>signal/motion_imu.py::MotionImuCalibration</code>、<code>::fit_motion_imu_calibration</code>、<code>::_run_roll_pitch_ekf</code>、<code>::preprocess_motion_imu_calibrated_ekf</code>；LPF 为独立函数/profile。 | reference 的动态加速度定义稳定；不会因失败悄悄变成另一算法。 | 在每次正式 run manifest 中保留 covariance、calibration source 和 unit conversion。 |
| S3 | internal 与 PTT 的 acceleration/gyro 单位必须按来源处理。 | MATCHED | 决策配置 65–77 行和 <code>signal/motion_imu.py</code> 的 source-specific conversion；PTT unit evidence 禁止乘 9.80665。 | 避免 PTT motion magnitude 被放大约 9.8 倍。 | 正式 PTT runner 对 unit evidence hash fail closed。 |
| S4 | 计算 A_dyn、Omega、J，同时遵守最终 frailty/motion 边界。 | MATCHED + DELIBERATE DEVIATION | <code>signal/motion_imu.py</code> 生成九个 processed/derived signals；决策配置 78–106 行限定 A_mag/Omega_mag/J_mag 只供 motion augmentation。 | Frailty 保持 exact 8ch thesis target；motion ablation仍能检验旋转不变强度信号。 | 不再把 derived channels 加回任何 frailty predictor allowlist。 |
| S5 | Frailty raw/fusion/ShapeFormer 为 exact 8ch；PPG 逐窗 robust normalization，六个 IMU 轴仅 outer-train fold scaler；禁止 IMU 逐窗同幅归一。 | MATCHED NUMERIC IMPLEMENTATION | <code>representations/imu_transform.py::RAW_CHANNEL_SCHEMA</code> 和 <code>::IMU_CHANNEL_SCHEMA</code>；<code>::fit_fold_imu_channel_transform</code>；<code>representations/raw.py::_robust_scale_ppg</code>、<code>::build_raw_windows</code>。 | 保存 motion intensity，同时避免 held-out participant 参与缩放。 | 保留 exact order/hash tests 和 six-axis provenance assertion。 |
| S6 | 运行配置和 provenance 必须如实描述六轴 frailty scaler。 | MATCHED IDENTITY | <code>signal/preprocess.py</code> 与四个 canonical YAML 的 <code>normalization.raw_imu</code> 均为 <code>outer_training_participant_only_median_iqr_over_1p349_population_sd_then_one_axes6</code>；11ch motion augmentation 仍有独立 nine-channel profile。 | 数值实现、config hash、run summary 和论文方法使用同一 six-axis 身份。 | 保持 config/provenance consistency test；不得重新使用旧裸-IQR或 <code>all_9</code> frailty 字符串。 |
| S7 | PPG 每窗 median；IQR/1.349，退化时 SD，再 finite fallback；clip [-8,8]；不对 IMU 执行同样逐窗归一。 | MATCHED | <code>representations/raw.py::_robust_scale_ppg</code>；PPG 与 IMU 分支在 raw builder 中分开。 | 保护 PPG 数值稳定而不抹去运动强度。 | 保持 PPG-only 单元测试和 non-finite/zero-IQR cases。 |
| S8 | raw 5 s / 2.5 s hop，engineering 10 s / 5 s hop；显式 DL sample rate/resampling 和 padding/mask。 | MATCHED | <code>data/windows.py</code>；canonical configs 的 window plans；<code>signal/resample.py</code>；representation tensors/masks。 | 四种表征使用可比较的物理窗口；padding 不被当作有效信号。 | 时间尺度变体只作为一因素 ablation。 |
| S9 | raw/fusion PPG reference 始终来自 <code>x_filter</code>，nonidentity <code>x_ar</code> 只能贡献 rate features，且无 silent substitution。 | MATCHED AT REPRESENTATION LEVEL | <code>signal/views.py::CanonicalSignalViews</code>；<code>representations/raw.py</code>；<code>docs/adr/ADR-012</code>。 | 防止 reducer 改变的波形被误解释为 morphology/amplitude。 | 与 Q4 一起修复 feature extraction 的 rate-only state propagation。 |

## 6. SQI、A1/A2 路由与 artifact reducer

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| Q1 | A1/A2 必须为 typed 七状态纯状态机；Q_rate pass、Q_shape fail 必须是 <code>rate_only_direct</code>。 | MATCHED AT LOW LEVEL | <code>contracts.py:30-47</code> 的 <code>RouteState</code>；<code>quality/routing.py::route_segment_pre_reduction</code> 和 <code>::finalize_rate_recovery</code>；状态校验在 <code>RouteResult</code>。 | 路由语义本身可测试，reducer failure 和 post-Q_rate 失败可显式表达。 | 保持低层状态机纯函数；正式启用前完成 Q3/Q4。 |
| Q2 | SQI 监督阈值未有证据时不得影响 retention、aggregation 或 prediction。 | MISSING / DEFERRED BY DECISION | canonical quality.mode=off；<code>experiment.py::_quality_mode</code> 禁止未授权 route；diagnostics_only 只记录；V2-009a/b/c deferred。 | 当前 frailty reference 不会因未经验证的 SQI 阈值产生选择偏倚。 | DEFERRED：只在注册监督 artifact id/hash 后启用 route，不重新询问当前阈值。 |
| Q3 | A1 路由必须在 heterogeneous segment 粒度执行，而不是把整条 recording 当一个 segment。 | PARTIAL | <code>experiment.py::_route_records</code> 已调用公共状态机，但当前每条 recording 只构造一个覆盖全长的 <code>SegmentIntegrity</code>，没有多 segment 切分与重组。 | 同一文件内短暂 motion/failure 会被整条记录单一状态掩盖；coverage 和 route-by-segment 不能按 workflow 解释。 | P0 before A1 claim：建立 segment planner，逐 segment 路由、保留 start/end/run identity，再按合法 run 聚合 pulse/features。 |
| Q4 | <code>rate_only_direct</code> 不得产生 morphology、amplitude 或 dual-wavelength optical predictors。 | CONFLICT / P0 | <code>experiment.py:582-586</code> 将 <code>RATE_ONLY_DIRECT</code> 折叠为 <code>SignalRoute.DIRECT</code>；<code>experiment.py:688-689</code> 对 DIRECT 调用 morphology，未传递 Q_shape/RouteState。 | 一旦监督 route 启用，形状不合格 segment 可能泄入 morphology/optical 特征，直接违反 A2。 | P0：在 runtime record/feature facade 持久化 <code>RouteState</code> 或 <code>morphology_eligible</code>；只有 <code>full_direct</code> 可提 morphology/optical，并加 tiny end-to-end segment test。 |
| Q5 | reducer failure 和 PISD failure 都必须显式；不得静默回退另一算法。 | MATCHED | artifact router/status contracts；<code>quality/routing.py</code> 显式 <code>FAILED_REDUCER</code>；<code>models/pisd_port.py:719,851-854</code> 禁止 effect-size fallback。 | 模型身份不会因错误路径无声改变。 | 把失败原因汇入 A3/A4 输出；不新增 silent rescue。 |
| Q6 | motion override 激活必须等待内部监督证据和 PTT。 | MISSING / DEFERRED BY DECISION | <code>configs/v2_decision_profile.yaml:419-423</code>；motion reference/augmentation可构造，但 override 保持 gated。 | 避免用未验证 motion classifier 改写 frailty 样本流。 | DEFERRED：完成规定证据前不启用。 |
| Q7 | SQI、coverage、route/reducer identity、技术元数据默认不进入 predictor allowlist。 | MATCHED | <code>features/registry.py:261-308</code>，provenance 明确 <code>sqi_and_coverage_predictors_excluded</code>；experiment 只把它们写 metadata/OOF。 | 减少技术状态捷径和数据质量代理变量导致的偏倚。 | 若未来比较，必须单独命名 ablation。 |

## 7. Pulse、PRV、morphology、optical 与 engineering features

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| F1 | PulseResult 必须保存真实时间、PPI validity、原始 adjacency 和 detection run identity。 | MATCHED | <code>peaks/resolver.py::detect_pulses</code>、<code>::detect_pulses_per_wavelength</code>；<code>peaks/aboy_project.py</code>；<code>contracts.py::PulseResult</code>；<code>signal/prv.py</code> 使用 <code>detection_run_id</code>、ordinals 与 adjacency。 | 拒绝 interval 不会把 gap 两侧错误拼成相邻 PPI；RED/IR 保持独立 detector provenance。 | Q3 分段后确保不同 route runs 不跨 run 连接。 |
| F2 | PRV 需显式资格：time-domain 60 s/80% coverage；frequency-domain static+contiguous 300 s/200 intervals；SampEn 至少 200 intervals。 | MATCHED | <code>signal/prv.py:112-309</code>，包含相邻 pair、最长连续 run、Lomb–Scargle 与逐字段 validity/reasons。 | 缺数据时为 invalid/NaN，而不是补出貌似有效 PRV。 | 保持 local manual 为 formal backend；Aura/rhenan仅固定 PPI comparison。 |
| F3 | Morphology 和 dual-wavelength optical 只允许 amplitude-preserving、Q_shape 合格的 direct branch。 | PARTIAL | <code>signal/morphology.py</code>、<code>signal/optical.py</code> 独立实现；当前 experiment 只检查 DIRECT/IDENTITY，未区分 <code>rate_only_direct</code>。 | 模块算法存在，但正式 route 打开后会触发 Q4 风险。 | 与 Q4 同一补丁闭合，不单独添加启发式。 |
| F4 | Engineering 10 s/5 s 必须冻结为 115 列/window、230 个 file summaries；RED/IR/A/Omega/J 有 7 time+4 spectral+family bands，六轴仅 7 time；Welch 为 Hann、1600/800。 | MATCHED IMPLEMENTATION / NOT SCIENTIFICALLY RUN | <code>features/engineering.py::engineering_feature_names</code>、<code>::engineering_welch_parameters</code>、<code>::extract_engineering_features</code>；A/Omega/J 取 canonical processed outputs，raw/fusion tensor 仍为 8ch。 | Engineering descriptors 与 eight-channel raw tensor 是不同 representation，不再错误地从 engineering 中删去 derived scalar summaries，也不把它们加入 raw tensor。 | 正式运行继续校验 115/230 schema、registry hash 和旧 94-column stale rejection。 |
| F5 | FeatureVector 必须是显式 ordered allowlist；FeatureMatrix 为 D×32，含 validity channels、padding mask 和 schema hash。 | MATCHED | <code>features/registry.py::default_registry</code>、<code>::build_feature_vector</code>、<code>::build_ordered_matrix</code>；registry/matrix schema 已随 115-column contract 提升。 | 不同 fold/model 使用相同可审计特征顺序；缺失不被默认为生理零。 | 正式 bundle round-trip 继续校验 schema hash。 |
| F6 | Canonical peak detector 必须统一用于 internal/A3，版本、wavelength 和 polarity 持久化。 | MATCHED PROJECT IMPLEMENTATION / ENDPOINT NOT RUN | <code>peaks/aboy_project.py</code> 实现 400 Hz、10 s block、HRI-adaptive project Aboy++-inspired detector；<code>peaks/resolver.py</code> 将旧 detector 只注册为 <code>dual_polarity_prominence_v1_ablation</code> 且无 fallback。 | internal path 已有统一、可审计 detector；但尚无 A3 ECG endpoint accuracy 结果，不能声称 published upstream Aboy++ exact reproduction。 | 在 A3 以同一 detector 做 raw/direct/reducer 比较；论文称 project Aboy++-inspired detector。 |
| F7 | Aura 1.0.2 与兼容 nolds 只作为隔离比较。 | MATCHED | <code>requirements/requirements-prv-aura-compare.txt</code> 固定 hrv-analysis 1.0.2、nolds 0.6.2；<code>features/prv_backend_compare.py</code> 接收未经再清洗的相同 PPI。 | 不污染主环境或主 predictor，且保留函数级对照。 | 无需将 Aura 加入 canonical dependencies。 |

## 8. 四表征、模型与 ShapeFormer

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| R1 | 四种 representation：raw、feature_vector、feature_matrix、fusion，使用共同 manifest/folds/Trainer/Evaluator。 | MATCHED | <code>representations/</code>；<code>models/factory.py</code>；formal catalog 覆盖四类。 | 模型比较可在同一数据与评估层级进行。 | 正式 study 中每次只改变声明的 ablation factor。 |
| R2 | CompactCNN、Inception full/small、matrix Inception 的结构身份和参数量可复核。 | MATCHED AT CODE/SAFE-SUITE LEVEL | <code>models/compact_cnn.py</code>、<code>models/inception.py</code>、factory architecture materialization；<code>tests/models/test_architectures.py:45-53</code> 锁定 exact 8ch counts：Compact 79,139，Inception full 456,579，small 57,027；合并后 conda-ml safe suite 295/295 通过。 | 避免把 small/full/single/ensemble 混名。 | 实现合同已验证；仍不得把 safe suite 当作真实数据性能证据。 |
| R3 | ROCKET reference 为 fold-local 10,000 kernels + ridge；MiniROCKET 只能具名 ablation。 | MATCHED | <code>models/rocket.py::RocketTransformer</code> 默认 10,000；<code>models/rocket_ridge.py</code>；catalog 单独注册 MiniROCKET ablation；fit provenance 绑定 outer train。 | 不把 MiniROCKET 结果冒充 ROCKET reference，也避免 held-out fitting。 | 正式运行保存 transform/ridge serialization parity。 |
| R4 | Literature-reference ShapeFormer 为 channel-specific OSD，8ch generic branch，fold-local discovery，变量长度 candidate 和完整 shapelet provenance。 | MATCHED ALGORITHM; NOT EMPIRICALLY BENCHMARKED | <code>models/pisd_port.py</code> 的常量、<code>PisdShapelets</code>、actual-T PIP count、same-channel distance、3/class、180-window balance和显式失败；<code>models/shapeformer_literature.py</code>；catalog reference entry。 | 当前可以准确称 <code>channel_specific_osd</code> 实现；仍不能把未运行结果写成 literature parity 或性能结论。 | 保持已有 focused fidelity tests，并按正式 5×5 执行；保留每个 shapelet 的 channel/start/end seconds/length。 |
| R5 | 固定 128 sample/stride 64 effect-size 只能叫 <code>effect_size_fixed_v1</code>；400/800 可作为额外固定长度 ablation。 | MATCHED DELIBERATE ABLATION | <code>configs/formal_experiment_catalog_v2.yaml:355-400</code> 与 effect-size discovery 模块；不会作为 OSD fallback。 | 历史 fixed candidate 不再污染 literature-reference 结论。 | 若运行 400/800，单独 config id，不写 PISDPort。 |
| R6 | Fusion 必须先对 raw windows 做 mask-aware file pooling，再只拼接一次 file feature vector。 | MATCHED | <code>models/fusion.py::FileBagFusionClassifier</code>；factory 要求 signal channels 和 file-feature width；leakage tests 定义 file vector 不向每个 raw window广播。 | 避免同一 file feature 被窗口数重复加权。 | 正式 matched ablation 保持相同 folds/sampling。 |
| R7 | Fixed-sample kernel 的 sample-rate、10 s context、dilation 等只是具名 time-scale ablation。 | MATCHED CONFIGURATION; NOT RUN | <code>configs/v2_decision_profile.yaml:239-245,387-404</code>；只允许 CompactCNN/InceptionTime。 | 不把 fixed sample kernels 误称为 physical-time matched。 | 按用户要求保持未运行；需要时一因素执行。 |
| R8 | Motion reference 8ch；11ch derived augmentation 不得进入 frailty factory/catalog。 | MATCHED | <code>configs/motion_detector_contract_v2.yaml</code>；frailty catalog input_channels=8；<code>models/factory.py</code> 对 canonical raw schema fail closed。 | 防止 motion augmentation 改变 frailty thesis target。 | 继续做跨 catalog stale scan，禁止新增 frailty 11ch entry。 |

## 9. 训练、CV、种子与 ensemble

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| T1 | 所有 scaler、imputer、selector、ROCKET、shapelet、calibration、epoch rule 只用 outer-train；trainer 不接受 outer-OOF labels。 | MATCHED AT CONTRACT LEVEL | <code>training/trainer.py::FrozenOuterSplit</code>、<code>::UnifiedTrainer</code>、fit provenance；representation/ROCKET/ShapeFormer 各有 train-roster hash；合并后 conda-ml safe suite 295/295 通过。 | 阻断最主要的外层信息泄漏路径。 | 每个正式 cell 继续持久化 fitted participant roster/hash；后续修改保持 leakage tests 通过。 |
| T2 | Deep reference 固定 10 epochs；7/15 是具名 ablation；禁止用 outer fold early stopping。 | MATCHED | <code>training/trainer.py::DEEP_EPOCH_PROFILES</code>、<code>::TrainerConfig</code>；canonical config 242–248 行；outer early stopping disabled。 | repeat/fold 间 epoch rule 一致，不使用 held-out label。 | 若以后启用 inner grouped selection，必须单独 ADR/config；当前不需要。 |
| T3 | 单模型在 CV 中使用 repeat-specific seed；最终单模型全数据 refit 才 seed=42。 | MATCHED | <code>data/folds.py::outer_cv_single_model_training_seed</code>；catalog seed policy；<code>configs/v2_decision_profile.yaml:323-330</code>。 | 解决“final refit seed 42 被误应用到 25 cells”的表述/实现风险。 | 在结果表同时列 split_seed 与 training_seed，避免再混淆。 |
| T4 | 五成员 ensemble 固定成员 seeds 50042–90042，member0 comparator 50042；每 fold 平均概率，拼 repeat OOF，每 repeat 计算一次指标；最终 refit 新训五个 all-29 models。 | MATCHED IMPLEMENTATION / VERIFIED CONTRACT; FORMAL SCIENCE NOT RUN | 接线位于 <code>models/inception.py::CANONICAL_ENSEMBLE_MEMBER_SEEDS</code>、<code>experiment.py</code>、formal catalog 和 decision profile；ensemble focused tests 87/87 与合并后 conda-ml safe suite 295/295 通过。<br><strong>ENSEMBLE_VERIFICATION_STATUS: VERIFIED</strong> | seed policy、member independence、exact averaging、OOF roster、final-refit roster 和 serialization 合同已有实现级验证；仍无正式 5×5 ensemble 性能证据。 | 算法合同无需继续修补；仅在人工触发 formal catalogue 时运行完整 repeated 5×5，并保留 member-level 与 averaged OOF。 |
| T5 | sampling/loss/class weighting 必须显式、outer-train-only，不得 silent stacking。 | MATCHED CONFIGURATION; EMPIRICAL CHOICE NOT SELECTED | canonical 明确 <code>balance_line_weighted_v2</code> sampler + outer-train inverse-frequency CE；<code>training/trainer.py:650-806</code> 计算权重并在 provenance 记录。 | 方法不是 silent，但 sampler 与 class-weighted loss 的组合效应尚无正式 one-factor 证据。 | P1：在 thesis 方法中预注册该组合，或以一因素 ablation 证明；不得事后按 outer score换方案。 |
| T6 | architecture、channel order、fs、window/hop、normalization、mask、feature hash、SQI、loss、weighting、sampler、epoch、optimizer、LR、WD、dropout、label smoothing、clip、seeds、fold hash、aggregation、calibration 都需冻结。 | MATCHED AT SCHEMA/CONFIG LEVEL | <code>configs/v2_decision_profile.yaml:298-322</code>；canonical configs；training provenance、model factory declarations和bundle schema。 | 运行身份可以重建并比较。 | 正式 run 前执行 resolved-config audit，并把完整 config hash 写入每个 cell/root artifact。 |
| T7 | 正式模型比较为完整 repeated 5×5，所有候选使用共同 folds，并生成完整 OOF。 | MISSING / EXPLICITLY NOT RUN | formal catalog 与 study plan 已注册，但 decision/status 明确本轮未执行完整 5×5、正式 ablation 或 PTT。 | 当前只能评价实现完整性，不能选择 winner 或形成论文性能表。 | 按用户人工命令执行；运行前通过 resolved-config audit；只有启用 A1/A2 route 时才需先闭合 Q3/Q4。 |
| T8 | 只有人工选定 final use case 后，才可全 29 人从头 refit；OOF 是性能证据，full refit 不产生内部泛化分数。 | MATCHED POLICY; NOT RUN | <code>configs/v2_decision_profile.yaml:323-335</code>；final selection/refit API 和 bundle模块。 | 避免在 use case 未选定前把 refit 配置扩散到 CV，或把训练集分数称 test。 | 等人工 winner selection；不要提前运行 final refit。 |

## 10. 聚合、OOF、指标、统计、报告与 bundle

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| AG1 | Canonical 聚合必须 window→file→role→participant；同 file 窗口均值、同 role 文件均值、available roles 等权。 | MATCHED | <code>training/aggregation.py::CANONICAL_AGGREGATION_LINE</code>、<code>::aggregate_hierarchy</code>、<code>::hierarchy_oof_rows</code>；role normalization。 | 长文件或窗口多的角色不会支配 participant probability。 | 保持 role-level OOF 和 coverage audit。 |
| AG2 | Equal-files Line A 只能是具名 ablation，不能再称 default/reference。 | MATCHED IN ACTIVE CONFIGS | <code>configs/reference_static_role_aware_v2.yaml</code> 为 canonical；Line A 有独立 ablation config；decision profile 52–56 行禁止称 default。 | 可检验角色等权的影响而不混淆主分析。 | 保持 Line A 与 Line B 的 config ID、hash 和报告标签分离。 |
| AG3 | OOF 必须有 window、file、role、subject 和 ensemble member 层级，且 Parquet fail closed。 | MATCHED AT WRITER/SAFE-SUITE LEVEL | <code>training/oof.py::OofPredictionRow</code>、role/member validation、<code>::write_oof_parquet</code>；experiment 写四层及 member rows；T4 的 roster/averaging 合同已通过 focused + safe suite。 | 可重建每层聚合，检查每个 held-out participant 和成员身份。 | 正式运行时仍需核对每 repeat 完整 29 participant coverage；实现验证不替代科学 OOF 产物。 |
| AG4 | 参与者级 BA、macro-F1、per-class recall/F1、worst-class、confusion、Brier/ECE、coverage；repeat summaries 与 paired statistics。 | MATCHED AT CODE LEVEL | <code>training/evaluator.py</code>；<code>training/statistics.py</code> 实现 10,000 participant-cluster bootstrap、100,000 paired permutation、Holm、BA 与 macro-F1 LCB95。 | 满足 V2-024e2 同时保留 BA 与 macro-F1 下界；避免窗口级伪样本量。 | 本次未计算科学指标；正式 report 中同时显示 point estimate、repeat SD/CI、两项 LCB。 |
| AG5 | 每次实验需输出配置、非变量、flow 位置、BA/F1、参数、learning curves、confusion、rankings、plots 和总结。 | MATCHED REPORTING SKELETON; NO FORMAL OUTPUT | <code>reporting/</code>、<code>study/</code>、pipeline/sweep entry scripts提供 summary、plots、ranking与独立目录逻辑。 | 工具存在不代表论文图表已生成。 | 正式运行后检查每个 run folder 的 machine-readable + human-readable completeness。 |
| AG6 | Canonical Line B 身份在所有有效配置中必须无矛盾。 | MATCHED IDENTITY | <code>configs/v2_decision_profile.yaml</code> 的 confirmed default 为 Line B；comparison profile 已改为 <code>line_a_equal_files_balance</code> 且 <code>formal_default=false</code>。 | Canonical 与 ablation 身份不再互相矛盾。 | 保持 resolved-config assertion：canonical aggregation 必须 role-aware。 |
| AG7 | CPU CI 应执行 lint/import/unit/synthetic integration。 | MISSING | 当前 repository 和 V2 下均无 <code>.github/workflows</code>。测试文件存在，但没有自动 CPU CI 证据。 | 修改后回归依赖人工记忆，无法声称 Definition of Done 的 CPU CI green。 | P1：建立轻量 CPU workflow；GPU/full-data 保持手工或 scheduled。 |
| AG8 | Bundle 必须保存预处理、feature schema、fold-local transforms、模型/ensemble members、calibration和 golden inference parity。 | MATCHED IMPLEMENTATION; FINAL WINNER ARTIFACT MISSING | <code>training/bundle.py</code>、<code>bundle/schema.py</code>、<code>bundle/save.py</code>、<code>bundle/load.py</code> 和 round-trip tests。 | 结构上可复现；尚无人工 winner 的真实 final bundle。 | T7/T8 后对实际 winner 执行 golden-record parity；硬件/ONNX不作为当前默认要求。 |
| AG9 | Workflow/spec 中逐项 acceptance 要求应有可追溯 test mapping。 | PARTIAL / MAPPING MISSING | 当前 safe suite 覆盖大量合同，且 35–210 bpm 边界已有明确测试，但仍没有一份 workflow 条目→测试名的完整矩阵；filter impulse/passband 等 acceptance mapping 尚未逐项闭合。 | 295/295 只能说明已纳入 safe suite 的测试通过，不能解释为两份要求文档的每条 acceptance 都已有自动测试。 | P1：建立 acceptance matrix；对缺项补 tiny deterministic tests，对已有但名称不直观的测试补精确链接。 |

## 11. 外部 PTT、A3/A4 与明确搁置项

| ID | 要求 | 状态 | 当前 V2 证据 | 科学后果 | 建议闭合 |
|---|---|---|---|---|---|
| E1 | A3 要在公开 500 Hz grid 锁定 selected PPG/IMU/ECG reference 并使用同一 pulse/routing/reducer endpoint。 | CONFLICT / PARTIAL | 外部 manifest 正确记录 source 500 Hz，但 <code>data/external_manifest.py:178,281,323-348,471-545</code> 把 PPG/ECG/IMU 同步 resample 为 400 Hz；workflow 11.1 明确要求 500 Hz sampling grid。 | 在未裁决前，A3 的 timing tolerance、peak sample identity 和与公开数据说明的可比性不唯一。 | P0 before A3：人工明确“原生 500 endpoint”或“同步 400 adapter”之一；若保留 400，需 ADR/论文明确 deviation，并报告源/目标时间误差。不得静默混用。 |
| E2 | ECG reference 需 one-to-one event matching，并报告 HR/IBI/PPI error、precision/recall/F1、timing error、coverage/failure，按 sit/walk/run。 | PARTIAL COMPONENTS | <code>peaks/pairing.py::match_events</code> 提供 one-to-one matcher；external manifest 有 ECG/activity；但没有完整 endpoint evaluator/规定输出。 | 可以复用组件，但当前不能声称 reducer 或 detector 已通过 ECG endpoint validation。 | 合入 E3 runner，增加 raw/direct/reducer 同记录 paired outputs。 |
| E3 | 独立 A3 command/config 必须生成 external_event_matches.parquet、external_rate_predictions.parquet、external_metrics_by_activity.json、external_failure_report.json、external_run_manifest.json。 | MISSING / EXPLICITLY NOT RUN | 当前 CLI 有 <code>motion-train-internal</code> 和 <code>motion-evaluate-ptt</code>，它们评估 activity detector；无 workflow 规定的 <code>benchmark-motion</code> endpoint runner和五项产物。 | Motion activity evidence 不能替代 heartbeat/rate endpoint evidence；V2-010 activation仍无外部依据。 | P0 for A3 claim：实现独立 endpoint runner，复用现有 signal/routing/reducer/pulse modules；不运行由用户决定。 |
| E4 | A4 internal dynamic command 必须生成 segment_routes、q_rate_pre_post、dynamic coverage、reducer failures、rate agreement 和诊断图。 | MISSING / EXPLICITLY NOT RUN | 无 <code>analyze-dynamic</code> command，也无完整规定 artifact set；现有 SQI diagnostics不是 A4 report。 | 无法量化内部 W/S 的 route diversity、pre/post quality 和 reducer coverage。 | P1 for A4 claim：Q3/Q4 后实现 runner；只声明 downstream quality evidence，不声明 clean-waveform accuracy。 |
| E5 | V2-006、009a/b/c、010、012、026、027 保持明确搁置。 | MATCHED DEFERRED POLICY | <code>configs/v2_decision_profile.yaml:406-432</code> 和 <code>docs/V2_DEFERRED_POINTS.md</code>；包括设备 QC、SQI route、motion override、最终 reducer winner、硬件/功耗/延迟、todo-only scope。 | 防止缺证据功能被默认启用。 | DEFERRED：不重新询问；只有出现规定证据或用户新裁决才变更。 |
| E6 | 正式 ablation、完整 5×5、PTT benchmark 不得伪装成已运行。 | MATCHED STATUS DISCIPLINE | 当前 decision/config/report文字把相关执行状态标为 not run/deferred；model cards声明内部结果只能叫 OOF validation。 | 保护 thesis claim 边界。 | 保持所有未来报告带 run manifest、数据/fold/config hashes。 |

## 12. 优先闭合顺序

### P0：在对应正式运行或 claim 前

1. 若要启用或评估 A1/A2 route，先完成 Q3/Q4：真正 segment-level 接线，并让 <code>rate_only_direct</code> 无法进入 morphology/optical。
2. A3 运行前先裁决 500→400 grid conflict，再实现规定 endpoint runner 和五项输出。

### P1：完整论文复现包前

1. 建立 CPU CI，并把当前已通过的 conda-ml safe suite（295/295）接入自动执行；本地通过不等于 CPU CI 已存在。
2. 在 Q3/Q4 后实现 A4 internal dynamic report。
3. 保持 ShapeFormer reference 的 focused fidelity tests，并运行与其他模型相同 protocol；正式 5×5 未运行前只称 algorithm implementation。
4. 对 sampler + class-weighted CE 保持预注册，或做严格一因素 ablation，不依据 outer results 临时选择。

### 由人工命令触发、当前不自动执行

1. 完整 5 repeats × 5 folds formal catalogue；
2. 正式 one-factor ablations 和 time-scale alternatives；
3. external PTT A3 benchmark；
4. 人工选择 final use case 后的 all-29 refit 和 final bundle。

### 继续搁置，不重新询问

V2-006、V2-009a/b/c、V2-010、V2-012、V2-026、V2-027；以及缺少规定证据时的 supervised SQI route、motion override、设备特定 ADC QC、最终 reducer winner和部署硬件门槛。

## 13. 本审查不作出的结论

- 不声称任何模型已在真实 29 人数据上达到某个 BA 或 macro-F1。
- 不声称 ShapeFormer、ROCKET 或 ensemble 已完成正式 5×5 性能验证。
- 不声称 PTT activity detector 输出等价于 ECG heartbeat endpoint benchmark。
- 不声称 artifact reducer 恢复了 clean waveform 或 morphology。
- 不声称当前 bundle 是人工选定 winner 的最终部署 bundle。
- 不把代码中存在的测试定义自动写成通过；本文只记录实际执行的 ensemble focused 87/87 与合并后 conda-ml safe suite 295/295，且不将其解释为科学性能证据。

## 14. 剩余不确定性

1. A3 的 500 Hz 原生 endpoint 与当前同步 400 Hz adapter 谁是最终 thesis reference 尚无后续人工裁决；本文将其列为真实 conflict，而不是自行选择。
2. SQI threshold、route supervision、motion override、reducer winner、设备 rail 和目标硬件均被明确 deferred；没有足够证据时不能把它们从“缺口”改为“默认算法”。
3. 合并后的 conda-ml safe suite 已实际执行并以 295/295 通过，但本次仍未运行任何正式科学任务；MATCHED/VERIFIED CONTRACT 不等于真实数据性能验证，正式 run 前仍需 resolved-config audit。
