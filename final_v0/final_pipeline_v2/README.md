# PPG Frailty 3-Class Pipeline V2

V2 是当前可读、可配置、可由人工直接运行的 thesis pipeline。旧的复杂门禁版本已由
`final_pipeline_v3` 保留为历史快照；V2 的普通实验入口不要求 agent 监督，也不依赖
source/lock/attestation gate。

## Pipeline 边界

- 任务：29 名 participant 的三分类，machine labels `0/1/2`，报告显示
  `Pre-Frail`、`Robust/Non-Frail`、`Young`。
- 输入：B 与 R 文件；R1–R4 是同一 relax role 的独立文件，文件间没有连续时间轴。
- 验证：participant-grouped 5-fold × 5-repeat。每个 repeat 的五个 folds 共用该
  repeat seed：`[42, 10042, 20042, 30042, 40042]`。固定 `seed=42` 只属于人工
  选出 final use case 后的 all-29 refit。
- reference 单模型 outer CV 使用各 repeat seed；具名 matched Inception comparator
  默认 member 0 seed `50042`。具名 five-member comparison 默认复用
  `[50042, 60042, 70042, 80042, 90042]`。普通 V2 ensemble 则由唯一、显式的
  `member_seeds` 决定任意正成员数，逐 fold 对成员概率取算术均值；final refit 使用
  被选配置中的同一 roster。
- reference aggregation 为
  `window → file → role → participant`（role-aware Line B）。Line A equal-files 与
  Line B 都是普通可选聚合模块，并与训练采样的 balance line 独立配置；报告可从同一
  held-out file OOF 并行重放 window/Line A/Line B 三种 participant 视图。
- reference quality mode 为 `off`。`diagnostics_only` 只记录诊断；`route` 直接选择
  可执行 SQI/calibration/route/recovery 状态机，不再依赖 readiness 或 Phase-0/C0 门禁。
- direct PPG view 使用 0.2–8 Hz；0.5–5 Hz 是 ablation。IMU reference 为同一
  participant role-B 校准的 roll–pitch EKF，无静默 fallback；LPF 0.3 Hz 是 ablation。
- frailty raw/fusion/ShapeFormer 输入固定为 8 通道：
  `RED, IR, A_dyn_X, A_dyn_Y, A_dyn_Z, GX, GY, GZ`。DL tensor 在每个 5 s
  window 内对八通道分别做 median/(IQR/1.349)、population-SD fallback、clip
  `[-8,8]`。这是独立副本；SQI、motion、denoiser 和 engineering 继续读取保留
  m/s²、rad/s、m/s³ 的 `processed_imu_physical`，peaks/PRV/morphology/optical
  继续读取幅值保真的 `x_analysis`/`x_native`。
- motion-model reference 同样使用上述 8 通道；加入
  `A_mag, Omega_mag, J_mag` 的 11 通道输入只属于具名 augmentation ablation，
  不进入 frailty raw-signal branch。
- 主结果是 OOF validation，不是 independent test；不会自动挑 winner。

上面都是 reference preset，不是 ordinary V2 的算法许可表。有效配置可独立选择
Adam/AdamW/SGD/RMSprop、replacement/exhaustive/subject/class-subject samplers、
participant/window/effective-number/no class weighting、cross-entropy/balanced-softmax/
focal loss、fixed/inner-grouped epoch 策略、Line A/Line B、quality 与 normalization
模块，以及正数 window length/hop、alignment、padding、绝对或比例 cap。默认值会在
hash 前完整物化；配置接受的每个算法字段必须有真实 runtime consumer，不适用于当前
model backend 的字段会明确报错。任务数据边界（29 participants、三分类、冻结 5×5
OOF、raw/fusion 的有序 8 通道与 400 Hz 内部信号网格）仍保持不变。

`features.enabled_groups` 是旧 `extra_input`/`manual_features` 的统一替代入口，可组合
选择 basic PPI/rate、HRV time-domain、HRV spectral、HRV nonlinear、morphology、
dual optical 与 engineering summary。默认全量 file vector 是 282 fields；feature
matrix 则是独立固定合同：10 s window、2 s hop、每窗 115 个 engineering predictors、
K=150，即 `OrderedFeatureMatrixV1[115,150]`。逐特征 validity 仅写入 provenance，
不再扩成 predictor channels；时间 padding 由 row mask 隔离。registry、vector、
fusion tensor 与 matrix 的 schema/count/hash 都从真实消费者派生，不能手工伪造。
迁移时，旧 `extra_input=PPI` 对应 `ppi_basic_rate`，旧 `HRV` 对应四个 PPI/HRV
groups 的并集；旧 `manual_features=morphology` 对应 `morphology + dual_optical`，
旧 `morphology_ppi_hrv_filelevel` 对应上述六组并集。旧实现中的 coverage/technical
columns 仍不作为 predictor；`engineering_summary` 是独立可选组，不被偷偷并入旧别名。

可选模型 `FileBagFusion` 通过嵌套 `model.signal_encoder` 组合 Compact、full/small
Inception、faithful channel-specific OSD ShapeFormer、其 scalar-distance ablation、
新版 effect-size ShapeFormer 或隔离的旧版 effect-size port；`features.enabled_groups`
决定与之拼接的 file vector。ShapeFormer
发现只展开已验证的 outer-train file bags，file features 不参与发现，随后每个文件仅在
window pooling 后拼接一次。原有 `FileBagFusionCompact`/`FileBagFusionInception` 名称
继续作为兼容路线。

旧版对照模块 `ShapeFormerLegacyEffectSizePort` 保留历史 channel-wise effect-map
发现和 `PortedShapeFormer` 中实际参与 forward 的 local/shape-token 两支。其默认参数
对应旧 `RunConfig` 的 3 个 shapelets/class、128 长度、64 stride、180 discovery
windows、8 candidates/class/channel、48/128 embedding、256 FFN、4 heads 和 0.30
dropout；这些参数均为可输入且进入运行架构哈希。旧 `processes`/`verbose` 只影响
执行调度或控制台输出，不作为算法参数；旧 `len_w` 的未使用 bookkeeping 也不作为
假开关，实际 local convolution width（默认 8）与 shapelet search span（默认 64）
分别显式配置。

四个 canonical 配置：

- `configs/reference_static_role_aware_v2.yaml` — raw / CompactCNN reference。
- `configs/reference_static_feature_vector_v2.yaml` — engineered feature vector。
- `configs/reference_static_feature_matrix_v2.yaml` — 115×150 ordered feature-matrix
  开发/合同 smoke harness；当前正式 matrix 模型待定。
- `configs/reference_static_fusion_v2.yaml` — raw + feature fusion。

## 安装

在 `final_pipeline_v2` 目录中：

```bash
python -m pip install -e '.[reporting,dashboard,formal-benchmark]'
# 运行深度模型时再安装：
python -m pip install -e '.[deep,reporting,dashboard,formal-benchmark]'
```

Aura PRV 对照固定使用 `hrv-analysis==1.0.2`，由于它需要旧版 NumPy/Astropy，必须
在独立环境中安装 `requirements/requirements-prv-aura-compare.txt`，不要覆盖主
pipeline 环境。

## 一条命令运行

完整单配置（默认全部 25 cells，并自动生成报告）：

```bash
python frailty_3class_pipeline_v2.py \
  --config configs/reference_static_role_aware_v2.yaml \
  --study-id compactcnn_role_aware_reference
```

单因素 ablation：

```bash
python frailty_3class_sweep_v2.py ablation \
  --base-config configs/reference_static_role_aware_v2.yaml \
  --factor training.fixed_epochs --values 7 10 15 --reference-value 10 \
  --study-id compactcnn_fixed_epochs \
  --purpose 'Compare only the fixed-epoch factor.' \
  --flow-position 'Training-capacity ablation before manual candidate review.'
```

Cartesian grid（CPU/classical case 可用 `--jobs 4` 并行；deep 默认保持 1）：

```bash
python frailty_3class_sweep_v2.py grid \
  --base-config configs/reference_static_feature_vector_v2.yaml \
  --vary 'model.logistic_max_iter=[1000,2000]' \
  --vary 'training.class_weighting=[outer_train_inverse_frequency,none]' \
  --reference model.logistic_max_iter=2000 \
  --reference training.class_weighting=outer_train_inverse_frequency \
  --study-id logistic_screen --purpose 'Descriptive screening.' \
  --flow-position 'Candidate screening before confirmation.' --jobs 4
```

先检查所有 resolved cases 而不执行：在上述命令末尾加 `--dry-run`。恢复失败或中断
的 study 使用 `--resume <existing-study-directory>`；已通过 case 跳过，重试会写入新
的 `attempt_NNN`，不会覆盖旧结果。terminal 只显示一条刷新式进度条，完整事件写入
JSONL。

## 输出与报告

每次运行创建独立目录，目录名含日期、study kind 与 ablation/grid 对象。目录包含：

- study plan、resolved configs、每个 case/attempt、cell summaries 与 OOF Parquet；
- BA、macro-F1、per-class、worst-class、coverage、calibration、参数量与可用的运行成本；
- repeat/fold stability、双侧 95% CI、learning curves、counts/normalized confusion
  matrices、parameter effect/interaction、route/role quality summaries；
- CSV/JSON/Markdown/HTML、论文用 PNG、缺失视图的明确 N/A 文件；
- `outputs_index.json`，列出 study 目录内全部普通文件、bytes 与 SHA-256。

重新生成报告不会训练：

```bash
python frailty_3class_sweep_v2.py report --study-dir <study-directory>
```

## 计划与差距文档

- [V2 搁置与未完成事项](docs/V2_DEFERRED_POINTS.md)
- [V2 算法结构差距分析](docs/V2_ALGORITHM_GAP_ANALYSIS.md)
- [V2 消融与测试详细计划](docs/V2_ABLATION_TEST_PLAN.md)

三份文档区分软件实现、科学运行、人工决定与明确搁置；计划条目不代表已经执行。

## Dash 人工检查

```bash
python frailty_pipeline_dashboard_v2.py --host 127.0.0.1 --port 8050
```

Dash 可选择 participant/record/config/segment，预览原始与处理后 signal、QC/route、
pulse/PPI、morphology、engineering 等阶段表与图；可开关并行预览线、下载当前 CSV/
metadata、启动或停止独立 study subprocess，并浏览/下载已完成 study 的表格、图和 ZIP。
Dash 只调用 canonical modules，不复制算法。

更完整的命令说明见 [RUNBOOK.md](RUNBOOK.md)。
