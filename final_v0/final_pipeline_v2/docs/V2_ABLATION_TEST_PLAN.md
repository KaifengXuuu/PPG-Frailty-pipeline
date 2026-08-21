# V2 消融与测试计划

状态：实验执行计划，不是结果记录
依据当前 V2 源码复核日期：2026-08-20
范围：仅 `final_v0/final_pipeline_v2`

## 1. 目的、依据与边界

本文件规定用于筛选和论证 thesis-grade V2 pipeline 的有序实验。它与结果报告严格分离：创建或阅读本文件不会自动启动训练，不表示任何实验已经运行，也不会把仅存在于 catalogue 的条目自动变成完整 study。普通 V2 不设证据/人审授权门禁；依赖顺序只约束科学解释和晋级结论。

本计划采用以下优先级：

1. 用户最新人工确认，包括 frailty/motion 最终通道合同和 split/member seed 合同；
2. `AA_TODO/workflow/CODEX_CANONICAL_PIPELINE_WORKFLOW_V1.md`；
3. `AA_TODO/3/CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md`；
4. 当前 V2 configs、catalog、study runner、reporting、model cards 和 tests，仅用于判断实现状态。

外围历史材料若仍含旧 seed roster 或更宽 tensor，以本文件所列最新人工确认值为准；当前 seed 合同已经在代码、配置、provenance 和 tests 中同步并通过安全测试。V3 是旧门禁状态的历史备份，不是当前执行目标。

当前 `materialization_index_v2.json` 明确记录 `training_executed=false`、`ablation_executed=false`、`ptt_benchmark_executed=false`，所以本计划中的科学结果全部仍待运行。

### 状态定义

| 状态码 | 含义 |
|---|---|
| `RUNNABLE_PATH` | 已有持久化 V2 pipeline config 和 study plan，或已有合法单轴 CLI；除非存在完整 dated study folder，否则仍不代表做过科学运行。 |
| `IMPLEMENTED_NEEDS_PLAN` | 算法/模式已实现，但仍缺完整 resolved config、study plan 或 dry-run 验证。 |
| `CATALOG_ONLY` | 仅登记供物化；catalog 明确为 `auto_run=false`、`materialization_only=true`。 |
| `PLANNED_NOT_RUNNABLE` | 科学问题有效，但精确可执行合同尚不完整。 |
| `DEFERRED_EVIDENCE` | 软件可执行，但科学证据、外部条件或人工研究决定尚未完成；不得伪造结论。 |
| `PROHIBITED` | 会违反冻结科学合同的路线。 |

`RUNNABLE_PATH` 只说明 study 层没有已知缺项，不等于数值正确、运行成本可接受或 5×5 已成功完成。

## 2. 所有 frailty study 的冻结协议

下列 U1–U7 是通用控制项。后文每个 group 只能改变其明确声明的因素，其余全部冻结。

### U1 — 数据、标签与 folds

- 内部 manifest：`manifests/internal_records_v2.csv`，261 条记录、29 名 participant。
- 类别顺序：`0=Pre-Frail`、`1=Robust/Non-Frail`、`2=Young`。
- 分组单位为 participant；任何 outer cell 内同一 participant 不得跨 train/test。
- 冻结 outer CV：`splits/sgkf5_repeated_grouped_5x5_v2.csv` 中 5 repeats × 5 `StratifiedGroupKFold`；禁止运行时重算划分。
- `repeat_split_seeds = [42,10042,20042,30042,40042]`。
- 普通非 ensemble、非 matched-comparator 的单网络，在一个 repeat 的五个 folds 中共同使用该 repeat 的 split seed 作为训练 seed。把全部 25 cells 固定成 seed 42 是错误的。
- 每个正式科学 case 必须有 25 outer cells；部分 repeat/fold 仅可作诊断，不能排名或晋级。

### U2 — ensemble/member seeds

- `ensemble_member_seeds = [50042,60042,70042,80042,90042]`。
- 该 roster 在每个 repeat/fold 中完全相同；split seed 不选择、不轮换 member。
- matched 单网络 comparator 必须是相同 architecture/route 的 `member_index=0`，所有 repeat/fold 固定 seed 50042。
- 每个 outer fold 内，对 members 0..4 的 probability 做精确 arithmetic mean。
- 只拼接各 held-out fold 的 ensemble mean，形成每个 repeat 一套完整 participant OOF；每个 repeat 只评分一次，再汇总五个 repeat。
- 禁止 member selection、把 member metric 平均当 ensemble metric、跨 repeats 汇成 25-member pool，或把 outer-fold models 当 deployment roster。
- 最终 full-data ensemble refit 使用相同 roster 训练五个全新的 all-29 models；若另行选择普通最终单模型，则 refit seed=42。

### U3 — signal 与 IMU 合同

- 原始物理输入顺序 `[RED,IR,AX,AY,AZ,GX,GY,GZ]`，内部采样率 400 Hz。
- direct PPG reference：3 阶 zero-phase Butterworth，0.2–8.0 Hz，无 notch、无第二次 filter。
- 内部 acceleration：`g * 9.81 -> m/s^2`；gyro：`degree/s -> rad/s`。
- 外部 SIT/PTT acceleration 已是 `m/s^2`，必须 identity conversion；禁止再乘 9.80665。
- reference gravity separation：same-participant calibrated roll–pitch EKF。必须保存单位转换、calibration identity、process covariance `[5.0,5.0,0.05,0.05,0.05]`/s、observation covariance `[0.5,0.5] rad^2`、initial covariance `[1.0,1.0,0.5,0.5,0.5]` 和单侧 dynamic observation scale `1+3*max(0,||a||-g)/g`；无 magnetometer 时只用 roll/pitch，不声称 yaw correction。
- 计算 `A_dyn_x/y/z`、`A_mag`、`Omega_mag`、`J_mag`；IMU robust center/scale 只能在 outer-training participants 上拟合。禁止按每个 window 把 IMU amplitude 归一到相同强度。
- EKF 失败必须显式 fail-closed，绝不能静默改走 low-pass profile。

### U4 — 最终 frailty 与 motion tensors

- 最终 raw/fusion/ShapeFormer frailty tensor 必须恰好为八通道：`[RED,IR,A_dyn_x,A_dyn_y,A_dyn_z,GX,GY,GZ]`。
- 用户/论文展示层可写 `A_dyn_X/Y/Z`；代码 schema 的 lowercase `A_dyn_x/y/z` 是同三轴的固定别名，
  数值、顺序和物理含义完全相同。本轮不为大小写改动 schema/hash。
- `A_mag`、`Omega_mag`、`J_mag` 可计算、缩放并用于 diagnostics/motion，但禁止进入最终 frailty predictor tensor。
- motion reference 使用同一 8-channel axes tensor。
- 唯一命名的 motion augmentation 是 11-channel tensor：`[RED,IR,A_dyn_x,A_dyn_y,A_dyn_z,GX,GY,GZ,A_mag,Omega_mag,J_mag]`；它不具备 frailty-predictor 资格，也不得成为 fallback。

### U5 — windows、normalization 与 route

- raw DL window/hop：5.0/2.5 s，400 Hz 下 2000 samples，仅完整 windows，每 file 上限 128。
- engineering window/hop：10.0/2.0 s，仅完整 windows；ordered matrix 固定 `115×150`，带显式时间位置 mask。
- raw/fusion/ShapeFormer 的 8 个 DL channels 均逐窗口 robust normalization；SQI、motion、reducer 与 engineering 继续读取物理单位 processed IMU 并行视图。
- reference 为 quality `off`、artifact reducer `identity`、motion override disabled。
- raw/fusion 使用 direct `x_filter` 加 processed IMU；非 identity reducer 只返回 aligned `x_ar` rate-only route，禁止在 `x_ar` 上声称 morphology 或 waveform shape。
- PISD/OSD、reducer、route、calibration、dependency 或 parquet 失败均不得静默切换算法。

### U6 — reference training 与 aggregation

- sentinel：`configs/reference_static_role_aware_v2.yaml`，raw `CompactCNN1D`，8 channels，kernels `[9,9,7]`，dilations `[1,1,1]`，pools `[4,4]`，dropout 0.2。
- reference training：fixed 10 epochs、batch 64、Adam、LR 0.001、weight decay 0.0001、outer-train inverse-frequency class weights、`balance_line_weighted_v2` sampler、cross entropy、label smoothing 0、无 gradient clipping、deterministic algorithms。
- reference balance line B：`equal_role_families`；probability hierarchy 为 `window -> file -> role -> participant`，每层 ordinary mean，participant 对 available roles 等权。
- quality weighting 和 direct-all-window participant mean 均关闭。

### U7 — evaluation 与 selection

- 评估单位为 participant OOF，primary metric 为 balanced accuracy (BA)。
- 必须同时保留 macro-F1、macro-F1 CI95 下界、per-class precision/recall/F1、worst-class recall/F1、confusion counts、row-normalized confusion、calibration 和 coverage。
- 报告五个 repeat scores；不得把相关的 25 fold scores 当 25 个独立最终估计。
- paired comparison 使用相同 participant/fold/repeat；统计遵守冻结的 participant-cluster bootstrap、paired permutation 和 Holm contracts。
- incomplete cases 不得排名；禁止自动 final selection。每个 comparison group 最多将 10 个完整 configs 送入人工 shortlist。

## 3. 运行方式、并行与强制输出

命令从 `final_v0/final_pipeline_v2` 运行。常规一条命令接口为：

```text
python frailty_3class_sweep_v2.py run --plan configs/studies/<plan>.yaml --repeats all --folds all --jobs <N>
```

对于 canonical config schema 已接受的真正单一 scalar factor：

```text
python frailty_3class_sweep_v2.py ablation --base-config <pipeline.yaml> --factor <dotted.path> --values <v0> <v1> ... --reference-value <vref> --study-id <id> --purpose <text> --flow-position <text> --repeats all --folds all --jobs <N>
```

命令会在 output root 下建立 `YYYYMMDD_HHMMSS_<kind>_<ablation-object>` 并默认生成 report。`--resume <exact-study-dir>` 跳过已通过 cases，并为 failed/incomplete case 创建新的 attempt directory；`report --study-dir <exact-study-dir>` 仅重建报告，不训练。

study 层只允许 case-level parallelism。建议：

- deep、fusion、ShapeFormer、motion、ensemble：`jobs=1`；只有实际 memory test 证明安全后才考虑 `allow_parallel_deep=true`；
- 独立 classical CPU cases：`jobs=2–4`，受内存限制，estimator/BLAS 内部 threads 固定为 1；
- 禁止在 study 层再建 fold/repeat/member nested pool。

正式 `run`、`ablation` 和报告生成必须使用原位刷新的 terminal 进度显示，不得按每个 event 逐行刷屏；细粒度状态同时追加到 progress JSONL，保证无人监督运行后仍可复核。

每次正式运行必须产生完整 report bundle `R0`：

- `study_plan.yaml`、每 case resolved config、resolved-cases 表、完整 varied-parameter 表和完整 controlled/non-variable 表，并在 summary 中列明当次参数、ablation 对象、固定非变量、`purpose`、`flow_position` 与 thesis section；
- case/attempt status，planned/reported/passed/failed/not-run case/cell counts，progress JSONL、timestamps、hashes 和 output index；
- window/file/role/participant OOF，以及 ensemble 的 member OOF；保存 model/config/manifest/fold/preprocessing/feature hashes 和 route/quality/retention 字段。role artifact 必须是 `oof_role_predictions.parquet`；reporter 能读并不证明 producer 已写出；
- participant-level BA、macro-F1、per-class、worst-class，repeat/fold tables，以及 mean、SD、CI95 low/high、margin、min、max；
- confusion counts、row-normalized confusion、top-case confusion CSV、calibration bins、paired deltas、coverage、route×role、quality distributions、worst-class-F1 stability、deployment measurements 和明确的 incomplete-case exclusions；
- leaderboard、stability、worst-class-F1 stability、fold heatmap、paired deltas、coverage、route/role、quality、calibration、两种 confusion、per-class、全部与 top-only learning curves、parameter effects/interaction 的 PNG 或明确 N/A marker；
- `STUDY_SUMMARY.md`、可选 `STUDY_SUMMARY.html`、`study_summary.json`、`outputs_index.json`。

同一 figure 的 PNG 与 `.NA.txt` 必须互斥。空表/N/A plot 必须说明缺少哪个输入 artifact。只有所有 requested cells 和 required OOF levels 完整后才可排名。

### 计数规则

- `case count` 指 study 中不同的 resolved scientific configs。
- 完整 frailty 或目标完整 motion study：`outer-cell count = case count × 25`。
- 一个 ensemble case 的 scientific outer cells 仍为 25，但实际 member fits 为 `25 × 5 = 125`。
- 后文计数是失败/排除前预算；deferred group 即使给出目标设计，当前 runnable cells 仍为 0。

## 4. 有序实验计划

### Phase 0 — 所有 sweep 之前的完整性基线

#### Group 0A — canonical raw reference 验收

- **科学问题：** 冻结 reference 能否完成全部 25 cells，且无 leakage、silent fallback、OOF hierarchy 缺失或报告不完整？
- **Reference/config：** `configs/reference_static_role_aware_v2.yaml`、`configs/studies/single_config_v2.yaml`。
- **唯一变量与取值：** 无；这是 acceptance run，不是 ablation。
- **并行模块开关：** direct 0.2–8 Hz PPG；calibrated EKF；artifact `identity`；quality `off`；raw representation；`CompactCNN1D`；balance line B；motion disabled。
- **固定控制：** U1–U7 全部不变。
- **适用模型/representation：** 直接验证 raw frailty reference，同时验证四类 representations 共用的基础设施。
- **执行并行：** `jobs=1`。
- **输出要求：** 完整 R0；25/25 cells；window/file/role/participant OOF。因 `ensemble_size=1`，member OOF 应明确为 N/A。
- **晋级规则：** 若非 25 cells 全部通过、hash/config identity 一致、每个 participant 每 repeat 恰有一个 held-out prediction、无 forbidden fallback 且 report 无 incomplete exclusion，则停止后续阶段。本组 metric 仅作基线诊断，不设性能 cutoff。
- **预计规模：** 1 case、25 outer cells、25 network fits。
- **当前状态：** `RUNNABLE_PATH`；尚未科学运行。

命令：

```text
python frailty_3class_sweep_v2.py run --plan configs/studies/single_config_v2.yaml --repeats all --folds all --jobs 1
```

### Phase 1 — sentinel 的训练容量与 optimizer screening

先用小而清晰的 reference 排查明显 under/over-training，再比较 architectures。该结论不得静默套用到其他 architecture；若 finalist learning curve 明显不同，必须重新做 matched confirmation。

#### Group 1A — fixed epochs

- **科学问题：** frozen 10-epoch rule 相比 lower/higher capacity controls 是否合适？
- **Reference/config：** raw CompactCNN reference；`configs/studies/ablation_fixed_epochs_v2.yaml`。
- **唯一变量与取值：** `training.fixed_epochs in [7,10,15]`，reference=10；`epoch_profile` 由 materializer 一致解析，只是 metadata，不是第二 axis。
- **并行模块开关：** 仅 epoch profile 改变，其余保持 Group 0A。
- **固定控制：** U1–U7；optimizer、LR、weight decay、batch、sampler、class weights、loss、dropout、windows、seeds、aggregation 不变。
- **适用性：** Adam deep models；现有 persisted plan 仅 raw CompactCNN。classical estimator/ROCKET 无 epoch axis。
- **执行并行：** deep cases，`jobs=1`。
- **输出要求：** R0；全部与 top-only learning curves 只能使用逐 epoch `train_loss`。当前 fixed-epoch 合同没有合法的 inner-validation split，因此 `validation_loss` 和 `validation_BA` 必须是明确的 N/A marker，不得要求、补算或推断。预声明 final epoch 的 held-out outer participant metrics 仍按 U7 报告，但不得改称 validation curve，也不得用于逐 epoch 选择；另报相对 epoch 10 的 repeat-paired deltas。
- **晋级规则：** 仅 25-cell complete cases 可人工复核；participant BA 为第一排序，同时保留 macro-F1 CI95 下界和 worst-class F1，禁止按单 fold/epoch 的视觉最佳点选择。
- **预计规模：** 3 cases、75 cells、75 fits。
- **当前状态：** `RUNNABLE_PATH`；未运行。

#### Group 1B — two-axis optimizer grid，仅探索

- **科学问题：** 是否存在值得后续单因素确认的粗粒度 LR/weight-decay 区域？
- **Reference/config：** raw CompactCNN；`configs/studies/grid_optimizer_v2.yaml`。
- **比较值：** 这是明确标为探索性的唯一 two-axis 例外：Cartesian screen `learning_rate in [0.0003,0.001] × weight_decay in [0.0001,0.001]`，reference=`(0.001,0.0001)`；任何确认性结论必须拆到 Groups 1C/1D 的单因素 studies。
- **并行模块开关：** model 和所有非 optimizer 模块保持 Group 0A。
- **固定控制：** U1–U7；Adam、10 epochs、batch64、sampler/loss/class weights/dropout 不变。
- **适用性：** 仅当前 raw CompactCNN plan，不代表 classical 或不同 optimizer models。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、parameter-interaction plot、完整四点 parameter-effect table。
- **晋级规则：** 只能为 Groups 1C/1D 提名候选值；不得把该 grid 当因果单因素证据或最终 winner。
- **预计规模：** 4 cases、100 cells、100 fits。
- **当前状态：** `RUNNABLE_PATH`，但仅 descriptive screening。

#### Group 1C — learning rate 单因素确认

- **科学问题：** weight decay 固定后，LR 是否改变 participant-level performance/stability？
- **Reference/config：** `configs/reference_static_role_aware_v2.yaml`，优先在 Group 1B 后运行。
- **唯一变量与取值：** `training.learning_rate in [0.0003,0.001]`，reference=0.001。
- **并行模块开关：** 只改变 LR。
- **固定控制：** U1–U7，尤其 weight decay=0.0001、epochs=10。
- **适用性：** raw CompactCNN sentinel；必要时对 selected deep finalist 重做。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、相对 LR 0.001 的 repeat-paired deltas。
- **晋级规则：** complete cases 后人工综合 BA、macro-F1 lower bound、worst-class F1、calibration、learning curve；不虚构未登记 numeric cutoff。
- **预计规模：** 2 cases、50 cells、50 fits。
- **当前状态：** scalar `ablation` CLI 为 `RUNNABLE_PATH`；无 persisted plan、未运行。

#### Group 1D — weight decay 单因素确认

- **科学问题：** LR 固定后，regularization 是否改变 participant-level generalization？
- **Reference/config：** `configs/reference_static_role_aware_v2.yaml`。
- **唯一变量与取值：** `training.weight_decay in [0.0001,0.001]`，reference=0.0001。
- **并行模块开关：** 只改变 weight decay。
- **固定控制：** U1–U7，尤其 LR=0.001、epochs=10。
- **适用性：** raw CompactCNN sentinel；必要时对 selected deep finalist 重做。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、相对 0.0001 的 repeat-paired deltas。
- **晋级规则：** 同 Group 1C。
- **预计规模：** 2 cases、50 cells、50 fits。
- **当前状态：** scalar CLI 为 `RUNNABLE_PATH`；无 persisted plan、未运行。

#### Group 1E — sampler 与 class weighting 解耦

- **科学问题：** reference 同时启用 `balance_line_weighted_v2` sampler 和
  `outer_train_inverse_frequency` class-weighted CE；性能变化来自哪一项，还是必须把二者共同
  预注册为不可拆的 thesis design？
- **Reference/config：** raw CompactCNN reference；reference case 保持现有 sampler + class weighting。
- **唯一变量与取值：** 采用一个共同 reference 加两个彼此独立的单因素 alternatives：
  (a) sampler=`exhaustive_shuffle_without_replacement`、class weighting保持
  `outer_train_inverse_frequency`；(b) sampler保持 `balance_line_weighted_v2`、
  class weighting=`none`。禁止构造 sampler 与 class weighting 同时关闭的第四个 case，除非
  另建明确标为 interaction/exploratory 的 study。
- **并行模块开关：** 仅训练 sampler 或 loss weight 二者之一变化；dataset、folds、model、batch、
  optimizer、LR/WD、epochs、aggregation、seeds 与全部信号/representation模块固定。
- **固定控制：** U1–U7；所有 sampling weights 和 class weights 只能由 outer-training labels/
  participant/file/role identity构造，held-out participants不得参与。
- **适用性：** 先用于 raw CompactCNN sentinel；若 thesis 直接预注册现有组合而不做该消融，
  必须在方法文档中明确，不能把组合效果解释为某一个单独机制。
- **执行并行：** deep cases `jobs=1`。
- **输出要求：** R0、实际 sampler/class-weighting identity、每 fold class counts/weights、
  repeat-paired deltas；只比较每个 alternative 对共同 reference，不把两个 alternatives互比
  解释为单因素因果证据。
- **证据规则：** alternatives 已进入统一 Trainer/config schema 并有 outer-fold isolation
  tests；只由完整25-cell OOF和人工审阅决定保留组合或简化。
- **预计规模：** 计划3 cases、75 cells、75 fits；runtime 可运行，尚未建立本组 plan。
- **当前状态：** `IMPLEMENTED_NEEDS_PLAN`；sampler 与 class weighting 已是独立模块，
  可通过显式 study axes 组合，不需要修改 validator 或申请执行许可。

### Phase 2 — signal/preprocessing sensitivity，每次只动一个 profile

#### Group 2A — direct PPG passband

- **科学问题：** 唯一命名的 direct-filter alternative 是否实质改变下游 frailty evidence？
- **Reference/config：** 0.2–8.0 Hz reference；`configs/formal_ablation_profiles_v2.yaml` 的 `direct_filter` family。
- **唯一变量与取值：** filter profile：`(low,high)=(0.2,8.0)` 对 `(0.5,5.0)` Hz。两个字段共同定义一个科学因素，必须物化为两个 full configs，不能伪装成两个独立 axes。
- **并行模块开关：** 只换 direct filter；artifact 保持 identity，禁止第二次 filter。
- **固定控制：** U1–U7；filter family/order/phase、windows、IMU、normalization、model/training/seeds/aggregation 不变。
- **适用性：** 先 sentinel；若保留，再对依赖 direct view 的 selected representations 确认；本组不含 non-identity `x_ar`。
- **执行并行：** deep `jobs=1`；classical confirmation 可 case-level `jobs=2–4`。
- **输出要求：** R0、route/source-signal counts、retained coverage、per-class deltas；可附 spectra/preview，但不进入 ranking。
- **晋级规则：** 两 profiles 必须同 manifest/folds 且 coverage 完整；依据 participant metrics 与生理合理性人工选择，不自动胜出。
- **预计规模：** 每 model 2 cases/50 cells；四 representations 各一次为 8/200。
- **当前状态：** `CATALOG_ONLY`，无 persisted study plan。

#### Group 2B — calibrated EKF 对 Profile-A low-pass

- **科学问题：** calibrated roll–pitch EKF 相对 mandatory LPF ablation 是否改善 evidence？
- **Reference/config：** `imu_gravity` family。
- **唯一变量与取值：** `calibrated_roll_pitch_ekf` 对 `profile_a_lowpass_0p3hz`。
- **并行模块开关：** 只换 gravity profile；两路均要求显式 same-participant calibration/bias，LPF 永远不是 EKF failure path。
- **固定控制：** U1–U7；unit conversion、20/40 Hz sensor low-pass（zero-phase SOS order 3）、outer-train scaling、channels/windows/model/training/folds 不变；0.3 Hz gravity low-pass 仍为 order 4。
- **适用性：** 所有由 processed IMU 派生的 frailty representations 和 motion profiles；先 sentinel，不能与 motion 8/11ch 同时改变。
- **执行并行：** deep `jobs=1`。
- **输出要求：** R0、calibration identity、unit conversions、covariances、profile ID、failure/drop counts、role-wise IMU coverage。
- **晋级规则：** EKF unit tests 和 25 cells 全通过才保留 reference；含 silent LPF substitution 的 run 无效。
- **预计规模：** 每 selected model/profile 2 cases/50 cells。
- **当前状态：** paired frailty study 为 `CATALOG_ONLY`；未运行。

#### Group 2C — fixed-sample-kernel DL resampling

- **科学问题：** kernel sample count 不变、物理时长允许变化时，sampling rate 如何影响模型？
- **Reference/config：** `fixed_kernel_samples` family。
- **唯一变量与取值：** `dl_fs_hz in [100,160,200,400]`，reference=400；5 s sequence lengths 分别 `[500,800,1000,2000]`。
- **并行模块开关：** 只改 anti-aliased resampling；window duration、kernel samples、dilation 不变。
- **固定控制：** U1–U7；CompactCNN kernels `[9,9,7]` 或 InceptionFull `[39,19,9]`，dilation1、5 s context。禁止声称 physical-time-matched kernels。
- **适用性：** 只登记给 raw `CompactCNN1D`、raw `InceptionTimeFull`；不自动外推 ShapeFormer/vector/matrix/ROCKET/fusion。
- **执行并行：** 每 study 一个 model family，`jobs=1`。
- **输出要求：** R0、resolved fs/sequence length/kernel samples、描述性 effective kernel seconds、parameter count、inference cost。
- **晋级规则：** complete cases 人工复核 performance/stability/time-scale；不可混入 context/dilation。
- **预计规模：** 每 model 4/100；两 models 8 case appearances/200 cells。
- **当前状态：** `CATALOG_ONLY`，`auto_run=false`。

#### Group 2D — raw context duration

- **科学问题：** 10 s context 是否优于 5 s reference？
- **Reference/config：** `fixed_kernel_samples` family。
- **唯一变量与取值：** 400 Hz 下 window `[5.0,10.0]` s，sequence `[2000,4000]`。
- **并行模块开关：** 只改 context/window plan。
- **固定控制：** U1–U7；400 Hz、full profile 中明确的 hop、kernel samples、dilation1、model/training/folds 不变。
- **适用性：** raw CompactCNN、raw InceptionFull。
- **执行并行：** 每 model `jobs=1`。
- **输出要求：** R0、window count/cap/coverage、learning curves。
- **晋级规则：** 不混入 sampling-rate/dilation；必须 25-cell complete 并复核 coverage。
- **预计规模：** 每 model 2/50；两 models 4 appearances/100。
- **当前状态：** `CATALOG_ONLY`，未运行。

#### Group 2E — dilation

- **科学问题：** input rate、context、kernel samples 固定时，dilation 2 是否有益？
- **Reference/config：** `fixed_kernel_samples` family。
- **唯一变量与取值：** dilation `[1,2]`，reference=1。
- **并行模块开关：** 只改变 dilation。
- **固定控制：** U1–U7；400 Hz、5 s/2000 samples、对应 kernels、training/folds 不变。
- **适用性：** raw CompactCNN、raw InceptionFull。
- **执行并行：** 每 model `jobs=1`。
- **输出要求：** R0、resolved receptive field、parameter count、inference cost、learning curves。
- **晋级规则：** paired complete evidence；不能与 rate/context 合并变化。
- **预计规模：** 每 model 2/50；两 models 4 appearances/100。
- **当前状态：** `CATALOG_ONLY`，未运行。

### Phase 3 — balance、aggregation 与非因果 quality diagnostics

#### Group 3A — role-family line B 对 equal-file line A

- **科学问题：** role families 等权能否减少拥有更多 repeated-role files 的 participant/file 主导结果？
- **Reference/config：** `aggregation_balance` family。
- **唯一变量与取值：** coherent balance line：`line_b_equal_role_families` 对 `line_a_equal_files`。Line B=`training_balance: equal_role_families`、hierarchy `[window,file,role,participant]`；Line A=`equal_files`、hierarchy `[window,file,participant]`。
- **并行模块开关：** training balance 与 probability hierarchy 共同定义一个 line，必须作为两个 full profiles，不能拆成两个 axes。
- **固定控制：** 除 balance line 外 U1–U7 全部不变；ordinary means、missing-role rule、manifest/folds/seeds、representation/model/preprocessing 固定。
- **适用性：** 四种 frailty representations；先 sentinel，再 finalists。binary motion 除非另冻 aggregation contract，否则不适用。
- **执行并行：** deep `jobs=1`；classical case-level `jobs=2–4`。
- **输出要求：** R0、file/role OOF、role availability/count/coverage、participant OOF、paired deltas。
- **晋级规则：** 缺 role OOF，或 training/aggregation 混用未声明 lines，case 无效；人工综合 BA、macro-F1 lower bound、worst-class F1、role coverage，不自动选择。
- **预计规模：** 每 model 2/50；四 representations 各一 finalist 为 8/200。
- **当前状态：** Line-A comparison identity/semantics 已解决；仍为 `CATALOG_ONLY`，尚无 compound-line persisted study plan。

#### Group 3B — quality `off` 对 `diagnostics_only` 等价性审计

- **科学问题：** 能否记录 quality diagnostics 而完全不改变 retention、aggregation、predictors、predictions？
- **Reference/config：** 任一 accepted reference；`configs/v2_decision_profile.yaml`。
- **唯一变量与取值：** `quality.mode in [off,diagnostics_only]`，reference=`off`。
- **并行模块开关：** 仅 diagnostics computation 切换；route/reducer/retention/aggregation/sampling/predictors/motion override 保持 reference。
- **固定控制：** U1–U7；`diagnostics_only` 禁止把 quality 当 feature/weight。
- **适用性：** common pipeline audit；先 sentinel，可选在 finalist 复核。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、role/route component distributions、retained flags 与全部 participant probabilities 的逐行 equality audit。
- **晋级规则：** 必须 prediction/retention 完全一致；任何 metric 变化都说明 diagnostics-only 合同失败，不代表模型受益。
- **预计规模：** 2 cases、50 cells。
- **当前状态：** `IMPLEMENTED_NEEDS_PLAN`；manual mode 已实现，无 persisted plan/科学运行。

#### Group 3C — quality weighting、route 与 legacy aggregation

- **科学问题：** SQI/routing、显式质量权重来源或 legacy hierarchy 是否改变结果？代码可执行不代表这些策略已被科学验证。
- **Reference/config：** quality `off`、identity、line B。
- **唯一变量与取值：** `quality.mode in [off,route]`；aggregation weight source 可选 `none`、`route_file_q_rate` 或兼容表征下的 `legacy_window_sqi`；阈值、权重、window selector 与 hierarchy 均由 effective config 明确给出。`direct_all_window_participant_mean=true` 仍不是已实现 aggregation module。
- **并行模块开关：** route、window selection、quality weight source、Line A/Line B 可独立组合；不适用组合在训练前 fail-fast。
- **固定控制：** U1–U7；calibrator 只用 outer-training participants，quality/SQI 不作为默认 predictor。
- **适用性：** raw/fusion 可使用 window selection；`route_file_q_rate` 从 file 层起加权；`legacy_window_sqi` 仅用于带逐窗 score 的 raw OOF。
- **执行并行：** deep case 默认 `jobs=1`；不同配置可作 case-level parallelism。
- **输出要求：** resolved thresholds/weights、route states、source signal、coverage、retained flags、质量权重来源和 paired OOF。
- **晋级规则：** 只有完整 25-cell、outer-train-only provenance 与 matched OOF 才能支持科学结论；不使用 readiness boolean/artifact gate。
- **预计规模：** 每个 paired design 至少 2 cases/50 cells。
- **当前状态：** runtime modules 已实现；完整 paired study 与科学结果未运行，属 `IMPLEMENTED_NEEDS_PLAN`。

### Phase 4 — model families 与四条 representation lines

本阶段是 candidate comparison，不是因果单因素 ablation；更换 model family 必然改变 architecture。但所有 cases 仍共享相同 data、folds、适用 preprocessing/training policy 和 participant evaluation。catalog 条目必须先物化成 full pipeline config；catalog 本身不是 run plan。

#### Group 4A — raw single-network candidates

- **科学问题：** 在 ShapeFormer/ensemble 前，哪个 registered raw architecture 给出最强完整 participant OOF？
- **Reference/config：** `reference_static_role_aware_v2.yaml`；`compact_cnn`、`inception_small`、`inception_full`。
- **比较值：** `CompactCNN1D`（channels `[32,64,128]`、kernels `[9,9,7]`）；`InceptionTimeSmall`（depth3、bottleneck/out16、kernels `[39,19,9]`）；`InceptionTimeFull`（depth6、bottleneck/out32、同 kernels）。均为 single network。
- **并行模块开关：** raw on；vector/matrix/fusion off；每 case 恰一 raw model。
- **固定控制：** U1–U7；8ch、400 Hz/5 s、dropout0.2、dilation1、training policy、普通 repeat-dependent single-model seeds。
- **适用性：** raw frailty。
- **执行并行：** `jobs=1`，除非以后 memory test 授权。
- **输出要求：** R0、learning curves、parameter count、inference measurements、architecture/model-card identity。
- **晋级规则：** 仅 complete cases；BA 排序，同时显示 macro-F1 lower bound、worst-class、calibration、stability、compute；最多 10，不自动 winner。
- **预计规模：** 3 cases、75 cells、75 fits。
- **当前状态：** CompactCNN 有 persisted config；两 Inception 为 `CATALOG_ONLY`；完整三 case study 尚不可一键运行。

#### Group 4B — feature-vector classical candidates

- **科学问题：** 同一 feature schema/folds 下，哪个可读 file-vector baseline 最强？
- **Reference/config：** `reference_static_feature_vector_v2.yaml`；`logistic_regression`、`rbf_svm`、`extra_trees`。
- **比较值：** L2 logistic (`lbfgs`, max_iter5000)；RBF SVM (`C=1.0`,`gamma=scale`, probability on)；ExtraTrees (500 trees, estimator jobs1)；catalog 中 class_weight 均 null。
- **并行模块开关：** feature-vector on，每 case 恰一 estimator；其余 classifier lines off。
- **固定控制：** U1–U7；同 feature schema/hash、outer-train-only imputation/scaling、无 technical metadata、同 aggregation/folds。
- **适用性：** feature-vector frailty；estimator-specific preprocessing 只在 outer train 拟合。
- **执行并行：** case-level `jobs=2–3`；ExtraTrees/BLAS threads=1。
- **输出要求：** R0；coefficients/importances 仅补充，不能替代 OOF metrics。
- **晋级规则：** 要求 complete participant OOF 和 feature-transform provenance，按 U7 人审。
- **预计规模：** 3 cases、75 cells。
- **当前状态：** logistic 有 persisted config；SVM/ExtraTrees 为 `CATALOG_ONLY`。

#### Group 4C/4D — feature-matrix model selection pending

- **固定输入合同：** 10 s/2 s-hop，每窗 115 个工程特征，时间轴 K=150。
- **当前状态：** matrix representation 可构建和验证，但尚未选定正式模型或 ablation 轴。
- **已退役：** ROCKET/Ridge 与 MiniROCKET 不再是可执行 module、catalog case 或 study case；历史结果只能作为历史证据，不能冒充当前 115×150 合同。
- **后续要求：** 先明确候选模型，再新增引用统一 matrix builder 的独立 plan；禁止把旧 ROCKET 实现复制进新 YAML/runner。

#### Group 4E — file-level fusion candidates

- **科学问题：** 哪个 file-bag encoder 能在不产生跨 file/window 错位时最好融合 raw signal 与 file features？
- **Reference/config：** `reference_static_fusion_v2.yaml`；`fusion_compact`、`fusion_inception`。
- **比较值：** CompactCNN encoder（`[32,64,128]`, kernels `[9,9,7]`）对 InceptionSmall（depth3、bottleneck/out16、kernels `[39,19,9]`）；两者均 feature hidden32、fusion hidden64、mean pooling、dropout0.2。
- **并行模块开关：** fusion on；每 file 内 direct raw 8ch 与 feature branches 同时 active；standalone lines off。
- **固定控制：** U1–U7；file identity/bag boundary、schema/hash、8ch、training/aggregation。
- **适用性：** fusion frailty。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、file/bag alignment audit、missing-feature masks、两 branch IDs、learning curves、compute。
- **晋级规则：** cross-file 或 held-out feature fitting 使 case 无效；complete 后按 U7 人审。
- **预计规模：** 2 cases、50 cells。
- **当前状态：** fusion Compact 有 persisted config；fusion Inception `CATALOG_ONLY`。

#### Group 4F — 四 representation finalists

- **科学问题：** raw、feature vector、ordered matrix、file-level fusion 中哪条 line 最适合 final use case？
- **Reference/config：** 四个 reference configs 或各自人工 shortlist 的 materialized finalist。
- **比较值：** `[raw,feature_vector,feature_matrix,fusion]`；因 representation-specific model 不同，这是 candidate comparison，不是因果单因素。
- **并行模块开关：** 每 case 恰一 representation；共享 common signal preparation/folds。
- **固定控制：** 语义允许范围内 U1–U7；同 manifest/folds/label order/quality/artifact/balance/participant metrics/reporting。
- **适用性：** final frailty selection；motion 不是第五 representation。
- **执行并行：** mixed-cost 建议分 studies；deep `jobs=1`，classical `jobs=2–4`。
- **输出要求：** 每 case R0；合并人审表含 representation/model、BA、macro-F1 CI95 lower bound、worst-class、calibration、coverage、parameters、cost。
- **晋级规则：** incomplete route 不排名；至少保留每 representation 最可信 finalist 直至 sensitivity/ensemble confirm；最终人工决定。
- **预计规模：** 4 cases、100 cells，后续 ensemble multiplier 另计。
- **当前状态：** `IMPLEMENTED_NEEDS_PLAN`；finalist IDs 依赖前序结果。

### Phase 5 — ShapeFormer fidelity 与 discovery-route comparisons

#### Group 5A — channel-specific OSD/PISD reference 验收

- **科学问题：** literature-reference ShapeFormer 能否按 fold-local discovery 合同忠实运行并显式失败？
- **Reference/config：** `shapeformer_channel_specific_osd`、`shapeformer_channel_specific_osd.md`。
- **唯一变量与取值：** 无；reference acceptance。
- **并行模块开关：** raw 8ch generic branch on；`channel_specific_osd` discovery；其他 discovery routes off。
- **固定控制：** U1–U7，加 `num_pip_ratio=0.20`（由实际 `T` 计算）、三 consecutive PIPs 限定 variable candidates、3 shapelets/class（共9）、participant/file-balanced max180 discovery windows、position-search neighbourhood128、400 Hz/2000 samples、local width8、local embed48、shape embed128、FF256、4 heads、dropout0.30。
- **适用性：** 仅 raw frailty ShapeFormer。每 candidate 只有一个 `source_channel`，discovery 与 best-fit search 都只在该 channel；shapelet bank 可来自八通道，generic branch 仍接收完整八通道，因此模型仍是 multivariate。
- **执行并行：** `jobs=1`；每 outer-training fold 独立 discovery。
- **输出要求：** R0；每 cell 完整 bank：source channel、start/end samples、start/end seconds、candidate length、class、source file/participant、IG/rank metadata、fold/preprocessing hashes。
- **晋级规则：** 25 cells 全完成且无 fallback。若 hard-code 64 PIPs、固定 shapelet length/candidate stride、把 neighbourhood128 当 shapelet length、跨 channel search、held-out leakage、class count 不等或替换 effect-size，run 无效。
- **预计规模：** 1 case、25 cells、25 high-compute fits 加 fold-local discovery。
- **当前状态：** algorithm/catalog 为 `implemented_not_benchmarked_high_compute`；属 `IMPLEMENTED_NEEDS_PLAN`，完整 5×5 前不能称 thesis-validated。

关键解释：64 PIPs 只是在 5 s、64 Hz 时的推导值 `0.20 × 5 × 64`，不是常量；历史 PISD `window_size=128` 是 position-search neighbourhood，绝不是 shapelet length。

#### Group 5B — faithful OSD 对 fixed effect-size route

- **科学问题：** literature-faithful channel-specific route 与保留的 historical fixed-length method 表现如何？
- **Reference/config：** `shapeformer_channel_specific_osd` 对 `shapeformer_effect_size_fixed_v1`。
- **比较值：** named route `channel_specific_osd` 对 `effect_size_fixed_v1`。fixed route：length128、candidate stride64、3/class、max128 candidates/class、hidden64、patch16、4 heads/1 layer、dropout0.2。
- **并行模块开关：** 每 case 恰一 ShapeFormer route；禁止 fallback/alias。OSD 始终是 literature reference，effect-size 即使得分更高也仍为 ablation。
- **固定控制：** U1–U7；同 manifest/folds、8ch、class order、outer training、aggregation、participant metrics。因 downstream architecture/discovery semantics 也不同，本组是 named route comparison，不是纯单参数因果实验。
- **适用性：** raw frailty ShapeFormer。
- **执行并行：** `jobs=1`，serial。
- **输出要求：** R0、route-specific discovery provenance/bank、shapelet visualization、compute/memory、explicit failure。
- **晋级规则：** 仅 25-cell complete cases 可比较；reference label 取决于 fidelity 而非分数；PISD failure 不能用 effect-size 结果填 OSD row。
- **预计规模：** 2 cases、50 cells、50 high-compute fits/discoveries。
- **当前状态：** 两 route 已 catalogued；paired study 为 `CATALOG_ONLY`/`IMPLEMENTED_NEEDS_PLAN`，未运行。

#### Group 5C — optional fixed-length 128/400/800

- **科学问题：** fixed-length effect-size family 对 candidate length 多敏感？
- **Reference/config：** 当前仅 `effect_size_fixed_v1` 已物化为 fixed-length reference。
- **唯一变量与取值：** `shapelet_length_samples in [128,400,800]`，reference=128；三者都只能叫 fixed-length ablation，绝不能标 PISD/OSD。
- **并行模块开关：** 只走 fixed effect-size；faithful OSD 属独立 Group 5A/5B。
- **固定控制：** U1–U7；所有 effect-size architecture/discovery 字段（含一个共同 candidate-stride policy）运行前必须冻结。
- **适用性：** raw fixed-effect ShapeFormer。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、完整 fixed bank、length samples/seconds、candidate count、compute。
- **证据规则：** 每个 case 显式记录 length、stride、candidate cap 与 feasibility；不得借用 PISD neighbourhood128 解释。
- **预计规模：** target 3/75；runtime 参数可运行，正式 plan 尚未建立。
- **当前状态：** `IMPLEMENTED_NEEDS_PLAN`。

#### Group 5D — optional joint multichannel PIP-centered IG

- **科学问题：** joint eight-channel candidate 与 canonical channel-specific route 是否不同？
- **Reference/config：** `channel_specific_osd`；optional route 必须命名 `multichannel_pip_centered_ig`。
- **唯一变量与取值：** candidate-channel policy `[single_source_channel,joint_eight_channel]`，仅在 joint route 有完整独立 config 后。
- **并行模块开关：** 每 case 一 route；joint route 不得叫 `PISDPort`、不得作 reference/fallback。
- **固定控制：** U1–U7 和兼容 ShapeFormer capacity/training。
- **适用性：** raw ShapeFormer ablation。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、route name、足以证明未混线的 per-candidate channel semantics。
- **晋级规则：** 单独 implementation/config/model card 登记并验证前不运行；literature claim 只属于 channel-specific OSD。
- **预计规模：** future 2/50；当前 0 runnable。
- **当前状态：** `PLANNED_NOT_RUNNABLE`；当前 decision profile 未登记旧 joint route。

### Phase 6 — matched five-member ensemble 因素

Phase 4 的普通 Inception 单网络使用 repeat-dependent training seeds，用于 architecture comparison；它不是 matched ensemble comparator。以下 one-network case 必须固定 member0 seed50042。

当前实现可分别构建 matched member0 与 five-member ensemble 的两个完整 configs，但尚无 persisted paired study plan，也没有把 member count 表达为 typed compound comparison axis。因此 6A/6B 目前不能通过一条 study command 成对运行；必须先新增并 dry-run 验证对应的 typed paired plan。

#### Group 6A — raw InceptionFull member0 对 five-member ensemble

- **科学问题：** raw InceptionFull route 固定时，五个独立初始化 members 的 probability mean 带来什么价值？
- **Reference/config：** matched comparator entry `inception_full_member0_comparator`（config stem
  `comparison_inception_full_member0_comparator`）对
  `inception_full_five_member_ensemble`；ordinary `inception_full` 只属于 Phase 4，不能用于本组。
- **唯一变量与取值：** ensemble size `[1,5]`。size1=`member_index=0`, seed50042；size5 roster=`[50042,60042,70042,80042,90042]`，arithmetic mean。
- **并行模块开关：** raw 与同一 InceptionFull architecture 保持，仅 member count 改变。
- **固定控制：** U1–U7、U2；同 split/input/training/fold-local preprocessing/aggregation/calibration evaluation。
- **适用性：** raw InceptionFull。
- **执行并行：** `jobs=1`；不得通过未记录 nested worker 混池 members。
- **输出要求：** R0；五 seeds 的 member OOF/provenance、fold ensemble means、matched member0 probabilities、五 repeats stitched participant metrics。
- **晋级规则：** 若平均 member metrics、选择 members、按 repeat 轮换 seeds，或用 ordinary repeat-seeded model 作 comparator，case 无效；人工比较 ensemble-minus-member0 paired evidence 与 cost。
- **预计规模：** 2 scientific cases、50 cells；实际 150 fits（25 comparator+125 ensemble）。
- **当前状态：** seed/model/runner/provenance 实现及相关安全测试已经通过，且两个 full configs 可分别构建；但缺 persisted typed paired study plan/compound axis，故为 `IMPLEMENTED_NEEDS_PLAN`。新增 plan 并通过 resolved-config dry-run 前不能称为一条命令可运行；尚未进行科学运行。

#### Group 6B — feature-matrix Inception member0 对 ensemble

- **科学问题：** 同一 fixed roster 是否帮助 ordered matrix 上的 InceptionTimeMatrix？
- **Reference/config：** matched comparator entry `inception_matrix_member0_comparator`（config stem
  `comparison_inception_matrix_member0_comparator`）对
  `inception_matrix_five_member_ensemble`；ordinary `inception_matrix` 只属于 Phase 4，不能用于本组。
- **唯一变量与取值：** ensemble size `[1,5]`；member0 seed50042；roster `[50042,60042,70042,80042,90042]`。
- **并行模块开关：** feature-matrix、schema/mask、member architecture 不变，只改 member count。
- **固定控制：** U1–U7、U2，包含完全相同 fold-local matrix transform。
- **适用性：** feature-matrix Inception；不是 ROCKET ensemble。
- **执行并行：** `jobs=1`。
- **输出要求：** R0、全部 member OOF、matrix-transform hashes、ensemble/member0 paired outputs。
- **晋级规则：** 同 6A；每 repeat 对 stitched held-out probabilities 评分一次，不平均 member/fold metric rows。
- **预计规模：** 2 cases、50 cells、150 fits。
- **当前状态：** seed/model/runner/provenance 实现及相关安全测试已经通过，且两个 full configs 可分别构建；但缺 persisted typed paired study plan/compound axis，故为 `IMPLEMENTED_NEEDS_PLAN`。新增 plan 并通过 resolved-config dry-run 前不能称为一条命令可运行；尚未进行科学运行。

两个 representation 的目标设计合计 4 scientific cases、100 outer cells、300 fits；在两个 typed paired plans 物化前当前 runnable case/cell 均为 0。final full-data refit 不计入此处。

### Phase 7 — artifact reducer 实现初筛、证据回返与 rate-only 下游路线

Phase 7/8 的编号表示模块分组，不是简单线性执行顺序。为避免 reducer winner 依赖尚未生成的 A3/A4 证据而自锁，必须按以下依赖执行：

1. **7-pre：** 只完成 reducer 实现、参数冻结、provenance、failure semantics 和内部非排名 smoke/feasibility；此时 identity 仍为 default，禁止选择 winner。
2. **8-core：** 完成 8A–8C 的 formal motion 5×5 reference、8ch/11ch augmentation 与 EKF/LPF evidence。
3. **identity evidence：** 在 reducer 仍为 identity、override 仍关闭时，完成 8D 的 identity A3 baseline 和 8E 的 identity A4 baseline。这是 winner claim 的证据依赖，不是 runtime authorization。
4. **7-return：** 返回 7A，使用同一冻结 A3/A4 endpoints 做 reducer-vs-identity paired evidence；之后才允许人工冻结 V2-012 reducer winner。
5. **downstream：** winner 冻结后才运行 7C，以及 8E 中依赖 selected reducer 或 motion override 的后续部分。

因此第一次进入 Phase 7 只能做实现/内部初筛；7A/7B 的正式 paired evidence 与 winner selection 是完成 identity A3/A4 后的回返阶段。

#### Group 7A — identity 对六个 registered reducers

- **科学问题：** 哪个 reducer（若有）能改善 heartbeat/rate evidence，同时不伪造 morphology、不产生不可接受 failure/coverage loss？
- **Reference/config：** artifact `identity`；`nlms_imu_anc`、`ssa_decomposition`、`spectral_mask`、`pca_bss`、`fastica_bss`、`nmf_bss`。
- **唯一变量与取值：** 每 case 恰一个 reducer ID，并始终与 identity 配对；reducer-specific hyperparameters 必须运行前冻结，不能看 outer-test 后再选。

当前源码默认值如下。它们是待预注册的 implementation defaults，不是已由 A3/A4 选出的 formal
参数；正式 materialization 必须逐字段保存并 hash，若人工改值则另建 parameter-ablation plan。

| Reducer | 当前 implementation defaults（正式运行前须冻结） |
|---|---|
| `identity` | 无 reducer 参数；原 direct `x_filter` 作为 control |
| `nlms_imu_anc` | `taps_per_delay=8`；`delay_taps=[0,4,8,16]`；`step_size=0.15`；`epsilon=1e-6`；`leakage=1e-5`；`update_gate_reference_rms=0.10`；IMU reference=`axes6` |
| `ssa_decomposition` | `embedding_samples=160`；`max_components=12`；`minimum_cardiac_concentration=0.45`；`cardiac_low_hz=0.5`；`cardiac_high_hz=3.5` |
| `spectral_mask` | `stft_window_s=4.0`；`stft_hop_s=1.0`；`imu_mask_quantile=0.75`；`mask_strength=0.80`；`preserve_band_hz=[0.5,3.0]`；IMU reference=`axes6` |
| `pca_bss` | deterministic PCA；rate-source selection 使用 cardiac concentration 减 motion correlation；IMU reference=`axes6`；不把 FastICA/NMF 的迭代字段解释为 PCA 参数 |
| `fastica_bss` | `random_state=42`；`max_iter=1000`；`tolerance=1e-5`；IMU reference=`axes6` |
| `nmf_bss` | `random_state=42`；`max_iter=1000`；`tolerance=1e-5`；`nmf_rank=2`；`nperseg=512`；`overlap_fraction=0.75`；IMU reference=`axes6` |

任何未列出的 selector/failure threshold 均不得在正式运行时临时调参；若实现中存在派生有效值
（例如短序列下的 effective `nperseg/noverlap`），必须同时保存 requested 与 effective provenance。

- **并行模块开关：** direct identity 或一个 non-identity rate-only `x_ar`；motion IMU reference 为 6 physical axes，11ch augmentation 属 Phase 8 独立因素。
- **固定控制：** 适用的 U1–U7；同 source records、signal preparation、endpoint evaluator、folds、calibration、failure rules。post-reducer quality 只评 rate；morphology/optical waveform features 应为 unavailable，不能填 0。
- **适用性：** 首次进入本组只做实现/内部初筛；正式比较必须先取得 Phase 8 identity A3 external PTT rate baseline 与 identity A4 internal dynamic baseline。回返完成 paired evidence 后，只有人工 selected reducer 可进入 vector/matrix 的 rate-only frailty comparison；raw/fusion waveform branch 必须 direct/identity。
- **执行并行：** memory/thread check 后 reducer cases 可 `jobs=2–4`；禁止 nested estimator threads。
- **输出要求：** A3 rate/beat error、按 reducer/motion state 的 coverage/failure、A4 rate-quality/availability、reducer version/params/status/hash、source signal、route state、R0-compatible paired summaries；A4 不得声称 waveform-restoration accuracy。
- **晋级规则：** 8A–8C 和 identity A3+A4 完成并人审前不得运行 winner selection；正式 reducer pairs 必须复用冻结 endpoints。failure 不能回退 identity 却保留 reducer label；此前 identity 保持 default。
- **预计规模：** 7 unique cases/175 external 5×5 eval cells；若六个 pair 分目录为 12 appearances/300 cells，可引用同一冻结 identity 结果；当前 runnable=0。
- **当前状态：** 算法已实现/登记，可运行实现初筛；正式 configs/hyperparameters 和前置 identity A3/A4 evidence 未完整，故 winner claim 为 `DEFERRED_EVIDENCE`。

#### Group 7B — historical EMD/CEEMD/DWT

- **科学问题：** 可为 provenance 复现历史行为；具名运行不会自动替换 V2 reference。
- **Reference/config：** identity 对 `emd_sifting_rate_only`、`ceemd_lite_nlms_legacy`、`dwt_a2_legacy`，使用 optional artifact-legacy environment。
- **唯一变量与取值：** 每次一个 historical reducer ID；必须从 provenance 恢复精确 versions/params，不得近似。
- **并行模块开关：** 只走 legacy rate-only；raw/fusion 不用，也不标 reference。
- **固定控制：** 每次 resolved config 冻结 legacy 参数，并保持 Group 7A endpoint inputs/evaluation。
- **适用性：** provenance/legacy comparison。
- **执行并行：** 可作为独立 named cases；缺少可选依赖时明确 unavailable。
- **输出要求：** historical dependency profile、精确 provenance、failure/coverage、`legacy_ablation` label。
- **晋级规则：** 只有完整 matched evidence 和人工选择才能改变 reference；普通 named run 无需授权。
- **预计规模：** 当前 0；未来三个 identity pairs 为 6 appearances/150 cells。
- **当前状态：** historical named modules 已注册；正式 paired plan/结果未完成。

#### Group 7C — selected reducer 进入 frailty feature routes

- **科学问题：** 若 A3/A4 选出 reducer，其 rate-only feature route 相对 direct identity 是否帮助 frailty prediction？
- **Reference/config：** identity direct vector/matrix finalist 对唯一人工 selected reducer。
- **唯一变量与取值：** signal route `[direct_identity,selected_rate_only_reducer]`，reducer ID 固定。
- **并行模块开关：** 仅 vector/matrix；`x_ar` 的 direct morphology/dual-wavelength 字段为 `unavailable/validity=false`；raw/fusion 不 eligible。
- **固定控制：** U1–U7、同 selected model、feature schema、folds/training/aggregation。
- **适用性：** engineered features；若要把结果称为“selected reducer downstream”，必须依次完成 8A–8C、identity A3/A4、7A 和 V2-012 人工选择；普通具名 reducer case 本身可执行。
- **执行并行：** classical `jobs=2`；matrix Inception `jobs=1`。
- **输出要求：** R0、source signal/route/coverage、feature availability/validity；SQI/coverage 仍是 routing/weighting metadata，不进入 physiological predictor vector。
- **晋级规则：** reducer winner 未在 Phase 7 回返阶段人工冻结前不能开始；把 unavailable morphology 当 0 或静默返回 direct features 均无效。
- **预计规模：** 每 representation 2/50；vector+matrix 为 4 appearances/100。
- **当前状态：** runtime route 已实现；V2-012 winner 与 A3/A4 科学证据为 `DEFERRED_EVIDENCE`。

### Phase 8 — motion branch、derived augmentation 与 endpoint evidence

frailty labels 绝不能训练 motion detector。内部 binary targets 是 protocol activity：B/R=static、S/W=motion，按 participant 分组。所有可比较 motion thesis 结果目标均为 5 repeats×5 grouped folds；当前 internal motion contract 只有 1 repeat×5 folds，不能冒充目标证据。

本阶段先完成 8A–8C，再以 identity route 依次建立 8D/A3 与 8E/A4 baseline；它们是回到 Phase 7 做 reducer paired comparison 的前置证据，不是 reducer winner 的结果。selected-reducer 和 override 相关的 8E 部分只能在 Phase 7 回返并人工选择后执行。

#### Group 8A — 8-channel motion reference 验收

- **科学问题：** frozen 8ch motion detector 能否生成完整 participant-grouped evidence，且不改变 frailty predictions？
- **Reference/config：** `configs/motion_detector_contract_v2.yaml`，`motion_8ch_axes_reference_v2`。
- **唯一变量与取值：** 无；reference acceptance。
- **并行模块开关：** binary endpoint motion detector on；frailty override off。tensor `[RED,IR,A_dyn_x,A_dyn_y,A_dyn_z,GX,GY,GZ]`，400 Hz，8 s/2 s（3200/800 samples）。
- **固定控制：** U3–U4；10 epochs、batch16、Adam LR0.001、WD0、BCE-with-logits、historical class/participant sampler、dropout0、clip1.0、无 augmentation。
- **适用性：** internal motion endpoint，不是 frailty representation。
- **执行并行：** `jobs=1`。
- **输出要求：** participant-grouped OOF scores/labels、只在 outer train 拟合的 frozen threshold、BA、macro-F1、worst-fold BA、ECE、params/cost、preprocessing hashes。
- **证据规则：** 必须 25/25 cells 才能支持目标强度的 downstream 科学结论；现有 1×5 不得改名 5×5，但它不作为任何 core runtime 的执行许可。
- **预计规模：** target 1/25；当前合同仅 5 diagnostic cells。
- **当前状态：** implementation/contract 已登记，但作为 thesis 5×5 为 `PLANNED_NOT_RUNNABLE`，需升级并验证 motion split/training contract。

#### Group 8B — motion 8ch 对 11ch derived augmentation

- **科学问题：** `A_mag`、`Omega_mag`、`J_mag` 是否在 6 physical IMU axes 之外增加 motion 信息？
- **Reference/config：** `motion_8ch_axes_reference_v2` 对 `motion_11ch_derived_augmentation_ablation_v2`。
- **唯一变量与取值：** channel profile `[8ch_axes_reference,11ch_derived_augmentation]`；11ch 只增加 `[A_mag,Omega_mag,J_mag]`。
- **并行模块开关：** 每 case 一个 motion tensor；frailty 始终8ch、override off。
- **固定控制：** 同 5×5 participants/folds/seeds、labels/model/training/windows、EKF/PPG normalization；8ch scaling 6 IMU，11ch scaling 全9 IMU，均 outer-training-participant-only。
- **适用性：** motion detector；11ch 永不具 frailty predictor 资格。
- **执行并行：** `jobs=1`，serial。
- **输出要求：** 8A 输出加 tensor order/unit/schema hash、paired deltas。
- **晋级规则：** matched 25 cells；禁止 per-window IMU amplitude equalization 和 silent profile substitution。
- **预计规模：** target 2/50。
- **当前状态：** 11ch constructor 已命名可构造，但 formal 5×5 为 `PLANNED_NOT_RUNNABLE`，未运行。

#### Group 8C — motion EKF 对 Profile-A LPF

- **科学问题：** motion endpoint 对 gravity profile 多敏感？
- **Reference/config：** calibrated EKF 对 `profile_a_sensor_lpf_order3_gravity_0p3hz_v4_ablation_only`。
- **唯一变量与取值：** 固定 channel profile 内 `[calibrated_roll_pitch_ekf,profile_a_lowpass_0p3hz]`。
- **并行模块开关：** 一次固定 8ch 或 11ch，不能与 derived augmentation 交叉变化。
- **固定控制：** 8A/B；两路都有显式 same-participant calibration、同 unit/scaling；LPF 不是 error recovery。
- **适用性：** motion endpoint；先 8ch，只有其晋级后才可选 11ch confirmation。
- **执行并行：** `jobs=1`。
- **输出要求：** motion R0-equivalent、calibration/covariance/profile provenance、failures、paired metrics。
- **晋级规则：** full 5×5、explicit failures、无 fallback。
- **预计规模：** 8ch 2/50；可选 11ch 再 2/50。
- **当前状态：** `PLANNED_NOT_RUNNABLE`，无 5×5 plan。

#### Group 8D — external PTT-PPG benchmark (A3)

- **科学问题：** 完整 internal evidence 后，frozen motion/rate pipeline 向 ECG-reference PTT endpoint transfer 如何？
- **Reference/config：** `splits/ptt_formal_repeated_grouped_5x5_v2.csv`、`manifests/ptt_imu_unit_evidence_v2_036.json`。
- **比较值：** 先一个 frozen 8ch reference；11ch/reducer 只能作另外命名的 matched comparisons；外部禁止 hyperparameter fitting。
- **并行模块开关：** evaluation only；external 只测试 motion/heartbeat/rate，绝不提供 frailty labels、training、calibration tuning、threshold optimization 或 outer-test 内 reducer selection。
- **固定控制：** external 5×5 seeds `[42,10042,20042,30042,40042]`；SIT acceleration identity m/s²、gyro deg/s→rad/s、same-participant SIT calibration、frozen internal model/threshold。
- **适用性：** A3 endpoint；当前 registry 明确不作 independent-test claim。
- **执行并行：** 未量测前 `jobs=1`。
- **输出要求：** participant-macro BA/F1、worst-fold BA、ECE、coverage/failure、可用的 heartbeat/rate errors、params/cost、unit-evidence hash、bundle IDs。
- **晋级规则：** complete internal motion 5×5 加精确 V2-036 source/unit evidence 后，结果才具备 matched scientific interpretation；先报告 identity A3 baseline。任何 external fit/recalibration 都使 benchmark 无效；reducer comparisons 必须等该 baseline 冻结后回到 7A/7B。
- **预计规模：** reference 1/25 external eval；8/11ch 为 2/50；reducer count 见 7A。
- **当前状态：** `DEFERRED_EVIDENCE`；未运行 full PTT benchmark。

#### Group 8E — identity A4 baseline、post-selection dynamic report 与 motion override

- **科学问题：** A4 记录 motion 下 rate quality/coverage/failure，不证明 clean-waveform accuracy；未来另问 supervised motion evidence 是否应改变 frailty routing。
- **Reference/config：** 先做 direct identity、override disabled 的 A4 baseline；selected reducer 与 override comparison 属 Phase 7 winner 冻结后的第二阶段。
- **唯一变量与取值：** identity A4 baseline 是 diagnostic report，不是 classifier ablation；后续一次只能比较 direct identity 对 selected reducer，或在另一个 study 比较 `[off,on]` override。
- **并行模块开关：** diagnostics 只进 parallel metadata，不得改变 retention/aggregation/frailty predictions；override off。
- **固定控制：** U1–U7、endpoint labels/route semantics。
- **适用性：** internal B/R/S/W diagnostics；supervised routing 必须等证据。
- **执行并行：** 可 batch diagnostics；performance claim 仍需完整 matched evidence。
- **输出要求：** route/source、reducer status/failure、motion state、rate confidence/agreement、feature availability、coverage、role/participant summaries。
- **晋级规则：** identity A4 baseline 必须在 7A/7B 正式 paired evidence 前冻结；selected-reducer A4 与 7C 必须等 V2-012 winner。override 还必须人审 paired A4、complete A3 和 internal formal motion；禁止 waveform-restoration claim。
- **预计规模：** diagnostic rows，不按 frailty cases；override 当前 0 runnable。
- **当前状态：** A4 planned/部分支持；V2-010 override 为 `DEFERRED_EVIDENCE`。

### Phase 9 — function-level PRV 与未来 feature ablations

#### Group 9A — fixed-PPI PRV backend comparison

- **科学问题：** local、Aura、historical Rhenan functions 在相同 synthetic PPI 上是否足以记录 backend 行为？
- **Reference/config：** formal `local`；optional `aura_hrv_analysis`、`rhenan_hrv`。
- **唯一变量与取值：** backend `[local,aura_hrv_analysis,rhenan_hrv]`；五个固定、未修改、millisecond fixtures：`steady_75bpm`、`alternating_75bpm`、`dual_modulated`、`slow_trend`、`single_outlier_unmodified`。
- **并行模块开关：** 仅 function comparison；cleaner off；classifier integration off。
- **固定控制：** 相同 input bytes/hash 与 native output names。Aura environment：`hrv-analysis==1.0.2`、Python3.11.14、`nolds>=0.4.1` 且最高已测兼容0.6.2（0.6.3不兼容）、Astropy5.2.2、NumPy1.26.4。
- **适用性：** 只作 PRV function documentation，不是 frailty model 或 5×5 classifier evidence。
- **执行并行：** sequential 即可；optional package 缺失必须给每 backend 显式 unavailable status。
- **输出要求：** 15 backend-fixture rows、package versions、input hashes、native outputs、explicit unavailable/failure。
- **晋级规则：** local 保持 formal backend；禁止声称 numeric identity、clinical superiority、interchangeability 或 classification performance。
- **预计规模：** 3×5=15 function evaluations，无 outer cells。
- **当前状态：** implementation 已有、optional isolated profile 已 pin；不属于 sweep-runner study，也未建立 classifier evidence。

#### Group 9B — leave-one-feature-family-out

- **科学问题：** 哪些 physiological feature families 对 selected vector/matrix/fusion model 有贡献？
- **Reference/config：** selected complete representation，加 frozen `FeatureVectorV1`/`OrderedFeatureMatrixV1`。
- **唯一变量与取值：** 每 study 一个 family `[present,removed]`。可执行入口为 `features.enabled_groups`；registered families 是 basic PPI/rate、HRV time-domain、HRV spectral、HRV nonlinear、direct morphology、direct dual-wavelength optical、engineering summary。
- **并行模块开关：** 固定一个 representation/model，只 disable 一个 predictor family；解释其余数据所需 validity/mask 保留。
- **固定控制：** U1–U7；source-route eligibility、fold-local transform、feature order/hash、model/training/aggregation 固定。non-identity `x_ar` 中 morphology/optical 本就 unavailable，不能冒充 removal ablation。
- **适用性：** finalists 之后的 vector/matrix/fusion，不适用 raw-only。
- **执行并行：** classical `jobs=2–4`；deep matrix/fusion `jobs=1`。
- **输出要求：** R0、完整 included/removed names、schema hashes、missingness/validity changes、paired deltas。
- **晋级规则：** 每次只去一个 family，禁止把 combinatorial subset search 冒充单因素 ablation；每个 resolved case 必须保存派生的 registry/vector/matrix schema 与 hash。
- **预计规模：** 五 pair studies=每 representation 10 appearances/250 cells；当前 runnable=0。
- **当前状态：** registry cropping 与全链路 schema 已 runnable；正式 family-removal cases 尚未加入 13-case catalog，也未运行。

#### Group 9C — technical/administrative metadata 边界

- **科学问题：** thesis 要验证的是 physiological predictors，而不是利用 identifiers/acquisition volume 泄漏标签。
- **Reference/config：** 所有 reference 的 `features.technical_metadata_allowed=false`。
- **唯一变量与取值：** 无授权 axis。participant/record/path、absolute file order、row count、duration、window count、administrative missingness、route/reducer identity、coverage、SQI 均不在 predictor allowlist。
- **并行模块开关：** 这些 metadata 只写 parallel audit tables，不进 tensor/vector。
- **固定控制：** U1–U7、frozen feature schema。
- **适用性：** 所有 feature/fusion routes。
- **执行并行：** N/A。
- **输出要求：** predictor name/schema audit 证明排除；字段仍可用于 coverage/A4 reporting。
- **晋级规则：** 除非另建命名且人工授权的 quality-aware ablation，使用上述字段的 case 无效；即使授权，identifiers 仍禁止。
- **预计规模：** 0。
- **当前状态：** identifiers 在 ordinary V2 predictor 中为 `PROHIBITED`；quality/SQI 可用于显式 routing/weighting，但仍不进入 physiological predictor vector。

### Phase 10 — finalists confirmation 与人工选择

#### Group 10A — finalist confirmation suite

- **科学问题：** 所有选定单因素已锁进各自 config 后，shortlisted use-case candidates 是否仍可辩护？
- **Reference/config：** 每 representation 最多一个 complete finalist，所有 upstream profiles 显式 resolved。
- **比较值：** candidate config ID；这是 final candidate comparison，不是因果 ablation。禁止把前面所有 winners 再做 Cartesian grid。
- **并行模块开关：** 每 case 一条完整 end-to-end line，声明 filter、IMU、artifact/quality、representation/model、balance line、training、seed policy、aggregation、calibration reporting。
- **固定控制：** U1–U7 和前阶段人工 accepted settings；必须复用同 frozen folds/seeds 才能 paired compare。
- **适用性：** final frailty use-case selection；motion/A3/A4 仍是独立 endpoint。
- **执行并行：** deep `jobs=1`；资源差异大时各自独立 dated study 归档。
- **输出要求：** 完整 R0；cross-study table 含 config/model/manifest/fold/preprocessing/feature hashes、BA、macro-F1 CI95 lower bound、worst-class、calibration、coverage、params、measured cost。
- **晋级规则：** 只人工。必须 25 cells 与全部 OOF levels 完整且无前序数据合同/fallback 违规；mean BA 第一排序，macro-F1 lower bound 与 worst-class 为强制 guard columns；不虚构未确认 threshold。
- **预计规模：** shortlist 决定 1–4 cases、25–100 cells。
- **当前状态：** upstream decisions/finalist configs 冻结前 `PLANNED_NOT_RUNNABLE`。

### Phase 11 — final-use-case full-data refit 与 bundle

#### Group 11A — final refit/bundle

- **科学问题：** 无；refit 只封装已选 use case，不得重新选择或调参。
- **Reference/config：** 唯一人工批准 final config，以及全部 frozen preprocessing/feature/model schemas。
- **唯一变量与取值：** 无；final refit 不属于 outer-CV comparison。
- **并行模块开关：** 只启用 selected end-to-end route；frailty raw/fusion/ShapeFormer 始终恰好8ch；motion 11ch 不得导入 frailty。
- **固定控制：** 所有 approved fields，包括 architecture、input order、sampling、windows/hops、normalization、padding/mask、feature schema hash、SQI/routing、loss/class weights/sampler、epoch rule、optimizer/LR/WD/dropout/label smoothing/clip、seeds、fold hash、aggregation、calibration metadata。
- **适用性：** 选择完成后的 research/deployment bundle。
- **执行并行：** 普通 single refit 一次，或按被选配置的显式 ensemble roster 逐成员 refit；永不把 outer-fold models 当 deployment members。
- **输出要求：** immutable resolved config、provenance/hashes、preprocessing objects、feature registry/schema、weights、model card、inference contract、bundle validation；ensemble bundle 列出全部 member IDs/seeds，不能写成一个 seed42 model。
- **晋级规则：** selected single/ensemble 在 all29 participants 上继承有效配置中的 seed/roster 并从头训练。具名 comparison 的五成员 roster 仍为 `[50042,60042,70042,80042,90042]`，但不是 core 限制。refit 不改变 CV ranking；通过 bundle 完整性检查、nonoverwrite atomic write 与 golden parity 前不 publication/deployment。
- **预计规模：** 无 outer cells；single 为1 full-data fit，ensemble 为配置 roster 大小个 fits。
- **当前状态：** 路径已实现；尚无人工 final selection 或正式 bundle，属 `DEFERRED_EVIDENCE`。

## 5. 阶段、规模与状态速查

本表仅用于预算，不能用来 pooling results。分开归档的 paired studies 中，重复 reference 按各 study 分别计数。conditional/deferred groups 不强行合并为一个总数。

| Group | 比较对象 | 目标 cases | 目标 5×5 cells | 当前状态 |
|---|---|---:|---:|---|
| 0A | raw CompactCNN 完整性 reference | 1 | 25 | `RUNNABLE_PATH` |
| 1A | epochs 7/10/15 | 3 | 75 | `RUNNABLE_PATH` |
| 1B | LR×weight-decay screen | 4 | 100 | `RUNNABLE_PATH`，探索性 |
| 1C | 仅 LR | 2 | 50 | `RUNNABLE_PATH`，ad hoc CLI |
| 1D | 仅 weight decay | 2 | 50 | `RUNNABLE_PATH`，ad hoc CLI |
| 1E | sampler 与 class weighting 单因素解耦 | 目标3（当前0） | 目标75（当前0） | runtime alternatives 已实现；`IMPLEMENTED_NEEDS_PLAN` |
| 2A | PPG 0.2–8 对 0.5–5 Hz | 2/model | 50/model | `CATALOG_ONLY` |
| 2B | EKF 对 LPF | 2/model | 50/model | `CATALOG_ONLY` |
| 2C | DL fs 100/160/200/400 | 4/model | 100/model | `CATALOG_ONLY` |
| 2D | context 5/10 s | 2/model | 50/model | `CATALOG_ONLY` |
| 2E | dilation 1/2 | 2/model | 50/model | `CATALOG_ONLY` |
| 3A | balance line B 对 A | 2/model | 50/model | `CATALOG_ONLY` |
| 3B | quality off 对 diagnostics-only | 2 | 50 | `IMPLEMENTED_NEEDS_PLAN` |
| 3C | SQI weighting/route/legacy aggregate | 当前0 | 当前0 | runtime modules 已实现；`IMPLEMENTED_NEEDS_PLAN` |
| 4A | raw single networks | 3 | 75 | reference 与 catalog-only 混合 |
| 4B | vector LR/SVM/ExtraTrees | 3 | 75 | reference 与 catalog-only 混合 |
| 4C/4D | feature-matrix model selection | 0 | 0 | model pending; ROCKET family retired |
| 4E | fusion Compact/Inception | 2 | 50 | reference 与 catalog-only 混合 |
| 4F | 四 representation finalists | 4 | 100 | 依赖前序结果 |
| 5A | faithful OSD 验收 | 1 | 25 | 已实现，待 plan/full 5×5 |
| 5B | OSD/effect-size route | 2 | 50 | 已登记，待 plan |
| 5C | effect-size length/stride 参数 | 目标3 | 目标75 | runtime 参数已实现；`IMPLEMENTED_NEEDS_PLAN` |
| 5D | channel-specific/joint discovery | 目标2 | 目标50 | 作为独立 registered model/discovery routes 可运行；matched plan 待建 |
| 6A | raw member0/ensemble | 目标2（当前 runnable 0） | 目标50（当前0） | `IMPLEMENTED_NEEDS_PLAN`；缺 typed paired plan |
| 6B | matrix member0/ensemble | 目标2（当前 runnable 0） | 目标50（当前0） | `IMPLEMENTED_NEEDS_PLAN`；缺 typed paired plan |
| 7A | identity + 六 reducers | 7 unique | 175 external eval | 首次仅实现初筛；等待8A–C与identity A3/A4后回返 |
| 7B | 三个 historical reducers | 目标3 | 目标75 | named modules 可执行；paired plan/结果未完成 |
| 7C | selected reducer 进入 vector/matrix | 2/representation | 50/representation | 仅在7-return人工 winner 后 |
| 8A | motion 8ch 验收 | 1 | 目标25 | 当前合同仅1×5 |
| 8B | motion 8ch/11ch | 2 | 目标50 | 5×5 不可运行 |
| 8C | motion EKF/LPF | 2/profile | 50/profile | 5×5 不可运行 |
| 8D | external PTT reference | 1 | 25 external eval | 8A–C 后先做 identity A3；科学证据未运行 |
| 8E | identity A4/selected reducer/override | identity baseline + conditional comparisons | override 当前0 | identity baseline 在7-return前；其余在winner后 |
| 9A | PRV backends/fixtures | 15 function calls | N/A | isolated diagnostic |
| 9B | 五个 leave-one-family-out pairs | 10 appearances/representation | 250/representation | 未冻结 |
| 9C | administrative metadata predictors | 0 | 0 | 禁止 |
| 10A | finalists | 1–4 | 25–100 | 依赖前序结果 |
| 11A | full-data refit | 1或5 fits | N/A | 人工选择后 |

表中 Phase 7/8 行必须按 `7-pre → 8A–C → identity A3/A4 → 7-return → 7C/8E downstream` 解释；不得按编号从 7A 一路线性执行到 8E。

## 6. 明确未完成的证据与外部条件

下列项目必须继续出现在计划和报告中；它们限制科学/部署结论，不授权或阻止普通 V2 模块执行。

| Decision | 未完成项目 | 尚缺内容 | 当前软件/结论边界 |
|---|---|---|---|
| V2-006 | device ADC rail、absolute scale、device-specific QC thresholds | device constants/evidence | 继续无需 device constants 的 physical rules；不虚构 rails |
| V2-009a/b/c | SQI weights/thresholds、route、quality-weighted prediction 的科学比较 | frozen effective config、outer-train provenance、完整 paired OOF | off/diagnostics/route 均可配置执行；不得声称未运行策略有效 |
| V2-010 | motion override evidence | complete internal 5×5 + external PTT comparison | motion evidence 与 core route 解耦；结论待运行 |
| V2-012 | final artifact reducer winner | complete A3/A4 matched comparison | identity default；reducers 只作 named comparisons |
| V2-026 | deployment hardware、power、end-to-end latency thresholds | target hardware 与 accepted limits | generic CPU cost 单独量测；不声称 deployment-ready |
| V2-027 | todo-only scope | future explicit authorization | 只记录，不执行 |
| ShapeFormer fixed 400/800 | 共同 stride/candidate-cap 可行性 | 精确的可执行 config | 保留 planned fixed-length，永不称 PISD |
| joint multichannel PIP | 独立 implementation/config/model card | 已登记的 named route | 不暴露为 PISD/fallback |
| motion formal 5×5 | 升级后的 motion training/study contract | 25-cell runner/provenance | 现有1×5 若运行也仅作 diagnostic |
| full PTT benchmark | internal matched evidence 与 unit/source identity | 完整前序证据 | 禁止 external fitting；未完整时不作 benchmark claim |
| feature-family ablations | 精确 family members 与 schema-hash policy | 已物化的 paired configs | 保持 complete reference registry |
| final refit/publication | manual selection + valid bundle | selected config、通过 bundle 完整性检查、nonoverwrite atomic write 与 golden parity | incomplete CV 不 refit/select |

## 7. 运行前检查与晋级检查表

开始任何 group 前：

1. 每 case 物化一个 complete config；profile catalog entry 不够。
2. 除非明确标 candidate comparison/exploratory grid，study 必须 `ablation` 且恰一 axis。
3. 先运行 CLI `--dry-run`，逐 case 查 resolved config、reference、varied table、controlled count、5 repeats、5 folds；invalid generated value 必须训练前失败。
4. 核对 manifest/fold hashes、representation/model ID、frailty 8ch order、normalization scope、balance line、seed policy、failure actions。
5. 确认 disk/memory 并选择 case-level jobs；deep/ensemble/ShapeFormer/motion 默认1。
6. 除显式同 plan `--resume` 外不得复用 output folder；每次 failed retry 必须独立 attempt directory。

任何 case/group 晋级前：

1. ranked frailty 或目标 formal motion case 的 planned/reported/passed 必须为 25/25/25。
2. required OOF hierarchy、ensemble member rows、hashes、route/coverage 必须存在；reporter 生成 N/A marker 不等于完成。
3. held-out participant 不得影响 preprocessing、shapelet discovery、sampling/class weights、threshold、calibration fitting、model selection。
4. 先看 participant BA，同时看 macro-F1 CI95 lower bound、worst-class F1/recall、calibration、coverage、stability、compute。
5. 使用相对 declared reference 的 paired participant/repeat evidence；绝不因单 fold/member/epoch 晋级。
6. 保存完整 dated folder/output index；人工 decision 单独记录，runner/report 不得静默提名 final model。

## 8. 本计划明确不做的事

- 不运行上述任何 study。
- 不因 catalog 登记就声称 13 candidates 均有 complete executable configs。
- 不把 optimizer grid 当 confirmatory one-factor evidence。
- 不把三个 derived motion channels 加入 final frailty model。
- 不把 effect-size/joint multichannel discovery 称为 PISD/OSD。
- 不恢复 SQI、motion、reducer、PTT、hardware 或 publication 的执行门禁；未运行证据只标 N/A/未验证。
- 不把 final-refit seed42 用于全部 outer-CV cells，也不把 split seeds 复用为 ensemble member seeds。

因此下一步始终保持小而明确：按顺序只物化/验证下一个 group，用一条人工命令运行，检查完整 dated report，记录人工晋级决定后再物化依赖组。
