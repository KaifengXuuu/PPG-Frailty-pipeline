# PROJECT_HANDOFF

状态：draft（包含第 0 节本会话 confirmed 补充；旧内容中与第 0 节冲突处，以第 0 节为准）  
最后手动更新时间：2026-06-10  
用途：给新的 Codex/AI agent 快速理解本项目目标、脚本演化、已完成组件、失败路线、未完成需求和下一步优先级。

## 0. 本会话纠正与补充：frailty3 三分类 pipeline

状态：confirmed  
来源：用户在本会话中明确要求整理并回复“同意录入”；结合本会话用户需求、代码检查、运行结果分析和脚本输出目录整理。  
最后手动更新时间：2026-06-10

### 0.1 当前项目主线纠正

| 项目 | 当前确认内容 |
|---|---|
| 当前主线 | 本会话主线是 `results_frailty3` 下的三分类 frailty-status pipeline：用 RED/IR 双通道 PPG + IMU 6 维信号区分 `Pre-Frail`、`Robust/Non-Frail`、`Young`。 |
| 数据来源 | `PPG_Testing_05_01_2026/StudyData_frailtyScored/StudyData_V7_standard.csv` 提供 `ID` 与 `FRAILTY-STATUS`；`PPG_Testing_05_01_2026/StudyData` 中受试者 ID 为文件名第一个 `_` 前字符串；`PPG_Testing_05_01_2026/TestDataYoungers` 全部视为 `Young`。 |
| 标签解释 | `FRAILTY-STATUS=2` 为 `Pre-Frail`；`FRAILTY-STATUS=3` 为 `Robust/Non-Frail`；`TestDataYoungers` 为 `Young`。 |
| 文件纳入规则 | `STE072` 已纠正，可以录入；只采用后缀/role 为 `B,R1,R2,R3,R4` 的文件。 |
| 主输入 | raw window 输入为 8 通道：RED、IR、AX、AY、AZ、GX、GY、GZ。深度模型输入张量形状为 `[N, 8, T]`。 |
| 手动特征 | 已增加 PPG peak detection，计算 PP intervals，并基于 PPI 计算 HRV/manual features。PPI/HRV 不是作为时序通道直接拼进 raw window，而是作为 fold 内标准化后的额外表格特征，经 MLP 与深度特征融合。 |
| 训练划分 | 主 CV 使用 `StratifiedGroupKFold`，按受试者分组，避免同一 subject 的 windows 同时出现在训练和验证中。 |
| 当前优先级 | 用户当前关注 `cnn` 和 `inceptiontime` 的最优分类表现，不把 runtime/cost/Pareto efficiency 作为 leaderboard 排名依据；`shapeformer` 暂时保留接口和实验记录。 |

### 0.2 本会话脚本总览

| 脚本 | 当前定位 | 详细程度 |
|---|---|---|
| `frailty_3class_classifier.py` | 当前主训练、交叉验证、模型 sweep 脚本。 | 本会话核心脚本，已多轮更新。 |
| `shapeformer_port.py` | ShapeFormer/ShapeFormer-PISD 移植模块，被主训练脚本调用。 | 本会话核心移植模块。 |
| `analyze_sweep.py` | sweep 输出结果分析脚本，当前只分析 `cnn` 与 `inceptiontime`，保留未来 shapeformer 接口。 | 本会话核心后处理脚本。 |
| `frailty_3class_holdout_eval.py` | 对 sweep top config 做独立 holdout 复核的脚本。 | 本会话核心验证脚本。 |
| `frailty_3class_cnn_fusion.py` | 早期 CNN/manual feature fusion 新脚本。 | 已被主脚本的 `extra_input=PPI/HRV` 功能部分吸收，保留为参考/旁支。 |

### 0.3 核心脚本详细交接表

| 脚本 | 初始需求 | 新增需求 | 已完成实现 | 算法/接口 | 用户评价与当前结论 | 未完成/风险 | 更新历史与原因 |
|---|---|---|---|---|---|---|---|
| `frailty_3class_classifier.py` | 读取 StudyData label 和 Young 数据；基于 RED/IR PPG + IMU6 做 3-class frailty-status 分类；根据样本量和差异性推荐模型。 | 加入 PPG peak detection、PPI、HRV/manual features；加入 1D-CNN；按 subject 分 train/validation，使用 `StratifiedGroupKFold`；只纳入 `B,R1,R2,R3,R4`；纳入 `STE072`；多跑模型并输出评估报告；移植 InceptionTime、ShapeFormer、ShapeFormer-PISD；绘制学习曲线；添加自动 sweep；优化进度条和日志；压制 windowing log 与 PISD verbose log。 | 支持 `cnn`、`inceptiontime`、`shapeformer`、`shapeformer_pisd` 以及若干 classical baseline；支持 `extra_input=0/PPI/HRV`；支持 CV report、subject/file/window 指标、混淆矩阵、学习曲线、模型保存、sweep manifest、逐 run 追加 CSV；支持按相同 window 参数聚合运行以减少重复 window 构建；支持总/子进度条。 | raw 输入为 `[N,8,T]`；PPI/HRV 作为额外 feature vector，经 fold 内 scaler 和 MLP fusion；深度模型使用 train fold 训练、validation fold early stopping；CV 使用 subject group。`shapeformer_pisd` 使用原版 PISD discovery 包装，耗时明显增加。 | 用户认为大规模 sweep 有必要，但预估时间过长；当前正式关注 `cnn` 和 `inceptiontime` 最优分类表现，shapeformer 暂不作为主分析对象。 | 样本量小，fold/repeat 方差大；Pre-Frail vs Robust/Non-Frail 仍混淆明显；overlap windows 会增加样本相关性；shapeformer_pisd discovery 耗时大；最终部署模型需要在选定 config 后重新训练。 | 从单一 CNN 分类脚本逐步扩展为统一训练和 sweep 框架；原因是用户不断要求比较多模型、多输入、多窗口长度、多 epoch/patience，并排除偶然性。 |
| `shapeformer_port.py` | 将 `/home/trinker/Code/github/multivariate-time-series-analysis/ShapeFormer` 移植到当前 pipeline。 | 检查是否完整移植原版结构；解释 `PISD discovery`、`gstride`、CPU/GPU 使用；添加 `shapeformer` 与 `shapeformer_pisd` 分支；压制 extract/discovery log。 | 实现 `ShapeBlock`、`PortedShapeFormer`、`discover_shapelets`、`discover_shapelets_pisd`；`shapeformer_pisd` 调用原版 `ShapeletDiscover`；`PortedShapeFormer.forward_features()` 支持与 PPI/HRV fusion 对接。 | 当前不是原仓库 bitwise/source-identical 原样拷贝，而是核心结构移植 + PPG pipeline 适配。`shapeformer` 使用较快的 effect-size discovery；`shapeformer_pisd` 使用原版 PISD discovery wrapper。原版 `gstride` 是 gait/frailty 数据接口，不适配当前 PPG 文件结构。 | 用户关注算法移植完整性与结果不提升原因；当前结论是结构已尽量对齐核心思想，但原版库不能无改动原样套用，需要数据接口、shapelet discovery、输入维度、batch/device、输出 head 等适配。 | PISD discovery CPU-heavy，导致训练时间大幅增加；小样本下 shapelet discovery 易受 fold subject 组成影响；原版 `gstride` 数据不存在/不适配；完整原仓库训练管线无法无需改动迁移。 | 最初移植 InceptionTime 后继续迁移 ShapeFormer；后来因用户发现某配置结果更高，继续检查 learning curve、loss 差异、early stopping 和 PISD log。 |
| `analyze_sweep.py` | 新建 sweep 输出分析脚本，输入如 `results_frailty3/20260527_1320_cnn_inceptionTime`，输出到 `results_frailty3/_sweep_analyse`，每次新建带日期时间和模型名的子文件夹。 | 当前只分析 `model in ["cnn","inceptiontime"]`；保留未来 shapeformer 接口；leaderboard 必须按参数组 ranking 而非单 run；runtime 只能作参考，不影响排名；添加 completeness check、class-level summary、confusion matrix aggregation；图表添加 top10 的 `worst_class_f1_mean` 和 std/stability 重排序。 | 输出 `clean_runs.csv`、`config_summary.csv`、`leaderboard_top_configs.csv`、`incomplete_configs.csv`、`class_level_summary.csv`、`top_config_confusion_matrices_long.csv`、`analysis_report.md`、figures、top confusion matrices 和 top learning curve png。 | config columns 固定为 `model,resolved_model,extra_input,cnn_epochs,cnn_patience,window_sec,hop_sec,overlap_pct,max_windows_fraction`；`seed` 与 `repeat` 不属于 config。主排序：`subject_balanced_accuracy_mean`；次排序：`subject_macro_f1_mean`；tie-breakers：`subject_balanced_accuracy_ci95_low`、`subject_macro_f1_ci95_low`、`worst_class_recall_mean`、`worst_class_f1_mean`、`subject_balanced_accuracy_std` ascending。 | 用户明确要求当前分析目标只看 cnn 和 inceptiontime 的最优分类表现，不考虑 runtime/cost/Pareto。分析报告用于选择 rank1/rank2/rank7 进入 holdout。 | shapeformer 分析接口保留但未作为当前默认目标；如果后续要纳入，需要确认排名指标是否仍同一套，且 runtime 不参与排序。 | 起因是 sweep 结果多、单 run 排名不公平，因此改为 groupby config 后聚合 repeats，再选 top-k configs。 |
| `frailty_3class_holdout_eval.py` | copy 一份 sweep 脚本，解锁全数据集训练和验证，方便与原 sweep 并排跑；对 `analyze_sweep.py` 的 rank1、rank2、rank7 各跑 5 repeats。 | 用户质疑 20% holdout 同时作为 early stopping validation 是否泄露；随后要求改为 `train:64% / inner validation:16% / test:20%`。 | 按 subject stratified split 生成 train/inner-val/test；默认读取最新 `_sweep_analyse` 的 `leaderboard_top_configs.csv`；默认 `--ranks 1,2,7 --repeats 5`；输出 holdout manifest、per-run report、learning curves、confusion matrices、`holdout_runs.csv` 和 `holdout_summary.csv`。 | inner validation 用于 early stopping；test 仅最终评估。未加前缀指标为最终 test 指标；`validation_*` 为 inner validation。 | 当前正式 holdout 结果较 sweep 低，属预期：严格 test 不参与 early stopping，且样本量小导致 split 方差大。最终建议按 config 的 mean/std/CI 选择，不选单次最好 repeat。 | 每个 repeat 的 test subject 可能不同，结果方差大；测试集约 6 个 subject，单个 subject 可显著影响分数；仍需外部数据或更多 subject 才能稳定。 | 从“holdout 同时当 validation”的初版改成严格三分法，原因是用户明确提出避免 test feedback/data leakage。 |
| `frailty_3class_cnn_fusion.py` | 用户问 CNN 能否采用手动特征，并要求用一个新脚本更新算法、给运行命令和试运行。 | 加入 handcrafted PPI/HRV/manual features 与 CNN/InceptionTime 深度特征融合。 | 实现 `cnn1d_fusion` 与 `inception_time_fusion` 风格的 raw window + feature MLP fusion；支持 subject-level split 和评估。 | 与主脚本后来的 `extra_input=PPI/HRV` 思路一致，但不是当前主入口。 | 保留为早期探索/参考脚本；当前统一训练、sweep 和后处理应优先使用 `frailty_3class_classifier.py` + `analyze_sweep.py` + `frailty_3class_holdout_eval.py`。 | 可能与主脚本存在功能重复；后续如继续维护，应避免两套 fusion 逻辑漂移。 | 该脚本是用户要求“新脚本”阶段的产物，后来主脚本吸收其核心功能。 |

### 0.4 主要实验结论与解释

| 主题 | 当前结论 |
|---|---|
| 为什么限制 windows/file | 早期限制如 `6 windows/file` 是为了 smoke test、节省时间、避免长文件贡献过多高度相关 windows；不是通用准则。后续 sweep 已改为按文件可切最大 windows 的百分比控制，例如 `90%` 或 `50%`。 |
| 为什么采样率处理不同 | 原始文件可能名义采样率或有效时间戳不同，脚本需要从数据/时间戳推断或重采样到统一建模长度。该点应继续核查每批原始数据的时间轴和采样率字段。 |
| early stopping vs inner validation | early stopping 是训练过程中的停止/模型选择机制；inner validation 是用于 early stopping 的数据子集。当前严格 holdout 脚本使用 inner validation early stopping，test 不参与训练反馈。 |
| 是否去掉 early stopping 看完整学习曲线 | 若目标是观察完整训练动态，可以去掉 early stopping 或设置更大 patience；若目标是模型选择和泛化，仍应保留 validation-based early stopping。 |
| epoch 为什么曾设为 8 | `8` 是早期快速试跑/排错选择，不是通用准则。实际应用应设较高上限并用 validation curve/early stopping 决定。 |
| 学习曲线淡红线 | 代表不同 fold 的验证 loss/metric 曲线。它们上下排列通常来自 fold 难度、subject 组成、类别分布和样本差异，不是时间顺序导致。fold 顺序只是枚举，训练不会从前一 fold 继承权重。 |
| 哪根学习曲线最代表模型能力 | 单根 fold 曲线不能代表整体能力；应看跨 fold/repeat 的平均与方差、CI、worst-class 指标和 subject-level 指标。 |
| 按平均 loss early stop 是否科学 | 可以作为常见做法，但本项目分类目标更适合关注 validation balanced accuracy、macro F1、worst-class recall/F1，尤其需防止只把 Young 分好。 |
| 提高 epoch 与提高 repeat 的区别 | epoch 是同一个模型训练更久；repeat 是同一参数组用不同 seed/split/init 重新训练，估计稳定性和偶然性。 |
| 最终模型如何从 CV 得到 | CV 的 5 个 fold 模型主要用于估计与选 config；最终部署应在选定 config 后用训练数据重新训练一个最终模型。不要从 5 个 fold 里挑最高分当最终模型。 |
| 是否选平均成绩或最好 repeat | 正式结果应按参数组 mean/std/CI 选择；最好 repeat 只能作为 best-case 辅助说明，不能作为最终模型能力估计。 |
| inner validation 是否让结果悲观 | 会减少训练集规模，可能使小样本结果更低；但它避免 test 泄露。若样本很少，可在最终 config 确定后重新训练部署模型，并把 holdout/CV 分数作为泛化估计。 |
| 当前主要瓶颈 | 不是 Young，而是 `Pre-Frail` 与 `Robust/Non-Frail` 之间混淆；subject 数少导致 fold/repeat 波动大。 |

### 0.5 已分析的重要结果

| 输出目录/报表 | 结论 |
|---|---|
| `results_frailty3/_sweep_analyse/20260601_0941_cnn_inceptiontime` | 分析了 360 runs、72 config groups，所有配置 complete。Top 10 主要由 InceptionTime/raw/5s 参数组成。 |
| sweep rank 1 | InceptionTime，raw extra input，5s window，30% overlap，patience 20。 |
| sweep rank 2 | InceptionTime，raw extra input，5s window，50% overlap，patience 20。 |
| sweep rank 7 | CNN，raw extra input，5s window，50% overlap，patience 20。 |
| `results_frailty3/_holdout_eval/20260607_0935_rank1-2-7_holdout` | 15 runs，3 ranks x 5 repeats complete；严格 train/inner-val/test 后分数低于 sweep，说明之前 CV/sweep 分数不能直接当最终泛化表现。 |
| holdout rank 1 | test balanced accuracy mean 约 0.600，macro F1 约 0.547，worst-class F1 约 0.133。 |
| holdout rank 2 | test balanced accuracy mean 约 0.600，macro F1 约 0.580，worst-class F1 约 0.380。当前更推荐 rank 2，因为更平衡。 |
| holdout rank 7 | test balanced accuracy mean 约 0.567，macro F1 约 0.513，worst-class F1 约 0.180。 |
| rank 2 aggregated subject confusion | true Pre-Frail: 6 Pre / 2 Robust / 2 Young；true Robust: 4 Pre / 4 Robust / 2 Young；true Young: 0 Pre / 2 Robust / 8 Young。主要混淆仍为 Pre-Frail vs Robust/Non-Frail。 |

### 0.6 当前已实现组件

| 组件 | 状态 |
|---|---|
| 数据读取与 label mapping | 已实现。 |
| 文件 role 过滤 `B,R1,R2,R3,R4` | 已实现。 |
| `STE072` 纳入 | 已实现。 |
| PPG peak detection、PPI、HRV/manual features | 已实现。 |
| 1D-CNN | 已实现。 |
| InceptionTime | 已移植/实现。 |
| ShapeFormer core port | 已实现，但不是原仓库无改动原样运行。 |
| ShapeFormer-PISD original discovery wrapper | 已实现，耗时大。 |
| PPI/HRV feature fusion | 已实现。 |
| Subject-level StratifiedGroupKFold | 已实现。 |
| 学习曲线 plot | 已实现。 |
| sweep 自动循环 | 已实现。 |
| sweep 逐 run CSV/report 追加 | 已实现。 |
| sweep 后处理与 leaderboard | 已实现。 |
| top config holdout 三分法复核 | 已实现。 |
| W&B | 未使用；当前采用本地 CSV/JSON/PNG/Markdown 可视化和分析。 |

### 0.7 当前未完成需求与建议给下一个 Codex 的任务

| 优先级 | 任务 | 原因 |
|---|---|---|
| 高 | 复查 `frailty_3class_classifier.py` 与 `frailty_3class_holdout_eval.py` 的最终命令行参数和默认值，确认是否与用户最新实验设计一致。 | sweep 默认参数和 holdout 默认参数经历多轮更新，继续实验前应避免旧默认值误用。 |
| 高 | 对 rank 2 或其他选定 config 做更稳定的 final training/export 方案。 | CV/holdout 用于估计；部署需要重新训练最终模型，并保存 scaler、label map、window 参数、feature schema。 |
| 高 | 针对 Pre-Frail vs Robust/Non-Frail 混淆做定向诊断。 | 当前最大错误来源不是 Young，而是两个老年组之间边界不清。 |
| 中 | 决定 PPI/HRV/manual features 是否保留在最终候选中，需基于 group-level sweep 而非单 run 判断。 | 额外特征理论上有意义，但小样本下可能增加方差或过拟合。 |
| 中 | 若继续 ShapeFormer，单独做小范围 ablation，而不是放入超大 sweep。 | `shapeformer_pisd` 运行时间长，且当前提升不明显；需要先证明必要性。 |
| 中 | 增加 subject-level calibration、per-class threshold 或 cost-sensitive objective 的实验。 | 可缓解 class imbalance 和 worst-class 指标差。 |
| 中 | 检查采样率/时间戳处理并写入明确数据规范。 | 用户曾质疑不同原数据 400 Hz 采样率，需避免隐含重采样错误。 |
| 低 | 评估是否接入 W&B 或继续使用本地报表。 | 当前未用 W&B；本地 CSV/PNG 已够用，但大型 sweep 可视化会更方便。 |

### 0.8 给新 Codex 的建议阅读顺序

| 顺序 | 文件/目录 | 目的 |
|---|---|---|
| 1 | `AGENTS.md`、`_agent/WRITE_RULES.md` | 先理解项目记录写入规则，尤其 `_agent` 写入必须先给用户审核并获“确认录入/同意录入”。 |
| 2 | 本文件第 0 节 | 获取当前 frailty3 分类主线，纠正旧交接内容中的误标注。 |
| 3 | `frailty_3class_classifier.py` | 理解主训练、CV、sweep、模型分支和 feature fusion。 |
| 4 | `shapeformer_port.py` | 理解 ShapeFormer/ShapeFormer-PISD 移植边界和耗时来源。 |
| 5 | `analyze_sweep.py` | 理解 sweep 后处理、leaderboard 排名逻辑和 top-k config 聚合。 |
| 6 | `frailty_3class_holdout_eval.py` | 理解最终候选 config 的 strict holdout 复核。 |
| 7 | `results_frailty3/_sweep_analyse/` 与 `results_frailty3/_holdout_eval/` | 查看实际输出结果、混淆矩阵、学习曲线和 summary CSV。 |

### 0.9 对旧内容的明确纠正

| 旧内容问题 | 纠正 |
|---|---|
| 旧第 5 节把 `frailty_3class_classifier.py`、`frailty_3class_cnn_fusion.py`、`frailty_3class_holdout_eval.py` 标为“非本会话，需补充代码审阅”。 | 对当前会话而言这些是已多轮实现/审阅/分析的核心脚本或旁支脚本，应按第 0 节理解。 |
| 旧内容主要围绕 denoiser、peak-gating、ASA classifier。 | 这些可作为项目其他路线历史，但不是本会话 frailty3 三分类主线。 |
| 旧内容中“最终 frailty classifier 应融合 IMU 静/动态、动态心搏、静态波形特征”属于更长期项目设想。 | 当前已落地的是 raw RED/IR/IMU windows + 可选 PPI/HRV/manual feature fusion 的三分类模型；尚未把旧动态 peak/IBI 模型完整并入最终 frailty classifier。 |

### 0.10 一句话交接摘要

本会话已把 frailty3 从单一 CNN 发展成包含 CNN、InceptionTime、ShapeFormer/ShapeFormer-PISD、PPI/HRV feature fusion、subject-level CV、自动 sweep、sweep 后处理和 strict holdout 复核的一套本地实验 pipeline；当前最可信候选是 sweep rank 2 的 InceptionTime raw 5s/50% overlap/patience20，但最终问题仍是小样本下 Pre-Frail 与 Robust/Non-Frail 混淆和 repeat/fold 方差较大。

### 0.11 2026-06-10 overfitting sweep 最新进度

状态：confirmed  
来源：用户连续要求修改 overfitting sweep；结合 `frailty_3class_overfitting_sweep.py` 代码检查、dry-run/smoke test、以及 `results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2` 输出分析。  
最后手动更新时间：2026-06-10

#### 0.11.1 当前 overfitting sweep 脚本状态

| 项目 | 当前确认内容 |
|---|---|
| 相关脚本 | `frailty_3class_overfitting_sweep.py` |
| 当前输出根目录 | `results_frailty3/_overfitting_sweep` |
| reference | 已由旧 `ref_original` 改为 `ref_rank2_fixed_epoch`。含义：保留 rank2 原始参数，但关闭 early stopping，即 `cnn_patience=0`、`cnn_select_best_epoch=False`。 |
| 评估协议 | 所有 stages 均改为 `5-fold StratifiedGroupKFold`，按 subject 分组；不再使用 holdout 作为 stage1/stage2 主评估协议。 |
| tuned configs | stage1/stage2 tuned configs 均为 no early stopping/fixed final epoch。 |
| 输出记录 | `overfitting_runs.csv` 额外写入 `eval_protocol`、`requested_cv_folds`、`n_splits`、`early_stopping_source`，便于确认协议。 |
| stage1 epoch grid | `epoch=[5,8,10,12,15]`。 |
| stage1 正则主效应筛选 | stage1 已扩展为主效应筛选：weight decay、dropout、label smoothing、max_windows_fraction 单因素变化，加少量 strong regularization combo。 |
| stage1 主效应字段 | 新增 `stage1_screen_group`、`stage1_regularization_factor`、`stage1_regularization_value`，便于后处理按正则因子聚合。 |

#### 0.11.2 最新 step1 输出分析

| 项目 | 内容 |
|---|---|
| 最新 step1 目录 | `results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2` |
| 完整性 | 完整。185 个 stage1 configs × 5 repeats + `ref_rank2_fixed_epoch` × 5 repeats，共 930 runs。 |
| 协议确认 | 全部为 `stratified_group_kfold`，`n_splits=5`，`early_stopping_source=none_final_epoch_fixed`。 |
| reference 表现 | `ref_rank2_fixed_epoch`: subject balanced accuracy mean 约 0.585，macro F1 约 0.589，worst-class F1 约 0.493。 |
| Top1 | `s1_085`: dropout=0, epoch=10，subject balanced accuracy mean 约 0.623，macro F1 约 0.626，worst-class F1 约 0.526。 |
| Top2 | `s1_091`: dropout=0.7, epoch=10，subject balanced accuracy mean 约 0.621，macro F1 约 0.622，worst-class F1 约 0.530。 |
| Top strong combo | `s1_105`: wd=0.005, dropout=0.5, label_smoothing=0.2, epoch=10，subject balanced accuracy mean 约 0.616，macro F1 约 0.625，worst-class F1 约 0.541。 |
| 最稳定/CI-low 代表 | `s1_163`: dropout=0.5, epoch=15，subject balanced accuracy mean 约 0.612，std 约 0.031，worst-class F1 约 0.557。 |
| 统计解释 | Top 配置相对 reference 有改善趋势，但没有任何 config 的 95% CI low 高于 reference mean，因此不能说显著稳定超过 reference。 |
| 重要 caveat | `ref_rank2_fixed_epoch` 保留原始 epoch=50，而 stage1 在 5/8/10/12/15 中筛选；因此部分 improvement 可能来自 shorter fixed epoch，而不完全是正则化本身。 |
| baseline 对照 | baseline epoch=10 (`s1_075`) subject balanced accuracy mean 约 0.605，高于 reference epoch=50 的约 0.585，提示 epoch=50 no early stopping 存在明显过训练。 |
| 主要错误 | Pre-Frail vs Robust/Non-Frail 混淆仍是核心问题；Top configs 并没有根本解决老年组内部混淆。 |
| 过拟合 | Top configs 训练 balanced accuracy 接近 0.99，validation balanced accuracy 约 0.56，train-val gap 仍约 0.42-0.44；强正则降低 loss gap，但分类边界提升有限。 |

#### 0.11.3 当前主效应结论

| 因子 | 当前观察 |
|---|---|
| epoch | 10 和 15 最值得保留；Top 单点在 epoch=10，平均表现 epoch=15 略高。 |
| dropout | 无单调规律。dropout=0 和 dropout=0.7 都进入 Top2；dropout=0.5 + epoch15 稳定性较好。 |
| weight decay | 无明显单调趋势；1e-4、1e-3、2e-2 都有可用结果，但不是决定性主效应。 |
| label smoothing | 单独主效应较弱；0.2 略好，但提升不明显。 |
| max_windows_fraction | 0.7 比 0.5/0.3/0.2 更有希望；太低会损失信息。 |
| strong combo | 有助于降低 loss gap；`s1_105` 是当前最好的 strong regularization combo。 |

#### 0.11.4 下一步建议

| 优先级 | 建议 |
|---|---|
| 高 | stage2 不建议只用自动 Top2，因为 `s1_085` 和 `s1_091` 都主要来自 dropout 主效应；建议使用 Top4：`s1_085`, `s1_091`, `s1_105`, `s1_163`。 |
| 高 | stage2 source 建议使用 `results_frailty3/_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2`。 |
| 高 | 如果继续当前脚本自动逻辑，`--stage2-top-n 2` 会选择 `s1_085` 和 `s1_091`，fixed epoch 会从 Top2 中更稳定者推导，通常为 epoch=10。 |
| 中 | 若希望兼顾稳定性和类别均衡，建议 `--stage2-top-n 4`，让 `s1_105` 和 `s1_163` 进入 stage2。 |
| 中 | 后续分析应避免只看 mean BA，应同时看 macro F1、worst-class F1、CI low、std，以及 Pre-Frail vs Robust 混淆矩阵。 |
| 中 | 需要明确最终报告中是否把 `ref_rank2_fixed_epoch` 作为唯一 reference，或额外加入 shorter-epoch baseline reference，以区分 epoch effect 与 regularization effect。 |

## 1. 项目目标与主线

| 层级 | 说明 |
|---|---|
| 最终目标 | 从 PPG 信号及 IMU 辅助信息中提取可靠心搏、HR/HRV、静/动态状态、波形特征，并用于 frailty 分类。 |
| 当前主线 | 静态段做 PPG 波形分析和特征提取；动态段先用 detector 判断运动状态，再用动态心搏提取模型获取可靠 peak、IBI、HR/HRV；最终将 IMU 静/动态状态、动态心搏信息、静态波形特征融合为 frailty classifier。 |
| 关键结论 | 早期 denoiser 路线泛化失败，不适合作为最终动态 PPG 去噪方案；但其中 motion/static gating 思路有价值，后续转向独立 motion detector 和端到端 peak/HR 模型。 |
| 模型定位 | 端到端一维神经网络只是“动态心搏信息提取模型”，不是最终 frailty classifier。 |
| 用户偏好 | 优先成功率、可解释性、泛化能力、持续迭代能力，不追求最小改动；所有重要实验需要 scorecard、图表、分层评估和可复现输出。 |

## 2. 本会话核心脚本详细交接

| 脚本 | 当前定位 | 初始需求 | 新增需求 | 已完成 | 未完成/风险 | 更新历史与原因 |
|---|---|---|---|---|---|---|
| `ppg_analyse4_calib.ipynb` | 当前主分析 notebook，取代旧 `ppg.py` | 在主 pipeline 中复用 detector/denoiser 模型，实现预处理、分类、降噪和 plot 预览。 | 用户后来要求纯 CPU、conda `ppg` 环境、尽量不用 Torch；考虑 ONNX runtime；再后来主线从 denoiser 转向 detector + dynamic heartbeat extractor。 | 曾加入 denoiser 复用、compare plot、ONNX runtime 方向。 | 需要继续整合新版 `ppg_peak_hr_gating_train.py` 输出的动态 peak/IBI/HR 模型；旧 denoiser 不应作为可靠动态修复模块。 | 初始是 denoiser 部署入口；因 denoiser 动态段失败，应改为主 pipeline 中的 motion detector + dynamic heartbeat extractor + 静态波形分析入口。 |
| `pttppg_denoiser_hybrid_train.py` | 早期 hybrid denoiser 训练脚本，现为 deprecated/experimental | 训练 motion artifact denoiser，保存参数供主脚本复用。 | 加进度条、ETA；支持 A/B：A=`raw+IMU`，B=`raw+IMU+linear baseline`；尝试 sit prior、peak prior、clean prior、artifact relation 学习。 | 支持模型 bundle 输出；支持 `raw_imu` 和 `raw_imu_baseline`；有进度和时间估计。 | 动态段泛化失败；不建议继续作为主去噪方向。 | 初始假设 IMU 可帮助估计 motion artifact；用户指出 IMU 与 artifact 的关系不能被预设为 teacher，关系本身才是需要学习的答案；后续结果证明该路线失败。 |
| `pttppg_denoiser_hybrid_preview.py` | denoiser 预览脚本，现为旧模型可视化工具 | 加载训练好的 denoiser，输出去噪前后 PPG plot。 | preview plot 不输出到根目录；自动写入 `denoiser_preview_output/`；文件名为 csv 文件名 + 日期 + 小时 + 分钟；plot 加低通/带通预处理显示。 | 支持去噪前后 plot、预处理 plot、自动命名输出。 | 只适合人工诊断旧 denoiser，不应作为最终评价依据。 | preview 暴露了 denoiser 动态段失败：动态信号没有周期规律，B 出双倍伪峰，峰谷错位。 |
| `pttppg_denoiser_hybrid_ab_compare.py` | denoiser A/B 对比脚本，现为旧路线诊断工具 | 同一批 preview 上比较 A=`raw+IMU` 与 B=`raw+IMU+linear baseline`。 | 将 compare plot 和模型复现模块放入主 notebook；支持对同一批样本做人工比较。 | 能读取两个模型并生成对比图。 | 缺少严格 quantitative score；只适合诊断，不适合作为最终算法依据。 | 用户观察 A 基本接近原始信号，B 出双倍伪峰且最高峰位置可能变峰谷，因此 A/B denoiser 都不适合主线。 |
| `pttppg_denoiser_hybrid_core.py` | hybrid denoiser 公共核心，现为 archive/reference | 抽出训练、preview、ONNX 复用公共逻辑。 | 支持不同 input mode、模型加载、特征构建和推理复用。 | 提供 denoiser 网络、特征构建、bundle 加载等公共代码。 | 后续新 peak/HR/detector 模型不应强依赖此模块。 | 因 denoiser 路线失败，保留为历史实验基础代码和 ONNX 经验参考。 |
| `pttppg_denoiser_hybrid_export_onnx.py` | denoiser Torch 到 ONNX 导出脚本，现为部署经验参考 | 用户要求部署端 CPU-only，避免 Torch。 | 导出现有 `results_hybrid_denoiser_raw_imu` 和 `results_hybrid_denoiser_raw_imu_baseline` 中的模型。 | 已支持旧 denoiser ONNX 导出。 | 对 denoiser 主线意义有限；但 ONNX 导出经验可迁移到新 detector/peak/IBI 模型。 | 起因是用户质疑“复用模型只涉及纯参数计算，为什么要用 Torch”，因此切到 ONNX runtime 思路。 |
| `pttppg_denoiser_onnx_runtime.py` | denoiser ONNX runtime 推理模块，现为部署经验参考 | 在主脚本 CPU-only 复用模型，不依赖 Torch。 | 用 ONNX runtime 加载旧 denoiser 模型。 | 满足旧 denoiser 的无 Torch 推理方向。 | 新 peak/HR/detector 模型仍需单独 ONNX/轻量 runtime。 | 作为 CPU-only 部署路线样板保留。 |
| `pttppg_pipeline_v7_4_noleak_viz_ae.py` | 旧 denoiser/AE 思路来源脚本，现为历史参考 | 用户最初希望以其思路训练 denoiser。 | 后续围绕 IMU reference、delay、sit prior、peak prior、artifact relation 做过多轮算法讨论。 | 提供 no-leak、AE/denoising、PTT-PPG pipeline 思路参考。 | 不再作为主路线；需要防止新 Codex 误以为它是当前核心。 | 因用户确认 motion artifact clean target 不可得、IMU 关系不可预设、部署端无可靠 motion peaks，该路线被降级。 |
| `ppg_peak_hr_gating_train.py` | 当前最重要的动态心搏提取训练脚本 | 新建端到端神经网络：PPG 原信号输入，同步 ECG 信号监督，输出准确 peaks 和 HR intervals；从 denoiser gating head 思路保留 detector 价值，去掉 artifact refiner。 | 加 ECG detector preflight；解析 atr/aux；beat-level peak 监督和评估；IBI Huber + 生理范围约束；dataset/activity-balanced training；domain-aware augmentation；instance norm/adversarial domain generalization；per-dataset/per-subject/per-activity scorecard；PPG-ECG delay analysis；LODO；GroupDRO/worst-domain；motion detector A/B benchmark；PPG 主指标改为 ±20 ms，并分层报告 ±10/20/30/40 ms。 | 支持 PTT、MIMIC、iAMwell、simultaneous、VitalDB 等多数据集；输出 cross-validation、holdout、extra-holdout、peak sequence、HR interval sequence、gate logit、detector benchmark、scorecard、曲线图、混淆矩阵、模型复用参数。 | 需要正式全量训练和系统评估新版 scorecard；需要决定最终部署模型是 peak/IBI 主模型、IMU motion detector，还是二者组合；需要严格外部验证。 | 这是从“恢复 clean waveform”转向“直接提取可靠心搏和 IBI”的主线转折脚本。原因是 dynamic denoising 失败，但 frailty pipeline 真正需要的是可靠 HR/HRV 中间参数。 |
| `pttppg_detector_v8_scores.py` / `pttppg_detector_v8_scores_audit_fix9.py` | 旧 detector 评分/审计脚本 | 早期用于静/动态 detector 的分数审计和修正。 | 用户后来要求检查旧 gate 是否偏 motion、threshold 是否导致 motion 容错小、验证端预测分布是否混乱。 | 可作为旧 detector baseline/audit 参考。 | 需要与新版 motion detector A/B benchmark 对照后决定是否废弃。 | 用户认为 denoiser/gating 的静止段识别更鲁棒，因此旧 detector 不再默认是最终方案。 |

## 3. 本会话出现但未重点审阅的脚本

| 脚本 | 标注 | 说明 |
|---|---|---|
| `cnnppg_v7.py` | 本会话出现，未重点审阅 | 可能是旧 CNN PPG 模型参考。若继续使用，需要重新检查输入、输出、split 和是否适配新 pipeline。 |
| `svm2_dataset_train.py` | 本会话出现，未重点审阅 | 可能是传统 SVM/tabular baseline。若继续使用，需要确认 label、split、特征是否无泄漏。 |
| `ppg.py` | 已过时 | 用户明确指出 `ppg.py` 已经过时，最新主脚本是 `ppg_analyse4_calib.ipynb`。除非抽取旧函数，否则不建议继续作为入口。 |

## 4. 旁支实验：ASA 相关

| 条目 | 标注 | 说明 |
|---|---|---|
| `asa_classifier.py` | 非本会话主线，且与本项目 frailty pipeline 无直接关系，仅作为模型试验/方法验证 | VitalDB ASA 1/2/3 三分类实验。不能当作项目主线，也不能直接等同 frailty 分类。 |
| `test_asa_classifier/` | 非本会话主线，模型试验输出 | 保存 ASA classifier 的 scorecard、summary、模型、图表、预测 CSV。 |
| VitalDB ASA 分布检查 | 非本会话主线，模型试验准备 | 曾检查 VitalDB 中 ASA、PLETH、ECG_II 分布。该内容只服务 ASA 模型试验。 |

ASA 脚本当前理解：只用 VitalDB 中同时包含 ASA、PLETH、ECG_II 的数据，删除 ASA 4/6/NaN，对 ASA 1/2/3 做三分类；支持 PPG-only、ECG-only、ECG-peaks-only 三个输入；使用 subject-level split、StratifiedGroupKFold、class weighting、fold 内 normalization 防泄漏。该内容仅作为方法试验，不纳入 frailty pipeline 主线结论。

## 5. 非本会话或未充分审阅内容

| 条目 | 标注 | 下一步 |
|---|---|---|
| `frailty_3class_classifier.py` | 非本会话，需补充代码审阅 | 检查输入特征、label 定义、subject-level split、是否无泄漏、最新结果。 |
| `frailty_3class_cnn_fusion.py` | 非本会话，需补充代码审阅 | 判断是否为最终 fusion 模型，以及是否接入 detector/peak/IBI 输出。 |
| `frailty_3class_holdout_eval.py` | 非本会话，需补充代码审阅 | 检查 holdout 评估是否严格独立、指标是否与项目目标一致。 |
| `results_frailty3/` | 非本会话，需补充结果审阅 | 读取最新 scorecard/summary，判断当前 frailty 分类进展。 |

## 6. 关键输出目录

| 目录 | 内容 | 用途 |
|---|---|---|
| `.CNN_results/` | `ppg_peak_hr_gating_train.py` 的训练结果、scorecard、模型、图表 | 动态 peak/IBI/gate/detector 主结果目录。 |
| `.CNN_results/20260427-01_smoke_detector_ab/` | detector A/B smoke run | 证明新版 detector benchmark 输出链路正常。 |
| `results_hybrid_denoiser_raw_imu/` | denoiser A 模型 | 旧 denoiser 对照。 |
| `results_hybrid_denoiser_raw_imu_baseline/` | denoiser B 模型 | 旧 denoiser 对照。 |
| `denoiser_preview_output/` | denoiser preview 图 | 人工查看去噪前后波形。 |
| `test_asa_classifier/` | ASA 分类实验结果 | 非主线模型试验输出。 |
| `test_asa_classifier/_vitaldb_signal_cache/` | VitalDB 波形缓存 | 非主线 ASA 实验缓存，避免重复 API 下载。 |

## 7. 算法路线演化

| 阶段 | 假设 | 结果 | 当前结论 |
|---|---|---|---|
| 1. IMU artifact denoiser | IMU 与 motion artifact 存在线性/非线性关系，模型可直接学习 artifact。 | 静态段表现好，动态段泛化失败。 | IMU 不能作为 teacher；motion artifact 的数学关系不能强假设。 |
| 2. clean prior / peak prior denoiser | 用 sit 段周期波形构造 motion clean，再学习 artifact。 | 部署端没有可靠 motion peaks，且伪峰/峰谷错位。 | 不适合作为部署方案。 |
| 3. 保留 gating 思路 | denoiser/gating 对静止/运动识别很鲁棒。 | 用户认为这是有价值成果。 | 转向独立 motion detector benchmark。 |
| 4. 动态心搏提取模型 | 不强行恢复完整 clean PPG，而是从动态 PPG 提取可靠 peak/IBI。 | 已实现多数据集训练和分层评估。 | 当前主线，更符合最终 HR/HRV 需求。 |
| 5. Frailty fusion | 最终 frailty classifier 应融合 IMU 静/动态、动态心搏、静态波形特征。 | 相关脚本需另行审阅。 | 下一阶段需要把动态模型接入主 notebook 和最终 frailty classifier。 |
| 6. ASA classifier | 测试 VitalDB ASA 与 PPG/ECG/peaks 的可分类性。 | 仅作为模型试验。 | 非本项目主线，不作为 frailty pipeline 成果。 |

## 7.1 初版 denoiser A/B 详细失败原因

| 模型 | 输入 | 观察结果 | 失败原因解释 | 结论 |
|---|---|---|---|---|
| Denoiser A | `raw PPG + IMU` | 部署端输出与原始 PPG 高度相似，动态段没有恢复出稳定周期波形。 | 模型没有学到足够有效的 artifact 去除映射，更像保守地复制输入；IMU 与 PPG artifact 的关系受佩戴位置、压力、设备延迟、运动方式、个体差异影响，单纯输入 IMU 不足以稳定泛化。 | 对动态段修复作用不足，不能作为主线。 |
| Denoiser B | `raw PPG + IMU + linear baseline` | 部署端比原信号出现双倍伪峰，部分原始最高峰位置被变成峰谷，motion 段可能呈梯形/非生理形态。 | linear baseline 给模型提供了过强但错误的先验；模型可能把 baseline 或其相位/形态偏差当成 clean PPG 结构，导致峰形重构过度、相位错位和伪周期增强。 | baseline 输入没有提升泛化，反而引入结构性伪峰。 |
| sit/static 段表现 | 静态 PPG 或 sit 段 | 绿色输出可与蓝色原始信号高度重叠，甚至看似“完美”。 | 这不是有效动态去噪能力，而更可能是模型在低噪声/静态分布上学会 identity mapping 或 gating 抑制修正。静态段原始 PPG 本身可用，复制输入容易得到好看的 plot。 | 静态段表现不能证明 denoiser 有效。 |
| motion 段表现 | 动态 PPG | 去噪后仍无可靠周期规律，或出现梯形、双峰、峰谷错位。 | 训练目标缺少真实 motion clean PPG；ECG peak、PPG peak 存在生理 delay；部署端 motion 段没有可靠 peaks 可校准；IMU 与 artifact 关系非固定函数。 | 说明“恢复 clean waveform”目标在当前监督条件下不可稳健实现。 |
| 核心监督问题 | IMU / artifact / clean PPG | 既不知道 motion artifact 真实形态，也不知道 motion clean PPG，且 IMU-artifact 数学关系未知。 | 早期算法实际上隐含了某些 teacher 或 prior 假设，但这些假设在跨数据集、跨设备、跨佩戴状态下不成立。 | 后续改为直接提取 peak/IBI，而不是恢复完整 clean waveform。 |
| 保留下来的价值 | gating 行为 | 模型稳定不对静止段误降噪，静止/运动识别看起来鲁棒。 | 这说明 encoder/gating 可能学到了运动状态或信号质量判别特征，而不是学到了可靠 artifact inverse mapping。 | 保留为独立 motion detector A/B benchmark 的来源。 |

## 8. 用户明确评价

| 主题 | 用户评价 |
|---|---|
| `ppg.py` | 已过时，不再作为主脚本。 |
| denoiser | 动态段完全不能用；A 接近原始信号，B 出双倍伪峰；motion 段变梯形/伪峰；不适合当前目标。 |
| IMU teacher 假设 | 不成立。IMU 与 artifact 有关系，但关系本身是要学习的答案，不能被假设为 teacher。 |
| 静态段 gating | 很有价值，静止段识别准确且不会误开降噪，可能比旧 detector 更鲁棒。 |
| 动态段部署 | 没有准确 peaks 可校准，所以部署端动态段不能依赖 peak prior 微调。 |
| 新目标 | 端到端神经网络只是动态心搏信息提取模块，不是最终 frailty classifier。 |
| 泛化评估 | aggregate 分数不够，需要 per-dataset/per-subject/per-activity、LODO、extra-holdout、delay 分析。 |
| ASA 实验 | 与本项目主线无关，仅作为模型试验。 |

## 9. 未完成需求与下一步优先级

| 优先级 | 未完成任务 | 说明 |
|---|---|---|
| 高 | 跑正式 `ppg_peak_hr_gating_train.py` 最新版本并读 scorecard | 需要用真实大样本配置，不是 smoke；重点看 PTT、MIMIC、VitalDB、iAMwell、simultaneous、AF/noisy/neonate 分层失败点。 |
| 高 | 决定最终 motion detector | 比较 A: denoiser encoder motion head 与 B: lightweight CNN detector；重点看 extra-holdout，而不是只看 PTT holdout。 |
| 高 | 将新 detector + peak/IBI 模型部署到 `ppg_analyse4_calib.ipynb` | 主 notebook 需要从旧 denoiser 路线切到“motion detector + dynamic heartbeat extractor”路线。 |
| 高 | 审阅最终 frailty 三分类脚本 | `frailty_3class_classifier.py`、`frailty_3class_cnn_fusion.py`、`frailty_3class_holdout_eval.py` 属于非本会话未审阅内容，需要单独检查。 |
| 中 | 为 `ppg_peak_hr_gating_train.py` 输出 ONNX/CPU-only 部署模块 | 当前有旧 denoiser ONNX 经验，但最终 runtime 模块仍需明确。 |
| 中 | 增强 ECG detector 与 PPG delay 校准 | PPG peak 与 ECG R peak 有生理 delay，动态 peak label 不能直接混用。 |
| 中 | HRV 指标层评估 | 最终 frailty 可能关心 SDNN、RMSSD、LF/HF 等；但训练目标应优先保证 beat timing、IBI 不漏检/不误检。 |
| 中 | 清理/归档旧 denoiser 脚本 | 避免后续 Codex 误以为 denoiser 是主路线。 |
| 低 | 整理 README/实验日志 | 当前信息散在 scorecard、summary、notebook、脚本里，建议后续写一个项目级 README。 |

## 10. 推荐新 Codex 阅读顺序

1. `AGENTS.md`
2. `_agent/WRITE_RULES.md`
3. `_agent/PROJECT_HANDOFF.md`
4. `ppg_peak_hr_gating_train.py`
5. `ppg_analyse4_calib.ipynb`
6. `pttppg_denoiser_hybrid_train.py`
7. `pttppg_denoiser_hybrid_preview.py`
8. `pttppg_denoiser_hybrid_ab_compare.py`
9. `.CNN_results/` 中最新 scorecard
10. `frailty_3class_classifier.py`
11. `frailty_3class_cnn_fusion.py`
12. `frailty_3class_holdout_eval.py`
13. `results_frailty3/` 中最新 scorecard

## 11. 给新 Codex 的一句话总结

本项目主线已经从“用 IMU 去噪动态 PPG”转向“用 detector 判断静/动态，再从动态 PPG 中提取可靠 heartbeats/IBI/HRV，并与静态 PPG 波形特征、IMU 状态一起用于 frailty 分类”。旧 denoiser 路线已基本判定失败，但它启发了新的 motion detector；当前最重要的主线脚本是 `ppg_peak_hr_gating_train.py`，主 notebook 是 `ppg_analyse4_calib.ipynb`。ASA 相关内容只是旁支模型试验，与本项目 frailty pipeline 无直接关系。
