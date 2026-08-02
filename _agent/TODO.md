# TODO

状态：confirmed
来源：用户当前要求、`_agent/arc/PROJECT_HANDOFF.md`、2026-06-10 后代码与结果复核
最后手动更新时间：2026-07-26
用途：记录明确可执行任务，并把未完成事项与对应脚本、验收条件和依赖关系对应起来。

## 最高优先级 P0

### P0-1 建立通用 Frailty3 Benchmark

- [ ] 新建统一 benchmark 入口，建议文件名：`frailty_3class_benchmark.py`。
- [ ] 固化 dataset manifest：subject、class、file、role、原始采样率、纳入/排除原因和数据版本。
- [ ] 固化当前可比协议：400 Hz、subject-level 5-fold `StratifiedGroupKFold`、
  fixed epoch、no early stopping、相同 folds/seeds/repeats。
- [ ] 为 flat InceptionTime、hierarchical InceptionTime、1D-CNN、
  Small InceptionTime 和 tabular baseline 提供统一 wrapper。
- [ ] 统一输出 window/file/subject-level metrics、三类 class report、
  confusion matrix、learning curves、fold subject IDs、完整 config、seed、
  数据版本和 completeness check。
- [ ] 将 holdout、early-stopping CV、no-early-stopping CV 和存在 data leakage
  的历史结果标成不同 protocol，禁止直接合并排名。
- [ ] 修正当前报告中把 OOF validation 命名为 `test_*` 的歧义；没有独立 test
  时必须明确写 `oof_validation_*`。
- [ ] 复核现有训练脚本 CLI/defaults，并由 benchmark 显式传参，避免旧默认值影响实验。
- 涉及文件：`frailty_3class_classifier.py`,
  `frailty_3class_overfitting_sweep.py`, `frailty_3class_holdout_eval.py`,
  `analyze_sweep.py`。
- 验收：同一 manifest/fold registry 可重复运行所有候选模型；报告可追溯到
  subject split、seed、epoch 和数据版本；不同 protocol 不进入同一 leaderboard。

### P0-2 整合全部 Sweep/Grid Search 并选出严格可比 Top 5

- [ ] 新建跨实验分析入口，建议文件名：`analyze_all_frailty_experiments.py`。
- [ ] 建立 parameter ontology 和 protocol registry；统一旧字段名、缺失字段和 reference 来源。
- [ ] 纳入至少以下结果：`20260527_1320_cnn_inceptionTime`、
  `overfitting_20260608_0752`、
  `20260608_1206_overfitting_sweep_stage1_rank2`、
  `20260625_2320_overfitting_sweep_stage1_rank2`、
  `20260630_0630_overfitting_sweep_generalization_rank2`。
- [ ] 对 2026-05-27 结果标记 data leakage；只允许用当前协议重跑的 reference
  进行绝对分数比较。
- [ ] 在相同 folds/seeds 下计算 paired metric differences、bootstrap confidence
  intervals、class-level effects 和 stability；不把不平衡 grid 的回归系数解释为因果效应。
- [ ] 输出参数覆盖图、主效应/交互图、class recall/F1、confusion matrix、
  repeat/fold stability 和 reference drift。
- [ ] 在严格相同协议中按 config-level repeat mean 排名，选出 Top 5 完整参数组；
  不按最佳单次 repeat 排名，不用 runtime/cost/Pareto 指标影响排名。
- [ ] 扩展 `analyze_sweep.py` 的 config identity，使 SQI、aggregation、
  manual features、loss、class weight、sampler、window quota 和 train overlap
  成为显式 config columns；默认模型列表加入 `small_inceptiontime`。
- 验收：Top 5 表包含完整参数、mean/std/CI、worst-class 指标、三类 confusion
  matrix 和可比性标签；不完整 config 不与完整 config 公平排名。

### P0-3 试验两层二分类 Hierarchical InceptionTime

- [ ] 上层分类器：`Young` vs `Old`。
- [ ] 下层分类器：只在 old training subjects 中分类 `Pre-Frail` vs
  `Robust/Non-Frail`。
- [ ] 初版使用两个独立 InceptionTime，不共享参数，以便判断每一层的真实贡献。
- [ ] 最终三类概率按以下方式组合：
  `P(Young)`、`P(Old) * P(Pre-Frail | Old)`、
  `P(Old) * P(Robust | Old)`。
- [ ] 与 flat InceptionTime 使用完全相同的 outer subject folds、seeds、
  epoch、window 和训练预算；至少 5 repeats。
- [ ] 分别报告 top-level BA、old-only oracle bottom-level BA、
  end-to-end subject-level 三分类 BA/F1 和错误传播。
- [ ] 下层训练和预处理只能使用 outer-train 中的 old subjects，禁止读取
  validation fold 标签或特征统计。
- 涉及文件：建议新建 `frailty_3class_hierarchical.py`，并接入 P0-1 benchmark。
- 验收：能够判断瓶颈来自 Young/Old、Pre/Robust，还是上层错误传播；与 flat
  baseline 完成配对比较。

### P0-4 建立 Base/Motion/Relax 分阶段生理特征路线

- [ ] 先确认文件 role `B/R/S/W` 与 Base/Motion/Relax 的真实实验阶段映射；
  未确认前不得凭文件名推断。
- [ ] 继续优化 IMU static/motion gating，并以 subject/dataset split 做独立评测。
- [ ] 使用带同步 ECG 的开放数据训练 motion-robust peak/PPI/HR extractor。
  ECG 只提供 peak/timing supervision；目标是可靠 beat/PPI/HR，不是重建 clean waveform。
- [ ] 分阶段提取 HR、HRV、SQI、IMU activity、阶段间变化和
  recovery slope/time/relative-to-baseline。
- [ ] 先试弱且可解释的模型：regularized logistic regression、SVM、
  ExtraTrees 或小型 gradient boosting。
- [ ] 定义特征 schema、缺失值策略、SQI 规则和 fold-only scaler；
  与 raw InceptionTime 在相同 subject CV 下比较。
- [ ] 完成正式非-smoke 的 `ppg_peak_hr_gating_train.py` 训练与 scorecard，
  包括 per-dataset/per-subject/per-activity、extra holdout、LODO 和 delay analysis。
- [ ] 稳定后再接入 `ppg_analyse4_calib.ipynb`，随后考虑 ONNX/CPU-only bundle。
- 验收：外部 heartbeat extractor 不依赖目标 frailty subject 的 validation/test
  标签；输出阶段级可解释特征，并通过 P0-1 benchmark 评估。

### P0-5 完成统一消融试验

- [ ] 至少覆盖：RED/IR、PPG-only、IMU-only、gravity removal、SQI、
  PPI/HRV、morphology、hierarchy、sampler、aggregation、
  Base/Motion/Relax stage 和 recovery features。
- [ ] 每次只改变一个因素；固定 folds、seeds、epoch、window、训练预算和基础 config。
- [ ] 报告 subject-level paired BA/macro-F1 delta、bootstrap CI、
  class recall/F1、confusion matrix、coverage 和失败样本。
- [ ] 把 PPI/HRV/manual features 是否保留在最终模型的旧任务并入本消融，
  不再依据单次 run 决定。
- 验收：能够区分“平均分提高”“稳定性提高”“仅某一类提高”和
  “因丢弃低质量窗口而改变评估覆盖”。

## 高优先级 P1

### P1-1 审计数据清洗与 Scaler

- [ ] 检查 NaN、flatline、saturation、异常振幅、异常 PPI、过短文件、
  channel missing、设备量纲和低 SQI windows。
- [ ] 比较 current per-window median/IQR、fold-level per-channel scaling、
  `StandardScaler`、`RobustScaler` 和保留绝对振幅的 hybrid scaling。
- [ ] 所有 scaler/imputer 只能在 training fold 拟合。
- [ ] 专门验证当前 per-window robust scaling 是否删除对 frailty 有用的
  pulse amplitude、IR/RED ratio 或恢复幅度信息。
- [ ] 记录 SQI gating 后每个 subject/class 的保留窗口数，防止 coverage
  不均造成表面性能变化。
- 验收：输出清洗审计表、各 scaler 的配对结果和 class/subject coverage。

### P1-2 检查异步、多时间尺度人工特征融合

- [ ] 明确每个特征的时间范围：window、file、stage 或 subject，以及真实推理时可用性。
- [ ] 审计当前 file-level features 复制到每个 window 的做法；检查是否按窗口数
  隐式放大某些文件，以及是否使用了窗口之后的整文件信息。
- [ ] 比较四种融合：当前 window early fusion、raw embedding 聚合后 file fusion、
  subject-level late fusion、out-of-fold stacking。
- [ ] stacking 的 meta-model 只能使用 OOF predictions；imputer/scaler 必须 fold-only。
- [ ] 输出 feature schema、subject/file/window alignment、缺失策略和消融结果。
- 验收：不存在跨 fold 或未来时间信息泄漏；不同时间尺度的特征贡献可单独解释。

## 保留与依赖任务

- [ ] 在 P0-1/P0-2 确定 final config 后，设计 final training/export。
  保存模型权重、scaler、label map、manifest、fold registry、window 参数、
  feature schema、训练配置和评估摘要。
- [ ] 在 P0-4 的 heartbeat extractor 稳定后，完成 ONNX/CPU-only smoke test
  并接入主 notebook。
- [ ] 增强 ECG detector 与 PPG delay calibration，系统评估 SDNN、RMSSD
  等 HRV 指标层误差。
- [ ] 清理并归档旧 `pttppg_denoiser_hybrid_*`、
  `pttppg_stage2_denoiser.py` 等 dynamic denoiser 路线，保留 reference。
- [ ] ShapeFormer 仅保留小规模 ablation；PISD 成本高且现有提升不明确，
  暂不进入大 sweep。
- [ ] W&B 保持低优先级；当前 CSV/JSON/PNG/Markdown 可满足本地分析，
  只有跨机器或实验规模继续扩大时再评估。

## 已完成或被后续实验替代

- [x] 已确认当前活动 frailty3 流程维持原始 400 Hz，不做 resampling；
  `PPG_Testing_05_01_2026/` 和 `physionet.org/` 已设为只读。
- [x] 已实现 PPG 基础预处理、IMU gravity removal、local Aboy++ morphology、
  file-level PPI/HRV/manual fusion、SQI gating、weighted CE/balanced
  softmax/focal loss、class weights 和 subject-aware samplers。
- [x] 2026-06-08 stage1 Top4 后续任务已被 2026-06-25 和 2026-06-30
  更完整 sweep 覆盖，不再直接作为当前 stage2 入口。
- [x] 2026-06-25 sweep：129 configs、645 runs，完整。
- [x] 2026-06-30 generalization sweep：232 configs、1160 runs，完整；
  新 config 未达到 BA 0.60，当前结果仍未达到 0.73。
- [x] 已完成 2026-06-16 和 2026-07-06 的独立 sweep reports；
  后续由 P0-2 统一整合。
- [x] 旧“检查采样率、时间戳和重采样规范”中的当前 frailty3 重采样问题已核准；
  外部数据的时间轴/设备量纲仍属于 P1-1。
