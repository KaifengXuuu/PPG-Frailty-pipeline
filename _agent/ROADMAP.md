# ROADMAP

状态：confirmed
来源：用户项目总纲、`_agent/arc/PROJECT_HANDOFF.md`、2026-06-10 后实验与代码复核
最后手动更新时间：2026-08-18

## 2026-08-18：活动 V2 的分阶段 Static Line B 筛选路线

- 状态：confirmed
- 来源：用户确认的六阶段省算力流程；当前代码、配置、dry-run 和测试复核。
- 总原则：原 39-case mega-study 保留作完整探索；日常模型筛选优先按六阶段流程，
  每阶段人工阅读报告并记录晋级决定，不自动继承赢家。

### Stage 1：Representation baseline

- 当前状态：可运行。
- 默认预算：4 cases × 1 repeat × 5 folds = 20 outer cells。
- 退出条件：判断 Raw、Feature vector、Feature matrix、Fusion 哪些线路值得进入
  Stage 2；结果接近时先升级 Stage 1 到完整 5×5。

### Stage 2：晋级 representation 内模型比较

- 当前状态：可运行模板；启动前必须人工删除未晋级 representation block。
- 最大候选：Raw 3、Feature vector 3、Feature matrix 2、Fusion 2；ShapeFormer、
  MiniROCKET 和 ensemble 不混入本阶段。
- 退出条件：每条晋级线路形成可解释的模型 shortlist。

### Stage 3：ShapeFormer 稳定性阶梯

- 当前状态：可运行。
- 顺序：one-cell implementation test → one-repeat diagnostic → stable 后 full 5×5。
- 退出条件：无 OOM/NaN、OOF/概率有效、资源可接受；失败不得阻断普通模型。

### Stage 4：选中 Inception route 的 ensemble 因素

- 当前状态：Raw InceptionFull 与 Matrix Inception 两组 registered pair 可运行，
  每次只选择其中一组。
- 比较：fixed member 0 对 five-member fold-level probability mean。
- 退出条件：完成 matched single/ensemble paired report；无 eligible Inception
  winner 时跳过。

### Stage 5：Finalist 的 SQI 与 motion 路线

- 当前状态：部分可运行、部分 deferred。
- 已可运行：finalist 上 `quality.mode=off` 对 `diagnostics_only`，且 prediction/
  retention 应一致。
- 待实现：supervised SQI route、8ch/11ch motion model input、EKF/Profile-A、
  reducer selection、motion override、formal motion 5×5、A3/A4 runners。
- 升级条件：对应正式 runner、provenance、focused tests 和 no-training dry-run 完成。

### Stage 6：最终模型的串行单因素消融

- 当前状态：可运行模板。
- 顺序：每条锁定线路独立运行一个 axis；人工写回 winner 后再切换下一个 axis。
- Deep 建议顺序：LR → batch size → fixed epochs → weight decay。
- Classical 使用各自 factor：Logistic C；SVM C/gamma；ExtraTrees max_features/
  min_samples_leaf；ROCKET kernels/ridge alpha。
- 禁止：把多个因素同时展开成大 grid，或跨 representation 混用一个 baseline。

## 当前总路线

项目不再把 motion PPG 的 clean-waveform reconstruction 作为主路线。当前目标是在
subject-level、可复现、无数据泄漏的统一 benchmark 下，提高
`Pre-Frail / Robust-Non-Frail / Young` 三分类泛化能力，并保留可解释的生理依据。

Frailty3 后续并行验证两条路线：

1. **Raw-signal route**：flat InceptionTime baseline 与
   Young-vs-Old -> Pre-Frail-vs-Robust 的 hierarchical InceptionTime。
2. **Physiology-feature route**：IMU motion gating + motion-robust peak/PPI/HR
   extraction，按 Base/Motion/Relax 提取 HR/HRV/activity/recovery features，
   再用较弱且可解释的模型分类。

两条路线必须先经过统一 benchmark、跨 sweep 整合和消融，才进入 final export。

## 阶段路线

### 0. 数据与协议基线

- 状态：当前 frailty3 已核准，持续维护。
- 已完成：原始输入目录只读；活动流程保持 400 Hz、无 resampling；
  subject-level 5-fold `StratifiedGroupKFold`、fixed epoch、no early stopping。
- 仍需：统一 manifest、fold registry、数据版本和 `oof_validation_*` 命名。

### 1. Benchmark 与历史实验整合

- 状态：最高优先级，未实现。
- 目标：建立 `frailty_3class_benchmark.py` 和
  `analyze_all_frailty_experiments.py`。
- 工作：
  - 统一 models、metrics、folds、seeds、repeats、outputs。
  - 隔离 holdout、early-stopping CV、fixed-epoch CV 和 leakage results。
  - 整合所有 sweep/grid，输出参数覆盖、稳定性、class effects 和严格 Top 5。
- 退出条件：可从任一最终表追溯 dataset、fold、seed、epoch、完整 config 和 predictions。

### 2A. Hierarchical Raw-Signal Classifier

- 状态：approved plan，未实现。
- 目标：比较 flat three-class 与两层二分类是否能改善标签边界。
- 工作：上层 Young/Old；下层只用 old train subjects 分
  Pre-Frail/Robust；概率组合为 end-to-end three-class。
- 退出条件：同 folds/seeds/budget 下完成 top-level、oracle bottom-level、
  end-to-end 和 flat baseline 配对报告。

### 2B. Base/Motion/Relax Physiological Features

- 状态：approved plan，依赖阶段定义和 heartbeat extractor 验证。
- 目标：绕开小样本 raw deep model 的部分复杂度，构建可解释的阶段级特征。
- 工作：
  - 确认 role 到阶段的真实映射。
  - 优化 IMU static/motion gating。
  - 用同步 ECG 开放数据监督 motion peak/PPI/HR extraction，处理 domain shift。
  - 提取 stage HR/HRV、SQI、activity、recovery slope/time/relative baseline。
  - 比较 logistic regression、SVM、ExtraTrees/small boosting。
- 退出条件：外部 heartbeat scorecard 和 frailty subject-CV 结果均完整，
  且无跨 subject/dataset 泄漏。

### 3. Preprocessing、Fusion 与统一消融

- 状态：未开始。
- 目标：找出真实有用的信号、质量控制和融合方式，而不是继续盲目扩大 grid。
- 工作：
  - 清洗、SQI coverage 和 scaler audit。
  - 检查 file-level features 复制到 windows 的隐式加权/时间范围。
  - 比较 early/file/subject late fusion 和 OOF stacking。
  - 消融 RED/IR、PPG/IMU、gravity、SQI、PPI/HRV、morphology、hierarchy、
    sampler、aggregation、stage/recovery features。
- 退出条件：每项均有相同 folds/seeds 的 paired delta、CI、class metrics、
  confusion matrix 和 coverage。

### 4. 最终模型选择与独立性能陈述

- 状态：被阶段 1--3 阻塞。
- 目标：从严格可比 Top 5 中按 config-level repeats 选择，而非选择最佳单次 repeat。
- 工作：确认最终 selection rule；如需要发表级独立性能，预留 untouched test
  或 external validation，并在测试前锁定 config。
- 注意：当前 5-fold OOF CV 不是额外独立 test；当前可比 BA 约 0.62，
  目标 0.73 尚未达到。

### 5. Final Training、Export 与 Notebook Integration

- 状态：未完成。
- 目标：用锁定 config 和明确 final-training protocol 重训部署模型。
- 输出：模型权重、scaler/imputer、label map、manifest、fold registry、
  feature schema、window/preprocessing 参数、训练配置和评估摘要。
- 后续：稳定的 heartbeat extractor 和 frailty model 接入
  `ppg_analyse4_calib.ipynb`；需要时再做 ONNX/CPU-only bundle。

## 保留的旁支

- ShapeFormer/ShapeFormer-PISD：仅小规模 ablation，暂不进入大 sweep。
- Dynamic clean-waveform denoising：deprecated/reference。
- ASA classifier：独立旁支，不进入 frailty 主线结论。
- W&B：低优先级；本地 CSV/JSON/PNG/Markdown 暂时足够。
