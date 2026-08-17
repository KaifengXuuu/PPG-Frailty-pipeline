# Phase 07 — Models, frozen-fold training, OOF evaluation, and bundles

## English report

### Scope and source review

This phase re-scanned the locked implementation specification, the V1 contracts,
ADR-006/007/008/011, the resolved configuration names, and the reviewed historical
model definitions.  It wrote only inside `final_v0/final_pipeline_v1`.  Historical
root code and `_agent` remained read-only.  Data ingestion, signal processing,
artifact reduction, SQI, and feature extraction were deliberately outside this
phase.

### Implemented model layer

- Added the reviewed `CompactCNN1D` with exactly 79,139 trainable parameters for
  eight channels and three classes.  The implementation preserves the reviewed
  32/64/128 channel widths, 9/9/7 kernels, pool-by-four stages, dropout schedule,
  global pooling, and three-class head.
- Added reviewed full and small InceptionTime single networks.  Exact parameter
  counts are 456,579 and 57,027 for eight channels and three classes.  Full uses
  depth six and 32 filters; small uses depth three and 16 filters.  Both use
  mask-aware pooling, and padded columns are zeroed before convolution.
- Added an optional five-member InceptionTime probability ensemble.  It requires
  exactly five distinct seeds, rejects shared parameter objects, trains members
  independently, and averages probabilities rather than logits.
- Added self-contained NumPy ROCKET plus ridge.  The reference factory route is
  restricted to `feature_matrix`, defaults to 10,000 kernels, fits robust scaling,
  kernels, and ridge state on the outer-training partition, and treats the matrix
  mask as invalid support.  `MiniROCKET` is an explicitly named engineering
  ablation and does not claim reference-algorithm parity.
- Added fold-local logistic-regression-L2, RBF-SVM, and ExtraTrees feature-vector
  baselines with frozen feature names and fitted-object provenance.
- Added a self-contained effect-size shapelet discovery route and experimental
  ShapeFormer.  It imports no external PISD implementation or machine-specific
  path and is explicitly marked experimental.
- Added corrected `FileBagFusionClassifier`: signal windows are encoded and pooled
  first, and the file feature vector is encoded exactly once per file afterward.
  Padded bag slots never reach the signal encoder.
- Added one canonical model-name registry.  Human specification names such as
  `CompactCNN1D`, `InceptionTimeFull`, `InceptionTimeMatrix`, and
  `LogisticRegressionL2` map to stable machine IDs.  Unknown aliases fail closed;
  bundles record both names.

### Implemented training and evaluation layer

- Added typed datasets for all four representation modes: `raw`,
  `feature_vector`, `feature_matrix`, and `fusion`.  The matrix dataset includes a
  strict adapter from schema-compatible `OrderedFeatureMatrixV1` objects.
- Added `FrozenOuterSplit` and `UnifiedTrainer`.  The fit API has no outer-OOF
  labels or dataset argument.  The primary rule trains a pre-registered fixed
  number of epochs.  The optional inner-grouped rule selects an epoch only inside
  outer training, discards the selection model, creates a fresh model, and refits
  on all outer-training rows.
- Added independently auditable provenance for every fitted scaler, imputer,
  ROCKET transform, classifier, shapelet bank, and network state.  Learned state,
  exact participant membership, registry hash, and fold hash are bound to the fit.
- Added prediction-only evaluation, finite probability validation, balanced
  accuracy, macro-F1, multiclass log loss, and confusion matrices.
- The trainer consumes the resolved YAML training block without ignored fields.
  Canonical fixed_epoch, Adam, cross-entropy, deterministic algorithms,
  participant-to-file-to-window balanced sampling, and outer-training
  participant inverse-frequency class weights are executed exactly.  The
  historical fixed spelling is accepted only as a recorded legacy alias.
  Classical and ROCKET fits receive the equivalent combined sample weights.
- PyTorch import is lazy at the training boundary.  Without the optional deep
  dependency, estimator datasets, OOF/evaluation utilities, bundles, and
  UnifiedTrainer.fit_estimator remain usable; only tensor/deep calls fail with a
  focused dependency message.
- Added strict OOF rows and an atomic Parquet writer.  `pyarrow` is optional; when
  it is unavailable the writer fails closed and never falls back to CSV.
- Added the frozen aggregation hierarchy
  `window -> file -> role -> participant`.  Windows, files, and roles are averaged
  at their direct parent level; SQI weighting is an explicit ablation only.
- Added one-factor ablation execution and paired subject-delta comparison APIs.
- Added integrity-checked model bundles for torch and estimator models.  Bundles
  contain model/input configuration, both canonical and machine model IDs,
  provenance metadata, optional fitted transforms, payload hashes, and golden
  inputs/probabilities.  Loading rejects missing, modified, or unexpected files;
  saving reloads the bundle and enforces golden parity.

### Stable public facade

- `ppg_frailty.models.create_model(model_config, input_spec)`
- `ppg_frailty.training.UnifiedTrainer`
- `ppg_frailty.training.FrozenOuterSplit`
- `ppg_frailty.training.evaluate_predictions`
- `ppg_frailty.training.aggregate_hierarchy`
- `ppg_frailty.training.OofWriter`
- `ppg_frailty.training.run_ablation_matrix`
- `ppg_frailty.training.save_bundle`
- `ppg_frailty.training.load_bundle`
- `ppg_frailty.training.predict_bundle`

### Verification

- AST parse: 68 V1 Python files parsed successfully at the time of the check.
- Model tests: 10/10 passed with standard-library `unittest`.
- Training/OOF/bundle tests: 13/13 passed with standard-library `unittest`,
  including an actual-reference-YAML constructor test and a subprocess that
  blocks every torch import while fitting an estimator successfully.
- Exact architecture parameter tests passed: 79,139 / 456,579 / 57,027.
- V1 validator passed required paths, spec hash, legacy-import rejection, strict
  JSON, and config loading.  Its bilingual check found two unrelated files owned
  by other phases (`artifacts/__init__.py` and `features/__init__.py`); no
  models/training file failed that check.

### Remaining scientific/runtime decisions

- PyTorch is available in the current environment, but its V1 dependency status
  remains `decision_pending`; runtime availability is not treated as approval.
- ShapeFormer effect-size discovery remains an experimental comparison route.
- MiniROCKET remains a separately named ablation, not the specified ROCKET route.
- This phase ran deterministic unit/smoke tests only.  Formal five-repeat,
  five-fold candidate reruns, resource profiling at 10,000 ROCKET kernels, and
  locked participant-level benchmarks remain downstream execution work.

## 中文报告

### 范围与源文件复核

本阶段重新扫描了锁定实现规范、V1 contracts、ADR-006/007/008/011、resolved
配置名称和已审查历史模型定义。所有写入严格位于 `final_v0/final_pipeline_v1`；
根目录历史代码与 `_agent` 均保持只读。数据导入、信号处理、伪影削减、SQI 与
特征提取不属于本阶段范围。

### 已实现模型层

- 落地已审查 `CompactCNN1D`；八通道三分类下参数量精确为 79,139。保留
  32/64/128 通道、9/9/7 卷积核、四倍池化、dropout、全局池化和三分类头。
- 落地完整/小型 InceptionTime 单网络；八通道三分类参数量分别精确为 456,579
  和 57,027。完整版本 depth=6/filters=32，小型版本 depth=3/filters=16；补齐列
  在卷积前归零，最终池化显式读取 mask。
- 落地可选五成员 InceptionTime 概率集成：必须恰好五个不同 seed，拒绝共享参数
  对象，成员分别训练，并在概率空间而非 logits 空间平均。
- 落地自足 NumPy ROCKET + ridge。规范 factory 路线只允许 `feature_matrix`，默认
  10,000 个核；稳健缩放、kernel 与 ridge 均只在 outer-training 拟合，mask 定义
  无效支持。`MiniROCKET` 明确为工程消融，不宣称参考算法等价性。
- 落地 feature-vector 的 L2 逻辑回归、RBF-SVM、ExtraTrees，并冻结特征名称与
  各拟合对象 provenance。
- 落地自足效应量 shapelet 发现和实验性 ShapeFormer；不导入外部 PISD 或机器
  特定路径，并明确标为 experimental。
- 修正 FileBag 融合：先编码/汇聚信号窗口，再对每个文件仅编码并拼接一次文件
  特征；补齐袋位置不会进入信号编码器。
- 建立唯一规范名称 registry。`CompactCNN1D`、`InceptionTimeFull`、
  `InceptionTimeMatrix`、`LogisticRegressionL2` 等人类名称映射到稳定 machine ID；
  未知别名关闭失败，bundle 同时记录二者。

### 已实现训练与评估层

- 为 `raw`、`feature_vector`、`feature_matrix`、`fusion` 四种 representation
  建立类型化 dataset；matrix dataset 可从 schema 完全兼容的
  `OrderedFeatureMatrixV1` 严格构造。
- 建立 `FrozenOuterSplit` 与 `UnifiedTrainer`。fit API 不接受 outer-OOF 标签或
  dataset。主协议采用预注册固定 epoch；可选 inner-grouped 路线只在 outer-train
  内选 epoch，随后丢弃选择模型、创建新模型并在完整 outer-train 从头重训。
- 对 scaler、imputer、ROCKET transform、classifier、shapelet bank 与网络权重
  分别记录可审计 provenance，并绑定学习后状态、参与者成员、registry hash 和
  fold hash。
- 建立只预测 evaluator、有限概率校验、BA、macro-F1、多分类 log loss 与混淆矩阵。
- trainer 对 resolved YAML 的 training 区块实行零忽略字段消费：实际执行规范
  fixed_epoch、Adam、cross-entropy、确定性算法、participant→file→window
  平衡抽样和仅基于 outer-training participant 的逆频率类别权重。历史拼写 fixed
  仅作为有记录的 legacy alias；经典模型与 ROCKET 使用等价组合 sample weight。
- torch 在训练边界延迟导入。缺少可选深度依赖时，estimator dataset、OOF/评估、
  bundle 与 UnifiedTrainer.fit_estimator 仍可使用；只有张量/深度调用会给出明确
  依赖错误。
- 建立严格 OOF 行与原子 Parquet 写入器；缺少可选 `pyarrow` 时关闭失败，禁止
  静默回退 CSV。
- 固定唯一聚合层级 `window -> file -> role -> participant`；每层对直接子节点
  等权平均，SQI 加权只能作为显式消融。
- 建立单因素消融和 subject 配对差值 API。
- 建立 torch/estimator 的完整性 bundle：记录模型/输入配置、两种模型名称、
  provenance、可选 fitted transforms、payload hash 与 golden 输入/概率；加载时
  拒绝缺失、被修改或未登记文件，保存后立即重载并校验 golden parity。

### 验证结果与剩余事项

- 检查时 68 个 V1 Python 文件 AST 全部解析成功。
- 模型测试 10/10 通过；训练/OOF/bundle 测试 13/13 通过，均使用标准库
  `unittest`；后者包含实际 reference YAML 构造测试，以及阻断所有 torch import
  后仍成功拟合 estimator 的独立 subprocess 测试。
- 精确参数量测试 79,139 / 456,579 / 57,027 全部通过。
- V1 validator 的 required paths、spec hash、禁止历史 import、strict JSON、配置
  加载均通过；双语检查仅发现其他阶段的 `artifacts/__init__.py` 与
  `features/__init__.py`，models/training 本批次文件没有失败。
- 当前环境可用 PyTorch，但 V1 依赖状态仍为 `decision_pending`；ShapeFormer
  仍是实验对照；MiniROCKET 仍是具名单独消融。
- 本阶段只运行确定性 unit/smoke test。正式 5 repeats × 5 folds 全候选重跑、
  10,000 核 ROCKET 资源画像和锁定 participant-level benchmark 属于后续执行。
