# V2 / V5 代码量与精简审计

代码行数是审阅成本指标，不是算法等价证据。V5 的精简边界是保留论文 workflow、
数值算法和全部 analyse 能力，同时删除重复 runner、嵌入式展示、历史复制模板和多层
包装。数值等价仍须在冻结环境下比较完整 25-fold 输出。

## 统一统计口径

本文分开使用三种口径，避免把测试、文档或外部依赖混进“核心生产代码”：

1. **旧 sweep 递归本地 Python 闭包**：根目录旧入口及其项目内直接/递归依赖；
2. **V5 核心生产代码**：`src/ppg_frailty/**/*.py`（排除独立 Dash 包）、非 Dash
   顶层生产入口和 `tools/*.py`；这是用户指定的 70,000 行目标口径；
3. **V5 全部 Python**：还包括 tests 与独立 Dash，只用于工程库存，不代表生产核心。

所有数字都是 physical lines，包含空行和注释。V5 核心口径排除 tests、docs、
README、`_agent`、独立 `src/ppg_frailty/dashboard/` 与 `dashboard.py`、历史输出、cache、
三个输出根、`__pycache__` 和第三方源码；`tools/*.py` 计入，Dash 另报自身行数。

当前 V5 核心生产代码值：**66,564 行**。同一命令在本轮开始时保存的只读会话
快照（临时路径 `/tmp/v5-global-refactor-before-EEAX7K/final_pipeline_v5`）上得到
**131,179 行**，净减
**64,615 行（49.26%）**，距离 70,000 行上限留有 3,436 行。此前人工库存记录为
131,213 行，比可复算快照多 34 行；本文的比例与净变化只使用可复算的同口径快照，
不混用两个基线。

## 旧 sweep 递归闭包

从仓库根目录 `frailty_3class_overfitting_sweep.py` 与 `analyze_sweep.py` 递归扫描本地
Python 引用，闭包为：

| 文件 | physical LOC | 角色 |
|---|---:|---|
| `frailty_3class_classifier.py` | 3,532 | 数据、特征、模型与训练主体 |
| `frailty_3class_overfitting_sweep.py` | 1,707 | sweep 入口与实验组合 |
| `analyze_sweep.py` | 1,372 | sweep 分析与图表 |
| `frailty_3class_holdout_eval.py` | 701 | holdout/配置辅助 |
| `shapeformer_port.py` | 512 | ShapeFormer 本地适配 |
| **递归本地闭包合计** | **7,824** | `3,532+1,707+1,372+701+512` |

其中两个用户直接入口合计 **3,079 行**：
`frailty_3class_overfitting_sweep.py` 1,707 加 `analyze_sweep.py` 1,372。

ShapeFormer PISD 的外部实现约 **636 行**，不属于此仓库的本地 Python 闭包，故不计入
7,824。若以后比较可部署 source bundle，必须把外部依赖另列，不能悄悄加进本地行数。

## 本轮核心精简

下表逐文件比较重构前快照和当前工作树，按净删减量排序。它能稳定复算，也避免把
交叉移动的文件重复计入人为分组；新增共享文件的行数已经包含在上面的全局净变化中。

| 顺序 | 文件 | 重构前 | 当前 | 净变化 | 主要精简 |
|---:|---|---:|---:|---:|---|
| 1 | `quality/stage5_pre.py` | 7,454 | 1,047 | −6,407 | 删除训练路径中的展示模板与重复 artifact 包装，保留 Stage5 数值 runner |
| 2 | `experiment.py` | 10,054 | 4,109 | −5,945 | 合并 workflow 分支、fold 产物与 refit 调度样板 |
| 3 | `reporting/analyze.py` | 5,442 | 408 | −5,034 | 用共享 registry/spec 取代逐图逐表重复分派 |
| 4 | `reporting/report.py` | 4,397 | 0 | −4,397 | 删除无调用的 V2/HTML 兼容入口，保留独立 analyse report |
| 5 | `reporting/incomplete.py` | 2,724 | 0 | −2,724 | 删除复制式失败报告，改用紧凑 execution audit |
| 6 | `study/hyperparameter.py` | 3,422 | 768 | −2,654 | 复用 phase runner、统计与 artifact writer |
| 7 | `reporting/historical_suite.py` | 2,218 | 0 | −2,218 | 删除无调用的旧 suite 包装，保留 specialized suite 实现 |
| 8 | `reporting/historical.py` | 2,080 | 241 | −1,839 | 历史 comparison 使用同一收集与渲染协议 |
| 9 | `models/factory.py` | 4,148 | 2,406 | −1,742 | 表驱动模型构造，保留架构和初始化方程 |
| 10 | `config.py` | 2,465 | 1,057 | −1,408 | 合并配置解析、默认值与验证路径 |
| 11 | `study/schema.py` | 2,186 | 799 | −1,387 | 用声明式 schema 复用 plan/case 字段处理 |
| 12 | `module_registry.py` | 2,698 | 1,355 | −1,343 | 模块目录、默认值与 CLI metadata 共用一份定义 |
| 13 | `pipeline.py` | 1,715 | 400 | −1,315 | 保留共享 preflight/record loader，删除无人调用的旧 smoke writer |
| 14 | `quality/motion_runner.py` | 2,363 | 1,220 | −1,143 | 统一 internal/PTT、OOF、evidence 与 bundle plumbing |
| 15 | `v5/specialized.py` | 1,497 | 390 | −1,107 | 专项执行入口共享 plan、report 与 provenance 调度 |
| 16 | `v5/model_config_export.py` | 1,291 | 257 | −1,034 | 合并 model selection、manifest 和复用参数导出 |
| 17 | `reporting/tabular.py` | 1,322 | 314 | −1,008 | 统一表格 spec、N/A 行和序列化 |
| 18 | `training/trainer.py` | 2,758 | 1,847 | −911 | 合并等价 epoch、checkpoint 与 prediction plumbing |
| 19 | `reporting/conclusions.py` | 1,585 | 699 | −886 | 共享结论表、统计投影和缺失处理 |
| 20 | `study/runner.py` | 1,774 | 911 | −863 | 统一 serial/thread/process、resume 与 fail-fast 生命周期 |

“模块保留”不等于逐字保留冗余代码。上述范围保留公开入口、数学公式、模型结构、
threshold、采样、split、workflow 顺序、统计搜索、表格字段和图表种类；删掉的是同一
逻辑的重复实现、嵌入式 presentation 和一次性历史包装。

新增的 `v5_reporting/execution_audit.py` 为 352 行，用统一 writer 恢复旧
`reporting/incomplete.py` 的失败/中断执行审计能力；它只读 metadata，不读取 OOF 或
weights，也不产生科学结论。与原 2,724 行复制式实现相比，该能力本身净减 2,372 行。

## 按删减量排序的大改动

1. **训练与展示彻底分离。** pipeline 只写预测、指标、Excel 和 weights；Stage5、
   hyperparameter 及普通 classification 的 figures/HTML 统一由只读 report 入口重建。
2. **共享 specialized 输出层。** 历史专项计算保留 strict loader、数值 runner、resume
   和模型导出，只把表/图/文字模板移到复用 reporter。
3. **统一 motion 数据路径。** internal/PTT、transfer/reverse、threshold、OOF 和窗口
   张量语义保留，共享 manifest/evidence 与 bundle plumbing。
4. **统一 study 生命周期。** single/grid/ablation/catalog/legacy bridge 通过同一
   expansion、cell executor、resume 和 `run/comparison/repeat/fold` 输出路径。
5. **报告与统计表驱动。** metric、空值、聚合和输出类型由紧凑 spec 生成，避免每个
   analysis 重复 CSV/Markdown/HTML 代码。
6. **配置目录动态生成。** module/default/range/path 使用一份映射，CLI、Dash 和 help
   不再各维护长表。

本轮又从上一工作树的 69,796 行净减 3,232 行核心生产代码：删除重复的
report/evaluate 兼容入口与 singular/plural 算法转发层，把内部调用直接绑定到唯一实现，
精简无逻辑 package re-export 和实现文件内 101 份重复符号清单，并移除仅检查源码 SHA
白名单的旧 parity 门禁；继续移除不可达的旧 smoke writer、已被 recording cache 取代
的缓存类和无消费者的 Dash 下载 helper，同时以 352 行共享实现保留失败运行审计。真正按
fold、row identity 和 `atol=1e-6` 比较结果的 `tools/compare_v2_v5_outputs.py` 保留。

## V5 新增而合理保留的代码

V5 不追求把所有新增能力压成旧入口的 7,824 行。以下是用户要求的生产功能：

- 完整 preset/manual/study CLI 和 live parameter catalog；
- 每 fold 多层预测、learned bundle、resume、并发和无泄漏 cache；
- 三根输出合同、pipeline/report 两种 Excel、自动 model_config；
- 独立可组合 analyse report；
- Dash 训练/停止/inference/queue/预览/下载；
- specialized/legacy comparison 的可复用入口。

因此 7,824 是旧 sweep 局部闭包的对照，不是整个 V5 的合理上限。

## 可复现统计

从仓库根目录执行，下面命令直接得到本文定义的核心生产代码 physical LOC：

```bash
find final_v0/final_pipeline_v5 -type f -name '*.py' \
  -not -path '*/tests/*' -not -path '*/dashboard/*' \
  -not -path '*/cache/*' -not -path '*/pipeline_output/*' \
  -not -path '*/report_output/*' -not -path '*/model_config/*' \
  -not -path '*/__pycache__/*' -not -name dashboard.py -print0 \
  | xargs -0 wc -l | tail -1
```

当前输出为 `66564 total`。独立 Dash 包为 4,942 行，入口 `dashboard.py` 为 30 行；
二者按既定目标口径另计。tests 为 8,116 行，同样不混入生产核心。

## 等价验证边界

代码更短、AST 相似、配置 hash 相同或单元测试通过，都不能单独证明最终输出不变。
正式 finalcase 需要相同 29 participants、冻结 5×5 split、GPU/CUDA/PyTorch/依赖，
逐 fold 比较 prediction identity、离散字段、模型/指标及结构化科学产物，浮点容差
`atol=1e-6, rtol=0`。截至当前，V5 尚未完成这次完整运行。
