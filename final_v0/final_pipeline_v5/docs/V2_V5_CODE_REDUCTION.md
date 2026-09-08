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

当前 V5 核心生产代码最终值：**69,796 行**。同一命令在本轮开始时保存的只读会话
快照（临时路径 `/tmp/v5-global-refactor-before-EEAX7K/final_pipeline_v5`）上得到
**131,179 行**，净减
**61,383 行（46.79%）**，距离 70,000 行上限留有 204 行。此前人工库存记录为
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
| 1 | `quality/stage5_pre.py` | 7,454 | 1,052 | −6,402 | 删除训练路径中的展示模板与重复 artifact 包装，保留 Stage5 数值 runner |
| 2 | `experiment.py` | 10,054 | 4,115 | −5,939 | 合并 workflow 分支、fold 产物与 refit 调度样板 |
| 3 | `reporting/analyze.py` | 5,442 | 409 | −5,033 | 用共享 registry/spec 取代逐图逐表重复分派 |
| 4 | `reporting/report.py` | 4,397 | 168 | −4,229 | 通用报告统一为 manifest 驱动的薄编排层 |
| 5 | `study/hyperparameter.py` | 3,422 | 778 | −2,644 | 复用 phase runner、统计与 artifact writer |
| 6 | `reporting/incomplete.py` | 2,724 | 188 | −2,536 | 合并 incomplete/N/A 分支与重复序列化 |
| 7 | `reporting/historical_suite.py` | 2,218 | 23 | −2,195 | 历史报告入口改为共享 specialized suite 适配器 |
| 8 | `reporting/historical.py` | 2,080 | 244 | −1,836 | 历史 comparison 使用同一收集与渲染协议 |
| 9 | `models/factory.py` | 4,148 | 2,406 | −1,742 | 表驱动模型构造，保留架构和初始化方程 |
| 10 | `config.py` | 2,465 | 1,060 | −1,405 | 合并配置解析、默认值与验证路径 |
| 11 | `study/schema.py` | 2,186 | 799 | −1,387 | 用声明式 schema 复用 plan/case 字段处理 |
| 12 | `module_registry.py` | 2,698 | 1,371 | −1,327 | 模块目录、默认值与 CLI metadata 共用一份定义 |
| 13 | `quality/motion_runner.py` | 2,363 | 1,227 | −1,136 | 统一 internal/PTT、OOF、evidence 与 bundle plumbing |
| 14 | `v5/specialized.py` | 1,497 | 404 | −1,093 | 专项执行入口共享 plan、report 与 provenance 调度 |
| 15 | `v5/model_config_export.py` | 1,291 | 260 | −1,031 | 合并 model selection、manifest 和复用参数导出 |
| 16 | `reporting/tabular.py` | 1,322 | 333 | −989 | 统一表格 spec、N/A 行和序列化 |
| 17 | `pipeline.py` | 1,715 | 749 | −966 | 顶层执行入口改为共享 workflow 编排 |
| 18 | `training/trainer.py` | 2,758 | 1,847 | −911 | 合并等价 epoch、checkpoint 与 prediction plumbing |
| 19 | `reporting/conclusions.py` | 1,585 | 705 | −880 | 共享结论表、统计投影和缺失处理 |
| 20 | `study/runner.py` | 1,774 | 911 | −863 | 统一 serial/thread/process、resume 与 fail-fast 生命周期 |

“模块保留”不等于逐字保留冗余代码。上述范围保留公开入口、数学公式、模型结构、
threshold、采样、split、workflow 顺序、统计搜索、表格字段和图表种类；删掉的是同一
逻辑的重复实现、嵌入式 presentation 和一次性历史包装。

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

当前输出为 `69796 total`。独立 Dash 包为 4,968 行，入口 `dashboard.py` 为 30 行；
二者按既定目标口径另计。tests 为 7,939 行，同样不混入生产核心。

## 等价验证边界

代码更短、AST 相似、配置 hash 相同或单元测试通过，都不能单独证明最终输出不变。
正式 finalcase 需要相同 29 participants、冻结 5×5 split、GPU/CUDA/PyTorch/依赖，
逐 fold 比较 prediction identity、离散字段、模型/指标及结构化科学产物，浮点容差
`atol=1e-6, rtol=0`。截至当前，V5 尚未完成这次完整运行。
