# V5 架构

V5 的结构目标是让一套科学 workflow 同时服务于 CLI、study YAML 和 Dash，同时将
高成本训练与可重复生成的展示分离。

## 数据流

```text
preset / complete YAML / --manual --module --set
                         │
                         ▼
             resolve + schema validation
                         │
                         ▼
 manifest + frozen split registry
                         │
                         ▼
 signal → windows → quality/artifact → features/representation
                         │
                         ▼
                 model + training
                         │
                         ▼
 per-fold OOF predictions → file/role/participant aggregation → metrics
                         │
                         ├──────────────► optional all-cohort refit
                         │
                         ▼
         pipeline_output + model_config
                         │
                         ├──────────────► analyse_report.py → report_output
                         └──────────────► infer / Dash previews
```

report 和 Dash 不另写一份科学算法：训练、推理和预览通过公共配置、模块 registry、
pipeline adapter 和 artifact reader 接入。

## 1. 配置层

`ppg_frailty.v5.configuration` 将三种输入统一为完整配置：

- 命名 preset，例如 `baseline`、`finalcase`；
- 完整 config YAML；
- `--manual` 加重复的 `--module`、`--set`、`--unset`。

模块 family 负责选择实现和关联默认值，dotted path 负责精确参数。解析顺序为基础
配置、模块、显式叶覆盖、unset、统一 schema validation。`manual-cli` 只把一个已知
配置展开成可审阅命令；实际训练仍走同一个 resolver。

模块和参数 surface 由 registry/config/study plans 动态投影：

```bash
python pipeline.py modules --help
python pipeline.py parameters --source-preset all --format markdown
```

## 2. study 与执行层

`StudyPlan` 表达 single、grid、ablation、catalog 和 legacy bridge。expansion 只负责
把计划解析为有序 cases；`StudyRunner` 对每个 case 执行声明的 repeat/fold，并统一
使用：

```text
pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
```

`pipeline.py run/ablation/grid/run-plan`、`sweep.py run` 和 Dash training 最终进入同一
V5 execution service。runner 提供 case 并发、进度、失败处理和 resume，但不重排
fold、不重算冻结 split，也不改变单 cell 内部调用顺序。

## 3. 科学 workflow

数值 workflow 的阶段为：

1. 读取 manifest、标签和冻结 participant-grouped split；
2. gap repair、PPG/IMU 滤波、静态校准和配置化重采样；
3. 共享 window planner；
4. 可选 SQI、motion、artifact/denoiser 与 routing；
5. peak/PPI/PRV/morphology 及其他 feature groups；
6. raw、feature-vector、feature-matrix 或 fusion 表征；
7. fold-local model materialization、training 和 learned bundle；
8. window→file→role→participant 聚合与 outer OOF evaluation。

并联模块通过统一 config/registry 接口切换。V5 重构可以共享 plumbing、数据类、序列化
和报告适配，但不能更改数学方程、模型架构、采样率、split 或上述调用顺序。若需要
这些科学变更，必须先单独决定并建立新 comparison/ablation 身份。

## 4. cache 与 resume

预处理 cache 只保存不依赖 outer-held-out label 的确定性中间结果。key 包含相应数据
和预处理配置身份；fold-local fitted transform、模型训练和 OOF 聚合不跨 fold 复用。
支持 `off`、`read_only`、`read_write` 及命名空间选择。

resume 以 `study_manifest`、case 状态和 fold 产物继续未完成工作。已完成 cell 复用
原有数据与 weights；run 根级 index、Excel 和 model export 可重新构建，因此短暂的
展示导出失败不使训练数据失效。

## 5. 数据输出层

cell 完成后直接写 window/file/role/participant/member OOF Parquet、metrics、训练/质量
审计数据和 model checkpoint。训练路径不导入 plotting，也不写 HTML。

run finalize 只读 cell artifacts，建立三个 CSV 索引、`v5_data_manifest.json` 和
经济型 `pipeline_data.xlsx`。Excel 是便捷副本，权威预测仍是 per-fold Parquet。

每 case 默认发布 OOF 指标排序后的中位 fold bundle。`--refit` 默认 false；打开时，
outer cells 之后为 run 中每个 case 执行既有 all-cohort refit。两类 bundle 随后进入
`model_config/<run>`。

## 6. 报告层

`analyse_report.py` 的四种 mode 为 `single`、`comparison`、`ablation` 和 `test`。
report registry 以声明表组合 audit、prediction、summary、ROC/AUC、confusion、
calibration、per-class、learning、coverage、hierarchy、quality、comparison、ablation、
ensemble、operations 和 historical 模块。

报告层只读 pipeline artifacts，按请求计算派生统计和图表，写入全新的
`report_output/<name>`。同一 pipeline run 可以用不同统计 seed、显著性单位或模块
反复分析，不影响训练结果。

## 7. model_config 与 inference

自动/独立 exporter 为每个 case 保存 resolved config、模块/参数默认值、fold 表和一份
选择后的 learned bundle。raw inference service 加载该 bundle，重放其支持的预处理、
window、model 和聚合路径，不执行 fit。

finalcase 的动态 R/S/W recording 依赖同 participant 静态 B 校准。missing-B 静默
校准是独立的 V5 TODO；当前不会在 inference 层偷偷替换训练合同。

## 8. Dash

Dash 是上述服务的本地控制面板：

- Configure 投影同一模块和参数 catalog；
- Workflow 读取 manifest 或已完成 artifacts 预览阶段结果；
- Run 构造 pipeline/sweep 请求，提供 Train、Stop、Infer 和 comparison queue；
- Analyse 构造 report 请求并预览表/图；
- Tools 暴露校验、索引、Excel/model export 和 specialized 路由。

界面展示并下载等价 CLI 与 resolved YAML。训练前必须选 YAML；Stop 终止当前后台
训练进程组；Infer 只加载 model_config 中可用的 learned bundle。

## 9. 数值等价边界

结构复用和单元测试不能替代端到端数值核验。V2/V5 等价要求相同输入、split、GPU、
CUDA、PyTorch 和依赖，以 25 个 finalcase outer cells 比较离散字段和 row identity，
浮点使用 `atol=1e-6, rtol=0`。截至当前，V5 尚未完成该 full 25-fold 运行。
