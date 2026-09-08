# V5 输出合同

V5 将训练数据、可重建展示和模型复用分到三个并列根目录。pipeline 不生成图；
公开 report CLI 不训练，也不改写作为输入的 pipeline 数据。

## 目录层次

```text
final_pipeline_v5/
├── pipeline_output/
│   └── <run>/
│       ├── <comparison>/
│       │   └── repeat_<RR>/fold_<FF>/
│       ├── tables/
│       └── models/
├── report_output/
│   └── <run-or-report-name>/
└── model_config/
    └── <run>/cases/<comparison>/
```

`<comparison>` 是一个 resolved case 的目录身份，准确映射记录在
`study_manifest.json`。repeat/fold 使用两位零填充，例如
`repeat_00/fold_03`。single、ablation、grid 和 catalog sweep 均使用相同层次。

## run 命名和恢复

- `--run-name NAME` 创建 `pipeline_output/NAME`。
- 未给名称时，`pipeline.py run/ablation/grid` 从配置来源 YAML stem（纯 manual 为
  `manual`）和 UTC 时间生成名称；study/sweep 同样使用 plan YAML stem 加 UTC 时间。
- 新 run 不静默覆盖同名目录。
- 未完成或中断的 run 用 `--resume pipeline_output/<run>` 继续。
- `sweep.py` 的一次 run 可以包含多个 comparison；它们不拆成多个顶层 run。

`pipeline_output` 和 `model_config` 可以独立提交到 Git。普通报告及 Stage5/static/
hyper 报告可由 pipeline 输出重建，不要求提交 `report_output`；decision-oracle 与
role-scope 等 analysis-only 历史计划还可能显式读取 `--source-root` 下的 V2 artifact。

## 每 fold 的权威产物

每个成功 cell 位于：

```text
pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
```

核心数据包括：

| 文件/类别 | 含义 |
|---|---|
| `oof_window_predictions.parquet` | window-level OOF 概率和身份 |
| `oof_file_predictions.parquet` | recording/file-level OOF 概率 |
| `oof_role_predictions.parquet` | role-level OOF 概率 |
| `oof_subject_predictions.parquet` | participant-level OOF 概率 |
| `oof_member_predictions.parquet` | ensemble member 概率；不适用时为空状态 artifact |
| `metrics_per_fold_seed.json` | 该 repeat/fold 的指标、seed 与运行摘要 |
| `model_checkpoint/` | 可重载 learned weights、模型输入合同及 golden sample |
| 质量/route/feature/training 文件 | 所选模块产生的结构化审计数据 |

所有层保留 `case_id`、repeat、fold、row/record/participant 身份、真实标签、预测类别
和 class probabilities 等适用字段。file 层是后续改换 file→role→participant 聚合或
file-level 显著性分析的直接输入；原始声明聚合仍由 resolved config 决定。

Parquet 是权威预测格式。空或不适用的可选层以显式状态表示，不用虚构记录填充。

## run 根级索引

训练 finalize 从已写入的 cell 只读建立：

```text
study_manifest.json
study_run_result.json
v5_data_manifest.json
tables/v5_fold_predictions.csv
tables/v5_fold_models.csv
tables/v5_config_parameters.csv
tables/pipeline_data.xlsx
pipeline_excel_status.json
models/<case>/median_fold/selection.json
```

- `study_manifest.json` 定义 case、配置路径、执行范围及状态。
- `v5_fold_predictions.csv` 是所有 per-fold Parquet 的轻量路径/shape 索引。
- `v5_fold_models.csv` 汇总每 fold 的模型、指标、provenance 和 checkpoint 路径。
- `v5_config_parameters.csv` 将 resolved config 展平成可检索表。
- `v5_data_manifest.json` 汇总完整性问题、发布模型和上述表路径。

### pipeline Excel

`tables/pipeline_data.xlsx` 是方便人工查看的经济视图，不是新的权威数据源。它包含
三个 compact index，以及容量允许时的 file、role、participant 和 member 预测。
window 层始终不复制进工作簿；任何预测层超过 Excel 行限制时也只在 status 中标为
skipped，继续使用 per-fold Parquet。

Excel 是可恢复后处理。生成失败不会删除已经完成的训练数据：

```bash
python pipeline.py export-excel \
  --pipeline-output pipeline_output/<run>
```

只有明确重建时加 `--replace`。

## learned weights 与 refit

每个成功 fold 保存一个可重载 bundle，并在保存后检查 golden prediction。每个 case
默认按 `(balanced_accuracy, repeat, fold)` 排序选择中位 fold；偶数候选使用 lower
middle。`models/<case>/median_fold/selection.json` 只引用权威 fold bundle，不重复复制
权重。

`refit` 默认 false。命令增加 `--refit` 后，outer-fold 训练结束，再对 run 中每个
case 分别执行 all-29 refit：

```text
models/<case>/all29_refit/
v5_refit_manifest.json
```

all-29 refit 不产生内部 self-evaluation；性能证据仍是 outer OOF。关闭 refit 时仍
有全部 fold weights 和中位 fold 发布模型。

## model_config

pipeline finalize 自动生成：

```text
model_config/<run>/
├── export_manifest.json
├── available_modules.json
└── cases/<comparison>/
    ├── resolved_pipeline_config.yaml
    ├── pipeline_module_defaults.yaml
    ├── model_reuse_parameters.yaml
    ├── fold_model_parameters.csv
    └── learned_model/
```

每个 case 导出 resolved config、模块/参数默认值、fold provenance 和一个选择后的
bundle。存在成功 all-29 refit 时优先导出它，否则导出中位 fold bundle。并非所有
历史表示/模型都支持 raw new-participant inference；能力和原因写在
`export_manifest.json`。

可从已完成 run 独立重建：

```bash
python export_model_config.py --pipeline-output pipeline_output/<run>
```

默认不覆盖已有导出；需要重建时使用 `--replace`。

## report_output

通用 `analyse_report.py run` 只读一个或多个 pipeline run，输出：

```text
report_output/<name>/
├── analysis_manifest.json
├── outputs_index.json
├── STUDY_SUMMARY.md
├── STUDY_SUMMARY.html
├── figures/*.png 或 figures/*.NA.txt
└── tables/
    ├── *.csv
    ├── *.json
    └── report_tables.xlsx
```

`pipeline_output/<run>` 作为单输入时，默认 report 名就是顶层 `<run>`；因此一个
sweep 的 comparison/ablation 不需要另给输出名。跨多个不相关 run 的分析可用
`--output-name` 指定一个新名字。report 目标必须是新目录，原有 report 不会被静默
覆盖。

Stage5/static-peak/hyperparameter 的 `specialized-report` 使用
`report_manifest.json`；hyper 另含 `phases/`。它们同样输出 summary、figures、CSV/JSON
和 workbook，但不伪造通用报告的 `analysis_manifest.json`/`outputs_index.json`。

report Excel 与 pipeline Excel 语义不同：

- `pipeline_data.xlsx` 是训练数据和预测索引的便捷视图；
- `report_tables.xlsx` 是指定分析参数生成的 derived tables 汇总。

通用 report Excel 同样可从已有 CSV 重建：

```bash
python analyse_report.py export-excel \
  --report-output report_output/<name>
```

该 `export-excel` 子命令要求通用报告的 `analysis_manifest.json`；specialized workbook
随 `specialized-report` 一次生成。

## 完整性与数值等价

manifest、索引、bundle 和输出表记录必要的路径、schema 与哈希。恢复和重建只读取
已完成 cell，不重新定义 split、模型或聚合算法。完整性通过不等于数值等价；正式
V2/V5 结论还必须完成相同冻结环境下的 25-fold 输出比较，浮点判据
`atol=1e-6, rtol=0`。截至当前，V5 尚未完成这次 full finalcase 运行。
