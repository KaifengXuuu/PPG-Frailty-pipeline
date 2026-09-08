# PPG Frailty Pipeline V5

PPG Frailty Pipeline V5 用 PPG 与同步 IMU recording 完成三分类 frailty
研究。它保留 V2 的数值算法、模型架构、数据划分与 workflow，同时把训练数据、
可重建报告、模型复用和现场演示拆成独立入口。

论文最终方案以可选预设 `finalcase` 提供：Rank 2
`tuned_all_roles_small_no_gravity`，运行 case ID 为
`tuned_all_roles__inception_small_no_gravity`。它不是隐式默认配置。

> 当前状态：CLI、输出合同、独立报告和 Dash 已实现；V5 尚未完成一次正式的
> 5 repeats × 5 folds finalcase 全量运行，因此现在不能声称 V2/V5 的 25-fold
> 输出已经核验一致。正式判据是在相同输入、split、GPU、CUDA、PyTorch 和依赖
> 下比较，浮点容差为 `atol=1e-6, rtol=0`。

当前核心生产代码按项目统一口径为 **69,796 行**，比可复算的重构前快照减少
61,383 行（46.79%）；统计范围、旧 sweep 的 7,824 行递归引用闭包及逐文件删减见
[代码量与精简审计](docs/V2_V5_CODE_REDUCTION.md)。Dash 与 tests 独立统计，不用测试
代码“冲抵”生产代码目标。

## 项目目的

- 复用 V2 的 signal、quality、artifact、feature、representation、model、training、
  aggregation 和 participant-grouped outer-CV 数值路径。
- 让每个算法模块及其参数可以由 YAML、纯 CLI 或 Dash 选择。
- 每 fold 保存 recording/file、role、participant、window 与 ensemble-member 预测，
  便于之后更换聚合单位和显著性计算单位而不重训。
- 训练阶段只生成数据、Excel 和 learned weights；所有图、统计展示和 HTML 由
  `analyse_report.py` 从已有结果重建。
- 每次成功 run 自动导出可复用 `model_config`；Dash 和 `pipeline.py infer` 可加载
  模型进行 no-fit participant inference。

## 功能概览

- 单配置、单因素 ablation、Cartesian grid、catalog/study sweep。
- 5×5 participant-grouped outer OOF、可恢复执行、并发和无泄漏预处理 cache。
- raw、feature-vector、feature-matrix、fusion 表征，以及经典、深度和 ensemble 模型。
- SQI、motion、artifact、peak、PRV 及论文历史 comparison 模块。
- 独立 single/comparison/ablation/test 分析；通用报告的 ROC/AUC、confusion、learning
  curves、calibration、box/distribution、paired inference 等图表和表格可组合。
- Stage5、static-peak、hyperparameter、decision-oracle 与 role-scope 通过同一报告入口
  的 specialized 子命令生成各自注册的完整套件。
- Dash 的训练、停止、no-fit 推理、comparison queue、逐阶段预览以及 CLI/YAML 下载。

模块和参数目录由运行时代码生成，不在 README 复制一份易漂移的长表：

```bash
python pipeline.py modules --help
python pipeline.py parameters --source-preset all --format markdown
python pipeline.py run --help
```

## 安装

从仓库根目录进入 V5：

```bash
cd final_v0/final_pipeline_v5
conda create -n ppg-v5 python=3.11.14
conda activate ppg-v5
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r requirements/requirements-finalcase-lock.txt
python -m pip install --no-deps -e .
python -m pip check
```

finalcase 的 exact 环境锁见
`requirements/environment-finalcase-lock.yaml`。每个新 shell 在 exact validate/run 前
设置：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

默认 `--environment-policy exact` 检查冻结的软件和硬件环境。CPU 或
`--environment-policy record` 只用于诊断，不能作为 V2/V5 数值等价证据。内部数据
不复制到 V5；manifest 中的仓库相对只读数据路径必须存在。

## 运行 finalcase：两条并行路径

### 路径一：真正纯 CLI

真正的训练命令使用 `--manual`、完整的 `--module`/`--set`/`--unset` 和执行参数，
不在训练时读取 preset、config 或 plan。完整 finalcase 有数百个叶参数，不应从文档
手抄。下面的命令只生成一条 shell-safe、可人工审阅的展开命令，本身不训练：

```bash
python pipeline.py manual-cli \
  --source-preset finalcase \
  --run-name finalcase_cli_01 > /tmp/finalcase_cli.sh
less /tmp/finalcase_cli.sh
bash /tmp/finalcase_cli.sh
```

生成器本身用完整 `--set`/`--unset` 表达每个叶值。若在审阅后的完整命令中加入
同值的 module selectors，使模块身份也直接可见，其中一段会是以下结构（省略号仅
用于展示，不能原样执行）：

```text
python pipeline.py run --manual \
  --config-id formal_inception_small_line_b_v2__tuned_all_roles_inception_small_no_gravity_fixed10 \
  --module representation=raw \
  --module model=InceptionTimeSmall \
  --module imu_gravity=sensor_filter_only_no_gravity_removal \
  --set signal.dl_resampling.target_fs_hz=64.0 \
  --set training.batch_size=16 \
  --set training.learning_rate=0.0003 \
  ...全部其余 --set/--unset... \
  --repeats all --folds all --jobs 1 --device cuda --no-continue-on-error \
  --no-measure-operational-costs --preprocessing-cache-mode read_write \
  --preprocessing-cache-root cache/preprocessing \
  --preprocessing-cache-namespaces imu_calibration,canonical_signal_views,raw_windows \
  --output-root pipeline_output \
  --environment-policy exact \
  --environment-lock requirements/environment-finalcase-lock.yaml \
  --study-id finalcase \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --run-name finalcase_cli_01
```

`manual-cli` 当前把每个叶值展开为 `--set`/`--unset`，并把 finalcase study YAML 的
5×5、失败策略、cache 和 comparison 名显式拼入命令；训练进程不再读取任何 YAML。
需要修改 workflow 时，可在
审阅后的命令中使用 `--module FAMILY=MODULE_ID` 选择模块，并用
`--set PATH=YAML_VALUE` 调参数。模块先应用，显式 `--set` 后应用。参数名、类型、
范围和 YAML 输入形式以 live catalog 为准：

```bash
python pipeline.py parameters --source-preset all --format markdown
python pipeline.py modules --help
python pipeline.py run --help
```

### 路径二：预制 study YAML

正式 one-case 5×5 计划已写入 `configs/studies/finalcase.yaml`：

```bash
python sweep.py validate \
  --plan configs/studies/finalcase.yaml \
  --environment-policy exact

python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v5_01 \
  --environment-policy exact
```

`refit` 是默认关闭的末端模块。需要全 cohort 权重时，仅在相同 run 命令上增加
`--refit`；它会在 outer-fold 训练完成后，对该 run 的每个 case 分别运行 all-29
refit。refit 模型没有内部无偏性能估计，性能证据仍来自 outer OOF。

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v5_refit_01 \
  --environment-policy exact \
  --refit
```

不加 `--refit` 时，每个使用模型的 case 仍保存全部 fold weights，并自动发布按 OOF
`(balanced_accuracy, repeat, fold)` 排序的中位 fold bundle，供复跑和 Dash 试运行。

## 输出结构

三个根目录与 README 同级：

```text
final_pipeline_v5/
├── pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
├── report_output/<run-or-report-name>/
└── model_config/<run>/cases/<comparison>/
```

- `pipeline_output`：权威 per-fold Parquet、指标/审计表、learned weights 和
  `tables/pipeline_data.xlsx`；不写 plots 或 HTML。
- `report_output`：`analyse_report.py` 生成的 figures（或显式 N/A）、CSV/JSON、
  HTML/Markdown 和报告 workbook。
- `model_config`：每 case 的 resolved config、模块/参数默认值和已选择模型包。

run 名可用 `--run-name` 指定；否则 pipeline 和 sweep 都用来源 YAML 的 stem 加 UTC
时间自动命名。已存在的 run 不会被静默覆盖，
继续执行使用 `--resume pipeline_output/<run>`。report 对同一个
顶层 run 默认写到同名子目录，所以 sweep 内多个 comparison 仍归入一个 report。
完整字段见 [输出合同](docs/OUTPUT_CONTRACT.md)。

## 分析与报告

先列出通用报告当前可组合的 mode、7 个 preset、18 个 module、36 个 figure 和
74 个 table：

```bash
python analyse_report.py list
```

验证并生成一个 run 的 classification 报告：

```bash
python analyse_report.py validate \
  --mode single \
  --input pipeline_output/finalcase_v5_01 \
  --preset classification

python analyse_report.py run \
  --mode single \
  --input pipeline_output/finalcase_v5_01 \
  --preset classification
```

comparison/ablation 可从同一 run 选择 cases；跨 run 时重复传
`--run NAME=PATH`。显式 `--figure` 或 `--table` 会精确替换 preset 的对应默认集合，
传 `none` 表示不生成该类产物。Stage5/static-peak/hyperparameter 使用
`specialized-report` 生成各自完整套件；decision-oracle/role-scope 使用
`specialized-run`。详见 [CLI 参考](docs/CLI_REFERENCE.md)。

## 模型导出与推理

每次 pipeline finalize 会自动更新 `model_config/<run>`。也可从一个已完成 run
独立重建：

```bash
python export_model_config.py \
  --pipeline-output pipeline_output/finalcase_v5_01
```

使用导出的 bundle 对输入 manifest 做 no-fit 推理：

```bash
python pipeline.py infer \
  --model-config model_config/finalcase_v5_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --input-manifest path/to/participant.yaml
```

动态 R/S/W 输入当前必须同时提供同 participant 的静态 B recording 完成校准。
缺少 B 的静默校准是 **V5 TODO**，尚未实现；当前会明确拒绝该输入。

## Dash 本地操作面板

```bash
python dashboard.py --host 127.0.0.1 --port 8050
```

打开 `http://127.0.0.1:8050`。Dash 提供与 CLI 同源的配置与执行面板：

- YAML/model_config 加载、模块下拉、参数表和有限范围滑条；
- `Train`、随时生效的 `Stop`、以及加载预训练 bundle 的 `Infer`；
- comparison 临时队列，可多次添加配置后训练；
- signal、window、quality、feature、model、prediction、aggregation 等阶段预览；
- pipeline 表、report 图表与表格预览；
- 当前请求及整个 comparison 序列的等价 CLI、resolved YAML 下载。

训练前必须选择 YAML；推理不执行 fit。单 participant 只能展示 QC、概率与分类；
ROC/AUC、cohort confusion matrix 和显著性检验需要多 participant、足够类别覆盖的
标注数据。

## 数值等价检查

完整 finalcase 运行后，使用相同 V2 case 做 25-fold 对比：

```bash
python tools/compare_v2_v5_outputs.py \
  --v2-output <V2-case-directory> \
  --v5-output pipeline_output/finalcase_v5_01/tuned_all_roles__inception_small_no_gravity \
  --expected-folds 25 \
  --atol 1e-6 \
  --write pipeline_output/finalcase_v5_01/v2_v5_numeric_equivalence.json
```

配置解析或源码静态对齐不能替代完整输出比较。

## 文档

- [CLI_REFERENCE.md](docs/CLI_REFERENCE.md)：所有入口、纯 CLI 和 study/sweep 用法。
- [OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md)：三根目录及数据、Excel、权重合同。
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)：运行层、科学层、报告层和 Dash 的关系。
- [THESIS_CODE_CONFLICTS.md](docs/THESIS_CODE_CONFLICTS.md)：论文与实现冲突，按影响排序。
- [V2_V5_CODE_REDUCTION.md](docs/V2_V5_CODE_REDUCTION.md)：统一口径的代码量与删减审计。
- [PLAN_COMPATIBILITY.md](docs/PLAN_COMPATIBILITY.md)：canonical 与历史专项 plan 的入口。
