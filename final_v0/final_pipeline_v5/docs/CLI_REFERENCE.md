# V5 CLI 操作参考

本文记录当前公开 CLI。所有示例均从 `final_pipeline_v5/` 根目录运行。训练入口只写
`pipeline_output` 与 `model_config`；图形、展示表和 report Excel 由
`analyse_report.py` 写入 `report_output`。

## 运行前

finalcase 数值运行使用冻结环境：

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python pipeline.py validate \
  --preset finalcase \
  --mode full \
  --environment-policy exact
```

`exact` 是默认 policy。`record` 或 CPU 可用于检查接口，但不能支持数值等价结论。
V2/V5 正式浮点比较使用 `atol=1e-6, rtol=0`。

## 命令地图

| 入口 | 子命令 | 作用 |
|---|---|---|
| `pipeline.py` | `run` | 一个完整配置 |
|  | `ablation` | 单一 dotted path 的多值比较 |
|  | `grid` | 多个 dotted paths 的 Cartesian 比较 |
|  | `run-plan` | 执行 study-plan YAML |
|  | `validate`, `show-config` | 校验或显示 resolved config |
|  | `modules`, `presets`, `parameters` | 查看实时配置目录 |
|  | `manual-cli` | 生成一条完整纯 CLI 命令，不执行训练 |
|  | `infer` | 加载已训练 bundle，no-fit 推理 |
|  | `index`, `export-excel`, `export-model-config` | 重建数据索引和导出 |
| `sweep.py` | `validate`, `run` | 校验/执行预配置 study YAML |
|  | `export-excel` | 重建 pipeline Excel |
| `analyse_report.py` | `list`, `validate`, `run` | 列目录、只读校验、生成报告 |
|  | `export-excel` | 从已有通用 report CSV 重建 report Excel |
|  | `specialized-*` | 历史专项分析/报告适配入口 |
| `export_model_config.py` | — | 从一个完成的 run 独立导出模型配置 |
| `dashboard.py` | — | 启动本地可视化面板 |

任何时候以解析器输出为准：

```bash
python pipeline.py --help
python pipeline.py run --help
python sweep.py run --help
python analyse_report.py run --help
```

## 模块与全部参数

不要从文档手抄静态参数表。以下命令实时给出模块 ID、全部 485 个跨
preset/study/module 联合叶路径、类型、默认值、范围和 YAML 输入形式：

```bash
python pipeline.py modules --help
python pipeline.py modules --family model
python pipeline.py parameters --source-preset all --format markdown
```

只看 finalcase 的 254 个配置叶：

```bash
python pipeline.py parameters --source-preset finalcase --format yaml
python pipeline.py parameters --help
```

常用输入形式：

```text
--module FAMILY=MODULE_ID
--set PATH=YAML_VALUE
--unset PATH
--config-id SAFE_ID
```

`--module` 处理一个模块及其关联默认值，之后 `--set` 覆盖具体叶参数；因此编辑
展开命令时，应同步删除或更新与新模块冲突的旧 `--set`。数组、mapping、字符串和
`null` 均按 YAML 解析，shell 中通常需要单引号。

## 配置来源

`run`、`ablation` 和 `grid` 必须且只能选择一种来源：

```text
--preset NAME
--config path/to/complete.yaml
--manual
```

三种来源最终都经过同一 schema validator，形成完整 resolved config 后进入同一
runner。`--manual` 还要求显式 `--config-id`，并接受重复的
`--module`/`--set`/`--unset`。

### 真正纯 CLI 的 finalcase

`manual-cli` 只是一个命令生成器：它读取 `finalcase` 一次，把所有叶值展开成
shell-safe 命令并退出，不训练。审阅并执行生成文件后，真正的训练进程只收到
`--manual` 与显式参数，不再选择 preset/config/plan。

```bash
python pipeline.py manual-cli \
  --source-preset finalcase \
  --run-name finalcase_cli_01 > /tmp/finalcase_cli.sh
less /tmp/finalcase_cli.sh
bash /tmp/finalcase_cli.sh
```

生成文件包含完整 `--set`/`--unset`，还显式带出 finalcase study YAML 的 5×5、CUDA、
`--no-continue-on-error`、cache root/namespaces、output、study ID 和 comparison case ID；
它还显式记录 exact environment policy 和 lock 路径。训练进程不再读取配置或 plan
YAML（environment lock 仍是冻结依赖证据）。refit 默认关闭，因此无需写一个反向开关。
若希望以模块名审阅或改造它，可加入例如：

```text
--module representation=raw
--module model=InceptionTimeSmall
--module imu_gravity=sensor_filter_only_no_gravity_removal
--module optimizer=adamw
--set training.batch_size=16
--set training.learning_rate=0.0003
```

这些是完整展开命令中的结构示例，不是可单独运行的 finalcase 命令。插入新模块时
要处理其后同路径的显式 `--set`，并先用 `show-config --manual ...` 或
`validate --manual ...` 审查最终值。

### preset/config 入口

```bash
python pipeline.py show-config --preset finalcase

python pipeline.py run \
  --config configs/presets/finalcase.yaml \
  --run-name finalcase_config_01 \
  --repeats all --folds all --jobs 1 --device cuda \
  --preprocessing-cache-mode read_write \
  --environment-policy exact
```

在 preset/config 上仍可追加 `--module`、`--set` 和 `--unset`，用于建立清楚的派生
配置；resolved config 会写入 run。

## execution 参数

详细定义以 `python pipeline.py run --help` 为准。核心形式如下：

| 参数 | 输入 | 说明 |
|---|---|---|
| `--repeats` | `all` 或 `0,1,...,4` | outer repeats 子集 |
| `--folds` | `all` 或 `0,1,...,4` | outer folds 子集 |
| `--jobs` | 正整数 | case 并发数 |
| `--device` | `cuda` / `cpu` | 也会解析到 training device |
| `--continue-on-error` | BooleanOptionalAction | case 失败后是否继续 |
| `--measure-operational-costs` | BooleanOptionalAction | 采集非科学运行成本 |
| `--preprocessing-cache-mode` | `off/read_only/read_write` | 无泄漏预处理 cache |
| `--preprocessing-cache-root` | path | cache 根目录 |
| `--preprocessing-cache-namespaces` | 逗号列表 | cache 命名空间 |
| `--output-root` | path | 默认 `pipeline_output` |
| `--run-name` | 安全单目录名 | 新 run 名；缺省自动命名 |
| `--case-id` | 安全单目录名 | 仅 single `run`；指定 comparison 子目录名 |
| `--resume` | 已有 run path | 从已有 run 恢复 |
| `--hash-predictions` | flag | 索引预测文件哈希 |
| `--dry-run` | flag | materialize/检查而不训练 |
| `--environment-policy` | `exact/record` | 环境检查方式 |
| `--environment-lock` | YAML path | 环境锁 |
| `--refit` | flag | 缺省不 refit；显式加入后执行末端 all-cohort refit |

新 run 不覆盖同名目录；恢复时使用 `--resume pipeline_output/<run>`。不指定
`--run-name` 时，config CLI 与 study/sweep 都用来源 YAML stem 加 UTC 时间命名；
纯 `--manual` 使用 `manual` 加 UTC 时间。正式运行建议总是给出可读的 `--run-name`。

## ablation 与 grid

单因素 ablation：

```bash
python pipeline.py ablation \
  --preset finalcase \
  --study-id gravity_ablation \
  --factor signal.imu.gravity_method \
  --values sensor_filter_only_no_gravity_removal profile_a_lowpass_0p3hz \
  --reference-value sensor_filter_only_no_gravity_removal \
  --run-name gravity_ablation_01 \
  --repeats all --folds all --jobs 1 --device cuda
```

Cartesian grid；每个 `--vary` 的右侧必须是至少两个值的 YAML list：

```bash
python pipeline.py grid \
  --preset finalcase \
  --study-id optimizer_grid \
  --vary 'training.learning_rate=[0.0001,0.0003]' \
  --vary 'training.weight_decay=[0.0001,0.001]' \
  --reference training.learning_rate=0.0003 \
  --reference training.weight_decay=0.001 \
  --run-name optimizer_grid_01 \
  --repeats all --folds all --jobs 1 --device cuda
```

`--module` 适合选择一个模块实现；`--factor`/`--vary` 用于定义会形成 comparison
cases 的实验轴。

## finalcase study YAML

推荐以预配置计划执行正式整套 5×5：

```bash
python sweep.py validate \
  --plan configs/studies/finalcase.yaml \
  --environment-policy exact

python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v5_01 \
  --environment-policy exact
```

`pipeline.py run-plan --plan ...` 使用同一执行服务；`sweep.py` 是 study 用户的精简
入口。plan 的每个 case 都写入同一 `<run>/<comparison>/repeat/fold` 树。

### refit

默认不 refit。需要 refit 时只增加一个开关：

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v5_refit_01 \
  --environment-policy exact \
  --refit
```

outer cells 完成后，该 run 中每个 case 各自运行一次 all-29 refit。关闭 refit 仍会
保存每 fold weights，并为每 case 导出 OOF 指标排序后的中位 fold bundle。all-29
bundle 不做内部 self-evaluation；报告性能仍取 outer OOF。

## 恢复、索引和导出

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --resume pipeline_output/finalcase_v5_01 \
  --environment-policy exact

python pipeline.py index \
  --study-dir pipeline_output/finalcase_v5_01 \
  --hash-predictions

python pipeline.py export-excel \
  --pipeline-output pipeline_output/finalcase_v5_01

python export_model_config.py \
  --pipeline-output pipeline_output/finalcase_v5_01
```

导出命令默认不覆盖已有目标；只有明确需要重建时才加 `--replace`。pipeline 正常
finalize 会自动建立索引、Excel 和 `model_config/<run>`。

## analyse_report

查看通用报告可用的 mode、7 个 presets、18 个 modules、36 个 figures 和 74 个
tables：

```bash
python analyse_report.py list
```

一个 sweep run 已含其全部 comparison cases，直接以顶层 run 为输入，report 默认
沿用 run 名。下例的 `<REFERENCE_CASE>` 必须替换为该 run 中真实的 reference case ID：

```bash
python analyse_report.py validate \
  --mode comparison \
  --input pipeline_output/gravity_ablation_01 \
  --reference-case <REFERENCE_CASE> \
  --preset comparison

python analyse_report.py run \
  --mode comparison \
  --input pipeline_output/gravity_ablation_01 \
  --reference-case <REFERENCE_CASE> \
  --preset comparison
```

结果为 `report_output/gravity_ablation_01/`。ablation 还传一个或多个
`--factor-path`；test 只接受输入中明确存在的独立 test evidence。内部 outer OOF
不能改名为独立 test。

跨 run 对比可重复给出具名输入：

```bash
python analyse_report.py run \
  --mode comparison \
  --run old=pipeline_output/run_old \
  --run new=pipeline_output/run_new \
  --reference-case <REFERENCE_CASE> \
  --preset comparison \
  --output-name old_vs_new
```

以下选择规则属于通用 `validate/run`：

- `--include-case` / `--exclude-case` 可重复使用；
- `--preset classification|comparison|ablation|test|minimal|ensemble|full`；
- `--module` 可重复组合分析模块；
- 显式 `--figure` / `--table` 精确替换默认集合，`none` 选择空集合；
- `--bootstrap-resamples`、`--permutation-resamples`、`--statistics-seed`、
  `--alpha` 和 `--calibration-bins` 控制报告统计；
- `--on-missing error|na|skip` 定义缺失输入策略。

从已有 CSV 表重建通用 report Excel：

```bash
python analyse_report.py export-excel \
  --report-output report_output/gravity_ablation_01
```

该命令要求通用报告的 `analysis_manifest.json`。Stage5/static-peak/hyperparameter 的
`specialized-report` 按各自注册的完整图表套件生成 workbook；decision-oracle 与
role-scope 使用 `specialized-run`。详细路由见
[PLAN_COMPATIBILITY.md](PLAN_COMPATIBILITY.md)。

## no-fit inference

```bash
python pipeline.py infer \
  --model-config model_config/finalcase_v5_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --input-manifest path/to/participant.yaml
```

输入 manifest 描述同一个 participant 的一条或多条 recording。动态 R/S/W 当前
必须同时提供静态 B 校准记录；missing-B 静默校准是未实现的 V5 TODO。推理只加载
bundle，不更新权重。

## Dash

```bash
python dashboard.py --host 127.0.0.1 --port 8050
```

Dash 只监听 loopback。训练必须先选择 YAML；`Train` 与 `Stop` 分离，`Infer` 加载
已导出的 bundle。Configure、Workflow、Run、Analyse、Tools 页面覆盖模块/参数、
comparison queue、逐阶段预览、pipeline/report 预览，以及等价 CLI 和 resolved YAML
下载。
