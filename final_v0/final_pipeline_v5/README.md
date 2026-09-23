# PPG Frailty Pipeline V5

Frailty（衰弱）是多个生理系统累积衰退所形成的状态，表现为生理储备减少、
稳态恢复能力下降，以及对外界压力源的易感性增加。较小的应激也可能引起明显的
健康变化。衰弱状态可以随时间改善或恶化，因此，低成本、可重复的测量对于研究
衰弱及前衰弱具有现实意义。

光电容积描记（photoplethysmography，PPG）通过光学传感器无创记录外周血容量
的脉动变化，脉搏的时序和形态可提供心血管及自主神经调节相关信息。惯性测量单元
（inertial measurement unit，IMU）通过三轴加速度和三轴角速度描述平移与旋转
运动。本项目使用同步采集的红光 PPG、红外 PPG 和六轴 IMU 共八通道信号，
将同一时段的生理变化与运动情境联系起来，并利用运动信息辅助识别和处理 PPG
运动伪影。

传统衰弱评估主要依赖问卷、体能测试和临床判断，通常按次进行。可穿戴 PPG 与
IMU 为重复测量提供了补充途径，但接触条件、运动干扰和同一受试者的重复记录会
影响信号解释与分类评价。本项目据此构建透明、模块化的 Python 分析流程，研究
同步信号能否支持可复现的受试者级分组辨别，并为后续远程监测研究提供基础。
研究背景和三个目标依据[论文草稿](../../Kaifeng_Masterarbeit_draft_v1_0.docx)
的 Introduction 与 Conclusion。

## 项目目的

1. 建立透明、可复现的同步 PPG/IMU 信号处理流程，涵盖预处理、脉搏峰检测、
   运动检测与运动伪影抑制。
2. 通过 PPG 衍生特征和受试者级分类，探索记录中的生理与运动相关信息，比较
   前衰弱老年组、非衰弱老年组和青年参考组。
3. 通过受试者独立的数据划分、与已有参考实现的比较，以及不同模型和信号处理
   配置的评估，检验整个 workflow 的可靠性。

本项目定位为工程方法研究和探索性分类。Young 是年龄参考组，并非衰弱等级；
三组分类结果本身不能证明衰弱特异性生物标志物或临床诊断效度。

## 数据与功能概览

论文数据包含 29 名受试者：9 名 Pre-Frail、12 名 Robust/Non-Frail 和 8 名 Young。
每人提供 9 条同步记录，共 261 条，包括静态基线 B、运动后恢复 R1–R4，以及
运动任务 S1/S2/W1/W2。manifest 保存原始数据路径、受试者、标签和 recording
角色；原始数据独立存放，使用时需要相应数据访问权限。

训练和评价采用以下 workflow。YAML、纯 CLI 和 Dash 共同配置同一套算法：

```text
manifest / 标签 / participant-grouped splits
    → PPG 与 IMU 预处理、校准和重采样
    → 分窗、质量评估、可选运动检测与伪影抑制
    → 特征提取或原始信号表征
    → fold 内训练与 held-out 预测
    → window → recording → role → participant 概率聚合
    → 每 fold 数据与权重 → pipeline_output
         ├─ fold 模型 / 可选全 cohort refit → model_config → 新 participant 推理
         └─ 统计分析与图表生成 → report_output
```

| 流程阶段 | 模块与功能 | 主要入口或实现位置 |
|---|---|---|
| 数据和实验计划 | 读取 manifest、标签与受试者分组 split；single、comparison、ablation、grid 和重复交叉验证 | `pipeline.py`、`sweep.py`；`data/`、`study/` |
| 信号预处理 | 缺失片段处理、PPG/IMU 滤波、静态校准、重采样、分窗与归一化；复用确定性预处理 cache | `signal/`、`data/windows.py`、`data/preprocessing_cache.py` |
| 质量与运动处理 | SQI、窗口选择、运动检测、artifact reducer 和 denoiser 等可选分支 | `quality/`、`artifacts/` |
| 特征和表征 | peak/PPI/PRV、脉搏形态与其他特征；raw、feature-vector、feature-matrix 和 fusion | `features/`、`representations/` |
| 模型与评价 | 经典模型、深度模型及 ensemble；fold 内训练、OOF 预测、分层聚合和指标 | `models/`、`training/` |
| 数据和模型输出 | 每 fold 预测与 learned weights、数据 Excel、模型选择、可选 refit 和模型复用参数 | `pipeline.py`、`export_model_config.py` |
| 分析与报告 | ROC/AUC、confusion matrix、learning curves、calibration、box plots、配对检验及专项比较 | `analyse_report.py`；`reporting/`、`v5_reporting/` |
| 交互操作 | 配置、训练与停止、预训练推理、comparison 队列、逐阶段预览和图表展示 | `dashboard.py` |

表中的模块目录均位于 `src/ppg_frailty/`。训练所需的拟合步骤仅使用当前训练
受试者；预处理 cache 不跨 fold 共享由训练数据拟合得到的状态。预测在每 fold
保存，后续可从同一批结果选择不同聚合方式、分析单位和图表，无需重新训练。

## 安装与环境复现

从仓库根目录进入 V5：

```bash
cd final_v0/final_pipeline_v5
```

一般安装使用 [pyproject.toml](pyproject.toml) 中的依赖范围，要求 Python 3.11
或以上；按所需功能安装 deep、reporting 和 dashboard：

```bash
conda create -n ppg-v5 python=3.11
conda activate ppg-v5
python -m pip install -e '.[deep,reporting,dashboard]'
python -m pip check
```

pipeline 模块测试及模型预测数值复现所用的参考版本清单位于
[requirements/requirements-finalcase.txt](requirements/requirements-finalcase.txt)。
参考环境为 Python 3.11.14、NVIDIA GeForce RTX 4080（驱动 560.81）、CUDA 12.6 和
PyTorch 2.9.1+cu126。需要同步该环境时，新建独立环境并安装清单中的精确版本：

```bash
conda create -n ppg-v5-finalcase python=3.11.14
conda activate ppg-v5-finalcase
python -m pip install -r requirements/requirements-finalcase.txt
python -m pip install --no-deps -e .
python -m pip check
```

清单已包含 CUDA 12.6 PyTorch wheel 的下载源。GPU 和驱动需在运行机器上单独
准备；pip 安装 CUDA 依赖不能替代 NVIDIA 驱动。
上述精确版本是测试和结果复现的参考，运行时不要求软件版本或 GPU 型号与清单
逐项匹配。依赖升级可通过修改安装清单进行，升级后应运行模块测试和所需的输出
数值对比。训练命令可用 `--device cpu` 选择 CPU（sweep 在 YAML 中配置设备）；
设备及数值库变化可能影响浮点结果。确定性 CUDA 训练会自动补齐缺失的
`CUBLAS_WORKSPACE_CONFIG=:4096:8`；已显式选择的后端设置不会被环境清单覆盖。

测试依赖与测试入口：

```bash
python -m pip install -e '.[test]'
python -m pytest
```

## 配置模块与参数

每个模块有默认参数，可通过 YAML、CLI 或 Dash 选择。CLI 接受重复的
`--module FAMILY=MODULE_ID`、`--set PATH=YAML_VALUE` 和 `--unset PATH`；模块
默认值先应用，显式叶参数随后覆盖。布尔值、列表、数值和字符串采用 YAML
输入形式，含空格或列表的值在 shell 中加引号。

完整模块名称、可调参数、类型、范围和默认值由代码中的 registry 生成：

```bash
python pipeline.py modules --help
python pipeline.py modules
python pipeline.py parameters --help
python pipeline.py parameters --source-preset all --format markdown
python pipeline.py run --help
python sweep.py --help
python analyse_report.py --help
```

## 运行 finalcase

论文最终方案作为可选预设 `finalcase` 提供，对应 Rank 2
`tuned_all_roles_small_no_gravity`，case ID 为
`tuned_all_roles__inception_small_no_gravity`。它使用全部 B/R/S/W 角色、64 Hz
八通道 raw 输入、5 秒窗口、InceptionTimeSmall 和固定 10 epochs，以
5 repeats × 5 participant-grouped folds 生成 OOF 预测。其他实验可独立选择
模块和参数。

### 纯 CLI

`pipeline.py run --manual` 接受完整的模块与参数定义，训练时不读取预设 YAML。
finalcase 的完整参数较多，可先展开成 shell 命令，再人工修改和执行：

```bash
python pipeline.py manual-cli \
  --source-preset finalcase \
  --run-name finalcase_cli_01 > /tmp/finalcase_cli.sh
less /tmp/finalcase_cli.sh
bash /tmp/finalcase_cli.sh
```

`manual-cli` 只生成命令；`--source-preset` 用于生成阶段。生成结果包含每个叶值的
`--set`/`--unset`、repeat/fold、cache 和输出参数，可脱离原 YAML 使用。也可直接
手写完整 `--manual` 命令。下例仅展示其中的可编辑参数片段：

```text
--module representation=raw
--module model=InceptionTimeSmall
--module imu_gravity=sensor_filter_only_no_gravity_removal
--set signal.dl_resampling.target_fs_hz=64.0
--set training.batch_size=16
--set training.learning_rate=0.0003
--set 'training.classifier_role_families=[B,R,S,W]'
```

### 预制 YAML

[configs/studies/finalcase.yaml](configs/studies/finalcase.yaml) 保存完整的 finalcase
study 计划，可先验证，再运行：

```bash
python sweep.py validate --plan configs/studies/finalcase.yaml
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v5_01
```

comparison 和 ablation 在同一 study 中定义多个 case，仍按统一 repeat/fold 流程
运行。论文各组实验、历史参数对应关系及运行命令见
[论文实验 YAML 索引](configs/studies/thesis/README.md)。通用模板与论文历史实验配置
分开存放，不能仅凭相同文件名认定为相同实验。更多计划及命令见 [CLI 参考](docs/CLI_REFERENCE.md) 和
[计划兼容性说明](docs/PLAN_COMPATIBILITY.md)。

`refit` 默认关闭。需要全 cohort 训练权重时增加 `--refit`，它会在 outer-fold
训练完成后为每个 case 执行 refit：

```bash
python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_refit_01 \
  --refit
```

关闭 refit 时仍保存全部 fold weights，并按 OOF
`(balanced_accuracy, repeat, fold)` 排序发布中位 fold 模型。refit 权重用于模型
复用，性能评价仍采用 held-out OOF 预测。

## 预处理 cache 开关

cache 复用不依赖标签、未经过 fold 内拟合的预处理结果，减少重复滤波、校准和
分窗的计算。它不是预测正确性的必要条件；关闭后会重新计算，不会删除已有 cache。
finalcase 的 study YAML 和 `manual-cli` 生成命令使用 `read_write`，目录为 V5 下的
`cache/preprocessing`。直接 `pipeline.py run` 未指定 cache 参数时，执行层默认
为 `off`，默认目录为 `artifacts/studies/cache`；缓存参数不属于 pipeline 预设本身。

| 模式 | 行为 |
|---|---|
| `off` | 不读写预处理 cache，每次重新计算。 |
| `read_only` | 命中时读取；未命中时计算但不新增 cache。 |
| `read_write` | 命中时读取；未命中时计算并保存。 |

直接运行 pipeline 时，通过 CLI 选择模式；下面是关闭 cache 的完整示例：

```bash
python pipeline.py run --preset finalcase \
  --run-name finalcase_no_cache_01 \
  --preprocessing-cache-mode off \
  --preprocessing-cache-root cache/preprocessing
```

将 `off` 换成 `read_only` 或 `read_write` 即可切换。使用 `--manual` 时也接受同一
参数；`manual-cli` 生成的命令中已有该参数，直接修改其值即可。
`--preprocessing-cache-root cache/preprocessing` 可指定 V5 内的缓存目录，
`--preprocessing-cache-namespaces imu_calibration,canonical_signal_views,raw_windows`
可选择缓存层；可选层另有 `motion_windows`，仅在对应运动分支运行时使用。
减少缓存层不会关闭相应算法，只会让未缓存的阶段重新计算。

使用 `sweep.py` 时，在所选 study YAML 的现有 `execution` 下修改
`preprocessing_cache`，保留其他 execution 配置，再照常运行计划：

```yaml
execution:
  # 保留原有 repeats、folds、device 等字段。
  preprocessing_cache:
    mode: off  # 或 read_only、read_write
    root: cache/preprocessing
    namespaces: [imu_calibration, canonical_signal_views, raw_windows]
    verify_source_sha256: true
```

Dash 的模型模块在 Train 模式下可用 cache 下拉框控制普通训练；sweep 使用所选 study YAML
中的 cache 配置。已有 pipeline 结果的报告生成及已导出模型的推理不需要这些
预处理 cache。清理磁盘前应确认没有训练任务正在使用缓存目录；关闭开关本身
不会释放已有文件占用的空间。

## 输出结构

三个输出根目录与 README 同级：

```text
final_pipeline_v5/
├── pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/
├── report_output/<run-or-report-name>/
└── model_config/<run>/cases/<comparison>/
```

| 目录 | 内容 |
|---|---|
| `pipeline_output` | 每 fold 的 window、recording/file、role、participant 和适用的 ensemble-member 预测；指标、learned weights、CSV 索引和 `tables/pipeline_data.xlsx` |
| `report_output` | 分析生成的 figures、CSV/JSON 派生统计表、HTML/Markdown 和报告 Excel |
| `model_config` | 每 case 的 resolved config、模块开关和默认参数、模型复用参数及所选 learned bundle |

Parquet 保存权威预测数据，pipeline Excel 提供数据与索引的便捷视图；report Excel
保存所选分析产生的派生统计。pipeline 不生成 plots 或 HTML，已有训练结果可用于
多次独立报告生成。目录和字段详见 [输出合同](docs/OUTPUT_CONTRACT.md)。

使用 `--run-name NAME` 命名 run；省略时从配置或计划名称加 UTC 时间生成名称。
已有运行可通过 `--resume pipeline_output/<run>` 继续。报告默认使用顶层 run 名，
同一 sweep 内的多个 comparison 共用该报告目录；另一次分析可用 `--output-name`
指定新的目录名。

## 分析与报告

`analyse_report.py` 从 pipeline 产物生成报告，支持 `single`、`comparison`、
`ablation` 和 `test`。先查看可组合的 preset、module、figure 和 table：

```bash
python analyse_report.py list
python analyse_report.py run --help
```

生成一个 run 的 classification 报告：

```bash
python analyse_report.py validate \
  --mode single --input pipeline_output/finalcase_v5_01 --preset classification
python analyse_report.py run \
  --mode single --input pipeline_output/finalcase_v5_01 --preset classification
```

comparison/ablation 可选择同一 run 内的 cases；跨 run 时重复传
`--run NAME=PATH`。显式 `--figure` 或 `--table` 替换 preset 的对应默认集合，
`none` 表示不生成该类产物。Stage5/static-peak/hyperparameter 使用
`specialized-report`；decision-oracle/role-scope 使用 `specialized-run`。

中断或失败运行可通过 `execution-audit --input pipeline_output/<run>` 生成执行
完整性和失败事件的表格、Excel、HTML/Markdown，便于检查已完成阶段与失败原因。
其产物位于 `report_output`，与模型性能报告分开。

## 模型复用与推理

pipeline 自动导出 `model_config/<run>`，也可独立导出：

```bash
python export_model_config.py --pipeline-output pipeline_output/finalcase_v5_01
python pipeline.py infer \
  --model-config model_config/finalcase_v5_01 \
  --case-id tuned_all_roles__inception_small_no_gravity \
  --input-manifest path/to/participant.yaml
```

推理加载模型和对应预处理参数，不重新训练。输入 manifest 提供 participant 的
一条静态记录或多条静态、动态记录。可用推理能力记录在导出的
`export_manifest.json`；动态 R/S/W 当前使用同 participant 的静态 B recording
进行校准。无 B 静态记录的静默校准 ablation 为 V5 待实现功能。

## Dash 操作面板

```bash
conda activate ml
# 在 final_v0/final_pipeline_v5 目录执行：
python dashboard.py --host 127.0.0.1 --port 8050
```

打开 `http://127.0.0.1:8050`。页面沿 workflow 纵向排列：输入 → PPG 预处理 →
IMU 预处理 → motion detector → SQI → denoiser → 特征工程 → 表征 → 分类模型 →
聚合 → report。每个阶段左侧为算法选择、开关、滑块和精确数值输入，右侧紧邻
时域/频域曲线、窗口/特征/预测表格。算法来自同一生产模块，界面不另写一套数学实现。
PPG 预处理时域图将原始 RED/IR 与滤波后的 RED/IR 分为两张图，分别使用独立纵轴；
补缺后的 native 曲线保留在原始图图例中，点击可显示。不作额外归一化或纵向平移。
时域及频域图例统一放在绘图区上方，预留换行空间，避免与横坐标重叠。
IMU 原始与处理后曲线按同一物理单位配对展示。
频域图使用完整采样率记录，不随时域预览起点/时长改变；同一单位的面板中叠加
原始/处理后 PSD，默认收起，点击 `Frequency domain / PSD` 展开；横轴为 Hz
（包含 0 Hz），正值使用对数纵轴，零值不改写；全零面板使用线性纵轴。
另默认展开 `Frequency domain / FFT amplitude`，采用与 Notebook 相同的
`abs(rFFT)/N`：不平方、不转 dB、不额外加倍或去趋势，正弦幅度 A 对应正频率峰 A/2。
FFT 使用线性纵轴，初始查看 0.01–8 Hz；保留 DC 至奈奎斯特频率的全部频点，
可用图表 Autoscale 展开。含缺口时只计算最长连续有效片段，不拼接缺口；
实际样本区间及频率分辨率在 Stage details 的 `fft` 字段中说明。
特征阶段在同一 PPG 时域图叠加 direct/processed 的 RED/IR 曲线，保留各来源及
actual-used 峰点，不重复绘制 actual-used 的底层波形；未降噪时只显示 direct 两条线。
表格按特征种类和
维度排列，保留全部行，可分页、筛选和排序。文件级特征分七类；窗口级分别展示
实际 115 维工程量和 146 维矩阵，另列逐搏形态、双波长配对及有效性/来源。
特征表默认收起，按 `engineering features`、`file level features`、
`time series features`、`freq domain features`、`morphology features` 五组展开。
工程窗口/矩阵归工程组；PPI、时域及非线性量归时序组；频域和形态量各归对应组；
其余文件级汇总及双波长表归文件级组。同一张表不重复展示，折叠不改变表内数据。
矩阵模式不再额外计算文件级向量；未执行的产物不补造，raw 模式的特征探索不作为
raw 分类模型的输入。
峰检测预览中的 `Beatwise PPI / HR` 分别显示逐搏间期（秒）及 `60/PPI`（bpm），
横坐标均为相邻两峰的时间中点；圆点/叉号区分有效/无效间期，direct、processed
和特征实际采用的间期通过图例区分。无效点默认隐藏，点击对应图例显示，数据不删除。
连续有效点直线相连，无效间期、缺失或来源
切换处断线，不作插值补齐。
所有表征模式点击特征工程的 `Analyse` 都会生成 `Window PPI / HR`，无需切换到
`feature_matrix`。已有矩阵时直接读取其结果；其他模式直接调用同一个 `WindowPlan`
和矩阵逐窗 rate 提取函数，复用已检测的峰，不重算完整 146 维矩阵，也不改变模型输入。
分窗采用 `windows.engineering` 的窗口长度、步长及边界设置，这些控件在所有模式下
均可调整，不改变 raw 模型独立的 `windows.raw_dl`。质量路由存在时使用
其实际有效区段；路由关闭时按整条记录分窗，不额外制造路由边界或质量等级。
逐窗图始终展开并展示整条记录，不受时域预览起点/时长裁剪。
均值、中位数和总体标准差全部默认显示，以窗口中心为横坐标，连接相邻有效窗口，
无效窗口处断线。窗口 HR 是逐搏 `60/PPI` 的统计值，不是窗口平均 PPI 的倒数。
缺失/不合格窗口不补值；没有可用峰或记录不满足分窗条件时，保留图框，
在图下及 Stage details 明示原因，不放宽原算法的有效性条件。

IMU、SQI、denoiser、特征工程、表征和模型各只有一个默认收起的参数区，点击
带箭头的参数标题即可展开。主要算法选择和主开关留在外面；Analyse、Run、Stop、
已有产物选择及预览不随参数区折叠。展开后切换算法或加载 YAML 仍保持展开状态。
每个参数名旁直接显示含义、算法作用及调大/调小或切换选项的影响，不必悬停查看。
说明也区分仅训练生效的参数和当前执行路径不读取的配置；折叠只改变显示，不改
参数值、默认值、CLI/YAML 导出或计算过程。

1. **配置**：顶部 YAML 默认不选，此时使用现有函数/dataclass 的参数默认值；
   初始表征/模型为 raw/CompactCNN1D。选择 pipeline YAML 会填入控件；选择
   study YAML 后可选择其中的 case。清空 YAML 恢复函数默认值。
   YAML 只提供初值，之后每次 Analyse 均以当前控件值为准。
2. **输入**：从 manifest 下拉框选择一条或多条 recording，并选定绘图记录；
   更换 manifest 配置后，记录选项随之更新。
   `roles` 主控件只显示 B/R/S/W，下面的 Resolved roles 显示实际记录编号。
   新勾选类别展开全部编号；YAML 已指定的部分编号保持不变，取消后重选才展开。
   `training.classifier_role_families` 仍独立控制分类范围，B 可仅用于校准。
   也可展开 Custom CSV files，添加路径、file_id、具体 role（B、R1–R4、S1–S2、W1–W2）和可选标签。
   同 participant 的动态记录需要 B 校准记录；校准文件也可在 IMU 区指定。
   起点/时长只控制图的显示范围，滤波和状态估计仍处理完整 recording。
3. **逐段 Analyse**：可直接点击任一下游阶段。缺失或过期的上游会先计算，
   不变的结果在当前会话内复用；修改参数后相关下游需要重算。
   关闭的可选模块按原流程旁路，denoiser 是否实际被调用由原 SQI/motion 路由决定，
   其状态显示在阶段详情中。Analyse 不训练分类器、motion detector 或 SQI 校准器。
4. **已有拟合产物**：motion 和 SQI 各有独立文件选择框，可下拉选择或输入路径。
   新生成或复制产物后，点击顶部 Refresh 更新可选项。
   使用训练分位数 SQI 时需要包含 bounds 和 fitted_on_participant_ids 的已有校准
   JSON；motion 使用已有 evidence JSON 所关联的模型与阈值。缺少时显示具体缺项，
   不对待分析 participant 临时拟合。
5. **模型 Analyse**：选择 model_config 的 case 或直接选择 learned bundle。
   权重选择不会覆盖当前控件。要从该模型的原参数开始，在顶部选择导出的
   resolved_pipeline_config.yaml；之后仍可修改参数。架构、通道和特征维度必须
   与所选权重相容，训练期拟合变换必须随 bundle 提供；阶段详情记录当前配置。
6. **训练**：模型区切换到 Train 后才出现 Run；运行前选择 YAML。Current controls
   训练当前控件配置；Comparison queue 执行缓存的参数组合；Selected study YAML
   执行所选计划的完整 case 序列，而非覆盖计划中的所有 case。
   Stop 可终止后台任务及其子进程。Refit 默认关闭，cache 模式可独立选择。
7. **Comparison**：填单位名、Add，修改参数后再次 Add；可 Remove last/Clear。
   队列可下载为完整 CLI 和可执行 comparison YAML，再由模型区的同一个 Run 启动。
8. **Report**：选择已有 pipeline output、分析方式、图表模块、统计参数后 Analyse。
   报告仍由 analyse_report.py 写入 report_output，完成后自动选择对应输出，
   面板提供图、HTML 和数据表预览，也可手动切换已有报告。
   Advanced tools 保留索引、模型导出、Excel、执行审计和专项研究入口；其中需要
   训练的操作只通过模型区 Train → Advanced tool request → Run 执行。
   命令参数直接从原 CLI 生成控件；需要 plan 时选择已有 YAML，逐项修改展开的
   参数。执行和下载均读取当前控件，CLI 文本只读。执行时将当前 plan 快照保存到
   `pipeline_output/.dashboard_requests/plans/`，不覆盖原 YAML；下载的 CLI 包含
   同一快照，可独立重放。专项训练选择此 plan 后也满足训练前必须选择 YAML 的要求。

页面底部可查看并下载当前阶段等价 CLI 和 resolved YAML；两份文件放在 V5 目录后
可从 CLI 执行同一阶段。也可不使用 YAML，直接调用同一个无训练阶段服务：

```bash
python stage_analyse.py --stage ppg --record-id <record-id> \
  --set signal.ppg_filter.low_hz=0.2 \
  --set signal.ppg_filter.high_hz=8.0 \
  --set signal.ppg_filter.order=3
python stage_analyse.py --help
```

阶段预览只保留有容量限制的会话内存结果，不写入磁盘预处理 cache，不改已有 run
或模型权重。浏览器刷新/服务重启后可重新 Analyse。当前生产实现的 feature-matrix
提取依赖 routing timeline；质量、motion 和 denoiser 全部关闭的组合不产生该
timeline，会报告缺项。需要分析此表征时可选择现有 diagnostics_only 路线，
Dash 不会静默改变所选流程。

单 participant 输入可查看质量、概率与分类；ROC/AUC、cohort confusion matrix
和显著性检验需要有标注、满足相应类别和样本要求的多 participant 数据。

## 文档

- [CLI_REFERENCE.md](docs/CLI_REFERENCE.md)：完整命令、参数与 study/sweep 用法。
- [OUTPUT_CONTRACT.md](docs/OUTPUT_CONTRACT.md)：目录、数据格式、Excel 与模型权重。
- [ARCHITECTURE.md](docs/ARCHITECTURE.md)：配置、科学流程、执行、报告和 Dash 的关系。
- [THESIS_CODE_CONFLICTS.md](docs/THESIS_CODE_CONFLICTS.md)：论文描述与实现的差异及影响。
- [PLAN_COMPATIBILITY.md](docs/PLAN_COMPATIBILITY.md)：通用与专项研究计划入口。
