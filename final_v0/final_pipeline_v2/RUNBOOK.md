# V2 人工运行手册

## 1. 进入环境

```bash
cd final_v0/final_pipeline_v2
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1
```

普通运行只检查所选 config 所需的 Python module 是否可 import；不会检查 Git branch、
源码 clean 状态、环境 lock 或 attestation。运行前由操作者自行确认当前代码与配置版本。

## 2. 先 dry-run

```bash
python frailty_3class_pipeline_v2.py \
  --config configs/reference_static_role_aware_v2.yaml --dry-run

python frailty_3class_sweep_v2.py run \
  --plan configs/studies/single_config_v2.yaml --dry-run
```

dry-run 会展开所有 cases，并用 canonical config validator 检查每个 override；错误类型、
未知 dotted path、非法 epoch/model/representation 会在执行前退出。

## 3. 单配置与指定 cells

完整 5×5：

```bash
python frailty_3class_pipeline_v2.py \
  --config configs/reference_static_role_aware_v2.yaml \
  --study-id raw_reference_5x5
```

仅运行一个 repeat/fold 做人工 smoke（这是 diagnostic，不是完整科学结果）：

```bash
python frailty_3class_pipeline_v2.py \
  --config configs/reference_static_feature_vector_v2.yaml \
  --study-id fv_smoke_r0f0 --repeats 0 --folds 0
```

`--repeats`、`--folds` 接受 `all` 或逗号分隔的 `0..4`。每个 outer-CV single-model
cell 的训练 seed 等于 repeat seed；同 repeat 的五个 folds 相同。`seed=42` 不会被
全局套用到 25 cells。

## 4. Ablation

推荐单次只变一个 thesis factor：

```bash
python frailty_3class_sweep_v2.py ablation \
  --base-config configs/reference_static_role_aware_v2.yaml \
  --factor signal.ppg_filter.low_hz \
  --values 0.2 0.5 --reference-value 0.2 \
  --study-id ppg_low_cutoff \
  --purpose 'Compare the named direct-filter factor only.' \
  --flow-position 'Signal-view ablation before model screening.'
```

Reusable plans 位于 `configs/studies/`。计划文件必须列出 purpose、flow position、
reference value、varied axes、controlled config 与 output policy；报告会原样归档。

## 5. Grid 与并行

```bash
python frailty_3class_sweep_v2.py grid \
  --base-config configs/reference_static_feature_vector_v2.yaml \
  --vary 'model.logistic_max_iter=[1000,2000,4000]' \
  --vary 'training.class_weighting=[outer_train_inverse_frequency,none]' \
  --reference model.logistic_max_iter=2000 \
  --reference training.class_weighting=outer_train_inverse_frequency \
  --study-id logistic_grid --purpose 'Descriptive model grid.' \
  --flow-position 'Screening before a confirmatory run.' --jobs 4
```

并行单位是 case，不在同一个 case 内嵌套并行 folds/models。deep studies 默认降为
`effective_jobs=1`，避免 GPU/内存争用。每个 child 把细粒度进度写入自己的 JSONL；
parent terminal 用 `\r` 刷新同一行。

## 6. Resume 与报告

```bash
python frailty_3class_sweep_v2.py run \
  --plan configs/studies/grid_optimizer_v2.yaml \
  --resume artifacts/studies/<existing-study-directory>

python frailty_3class_sweep_v2.py report \
  --study-dir artifacts/studies/<existing-study-directory>
```

Resume 精确读取 case result 中的 `artifact_root`；不会用目录搜索猜结果。passed case
跳过，failed/incomplete case 写到下一个 `attempt_NNN`。如果 `continue_on_error=false`
提前结束，manifest/summary 会明确记录 planned/passed/failed/not-run case 与 cell 数量。

报告排名只使用有限且完整的 participant OOF 指标；incomplete configs 单独列出，不会
混入 predictive leaderboard。部署测量（parameter count、latency、bundle size）与
预测性能分表显示，缺失部署成本不会伪装成 0。

## 7. 输出目录

默认根目录为 `artifacts/studies/`。每个 study 是新的日期目录，例如：

```text
20260817_153012_ablation_training-fixed-epochs/
  study_plan.yaml
  study_manifest.json
  study_run_result.json
  progress_events.jsonl
  resolved_configs/
  cases/<case-id>/attempts/attempt_001/experiment/
  tables/
  figures/
  STUDY_SUMMARY.md
  STUDY_REPORT.html
  outputs_index.json
```

报告刷新时，同名 PNG 与 `.NA.txt` 互斥；旧图不会在输入变为 N/A 后残留。ZIP 下载和
`outputs_index.json` 都覆盖整个 study，而不只覆盖 report 子目录。

## 8. Dash

```bash
python frailty_pipeline_dashboard_v2.py --host 127.0.0.1 --port 8050
```

浏览器打开 `http://127.0.0.1:8050`。推荐流程：

1. 选 config、participant、role、record 和 segment；
2. 勾选 signal traces 与 module stages，点击 Preview；
3. 检查同 participant role-B calibration、signal/PSD、QC-route、pulse/PPI、
   morphology/engineering 表；
4. 下载当前 trace CSV 与含 participant/role/class/fs/segment/stage 的 metadata；
5. 在 Study Jobs 选择 plan 并 Start；Stop 只停止当前 job，没有 active job 时绝不会
   启动新任务；一个 job 运行时拒绝第二次 Start；
6. 在 Completed Studies 并排浏览 figures、表格并下载完整 ZIP。

Dash job 在独立 subprocess/process-group 中运行，页面 callback 不执行训练。每个
Dashboard job 使用独立 output root，进度只读取该 job 的目录。

## 9. 特殊边界

- Ensemble outer CV：每个 repeat/fold 固定复用 member seeds
  `[50042,60042,70042,80042,90042]`；matched single comparator 固定 seed `50042`。
  每个 fold 只对五成员 probabilities 取均值，不能选择最佳 member 或平均 member metrics。
- Frailty raw/fusion/ShapeFormer：固定 8 通道
  `RED,IR,A_dyn_X,A_dyn_Y,A_dyn_Z,GX,GY,GZ`。motion detector/denoiser 默认也是
  8ch axes reference；额外 magnitude/jerk 组成的 11ch 只在具名 augmentation
  ablation 中启用。
- Final use-case refit：只在人工选择完成后运行；single seed=42，final five-member
  roster 为 `[50042,60042,70042,80042,90042]`。
- PTT external：V2-036 将 sit acceleration 按源 `m/s²` 直接使用，不乘 9.80665；
  gyro 根据 header 从 `deg/s` 转 `rad/s`。这不改变 internal frailty source 的 `g` 单位。
- Aura PRV：固定 `hrv-analysis==1.0.2`，使用独立 requirements 文件；普通 pipeline
  不会自动安装或调用它。
- V3：只读历史快照；所有新实现、实验和输出都属于 V2。

## 10. 非科学回归

修改代码后可运行普通测试；这些命令不训练模型：

```bash
python -m unittest tests.contracts.test_v2_configuration
python -m unittest tests.study.test_study_product_v2
python -m unittest tests.dashboard.test_dashboard_services
python tools/run_test_suite.py --suite safe
```

测试通过只说明软件合同与 synthetic fixtures 通过，不代表任何科学候选表现。
