# PPG Frailty 3-Class Pipeline V2

V2 是当前可读、可配置、可由人工直接运行的 thesis pipeline。旧的复杂门禁版本已由
`final_pipeline_v3` 保留为历史快照；V2 的普通实验入口不要求 agent 监督，也不依赖
source/lock/attestation gate。

## Pipeline 边界

- 任务：29 名 participant 的三分类，machine labels `0/1/2`，报告显示
  `Pre-Frail`、`Robust/Non-Frail`、`Young`。
- 输入：B 与 R 文件；R1–R4 是同一 relax role 的独立文件，文件间没有连续时间轴。
- 验证：participant-grouped 5-fold × 5-repeat。每个 repeat 的五个 folds 共用该
  repeat seed：`[42, 10042, 20042, 30042, 40042]`。固定 `seed=42` 只属于人工
  选出 final use case 后的 all-29 refit。
- 普通单模型 outer CV 使用各 repeat seed；matched Inception single comparator 固定
  member 0 seed `50042`。five-member ensemble 在所有 repeats/folds 固定复用
  `[50042, 60042, 70042, 80042, 90042]`，每 fold 先平均五成员概率，再拼接每个
  repeat 的完整 participant OOF。final ensemble refit 也使用同一五成员 roster。
- canonical aggregation 为
  `window → file → role → participant`（role-aware Line B）；equal-files Line A
  仅是具名 ablation。
- reference quality mode 为 `off`。`diagnostics_only` 只记录原始诊断；A1/A2
  七状态 route/recovery 是纯状态机，只有在监督阈值配置完整时才影响数据路线。
- direct PPG view 使用 0.2–8 Hz；0.5–5 Hz 是 ablation。IMU reference 为同一
  participant role-B 校准的 roll–pitch EKF，无静默 fallback；LPF 0.3 Hz 是 ablation。
- frailty raw/fusion/ShapeFormer 输入固定为 8 通道：
  `RED, IR, A_dyn_X, A_dyn_Y, A_dyn_Z, GX, GY, GZ`。6 个 IMU 轴只用
  outer-train participants 拟合 scaler，不做逐窗 IMU 幅值归一。
- motion-model reference 同样使用上述 8 通道；加入
  `A_mag, Omega_mag, J_mag` 的 11 通道输入只属于具名 augmentation ablation，
  不进入 frailty raw-signal branch。
- 主结果是 OOF validation，不是 independent test；不会自动挑 winner。

四个 canonical 配置：

- `configs/reference_static_role_aware_v2.yaml` — raw / CompactCNN reference。
- `configs/reference_static_feature_vector_v2.yaml` — engineered feature vector。
- `configs/reference_static_feature_matrix_v2.yaml` — ordered feature matrix。
- `configs/reference_static_fusion_v2.yaml` — raw + feature fusion。

## 安装

在 `final_pipeline_v2` 目录中：

```bash
python -m pip install -e '.[reporting,dashboard,formal-benchmark]'
# 运行深度模型时再安装：
python -m pip install -e '.[deep,reporting,dashboard,formal-benchmark]'
```

Aura PRV 对照固定使用 `hrv-analysis==1.0.2`，由于它需要旧版 NumPy/Astropy，必须
在独立环境中安装 `requirements/requirements-prv-aura-compare.txt`，不要覆盖主
pipeline 环境。

## 一条命令运行

完整单配置（默认全部 25 cells，并自动生成报告）：

```bash
python frailty_3class_pipeline_v2.py \
  --config configs/reference_static_role_aware_v2.yaml \
  --study-id compactcnn_role_aware_reference
```

单因素 ablation：

```bash
python frailty_3class_sweep_v2.py ablation \
  --base-config configs/reference_static_role_aware_v2.yaml \
  --factor training.fixed_epochs --values 7 10 15 --reference-value 10 \
  --study-id compactcnn_fixed_epochs \
  --purpose 'Compare only the fixed-epoch factor.' \
  --flow-position 'Training-capacity ablation before manual candidate review.'
```

Cartesian grid（CPU/classical case 可用 `--jobs 4` 并行；deep 默认保持 1）：

```bash
python frailty_3class_sweep_v2.py grid \
  --base-config configs/reference_static_feature_vector_v2.yaml \
  --vary 'model.logistic_max_iter=[1000,2000]' \
  --vary 'training.class_weighting=[outer_train_inverse_frequency,none]' \
  --reference model.logistic_max_iter=2000 \
  --reference training.class_weighting=outer_train_inverse_frequency \
  --study-id logistic_screen --purpose 'Descriptive screening.' \
  --flow-position 'Candidate screening before confirmation.' --jobs 4
```

先检查所有 resolved cases 而不执行：在上述命令末尾加 `--dry-run`。恢复失败或中断
的 study 使用 `--resume <existing-study-directory>`；已通过 case 跳过，重试会写入新
的 `attempt_NNN`，不会覆盖旧结果。terminal 只显示一条刷新式进度条，完整事件写入
JSONL。

## 输出与报告

每次运行创建独立目录，目录名含日期、study kind 与 ablation/grid 对象。目录包含：

- study plan、resolved configs、每个 case/attempt、cell summaries 与 OOF Parquet；
- BA、macro-F1、per-class、worst-class、coverage、calibration、参数量与可用的运行成本；
- repeat/fold stability、双侧 95% CI、learning curves、counts/normalized confusion
  matrices、parameter effect/interaction、route/role quality summaries；
- CSV/JSON/Markdown/HTML、论文用 PNG、缺失视图的明确 N/A 文件；
- `outputs_index.json`，列出 study 目录内全部普通文件、bytes 与 SHA-256。

重新生成报告不会训练：

```bash
python frailty_3class_sweep_v2.py report --study-dir <study-directory>
```

## 计划与差距文档

- [V2 搁置与未完成事项](docs/V2_DEFERRED_POINTS.md)
- [V2 算法结构差距分析](docs/V2_ALGORITHM_GAP_ANALYSIS.md)
- [V2 消融与测试详细计划](docs/V2_ABLATION_TEST_PLAN.md)

三份文档区分软件实现、科学运行、人工决定与明确搁置；计划条目不代表已经执行。

## Dash 人工检查

```bash
python frailty_pipeline_dashboard_v2.py --host 127.0.0.1 --port 8050
```

Dash 可选择 participant/record/config/segment，预览原始与处理后 signal、QC/route、
pulse/PPI、morphology、engineering 等阶段表与图；可开关并行预览线、下载当前 CSV/
metadata、启动或停止独立 study subprocess，并浏览/下载已完成 study 的表格、图和 ZIP。
Dash 只调用 canonical modules，不复制算法。

更完整的命令说明见 [RUNBOOK.md](RUNBOOK.md)。
