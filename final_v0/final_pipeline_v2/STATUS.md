# Final Pipeline V2 status

## 当前可用

| Area | Status |
|---|---|
| Single config | `frailty_3class_pipeline_v2.py`，支持完整或指定 repeat/fold、resume、自动报告 |
| Grid / ablation | `frailty_3class_sweep_v2.py`，config-driven、case-level parallel、dry-run、resume |
| Progress | terminal 单行刷新；完整事件写 JSONL；Dash 只监控对应 job |
| Reports | CSV/JSON/Markdown/HTML、learning curves、confusion、排名、CI、route/role、论文图与完整索引 |
| Dash | canonical data/config 选择、阶段预览、下载、独立 job start/stop、completed-study 浏览 |
| Canonical configs | raw、feature vector、feature matrix、fusion 四个 role-aware Line B 入口 |
| Ordinary dependencies | 按所选 config 检查 import availability；不使用 source/lock/attestation gate |

普通 outer/full 路径的软件接线已完成：participant-local role-B EKF calibration、A1/A2
route artifacts、role OOF、8ch frailty raw/fusion/ShapeFormer、motion 8ch reference 与
具名 11ch augmentation ablation、predictor/metadata
隔离、training history 与原子 non-overwrite 输出均已覆盖 focused tests。

## 科学边界

- 没有在本实现过程中运行正式训练、完整 5×5、ablation、A3、PTT benchmark 或
  final refit。
- Outer CV single model 使用每个 repeat 自己的 seed：
  `[42,10042,20042,30042,40042]`；同 repeat 的五个 folds 共用该 seed。
- `seed=42` 只属于人工选择后的 single-model all-29 final refit。
- Ensemble outer CV 等待人工冻结 repeat × member seed matrix，当前明确拒绝运行。
- canonical aggregation 是 role-aware Line B；equal-files Line A 是具名 ablation。
- Internal accelerometer source 保持 `g→m/s²`；V2-036 仅对 PTT external sit data
  采用源 `m/s²` identity conversion。
- Aura comparison 固定 `hrv-analysis==1.0.2`，使用独立 requirements/environment。

## 已知 pending

1. Ensemble outer-CV 的 repeat × member seed matrix 尚待人工确认；当前不会运行。
2. SQI supervised thresholds/weights、最终 artifact reducer、device rail QC 与 deployment
   hardware targets 尚未人工冻结；reference 保持 SQI off、identity reducer、描述性成本。
3. 本轮没有执行正式 final refit；其软件路径继续保留 selection/OOF/config/dataset/model/
   file hashes、all-29 roster、nonoverwrite atomic write 与 golden parity。

旧 tracked-clean/source、exact-lock、attestation/prepublish、ONNX winner 和 acceptance
门禁已从 V2 删除；原版本保存在 V3 历史快照。

## 最近非科学验证

- 8ch frailty/motion focused：61/61。
- model registry：6/6。
- reporting + dashboard fake/service tests：19/19。
- canonical calibrated real-input smoke：1/1（只构建/预处理，不训练）。
- V3 未修改。

完整人工命令见 [RUNBOOK.md](RUNBOOK.md)。
