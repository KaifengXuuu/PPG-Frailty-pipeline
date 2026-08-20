# Final Pipeline V2 status

## 当前可用

| Area | Status |
|---|---|
| Single config | `frailty_3class_pipeline_v2.py`，支持完整或指定 repeat/fold、resume、自动报告 |
| Grid / ablation | `frailty_3class_sweep_v2.py`，config-driven、case-level parallel、dry-run、resume |
| Progress | terminal 单行刷新；完整事件写 JSONL；Dash 只监控对应 job |
| Reports | CSV/JSON/Markdown/HTML、learning curves、confusion、排名、CI、route/role、论文图与完整索引 |
| Dash | canonical data/config 选择、阶段预览、下载、独立 job start/stop、completed-study 浏览 |
| Reference configs | raw、feature vector、feature matrix、fusion 四个 Line-B-default 入口；默认值不限制普通 V2 组合 |
| Ordinary dependencies | 按所选 config 检查 import availability；不使用 source/lock/attestation gate |

普通 outer/full 路径的软件接线已完成：participant-local role-B EKF calibration、A1/A2
route artifacts、role OOF、8ch frailty raw/fusion/ShapeFormer、motion 8ch reference 与
具名 11ch augmentation ablation、predictor/metadata
隔离、training history 与原子 non-overwrite 输出均已覆盖 focused tests。

普通 V2 已把 optimizer、sampler、loss、class weighting/count basis、training
balance、Line A/Line B reporting aggregation、窗口计划、DL 重采样、PPG/IMU
normalization、quality route/window selection/quality weighting、artifact reducer、feature
groups、model architecture 与 ensemble roster 解析成独立的有效配置。省略字段会物化
reference default；合法非默认值进入真实 runtime 与 provenance/hash。仅改变无消费者的
inactive 参数会在配置阶段报错，不会制造 hash-only 假开关。

## 科学边界

- 没有在本实现过程中运行正式训练、完整 5×5、ablation、A3、PTT benchmark 或
  final refit。
- Outer CV single model 使用每个 repeat 自己的 seed：
  `[42,10042,20042,30042,40042]`；同 repeat 的五个 folds 共用该 seed。与 ensemble
  匹配的 Inception single comparator 是例外，固定使用 member 0 seed `50042`。
- `seed=42` 只属于人工选择后的 single-model all-29 final refit。
- 具名 five-member comparison preset 在每个 repeat/fold 复用
  `[50042,60042,70042,80042,90042]`；普通 ensemble 接受任意非空、唯一 uint32
  roster，final refit 继承被选配置的同一 roster。
- Reference aggregation 是 role-aware Line B；equal-files Line A 与 Line B 都是
  普通可选 reporting modules，且与 training balance 独立。同一 held-out OOF 同时生成
  window-balanced、Line A 与 Line B 视图。
- Internal accelerometer source 保持 `g→m/s²`；V2-036 仅对 PTT external sit data
  采用源 `m/s²` identity conversion。
- Aura comparison 固定 `hrv-analysis==1.0.2`，使用独立 requirements/environment。

## 已知 pending

1. SQI route/weights 与 registered reducers 已可执行；其优劣、最终 reducer、device rail
   QC 与 deployment hardware targets 尚无正式科学/部署证据。reference 仍保持 SQI off、
   identity reducer 与描述性成本，但这些默认值不授权或阻止其他配置。
2. 本轮没有执行正式 ensemble CV 或 final refit；其软件路径继续保留
   selection/OOF/config/dataset/model/file hashes、all-29 roster、nonoverwrite atomic
   write 与 golden parity。

旧 tracked-clean/source、exact-lock、attestation/prepublish、ONNX winner 和 acceptance
门禁已从 V2 删除；原版本保存在 V3 历史快照。

## 最近非科学验证

- 8ch frailty/motion focused：61/61。
- Ensemble seed/probability/OOF/final-refit focused：87/87。
- Conda-ml safe suite：258/258。
- V2 validator：7/7，220 个 Python 文件 syntax passed。
- model registry：6/6。
- reporting + dashboard fake/service tests：19/19。
- canonical calibrated real-input smoke：1/1（只构建/预处理，不训练）。
- V3 未修改。

完整人工命令见 [RUNBOOK.md](RUNBOOK.md)。
