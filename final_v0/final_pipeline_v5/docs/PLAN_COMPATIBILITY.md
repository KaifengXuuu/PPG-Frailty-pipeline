# V2 study-plan compatibility

V2 原有 28 份 study YAML；V5 全部保留，并新增 1 份 `finalcase.yaml`，合计 29 份。V5 不会把不同 schema
强行解释成同一种 grid。每份计划都先由它自己的严格 loader 校验，再进入对应的
计算或分析入口。这样既保留 V2 workflow，也能保证训练产物和展示产物分离。

## Canonical plans：19 份

19 份 `ppg_frailty.study_plan.v2` 可直接通过以下任一入口运行：

```bash
python pipeline.py run-plan --plan configs/studies/finalcase.yaml
python sweep.py run --plan configs/studies/finalcase.yaml
```

这里以 `finalcase.yaml` 为可直接执行的例子；其余 canonical 文件可由同一路径替换
`--plan` 值。

它们覆盖 single、grid、catalog、representation/model、ensemble、SQI/motion、
sequential ablation、Legacy Bridge、finalcase、gravity 和 ShapeFormer 流程。V5
只把 plan 的 HTML/plot/report policy 强制设为关闭；scientific config、case
expansion、repeat/fold、seed、split 和训练算法不变。输出统一位于
`pipeline_output/<run>/<comparison>/repeat_<RR>/fold_<FF>/`。

其中两份 motion-finalist plan 引用同一个冻结 detector bundle。V5 不复制约 137.73 GiB
历史 artifact 树，只随计划提供运行所必需的 evidence JSON 和 all-29 motion model：

- `motion_internal_evidence.json`：SHA-256 `10f02a9d784e06471c7109ff8dc92d28f1a8d7753f8fdf179bebce5699fb446c`；
- `formal_motion_model.pt`：SHA-256 `62a09c53fecf90dfb9388900df19efccc62facf9f72b221b09c7d06c999c6eca`。

原 V2 absolute provenance 路径可用时仍需 hash 相同；在独立 V5 checkout 中，loader
使用计划旁已映射的相同字节副本。缺失或 hash 漂移会 fail closed。baseline 和
finalcase 不依赖该 bundle。

## Specialized plans：10 份

### 分析型：5 份

- 4 份 `ppg_frailty.stage0_decision_bias_oracle.v1`；
- 1 份 `ppg_frailty.role_scope_decomposition.v1`。

它们只消费既有 prediction/ranking artifact，必须从报告入口运行，只写
`report_output`：

```bash
python analyse_report.py specialized-validate \
  --plan configs/studies/static_line_b_staged_v2/stage0_decision_bias_oracle.yaml \
  --source-root ../final_pipeline_v2

python analyse_report.py specialized-run \
  --plan configs/studies/static_line_b_staged_v2/stage0_decision_bias_oracle.yaml \
  --source-root ../final_pipeline_v2
```

`--source-root ../final_pipeline_v2` 是只读历史 artifact 的显式来源；若计划引用的
输入已经在 V5，可改成相应 V5 内路径。adapter 不会拼接、猜测或近似替代缺失输入。

### 计算型：5 份

- `ppg_frailty.stage5_pre_motion_ptt.v1`；
- `ppg_frailty.stage_ablation_01_static_peaks.v3`；
- 3 份 `ppg_frailty.hyperparameter_study_plan.v1`。

它们使用保持 V2 数值语义的 V5 提取计算 runner，只写 `pipeline_output`。Stage5
示例为：

```bash
python specialized_pipeline.py validate \
  --plan configs/studies/static_line_b_staged_v2/stage5_pre.yaml \
  --source-root ../final_pipeline_v2

python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage5_pre.yaml \
  --source-root ../final_pipeline_v2 \
  --run-name stage5_pre_v5_01
```

static-peak 使用同一形式并把 plan 换成
`stage_ablation_01_static_peak_detectors.yaml`。三个 hyperparameter plan 不消费
`--source-root`；第一阶段可直接运行，后两个阶段必须显式连接前一阶段的
`selected_configuration.json`：

```bash
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage6_batch_LR_search.yaml \
  --run-name stage6_batch_lr_01

python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage6_regula_search.yaml \
  --upstream-study pipeline_output/stage6_batch_lr_01 \
  --run-name stage6_regularization_01
```

`stage_ablation_channels.yaml` 同样通过 `--upstream-study` 指向完成的 regularization
run。

这些计算入口本身只写结构化数据，不再先生成展示文件、再靠递归门禁删除或拒绝它们。
结束后 adapter 索引产物并写数据便利视图 `tables/pipeline_data.xlsx`。Stage5 导出 frailty29/PTT22 合计
10 个 outer-fold 与 2 个 final motion weights 到 `model_config/<run>`；hyperparameter 的每个
V5 phase 导出所有 frailty fold weights 和相应 case 配置；static-peak 不训练模型，
因此明确记录 `model_trained=false`、`model_kind=not_applicable`。
Stage5/static-peak/hyperparameter 报告由 `reporting.specialized` 直接读取持久化数据；
hyperparameter 的各 phase 复用普通 classification renderer。公开 CLI 不会写回
pipeline source：

```bash
python analyse_report.py specialized-report \
  --input pipeline_output/SPECIALIZED_RUN
```

V2 successive-halving 唯一已有的 completion 流程也保留：

```bash
python specialized_pipeline.py complete \
  --study-dir pipeline_output/SPECIALIZED_RUN
```

`specialized_pipeline.py run --resume pipeline_output/<run>` 可恢复同一个 V5
Stage5/static-peak run；hyperparameter orchestration 也会恢复其已有的持久化 V5
phase。V5 不发明新的候选晋级或 promotion 数学；`complete` 仍只执行 V2 原来明确定义
的未晋级候选 full-CV completion。

## 数值边界

展示调用已从计算 runner 中抽离为 `reporting.specialized` 的兼容入口；计算入口不含
PNG/HTML writer，也不需要训练后的展示清理步骤。数学计算、schema、排序、threshold、
split/seed 和产物字段由定向 golden/数值测试审计；最终 scientific equivalence 仍以
冻结环境中的完整输出比较为准，不能由静态源码 hash 代替。
