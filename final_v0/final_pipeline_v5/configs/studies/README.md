# V5 study YAML 入口

本目录保留 V2 的全部 reusable study YAML，并新增论文最终方案
`finalcase.yaml`。请从 `final_pipeline_v5/` 根目录运行，不要调用 V2 顶层脚本。

## Canonical study plan

`schema_version: ppg_frailty.study_plan.v2`：

```bash
python sweep.py validate --plan configs/studies/PLAN.yaml --environment-policy exact
python sweep.py run --plan configs/studies/PLAN.yaml --environment-policy exact
```

也可用 `python pipeline.py run-plan --plan ...`。两者调用同一 V5 data-only service；
原 plan 中的 plotting/report flags 会被关闭，训练参数、case expansion、split 和 seed
保持原定义。

论文 finalcase：

```bash
python sweep.py validate \
  --plan configs/studies/finalcase.yaml \
  --environment-policy exact

python sweep.py run \
  --plan configs/studies/finalcase.yaml \
  --run-name finalcase_v5_01 \
  --environment-policy exact
```

## Specialized study plan

非 canonical schema 不会被 generic grid 静默解释：

- decision-oracle、role-scope：`analyse_report.py specialized-*`；
- Stage5/static-peak/hyperparameter：`specialized_pipeline.py validate|run|complete`；
- 计算完成后的专项报告：`analyse_report.py specialized-report`。

这些历史计划通常绑定 V2 dated artifact。V5 不复制约 137.73 GiB 的 artifact 树；需要时用
`--source-root ../final_pipeline_v2` 只读解析原输入。两份 canonical motion-finalist
计划所需的同字节 evidence/model 已作为小型 hash-bound authority 随 V5 提供；其余
缺失输入或任何 hash 不符仍会 fail closed。专项计算同样只在 `pipeline_output` 写数据
和 pipeline Excel；训练模型的专项计划还自动导出 learned weights 与 `model_config`。

完整 schema/入口/限制见
[`docs/PLAN_COMPATIBILITY.md`](../../docs/PLAN_COMPATIBILITY.md)。
