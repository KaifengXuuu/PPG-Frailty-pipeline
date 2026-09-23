# 论文实验配置

本目录按论文 `Kaifeng_Masterarbeit_draft_v1_0.docx` 的比较单位组织运行计划。
一个 YAML 对应一个 test，文件内 `cases` 是该 test 的比较组，不是不同的 test。
配置优先依据 V2 成功运行保存的 `study_plan.yaml` / `resolved_plan.yaml` 与
各组 `resolved_config.yaml`，而不是后来可能修改过的同名模板。

以下命令均在 `final_pipeline_v5` 根目录执行。`validate` 只解析和展开配置，
不训练；它不等同于验证依赖产物齐备或已经完成数值复现。
普通分类实验使用 `sweep.py`；峰检测、motion、denoiser 与分阶段搜索使用
`specialized_pipeline.py`；事后预测分解使用 `analyse_report.py`。
数据写入 `pipeline_output/<run>`，图表单独写入 `report_output/<run>`。

## 实验索引

| 论文 test | YAML | 比较组与执行资源 | 入口 |
|---|---|---|---|
| 4.1.1 / 表 18–19：静态峰检测 | [stage_ablation_01_static_peak_detectors.yaml](../static_line_b_staged_v2/stage_ablation_01_static_peak_detectors.yaml) | MSPTDfast 与 Aboy project v2；22 名 PTT 参与者的 sit 记录，RED/IR 分开 | specialized |
| 4.1.2 / 表 21：motion 双向迁移 | [motion_detector.yaml](motion_detector.yaml) | Frailty29 → PTT22、PTT22 → Frailty29；motion 专用分组五折，不运行 denoiser | specialized |
| 4.1.3 / 表 22–23：denoiser | [denoiser.yaml](denoiser.yaml) | identity 加六种 reducer；PTT 静态/动态分别评分，不训练 motion | specialized |
| 4.2.1 / 表 24：表征比较 | [representation_screening.yaml](representation_screening.yaml) | raw/CNN、feature-vector/logistic、feature-matrix/Small Inception；3 × 5 repeats × 5 folds | sweep |
| 4.2.2 / 表 25：早期三模型 matched 比较 | 尚无数值等价的 V5 训练 YAML | 旧 CNN、InceptionTime、ShapeFormer；见下方历史边界 | 待决策 |
| 4.2.3 / 表 26–27：SQI/motion 路由 | [sqi_motion_routing.yaml](sqi_motion_routing.yaml) | 两开关四种组合，denoiser 均关闭；4 × 5 × 5 | sweep |
| 4.2.4 / 表 28–30：B0–B7 单因素比较 | [legacy_bridge_ablation.yaml](legacy_bridge_ablation.yaml) | CNN 与 InceptionTime，各 B0–B7；16 × 5 × 5 | sweep |
| 4.2.4 / 表 31、附录 D-A：batch/LR | [batch_learning_rate_search.yaml](batch_learning_rate_search.yaml) | 6 组；先筛选、晋级，再补齐未晋级组的完整 CV | specialized + complete |
| 4.2.4 / 表 32、附录 D-B：正则化 | [regularization_search.yaml](regularization_search.yaml) | 9 组；读取 batch/LR 选中值；9 × 5 × 5 | specialized |
| 4.2.4.5 / 表 33–34：最终五配置 | [final_five_configurations.yaml](final_five_configurations.yaml) | 论文 Rank 1–5 汇总重跑；5 × 5 × 5，不自动选择赢家 | sweep |

`final_five_configurations.yaml` 保留每组历史参数，全部组不共享相同角色范围、
epoch、batch size、类别计数基准或重力处理。Rank 4 是实际运行的
`s1_163_v2_port_all_roles_modules_off`，不是后来同名模板里的 tuned all-role 组。
用户选定的单配置入口仍是 [finalcase.yaml](../finalcase.yaml)，对应 Rank 2
`tuned_all_roles__inception_small_no_gravity`；本目录不改变这一选择。

## 普通分类比较

以三路表征比较为例；替换 `--plan` 和 `--run-name` 即可运行其余 sweep：

```bash
python sweep.py validate --plan configs/studies/thesis/representation_screening.yaml
python sweep.py run --plan configs/studies/thesis/representation_screening.yaml \
  --run-name thesis_representation
python analyse_report.py run --mode comparison \
  --input pipeline_output/thesis_representation --preset classification
```

各普通计划保留五个 repeat、每个 repeat 五个参与者隔离 fold。
SQI/motion 对比复用原实验的 all-29 frozen motion 权重；这是在同一内部队列拟合过的
辅助处理，不能表述为 motion 模型的外层独立 OOF。运行前需要该计划所引用的
`artifacts/studies/.../20260820_225546_.../motion_internal/motion_internal_evidence.json`
及其权重。它们是模型输入产物，不是依赖版本锁；不能用任意其他 motion 模型替代后
仍宣称复现原实验。

## 峰检测、motion 与 denoiser

```bash
python specialized_pipeline.py run \
  --plan configs/studies/static_line_b_staged_v2/stage_ablation_01_static_peak_detectors.yaml \
  --run-name thesis_static_peaks
python specialized_pipeline.py run --plan configs/studies/thesis/motion_detector.yaml \
  --source-root ../final_pipeline_v2 --run-name thesis_motion
python specialized_pipeline.py run --plan configs/studies/thesis/denoiser.yaml \
  --run-name thesis_denoiser
python analyse_report.py specialized-report --input pipeline_output/thesis_denoiser
```

运行前可将 `run` 换成 `validate`，去掉 `--run-name`。
motion 的 `--source-root` 用于只读访问 V2 历史 comparison 权重及 OOF 证据；
输出仍由 V5 写入 `pipeline_output`。denoiser 只需要本仓库的 PTT 数据。
各自 YAML 的 `execution` 选择计算阶段；不会为了 denoiser 重训无关 motion 模型。
PTT 数据需要本地存在，以上命令不自动下载。

## Batch/LR 与正则化搜索

```bash
python specialized_pipeline.py run \
  --plan configs/studies/thesis/batch_learning_rate_search.yaml --run-name thesis_batch_lr
python specialized_pipeline.py complete --study-dir pipeline_output/thesis_batch_lr
python specialized_pipeline.py run \
  --plan configs/studies/thesis/regularization_search.yaml \
  --upstream-study pipeline_output/thesis_batch_lr --run-name thesis_regularization
python analyse_report.py specialized-report --input pipeline_output/thesis_batch_lr
python analyse_report.py specialized-report --input pipeline_output/thesis_regularization
```

第一步对全部六组执行 5 epochs、每个 repeat 的 fold 0，并将前三组从头训练到
10 epochs、完整 5 × 5 CV。`complete` 再训练其余三组的完整 CV；不执行这一步，
就没有论文表 31 的六组完整资源结果。正则化读取上游选中配置的 batch/LR，
不是重新选择固定值。如果重跑导致上游选择改变，下游也会随之改变；这是原有
依赖式搜索流程，而不是冻结为另一条实验流程。

## 历史来源与复现边界

下列运行目录均位于 V2 的 `artifacts/studies/static_line_b_staged_v2/`。

| 配置/结果 | 历史运行目录 |
|---|---|
| 静态峰检测 | `20260823_104616_stage-ablation-01-static-peak-detectors-v3` |
| motion 双向迁移 | `20260820_225546_staged-static-05-pre-motion-ptt-v1` |
| denoiser | `20260820_182324_staged-static-05-pre-motion-ptt-v1` |
| 三路表征 | `20260823_075821_catalog_sweep_staged-static-01-representation-baselines-v2` 的前三组 |
| SQI/motion | `20260824_031024_catalog_sweep_staged-static-05-sqi-motion-compact-cnn-v6` 的前四组 |
| B0–B7 | `20260821_162454_catalog_sweep_staged-static-03-centered-star-v1` |
| Batch/LR | `20260822_190832_hyperparameter_staged-static-06-batch-lr-successive-halving-v1` 及各 phase |
| 正则化 | `20260823_031621_hyperparameter_staged-static-06-regularization-grid-v1` 及 full_cv phase |
| 最终 Rank 1/3 | `20260824_160517_catalog_sweep_final-case-all-roles-inception-architecture-comparison-v1` |
| 最终 Rank 2 | `20260824_175009_catalog_sweep_stage0-inception-small-no-gravity-supplement-v1` |
| 最终 Rank 4/5 | `20260824_111943_catalog_sweep_final-case-comparison-inception-full-v1` |

重要区别：

- 历史三路表征与两轮超参搜索使用 `aboy_project_v1`；当前公共 catalog 默认
  MSPTDfast。因此本目录显式保留旧 detector，并不修改全局默认。
- 两轮搜索的未启用 motion/feature-matrix 配置字段采用当前默认；这些字段不进入
  本次 raw 分类计算。阶段资源、实际训练、输入及峰检测设置保留历史值。
- 静态峰检测的两个 detector 与对齐参数未换算法；现有统计配置使用五个指标 ×
  两个通道的 Holm–Šidák family，历史早期 snapshot 仅声明 F1 family。
  报告的多重比较范围因此需要与所引用的论文表保持一致。
- 历史 denoiser 评分使用 Aboy v1；后来的 `stage5_pre.yaml` 改为 MSPTDfast。
  本目录的 denoiser 保留旧评分器，不把替换评分器后的结果称为原结果复现。
- 最终五组 YAML 是对来源组的汇总重跑，不是回放旧的七组 merged reporting archive。
  它不将不同配置组之间的全部比较称为单因素 ablation。

论文早期三模型 matched 比较来自旧 runner 的
`results_frailty3/_overfitting_sweep/20260527_1320_cnn_inceptionTime` 与
`20260528_1045_shapeformer_0extra`。V2 只对它们进行了历史结果再分析，见
`results_frailty3/_v2_reanalysis/20260824_historical_search_reports_v2/01_early_three_model_matched/report_manifest.json`。
其外层评估标签参与 early stopping，不能用当前固定 epoch 的 V5 study 冒充等价重跑。
保留历史分析还是明确恢复旧训练协议，需要单独决定；当前不提供误标为可复现的 YAML。
通用历史汇总也不能视为该 matched 报告的数值等价替代。

附录 D-C 的图像来源可以进一步定位：窗口长度（5/10/15 秒）与重叠率（30/50%）
两图与 `results_frailty3/_sweep_analyse/20260608_0659_combined_cnn_inceptiontime_shapeformer/figures/`
中的对应 PNG 字节一致；dropout、weight decay、label smoothing、窗口保留比例、
epoch 和正则化因子六图与 `20260616_1139_overfitting_inceptiontime/figures/` 及
`20260616_1143_overfitting_inceptiontime/figures/` 中对应 PNG 一致（同属 `_sweep_analyse/`）。
后两份分析的 `clean_runs.csv` 指向 `_overfitting_sweep/20260608_1206_overfitting_sweep_stage1_rank2`。
这些是旧 runner 的探索结果，不是上表中的 V2 batch/LR 或九组正则化实验。
其中固定 epoch 探索与早期三模型的 early-stopping 流程也不能混为一谈；其与
V5 的完整输入、训练和评估等价性尚未核对，不根据箱线图的单个横轴标签伪造新训练配置。
现有 [ablation_fixed_epochs_v2.yaml](../ablation_fixed_epochs_v2.yaml)、
[stage3_v3.yaml](../static_line_b_staged_v2/stage3_v3.yaml) 等开发计划可以运行，
但不能仅凭文件名声称复现这些旧实验。
