# ADR-011: Representation modes and feature matrix / 表征模式与特征矩阵

- 状态 / Status: accepted_for_v2
- 依据 / Source: contract §§5.7, 5.8, 5.9, 5.11

## Decision / 决策

唯一配置字段：

```text
representation_mode = raw | feature_vector | feature_matrix | fusion
```

- `raw`: 8-channel windows，window→file→participant；
- `feature_vector`: 每 recording 一个 `FeatureVectorV1`，默认全量为 282 个 allowlist
  predictor，也可由 `features.enabled_groups` 选择非空组合；
- `feature_matrix`: 每 recording 一个 `OrderedFeatureMatrixV1[115,150]` 与 row mask；
  115 个 predictors 来自 10 s window、2 s hop 的工程特征；
- `fusion`: raw windows 先 mask-aware pooling 成 file embedding，再与一次编码的
  `FeatureVectorV1` 拼接。

Feature matrix 的 115 个 time-varying engineering channels 按时间排列。逐特征 validity
只写入 provenance，不再扩大 predictor tensor；无效项在 outer-train standardized 空间
以中性 0 表示。长文件均匀取 150 rows，短文件 transform 后右填零、mask=false。
matrix rows 绝不是独立样本。ROCKET/Ridge 已退出可执行 pipeline；后续 matrix 模型待定。

四种表征共享 manifest、folds、labels、roles、aggregation、metrics 和 provenance。
