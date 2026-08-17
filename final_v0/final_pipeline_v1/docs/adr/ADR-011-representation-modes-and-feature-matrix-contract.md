# ADR-011: Representation modes and feature matrix / 表征模式与特征矩阵

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§5.7, 5.8, 5.9, 5.11

## Decision / 决策

唯一配置字段：

```text
representation_mode = raw | feature_vector | feature_matrix | fusion
```

- `raw`: 8-channel windows，window→file→participant；
- `feature_vector`: 每 recording 一个 `FeatureVectorV1`，只含 allowlist predictor；
- `feature_matrix`: 每 recording 一个 `OrderedFeatureMatrixV1[D,32]` 与 row mask；
- `fusion`: raw windows 先 mask-aware pooling 成 file embedding，再与一次编码的
  `FeatureVectorV1` 拼接。

Feature matrix 的 time-varying channels 按时间排列；完整 fold-standardized file
context 只在 valid positions 重复。长文件均匀选 32 位置，短文件 transform 后右填零，
mask=false。ROCKET 和 Inception 都必须显式消费 mask；matrix rows 绝不是独立样本。

四种表征共享 manifest、folds、labels、roles、aggregation、metrics 和 provenance。

