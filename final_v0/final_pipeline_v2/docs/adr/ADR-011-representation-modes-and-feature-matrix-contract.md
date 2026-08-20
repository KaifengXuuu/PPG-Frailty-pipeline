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
- `feature_matrix`: 每 recording 一个 `OrderedFeatureMatrixV1[D,K]` 与 row mask；默认
  全量且 `K=32` 时 `D=2×(115+282)=794`；
- `fusion`: raw windows 先 mask-aware pooling 成 file embedding，再与一次编码的
  `FeatureVectorV1` 拼接。

Feature matrix 的 115 个 time-varying engineering channels 按时间排列，并各带 validity；
selected fold-standardized file context 也成对携带 value/validity，只在 valid positions
重复。长文件按配置的 K 均匀取样，短文件 transform 后右填零，
mask=false。ROCKET 和 Inception 都必须显式消费 mask；matrix rows 绝不是独立样本。

四种表征共享 manifest、folds、labels、roles、aggregation、metrics 和 provenance。
