# Representations and parallel model families / 表征与并行模型族

```mermaid
flowchart LR
    A["One manifest + one fold registry"] --> B{"representation_mode"}
    B --> C["raw<br/>8×T windows + mask"]
    B --> D["feature_vector<br/>one ordered vector/file"]
    B --> E["feature_matrix<br/>D×K, K=32 + row mask"]
    B --> F["fusion<br/>raw file bag + vector"]
    C --> C1["CompactCNN1D"]
    C --> C2["Inception Full/Small single"]
    C --> C3["ShapeFormer effect-size experimental"]
    D --> D1["L2 Logistic"]
    D --> D2["RBF SVM"]
    D --> D3["ExtraTrees / optional MLP"]
    E --> E1["Mask-aware Inception single"]
    E --> E2["Five-member probability ensemble"]
    E --> E3["ROCKET-10000 + ridge"]
    E --> E4["MiniROCKET named ablation"]
    F --> F1["raw window encoder"]
    F1 --> F2["mask-aware mean file embedding"]
    F --> F3["file vector encoder once"]
    F2 --> F4["concat once → file probability"]
    F3 --> F4
```

All branches share labels, memberships, role definitions, aggregation, OOF writer, and
participant metrics. / 所有分支共享标签、折、role、聚合、OOF 和 participant 指标。
