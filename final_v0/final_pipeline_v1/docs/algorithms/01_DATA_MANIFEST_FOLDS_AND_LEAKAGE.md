# Data, frozen folds, and fit boundary / 数据、冻结折与拟合边界

```mermaid
flowchart LR
    A["Read-only raw files<br/>只读原始数据"] --> B["Versioned ManifestRow<br/>source hash + QC reason"]
    C["M2 corrected registry<br/>5 repeats × 5 folds"] --> D["Materialized membership CSV<br/>禁止运行时重算"]
    B --> E["Join by participant_id"]
    D --> E
    E --> F{"One outer fold"}
    F --> G["outer-training participants"]
    F --> H["OOF held-out participants"]
    G --> I["fit scaler / imputer / SQI / shapelets<br/>ROCKET / selector / calibrator / epoch"]
    I --> J["Frozen fitted artifacts<br/>training IDs + hashes"]
    J --> K["transform held-out without refit"]
    H --> K
    K --> L["exactly one OOF prediction<br/>per participant/fold/seed/config"]
    H -. "labels inaccessible to trainer" .-> X["Evaluator only"]
```

- Membership is imported, not regenerated. / 折成员从权威注册表导入，不重新生成。
- Every fitted artifact records training participant IDs and rejects OOF IDs.
- Changing held-out values must not change any fitted training object.
