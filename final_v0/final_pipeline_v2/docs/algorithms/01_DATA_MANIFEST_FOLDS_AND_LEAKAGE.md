# V2 data, folds and leakage boundary

```mermaid
flowchart LR
    A["Read source bytes once"] --> B["pre-load SHA/bytes"]
    B --> C["parse full 8-channel recording"]
    C --> D["post-load SHA/bytes equality"]
    D --> E["manifest-bound physical QC"]
    E --> F["optional reduced slice"]
    G["frozen participant CSV"] --> H["outer train / OOF roster"]
    F --> H
    H --> I["fit on outer train only"]
    I --> J["transform held-out without refit"]
```

Physical admission currently checks finite values, nonfinite gap zero,
minimum-duration requirements and exact constant-channel rejection. Device
rails, clipping/saturation and absolute limits are explicitly deferred and
represented as null, never placeholder numbers. CSV timestamps are not
required; the manifest sampling grid is used.

The single seed42 fold artifact is for the internal motion reference. Frailty
formal execution uses the distinct repeated 5×5 artifact. Membership is copied
from the frozen authority; runtime splitter calls are prohibited.

