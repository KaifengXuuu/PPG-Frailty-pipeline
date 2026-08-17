# End-to-end V2 workflow

```mermaid
flowchart TD
    A["V2 config + frozen source snapshot"] --> B["Hash-verified manifest + imported folds"]
    B --> C["Full-record physical QC, then optional reduced slice"]
    C --> D["Signal views and unit conversion"]
    D --> E{"SQI mode"}
    E -->|"off (default)"| F["B/R direct inputs unchanged"]
    E -->|"diagnostics_only"| G["Save raw diagnostics; same inputs/weights/predictions"]
    E -->|"route (disabled)"| H["Fail closed pending supervised design"]
    F --> I{"representation"}
    G --> I
    I --> J["raw"]
    I --> K["feature vector"]
    I --> L["feature matrix"]
    I --> M["fusion"]
    J --> N["fold-local fit"]
    K --> N
    L --> N
    M --> N
    N --> O["typed window/file/member/participant OOF"]
    O --> P["Line A or matched Line B aggregation"]
    P --> Q["25-cell trusted metrics + hashes"]
    Q --> R["explicit comparison archive; no auto-selection"]
```

All fitted preprocessing and models see outer-training participants only.
Source bytes are checked before and after numeric parsing. Outputs are staged
and published without overwrite. Reduced execution cannot be promoted to a
complete 5×5 result.

The safe gate exercises build-data identity and representation construction,
not training or scientific metrics. Formal ShapeFormer, motion, ablation,
comparison, final-refit and ONNX routes have explicit gates and are never
triggered by validation.

