# Ablation and comparison execution / 消融与对照执行

```mermaid
flowchart TD
    A["Locked base config"] --> B["Declare one factor"]
    B --> C["Materialize paired variants"]
    C --> D["Same manifest / folds / seeds / budget"]
    D --> E["Run complete repeat×fold cells"]
    E --> F{"All cells complete?"}
    F -->|"no"| G["incomplete: do not rank"]
    F -->|"yes"| H["Join paired OOF by participant/fold/seed"]
    H --> I["BA, macro-F1, class metrics,<br/>coverage, calibration, runtime, memory"]
    I --> J["paired delta + CI + failure cases"]
    J --> K["machine report + human report"]
```

Registered factor families include preprocessing, EKF-vs-LPF, quality/drop policy,
artifact reducers, feature families, sampling rate/kernel duration, representation,
ROCKET/MiniROCKET, single/ensemble, pooling, and aggregation. A comparison command
changes only its declared factor; all other identity fields must hash equally.
