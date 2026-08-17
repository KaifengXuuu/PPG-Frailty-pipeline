# M2 数据 Manifest、双 Fold 注册表与评估协议图

> 当前权威 M2 数据/分折图；实线为数据或物化 membership，虚线为审计、限制或结果引用。

```mermaid
flowchart LR
    A["Frailty3 raw read-only<br/>StudyData + TestDataYoungers"] --> B["Full-byte and numeric scan<br/>261 CSV / 29 subjects / 8 channels"]
    L["Standard label CSV<br/>STE072 is active join key"] --> B
    B --> FM["File manifest<br/>hash / role / fs / unit / duration / reference"]
    B --> SM["Subject manifest<br/>class / cohort / all file IDs"]
    U["User-confirmed roles<br/>B baseline · R relax/recovery<br/>S stand-and-sit · W walk"] --> RM["Stage registry<br/>partial order S/W before R"]
    RM --> FM
    SM --> H["Historical SGKF 1.4.2 reproduction<br/>shuffle mapping defect retained"]
    SM --> C["Corrected SGKF semantics<br/>permute groups + count rows together"]
    H --> HR["Historical registry<br/>reproduction only · 6 class-missing folds"]
    C --> FR["Future registry<br/>5 repeats × 5 folds<br/>all folds contain 3 classes"]
    FR --> P["Fixed-epoch OOF protocol<br/>no early stopping · OOF unseen during training"]
    P --> O["oof_validation_* results<br/>manifest + registry + protocol + preprocessing provenance"]
    HR -. "old result attribution only" .-> O
    FM -. "dataset version gate" .-> O
```

## 算法不变量

- Group 的最小单位始终是 subject；文件和窗口不得独立重新分折。
- 未来注册表逐折三类齐全且每类 fold count 差不超过 1。
- 历史/未来 registry ID 永不互换；全部候选必须在未来 registry 统一重跑。
- OOF 只在 fixed epochs 全部训练完成后进入一次评估。
