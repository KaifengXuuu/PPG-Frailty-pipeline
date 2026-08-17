# Activity/Motion 迁移重训、SQI 与恢复特征流程图

> 状态：监督语义已确认；实现、训练与 benchmark 尚未开始。

## 1. 历史证据与路线纠偏

```mermaid
flowchart LR
    U["用户回忆：三分类尝试后合并动态类"] --> A["全库代码、Git、Notebook、模型与结果审计"]
    A --> S["已核验：Rest / Sit&Stand / Walk 三分类 SVM 数据"]
    A --> N["未找到：三分类 CNN 源码、权重、3×3 CM"]
    S --> Q["Notebook：Rest 较好；Walk 与 Sit/Stand 混淆"]
    Q --> B["主任务收敛为 Static vs Motion"]
    N -. "不能把记忆写成已验证 CNN" .-> P["论文证据边界"]
    B --> P
```

## 2. PTT 结构迁移到 29 人 Activity detector

```mermaid
flowchart TB
    PTT["旧 PTT Light CNN<br/>sit=0, walk/run=1<br/>10 channels"] --> T["可迁移结构/兼容权重"]
    D1["StudyData 21人×9角色"] --> M["StageManifest"]
    D2["TestDataYoungers 8人×9角色"] --> M
    M --> L["主标签<br/>B/R=0；S/W=1"]
    M --> R["保留角色<br/>B,R1-R4,S1-S2,W1-W2"]
    T --> C["from-scratch vs fine-tune<br/>IR vs RED vs dual PPG"]
    L --> C
    R --> C
    C --> O["5 seeds × outer 5-fold subjects"]
    O --> I["inner train/validation<br/>early stop + calibration + threshold"]
    I --> X["封存 outer-test subject"]
    X --> Y["OOF p_active + frozen class"]
    Y --> E["BA/F1/AUC/AUPRC/ECE<br/>binary CM + B/R/S/W strata"]
```

## 3. Motion→SQI 与 S/W→Recovery→Frailty

```mermaid
flowchart TB
    RAW["raw RED / IR / ACC / GYRO"] --> MD["OOF Activity detector"]
    RAW --> SQ["SQI-v2 morphology / periodicity / entropy / IBI / RED-IR"]
    MD --> PA["calibrated p_active"]
    PA --> F["soft Motion-SQI fusion"]
    SQ --> F
    F --> ROUTES["四条共后端路线<br/>spectral / BSS / nonstationary / adaptive"]
    ROUTES --> TRACK["OOF HR / PPI + SQI + coverage + reason"]
    STAGE["B → active → R1 → active → R2 → active → R3 → active → R4"] --> PAIR["preceding_active_role ↔ Rk"]
    TRACK --> PAIR
    PAIR --> AF["Active：peak/P95/ΔHR/PPI/motion dose"]
    PAIR --> RF["Recovery：HRR30/60/120、slope、tau、AUC、time-to-baseline"]
    AF --> FV["route-specific frailty feature blocks"]
    RF --> FV
    FV --> NCV["nested subject CV<br/>same folds + same seeds"]
    NCV --> LOCK["稳定赢家与 final_locked_config"]
```

## 4. 关键解释

- `Rk` 与它前面的实际活动配对，不按 S/W 编号硬编码。
- `p_active` 是 activity probability，不是 optical-artifact probability。
- PTT 内部满分与 external SIM 明显下降共同说明必须在本地设备域重训并重新校准阈值。
- 四阶段辅助分类可以探索 B/R/S/W 的生理差异，但主 detector 仍以 Static/Motion 二分类为验收目标。
