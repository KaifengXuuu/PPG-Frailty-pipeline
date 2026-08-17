# 非 M0 根脚本与 Notebook 图册 / Non-M0 Root Script and Notebook Atlas

本图册覆盖根目录中不属于 M0 的8个Python入口和5个Notebook；每节描述当前保存代码的直接职责、输入和输出，不代表未来TODO已完成。

## 1. `frailty_3class_classifier.py`

```mermaid
flowchart TD
    CSV["StudyData / Youngers signals + frailty labels"] --> PRE["RED/IR filtering + IMU gravity removal"]
    PRE --> FEAT["Peaks, PPI, HRV, morphology, spectrum, motion-normalized features"]
    PRE --> WIN["CNN windows and SQI metadata"]
    FEAT --> ML["LogReg / RBF-SVM / ExtraTrees"]
    WIN --> DL["CNN / InceptionTime / ShapeFormer"]
    ML --> GROUP["Subject-grouped CV and role/quality aggregation"]
    DL --> GROUP
    GROUP --> OUT["datasets cache + results reports + models"]
```

## 2. `frailty_3class_cnn_fusion.py`

```mermaid
flowchart LR
    NPZ["Signal windows NPZ"] --> ENC["CNN or Inception encoder"]
    CSV["Handcrafted feature CSV"] --> IMP["Fold-train imputer + scaler"]
    IMP --> MLP["Feature MLP"]
    ENC --> CAT["Concatenate embeddings"]
    MLP --> CAT
    CAT --> CLS["Frailty three-class head"]
    CLS --> OUT["Fusion report + PT + scaler"]
```

## 3. `frailty_3class_overfitting_sweep.py`

```mermaid
flowchart LR
    CACHE["Frailty caches + fixed reference configs"] --> S1["Stage 1 regularization / SQI / aggregation grid"]
    S1 --> S2["Stage 2 fixed-epoch finalists"]
    S2 --> GEN["Generalization grid: architecture / sampling / roles"]
    GEN --> CV["Subject-grouped repeated CV without early stopping"]
    CV --> OUT["Manifest + runs + reports + curves + summary"]
```

## 4. `analyze_sweep.py`

```mermaid
flowchart LR
    RUNS["Sweep manifests, runs, reports and curves"] --> CLEAN["Validate completeness and deduplicate"]
    CLEAN --> AGG["Config-level mean, SD, t-CI and class metrics"]
    AGG --> RANK["Ranking and incomplete-config registry"]
    RANK --> OUT["CSV tables + confusion plots + analysis_report.md"]
```

## 5. `frailty_3class_holdout_eval.py`

```mermaid
flowchart LR
    LEAD["Selected leaderboard ranks"] --> SPLIT["Subject train / inner-validation / independent test"]
    SPLIT --> TRAIN["Inner-validation early stopping"]
    TRAIN --> TEST["One final test evaluation"]
    TEST --> REP["Repeated seeds + CI + aggregate confusion"]
    REP --> OUT["Holdout manifest, runs, summary, reports and plots"]
```

## 6. `shapeformer_port.py`

```mermaid
flowchart TD
    X["Window tensor N×C×T + labels"] --> LOCAL["Local convolution + position + attention"]
    X --> DISC["Effect-size or external PISD shapelet discovery"]
    DISC --> DIST["Shapelet distance and position embedding"]
    DIST --> GLOBAL["Global attention branch"]
    LOCAL --> CAT["Concatenate local and global embeddings"]
    GLOBAL --> CAT
    CAT --> LOGIT["Frailty logits / reusable bundle"]
```

## 7. `asa_classifier.py`

```mermaid
flowchart TD
    VDB["VitalDB clinical + PLETH / ECG tracks"] --> CACHE["Case cache and signal quality checks"]
    CACHE --> BR1["PPG waveform branch"]
    CACHE --> BR2["Spectrum branch"]
    CACHE --> BR3["RR / HRV branch"]
    BR1 --> POOL["Case mean / std / top-k pooling"]
    BR2 --> POOL
    BR3 --> POOL
    POOL --> CV["Subject CV + OOF threshold optimization"]
    CV --> HOLD["Subject holdout ASA1/2/3 evaluation"]
    HOLD --> OUT["PT / JSON / CSV / plots / scorecard"]
```

## 8. `svm2_dataset_train.py`

```mermaid
flowchart LR
    SIG["Raw PPG / IMU files"] --> LABEL["Dash interval annotation"]
    LABEL --> RAW["Long-table labeled samples"]
    RAW --> WIN["PPG / IMU / orientation / Welch / entropy window features"]
    WIN --> SCALE["Scaler + optional PCA"]
    SCALE --> SVC["Linear or RBF SVC with Group CV / holdout"]
    SVC --> OUT["train_* CSV + SVM pickle + preview"]
    SYNTAX["Current .py future-import placement error"] -. "blocks script" .-> LABEL
```

## 9. `PPG_Analy_Visual_test.ipynb`

```mermaid
flowchart LR
    ENV[".env + PPG / IMU CSV"] --> DASH["Legacy Dash callbacks"]
    DASH --> PROC["Peaks, HR/PPI/HRV, SpO2, IMU, ANC / CEEMD"]
    PROC --> VIEW["Interactive plots and tables"]
    PROC --> CSV["42-row hrv vs hrvanalysis export"]
    ERR["Saved cubic-spline error on two RR intervals"] -.-> CSV
```

## 10. `ppg_analyse3.ipynb`

```mermaid
flowchart LR
    CSV["PPG / IMU CSV + environment paths"] --> V8["Legacy v8 NPZ detector runtime"]
    V8 --> HILITE["Window/sample motion highlighting"]
    HILITE --> HRV["PPG processing and HRV Dash"]
    PATH["Wrong default path / abnormal saved calibration"] -.-> V8
```

## 11. `ppg_analyse4_calib.ipynb`

```mermaid
flowchart LR
    REC["PPG / IMU records + detector bundle"] --> CAL["Anchor and detector calibration"]
    CAL --> NPZ["Calibration NPZ contract"]
    REC --> AB["Denoiser A/B via Dash utilities"]
    AB --> VIEW["Waveform / HRV visualization"]
    UNRUN["Saved notebook has no execution"] -.-> CAL
```

## 12. `svm2_dataset_train.ipynb`

```mermaid
flowchart LR
    SIG["Raw signal files"] --> UI["Interval-label Dash"]
    UI --> TABLE["Labeled raw and window CSV"]
    TABLE --> FEAT["88-feature saved workflow"]
    FEAT --> SVM["Group-CV SVM and model files"]
    SVM --> TEXT["Saved process text; no durable numeric scorecard"]
```

## 13. `template_test.ipynb`

```mermaid
flowchart LR
    API["VitalDB API"] --> CLIN["Clinical ASA distribution exploration"]
    API --> TRACK["PLETH + ECG availability and waveform exploration"]
    CLIN --> NB["Saved notebook tables / plots"]
    TRACK --> NB
    ERR["Two saved API-usage errors"] -.-> API
```

## 覆盖声明

13个图块与 `ROOT_FILE_IO_INVENTORY.md` 的8个非M0 Python脚本、5个Notebook一一对应。M0的16个根代码入口由 `m0/05_SCRIPT_ALGORITHM_ATLAS.md` 覆盖。

