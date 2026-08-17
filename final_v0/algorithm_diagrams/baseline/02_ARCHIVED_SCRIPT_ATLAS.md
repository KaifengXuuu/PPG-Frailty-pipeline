# 23个归档代码/Notebook结构图册 / Archived Script Atlas

## 1. `archiv/frailty_3class_classifier - Copy_8channels_08062026.py`
```mermaid
flowchart LR
    DATA["8-channel frailty signals + labels"] --> FEAT["PPI / HRV and raw windows"] --> MODELS["ML / CNN / Inception / ShapeFormer-PISD"] --> OUT["fs64 caches + early experiment reports"]
```

## 2. `Arc/ppg_analy2.ipynb`
```mermaid
flowchart LR
    ENV[".env + CSV + old NPZ"] --> DASH["Single-cell PPG / IMU / HRV Dash"] --> ERR["Saved key mismatch then UnboundLocalError"]
```

## 3. `Arc/ppg_with_detector_v8.py`
```mermaid
flowchart LR
    CSV["CSV + fixed old bundle"] --> DET["Legacy v8 detector"] --> UI["Dash waveforms / HRV export"]
```

## 4. `Arc/ppg_with_detector_v8_npz_select.py`
```mermaid
flowchart LR
    PICK["Directory and NPZ selectors"] --> LOAD["CSV + chosen old bundle"] --> UI["Dash and HRV CSV"]
```

## 5. `Arc/ppg_with_detector_v8_npz_select_viz.py`
```mermaid
flowchart LR
    LOAD["Selected CSV / NPZ"] --> DET["Window motion prediction"] --> COLOR["Green static / red motion overlays"] --> UI["Dash"]
```

## 6. `Arc/pttppg_dash.ipynb`
```mermaid
flowchart LR
    V7["v7 setup1 / setup2 JSON"] --> BROWSE["Dash result browser"] --> FAIL["Saved duplicate-groups syntax failure"]
```

## 7. `Arc/pttppg_detector_v8_scores.py`
```mermaid
flowchart LR
    PTT["PTT windows"] --> F37["10 PPG + 27 IMU features"] --> MAH["Mahalanobis + lag + logistic"] --> OUT["results_detector_v8 old schema"]
```

## 8. `Arc/pttppg_detector_v8_scores_audit.py`
```mermaid
flowchart LR
    SCORE["Base detector scores"] --> AUDIT["PPG-only / IMU-only / fusion / walk-run audits"] --> OUT["Bundle + summary + figures"]
```

## 9. `Arc/pttppg_detector_v8_scores_audit_fix2.py`
```mermaid
flowchart LR
    AUDIT["Audit detector"] --> PLOT["ROC / PR + pooled confusion matrices"] --> OUT["Early results_v8_audit lineage"]
```

## 10. `Arc/pttppg_detector_v8_scores_audit_fix3.py`
```mermaid
flowchart LR
    SCORE["Detector scores"] --> THR["Precompute IMU threshold"] --> CM["Corrected confusion naming"] --> OUT["Audit artifacts"]
```

## 11. `Arc/pttppg_detector_v8_scores_audit_fix6.py`
```mermaid
flowchart LR
    RESULT["Detector result objects"] --> SAN["Attempt JSON sanitization"] --> NAMEERR["Undefined _json_sanitize blocks writes"]
```

## 12. `Arc/pttppg_detector_v8_scores_audit_fix8.py`
```mermaid
flowchart LR
    AUDIT["Headless audit + sanitizer"] --> WRITE["Summary / figures"] --> BUNDLE["Final bundle"] --> TYPEERR["dict(dataclass) TypeError"]
```

## 13. `Arc/pttppg_pipeline_v7_2_noleak_viz.py`
```mermaid
flowchart LR
    PTT["pleth4/5/6 + IMU + ECG"] --> AE["CNN-BiLSTM detector AE"] --> U["1D U-Net proxy / ECG-HR setups"] --> VIZ["No-leak visualization branch"]
```

## 14. `Arc/pttppg_stage1_detector.py`
```mermaid
flowchart LR
    PTT["PPG / IMU activity windows"] --> LR["Scaled logistic models"] --> FUSE["Threshold + lag + OR / AND"] --> OUT["results_stage1"]
```

## 15. `Arc/pttppg_pipeline_v7_3_noleak_viz.py`
```mermaid
flowchart LR
    PTT["pleth1/2 + IMU + ECG"] --> RULE["OR / AND detector + lag"] --> MASK["STFT MaskNet"] --> OUT["results_v7_3"]
```

## 16. `Arc/svm_dataset_train.ipynb`
```mermaid
flowchart LR
    SEG["Labeled segments"] --> F45["45 window features"] --> SVM["Scaler / PCA / SVC"] --> OUT["Early motion CSV + pickle"]
```

## 17. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis.py`
```mermaid
flowchart LR
    CSV["One s2_sit pleth file"] --> MID["Middle segment + Chebyshev + Savitzky-Golay"] --> BIO["HR / SDNN / SpO2"] --> VIEW["Print + plot"]
```

## 18. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysisv2.py`
```mermaid
flowchart LR
    DIR["PPGdf CSV directory"] --> LOOP["Per-file prototype processing"] --> PNG["Archive PPGdf plots"]
```

## 19. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis_esther.py`
```mermaid
flowchart LR
    ABS["Esther absolute desktop path"] --> MID["Middle-third 5 s windows"] --> BIO["Filter + HR / SDNN / SpO2"] --> EXT["External plots"]
```

## 20. `PPG_Testing_05_01_2026/Archive/ptt_ppg_dataset_analysis_fingertiponly.py`
```mermaid
flowchart LR
    FINGER["RedFinger / IrFinger at 400 Hz"] --> WIN["Filter + valid windows"] --> BIO["HR / SDNN / SpO2"] --> DUP["Duplicate source path; no unique algorithm"]
```

## 21. `PPG_Testing_05_01_2026/Archive/7-8-2025/ptt_ppg_dataset_analysis_fingertiponly.py`
```mermaid
flowchart LR
    FINGER["7-8-2025 fingertip CSV"] --> WIN["400 Hz filtered windows"] --> BIO["HR / SDNN / SpO2"] --> OUT["9 PNG + 9-row summary"]
```

## 22. `PPG_Testing_05_01_2026/Archive/FilteredWalkTest/FilteredWalkTest.ipynb`
```mermaid
flowchart LR
    BASE["base.csv Timestamp / Ir2 / Red2"] --> CMP["Raw vs baseline subtraction vs Chebyshev vs Butterworth"] --> NB["Embedded exploratory plots"]
```

## 23. `PPG_Testing_05_01_2026/ptt_ppg_dataset_analysis_16July2025.py`
```mermaid
flowchart LR
    CSV["25July25 IR / RED CSV"] --> FILTER["400 Hz Butterworth 0.5–5 Hz"] --> BIO["IR peaks, RR rejection, HR / SDNN / SpO2"] --> OUT["PNG + 12-row current summary mixed with stale plots"]
```

## 覆盖声明

23个图块与 `CODE_FILES.jsonl` 的23个非根路径一一对应。它们只用于历史溯源和输出归因；当前算法候选仍以根代码为准。

