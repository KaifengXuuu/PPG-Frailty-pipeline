# M0 逐脚本算法结构图册 / Per-script Algorithm Atlas

本图册覆盖 M0 范围内承担算法、训练、评价、导出或运行职责的每个脚本。每节只画该文件的直接职责；跨脚本关系见上级总图。

## 1. `funcs.py`

```mermaid
flowchart LR
    IN["Arrays + sampling rate + parameters"] --> FILTER["PPG / IMU filters"]
    IN --> PEAK["Aboy++ peak and artifact rejection"]
    IN --> EKF["IMU bias, EKF, gravity removal"]
    FILTER --> ANC["NLMS ANC / CEEMD-lite"]
    EKF --> ANC
    PEAK --> BIO["PPI, HR and HRV values"]
    ANC --> WAV["Artifact and residual arrays"]
```

## 2. `ppg.py`

```mermaid
flowchart LR
    ENV["Environment keys + Dash selections"] --> CSV["CSV loader and column mapping"]
    CSV --> CALLBACK["Interactive callback branches"]
    CALLBACK --> PROC["Filter / IMU / ANC / detector / peaks"]
    PROC --> GRAPH["Waveform and motion figures"]
    PROC --> HRV["PPI / HRV table and optional CSV"]
```

## 3. `pttppg_pipeline_v7.py`

```mermaid
flowchart LR
    CSV["PTT records"] --> WIN["6 s / 1 s windows"]
    WIN --> DWT["DWT features"]
    DWT --> AE["CNN + BiLSTM detector AE"]
    WIN --> U1["Setup1 U-Net"]
    WIN --> U2["Setup2 U-Net with ECG / peaks / p6"]
    AE --> JSON["Detector fold JSON / figures"]
    U1 --> RES["Denoiser fold metrics / models"]
    U2 --> RES
```

## 4. `cnnppg_v7.py`

```mermaid
flowchart LR
    REC["PTT records grouped by subject"] --> HOLD["Fix external subject holdout"]
    HOLD --> INNER["Inner train / validation"]
    INNER --> CV["Five-fold train-subject CV"]
    CV --> UNET["8-channel proxy U-Net"]
    UNET --> MET["SNR and ECG-HR proxy metrics"]
    MET --> HTEST["One-time holdout evaluation"]
```

## 5. `pttppg_pipeline_v7_4_noleak_viz_ae.py`

```mermaid
flowchart TD
    REC["pleth1/2 + IMU + ECG reference"] --> SPLIT["Subject train / holdout"]
    SPLIT --> FEAT["10 PPG + 27 IMU features"]
    FEAT --> DET["Rule, AE and fused detector"]
    FEAT --> BC["Broadcast features"]
    REC --> STFT["PPG STFT magnitudes"]
    STFT --> MASK["39-channel 2D MaskNet"]
    BC --> MASK
    DET --> DOUT["Detector JSON / NPZ / PT / figures"]
    MASK --> MOUT["Walk/run PT / ONNX / meta"]
```

## 6. `pttppg_denoiser_v8_masknet.py`

```mermaid
flowchart LR
    REC["PTT windows + peaks"] --> CH["8 time channels + 37 broadcasts"]
    CH --> NET["Time-mask network"]
    NET --> LOSS["Waveform / shape / smooth proxy losses"]
    LOSS --> EXPECT["Expected model and metrics"]
    BUG["F variable shadowing + variable peaks collate"] -. "blocks" .-> NET
    EXPECT --> EMPTY["Actual output: empty"]
```

## 7. `pttppg_stage2_denoiser.py`

```mermaid
flowchart LR
    REC["PTT records"] --> PSEUDO["ECG impulses + pseudo shape + phase"]
    REC --> C47["47-channel input"]
    PSEUDO --> NET["Stage-2 mask model"]
    C47 --> NET
    NET --> CV["Subject CV then final train"]
    CV --> EXPECT["Expected PT / ONNX / summary"]
    ISSUES["Collate, detached a, phase leakage, holdout epoch choice"] -. "invalidates" .-> CV
    EXPECT --> EMPTY["Actual output: empty"]
```

## 8. `pttppg_detector_v8_scores_audit_fix9.py`

```mermaid
flowchart TD
    REC["PTT sit / walk / run records"] --> WIN["1, 2 or 6 s windows"]
    WIN --> PPGF["10 PPG features"]
    WIN --> IMUF["27 IMU features"]
    PPGF --> AE["Sit-clean AE and Mahalanobis score"]
    IMUF --> SCORE["IMU score"]
    AE --> LAG["Global lag search"]
    SCORE --> LAG
    LAG --> LR["Logistic fusion"]
    LR --> OUT["NPZ bundle + summary + audit figures"]
```

## 9. `pttppg_denoiser_hybrid_core.py`

```mermaid
flowchart TD
    CSV["PTT CSV"] --> LOAD["Load, resample and standardize"]
    LOAD --> IMU["9-channel dynamic IMU"]
    LOAD --> PPG["2-channel filtered PPG"]
    IMU --> RIDGE["81-reference lagged ridge baseline"]
    PPG --> RIDGE
    RIDGE --> DATASET["11- or 15-channel window dataset"]
    DATASET --> UNET["Residual artifact U-Net"]
    UNET --> LOSS["Proxy loss components"]
    UNET --> OLA["Overlap-add inference"]
```

## 10. `pttppg_denoiser_hybrid_train.py`

```mermaid
flowchart LR
    CLI["CLI config and seed"] --> DISC["Discover PTT records"]
    DISC --> SPLIT["Subject train / validation / holdout"]
    SPLIT --> CORE["Build Hybrid datasets and model"]
    CORE --> TRAIN["Epoch loop + validation selection"]
    TRAIN --> OUT["PT, meta, history, split and delay JSON"]
```

## 11. `pttppg_denoiser_hybrid_preview.py`

```mermaid
flowchart LR
    BUNDLE["One hybrid bundle"] --> REC["Select CSV records and windows"]
    REC --> INFER["Core preprocessing + model inference"]
    INFER --> PLOT["Raw, baseline and hybrid traces"]
    PLOT --> PNG["Preview PNG files"]
```

## 12. `pttppg_denoiser_hybrid_ab_compare.py`

```mermaid
flowchart LR
    A["Raw+IMU bundle"] --> SAME["Same selected records / windows"]
    B["Raw+IMU+baseline bundle"] --> SAME
    SAME --> IA["Inference A"]
    SAME --> IB["Inference B"]
    IA --> FIG["Side-by-side qualitative figure"]
    IB --> FIG
```

## 13. `pttppg_denoiser_hybrid_export_onnx.py`

```mermaid
flowchart LR
    PT["Hybrid PT + meta"] --> LOAD["Rebuild architecture and load state"]
    LOAD --> EXPORT["Export opset 17; dynamic batch; fixed time"]
    EXPORT --> ONNX["ONNX + external data"]
    ONNX --> CHECK["Random-tensor numerical comparison"]
    CHECK --> META["Update ONNX contract JSON"]
```

## 14. `pttppg_denoiser_onnx_runtime.py`

```mermaid
flowchart LR
    CSV["CSV record"] --> PRE["Runtime-local preprocessing and ridge"]
    META["Model metadata"] --> PRE
    PRE --> WIN["Build fixed windows"]
    WIN --> ORT["ONNX Runtime artifact_hat"]
    ORT --> CLEAN["Subtract artifact and overlap-add"]
    CLEAN --> ARR["Waveforms + masks + diagnostics"]
```

## 15. `ppg_denoiser_dash_utils.py`

```mermaid
flowchart LR
    UI["Dash request and selected record"] --> CACHE["Bundle/runtime cache"]
    CACHE --> RUN["ONNX denoiser runtime"]
    RUN --> MASK["Motion mask truncate / pad"]
    MASK --> TRACE["Plotly traces and display labels"]
```

## 16. `ppg_peak_hr_gating_train.py`

```mermaid
flowchart TD
    SOURCES["Five internal sources + SIM external"] --> REG["Record / subject registry and preflight"]
    REG --> SPLIT["Train / internal holdout / extra holdout"]
    SPLIT --> WIN["PPG windows + ECG timing / RR / gate targets"]
    WIN --> PPGNET["PPG-only three-head U-Net"]
    PPGNET --> CV["Five-fold OOF thresholds"]
    CV --> FINAL["Final PT / ONNX / scorecards"]

    REG --> M10["10-channel PPG+IMU windows"]
    M10 --> MA["Model A encoder classifier"]
    M10 --> MB["Model B light CNN"]
    MA --> BENCH["PTT holdout + SIM external benchmark"]
    MB --> BENCH
```

## 图册覆盖声明 / Coverage statement

以上16个入口覆盖 M0 中实际承担算法或运行职责的脚本。`funcs.py` 与 `ppg.py` 的重复实现被分别绘制；同一大脚本中的 PPG-only heartbeat 与 PPG+IMU A/B 被画成两条分支，防止证据混用。

