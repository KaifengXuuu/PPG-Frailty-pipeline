# 当前项目端到端算法总图 / Current End-to-End Project Pipeline

本图表示当前仓库中实际存在的研究路线与产物，而不是未来 M1–M10 的已实现状态。虚线表示监督、评价、分析或历史依赖；实线表示主要数据/产物流。

```mermaid
flowchart LR
    subgraph INPUTS["Local inputs / 本地输入"]
        STUDY["StudyData + Youngers + frailty labels"]
        PTT["PhysioNet PTT: PPG + IMU + ECG"]
        VITAL["VitalDB clinical + PLETH + ECG"]
        MOTION["SVM labeled / raw / window CSV"]
        ENV["Environment and runtime configuration"]
    end

    subgraph SIGNAL["Signal processing / 信号处理"]
        BASE["Filters, IMU gravity removal, Aboy++ / PPI / HRV"]
        MA["Historical motion artifact / denoising routes"]
        HB["Heartbeat / IBI / gate experiments"]
        SQI["SQI, coverage and role-level aggregation"]
    end

    subgraph MODELS["Prediction branches / 预测分支"]
        FRAIL["Frailty3: ML, CNN, InceptionTime, ShapeFormer, fusion"]
        ASA["ASA 1/2/3: PPG + spectrum + RR/HRV branches"]
        SVM["Five-class motion SVM prototype"]
    end

    subgraph EVAL["Evaluation and synthesis / 评价与汇总"]
        SWEEP["Grouped CV / sweeps / fixed-epoch studies"]
        HOLD["Independent subject holdout"]
        ANALYZE["Run completeness, CI, ranking and scorecards"]
        PAPER["Paper tables, claim boundaries and failure evidence"]
    end

    STUDY --> BASE
    PTT --> BASE
    VITAL --> BASE
    ENV --> BASE
    BASE --> MA
    PTT -. "ECG / activity reference" .-> MA
    BASE --> HB
    PTT -. "ECG reference" .-> HB
    BASE --> SQI

    STUDY --> FRAIL
    SQI --> FRAIL
    VITAL --> ASA
    BASE --> ASA
    MOTION --> SVM

    FRAIL --> SWEEP
    FRAIL --> HOLD
    ASA --> HOLD
    SVM --> HOLD
    SWEEP --> ANALYZE
    HOLD --> ANALYZE
    MA -. "negative / proxy evidence" .-> PAPER
    HB -. "external failure / candidate detector" .-> PAPER
    ANALYZE --> PAPER
```

## 当前证据边界

- Motion/full-waveform 分支已经完成 M0 审计，结论是不恢复完整clean reconstruction。
- Frailty3 存在旧sweep与独立holdout两种证据层；主口径待用户决策。
- ASA最终结果依赖OOF threshold，argmax存在类别坍缩。
- SVM `.py` 当前不可编译且没有持久化论文级指标，只能画为原型分支。

