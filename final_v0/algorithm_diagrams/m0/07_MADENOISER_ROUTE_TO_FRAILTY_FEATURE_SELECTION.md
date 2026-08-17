# MAdenoiser 已确认路线到 Frailty 特征选择算法图

状态：路线已确认；实现、训练与 benchmark 尚未开始。

## 1. 串行路线

```mermaid
flowchart TD
    A["协议冻结<br/>cohort / roles / seeds / folds / metrics"] --> B["SQI-v2<br/>形态 / 周期 / 熵 / IBI / RED-IR"]
    B --> C["Motion-29<br/>subject-grouped threshold + CV"]
    C --> D["OOF motion probability 融入 SQI"]
    D --> E1["R1 spectral track + SQI"]
    D --> E2["R2 dual-PPG BSS + SQI"]
    D --> E3["R3 nonstationary + SQI"]
    D --> E4["R4 adaptive + SQI + safety gate"]
    E1 --> F["PTT ECG-peaks HR/PPI benchmark"]
    E2 --> F
    E3 --> F
    E4 --> F
    F --> G["四套 route-specific Frailty feature matrix"]
    G --> H["相同 seeds + subject 5-fold nested selection"]
    H --> I["锁定最高稳定 BA 组合"]
    I --> J["未来 untouched/external validation"]
```

## 2. SQI 与 Motion detector 的正确接入点

```mermaid
flowchart LR
    R["原始 RED / IR / ACC / GYRO"] --> U["单位校准与统一窗口索引"]
    U --> M["Motion detector preprocessing<br/>gravity / jerk / physical magnitude"]
    U --> Q["SQI raw components<br/>shape / ACF / template / entropy / IBI / dual-PPG"]
    U --> X["classifier-only standardization"]
    M --> P["inner-CV calibrated motion_probability"]
    P --> S["SQI-v2 composite + reject reasons"]
    Q --> S
    S --> O["accepted / no_estimate / coverage"]
    X --> C["Frailty model input"]
    O --> C
```

## 3. 四路线公平对照

```mermaid
flowchart TB
    I["同一 SignalRecord 与 split manifest"] --> A["R1 谱域污染抑制与轨迹"]
    I --> B["R2 PCA / ICA / STFT-NMF"]
    I --> C["R3 DWT / WPT / EEMD / VMD / SSA"]
    I --> D["R4 Wiener / LMS / NLMS / RLS"]
    A --> K["公共 peak / event merge / HR / PPI 后端"]
    B --> K
    C --> K
    D --> G["真实 HR 与 pulse-energy 安全门"]
    G --> K
    K --> Q["同一 SQI-v2 / coverage / failure schema"]
    Q --> P["PTT subject-level HR/PPI paired metrics"]
```

## 4. Frailty 选择与防泄漏门

```mermaid
flowchart TD
    F["冻结 29 subjects / 145 static rows"] --> S["5 seeds × 5 outer subject folds"]
    S --> T["outer-train"]
    S --> X["封存 outer-test"]
    T --> N["inner group CV<br/>route / HR-PPI block / SQI / threshold / model"]
    N --> L["锁定本折配置并重训 outer-train"]
    L --> X
    X --> O["一次性 subject OOF predictions"]
    O --> R["BA / macro-F1 / worst recall / coverage / stability"]
    R --> W["稳定赢家与 final_locked_config"]
    W --> E["未来独立 cohort 性能"]
    R -. "非嵌套最高分只能称 development-selected CV BA" .-> Z["禁止冒充独立 test"]
```

## 5. 状态语义

- 实线表示用户确认的未来执行顺序，不表示已完成。
- Motion-29 在监督目标明确前保持 `blocked_on_target_semantics`。
- PTT `peaks` 是 ECG R-peak reference；绝对 PPG pulse timing 必须 delay-aware。
- 最终版本按 nested subject-level BA 选择；非嵌套 CV 最高分只能作开发选择证据。
