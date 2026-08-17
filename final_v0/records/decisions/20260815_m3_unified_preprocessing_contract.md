# M3-PREPROCESS-001 — 统一预处理、无预校准 EKF 主路线与公共生理后端

- 日期 / Date：2026-08-15
- 状态 / Status：user_confirmed_contract_frozen
- 来源 / Source：用户在本回话逐项确认 D1–D8，并补充 D5 以 EKF 为主、LPF 重力分量为对照、采用无预校准 EKF
- 影响 TODO / Affected milestones：M3 以及依赖同一 preprocessing/fold/profile 的 M4–M9

## 决定 / Decision

1. D1：未来活动路线统一为 corrected、versioned 公共实现；根目录旧脚本仅用于历史复现和差异审计。
2. D2：未来主采样网格固定 400 Hz；Frailty3 原生 400 Hz 不重采样，外部 256/500 Hz 必须用显式、登记的 polyphase 路线转换。历史 256/500 Hz 模型不得伪装数值兼容，必须重训。
3. D3：static PPG 为 0.2–8 Hz、motion/peak/denoiser PPG 为 0.4–8 Hz，均为三阶 Butterworth SOS；notch 默认关闭。offline 使用 zero-phase，mobile 使用 causal stateful，二者分别训练和评价。
4. D4：原始输入顺序固定 RED、IR、AX、AY、AZ、GX、GY、GZ；保留双波长 raw/DC/filtered-AC/amplitude/ratio proxy。PPG model view 用窗内 median/IQR；IMU scaler 仅可在 M2 training fold 拟合。
5. D5：六轴无预校准 quaternion error-state EKF 是重力估计主路线；0.3 Hz causal LPF 是独立对照。二者共享单位、传感器低通、时间轴、mask 与输出 schema，不得静默互相 fallback。
6. D6：长度不超过 0.25 s 的内部 gap 可插值但保留 repair mask；边界 gap、超过 0.25 s、非有限比例超过 1%、全非有限、至少 1 s flatline、必需通道缺失均 fail-closed。clipping 与异常振幅只作为有证据级别的 SQI risk；timestamp、单位和通道顺序必须验证。
7. D7：peak/PPI/HR/PPG-derived PRV 使用单一 corrected 后端；自动双极性，RED/IR 独立检测，以有效状态和 SQI 选择 primary，不生成或移动 consensus peaks。PPI 有效范围 0.30–2.00 s；HR/PPI 至少 8 s 且至少 5 peaks；time-domain PRV 至少 60 s；frequency PRV 120 s 仅探索、300 s 才可确认；SDNN ddof=1，pNN50 为 0–1 fraction。
8. D8：PTT ECG→PPG transit delay 只能在 training subjects 拟合；评价同时保留 uncorrected/corrected timing、peak F1、PPI/HR error、coverage 与 failure。任何 evaluation subject 与 delay-fit roster 重叠均拒绝。

## 无预校准 EKF 的物理边界 / Physical boundary

- 首个有效加速度仅在线建立 roll/pitch，yaw 设为零但明确不可观。
- 持续线性加速度与倾斜在六轴条件下不可完全区分；绕重力轴 gyro bias 也不可观。
- 低 covariance 只表示滤波器内部统计不确定性，不等于真实重力方向已被外部真值验证。
- Frailty3 没有姿态/重力真值；其 EKF/LPF 结果只能作为角色级 proxy，不能据此宣布真实场景绝对准确率。

## 排除项 / Rejected interpretations

- 不允许 per-record 或全数据拟合 scaler/imputer/阈值后再做 OOF。
- 不允许用旧 256/500 Hz 权重直接声称兼容 400 Hz corrected profile。
- 不允许以 LPF 自动替代 EKF 失败，或以 EKF 结果掩盖 LPF 对照。
- 不允许把 PPG-derived PRV 继续无版本地称为 ECG HRV。
- 不允许把 profile ID、algorithm ID 或数据来源塞入 reason codes。
