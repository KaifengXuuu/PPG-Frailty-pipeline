# EKF 与 LPF 结果对比 / EKF versus LPF Results

## 1. 结论先行 / Outcome first

在有真值的固定合成 IMU fixture 上，无预校准 quaternion ESKF 的 gravity/dynamic-acceleration error 明显低于 0.3 Hz LPF，但 coverage 因显式在线初始化略低。Frailty3 没有 gravity 或姿态真值；其首 6 s 结果只能说明两路线产生不同的信号分解与 coverage，不能用于计算姿态/重力准确率，也不能单独决定临床或分类优越性。

The current evidence supports keeping ESKF as the engineering primary and LPF as a controlled comparator. It does not establish clinical superiority.

## 2. 比较条件 / Comparison conditions

两路线共享：

- 同一输入与 M2 manifest；
- 400 Hz；
- 显式 g/deg/s→SI；
- 同一 quality/timestamp contract；
- causal acceleration 20 Hz、gyro 40 Hz；
- 同一 segment、float64、dynamic acceleration 与 vector jerk 定义；
- no silent fallback。

仅 gravity estimator 不同：ESKF orientation+bias model vs second-order 0.3 Hz acceleration LPF。

## 3. 合成真值结果 / Synthetic-truth results

Evidence ID: `m3_ekf_lpf_synthetic_truth_v1`; fixture SHA-256 `bcb0e796b97d10e96a32b61c4fe17eb31de8729724411735158c2abd82e00f24`.

| Metric | ESKF primary | LPF 0.3 Hz |
|---|---:|---:|
| profile | `imu_ekf_si_400_causal_v1` | `imu_lpf_si_400_causal_v1` |
| valid samples | 4600 | 4799 |
| coverage | 0.958333 | 0.999792 |
| gravity vector RMSE | 0.149350 m/s² | 2.151857 m/s² |
| gravity angle P95 | 1.778540° | 18.308295° |
| dynamic acceleration vector RMSE | 0.118810 m/s² | 2.123582 m/s² |
| terminal state | tracking | tracking |
| silent fallback | false | false |

解释：fixture 包含 roll/gravity 与已知 dynamic acceleration，适合检查坐标、传播、gate 和分解实现。ESKF coverage 排除 online initialization；LPF 只有 first jerk sample 无效。该 fixture 是工程构造，不代表真实佩戴、bias、motion morphology 或临床 population。

## 4. Frailty3 首 6 s 代理结果 / Frailty3 first-six-second proxies

Evidence ID: `m3_ekf_lpf_frailty3_first6s_proxy_v1`; dataset `frailty3_m2_20260815_a054800abda272f6`; 29 subjects、261 records；每 record 读取前 6 s、无 padding。

| Role family | Route | Median coverage | Median dynamic RMS (m/s²) | Records with any valid / total | Terminal no-estimate |
|---|---|---:|---:|---:|---:|
| B | ESKF | 0.916667 | 0.273592 | 29/29 | 0 |
| B | LPF | 0.999583 | 0.099511 | 29/29 | 0 |
| R | ESKF | 0.916667 | 0.327006 | 116/116 | 0 |
| R | LPF | 0.999583 | 0.229736 | 116/116 | 0 |
| S | ESKF | 0.916667 | 1.675688 | 58/58 | 0 |
| S | LPF | 0.999583 | 1.888982 | 58/58 | 0 |
| W | ESKF | 0.916667 | 1.698317 | 58/58 | 0 |
| W | LPF | 0.999583 | 1.588134 | 58/58 | 0 |

### 4.1 正确解释 / Correct interpretation

- S/W 的 dynamic RMS 高于 B/R，符合 activity 能量差异的直觉，但这不是 Motion detector BA；监督 BA 必须在 M2 folds 上计算。
- ESKF 0.916667 coverage 主要反映 0.5 s online tracking gate；LPF 0.999583 主要只损失第一个 jerk sample。
- 某一 role 上 dynamic RMS 更低不等于 gravity 更准确：真实 dynamic acceleration 未知。
- ESKF 的 median gravity-norm absolute error 在 evidence 中为 0，是因为 `||g_B(q)||=9.80665` 被模型**按构造固定**，不是数据证明的准确率。
- LPF 的 gravity norm deviation 会混合 sensor scale/DC bias、低频线性加速度和真实姿态动态；不能单独解释为 LPF error。
- record 可能从运动中开始，正是 no-precal initialization 的不利条件。

### 4.2 禁止表述 / Prohibited interpretation

不得写 “ESKF posture accuracy = …”、 “ESKF gravity accuracy is perfect on Frailty3”、 “LPF removes more motion because RMS is lower” 或 “ESKF/LPF improves Frailty classification”。该 evidence 没有支持这些命题所需真值或 OOF classifier result。

## 5. 为什么保留 ESKF 主路线 / Why ESKF remains primary

1. 合成有真值 fixture 上，对旋转+动态加速度的 gravity/dynamic 分解误差显著更低。
2. 输出 uncertainty、NIS、accepted/downweighted update、gyro bias 与状态机，可显式 abstain。
3. quaternion 避免历史 Euler singularity。
4. LPF 无法表示 orientation/bias uncertainty，并会把低频真实运动吸收到 gravity。

代价是 initialization coverage、计算量、parameter sensitivity 与无预校准不可观性。因此 LPF 必须保留为低复杂度、确定性 comparator；不把它降格为“错误算法”，也不把 EKF 升格为未经真实 reference 验证的“真值算法”。

## 6. 下一阶段公平验收 / Next-stage fair acceptance

### 6.1 物理/工程层

- 扩展 synthetic sweeps：initial tilt、yaw、rotation rate、gyro bias、accelerometer scale/DC、持续加速度与 start-in-motion；
- 使用 turntable/motion-capture 或其他独立 reference 获取真实 gravity/orientation；
- 报告 error 分布、coverage、time-to-tracking、prediction-only/no-estimate、reset、latency/RAM。

### 6.2 任务层

- M5：B/R vs S/W Motion detector，在 corrected subject folds 上报告 BA/confusion matrix/coverage；
- M8：以同 seeds/folds 对比 ESKF/LPF 下游 Frailty BA、macro-F1、risk–coverage 与 paired differences；
- D8：相同 preprocessing route 下报告 ECG-referenced PPG peak/PPI/HR；
- M9：三档硬件上的 latency、RAM、bundle、power 和 1 h streaming。

最终判断先要求物理/信号门和最低 coverage，再比较 paired predictive utility，最后用资源成本打破平局。任何路线失败都保留自己的 no-result，不允许通过 silent fallback 抬高 coverage。

