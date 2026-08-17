# EKF 主路线与 LPF 对照合同 / EKF Primary and LPF Comparator Contract

## 1. 共同前端与公平比较 / Shared frontend and fair comparison

EKF 与 LPF 只能改变 gravity estimator。两路线必须共享：

- 同一 400 Hz 六轴输入 AX, AY, AZ, GX, GY, GZ；
- 显式单位转换：g/mg/m/s² → m/s²，deg/s/rad/s → rad/s；
- 同一质量门、timestamp gate、source/repair mask；
- 三阶 causal SOS：acceleration 20 Hz，gyro 40 Hz；
- 同一窗口/chunk 边界、初始输入、float64 与 vector jerk；
- 同一后续 SQI/Motion、fold、seeds、coverage denominator 和 feature schema。

There is no silent route substitution. EKF failure returns its own state and mask; it never falls back to LPF. LPF is an independently registered comparator.

## 2. 坐标与名义状态 / Frames and nominal state

实现使用 scalar-first `q_NB=[w,x,y,z]`，表示 body 到 navigation 的旋转；`R_NB=R(q_NB)`。navigation 重力向量为

```text
g_N = [0, 0, 9.80665] m/s²
g_B(q) = R_NB(q)^T g_N
a_dynamic = a_filtered - g_B
```

名义状态为 quaternion 与 gyro bias，右乘六维误差状态为

```text
delta_x = [delta_theta, delta_b_g]
q_true ≈ q_nominal ⊗ Exp(delta_theta)
```

绝对 yaw 没有观测源；输出 yaw 仅是相对积分坐标。accelerometer bias 当前不在状态中。

## 3. 在线无预校准初始化 / Online initialization without precalibration

系统从 `initialization_pending` 开始。首个满足 `0.5g ≤ ||a|| ≤ 1.5g` 的过滤后加速度向量用于对齐 tilt；该样本不再重复传播。初始化值：

- gyro bias mean = [0,0,0] rad/s；
- tilt sigma = 20°；
- yaw sigma = 180°；
- gyro bias sigma = 5°/s；
- quaternion 每次构造和更新后归一化。

“无预校准”不等于“无初始化”：它表示没有独立、已验证静止区间来估计 bias/姿态。首个可接受向量可能包含真实线性加速度，因此 tilt、gravity 和 gyro bias 的物理可观性保持 `unverified_no_static_precalibration`。

No-precalibration does not mean that tilt is physically identifiable from arbitrary motion. Persistent linear acceleration and tilt remain confounded.

## 4. 预测模型 / Propagation model

令 `omega = gyro_filtered - b_g`，时间步 `dt=1/400`：

```text
q(k+1) = normalize(q(k) ⊗ Exp(omega * dt))
F = [ -skew(omega)  -I
          0           0 ]
Phi = I + F*dt + 0.5*(F*dt)^2
P- = Phi P Phi^T + Q
```

过程噪声包含 gyro white noise density `sigma_g=0.002` 与 bias random walk `sigma_b=0.0002`，并保留 bias random walk 在同一步内对 attitude 的 `dt³/3` 与交叉 `-dt²/2` 项。省略这些项会系统性低估 attitude/bias covariance。

## 5. 动态加速度门控与切平面更新 / Dynamic-acceleration gating and tangent update

每 4 个样本尝试一次 gravity-direction update。预测与观测单位方向为：

```text
h = R(q)^T [0,0,1]
z = a / ||a||
r = T(h) (z - h)
H = [ T(h) skew(h)   0 ]
```

`T(h)` 是与 h 正交的确定性 2-D tangent basis。measurement covariance 自适应放大：

```text
rho = abs(||a||/g - 1)
eta = ||a(k)-a(k-1)|| / (g*dt)
alpha = clip(1 + (rho/0.05)^2 + (eta/2)^2, 1, 100)
R = (5 deg)^2 * alpha * I2
```

`alpha ≥25` 记录为 dynamic-acceleration downweighted。只有 norm gate 通过且 2-D NIS ≤13.8155 才接受更新。该结构降低运动加速度对 tilt 的支配，但不能把真实持续加速度与重力完全分离。

## 6. 校正与数值稳定 / Correction and numerical stability

接受更新时：

- 解线性系统得到 Kalman gain，不显式求逆；
- `q ← normalize(q ⊗ Exp(delta_theta))`；
- `b_g ← b_g + delta_b_g`；
- 使用 Joseph form 更新 covariance；
- 应用右乘误差 reset Jacobian；
- covariance 对称化；
- 最小 eigenvalue <−1e−10 视为数值失败；0 至 1e−15 之间 floor 到 1e−15 并记录；
- 任一 `|b_g|>0.35 rad/s`、非有限 quaternion/covariance 或超时条件进入 `no_estimate`。

These safeguards improve numerical consistency; they do not make unobservable physical quantities observable.

## 7. 状态机 / State machine

| 状态 / State | 进入条件 / Entry | 输出语义 / Output semantics |
|---|---|---|
| initialization_pending | 未找到可接受初始化向量；或尚未同时满足 0.5 s、20 accepted updates、tilt sigma≤10° | gravity/dynamic/jerk feature mask false |
| tracking | 已达到 tracking gate，且当前更新/预测仍在边界内 | 可输出；受共同 finite/jerk mask 约束 |
| prediction_only | 已 tracking；当前未接受更新，且距离最近 accepted update 未超过 2 s、tilt sigma≤20° | 可输出但诊断必须保留 |
| no_estimate | prediction-only >2 s、tilt sigma>20°、bias/covariance/quaternion divergence | terminal latch；仅显式 `reset_for_new_session` 可恢复 |
| invalid | 单样本非有限/形状异常等 | 不伪造 gravity |

流式调用必须长期持有同一个 `CausalImuProcessor`。每 chunk 新建 processor 会反复初始化并破坏 filter/ESKF/jerk/timestamp 连续性。

## 8. LPF 受控对照 / Controlled LPF comparator

`imu_lpf_si_400_causal_v1` 在共同 20 Hz acceleration frontend 后，对每轴使用二阶 0.3 Hz causal SOS：

```text
g_B_LPF = LPF_0.3Hz(a_filtered)
a_dynamic_LPF = a_filtered - g_B_LPF
```

LPF 不估计 quaternion、gyro bias 或 uncertainty。它可能把缓慢真实运动加速度吸收到 gravity，也会在转动时产生相位延迟；但成本低、确定性强，是必要的工程 baseline。

## 9. 因果/离线边界 / Causal/offline boundary

当前 registry 仅激活 `imu_ekf_si_400_causal_v1` 与 `imu_lpf_si_400_causal_v1`。M3 没有活动 offline IMU profile，因而：

- 不得把整段 causal one-shot 称为 offline zero-phase；
- 不得用 filtfilt LPF 与 causal EKF 比较后声称只有 gravity estimator 不同；
- 如未来增加 offline 路线，必须为 EKF 和 LPF 同时冻结 initialization、padding、phase、mask 与 latency 语义并创建新 IDs。

## 10. 每次输出必须携带 / Required diagnostics

profile/registry ID 与 hash、gravity method、state per sample/counts、terminal state、valid fraction、first/last valid index、silent_fallback=false；EKF 另含 quaternion、gyro bias、tilt sigma、NIS、accepted/downweighted flags，以及 no-precalibration、yaw、bias observability 限制。

