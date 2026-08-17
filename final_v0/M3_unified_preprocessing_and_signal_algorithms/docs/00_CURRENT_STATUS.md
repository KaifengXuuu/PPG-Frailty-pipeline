# M3 当前状态 / Current Status

## 1. 里程碑结论 / Milestone conclusion

- 里程碑 / Milestone: **M3 — unified preprocessing and signal algorithms**
- 状态 / Status: **核心合同与参考实现已建立；工程参考测试通过；真实监督 benchmark 尚未完成。 / Core contracts and reference implementation are established; engineering reference tests pass; real supervised benchmarks remain pending.**
- 当前日期 / Date: 2026-08-15
- 未来主采样率 / Future primary sampling rate: 400 Hz
- IMU 主路线 / Primary IMU route: 无静态预校准、scalar-first quaternion error-state EKF
- IMU 受控对照 / Controlled IMU comparator: 0.3 Hz 二阶因果低通重力分量
- 写入边界 / Write boundary: 本里程碑产物位于 `final_v0/M3_unified_preprocessing_and_signal_algorithms/`；根目录代码只作为历史复现证据。

M3 已把过去散落在 classifier、denoiser、peak helper 和 visualization 脚本中的预处理语义收敛到一个 registry-bound 公共边界。M4 及以后不得复制出第二套“活动”预处理实现。

M3 consolidates preprocessing semantics previously scattered across classifiers, denoisers, peak helpers, and visualization scripts into one registry-bound public boundary. M4 and later milestones must not create a second active preprocessing implementation.

## 2. 权威层级 / Authority order

当说明文档与机器文件发生冲突时，按下列顺序判定：

1. `registries/preprocessing_profiles_v1.json` 与 `registries/physiology_algorithms_v1.json`：冻结 profile、采样率、滤波与生理算法参数。
2. `src/m3_signal_core/`：实际运行语义。
3. `tests/`、`fixtures/` 与实时测试结果：可执行验收。
4. `evidence/`：固定输入上的工程证据与历史 crosswalk。
5. `docs/` 与 `algorithm_diagrams/m3/`：面向人阅读的解释，不覆盖前四项。

If explanatory text conflicts with machine-readable artifacts, the registries and implementation prevail, followed by executable tests and evidence. Documentation is explanatory, not a parameter override.

## 3. 已冻结的活动合同 / Frozen active contracts

| 域 / Domain | 当前合同 / Current contract | 关键不变量 / Key invariant |
|---|---|---|
| 输入质量 / Input quality | 滤波和缩放前检查形状、通道、时间轴、非有限值、gap、flatline；有限内部短 gap 才允许插值 | invalid/insufficient 不生成伪零特征 |
| PPG | 静态 0.2–8 Hz；motion/peak/denoiser 0.4–8 Hz；三阶 Butterworth；notch disabled | 原始 DC/AC 与 RED/IR 比例在归一化前保存 |
| 离线/移动 / Offline/mobile | 离线 PPG 为 zero-phase；移动 PPG 为 causal stateful，profile ID 分离 | 不声称两者逐样本数值相同 |
| IMU 单位 / IMU units | 输入单位必须显式给出；Frailty g、deg/s 转为 m/s²、rad/s | 禁止幅值猜测单位 |
| IMU 前端 / IMU frontend | acceleration 20 Hz、gyro 40 Hz 三阶因果 SOS | EKF/LPF 完全共享 |
| 重力主路线 / Gravity primary | 无预校准 quaternion ESKF，状态为 initialization_pending/tracking/prediction_only/no_estimate | no_estimate 锁存至显式 session reset；禁止静默 LPF fallback |
| 重力对照 / Gravity comparator | 0.3 Hz 二阶 causal LPF | 仅重力估计器与 EKF 不同 |
| 峰与生理 / Physiology | corrected_v1 双极性峰、PPI 0.30–2.00 s、HR、PPG-derived PRV | 不删除导致异常 PPI 的源峰；不生成 RED/IR 共识峰 |
| scaling | window view 可逆 robust scaling；模型 scaler/imputer 只在 M2 training subjects 拟合 | OOF subject 混入即 fail closed |
| 路由绑定 / Routing binding | M1 V3：SQI 必做，Motion 可选；high quality bypass；low/motion 预选 drop XOR denoise | drop 为合法 abstention，failure 与 drop 分开 |

## 4. 公共入口 / Public entry points

- `preprocess_ppg`：registry-bound PPG 清洗、滤波、原始幅值上下文和显式重采样合同。
- `CausalImuProcessor.process_chunk`：移动/流式主入口，跨 chunk 保留 SOS、ESKF/LPF、jerk 和 timestamp 状态。
- `preprocess_imu`：one-shot facade；用于固定窗口和测试，不替代长期 processor。
- `detect_peaks_corrected`、`derive_ppi`、`compute_hr`、`compute_prv`：共享生理后端。
- `resolve_m2_fold`、`fit_fold_scaler`：绑定 M2 修正后的物化 subject folds。
- `fit_transit_delay`、`evaluate_ppg_against_ecg`：D8 ECG-reference 评价边界。

Each result carries explicit status, masks, reason/issue metadata, and profile provenance. A non-success state is data, not an invitation to substitute stale values.

## 5. 当前验证状态 / Current verification state

2026-08-15 对当前工作树执行不写字节码、且不改报告文件的完整 `unittest` discovery：

- 38 tests run;
- 38 passed;
- 0 failures / 0 errors;
- live runtime: 3.879 s.

该实时结果覆盖合同边界、M2 fold 绑定、D8 train-only transit delay、PPG、scaling、ESKF/LPF、stateful chunk parity、peak/PPI/HR/PRV。已保存的 `M3_REFERENCE_TEST_RESULTS.json` 可能对应较早的 22-test 子集；因此论文或交付引用必须同时写明报告文件 hash 和执行日期，不得用旧计数代表当前测试集。

The live 38-test pass is an engineering regression result. It is not clinical validation and does not replace a versioned persisted report.

## 6. 未完成与禁止过度解释 / Pending work and prohibited over-claims

1. 29-subject 所有候选尚未在 M2 唯一主 folds 上统一重跑；Frailty BA、macro-F1、risk–coverage 与 no-result 仍待 M5/M8。
2. D8 已有无泄漏 evaluator 和合成测试，但尚未在 pulse-transit-time-ppg 的真实 ECG/PPG 标注上完成 OOF benchmark。
3. 现有 Frailty EKF-vs-LPF evidence 只有无真值的信号代理量。它**不是姿态准确率、不是重力准确率、也不是临床优越性证据**。
4. 无预校准六轴 IMU 中，持续线性加速度与倾角不可完全区分；绝对 yaw 不可观；accelerometer bias 未估计；gyro bias 仅部分可观且未由静态校准确认。
5. 当前 registry 只激活 causal IMU profiles；没有活动的 offline EKF/LPF profile。离线 IMU 若未来需要，必须新建 ID、明确状态初始化与边界处理并重新验证。
6. 三档移动平台仍是候选预算，尚无真实设备 latency/RAM/power/bundle 测量。

## 7. 下游使用门 / Downstream gate

M4–M9 只有在同时保存以下 provenance 时才可消费 M3 输出：profile ID、registry SHA-256、输入 manifest/dataset ID、时间/样本边界、单位、quality status、source/repair/feature-valid masks、fold artifact（若拟合统计量）、算法 ID、failure/drop reason 和软件 source hash。

M4–M9 may consume M3 output only when profile, registry, data, window, unit, mask, fold, algorithm, status, and source provenance remain attached.

