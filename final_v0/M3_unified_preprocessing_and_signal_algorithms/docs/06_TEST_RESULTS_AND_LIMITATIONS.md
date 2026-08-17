# 测试结果与局限 / Test Results and Limitations

## 1. 两类测试记录必须区分 / Two test records must remain distinct

### 1.1 已保存历史报告 / Persisted earlier report

`M3_REFERENCE_TEST_RESULTS.json` 当前记录：

- report ID `m3_reference_tests_v1`;
- 22 tests；
- 22 passed；
- 0 failures / 0 errors / 0 skipped；
- elapsed 1.098176950996276 s；
- 只列出 `test_imu_physiology.py` 与 `test_quality_ppg_scaling.py` 的旧 source hashes。

### 1.2 当前工作树实时验证 / Live current-tree verification

2026-08-15 使用 `PYTHONDONTWRITEBYTECODE=1`、不改报告文件，执行完整 unittest discovery：

- 38 tests；
- 38 passed；
- 0 failures / 0 errors；
- elapsed 3.879 s。

新增覆盖来自 `test_contract_edges.py`、`test_fold_reference.py` 以及扩展后的 IMU tests。因而 persisted 22-test report 是可复现历史证据，但已不完整描述当前测试面。正式交付前应由受控 build 工具生成新 report ID 或 version，保存四个测试文件 hashes 和执行环境；不得原位伪装旧报告从未存在。

The 38-test live run is the current engineering verdict; the 22-test JSON is the current persisted artifact. Both facts must be reported until a new versioned report is generated.

## 2. 当前测试覆盖 / Current test coverage

| Test area | 已验证 / Verified |
|---|---|
| strict contracts | non-finite JSON→null；profile provenance 不混入 reason codes；M1 end-of-stream-aware status mapping |
| quality/timestamps | 0.25 s gap exact boundary；1% non-finite；boundary gap；1 s flatline；channel order；repeated timestamp/profile mismatch |
| PPG | frozen filter anchors；raw DC preservation；explicit 500→400 polyphase metadata；dual raw ratios |
| scaling | non-training fit rejected；test distribution 不 refit；reversible window scale；zero IQR fail closed；training-only amplitude gate |
| M2 binding | exact training roster artifact；故意混入 OOF subject 被拒绝 |
| IMU units/filter | g/deg/s 与 SI 等价；causal SOS arbitrary chunk parity |
| ESKF/LPF | initialization pending；static tracking；独立 profile/no fallback；stateful chunk parity；synthetic truth RMSE gate；finite common mask；no_estimate latch |
| physiology | polarity invariance；synthetic event recall；PPI inclusive boundaries；PRV formulas；120/300 s tiers；RED tie/no consensus |
| D8 | train-only transit delay；disjoint evaluation；delay correction；fit-role/subject-overlap leakage rejection |

## 3. 固定 engineering evidence / Frozen engineering evidence

- `ekf_lpf_synthetic_comparison.json`：有明确 gravity/dynamic truth 的合成 IMU；工程 gate pass。
- `filter_response_comparison.json`：PPG SOS coefficients 与频率 anchor。
- `legacy_peak_parity.json`：同一 fixture 上 funcs/ppg duplicates exact；classifier adaptation 与它们存在一个额外 legacy peak，差异被保留。
- `frailty3_signal_integrity_summary.json`：绑定 M2 full-byte/full-numeric scan，29 subjects、261 files、18,152,248 numeric rows、8 columns finite。
- `ekf_lpf_frailty3_role_proxy.json`：261 records 首 6 s paired route proxy；无 gravity truth。
- `historical_preprocessing_crosswalk_v1.json`：12 个根目录历史脚本的 hashes、风险与迁移边界。

## 4. 这些测试没有证明什么 / What these tests do not prove

1. 合成 PPG peak recall ≥0.90（±20 samples）不等于真实 motion PPG 的 ECG-referenced performance。
2. 水平静态 synthetic EKF gravity error 很小，只验证实现与构造真值一致，不证明任意佩戴姿态、持续加速度或设备 bias 下准确。
3. chunk parity 证明 state persistence 一致，不证明算法物理模型正确。
4. filter response 证明数值设计符合 profile，不证明 cutoff 对 frailty 分类临床最优。
5. Frailty dynamic-acceleration RMS 是输出能量，不是误差；没有 motion-capture/gravity truth 时不能转写为 posture/gravity accuracy。
6. PPG-derived PRV formulas 正确不等于与 ECG-HRV 等价。
7. 单元测试不覆盖真实设备丢包、时钟漂移、温度降频、长期 backlog、BLE/USB transport 或 UI。
8. 38-test pass 不替代 M2 corrected 5×5 OOF rerun。

## 5. 待补测试矩阵 / Required future test matrix

### 5.1 IMU

- 多初始 tilt/yaw、不同旋转轴与 angular-rate sweep；
- 已知 gyro bias/random walk、accelerometer scale/DC bias；
- session 从强运动开始、短静止后再运动、持续匀加速；
- NIS gate、dynamic-R downweight、prediction-only 2 s 边界；
- covariance floor/divergence injection 与 reset；
- 随机 chunk sizes、timestamp jitter/gap/duplicate/out-of-order；
- 有外部姿态/重力 reference 的 turntable 或 motion-capture 评价。

### 5.2 PPG/physiology

- 不同 HR、pulse morphology、polarity、baseline drift、clipping、dropout；
- RED/IR 相位差、幅值差和一通道失败；
- ectopic/missed/extra peaks 的 PPI mask；
- 60/120/300 s exact duration/coverage edges；
- pulse-transit-time-ppg 真实 ECG annotation 上的 subject-disjoint D8。

### 5.3 End-to-end

- M1 四个 factorial routing arms；
- M2 5 repeats ×5 folds；
- route-level HR/PPI error、risk–coverage、Frailty BA/macro-F1；
- mobile causal/offline PPG parity envelope，而不是逐样本 equality；
- 三档真实硬件 latency/RAM/bundle/power/1 h stability。

## 6. 论文表述模板 / Paper-language boundary

允许： “The reference implementation passed 38 current engineering regression tests; the archived JSON report covers an earlier 22-test subset.”

不允许： “M3 is clinically validated,” “EKF posture accuracy is X%,” 或 “LPF/EKF is superior on Frailty3”——当前证据尚不支持这些结论。

