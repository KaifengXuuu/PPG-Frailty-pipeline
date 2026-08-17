# M0 论文证据、结果评价与表述边界 / Paper Evidence and Claim Boundaries

- 状态 / Status：`confirmed_by_code_and_outputs`
- 目的 / Purpose：区分可写入论文的事实、只能作为探索性结果的证据，以及禁止作出的性能声明。

## 1. 关键定量结果 / Key quantitative results

| 路线 | 协议/数据 | 关键结果 | 证据等级 | 论文评价 |
|---|---|---|---|---|
| v7 detector | GroupKFold；阈值在被评估数据拟合 | F1≈0.095，BA≈0.050；PR-AUC=1、ROC-AUC=NaN | E1 invalid | 标签几乎全 motion，指标组合自相矛盾，不可作为有效 detector |
| v7 setup1 denoiser | 5 folds；同一 noisy PPG proxy target | SNR `-5.83,-6.74,+3.30,+2.50,-7.33 dB` | E1 invalid | `holdout` 只是最后一折 |
| v7 setup2 denoiser | ECG/peaks/p6 进入推理输入 | SNR `-5.92,-6.66,-6.17,-8.60,+4.29 dB` | E0 leakage | 直接输入泄漏、不可部署 |
| v7.2 setup1 | true subject holdout | walk SNR -7.39 dB、HR MAE 35.64 bpm；run -5.43/33.70 | E2 negative | 切分更好，但效果明确为负 |
| v7.2 setup2 | true subject holdout | walk -6.40 dB/33.44 bpm；run -6.68/37.80 bpm | E2 negative | ECG-HR proxy 未改善泛化 |
| v7.4 detector | subject holdout | OR BA .500；AND .573；AE .649；fused AND .670 | E2 limited | 标签是 activity，不是窗口 artifact truth；仅可作 activity/motion proxy |
| v7.4 MaskNet | inner validation only | walk best val loss .9878；run .7024 | E1 proxy | 无 holdout/SNR/peak/IBI 指标；`a` 全为 1.0 |
| v8/Stage-2 | 预期训练 | 实际结果目录均 0 文件 | E0 blocked | 存在确定性/高概率运行阻断，不能称已完成 |
| legacy v8 detector | PTT internal holdout | fused BA .9880/F1 .9881；IMU-only BA .9992；PPG-only BA .7203 | E1 biased | 近满分由 IMU activity 主导；CV transform leakage |
| hybrid raw+IMU | train/val proxy objective | best val .54578，epoch 8 | E1 proxy | 无 clean reference、无 holdout scorecard |
| hybrid + linear baseline | 相同 split/proxy objective | best val .45273，epoch 2 | E1 proxy | 相对 objective 下降≈17%，不等于真实去噪提升 |
| PPG peak model | 5-fold OOF | event F1@20 ms .3870 | E2 negative | target 是 ECG timing，且窗口重叠重复计 beat |
| PPG peak model | internal holdout | event F1 .3780 | E2 negative | 仍非 PPG pulse-peak accuracy |
| PPG peak model | extra holdout | event F1 .1540；SIM .0948 | E3 external negative | 明确显示外部泛化失败 |
| PPG-only gate | CV / internal / extra | F1 .9066 / .8695 / .4690；extra AUC .4088 | E3 negative | 外部 score 方向反转，不可部署 |
| IBI dense output | internal holdout | aggregate MAE 4.8271 s；VitalDB 18.2699 s | E2 invalid aggregate | pseudo-label异常与有界预测不相容 |
| IMU detector A | external SIM | F1 .7542，BA .7699，AUC .8269 | E3 promising | 最有希望之一，但缺 subject-level CI |
| Light CNN detector B | external SIM | F1 .7634，BA .7802，AUC .8642 | E3 promising | 略优于 A；仍是 pooled overlapping windows |

## 2. Evidence tier 定义

- `E3 external`：真正外部数据或严格保留集；仍需检查 aggregation 与 CI。
- `E2 holdout`：subject-disjoint holdout，关键统计未在该 holdout 拟合。
- `E1 proxy/biased`：只有 validation、代理目标、CV 选择偏差或活动标签替代 artifact truth。
- `E0 invalid/blocked`：明确泄漏、运行阻断、空结果或指标定义无效。

## 3. 可安全写入论文的事实 / Claims supported by evidence

1. 多代方法系统探索了透明滤波、IMU ANC、DWT/AE/UNet、STFT mask、pseudo-supervised hybrid、direct peak/IBI 和 motion detector。
2. subject-disjoint v7.2 holdout 的 proxy SNR 全部为负，支持放弃 full-waveform reconstruction 主线。
3. v7.4 activity detector 的最佳 holdout BA 为 0.670，不能证明窗口级 artifact accuracy。
4. hybrid 已实现 Python/ONNX 工程链，但没有真实 clean/holdout 性能证据。
5. direct PPG-only peak route 在 external SIM 的 event F1@20 ms 仅 0.0948，表明明显 domain shift。
6. 10-channel PPG+IMU Light CNN 在 external SIM 上 BA 0.7802、F1 0.7634，是后续 motion-state 候选，但尚需 subject-level统计。
7. ECG→PPG delay 跨数据集显著不同：PTT .2305 s、iAMwell .2796 s、MIMIC .3228 s、SIM .2923 s、VitalDB .4626 s；监督不能忽略该差异。

## 4. 禁止或必须改写的表述 / Unsupported claims

- 禁止：“模型恢复了真实 clean dynamic PPG。”
- 禁止：“legacy v8 以接近 100% 准确率检测 PPG motion artifacts。”
- 禁止：“当前 peak model 检测 PPG pulse peaks。”应改为“尝试从 PPG 估计未校正 ECG R timing”。
- 禁止：“已经输出 HR accuracy。”当前代码未计算 bpm HR MAE/MAPE。
- 禁止：“hybrid B 比 A 去噪提高 17%。”只能写“proxy validation objective 低约 17%”。
- 禁止：“v7/v7.2 的 holdout 均为独立 test。”v7 的 holdout 只是最后一折；v7.2 才有固定 subject holdout。
- 禁止：仅凭 preview 图、平滑波形或静态 sit 表现宣称 motion waveform 恢复成功。
- 禁止：把目录、模型文件或 ONNX 存在等同于严格验证完成。

## 5. 建议论文叙事 / Thesis-ready interpretation

历史实验表明，在缺少真实运动条件下 clean PPG、ECG–PPG delay 跨域变化且 IMU 与光学伪影关系非固定的条件下，完整波形重建缺乏可辨识监督。项目因此将研究重点从视觉上“更干净”的波形转向三个可检验目标：

1. 明确输出 coverage 的 high-quality-only/SQI 策略；
2. 使用独立外部数据评价的 motion-state detector；
3. 直接输出唯一 beat、IBI、HR/HRV、置信度和失败状态，并以 ECG 仅作训练/评价 reference。

后续实验必须以 raw/no-denoising 与 high-quality-only 为基线；任何 coarse denoising 只有在 HR/PPI/coverage 指标上显著且稳定优于基线时才可保留。

