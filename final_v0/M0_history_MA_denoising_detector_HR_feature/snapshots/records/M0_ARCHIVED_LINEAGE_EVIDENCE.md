# M0 归档版本与实际输出生产关系 / Archived Lineage and Output Provenance

- 状态 / Status：`complete_supplement`
- 目的 / Purpose：解释中间修复版本与现存目录的精确关系；不把修复链重复登记为独立科学方法。

## 1. 有明确生产关系的历史输出

| 实际输出 | 最匹配代码 | 关键结果/结构 | 方法学定位 |
|---|---|---|---|
| `results_stage1/`（17文件） | `Arc/pttppg_stage1_detector.py` | holdout OR BA `.6920`、F1 `.8656`、recall `1.0`；AND BA `.8623`、F1 `.8427`、precision `.9961` | 独立feature-threshold/logistic activity detector前身；lag在拼接记录上，activity≠artifact truth |
| `results_detector_v8/`（6文件） | `Arc/pttppg_detector_v8_scores.py` | `version=detector_v8_scores`；holdout BA/F1约`.9966`；旧NPZ schema | legacy v8基础评分器；全train anchor/mu/cov/lag先于CV，IMU活动主导 |
| `results_v7_3/`（33文件） | `Arc/pttppg_pipeline_v7_3_noleak_viz.py` | rule detector + walk/run MaskNet PT/ONNX/meta；OR BA约`.5023`、AND约`.5786` | v7.4直系前身；无独立denoiser holdout，离散subject参数/对齐风险 |
| `results_v8_audit/{1_0.5,2_0.5,6_1}` | 根`pttppg_detector_v8_scores_audit_fix9.py` | 三套完整summary/audit/ROC/PR与按配置NPZ | fix9是首次补齐`asdict`和曲线import的近完成版本；核心CV transform leakage仍在 |
| `results_v72_noleak/` | 根`cnnppg_v7.py`最无歧义；Arc v7.2可通过显式outdir写相似schema | AE skip；四组holdout SNR均负 | 默认目录证据更支持根脚本；不得仅凭高代码相似度归因给Arc分支 |

## 2. Detector audit fix链

```text
scores.py
  → scores_audit.py        (未定义局部变量 / dict(dataclass))
  → audit_fix2.py          (best_thr等在定义前使用)
  → audit_fix3.py          (ndarray JSON序列化失败)
  → audit_fix6.py          (_json_sanitize未定义)
  → audit_fix8.py          (dict(dataclass) TypeError；部分绘图异常被吞)
  → root audit_fix9.py     (修复序列化/曲线导入；方法学偏差仍在)
```

这些文件表示工程修复lineage。M0 registry只保留一个 `D01 legacy v8 handcrafted detector`方法ID；否则会把同一算法的错误修复次数误写成方法多样性。

## 3. PPG Dash bundle contract lineage

- 早期Arc Dash loader要求`ppg_mu/ppg_cov/imu_mu/imu_cov/logreg_coef/logreg_intercept`。
- 实际trainer bundle使用`mu_ppg/cov_ppg/mu_imu/cov_imu/coef/intercept`。
- 因此早期Arc Dash不能直接加载现存trainer NPZ；根`ppg.py`虽改成实际键名，仍有默认文件名不存在、window contract和runtime错误。

## 4. 论文使用边界

1. Stage1的`.8623`只能称“同域activity-label holdout AND规则结果”，不能称artifact detector accuracy。
2. v8近满分不能覆盖PPG-only `.7203`及IMU-only `.9992`的审计事实。
3. 任何Arc fix结果必须先证明运行跨过其确定性失败点；代码文件存在不等于结果由它成功生成。
4. 输出归因不确定时写“schema/lineage compatible”，不得写“exactly reproduced”。

