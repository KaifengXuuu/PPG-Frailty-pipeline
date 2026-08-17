# Fold Scaling 及 M1/M2 绑定 / Fold Scaling and M1/M2 Binding

## 1. M2 唯一未来主协议 / Sole future protocol from M2

M3 的所有可学习 preprocessing statistics 必须绑定：

- fold registry: `frailty3_future_corrected_sgkf5_v2`;
- 29 subject Frailty3 snapshot；
- subject-level 5-fold；
- repeats = 5；
- split seeds = 42, 10042, 20042, 30042, 40042；
- training seed = split seed + zero-based fold index；
- protocol = `frailty3_fixed_epoch_oof_v2_corrected_sgkf`;
- fixed epoch、no early stopping；
- OOF 在训练完成后只评估一次。

历史 `frailty3_historical_sgkf5_sklearn142_bug_v1` 只用于复现。它的 25 个 folds 中有 6 个缺类别；不得用于未来路线选择或 leaderboard。

The corrected registry materializes membership. Runtime code resolves the saved roster and must not call a splitter again.

## 2. Fold artifact / Fold artifact

`fit_fold_scaler` 只接受与选定 repeat/fold 的 exact training subject set 完全相同的 observed subject IDs。少一个、多一个、重复、或混入 OOF subject 均 fail closed。artifact 至少记录：

- dataset manifest ID/SHA；
- fold registry ID、完整 payload SHA；
- repeat index、fold index、split seed、training seed；
- training 与 OOF subject rosters 及 roster hash；
- feature names/order；
- imputer、center、scale、method、clip policy；
- M3 preprocessing registry hash 与 producer source hash。

This artifact travels with every derived feature/model result so that scaling can be reproduced without reading OOF values.

## 3. Scaling 层次 / Scaling layers

| 层次 / Layer | Fit scope | 规则 / Rule | 泄漏边界 / Leakage boundary |
|---|---|---|---|
| waveform window view | 当前 window | median, IQR/1.349, no clip；可逆 | 不替代 fold scaler |
| IMU/tabular feature scaler | exact training subjects | robust median/IQR 或 standard mean/std | 禁止 OOF refit |
| imputer | exact training subjects | per-feature median | OOF transform only |
| amplitude-risk model | exact training subjects | log AC、log |DC|、log AC/DC 的 robust center/scale；|z|>6 heuristic | 仅 SQI risk，不作设备真值 |

zero IQR/zero scale 统一置为不可除或 scale=1 的明确合同位置；不得静默产生 Inf/NaN，也不得根据 OOF 分布选择修正值。

## 4. M1 V3 顺序路由绑定 / Binding to M1 V3 sequential routing

```text
input validation
  -> common M3 preprocessing
  -> SQI (mandatory) + Motion detector (optional)
  -> join on identical window/sample/time bounds
  -> high + static/not_evaluated: bypass denoiser, shared features
  -> low OR motion:
       pre-frozen manual policy = drop XOR denoise_then_extract_features
  -> unrecoverable/invalid/failure: explicit null/no-result
```

Motion activity 与 signal quality 是两个轴。B/R 只提供 static activity 标签，S/W 提供 motion 标签；它们不是 SQI 真值。启用 detector 后仍未返回时，不得提前进入 high-quality branch。

Activity state and signal quality state must remain independent. Motion detector output may inform routing but does not replace SQI.

## 5. M3 状态到 M1 的边界 / M3-to-M1 status boundary

| M3 status | M1 语义 / M1 semantics |
|---|---|
| valid | ok |
| repaired / partial | partial，保留 repair/coverage |
| invalid | invalid_input |
| insufficient / no_estimate | insufficient_quality |
| initialization_pending, stream ongoing | processing_lag |
| initialization_pending, end of stream | insufficient_quality |

drop 是用户预选策略导致的合法 abstention；failure 是算法没有产生合同输出。两者的计数、reason、coverage 与 denominator 必须分开。

## 6. 数据阶段与时序边界 / Stage and temporal boundary

仅冻结以下事实：

- B = baseline；
- R = relax/recovery；
- S = stand-and-sit；
- W = walk；
- B/R 属于 static activity 范围，S/W 属于 motion；
- S/W 在 Relax 前。

R/S/W 编号的完整动作顺序、每次动作时点与重复定义尚未确认。因此 M3 只保存 record/stage/window 时间边界，不从文件编号推断更细生理事件。未来 recovery speed、运动上下限 HR 等 feature 必须先有 M2 可审计的时间锚点。

## 7. Benchmark 共同约束 / Common benchmark constraints

所有 denoiser、Motion detector、SQI threshold、EKF/LPF、peak route 和 classifier candidate 必须：

1. 使用相同 corrected membership、5 repeats、5 folds 与 seeds；
2. 训练内拟合 threshold/scaler/imputer/transit delay；
3. 使用相同 candidate-window hash 与预路由 denominator；
4. 同时报告 subject、stage、window、time、HR/PPI event coverage；
5. 报告 no-result、policy drop、denoiser failure 和 feature failure；
6. 先满足 coverage/risk 门，再比较 paired BA/macro-F1；不能删除困难 subject 后重算。

## 8. 尚未完成 / Not yet complete

M3 已实现 fold resolver/scaler artifact 和泄漏回归测试，但尚未替所有历史候选完成 5×5 corrected-fold rerun。当前历史绝对分数与 confusion matrix 只能作为历史材料，不能直接选出最终路线。

