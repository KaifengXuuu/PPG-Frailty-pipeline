# Frailty3 双 Fold 注册表与唯一未来主协议

## 为什么需要双注册表

历史运行调用 scikit-learn 1.4.2 的 `StratifiedGroupKFold(..., shuffle=True)`。该版本在 shuffle 后移动了 group class-count rows，却没有同步重映射 `groups_inv`，因此受试者仍不跨 fold，但类别计数被错误关联到别的 group。结果是 25 个历史 OOF folds 中 6 个缺少至少一个类别。

M2 不静默改写历史 membership：

1. `frailty3_historical_sgkf5_sklearn142_bug_v1.json` 精确物化历史分折，只用于复现和解释旧结果。
2. `frailty3_future_corrected_sgkf5_v2.json` 同步置换 group 与 class-count，再执行 SGKF 贪心分配；它是未来所有路线和 benchmark 的唯一主协议。

## 修正算法

对每个 seed：

1. 以稳定 UTF-8 byte order 排列 29 个 `(subject_id,class_id)`，一个 subject 一行。
2. 建立每个 group 的 class-count vector。
3. 使用 NumPy `RandomState(seed)` 产生 group permutation。
4. 同时置换 group IDs 与对应 count rows；这是历史版本缺失的关键步骤。
5. 按 class-distribution standard deviation 稳定降序，逐 group 放入使全局 class-proportion deviation 最小的 fold；平局选择当前样本数较少的 fold。
6. 物化 membership，训练时禁止重新调用 splitter。

## 主协议冻结值

- dataset：M2 Frailty3 29-subject snapshot
- raw sampling rate：400 Hz
- split：subject-level 5-fold corrected SGKF
- repeats：5
- split seeds：`42,10042,20042,30042,40042`
- training seed：`split_seed + zero_based_fold_index`
- epoch：每个 config 预先冻结的 fixed epoch
- early stopping：禁止
- outer OOF data in training loop：禁止；训练完成后只评估一次
- output role：`oof_validation`

## 强制不变量

- 同一 repeat 内 train 与 OOF subject 不相交；5 个 OOF folds 两两不交且并集为全部 29 subjects。
- subject 的所有 B/R/S/W 文件始终随 subject 移动，不得按文件或窗口再分折。
- 未来主注册表每折三类齐全；每类在 5 folds 的计数最大差不超过 1。
- registry、dataset manifest、label map 或 role filter 任一改变都必须创建新 ID，不能覆盖。
- 所有候选在同一物化 membership、seeds、repeats 和训练预算上重跑；历史分数不直接进入未来 leaderboard。
