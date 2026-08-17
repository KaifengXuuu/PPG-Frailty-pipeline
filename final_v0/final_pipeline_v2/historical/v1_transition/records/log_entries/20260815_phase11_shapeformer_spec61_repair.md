# Phase 11 — ShapeFormer §6.1 strict repair / ShapeFormer §6.1 严格修复

- Status / 状态：implemented and CPU-regression tested; experimental status retained.
- Scope / 范围：`models/shapeformer.py`, strict model factory, model tests,
  generated ShapeFormer model card and its single generation authority.
- Scientific claim boundary / 科学声明边界：this is an implementation and protocol
  repair only. It is not PISD/original ShapeFormer parity and contains no frailty
  performance claim；本阶段只证明接口与算法约束落地，不提供衰弱分类性能结论。

## Process / 流程

1. Re-read merged specification §5.6 and §6.1 and compared every requirement with
   the existing self-contained effect-size implementation.
   重新逐条核对规格，确认旧实现虽然自足但缺少 attention 前的 patch/downsample，
   且 discovery、物理时间与 outer-fold 身份尚未成为必填契约。
2. Preserved the explicitly named local `effect_size_shapelets_v1` discovery
   method and rejected PISD/unknown values instead of falling back.
   保留具名效应量发现方法；PISD 或未知方法均关闭失败，不发生静默替换。
3. Bound every fitted bank to sorted outer-train participant IDs, a SHA-256 roster
   hash, repeat/fold indices, input sampling rate, and shapelet length in samples
   and seconds.
   每个拟合库绑定 outer-train 名单、名单哈希、repeat/fold、采样率以及样本/秒双尺度。
4. Replaced raw/local encoder use before classification with non-overlapping Conv1d
   patch embedding, deterministic positional encoding, mask-aware Transformer
   attention and mask-aware patch pooling. `patch_size_samples < 2` is rejected.
   分类前路线改为非重叠 patch embedding→位置编码→掩码 Transformer→掩码池化；
   从结构上拒绝 400 Hz 原始采样点直接作为通用注意力 token。
5. Kept trainable shapelet-distance features as a parallel experimental branch and
   applied the same validity mask before both patch and shapelet computations.
   保留可训练 shapelet 距离实验分支，并在两个分支前统一清除无效补齐值。
6. Tightened the factory: discovery method, input sampling rate, outer repeat/fold
   and outer-train roster hash must all match the fitted bank.
   工厂要求上述身份与拟合库逐项一致，并验证完整 mapping 恢复路径。
7. Regenerated all cards from the unique generator; only ShapeFormer semantic
   content changed.
   通过唯一生成器更新模型卡；语义变化仅涉及 ShapeFormer 身份与限制说明。

## Algorithm / 算法

For input `x[B,C,T]`, the reference experimental route uses non-overlapping patches
of `P>=2` samples. A Conv1d patch projection produces
`N=floor(T/P)` tokens, sinusoidal position encodings preserve order, and generic
self-attention receives a key-padding mask constructed from fully valid input
patches. Masked mean pooling yields one patch embedding. In parallel, each fitted
shapelet computes the negative minimum squared distance over fully valid sliding
windows. The two file/window embeddings are concatenated before the classifier.

对输入 `x[B,C,T]`，先以 `P>=2` 做非重叠 patch 投影，得到
`N=floor(T/P)` 个 token；确定性位置编码保留时序，完整有效 patch 构成注意力
padding mask，最后做掩码均值池化。并行 shapelet 分支只在完整有效滑窗上计算
负最小平方距离；二者拼接后进入分类器。该结构明确不是 raw sample-token attention。

## Results / 结果

- Focused model suite: **19/19 passed**, zero failures/errors/skips.
- Full V1 CPU suite at this checkpoint: **149/149 passed**, zero
  failures/errors/skips.
- Tested boundaries: patch construction, output shape, invalid-tail invariance,
  explicit discovery selection, PISD rejection, physical-time equality,
  outer repeat/fold mismatch, roster-hash mismatch, mapping restoration, generated
  card identity, and unchanged CompactCNN/Inception parameter snapshots.
- No independent test or corrected 5×5 scientific benchmark was run in this phase;
  `independent_test=false` remains unchanged.

## Review / 自审

- No external path/package was introduced; effect-size discovery remains local.
- No outer-held-out labels enter discovery or factory construction.
- Generic self-attention receives patch tokens only; `patch_size=1` fails closed.
- Invalid tail values were changed by four orders of magnitude in test and produced
  identical masked logits within `atol=1e-6, rtol=0`.
- Potential limitation: non-overlapping patch projection discards an incomplete
  terminal patch. This is deterministic and mask-safe but remains an experimental
  architectural choice for matched benchmark evaluation.
- Potential limitation / 局限：本地效应量发现并非 PISD；因此模型继续标记为
  `experimental_ineligible_for_parity_claim`，不得据此宣称原方法复现。
