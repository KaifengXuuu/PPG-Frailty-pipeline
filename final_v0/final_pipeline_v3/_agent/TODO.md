# Final Pipeline V2 TODO / V2 待办

## 当前实施批次：完整修复但不运行科学 comparison

- [ ] P0：闭合 manifest/fold/raw-source 字节身份与 V2 build-data。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：正式 loader 核对冻结 manifest/fold SHA、每条 source SHA、header、n_samples、
    units；产物记录 observed hashes；禁止 runtime split 重算。

- [ ] P0：修复 quality diagnostics-only 与 recording QC。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：只保存 raw component/value/validity/reason；不得生成未监督融合分数、drop、
    weight、route 或 prediction effect。设备依赖 QC 保持 deferred。

- [ ] P0：实现 formal 11-channel motion 数据与模型链。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：8 s@400 Hz/hop2；单位转换；calibrated roll–pitch EKF；九个 IMU channels
    train-only robust scale；单次 SGKF5 seed42；删除 role-label oracle；严格 evidence gate。
  - 边界：只实现和做合成/单作业 smoke，不运行29人CV或PTT。

- [ ] P0：实现 PTT distal 适配与同步500→400重采样合同。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：record/source/mapping/resampling/schema hash；ECG reference evaluation接口；
    不运行正式外测。

- [ ] P0：实现 channel-specific variable-length OSD/PISD ShapeFormer。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：V2-029a/b；无 fixed length/stride；explicit failure；内存有界的等价距离；
    EffectSize与multichannel路线只能是具名ablation。

- [ ] P0：使13候选、两种ensemble和四种representation具备formal config/runner入口。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：所有超参数显式；Line A默认，Line B可选但不自动运行；7/15、filter、LPF、
    fixed-sample等均为具名可构造路线。

- [ ] P0：闭合 typed OOF、统计归档和不可覆盖产物。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：固定Arrow schema/readback；split/training seed分离；member/average语义；BA与
    macro-F1 bootstrap/LCB/permutation/Holm；artifact SHA/bytes index。

- [ ] P0：升级 V2 bundle、final-refit与winner ONNX interfaces。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 边界：不训练final model、不导出winner ONNX；实现严格版本/provenance/golden gates。

- [ ] P0：建立 V2 dependency gate、acceptance与CPU-CI。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：按config需要的profile fail closed；scientific slow tasks保持显式opt-in；safe
    gate不得导入comparison/ablation。

- [ ] P1：完成 Aura/nolds 官方兼容性调研。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 边界：不更改conda ml；不静默降级；保存官方来源与isolated smoke结果。

- [ ] P1：清理 V1 current evidence、旧model cards与tracking漂移。
  - 日期：2026-08-16
  - 状态：`in_progress`
  - 要求：V1只保留historical身份；V2重新生成current证据。

- [ ] 验证：仅运行非科学全量回归和简单真实 reduced smoke。
  - 日期：2026-08-16
  - 状态：`pending`
  - 禁止：正式ablation、完整5×5、PTT benchmark、winner/final训练。

- [ ] 最终：独立只读conformance审查、diff预览与自审。
  - 日期：2026-08-16
  - 状态：`pending`

