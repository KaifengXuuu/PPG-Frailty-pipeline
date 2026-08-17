# 待录入 `_agent` 内容草稿簿 / Pending `_agent` Update Drafts

> 本文件由 `tools/sync_tracking_docs.py` 从 `records/pending_agent_updates/*.md`
> 自动重建。除非用户明确要求草拟或展示，否则不得写入 `_agent/`。

## 强制规则 / Mandatory rules

- 候选内容默认为 `draft`，必须注明目标文档、来源、证据和待确认项。
- 用户明确要求后才整理成逐文档可审核正文。
- 只有用户明确回复“确认录入”或“同意录入”后才可写入 `_agent/`。
- 本会话当前写入边界仍限制为 `final_v0/`。

# Baseline + M0 待录入主题（内部草稿）

- 状态：`draft; do_not_write_without_user_request`
- 候选目标：`_agent/MODULES.md`、`PROJECT_STRUCTURE.md`、`NOTES.md`、`PROJECT_HANDOFF.md`、`TODO.md`、`CHANGELOG.md`。
- 主题：workspace扫描范围与证据、52份代码I/O与lineage、M0方法registry、失败/代理/空结果、论文claim边界、full-waveform不恢复、P02候选、未来人工决策门。
- 必须保留：M0任何方法均未达到strictly_validated；v7.2负holdout；v8/Stage2空目录；hybrid无clean/holdout；P01目标/外部泛化问题；legacy v8 IMU活动主导。
- 写入前：按`_agent/WRITE_RULES.md`拆分为逐目标文档草案并向用户展示；只有用户明确“确认录入/同意录入”才写入。

---

# M0 Activity/Motion supervision private draft

> Private staging note only. Do not copy to `_agent` or display drafted content until the user explicitly asks for a draft and approves the target text.

## Candidate decision

- Decision `M0-MOT-001` confirms the 29-subject target as activity/motion state.
- Map B and R1–R4 to static; map S1–S2 stand-and-sit and W1–W2 walking to motion.
- Preserve the full role and acquisition sequence for recovery and frailty-feature exploration.
- Reuse the PTT detector architecture/preprocessing concept, but retrain and recalibrate within the local device domain.

## Candidate evidence correction

- Verified early assets are three-class/multiclass SVM datasets, SVM training code, and 649 SVM model files.
- The saved qualitative result says Rest was recognized well while Walking and Sitting/Standing were mixed.
- No verifiable three-class motion CNN source, checkpoint, numeric result, or 3×3 confusion matrix was found.
- Existing A/B CNN motion results are direct sit-versus-walk/run binary experiments and must not be described as a collapsed three-class CNN.

## Candidate TODO impact

1. Add a frozen StageManifest for 29 subjects and all nine roles.
2. Implement subject-disjoint nested 5-fold detector transfer/from-scratch comparisons and fold-local threshold calibration.
3. Produce strict OOF `p_active`, integrate it softly into SQI-v2, and retain hard rejection only as an ablation.
4. Pair each active bout with the immediately following Rk and build route-specific active/recovery HR/PPI features.
5. Keep all route and frailty selection nested within subject folds and preserve coverage/failure metadata.

## Evidence boundaries

- Activity labels are not window-level optical-artifact truth.
- File modification time supports the active→R sequence and is provisional until a formal acquisition manifest is available.
- Existing label-table HRrecovery/maxHR fields have unknown generation provenance and are not accepted as supervision truth.
- No training or new benchmark was run during this documentation step.

---

## M0 历史 motion-processing 审计候选更新

- 状态 / Status：`draft`
- 来源 / Source：完整代码读取、输入头部扫描、输出文本 EOF 扫描、历史记录及结果交叉核验。
- 目标文档 / Candidate targets：`_agent/MODULES.md`、`_agent/TODO.md`、`_agent/NOTES.md`、`_agent/CHANGELOG.md`、`_agent/docs/decision-log.md`。
- 当前处理 / Current handling：仅保留候选主题；用户明确要求后再生成逐文档可审核正文。
- 候选主题 / Candidate topics：
  - v7/v7.2/v7.4/v8/Stage-2/hybrid/heartbeat 的真实实现与验证状态；
  - 已确认的 leakage、运行阻断、ECG–PPG delay 和输出 schema 问题；
  - 可保留的 motion detector/ONNX 经验与不得恢复 full-waveform reconstruction 的依据；
  - M0 完成状态以及 M1/M3/M4 的依赖与风险。

---

## 草稿主题：M0 五类方法与候选路线扩展

- 状态：`draft_not_for_agent_write`
- 来源：2026-08-03 本地代码/输出扩展审计
- 待用户要求后整理：
  - 自适应滤波只保留为风险基线，记录真实 HR 被吞的假设破坏问题；
  - DWT/CEEMD-lite 的准确命名及 CWT/WPT/EEMD/VMD/SSA 未实现状态；
  - 谱域抑制、候选峰与 Viterbi/Kalman/Particle 的优先实现路线；
  - 双波长 PCA/FastICA/STFT-NMF 的数据可用、实现缺失结论；
  - SQI 现有轻微正信号、实现缺口、统一质量/拒绝层要求；
  - M0 扩展仍须用户确认，未进入 M1。
- 禁止动作：用户未明确要求前，不展示为 `_agent` 拟录入正文，不写入 `_agent/`。

---

# M0 MAdenoiser confirmed-route private draft

> Private staging note only. Do not copy to _agent or display its drafted content until the user explicitly requests a draft and approves the target text.

## Candidate decision entry

- Decision ID: M0-MAD-001
- Status: confirmed_route_implementation_not_started
- Scope remains M0 extension; M1 is not authorized.
- Writes remain restricted to final_v0; root code, data, output, AGENTS.md, and _agent remain read-only.
- No network use or TROIKA/JOSS source retrieval has been authorized.

## Candidate TODO impact

1. Implement and validate SQI-v2 components: skewness, kurtosis, autocorrelation periodicity, template correlation, normalized spectral entropy, complete IBI plausibility, RED/IR agreement, interpretable flags, and fold-local calibration.
2. Define the 29-subject Motion supervision target, then implement subject-grouped nested threshold/CV and integrate fold-local motion probability into SQI before classifier standardization.
3. Implement four comparable route front ends with one common backend:
   - spectral_track_sqi
   - dual_ppg_bss_sqi
   - nonstationary_sqi
   - adaptive_sqi
4. Evaluate preliminary HR/PPI on pulse-transit-time-ppg using ECG R-peaks as HR/RRI reference; use train-fit ECG→PPG delay for absolute PPG event timing.
5. Produce OOF frailty feature blocks for motion HR/PPI, compare identical subject folds and seeds, and select by nested-CV subject BA.

## Candidate evidence boundaries

- The 29-subject frailty cohort has 261 raw role files and no current window-level optical-artifact truth.
- Existing PTT Motion A/B is a 22-subject activity-state experiment, not a 29-subject artifact CV.
- Current SQI has partial components and leakage/unit risks; route confirmation does not validate it.
- PTT peaks are ECG R-peaks, not PPG pulse peaks.
- Initial frailty comparison space is baseline plus four routes × HR-only/PPI-only/HR+PPI.
- Direct winner selection on the same 5-fold result is development selection, not independent final performance.

## Open decision to request before implementation

Define the 29-subject Motion target as one of:

1. manually annotated window-level optical artifact;
2. B/R/S/W activity/motion proxy, named only as activity/motion state;
3. independently defined peak/HR unavailability.

This choice changes label construction, threshold meaning, validation metrics, and paper claims.

---

# 待录入 `_agent` 草稿：M1 架构与移动平台合同

> 私有待审草稿。除非用户明确要求，不展示正文、不写入 `_agent`。

建议目标文档：`TODO.md`、`MODULES.md`、`ROADMAP.md`、`docs/decision-log.md`、`CHANGELOG.md`、`PROJECT_STRUCTURE.md`。

待记录主题：M1 端到端模块顺序、SignalBatch/PipelineResult 合同、SQI诊断与质量动作互斥、训练/部署隔离、允许依赖、三档中心平台、provisional资源预算、M2/M3/M4/M9后续门。

状态边界：用户确认产品形态和依赖；架构合同已验证；具体处理器采购、连接链路、模块实现、ONNX runtime smoke 和真实硬件指标仍待完成。

---

# 私有待录入草稿：M1 V2

- 仅供未来用户明确要求草拟 `_agent` 更新时使用；当前不得写入 `_agent`，最终报告也不展示正文。
- 待录入主题：M1 V2 有界流式合同、preprocessing execution mode、窗口坐标/coverage、单一 action owner、完整 artifact hashes、CPU fallback、当前未实现/未 benchmark 边界。

---

# 私有待审草稿：M1 V3 质量路由 / Private pending draft

## 写入边界

- 本文件只保存未来可能写入 `_agent` 的候选内容。
- 未收到用户明确要求时，不展示本文件内容，不写入 `_agent`。
- 真正草拟时须重新读取 `_agent/WRITE_RULES.md` 并按职责拆分。

## 待同步事实

- M1 当前 quality-routing authority 为 `m1.architecture.v3`；V3 只取代 V1/V2 冲突的质量路由语义。
- SQI 必做，Motion detector 可选；两者 join 后才路由，denoiser 只能后置。
- high/non-motion 绕过去噪进入共享 feature extractor。
- low 或 motion 由 run/session 级人工配置互斥选择 drop 或 denoise-then-features。
- Motion 与 signal quality 为正交轴；B/R vs S/W 是 activity supervision。
- invalid/unrecoverable 强制 drop；module failure fail-closed，无 stale result/raw fallback。
- V3 CURRENT 合同验证与 24 项路由语义 fixture 已通过；模型/ONNX/硬件仍未执行。
- M2–M9 必须同步 coverage/no-result、factorial benchmark、FeatureBlock compatibility 与移动 worst-case 路径。

## 建议目标文档

- `_agent/docs/decision-log.md`：记录 M1-ARCH-003。
- `_agent/PROJECT_HANDOFF.md`：更新当前 M1 authority 和暂停点。
- `_agent/TODO.md` / `_agent/ROADMAP.md`：仅在用户要求草拟并确认录入后，修正后续 M4/M8 路由措辞。
- `_agent/CHANGELOG.md`：记录 V3 合同与验证结论。

---

## M2 数据 manifest、阶段语义与双注册表候选记录

- 状态：`draft_not_authorized_for_agent_write`
- 目标文档候选：`MODULES.md`、`TODO.md`、`PROJECT_STRUCTURE.md`、`docs/decision-log.md`、`CHANGELOG.md`、`NOTES.md`
- 来源：用户 2026-08-15 确认、M2 只读代码/数据/结果审计、M2 机器验证。
- 规则：仅在用户要求草拟时展开逐文档正文；用户明确“确认录入/同意录入”前不得写入 `_agent/`。
- 核心候选主题：Frailty3 数据版本与 manifest、B/R/S/W 部分阶段语义、历史/未来 SGKF 双注册表、5×5 fixed-epoch OOF 主协议、外部数据资格和未决元数据门、`oof_validation_*` 命名合同。

---

# M3 统一预处理与信号算法待录入主题

- 状态：draft；仅在用户要求草拟 `_agent` 更新时整理，不直接写入 `_agent`。
- 候选主题：M3 冻结的 400 Hz profiles、corrected/legacy 边界、无预校准 EKF 主路线、
  LPF 对照、异常门控、fold-only scaling、peak/PPI/HRV 公共 API 与验收结果。
- 证据位置：`final_v0/M3_unified_preprocessing_and_signal_algorithms/`。
- 当前进展：M3 公共实现已完成 profile-bound PPG、stateful causal IMU、train-fold scaler、
  corrected peak/PPI/HR/PRV、PTT train-only delay evaluator；38 项 reference tests 暂时全部通过。
- 已形成证据：合成真值 EKF/LPF 对照、261 文件 Frailty3 完整性审计和 B/R/S/W 角色级代理统计。
- 待完成：机器 schemas/registries、正式 M3 validator、完整文档/算法图和全局回归均通过后再冻结结论。
