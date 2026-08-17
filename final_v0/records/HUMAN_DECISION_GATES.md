# 人工决策门 / Human Decision Gates

- 状态 / Status：`no_current_M3_blocker_future_gates_remain`
- 规则 / Rule：发现会改变研究主线、论文口径、数据cohort或依赖范围的选择时停止；这里只记录选项与影响，不代替用户决定。

## 当前必须暂停确认 / Current pause point

- M3：无。D1–D8 已由用户确认并记录为 `M3-PREPROCESS-001`。
- 进入 M4 前仍按逐 TODO 验收规则等待用户确认。

### D-000 — M0 验收与是否进入 M1

- 事实：M0 registry、crosswalk、risk、paper evidence与算法图已形成；历史full-waveform reconstruction没有严格有效路线。
- 建议默认：接受 M0；维持 `failed_or_deprecated`；进入 M1，但不恢复任何历史完整波形重建。
- 用户决定：`accepted`；M0 结论保留，已进入并完成 M1/M2 合同阶段。

## 已发现、在对应 TODO 前必须决定 / Future blocking gates

### D-001 — Frailty3 论文主结果口径（最晚 M5 前）

- 选项A（建议）：以15-subject independent holdout BA `.5444`及CI为主结果；旧 `.737` 只作探索性/选择偏差案例。
- 选项B：旧sweep为主；这会与独立验证原则冲突，必须明确披露泄漏/选择偏差。
- 状态：`pending`。

### D-002 — 重复/不完整结果目录的归档口径（最晚 M1 前）

- 选项A（建议）：用 `complete / historical / aborted / empty` 四态登记，不删除原目录。
- 选项B：只保留人工指定的“最新”run用于final registry；其他仍保留但不进入论文证据。
- 状态：`pending`。

### D-003 — ShapeFormer-PISD 是否保留（最晚 M5 前）

- 选项A（建议）：若外部依赖不能离线固化，降级为历史候选，不进入final benchmark。
- 选项B：允许在 `final_v0` 实现/封装自包含CPU版本，再按同协议重跑。
- 状态：`pending`。

### D-004 — SVM 是否作为最终主线模型（最晚 M5 前）

- 选项A（建议）：先作为必须修复并重跑的baseline；只有新scorecard达标才升级为主线。
- 选项B：直接指定SVM为最终主线并优先修复；当前没有足够结果支持。
- 状态：`pending`。

### D-005 — ASA 决策规则（最晚 M8 前）

- 选项A（建议）：论文采用OOF threshold版本，完整披露BA/macro-F1/每类recall与argmax collapse。
- 选项B：采用argmax；会产生ASA2不预测问题。
- 状态：`pending`。

### D-006 — 28/29-subject缓存cohort（最晚 M2/M6 前）

- 选项A（建议）：以可由manifest重建的29-subject版本为当前cohort，28-subject缓存标记legacy。
- 选项B：继续兼容两套cohort；必须分别报告且禁止混合split。
- 状态：`closed`。
- 用户决定：采用双注册表路线 C；29-subject corrected subject-level SGKF 作为未来唯一主协议，历史 SGKF 仅复现；5 repeats/seeds 已冻结。

### D-008 — M3 统一预处理与 IMU 重力路线

- 状态：`closed`。
- 用户决定：D1–D4、D6–D8 按推荐冻结；D5 采用无预校准 quaternion error-state EKF 为主，0.3 Hz LPF 为独立对照。
- 影响：全部 future-active 模块必须引用 M3 公共实现/registry；root 实现只可历史复现。详见 `M3-PREPROCESS-001`。

### D-007 — 是否受控读取二进制模型/缓存内部元数据（按需）

- 选项A（建议）：优先文本/代码证据；只有任务确需且没有安全替代时，再单项请求授权。
- 选项B：授权批量受控反序列化；会扩大安全和复现审计范围。
- 状态：`pending`；当前不阻塞 M0。

## 决策记录格式 / Decision recording format

用户确认后，新建不可变决策片段，至少记录：`decision_id`、选择、理由、影响TODO、被排除选项、日期；随后由同步工具更新工作日志与本登记表的派生视图。任何 `_agent` 更新仍必须另行由用户要求并确认。
