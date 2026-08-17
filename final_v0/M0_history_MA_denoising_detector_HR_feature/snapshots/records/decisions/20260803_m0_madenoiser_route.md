# M0-MAD-001 — MAdenoiser 后续路线决定

- 日期 / Date：2026-08-03
- 状态 / Status：`confirmed`
- 决策者 / Decider：用户
- 记录范围 / Scope：M0 后续技术路线，不等于批准进入 M1

## 决定 / Decision

1. 完善 SQI 的偏度、峰度、自相关、模板相关、归一化谱熵、完整 IBI 合理性和 RED/IR 一致性，并允许由严格消融决定最终进入 composite 的分量。
2. 在本地 29-subject frailty cohort 上完善 Motion detector 的阈值与 subject-grouped CV，将 OOF motion probability 融入 SQI。
3. 按 `spectral+SQI → dual-PPG BSS → nonstationary decomposition → adaptive filtering` 顺序实现四条共同后端路线；先在 PTT `peaks` 参考上评 HR/PPI，再在相同 seeds/5-fold 的 frailty feature matrix 中选择最高 BA 稳定组合。

## 影响 / Impact

- 影响 M1/M2 的架构、manifest 与 split registry。
- 直接定义 M4.1、M4.2、M4.3 的实现顺序。
- 定义 M6.2 的 PTT HR/PPI benchmark 和 M6.1/M7/M8 的 Frailty route-feature 比较。
- 最终移动端 M9 必须保存 SQI、motion detector、route、feature schema 与 locked config。

## 解释边界 / Interpretation boundaries

- 29 subjects 是 frailty cohort；现有 P02 结果来自 PTT/SIM，不是 Motion-29 CV。
- PTT `peaks` 是 ECG R-peaks；HR/RRI 可作参考，PPG pulse timing 必须校正 delay。
- 最高 BA 是配置选择规则；无偏性能必须来自 nested outer-test 或未来独立 cohort。
- Adaptive 允许作为第四路线实现，但必须经过真实 HR 保护门；失败时仍是风险对照。

## 仍待用户决定 / Open gate

29-subject 数据没有窗口级 artifact 真值。开始 Motion-29 实现前，必须确认监督目标是 optical artifact、B/R/S/W activity proxy，还是独立定义的 peak/HR 不可用性。
