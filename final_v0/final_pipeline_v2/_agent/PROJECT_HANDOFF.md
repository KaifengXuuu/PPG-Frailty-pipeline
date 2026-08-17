# Final Pipeline V2 handoff / V2 项目交接

## 当前状态（2026-08-16）

- 状态：`repair_in_progress`
- 用户目标：把 V2 修复为严谨、可作为论文方法与结果基础的 final pipeline。
- 当前执行边界：只实施代码、合同、依赖审计、non-scientific tests 和简单 reduced
  smoke；不运行正式 ablation、完整5×5或PTT benchmark。
- V1 边界：V1 是过渡/历史版本，不构成 V2 current evidence。

## 已确认的核心默认

- Frailty默认输入 roles：B/R；quality mode：off；balance：Line A equal files。
- Line B equal role families、SQI diagnostics、artifact reducers、epoch7/15、filter0.5–5、
  LPF gravity、fixed-sample resampling 和ensemble均为显式 comparison routes，不自动运行。
- ShapeFormer literature reference：channel-specific variable-length OSD/PISD；EffectSize
  fixed128/stride64与multichannel-PIP只能具名ablation。
- Motion formal input：11 channels、8s@400Hz/hop2、单位转换、calibrated roll–pitch EKF、
  nine-channel outer-train robust scaling。
- 单模型 final seed 42；五成员 InceptionTime seeds为42/10042/20042/30042/40042。

## 当前禁止声明

- 不得宣称13候选均完成正式5×5 benchmark。
- 不得宣称已有 reducer winner、SQI threshold、motion override evidence 或 PTT test结果。
- 不得宣称 Aura最新版当前兼容成功。
- 不得把V1 acceptance/current artifacts描述为V2证据。
- 不得把合成forward、contract test或safe smoke描述为科学性能结果。

## Deferred gates

V2-006、V2-009a/b/c、V2-010 activation、V2-012 winner、V2-026、V2-027，以及
正式ablation/完整5×5/PTT benchmark保持deferred；详见`docs/decision-log.md`。

## 下一交接条件

修复完成后必须更新：实际变更清单、测试报告、依赖lock状态、remaining gaps、diff
自审与独立conformance结果。未达到这些条件前，状态不得改为`release_ready`。

