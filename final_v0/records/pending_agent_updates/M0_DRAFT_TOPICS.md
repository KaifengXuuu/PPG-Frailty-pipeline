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

