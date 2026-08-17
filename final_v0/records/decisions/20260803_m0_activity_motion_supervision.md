# M0-MOT-001 — 29-subject Activity/Motion 监督与时序特征决定

- 日期 / Date：2026-08-03
- 状态 / Status：`confirmed`
- 决策者 / Decider：用户
- 范围 / Scope：M0 后续路线语义；不等于批准开始 M1 或完成模型实现

## 决定 / Decision

1. 29-subject Motion detector 的监督目标定义为 activity/motion state。
2. `B` baseline 与 `R1–R4` relax/recovery 映射为 static；`S1–S2` stand-and-sit 往复与 `W1–W2` walking 映射为 motion。
3. 沿用 pulse-transit-time-ppg 检测器的结构/预处理思想，在本地29人设备域重训，以处理设备误差和环境噪声；旧权重、阈值与内部满分不得直接当作本地结果。
4. 保留 B/R/S/W 阶段语义和实际顺序，探索运动 HR 上下限、活动响应、恢复速度、重复性和时序关系的 frailty 特征。
5. 主任务保持 static-vs-motion 二分类；B/R/S/W 多阶段分类只可作为预注册辅助头或探索路线。

## 历史证据纠偏 / Historical evidence correction

- 已找到 Rest/Sit&Stand/Walk 三分类 SVM 数据、训练实现与649个 SVM 权重文件。
- 已找到历史文字记录：Rest 较好，Walking 与 Sitting/Standing 混淆。
- 未找到三分类 CNN 源码、权重、数值报告或 3×3 confusion matrix；不得在论文中把已核验 SVM 写成 CNN。
- 当前可复核 A/B CNN 从一开始就是 `sit=0, walk/run=1` 二分类。

## 验证边界 / Validation boundaries

- group 必须为 subject；threshold、概率校准、early stopping 与模型选择只在 inner training data 完成。
- 所有29人只通过 OOF 推理生成 `p_active`，再融入 SQI 和 frailty 特征。
- activity label 不是 optical-artifact truth。
- `Rk` 必须与时间顺序中前一活动配对，不可按 S/W 编号假设。
- 恢复特征与最高 BA 组合尚未实现或测试，当前仅为 confirmed specification。
