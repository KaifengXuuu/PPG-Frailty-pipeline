# NOTES

状态：draft  
来源：用户评价、`_agent/PROJECT_HANDOFF.md`、代码结构检查  
最后手动更新时间：2026-06-23

## 2026-06-23：关键观察与风险

- 类型：用户偏好
- 内容摘要：用户优先关注成功率、可解释性、泛化能力和持续迭代能力；重要实验需要 scorecard、图表、分层评估和可复现输出。
- 涉及文件/模块：全部实验模块
- 可能下一步：所有主线实验输出应至少包含 summary CSV、confusion matrix、learning curve、scorecard 或 markdown report。

## 2026-06-23：dynamic denoising 失败原因

- 类型：观察/风险
- 涉及文件/模块：`pttppg_denoiser_hybrid_*`, `pttppg_stage2_denoiser.py`
- 内容摘要：Denoiser A 接近复制 raw PPG；Denoiser B 出现双倍伪峰、峰谷错位和非生理形态；static/sit 段表现好不能证明 motion 段去噪有效。
- 证据或线索：`PROJECT_HANDOFF.md` 第 7.1 节和用户明确评价。
- 待验证问题：是否仍值得把 coarse-denoised dynamic signal 作为 frailty classifier 探索输入。
- 可能下一步：仅作为 high-risk exploratory input，不作为可靠去噪模块。

## 2026-06-23：frailty3 当前风险

- 类型：风险
- 涉及文件/模块：`frailty_3class_classifier.py`, `analyze_sweep.py`, `frailty_3class_holdout_eval.py`
- 内容摘要：当前主要瓶颈是 Pre-Frail 与 Robust/Non-Frail 混淆；subject 数少导致 fold/repeat 方差大；overlap windows 会增加样本相关性。
- 可能下一步：按 subject、role、静/动态状态、HRV、窗口质量分层分析错误。

## 2026-06-23：数据规范风险

- 类型：待确认
- 涉及文件/模块：`funcs.py`, `frailty_3class_classifier.py`, `ppg_analyse4_calib.ipynb`
- 内容摘要：采样率、时间戳、重采样策略必须明确，否则会影响 peak timing、HRV 和 window 模型。
- 可能下一步：建立单独数据规范记录，注明原始采样率、有效时间轴、重采样规则。
