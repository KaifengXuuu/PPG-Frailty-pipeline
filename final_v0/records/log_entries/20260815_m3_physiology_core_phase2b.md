# 2026-08-15 M3 Peak、PPI、HR 与 PRV 公共实现

- 新增 corrected_v1 双极性 peak detector，固定 10 秒窗口、5 秒 hop 和 0.15 秒事件合并。
- PPI 固定为 0.30–2.00 秒；无效 PPI 不删除源峰，raw/valid/corrected NNI 分列。
- HR 门固定为至少 8 秒、5 个峰和 4 个有效 PPI。
- PPG-derived variability 使用 PRV 名称；60 秒 time-domain、120/300 秒 frequency tiers。
- RED/IR 分别检测，以 SQI 选主通道，平局选 RED，禁止 consensus 移动峰。
- 新增 physiology/reason-code registries 和公共后端算法图。
- 状态：算法已落盘，尚待 fixtures、真实片段审计和 validator。

