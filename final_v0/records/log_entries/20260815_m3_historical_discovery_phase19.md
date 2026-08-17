# M3 historical preprocessing discovery phase 19 / M3 历史预处理全量发现第 19 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_evidence_rebuilt
- 流程 / Process：按 M3 TODO 重扫 final_v0 外全部 Python，以滤波、缩放、重采样、重力和 Aboy/peak 关键词的固定正则发现相关实现；逐文件读取并计算 bytes/SHA256。
- 算法 / Algorithm：UTF-8 bytewise 排序；root/active-candidate 与 Arc/archiv/Archive 分别标 historical_reproduction_only 和 historical_archive_reproduction_only；两类都不得成为 future-active。
- 结果 / Result：发现 35/35 个相关脚本、0 missing；其中 archive 17、root/candidate 18。evidence crosswalk 与 M3_BUILD_REPORT 已重建并登记完整 hash。
- 边界 / Boundary：该 evidence 是扫描快照；最终机器 authority 由 registries/historical_preprocessing_crosswalk_v1.json 定义并应覆盖同一 discovery roster。
