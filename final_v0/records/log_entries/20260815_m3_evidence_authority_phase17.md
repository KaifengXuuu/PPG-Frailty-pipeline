# M3 evidence authority phase 17 / M3 证据权威边界第 17 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_evidence_rebuilt
- 流程 / Process：为避免 evidence 与 registries 出现双权威，重定义历史 crosswalk evidence 为 byte-hashed audit snapshot，并改进核心证据重建的合并逻辑。
- 算法 / Algorithm：核心 builder 重算四类核心证据及历史快照，同时逐文件复核并保留独立生成的 261-record EKF/LPF proxy 与 legacy peak parity；build report 登记 producer SHA 和每项 bytes/SHA256。
- 结果 / Result：M3_BUILD_REPORT 为 pass，共登记 6 项证据；全数据 proxy 和 parity 未在重建中丢失。
- 边界 / Boundary：未来 registries/historical_preprocessing_crosswalk_v1.json 是机器 authority；evidence 同名文件只保留扫描时点和源码哈希。
