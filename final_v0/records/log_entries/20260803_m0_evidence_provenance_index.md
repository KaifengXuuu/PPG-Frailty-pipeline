## 2026-08-03 — M0 证据与来源索引 / M0 evidence and provenance index

- 操作 / Action：新增 M0 源码、输入、输出、机器验证和快照来源链索引。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/05_EVIDENCE_INDEX_AND_PROVENANCE.md`。
- 内容 / Content：列出关键函数行段、真实数据根/header、历史结果目录/关键文本文件、canonical generated manifests 和快照规则。
- 完整性策略 / Integrity strategy：小型人类报告与验证摘要进入快照；大型逐文件 manifest 保留单一 canonical 原位，通过路径与 SHA manifest 引用，避免两套证据漂移。
- 隐私 / Privacy：`.env` 值继续不写入；历史外部盘路径仅作不可移植运行证据。
- 文献边界 / Literature boundary：本轮未联网；TROIKA/JOSS 不提供伪引用，精确论文复现需用户另行授权。
- 验证 / Verification：待快照构建工具写入后执行 source/snapshot 字节一致性和 package tree 检查。
