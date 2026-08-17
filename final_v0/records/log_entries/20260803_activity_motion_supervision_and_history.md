## 2026-08-03 — Activity/Motion 监督确认与早期三分类历史追溯

- 操作 / Action：把用户确认的 B/R 静态、S/W 动态监督语义写入 M0，并追溯早期多类模型、结果与混淆矩阵。
- 数据核验 / Data audit：逐字节读取29人261份CSV；确认两个数据目录、统一8列结构、每角色29份、角色持续时间与全部活动后恢复顺序。
- 历史结论 / History：找到三分类 SVM 数据和649个 SVM 权重；找到“Rest好、Walk与Sit/Stand混淆”记录；未找到三分类 CNN 或 3×3 confusion matrix。
- 当前模型 / Current model：核验 PTT/SIM A/B CNN 为直接二分类；balanced_v2 external SIM 中 Light CNN BA `.7802`、F1 `.7634`，与内部满分共同显示域偏移。
- 新增 / Added：专题文档09、算法图08、决策 `M0-MOT-001`、两份机器审计JSON与既有二分类 confusion/result 证据副本。
- 状态 / Status：监督阻塞已解除；Motion-29 适配、nested 5-fold、阈值、SQI融合、恢复特征和frailty比较仍未实现。
- 边界 / Boundary：未训练、未反序列化pickle/PT/ONNX、未联网、未修改final_v0外文件、未写入 `_agent`。
- 同步 / Synchronization：随后生成追加式 v3 manifest/verification/tree，并刷新算法索引、工作记录和总文件树；追踪更新不递归记日志。
