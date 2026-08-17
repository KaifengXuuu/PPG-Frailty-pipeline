# M0 当前状态 v3：Activity/Motion 监督已确认

> 本文件是追加式 current-status 入口，不覆盖 v1/v2 历史报告。若旧文档仍显示 `blocked_on_target_semantics`，以本文件、`09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md` 和决策 `M0-MOT-001` 为当前状态。

## 当前结论

1. 监督已确认：`B/R→static`，`S/W→motion`；S 为 stand-and-sit 往复，W 为 walking。
2. 目标名称是 activity/motion state，不是 optical-artifact ground truth。
3. 29 人来自 `StudyData` 21 人和 `TestDataYoungers` 8 人，共261份九角色 CSV。
4. 所有人的协议结构都是 `B→active→R1→active→R2→active→R3→active→R4`；Rk 绑定其前一活动。
5. 早期可核验三分类资产是 SVM；未找到三分类 CNN、三输出 head、CNN 权重或 3×3 confusion matrix。
6. 当前 PTT Light CNN 是直接二分类；其 external SIM BA `.7802`、F1 `.7634`，说明可复用结构但必须在本地设备域重训和重校准。
7. 本轮只完成证据、算法合同与归档；Motion-29 代码、CV、阈值、SQI融合和恢复特征仍未实现。

## 当前导航

| 文件 | 用途 |
|---|---|
| `09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md` | 完整数据审计、历史追溯、迁移重训、SQI和恢复特征合同 |
| `evidence/MOTION29_DATA_AUDIT.json` | 29人、角色、时长、监督与顺序的机器记录 |
| `evidence/EARLY_MULTICLASS_SEARCH_AUDIT.json` | 三分类SVM存在、三分类CNN/3×3 CM未找到的机器记录 |
| `evidence/current_binary_detector_balanced_v2/` | PTT/SIM A/B主运行摘要与混淆矩阵副本 |
| `evidence/current_binary_detector_smoke/` | Smoke运行与阈值/域偏移稳定性副本 |
| `snapshots/algorithm_diagrams/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md` | 本轮算法图快照 |
| `M0_SOURCE_SNAPSHOT_MANIFEST_V3.json` | v3 源—快照哈希清单 |
| `M0_PACKAGE_VERIFICATION_V3.json` | v3 完整性与源漂移验证 |
| `10_M0_PACKAGE_TREE_V3.md` | v3 包全部文件的树、字节、SHA-256与说明 |

## 与旧文档的关系

- `README.md` 与 `07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md` 中的“监督语义待决定”是 v2 时间点的历史状态。
- `M0-MOT-001` 已解除该阻塞，但没有自动授权进入下一 TODO 实施。
- v1/v2 manifest、verification、tree 与 snapshots 保留原字节，不以当前结论覆盖。
