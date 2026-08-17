# M2 数据 Manifest、阶段映射与评估协议

## 当前结论

M2 冻结数据身份、文件/受试者清单、经用户确认的阶段语义、外部同步 ECG/PPG/IMU 证据、协议命名和 Frailty3 双 fold 注册表。这里不训练模型，也不产生性能结论。

唯一未来主协议为：原始 400 Hz Frailty3、subject-level 5-fold、5 repeats、seeds `42, 10042, 20042, 30042, 40042`、fixed epoch、no early stopping、仅训练完成后计算 OOF validation。历史 scikit-learn 1.4.2 shuffle 映射错误的 SGKF membership 只保留用于复现，所有候选必须在修正且类别均衡的注册表上统一重跑。

## 文件导航

| 文件/目录 | 内容 |
|---|---|
| `00_CURRENT_STATUS.md` | M2 当前权威状态、已完成/未完成边界 |
| `01_DATASET_MANIFEST_AND_PROVENANCE.md` | Frailty3 文件/受试者 manifest、数据版本与异常证据 |
| `02_STAGE_ROLE_MAPPING.md` | B/R/S/W 确认语义、部分时序与明确未知项 |
| `03_DUAL_FOLD_REGISTRY_AND_MAIN_PROTOCOL.md` | 历史复现与未来主协议双注册表、SGKF 修正和防泄漏规则 |
| `04_EXTERNAL_SYNCHRONIZED_DATA_MANIFEST.md` | PTT、Simultaneous、iAMwell、MIMIC-PERform、VitalDB 的资格与限制 |
| `05_RESULT_PROVENANCE_AND_NAMING_CONTRACT.md` | 结果最小溯源字段和 `oof_validation_*` 命名合同 |
| `schemas/` | 数据、fold 和结果溯源的机器合同 |
| `registries/` | 阶段、协议和外部数据集注册表 |
| `manifests/` | 生成的逐受试者、逐文件、外部数据集和逐记录清单 |
| `splits/` | 历史复现与未来主协议的物化 membership |
| `tools/` | 只读源数据、只写本 M2 包的双语生成/验证工具 |
| `M2_BUILD_REPORT.json` | 全字节哈希、数值结构扫描和生成摘要 |
| `M2_CONTRACT_VERIFICATION.json` | M2 自动验收结果 |
| `M2_PACKAGE_TREE.md` | 包内逐文件 SHA-256 与说明 |

## 不可越过的边界

- `PPG_Testing_05_01_2026/` 与 `physionet.org/` 只读；生成器具有目标路径保护。
- R1–R4、S1/S2、W1/W2 的编号含义和精确次序未确认，必须保持 `unverified`。
- “S/W 在 Relax 前”只建立部分顺序，不推断 S 与 W 之间或各编号之间的总顺序。
- Frailty3 没有同步 ECG/人工 peak ground truth；一致性指标不得表述为 HR/PPI accuracy。
- 没有独立测试集时不得使用 `test_*` 字段或“test performance”表述。
