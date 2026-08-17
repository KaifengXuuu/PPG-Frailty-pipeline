# Frailty3 数据 Manifest 与溯源

## 权威原始集合

- `PPG_Testing_05_01_2026/StudyData/`：21 名 older subjects，189 个 CSV。
- `PPG_Testing_05_01_2026/TestDataYoungers/`：8 名 young subjects，72 个 CSV。
- `PPG_Testing_05_01_2026/StudyData_frailtyScored/StudyData_V7_standard.csv`：当前代码实际使用的标签连接源。
- 每位受试者均有 `B,R1,R2,R3,R4,S1,S2,W1,W2` 九个文件；总计 29 subjects、261 files。

生成器对每个 CSV 完整读取字节、计算 SHA-256、验证精确八列 header、把全部数值 token 解析为有限浮点数，并记录样本数、400 Hz 推导时长、角色、类别、单位、warning 和 reference 可用性。机器结果见 `manifests/frailty3_file_manifest.csv` 与 `M2_BUILD_REPORT.json`。

## 类别合同

| class_id | class_name | subjects | 来源 |
|---:|---|---:|---|
| 0 | `pre_frail` | 9 | older 标签表 `FRAILTY-STATUS=2` |
| 1 | `robust_non_frail` | 12 | older 标签表 `FRAILTY-STATUS=3` |
| 2 | `young` | 8 | young cohort 规则；不是 FRAILTY-STATUS 的第三严重度 |

Young 类按 cohort 定义。标签表中若干 young ID 即使存在 status=3，也不得覆盖 cohort class；AB/EE 没有可连接标签行也不构成排除理由。

## 通道与单位

| 通道 | 原始单位状态 | 可确认信息 |
|---|---|---|
| RED, IR | `raw_device_counts` | 双通道原始数值；ADC 量纲、波长 nm、placement、polarity 未由第一方元数据确认 |
| AX, AY, AZ | `g_source_declared` | 第一方项目代码按 g 解释，M3 再冻结转换实现 |
| GX, GY, GZ | `degree_per_second_source_declared` | 第一方项目代码按 deg/s 解释，M3 再冻结转换实现 |

400 Hz 来自用户确认与现有代码默认；CSV 无 timestamp，不能由文件内部独立推导采样率。M2 必须同时保存 `sampling_rate_source=user_and_first_party_code`。

## 标签与源异常

- standard CSV 使用 `STE072`，与真实文件 `STE072_01_*` 一致；standard XLSX 与旧分号 CSV 使用 `STE02`，标记为冲突 legacy source，不用于连接。
- 标签源另有 BAE28、NRE29、PSR16、PSS22 四个无相应原始文件的 ID；不纳入 29-subject roster。
- 30 s 窗口会使全部 W、几乎全部 S 大量 padding；M2 只发 warning，不自动排除。M3/M4 必须显式选择短片段策略。
- Frailty3 无同步 ECG、人工 peak、HR 或 PPI ground truth；`reference_available=false`。
