# 外部同步 ECG/PPG/IMU 数据 Manifest

## 可直接作为 ECG 监督/运动验证的数据

### Pulse Transit Time PPG 1.1.0

- 22 subjects × sit/walk/run = 66 records；CSV container grid 500 Hz。
- 左食指 distal/proximal 各含多色 PPG；ECG、accelerometer、gyroscope 同步；`peaks`/`.atr` 为自动检测后人工复核 R peaks。
- placement 与 activity 可用；PPG hardware/update rate、container rate 和其他低频通道必须分字段保存。
- README 两处对 pleth red/IR 映射互相冲突，故 wavelength mapping=`unresolved_conflict`，双波长 BSS 暂停。
- README 本地文件整文件 SHA 失败且尾部被本项目文字污染；其余官方 SHA 清单文件全部通过。波形/annotation 可用，但 README 不能单独作为权威。
- accelerometer 声明单位与实值/代码推断冲突，保存 `unit_declared` 与 `unit_inferred`，M3 前不强制统一。

### Simultaneous Measurements 1.0.0

- 13 个完整可用 record；统一 generated grid 256 Hz，SOT/Pleth + SOT/EKG + FAROS 3-axis accelerometer；无 gyro。
- `.atr` 为人工修订 consensus beat annotations；`.aux` 为人工修订阶段 marker。
- 四阶段包含 standing rest、flat walk、standing 2-back、uphill walk；二元 activity 必须由 interval marker 归一，不能把整条 record 赋单一标签。
- 右前臂/腕部 SOT 主机连接右手指端 pleth（示意图证据，具体手指未知）；波长未知。
- `x001a` 缺 SOT/Pleth，不进入 PPG benchmark；整个源 SHA 清单通过。

## 条件使用或暂停的数据

| dataset | heartbeat | motion | BSS | 关键限制 |
|---|---|---|---|---|
| iAMwell local 15 | conditional pseudo-ECG peaks | no | no | 无 IMU；阶段表尚未接入，A12 有 ERROR，RECOVERY4 需受控映射 |
| MIMIC-PERform local | conditional pseudo-ECG peaks | no | no | 125 Hz、无 IMU；重复 record、partition 截断和逐记录 ECG lead 必须先冻结 |
| VitalDB online | hold | no | no | 本地无 waveform；历史 80-case API run 未冻结 case IDs、包版本和 source hashes |

外部数据集级事实见 `registries/external_dataset_registry.json`，PTT/Sim 逐记录清单见 `manifests/external_record_manifest.csv`。
