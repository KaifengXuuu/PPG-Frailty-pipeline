# M1 V3 当前验证入口修正 / Current V3 Validation Entry

## 1. 权威关系 / Authority

- 架构语义仍为 `m1.architecture.v3`；本文件不改变路由算法、schema 或三档配置。
- 本文件仅修正 V3 首版 validator 对迁移元数据的误报。当前机器验证入口为：
  - `tools/validate_m1_contracts_v3_current.py`；
  - `registries_v3/quality_routing_registry_v3_active.json`；
  - `M1_CONTRACT_VERIFICATION_V3_CURRENT.json`；
  - `M1_PACKAGE_TREE_V3_CURRENT.md`。
- `tools/validate_m1_contracts_v3.py` 与 `registries_v3/quality_routing_registry_v3.json` 保留为首版及迁移说明历史。首版 registry 的 `legacy_migration` 故意写出旧字段原名，不代表这些字段仍在活动合同中。
- 若首版验证报告与 CURRENT 报告冲突，以 `*_V3_CURRENT` 为准。

## 2. 修正原因 / Reason

首版 validator 对整个 registry 做禁止词搜索，把 `legacy_migration` 中用于明确“禁止/需迁移”的旧名也误判为活动字段。CURRENT 入口改为校验一份不含迁移叙述的 active registry，同时继续保留原迁移证据。

该修正不放宽活动合同：V3 schema、examples 与 active registry 仍禁止旧 `action owner`、SQI weighting、coarse waveform replacement 和 denoiser-before-route 语义。

## 3. 环境记录 / Environment note

工作区终端的默认沙箱刷新仍报告 `helper_unknown_error`，导致已有文件无法由补丁通道原位读取。为遵守“使用 apply_patch 编辑”和保留历史证据的规则，本次采用追加式校正文件；没有用 shell 覆盖旧文件。

