# M3 fold artifact envelope phase 22 / M3 训练折 artifact 完整封装第 22 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：在 schema 设计复核中发现 runtime artifact 已声明 m3.fold_fitted_artifact.v1，却缺少初稿 schema 要求的完整身份与 transformer envelope；选择升级 runtime 而不是放宽同版本 schema。
- 算法 / Algorithm：artifact 绑定 M2 fold file SHA 与 payload SHA、dataset/protocol/repeat/fold/seeds、exact train/OOF partition、M3 registry/profile payload hashes、feature schema、median imputer、RobustScaler/no-clip 参数及 canonical parameters SHA。
- 结果 / Result：成功 artifact 与负例测试均通过；全量 reference tests 42/42。
- 边界 / Boundary：调用方必须显式传入非空唯一 preprocessing_profile_ids；历史或 deprecated profile 不可生成 future artifact。
