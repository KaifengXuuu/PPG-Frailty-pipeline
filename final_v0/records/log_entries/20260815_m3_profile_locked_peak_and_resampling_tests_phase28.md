# M3 profile-locked peak and resampling tests phase 28 / M3 Profile 锁定峰检测与重采样测试第 28 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：对第 27 阶段新增的 peak-purpose gate、external resampling facade、公共导出和离散索引映射运行完整参考测试。
- 算法 / Algorithm：500→400 Hz 使用 4/5 polyphase 波形重采样；时间按 target grid 构造，valid mask 使用 target→nearest-source，峰标注使用 source→target rounding。峰检测同时验证 profile status、modality、purpose、400 Hz 与冻结滤波合同。
- 结果 / Result：M3 参考测试由 42 增至 44，44/44 通过，0 failures、0 errors、0 skipped。正例确认波形/时间/mask/峰同步，负例确认 125 Hz、motion-purpose profile 和错误峰检测采样率均 fail closed。
- 边界 / Boundary：external facade 当前只登记 PTT 500 Hz 与 Sim 256 Hz；MIMIC 125 Hz 必须未来新建独立 profile，禁止复用本合同。mask 映射用于对齐有效性，不会把 invalid source 样本静默改为 valid。
