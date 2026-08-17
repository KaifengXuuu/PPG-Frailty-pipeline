## 2026-08-03 — 五类方法逐源码审计 / Five-family line-level audit

- 操作 / Action：新增五类方法的代码、理论、应用、测试、缺口与实现可行性总审计。
- 写入 / Written：`M0_history_MA_denoising_detector_HR_feature/03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md`。
- 只读复核 / Read-only review：三项并行审计覆盖 52 份代码/notebook、4,190,267 bytes，SHA-256 mismatch 为 0；主进程复核关键函数行段和实际结果表。
- 代码发现 / Code findings：现有真实实现限于 IMU-NLMS、CEEMD-lite、DWT-A2、STFT工具和部分SQI；Wiener/RLS、标准小波阈值、CWT/WPT、EEMD/VMD/SSA、完整谱追踪和BSS均不存在。
- 算法发现 / Algorithm findings：ANC 独立性前提被运动诱发真实 HR 破坏；现有双路STFT没有IMU频率局部证据；BSS数据可用但实现为0；SQI只覆盖四个加权分量。
- 测试发现 / Test findings：历史 adaptive/decomposition 缺定量 scorecard；谱追踪/BSS 无测试；SQI只有轻微正向 generalization 消融而无独立质量验证。
- 输出 / Outcome：为五族分别规定统一接口、至少 5–10 类可执行测试、安全门和状态判定。
- 边界 / Boundary：没有把“可实现”写成“已实现”；没有联网读取 TROIKA/JOSS，也没有进入新算法编码。
