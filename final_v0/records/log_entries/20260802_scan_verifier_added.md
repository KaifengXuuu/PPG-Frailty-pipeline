# 扫描证据校验器建立 / Scan-evidence verifier added

- 日期 / Date：2026-08-02
- 状态 / Status：`implemented_unverified`
- 文件 / File：`final_v0/tools/verify_scan_evidence.py`
- 目的 / Purpose：核对 baseline、7 个输入目录、17 个输出目录及扫描账本的计数、总字节、EOF、SHA-256 和错误状态。
- 写入边界 / Write boundary：只写 `final_v0/records/generated/SCAN_VERIFICATION.json`。
- 下一步 / Next：运行校验器；通过后把统计结果写入 M0 扫描与结果报告。

