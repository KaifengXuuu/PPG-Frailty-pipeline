## 2026-08-02：M0 全量扫描与证据校验 / M0 full scan and evidence verification

- 状态 / Status：`reporting_in_progress`
- 代码 / Code：52 个代码文件逐字节完整读取，均保留 SHA-256 或明确错误记录；错误数 0。
- 输入 / Inputs：7 个目录、6,405 个文件、34,581,621,834 bytes，逐文件读取 65,536 bytes 以内头部并识别结构；错误数 0。
- 输出 / Outputs：17 个目录、28,670 个文件、8,165,668,577 bytes。
- 输出文本 / Output text：9,314 个文本文件完整读取至 EOF，均满足 `bytes_read == file_size` 并记录 SHA-256。
- 非文本输出 / Binary outputs：19,356 个文件仅登记名称、格式和大小。
- 一致性 / Consistency：25 个预期扫描事务全部存在；`SCAN_VERIFICATION.json` 状态 `pass`，失败数 0。
- 当前工作 / Current work：编制 M0 registry、代码—输入—输出 crosswalk、论文证据报告及算法图。

