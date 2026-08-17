## 2026-08-02 — M0 crosswalk 文件数按 manifest 更正

- 发现：人工整理表中的若干目录文件数按预期结构估算，与输出扫描manifest不一致。
- 更正：以 `records/generated/outputs/*.summary.json` 的 `file_count` 为唯一依据，更新 `results=5`、`v72=16`、`v7_4=55`、`v7_3=33`、`v8_audit=30`、两个hybrid variant各8、legacy hybrid=6。
- 工具：`correct_m0_crosswalk_manifest_counts.py` 对6个完整字符串执行唯一命中替换。
- 影响：只纠正实际文件数量；算法、指标、证据等级和M0结论不变。

