## 2026-08-02 — M0 Dash 工具文件名更正

- 发现：三处审计引用写成 `dash_denoiser_utils.py`，实际根文件为 `ppg_denoiser_dash_utils.py`。
- 更正：使用只允许每目标命中一次的 `correct_m0_dash_filename_reference.py`，精确修改 crosswalk、逐脚本图册与图覆盖校验器。
- 复核：更正后算法图验证仍为 `pass`；6份图文档、29个 Mermaid 图块、16个 M0 入口全部覆盖。
- 影响：只改审计文件名，不改变算法、指标、风险等级或项目源文件。

