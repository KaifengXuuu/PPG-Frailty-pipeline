# M0 执行、算法与结果总报告 / M0 Execution, Algorithm, and Results Report

- TODO：M0 完整审计历史 Motion Artifact、动态降噪和 Heartbeat 路线
- 状态 / Status：`complete`
- 日期 / Date：2026-08-02
- 写入范围 / Write scope：仅 `final_v0/`
- 源项目状态 / Source project state：只读；未改动代码、数据、结果、`AGENTS.md` 或 `_agent/`
- 外部联网 / External network：未使用

## 1. 结论先行 / Outcome

M0 验收目标已经达到：已建立可追溯的历史 motion-processing registry、代码—输入—输出 crosswalk、论文证据边界、风险登记和算法结构图。审计结论是：

1. 现有任何一条动态 PPG 去噪路线都不同时具备“真实 clean reference、独立 subject holdout、可复现推理程序、正向生理指标改善”四项证据。
2. 历史 full-waveform reconstruction 结论维持 `failed_or_deprecated`；不得恢复为主线。今后粗处理只允许作为 HR/PPI 提取或 Frailty3 分类输入候选，并必须与 raw/no-denoising 和 high-quality-only 比较。
3. v7 有阈值与输入泄漏；v7.2 切分更严格，但独立 holdout SNR 全部为负；v7.4 生成了模型/ONNX，却没有独立去噪效果；v8 MaskNet 和 Stage-2 均存在运行阻断与校准错误。
4. hybrid 路线有训练、预览、A/B 和 ONNX 工程产物，但没有 clean ground truth 或 holdout scorecard；其 artifact loss 与 clean loss 数学等价，属于同一代理目标重复计权。
5. `ppg_peak_hr_gating_train.py` 的主模型实际是 PPG-only，并以未做 ECG→PPG delay 校正的 ECG R-peak 时刻作为 peak target。它不能被表述为已经完成 PPG pulse-peak/HR 提取。
6. 当前最有希望的 motion detector 证据是独立的 10-channel PPG+IMU A/B benchmark；external SIM 上 Light CNN 略优，但仍只有 pooled overlapping-window 指标，没有 subject-level CI。
7. legacy v8 在同域获得近满分，但几乎由 IMU 活动强度完成 sit 与 walk/run 区分；它不是 PPG artifact ground-truth detector，并且 CV preprocessing/lag 存在泄漏与跨记录错位。

## 2. 详细流程报告 / Detailed process report

### 2.1 规则、TODO 与 Git 基线

- 完整读取 `AGENTS.md`、`_agent/WRITE_RULES.md` 与权威 `_agent/TODO.md`。
- Git 分支为 `dev0`；开始前已有 5 个用户修改文件，本轮保持不变。
- `final_v0` 开始时为空；所有新文件均在该目录内。

### 2.2 Workspace 基线扫描

| 项目 | 数量/大小 | 读取方式 | 结果 |
|---|---:|---|---|
| workspace 文件（排除 `.git`、`final_v0`） | 35,214 / 42,794,025,593 bytes | 全树元数据 | 0 错误 |
| 根目录文件 | 45 | 41 个文本/代码逐字节；4 个非文本元数据 | 0 错误 |
| 代码/notebook | 52 | 全部逐字节至 EOF + SHA-256 | 0 错误 |
| 静态路径引用 | 2,387 | AST/notebook 字符串提取 | 已登记 |
| 输入目录 | 7 / 6,405 文件 / 34,581,621,834 bytes | 每文件最多 65,536 bytes 头部与结构 | 0 错误 |
| 输出目录 | 17 / 28,670 文件 / 8,165,668,577 bytes | 文本 EOF；非文本名称/格式/大小 | 0 错误 |
| 输出文本 | 9,314 | 全字节、SHA-256、行数/schema/指标行 | 全部 `bytes_read == file_size` |
| 非文本输出 | 19,356 | 不读取载荷，仅登记元数据 | 全部登记 |
| 扫描事务 | 25 | baseline + 7 input + 17 output | 全部存在 |

证据一致性由 `tools/verify_scan_evidence.py` 独立复核，`SCAN_VERIFICATION.json` 为 `pass`，失败数 0。

### 2.3 输入结构复核

- `physionet.org`：4,920 文件，覆盖 PTT、simultaneous、iAMwell、MIMIC 等本地数据；头部识别为 3,374 文本、119 MAT、30 ZIP、1,397 其他二进制。
- `PPG_Testing_05_01_2026`：1,134 文件；1,068 文本、1 个 XLSX/ZIP、65 个图片或其他二进制。
- `datasets`：327 文件；290 CSV、37 NPZ。
- `train_raw/train_labeled/train_val/train_window`：24 文件，发现三套不兼容的 SVM 窗口 schema（45、92、116 列）；该问题不属于 M0 修复范围，已进入未来决策门。
- `.env` 虽参与完整字节校验，但任何值均未写入证据；只保留变量名与 `values_redacted=true`。

### 2.4 输出完整读取与代码对应

- 所有 `results_*`、`.CNN_results`、`models`、`denoiser_preview_output` 与 `test_asa_classifier` 均已建立逐文件 manifest。
- `results_denoiser_v8` 和 `results_stage2` 实际文件数均为 0；“目录存在”不能视为实验完成。
- M0 相关 JSON/CSV/Markdown 逐份与代码声明的字段、默认路径和实际产物核对。
- 对 PT/ONNX/NPZ/PNG/Pickle 等非文本只登记名称/格式/大小；除已有安全元数据外未反序列化模型。

### 2.5 代码、结果与历史记录三方交叉

- 代码证据：17 组历史方法、16+ 个脚本入口、关键函数和确定性错误。
- 结果证据：v7/v7.2/v7.4/v8/hybrid/heartbeat/motion A-B 的真实指标和缺失产物。
- 历史证据：`_agent/MODULES.md`、归档 handoff、NOTES、decision log、CHANGELOG、ROADMAP。
- 冲突处理：以当前代码和实际文件为事实依据；历史结论若超出证据，在 registry 中降级为 `implemented_unverified`、`smoke_only` 或 `failed_or_deprecated`。

## 3. 算法路线报告 / Algorithm-route report

### 3.1 基础透明算法

`funcs.py`/`ppg.py` 包含 Butterworth 滤波、Aboy++ 风格峰值、EKF/重力去除、IMU motion classification、多参考 NLMS ANC 和 CEEMD-lite+NLMS。其优势是透明、无训练权重；但当前实现存在会直接改变 HR/HRV、滤波和默认 UI 行为的错误，不能当作已验证公共实现。

### 3.2 v7 → v7.2 → v7.4 → v8/Stage-2 演化

- v7：DWT+CNN/BiLSTM AE detector；1D U-Net 去噪；setup2 把 ECG/peak/p6 直接作为输入。
- v7.2：先固定真正 external subject holdout；ECG peak 只作 HR proxy supervision，不作为推理输入。
- v7.4：activity-rule/AE/fused detector；两路 PPG STFT magnitude + 37 个广播特征进入 2D magnitude MaskNet；加入 sit template 和 ECG delay。
- v8/Stage-2：尝试时域通道 + 广播特征 + mask；但变长 peak tensor 默认 collate、变量覆盖、phase 符号、无梯度 subject 参数和 holdout 选择等问题阻断可信验证。

### 3.3 Hybrid pseudo-supervised reconstruction

- 输入 A：`raw PPG + IMU`（11 通道）。
- 输入 B：`raw PPG + IMU + linear baseline clean/artifact`（15 通道）。
- 以 sit template、ECG/PPG peak、linear baseline 和 clean/identity/anchor 代理项训练 residual artifact U-Net。
- B 的 validation proxy objective 比 A 低约 17%，但该指标不是 clean waveform accuracy，不能解释为真实去噪提升。

### 3.4 Direct heartbeat / IBI / gate

- 主网络：PPG-only 1D U-Net + dense peak、dense IBI、window gate 三头。
- target：未 delay-correct 的 ECG R timing；IBI 为 ECG RR track；gate 仅部分数据有标签。
- 辅助 motion benchmark：独立使用 PPG + gravity-removed IMU 的 10 通道 A/B 分类器，不应与 PPG-only gate 混称。
- 当前 direct route 的代码/scorecard完整，但外部 peak/gate 泛化不足、HR(bpm)未实现，不能作为最终生理模块。

## 4. 验收条件逐项对照 / Acceptance mapping

| M0 验收要求 | 完成状态 | 证据 |
|---|---|---|
| 重新阅读代码、结果、scorecard、preview、历史记录 | 完成 | 代码/输出 manifests、三方审计、M0 报告 |
| 覆盖 TODO 指定全部方法 | 完成 | `M0_METHOD_REGISTRY.md` |
| 每法记录输入、输出、监督、预处理、参数、数据、split、协议、指标、结果、失败、部署 | 完成 | method registry 逐法字段 |
| 区分严格验证/未验证/smoke/失败/未实现 | 完成 | 统一状态词汇与 evidence tier |
| 检查 leakage、阈值拟合、ECG–PPG 对齐、纯视觉评价 | 完成 | `M0_RISK_REGISTER.md` |
| 保留“不恢复完整 clean waveform”结论 | 完成 | 本报告结论、论文证据边界 |
| 形成可追溯 registry，避免重复试验 | 完成 | registry + crosswalk + algorithm diagrams |

## 5. 已完成、未完成与验证状态 / Completion and verification

### 已完成

- 全 workspace 代码/输入/输出基线与可机器验证 manifests。
- M0 所有历史方法的实现、验证、结果、泄漏和部署状态登记。
- M0 quantitative evidence、claim boundary 和算法结构图。
- 待录入 `_agent` 的候选主题已保存于 `PENDING_AGENT_UPDATES.md`，未写 `_agent`。

### 本项明确未做

- 未训练、重跑、修复或替换任何历史模型。
- 未反序列化 Pickle/PT；未读取二进制模型载荷。
- 未把视觉 preview 重新解释为数值证据。
- 未在 M0 提前实施 M1–M10。

### 验证状态

- 扫描证据：`pass`。
- 代码语法：M0 审计脚本 AST 均可解析；这不覆盖已识别的运行时阻断。
- 结果真实性：仅陈述实际存在的文件与字段；目录存在但无核心结果的一律不标记完成。

