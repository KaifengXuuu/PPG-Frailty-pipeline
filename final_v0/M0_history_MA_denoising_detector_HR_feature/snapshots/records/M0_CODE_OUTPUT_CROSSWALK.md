# M0 代码—输入—输出对应表 / Code–Input–Output Crosswalk

- 状态 / Status：`complete`
- 对应范围 / Scope：M0 Motion Artifact、动态降噪、Heartbeat 与其公共基础函数。
- 核对方法 / Method：代码逐字节读取与静态路径提取；输入文件头部/schema 扫描；输出文本逐字节至 EOF；二进制输出登记名称、格式、大小；结果字段与代码写出结构逐项比对。
- 重要限制 / Limitation：PT、ONNX、NPZ、PNG 等二进制载荷没有反序列化；映射依据代码写出语句、同目录文本元数据和文件名契约。

## 1. 汇总映射 / Summary mapping

| ID | 代码入口 | 主要输入 | 代码声明输出 | 实际输出对应 | 一致性结论 |
|---|---|---|---|---|---|
| F01–F05 | `funcs.py`, `ppg.py` | 用户选择的 PPG/IMU/ECG CSV；`.env` 中路径变量 | Dash 图、HR/HRV 表及用户指定 CSV | 无单一固定结果目录；存在交互式/按用户路径写出 | `partial`：算法存在，但路径契约与结果基线不固定 |
| V01 | `pttppg_pipeline_v7.py` | `physionet.org/.../PTT...` CSV | `results/` 下 detector/denoiser/comparison JSON、PT、PNG | `results/`：5 文件、结构与 v7 写出项相符 | `matched_with_invalid_protocol` |
| V02 | `cnnppg_v7.py` | PTT CSV | `results_v72_noleak/` split、AE、CV/holdout、图表 | 目录存在，16 文件；文本/二进制角色与代码相符 | `matched_negative_result` |
| V03–V04 | `pttppg_pipeline_v7_4_noleak_viz_ae.py` | PTT CSV | 默认名 `results_v7_3`；可由参数指定；detector + denoiser artifacts | `results_v7_4/` 55 文件为显式参数运行；`results_v7_3/` 另有旧/不同批次 33 文件 | `matched_but_name_ambiguous` |
| V05 | `pttppg_denoiser_v8_masknet.py` | PTT CSV、v7.4 风格 features/peaks | `results_denoiser_v8/` metrics/model/a-table | 目录存在但 0 文件 | `declared_but_not_produced` |
| V06 | `pttppg_stage2_denoiser.py` | PTT CSV、Stage-1/ECG/shape pseudo labels | `results_stage2/` CV/final PT、ONNX、metrics | 目录存在但 0 文件 | `declared_but_not_produced` |
| D01 | `pttppg_detector_v8_scores_audit_fix9.py` | PTT CSV | `results_v8_audit/` 三个 window 配置的 bundle、summary、audit 图 | 30 文件；三组 summary/audit/NPZ 均存在 | `matched_but_biased_protocol` |
| H01 | `pttppg_denoiser_hybrid_train.py` + `pttppg_denoiser_hybrid_core.py` | PTT CSV | PT、meta、history、split、delay；可导出 ONNX | `results_hybrid_denoiser_raw_imu/` 与 `_baseline/` 各 8 文件；契约相符 | `matched_proxy_only` |
| H01 legacy | 同一 hybrid 系列的较早运行 | PTT CSV | `results_hybrid_denoiser/` | 6 文件；best val/schema 与后两组不同 | `historical_noncomparable` |
| H02 | `pttppg_denoiser_hybrid_preview.py`, `..._ab_compare.py` | Hybrid bundle + PTT CSV | preview PNG / A-B 视觉比较 | `denoiser_preview_output/` 8 PNG | `matched_smoke_only` |
| H03 | `pttppg_denoiser_hybrid_export_onnx.py`, `pttppg_denoiser_onnx_runtime.py`, `ppg_denoiser_dash_utils.py` | PT/meta/CSV/ONNX(+`.data`) | ONNX contract、runtime 波形、Dash traces | raw variants 含 ONNX、`.onnx.data` 与 JSON；未发现端到端 parity scorecard | `artifacts_present_contract_incomplete` |
| P01–P02 | `ppg_peak_hr_gating_train.py` | PTT、simultaneous、iAMwell、MIMIC、VitalDB；SIM external | `.CNN_results/<run>/` PT/ONNX、scorecard、split、plots、A/B benchmark | 687 文件覆盖多次运行；最新相关 run 产物齐全 | `matched_but_main_model_negative` |

## 2. 公共基础入口 / Foundation entry points

### `funcs.py`

- 输入路径：本文件本身主要提供函数，不固定数据目录；路径由 `ppg.py` 或调用脚本传入。
- 输入结构：一维 PPG；三轴加速度/角速度；采样率；峰索引；滤波/ANC参数。
- 输出结构：NumPy array、peak indices、PPI/HRV统计、motion score/label、ANC artifact/residual。
- 文件输出：没有稳定的模块级输出目录。
- 对应结论：无法用单个现存结果目录证明每个函数正确；必须在 M3 以固定 fixture/parity test 建立新证据。

### `ppg.py`

- 输入路径：由 `.env` 和 Dash 交互参数选取，允许任意 CSV 目录；`.env` 值在扫描证据中已脱敏。
- 输入结构：PPG、IMU、ECG/peak 等列，具体列名随数据源/页面分支变化。
- 输出结构：交互图、状态、peak/PPI/HRV 表；部分分支写 HRV CSV。
- 模型依赖：legacy v8 默认查找 `results_v8_audit/detector_v8_bundle.npz`，实际目录没有这个准确文件名，只有按 window 配置命名的 bundle。
- 对应结论：当前默认 UI 路径不能被现存产物无歧义满足；且 runtime feature contract 与训练脚本存在 jerk/参数差异。

## 3. v7 系列 / v7 family

### `pttppg_pipeline_v7.py` → `results/`

- 输入：PTT CSV，主要列 `pleth_4/5/6`、accelerometer、gyroscope、ECG及peak。
- 主要写出：detector fold JSON、denoiser setup1/setup2 fold JSON、comparison JSON、PT/PNG。
- 实际对应：目录与字段存在；setup2 输出可以追溯到 ECG/peaks/p6 被送入推理输入。
- 不一致不是文件缺失，而是协议含义：所谓 holdout 仅为最后一折，阈值在被评价数据上拟合。

### `cnnppg_v7.py` → `results_v72_noleak/`

- 输入：PTT CSV；8 通道推理输入；ECG peak 只用于 HR proxy supervision/evaluation。
- 主要写出：`splits.json`、AE状态、CV/holdout CSV/JSON、SNR/HR图。
- 实际对应：AE 跳过原因、negative holdout SNR 与代码分支一致；不是扫描缺失。
- 解释：输出完整但结果为负，不能因为文件齐全而标记方法成功。

### `pttppg_pipeline_v7_4_noleak_viz_ae.py` → `results_v7_4/`

- 输入：PTT `pleth_1/2`、IMU、ECG/peaks用于训练约束。
- detector 输出：`cv_summary.json`、`detector_artifact.json`、rule NPZ、AE PT、fold/holdout confusion 与 lag 图。
- denoiser 输出：walk/run PT、ONNX、外部 `.onnx.data`、meta、subject-a table。
- 实际对应：上述文件均存在；walk/run meta 的 best val loss 分别 `.987795...` 与 `.702435...`，a-table 全为 `1.0`。
- 路径歧义：脚本默认目录名仍为 `results_v7_3`；当前 `results_v7_4` 必须是显式参数运行。论文引用必须写实际目录与 config，不得仅按脚本默认名推断版本。

### v8/Stage-2 空目录

- `pttppg_denoiser_v8_masknet.py` 声明 `results_denoiser_v8`，实际 0 文件。
- `pttppg_stage2_denoiser.py` 声明 `results_stage2`，实际 0 文件。
- 代码内又存在可确定的运行时/训练协议问题，因此空目录被解释为“未产出”，不是“扫描遗漏”。

## 4. Detector 与 Hybrid / Detector and hybrid

### `pttppg_detector_v8_scores_audit_fix9.py`

- 输入：PTT sit/walk/run CSV。
- 输出：`results_v8_audit/{1_0.5,2_0.5,6_1}/detector_v8_summary.json`、各自 `audit_summary.json` 与图；根下按窗口命名的 NPZ。
- 实际对应：三组均存在；2 s/0.5 s 的 fused holdout F1 `.988133...`，同时 IMU-only 近满分而 PPG-only BA `.7203`。
- 旧目录：`results_detector_v8/` 是不同 schema/历史版本，不应与 `results_v8_audit` 混合聚合。

### Hybrid 训练、预览和部署

- `pttppg_denoiser_hybrid_train.py` 负责 CLI/split/train/write；`..._core.py` 负责 loader、preprocessing、ridge baseline、window、network/loss/OLA。
- raw+IMU 输出目录含 `hybrid_denoiser.pt`、`hybrid_denoiser_meta.json`、`train_history.json`、`splits.json`、delay 与 ONNX contract。
- baseline variant 使用相同结构但 15 通道；两目录可以按 meta 的 `input_mode` 区分。
- `pttppg_denoiser_hybrid_preview.py` 与 `..._ab_compare.py` 对应 8 个 PNG；没有 CSV/JSON metric，因此只能证明推理路径可画图。
- export/runtime 只把 `model_input → artifact_hat` 放入 ONNX；PPG/IMU preprocessing、ridge、normalization 与 overlap-add 都在 Python，部署包不是单文件自包含。

## 5. Heartbeat 与独立 motion A/B / Heartbeat and motion A/B

### `ppg_peak_hr_gating_train.py` → `.CNN_results/`

- 主输入：PPG 单通道；ECG/annotation用于 target 和评价；主 gate 不使用 IMU。
- 独立 A/B 输入：10 通道 PPG+dynamic IMU；这是同脚本中的另一路 benchmark。
- 主要写出：run config/splits、fold curves、OOF/holdout/extra scorecards、PT/ONNX/meta、motion A/B benchmark。
- 实际目录包含多个日期版本；跨 run 的 subject count、target 或指标定义不同。M0 论文数值固定引用最新审计 run，而不是把 687 个文件混合平均。
- 输出语义：`peak_events` 是对 ECG timing target 的窗口级事件匹配，并非已校正的 PPG pulse-peak；HR(bpm)结果并未写出。

## 6. 对应关系使用规则 / Usage rules

1. 引用任何数值时同时写明脚本、实际结果目录、run/window配置、split 和目标定义。
2. “目录存在”“模型存在”“ONNX存在”均不能替代 holdout/external scorecard。
3. `results_v7_3`/`results_v7_4`、`results_detector_v8`/`results_v8_audit`、三个 hybrid 目录、多个 `.CNN_results` run 不得合并为同一实验。
4. 对空目录、skipped状态、NaN与负结果保持原义；不得按预期写出结构补写不存在的结果。
5. 二进制模型的网络/参数声明来自代码与文本 meta；未反序列化时不声明其内部权重完整或可运行。

