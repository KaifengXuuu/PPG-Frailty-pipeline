# 根目录逐文件 I/O 与内容清单 / Root-file I/O and Content Inventory

- 状态 / Status：`complete`
- 覆盖 / Coverage：workspace 根目录 45 个文件，逐份完整读取或按非文本规则登记；其中29个代码/Notebook也在 `CODE_FILES.jsonl` 中逐字节校验。
- 安全 / Safety：`.env` 值和下载来源URL未复制；`detector_v8_calib.npz`只登记ZIP/NPY成员元数据，不读取数组载荷。
- 机器证据 / Machine evidence：`records/generated/ROOT_FILES.jsonl`（45行）与 `CODE_FILES.jsonl`（全workspace 52行）。

## 1. M0 根代码入口（16个）

| 文件 | 直接职责 | 输入路径/结构 | 输出路径/结构 | 实际对应与状态 |
|---|---|---|---|---|
| `funcs.py` | 滤波、Aboy++ peak/PPI/HRV、IMU EKF/重力、ANC、CEEMD-lite公共函数 | 调用方传入PPG/IMU数组、fs、peaks和参数；无固定文件路径 | 返回array/dict/Plotly对象；无固定目录 | 无单一结果目录；`implemented_unverified`，详见M0 F01–F05 |
| `ppg.py` | Dash PPG/IMU/HRV交互管线及legacy v8 runtime | `.env`键、用户选择CSV；IR/RED/AX–GZ等 | Dash图表；用户指定目录HRV CSV | 默认v8 bundle名不匹配；多项runtime/API错误；研究原型 |
| `pttppg_pipeline_v7.py` | v7 DWT-AE detector与setup1/2 U-Net | PTT `pleth_4/5/6`、IMU、ECG/peaks | `results/` detector/denoiser/comparison文件 | 5文件；协议泄漏/负结果；`failed_or_deprecated` |
| `cnnppg_v7.py` | v7.2 subject-holdout proxy denoiser | PTT 8通道；ECG peaks只作proxy监督 | `results_v72_noleak/` split、CV/holdout、图 | 16文件；holdout SNR全负；`failed_or_deprecated` |
| `pttppg_pipeline_v7_4_noleak_viz_ae.py` | activity rule/AE/fusion与STFT MaskNet | PTT pleth1/2、IMU、ECG/peaks | 默认`results_v7_3`，显式run为`results_v7_4` | 33/55文件；detector有限，denoiser无holdout效果 |
| `pttppg_denoiser_v8_masknet.py` | 8+37通道time-mask实验 | PTT windows、变长peaks | `results_denoiser_v8/` model/metrics/a-table | 0文件；确定性forward阻断；deprecated |
| `pttppg_stage2_denoiser.py` | 47通道Stage-2 mask与ECG/shape proxy loss | PTT、phase/ECG/pseudo shape | `results_stage2/` CV/final PT/ONNX/summary | 0文件；collate/gradient/phase/holdout问题 |
| `pttppg_detector_v8_scores_audit_fix9.py` | 10 PPG+27 IMU手工特征、Mahalanobis、lag、logistic fusion | PTT sit/walk/run，500Hz | `results_v8_audit/`三配置summary/NPZ/audit图 | 30文件；IMU活动主导且CV transform泄漏；legacy |
| `pttppg_denoiser_hybrid_core.py` | Hybrid loader、IMU特征、lag-ridge、dataset、UNet/loss/OLA | PTT双PPG+IMU+可选ECG/peaks | 由train/preview/runtime调用；本身不固定run目录 | 核心实现存在单位、proxy重复、边界coverage风险 |
| `pttppg_denoiser_hybrid_train.py` | Hybrid CLI、subject split、train和bundle写出 | PTT CSV，core配置 | `results_hybrid_denoiser*` PT/meta/history/splits/delay | 三历史目录6/8/8文件；无holdout scorecard |
| `pttppg_denoiser_hybrid_preview.py` | 单bundle定性预览 | Hybrid PT/meta + PTT CSV | `denoiser_preview_output/*.png` | 与A/B合计8 PNG；仅smoke |
| `pttppg_denoiser_hybrid_ab_compare.py` | raw_imu与baseline variant同窗视觉A/B | 两个bundle + PTT CSV | 对比PNG | 无数值score/CI；仅smoke |
| `pttppg_denoiser_hybrid_export_onnx.py` | 从PT/meta导出ONNX并做随机tensor差异 | Hybrid PT/meta | ONNX、外部`.onnx.data`、contract JSON | 两个新variant均有产物；非端到端parity |
| `pttppg_denoiser_onnx_runtime.py` | Python侧预处理、window、ORT、artifact subtraction、OLA | CSV + ONNX/meta/`.data` | 内存waveform/mask/diagnostic | 无独立固定scorecard；preprocessing重复实现 |
| `ppg_denoiser_dash_utils.py` | Dash接入Hybrid runtime、cache、motion mask与trace | Dash record/request + bundle | Plotly trace/display状态 | mask时间对齐与cache失效风险；无固定文件输出 |
| `ppg_peak_hr_gating_train.py` | PPG-only peak/IBI/gate；另含10通道PPG+IMU A/B | PTT/iAMwell/MIMIC/VitalDB/SIM等 | `.CNN_results/<run>/` PT/ONNX/scorecards/plots | 687文件多run；P01外部失败，P02 SIM BA .7802候选 |

M0 逐法算法、输入、监督、split、指标和错误以 `M0_METHOD_REGISTRY.md` 为准；逐目录对应以 `M0_CODE_OUTPUT_CROSSWALK.md` 为准。

## 2. 非 M0 主 Python 脚本（8个）

| 文件 | 主要算法/职责 | 输入路径/结构 | 输出路径/结构 | 实际结果与关键问题 |
|---|---|---|---|---|
| `analyze_sweep.py` | 合并sweep；配置级均值/SD/t-CI、逐类、混淆矩阵、完整性和排名 | `results_frailty3/<sweep>`或`_overfitting_sweep/<run>`的runs/manifest/reports/curves | `_sweep_analyse/<run>/` clean runs、leaderboard、incomplete、PNG、Markdown | 14目录；12份主报告可用，1个无核心文本、1个有表无报告；不消除上游泄漏 |
| `asa_classifier.py` | VitalDB ASA1/2/3；PPG/频谱/RR-HRV多分支、case pooling、subject CV/holdout、OOF阈值 | VitalDB clinical/cases/tracks与本地PTT ECG预检 | `test_asa_classifier/_vitaldb_signal_cache`和run目录config/PT/JSON/CSV/plots/scorecard | 最终1,196 test cases；OOF BA .4728、macro-F1 .4667；argmax不预测ASA2 |
| `frailty_3class_classifier.py` | PPG/IMU特征、Aboy/PPI/HRV/形态；LogReg/SVM/ET/CNN/Inception/ShapeFormer | StudyData、Youngers、frailty label CSV；RED/IR/AX–GZ | `datasets/frailty3_*`、`results_frailty3` reports/curves、`models` | 新缓存29 subjects；旧深度CV以test fold早停后同折报告，存在选择泄漏 |
| `frailty_3class_cnn_fusion.py` | CNN/Inception信号编码+手工特征MLP，折内imputer/scaler | frailty NPZ与完整手工feature CSV | fusion report、PT和scaler | 两报告各870 windows/145 files/29 subjects；同样有外折早停泄漏 |
| `frailty_3class_holdout_eval.py` | leaderboard候选的train/inner-val/test独立留出、多seed CI | `_sweep_analyse/.../leaderboard_top_configs.csv`与CNN cache | `_holdout_eval/<run>/` manifest/runs/summary/reports/curves/CM | 6目录：2完整、2为1/15、2空；当前frailty最可信协议，CI仍宽 |
| `frailty_3class_overfitting_sweep.py` | 固定epoch Stage1/2/generalization网格，无早停 | leaderboard/前序summary/fixed reference与frailty cache | `_overfitting_sweep/<run>/` manifest/runs/summary/reports/curves | 27目录：9完整、3部分；`test`多为OOF validation别名；gap约.46 |
| `shapeformer_port.py` | local conv/attention + global shapelet/PISD | `[N,C,T]`窗口与label；PISD硬编码外部ShapeFormer仓库 | 返回bundle/model，由主分类脚本写report/PT | 无独立CLI；外部依赖不可自包含；complexity量纲不一致 |
| `svm2_dataset_train.py` | Dash区间标注、PPG/IMU/姿态/Welch/熵特征、Scaler/PCA/SVC、Group CV | 硬编码`/mnt/d/Tubcloud/...`；labeled/raw/window/val CSV多schema | `train_*`、`datasets/motion_dataset*`、`models/svm_motion_*.pkl` | **不可编译**：future import位置错误；推理窗错配、IMU可选矛盾、无imputer、test_ratio=0泄漏；无论文级持久指标 |

## 3. Notebook（5个）

| 文件 | 职责 | 输入 | 输出/实际对应 | 保存状态与问题 |
|---|---|---|---|---|
| `PPG_Analy_Visual_test.ipynb` | 旧Dash PPG peaks、HR/PPI/HRV、SpO2、IMU、ANC/CEEMD；两HRV库比较 | `.env`路径与PPG/IMU CSV | `datasets/*-HRV-*.csv`三列42行；282份同类文件 | 22 code cells/16执行；保存运行因2个RR做cubic spline报错；浏览器启动失败 |
| `ppg_analyse3.ipynb` | 旧Dash接legacy v8 detector与IMU校准 | `.env`、缺失的默认bundle、CSV | Dash与HRV CSV | 巨型cell `execution_count=None`但有保存输出；多次Wrong Path，acc bias异常 |
| `ppg_analyse4_calib.ipynb` | v8 detector校准、anchor、denoiser A/B、HRV | `.env`、bundle、`ppg_denoiser_dash_utils.py`、CSV | calibration NPZ、Dash、HRV CSV | 保存Notebook未执行；根`detector_v8_calib.npz`结构吻合但provenance未证实 |
| `svm2_dataset_train.ipynb` | SVM标注/数据集/训练/预览的分cell版本 | 硬编码原始数据与train_* | train_*与649个PKL工作流 | 18 cells/13执行；保存60 files/88 features文字，无持久化性能指标 |
| `template_test.ipynb` | VitalDB clinical/ASA/PLETH/ECG勘探 | VitalDB API | 仅Notebook output | 保存6388 cases与6156 PLETH+ECG统计；含两次API误用，不是完整导入验证 |

## 4. 配置、文本、二进制和附属文件（16个）

| 文件 | 内容/职责 | 输入/输出关系 | 状态与风险 |
|---|---|---|---|
| `.codex` | 0字节占位 | 无I/O | 遗留空文件 |
| `.dockerignore` | Docker context排除规则 | Docker build读取；不写仓库文件 | 未排除大型data/results/models，context可能巨大 |
| `.env` | 端口、数据路径、fs/BPM/PPI/HRV参数 | dotenv/Compose读取，间接控制Dash/HRV | 敏感；值与hash不记录 |
| `.env.example` | 非敏感配置模板 | 用户复制为`.env` | 旧示例路径；`DEDUPLICATE_MS`空值 |
| `.gitignore` | Git排除规则 | 影响索引，不产生项目输出 | 未统一排除results、pkl/joblib等 |
| `AGENTS.md` | Agent治理、先审后写、diff/review规则 | 引用`_agent/WRITE_RULES.md` | 本轮只读；final_v0例外来自用户明确授权 |
| `detector_v8_calib.npz` | v8校准容器：mu/sd、IMU bias、anchor、win/hop/method/bundle path | 由校准Notebook风格流程写、Dash读取 | 3,900 bytes/12 members；未读数组；provenance未完全证实 |
| `docker-compose.yml` | 构建ppg-app、`.env`、端口、整仓bind mount | 产生运行容器 | `${PWD}`平台依赖且受Dockerfile CMD错误阻断 |
| `Dockerfile` | Python3.9 slim、pip install、copy、gunicorn | 读取pyproject/README/全context，输出image | `sh -c` JSON CMD参数语义错误，gunicorn不能按预期启动 |
| `LICENSE` | 标准MIT许可证 | 无算法I/O | 有效文本 |
| `pttppg_detector_v8_scores_audit_fix9.py:Zone.Identifier` | Windows下载来源附件 | 对应同名脚本；不参与运行 | 333 bytes；不复制URL/查询参数 |
| `pttppg_pipeline_v7_4_noleak_viz_ae.py:Zone.Identifier` | 同上 | 对应v7.4脚本 | 329 bytes；仅provenance |
| `pttppg_stage2_denoiser.py:Zone.Identifier` | 同上 | 对应Stage-2脚本 | 333 bytes；仅provenance |
| `pyproject.toml` | Hatchling元数据、Python/依赖、CLI | Docker/pip install读取，生成package/CLI | CLI指向`.ipynb`不可导入；依赖缺frailty/ASA/SVM栈；Python精确锁定过严 |
| `README.md` | 旧PPG frailty/Conda/.env/Notebook说明 | 指向`.env.example`和旧Notebook | 已过时；未覆盖frailty/ASA/denoiser/sweep/holdout；Python版本冲突 |
| `Script_struc.txt` | 理想目录结构与2025历史备注 | 无I/O | 与当前平铺结构不一致，不可作真实结构依据 |

## 5. 完整性和使用结论

1. 45个根文件均已列名；文本/代码均完整读取，非文本按用户规则登记结构。
2. 根代码与实际结果的精细对应分散于本清单、M0 crosswalk及project-wide findings；不明确者标为历史、代理或待确认。
3. 目录存在不等于run完成；frailty/ASA的完整性必须按manifest、runs、summary和scorecard联合判定。
4. 后续每个TODO开始仍需重扫相关文件；本清单是2026-08-02基线，不代替未来变更检测。

