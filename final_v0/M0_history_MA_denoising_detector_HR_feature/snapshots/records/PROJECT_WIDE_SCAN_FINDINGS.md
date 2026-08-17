# Workspace 全项目扫描发现 / Project-wide Scan Findings

- 状态 / Status：`baseline_complete; future_TODO_items_not_executed`
- 目的 / Purpose：保存 M0 前置全量扫描中发现、但属于 M1–M10 的事实，避免后续重复扫描后遗失上下文。
- 边界 / Boundary：以下不是对 M1–M10 的完成声明，也没有修改或运行对应代码。

## 1. 全局文件与证据覆盖

| 范围 | 文件数 | 字节数 | 读取策略 | 状态 |
|---|---:|---:|---|---|
| Workspace（排除 `.git`、`final_v0`） | 35,214 | 42,794,025,593 | 全文件树元数据 | 完成 |
| 根目录文本/代码 | 41 | 已纳入根文件总数45 | 逐字节至EOF+SHA | 完成 |
| 全代码/notebook | 52 | — | 逐字节至EOF+SHA | 完成 |
| 输入目录 | 6,405 | 34,581,621,834 | 每文件头部≤65,536 bytes+结构 | 完成 |
| 输出目录 | 28,670 | 8,165,668,577 | 文本EOF；二进制名称/格式/大小 | 完成 |

## 2. Frailty3 历史结果（供 M5–M8）

- `results_frailty3`：14,496 文件，其中8,974文本已完整读取，5,522二进制已登记。
- 历史 overfitting sweep 中 InceptionTime 曾出现 balanced accuracy 约 `.737`；审计显示使用validation/early-stopping选择，不能作为独立最终性能。
- 固定epoch参考约 `.623`，说明移除选择优势后性能下降。
- 已有15-subject independent holdout 的 rank-2候选 balanced accuracy `.5444`，bootstrap CI约 `[.4634,.6255]`；它与旧 `.737` 不是同一证据层。
- 论文主结果应选择“独立holdout”还是“旧sweep探索值”属于人工研究叙事决策，不能由扫描器替用户决定。

## 3. ASA 结果（供 M5/M8）

- `test_asa_classifier`：12,642 文件，其中153文本完整读取、12,489二进制登记。
- 最终OOF threshold策略：balanced accuracy约 `.4728`，macro-F1约 `.4667`，ASA3 recall约 `.3016`。
- argmax基线出现预测类别坍缩，未预测ASA2；因此不得只报告overall accuracy或隐藏每类召回。
- 是否把OOF threshold版作为论文唯一ASA结果，并如何表述argmax collapse，需要后续人工确认。

## 4. SVM/缓存/schema（供 M5–M7）

- `train_raw`、`train_labeled`、`train_val`、`train_window` 中识别出45、92、116列等不兼容schema；不能静默拼接。
- 当前 SVM 训练脚本存在语法/import或字段契约风险，且未发现可与代码无歧义对应的正式scorecard。
- 存在旧28-subject缓存与新29-subject缓存并存；是否将旧缓存整体标记legacy，必须在数据合同和论文cohort定义后决定。
- M6必须先生成统一manifest、group split和feature registry，再运行模型；不能沿用“目录名看起来相同”的假设。

## 5. ShapeFormer-PISD 与外部依赖（供 M5）

- ShapeFormer移植脚本依赖仓库外硬编码路径/外部实现。
- 在不允许恢复或封装该依赖时，它不能满足 M1 的offline CPU重跑合同。
- 是否保留为正式候选、降级为历史实验或在 `final_v0` 重写最小自包含版本，需要用户选择。

## 6. 输出目录版本与完整性

- 多个结果目录包含重复、历史或中断批次；仅凭目录名不能判断“最新”或“完成”。
- `results_denoiser_v8`、`results_stage2` 是明确空目录。
- `.CNN_results` 与 `results_frailty3` 中 run 数量多，必须按 run-level config/code hash/data split 选取，不能跨run混合。
- M1需先定义“完整run”的最小文件集；M8再按该规则生成论文表格。

## 7. 安全读取边界

- 输出中的Pickle/PT/ONNX/NPZ等按用户规则只登记非文本文件名/格式/大小，没有反序列化未知对象。
- 已扫描到大量Pickle类元数据；如果后续必须读取内部内容，应优先寻找JSON/CSV等并行文本，或由用户授权受控反序列化。
- `.env` 值没有写入 `final_v0`，只有键名、字节校验与脱敏标记。

## 8. 后续使用规则

1. 每个新 TODO 开始前重新扫描相关源文件、对应输入头部与输出manifest，不能把本基线直接当作当前状态。
2. 本文中的数值是导航线索；正式论文值必须回到具体run文件和协议确认。
3. 任何需要用户选择的路线均进入 `HUMAN_DECISION_GATES.md`；未确认前不得实施会固定研究口径的写入。

