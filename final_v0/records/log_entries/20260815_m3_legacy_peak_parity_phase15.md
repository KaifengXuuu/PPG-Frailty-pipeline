# M3 legacy peak parity phase 15 / M3 历史峰算法一致性第 15 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_evidence_persisted
- 流程 / Process：逐字节读取根目录 funcs.py、ppg.py 与 frailty_3class_classifier.py，通过 AST 白名单隔离执行所需函数，避免导入 UI/训练依赖及模块副作用。
- 算法 / Algorithm：对同一 400 Hz 固定 PPG fixture 比较 funcs/ppg 重复 Aboy++、classifier 的 aboypp_detect_peaks 及 detect_ppg_peaks alias；结果按 int64 peak 序列哈希冻结。
- 结果 / Result：funcs.py 与 ppg.py 36 峰逐值完全一致；classifier alias 35 峰完全一致；两类历史实现不等价，classifier 相比 funcs/ppg 少 index 8318。新增 2 项测试后全量 40/40 通过。
- 判定 / Decision：差异不是 corrected_v1 失败，而是论文/复现必须保留的历史实现分叉；future-active 入口仍唯一指向 m3_signal_core。
- 证据 / Evidence：M3 evidence/legacy_peak_parity.json，并已登记 M3_BUILD_REPORT.json。
