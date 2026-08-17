# M3 D8 symmetric scorecard phase 21 / M3 D8 对称评价第 21 阶段

- 时间 / Date：2026-08-15
- 状态 / Status：implemented_and_reference_tested
- 流程 / Process：复扫 PTT evaluator 后，将只在顶层输出 corrected PPI/HR 的非对称结果改为 raw 与 delay-corrected 两个同 schema scorecard。
- 算法 / Algorithm：两分支均报告 one-to-one precision/recall/F1、timing samples/ms、PPI MAE、HR error、coverage 与 failure；delay artifact 新增 dataset、training split、preprocessing profile 与 algorithm provenance。
- 结果 / Result：合成训练/独立评价 fixture 中 raw 因 200 ms 生理延迟在 50 ms 门下无匹配，corrected F1=1、timing/PPI/HR error=0；全量 reference tests 42/42 通过。
- 边界 / Boundary：这是公式与无泄漏合同验证；真实 PTT OOF benchmark 留给 M4/M5，不能把 ECG detector preflight 当作 PPG peak 成绩。
