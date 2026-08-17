## 2026-08-02 — M0 算法图与逐脚本图册

- 写入：项目历史信号总图、基础函数图、v7→Stage-2演化图、Hybrid套件图、Heartbeat/Motion A-B图、16入口逐脚本图册。
- 图示约定：实线为运行数据流；虚线为监督、评价、风险或审计引用；阻断和空结果直接落在相应算法节点旁。
- 自动维护：新增 `sync_algorithm_index.py` 与 `verify_algorithm_diagrams.py`，用于重建图索引并检查 Mermaid fence、图类型和脚本覆盖。
- 边界：图是历史代码结构与证据的审计表示，不代表历史算法被批准恢复。

