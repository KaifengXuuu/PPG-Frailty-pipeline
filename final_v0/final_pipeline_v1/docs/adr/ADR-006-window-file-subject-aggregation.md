# ADR-006: Window→file→role-aware participant aggregation / 分层聚合

- 状态 / Status: accepted_for_v1
- 依据 / Source: contract §§3, 5.7, 5.11, 5.13, 8.3

## Decision / 决策

参考聚合严格为：

```text
window probabilities -> file probability -> role probability -> participant probability
```

Window→file 默认 ordinary mean；只有验证通过且预注册的 quality weighting 才可使用
质量权重。File→role 对同一 role 的重复记录等权平均；role→participant 对当前
配置允许且实际可用的 role 等权平均。缺失 role 不以零补齐，而是报告 coverage。

Feature-vector/matrix 路线从 file 层进入；fusion 在 raw windows 已池化为一个 file
embedding 后，只拼接一次完整 file feature vector。禁止把 file vector 重复给每个
raw window。

## Named ablations / 命名消融

- `direct_all_window_subject_mean_legacy`: 历史敏感性对照，禁止成为 reference；
- `quality_weighted_window_to_file`: 仅训练折拟合/冻结权重后允许；
- `attention_file_pooling`: 与 mask-aware mean 独立比较。

Role 等权是否应改为协议预定权重列入 V2 人工确认点。

