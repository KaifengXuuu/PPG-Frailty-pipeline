# B/R/S/W 阶段映射与部分时序合同

## 用户已确认语义

| 文件 token | 阶段语义 | activity state | 确认状态 |
|---|---|---|---|
| `B` | baseline | static | confirmed |
| `R1..R4` | relax/recovery | static | confirmed at family level |
| `S1,S2` | stand-and-sit | motion | confirmed at family level |
| `W1,W2` | walk | motion | confirmed at family level |

## 只允许的时序关系

当前只冻结下列部分顺序：

```text
S-family ─┐
          ├── precedes ──> R-family (Relax/Recovery)
W-family ─┘
```

该关系允许后续定义运动后恢复特征，但不代表已知全部采集 protocol。

## 必须保持未知

- R1–R4 的编号是否表示连续时间点、不同动作后的 recovery、重复次数或其他含义：`unverified`。
- S1/S2、W1/W2 的编号含义、两者之间先后关系及重复含义：`unverified`。
- S 与 W 谁先、是否交替、B 相对每个动作的精确时间位置：`unverified`。
- 文件名中其他 `_01.._04` token 的实验语义：`unverified`；它属于 subject/base identifier，不得当作 stage time index。

## 下游约束

Motion detector 可用 B/R 对 S/W 监督 activity，但不得把该标签当作 SQI 真值。Recovery feature 必须带 `feature_available_after_stage`，只在相应 R 数据完成后可用；禁止把未来 R 段信息回填到先前窗口。
