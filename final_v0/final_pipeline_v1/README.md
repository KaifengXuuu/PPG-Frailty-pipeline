# PPG Frailty Final Pipeline V1

状态 / Status: **engineering acceptance checkpoint passed; scientific benchmark not run**

本目录是依据 `CODEX_IMPLEMENTATION_SPEC_PPG_FRAILTY_DEV0_MERGED.md` 建立的独立、
可运行、可审计实现。它不修改根目录历史脚本，不覆盖历史结果，也不把尚未通过测试的
模块描述为已验证性能。

This directory is the isolated, runnable, and auditable implementation of the merged
dev0 contract. It neither modifies historical root scripts nor overwrites historical
results, and it never presents an untested module as validated performance.

## Two-axis status / 双轴状态

- **Engineering / 工程：current acceptance checkpoint passed.** Frozen data/folds, typed signal/SQI/artifact/feature routes, four representations, thirteen model routes, training/evaluation/OOF/bundle contracts, public comparisons/ablations, and strict CPU gates are present.
- **Science / 科学：benchmark not completed.** No full 5 repeats × 5 folds candidate matrix, independent Frailty3 test, external-PTT reducer ranking, or mobile/ONNX parity result is claimed.

工程门通过只证明实现、协议和证据边界可复核；它不证明模型性能、临床有效性或跨设备泛化。最新边界与机器证据见 [STATUS.md](STATUS.md)。

## Frozen boundaries / 冻结边界

- 三分类标签：`Pre-Frail`、`Robust/Non-Frail`、`Young`。
- 外层分组单位：participant；任何拟合对象不得使用外层 held-out 数据。
- 内部 acquisition/audit/morphology 时间轴：400 Hz。
- 非恒等去伪影输出 `x_ar` 只用于 rate recovery；`Q_morph=not_applicable`。
- 权威规格：41,122 bytes、766 lines、SHA-256
  `cd7c4907a8beccd301048dea07ae6fdc4e9d2dc839759cb8dddd4a461e3c5000`。

## Quick start / 快速开始

```bash
cd final_v0/final_pipeline_v1
export PYTHONPATH="$PWD/src"
export PYTHONDONTWRITEBYTECODE=1

python3 -B -m ppg_frailty.cli --help
python3 -B -m ppg_frailty.cli list-modules --family all
python3 -B -m ppg_frailty.cli validate --all-configs
python3 -B -m ppg_frailty.cli test --suite all --verbosity 1
python3 -B tools/acceptance_gate.py
```

完整依赖配置、输出路径、所有并行 reducer/model、EKF 对 LPF、消融与失败语义见 [RUNBOOK.md](RUNBOOK.md)。

## Public evidence boundary / 公共证据边界

| Route / 路线 | Current proof / 当前证明 | Not proved / 尚未证明 |
|---|---|---|
| `run --mode smoke|full` | Frozen real-input and protocol integration | Trained classifier performance |
| `run-experiment --budget reduced-smoke|full` | Real frozen outer-fold training/evaluation; current formal executor is feature_vector-only | Reduced or selected-cell output is not the complete 5×5 benchmark |
| `compare artifacts` | Synthetic execution of identity and six non-identity reducers | External-PTT ECG-reference ranking |
| `compare models` | Reduced synthetic execution of all thirteen model routes | Frailty3 leaderboard |
| `compare imu-gravity` | Synthetic known-truth ESKF/LPF comparator | 29-subject human-motion validity |
| `ablate` | One-factor execution/schema contract | Frozen 5×5 outcome superiority |
| strict acceptance / CPU CI | Current source, imports, CLI, tests, claim boundaries | Medical validity or independent generalization |

## Real outer-fold experiment / 真实 outer-fold 实验

`run-experiment` is the training/evaluation entry; `run` remains input/protocol audit only.
The passing public reduced example must use the feature-vector motion config:

```bash
python3 -B -m ppg_frailty.cli run-experiment   --config configs/motion_benchmark_v1.yaml   --budget reduced-smoke   --repeat 0   --fold 0   --output-dir artifacts/experiments/my_reduced_r0_f0
```

The public reduced budget fixes 60 seconds, one record per participant, and one epoch-equivalent
training budget while retaining the complete frozen participant roster. Current formal runner
dispatch is **feature_vector-only**: `reference_static_v1.yaml` (raw),
`feature_matrix_v1.yaml`, and fusion routes fail closed rather than silently changing
representation. Omitting repeat/fold with `--budget full` requests all 25 cells; an explicit
pair requests one full-length cell. Neither a reduced nor selected single-cell result is the
complete 5×5 benchmark.

公共 reduced 固定 60 秒、每 participant 一份记录及一轮训练预算，但不缩减冻结名单。
当前正式 runner 仅支持 feature_vector；其他表征传入时关闭失败，不能把模块/合成合同
覆盖写成真实 5×5 结果。

## Package and report navigation / 包与报告导航

- [Current dual-axis status / 当前双轴状态](STATUS.md)
- [Copyable operational runbook / 可复制运行手册](RUNBOOK.md)
- [Algorithm workflows / 算法流程图](docs/algorithms/README.md)
- [Specification vs TODO / 规范与 TODO](docs/comparisons/01_SPEC_VS_TODO_OVERLAP_AND_DIFFERENCES.md)
- [Specification vs completed M0–M3 / 规范与已完成 M0–M3](docs/comparisons/02_SPEC_VS_COMPLETED_TODO.md)
- [Specification vs local frozen workflow / 规范与本地冻结 workflow](docs/comparisons/03_SPEC_VS_LOCAL_FROZEN_WORKFLOW.md)
- [Algorithm reasonableness / 算法合理性](docs/comparisons/04_ALGORITHM_REASONABLENESS_AND_TRADEOFFS.md)
- [V1→V2 confirmation summary / V2 确认摘要](docs/comparisons/05_V1_TO_V2_CONFIRMATION_SUMMARY.md)
- [Detailed V2 points / 详细人工确认点](records/v2_decision_points/HUMAN_CONFIRMATION_POINTS.md)
- [Model cards / 模型卡](model_cards/README.md)
- [Generated detailed tree / 自动详细文件树](PROJECT_TREE.md)
