# M0 v3 文件树与逐文件说明 / M0 v3 Tree and Per-file Descriptions

> 由 `final_v0/tools/build_m0_history_package_v3.py` 自动生成；本索引不自哈希。

## 树状结构 / Tree

```text
M0_history_MA_denoising_detector_HR_feature/
├── evidence
│   ├── current_binary_detector_balanced_v2
│   │   ├── A_external_motion_confusion_matrix.png
│   │   ├── A_holdout_motion_confusion_matrix.png
│   │   ├── B_external_motion_confusion_matrix.png
│   │   ├── B_holdout_motion_confusion_matrix.png
│   │   ├── config_summary.json
│   │   ├── detector_ab_f1_comparison.png
│   │   └── detector_benchmark_summary.json
│   ├── current_binary_detector_smoke
│   │   ├── A_external_motion_confusion_matrix.png
│   │   ├── B_external_motion_confusion_matrix.png
│   │   ├── config_summary.json
│   │   └── detector_benchmark_summary.json
│   ├── EARLY_MULTICLASS_SEARCH_AUDIT.json
│   └── MOTION29_DATA_AUDIT.json
├── snapshots
│   ├── algorithm_diagrams
│   │   ├── 00_PROJECT_HISTORICAL_SIGNAL_FLOW.md
│   │   ├── 01_FOUNDATION_FUNCS_PPG.md
│   │   ├── 02_V7_TO_STAGE2_EVOLUTION.md
│   │   ├── 03_HYBRID_SUITE.md
│   │   ├── 04_HEARTBEAT_AND_MOTION_AB.md
│   │   ├── 05_SCRIPT_ALGORITHM_ATLAS.md
│   │   ├── 06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md
│   │   ├── 07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md
│   │   └── 08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md
│   ├── records
│   │   ├── decisions
│   │   │   ├── 20260803_m0_activity_motion_supervision.md
│   │   │   └── 20260803_m0_madenoiser_route.md
│   │   ├── ARCHIVED_CODE_IO_INVENTORY.md
│   │   ├── CODE_IO_MASTER_INDEX.md
│   │   ├── HUMAN_DECISION_GATES.md
│   │   ├── M0_ARCHIVED_LINEAGE_EVIDENCE.md
│   │   ├── M0_CODE_OUTPUT_CROSSWALK.md
│   │   ├── M0_EXECUTIVE_REPORT.md
│   │   ├── M0_METHOD_REGISTRY.md
│   │   ├── M0_PAPER_EVIDENCE.md
│   │   ├── M0_RISK_REGISTER.md
│   │   ├── PROJECT_WIDE_SCAN_FINDINGS.md
│   │   ├── ROOT_FILE_IO_INVENTORY.md
│   │   └── SCAN_PROTOCOL.md
│   └── verification
│       ├── inputs
│       │   ├── physionet.org.summary.json
│       │   └── PPG_Testing_05_01_2026.summary.json
│       ├── outputs
│       │   ├── CNN_RESULTS.summary.json
│       │   ├── denoiser_preview_output.summary.json
│       │   ├── results.summary.json
│       │   ├── results_denoiser_v8.summary.json
│       │   ├── results_frailty3.summary.json
│       │   ├── results_hybrid_denoiser.summary.json
│       │   ├── results_hybrid_denoiser_raw_imu.summary.json
│       │   ├── results_hybrid_denoiser_raw_imu_baseline.summary.json
│       │   ├── results_stage1.summary.json
│       │   ├── results_stage2.summary.json
│       │   ├── results_v72_noleak.summary.json
│       │   ├── results_v7_4.summary.json
│       │   └── results_v8_audit.summary.json
│       ├── ALGORITHM_DIAGRAM_VERIFICATION.json
│       ├── ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V2.json
│       ├── ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V3.json
│       ├── BASELINE_SUMMARY.json
│       ├── CODE_DIAGRAM_COVERAGE.json
│       ├── CODE_FILES.jsonl
│       ├── CODE_PATH_REFERENCES.jsonl
│       ├── ROOT_FILES.jsonl
│       ├── SCAN_RUNS.jsonl
│       ├── SCAN_VERIFICATION.json
│       └── TOP_LEVEL_DIRECTORIES.json
├── 00_CURRENT_STATUS_V3.md
├── 01_M0_COMPLETE_RESULTS_AND_DECISIONS.md
├── 02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md
├── 03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md
├── 04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md
├── 05_EVIDENCE_INDEX_AND_PROVENANCE.md
├── 06_M0_PACKAGE_TREE.md
├── 07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md
├── 08_M0_PACKAGE_TREE_V2.md
├── 09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md
├── 10_M0_PACKAGE_TREE_V3.md
├── M0_PACKAGE_VERIFICATION.json
├── M0_PACKAGE_VERIFICATION_V2.json
├── M0_PACKAGE_VERIFICATION_V3.json
├── M0_SOURCE_SNAPSHOT_MANIFEST.json
├── M0_SOURCE_SNAPSHOT_MANIFEST_V2.json
├── M0_SOURCE_SNAPSHOT_MANIFEST_V3.json
└── README.md
```

## 完整性与内容 / Integrity and content

| 文件 / File | 字节 / Bytes | SHA-256 | 内容 / Content |
|---|---:|---|---|
| `00_CURRENT_STATUS_V3.md` | 2434 | `17d35975bd1ae6e717e555a14ca1877306e3ccc9582f725aee183fd7d47900b1` | Markdown《M0 当前状态 v3：Activity/Motion 监督已确认》 |
| `01_M0_COMPLETE_RESULTS_AND_DECISIONS.md` | 14337 | `af2c221f18aa6740149ab923454961ea354e003a9303942528e8dfb0fd1e4ead` | Markdown《M0 完整结果、算法结论与路线决定》 |
| `02_CANDIDATE_SCRIPTS_MOTION_DENOISING_DYNAMIC_HR.md` | 21353 | `adae474bcff301454b75a0b9872c04ba45614e54a8712cd48aa916f4481630cc` | Markdown《Motion Detector、Denoising 与动态 HR 候选脚本总表》 |
| `03_FIVE_METHOD_FAMILIES_CODE_TEST_IMPLEMENTATION_AUDIT.md` | 29845 | `99a728f93c38b159a4b430b21c27d1427d1ce9886f86dbd58e96842556fa031c` | Markdown《五类方法：代码、算法、应用、测试与可实现性审计》 |
| `04_UNIFIED_IMPLEMENTATION_AND_BENCHMARK_CONTRACT.md` | 15105 | `ea1f3f1ee7a7d9778e9fd7cd06a4ea189ca3a290416a6f2f5684653bb5f11b1e` | Markdown《五类路线统一实现与 Benchmark 合同》 |
| `05_EVIDENCE_INDEX_AND_PROVENANCE.md` | 11465 | `7de41e07b9abaf3af10a91110e44597222f587a10720f2dd11b2e68e9c91f91d` | Markdown《M0 证据索引与来源链》 |
| `06_M0_PACKAGE_TREE.md` | 15519 | `f64eaf6447835f74852ac0eae8d16585a3c6ce7b8095ba78a9dc03c30f8344ba` | Markdown《M0 历史归档文件树与逐文件说明 / M0 Package Tree and Per-file Descriptions》 |
| `07_CONFIRMED_MADENOISER_FOLLOWUP_ROADMAP.md` | 12054 | `84b5ee74268985ab0871699b90d36bdc1c378e84b7ac25e3d272b3afc9a211da` | Markdown《MAdenoiser 已确认后续路线：从 SQI、Motion-29 到 Frailty 特征选择》 |
| `08_M0_PACKAGE_TREE_V2.md` | 17644 | `d41d1599dbcdf2844e343ac85a3d4a4e49ba261ceefe17b97d2c5adadca64e3e` | Markdown《M0 历史归档文件树与逐文件说明 / M0 Package Tree and Per-file Descriptions》 |
| `09_ACTIVITY_MOTION_SUPERVISION_THREE_CLASS_HISTORY_AND_RECOVERY.md` | 16700 | `81b82f55729b5f66ce4b6f977e249205ca5d03a37c5a2c728275c2986d800928` | Markdown《29 人 Activity/Motion 监督合同、早期三分类历史与运动后恢复特征路线》 |
| `M0_PACKAGE_VERIFICATION.json` | 302 | `0c580a7b316d9cfb34e2920fb3bf5d7907f3dad531766b9735e68752cb2efb7c` | 机器JSON；顶层字段=algorithm_diagram_file_count,mermaid_block_count,missing_documents,package_id,required_document_count,snapshot_count,snapshot_failures,snapshot_total_bytes |
| `M0_PACKAGE_VERIFICATION_V2.json` | 302 | `83fd3894c5060a5bb243359f31a40378c080157d1526c6f38a5e8d7c11bb90fb` | 机器JSON；顶层字段=algorithm_diagram_file_count,mermaid_block_count,missing_documents,package_id,required_document_count,snapshot_count,snapshot_failures,snapshot_total_bytes |
| `M0_PACKAGE_VERIFICATION_V3.json` | 319 | `8b04a1f0dbab74197221c6cdb26639b452d47a2ff8fbccba18babc0a496243f1` | 机器JSON；顶层字段=algorithm_diagram_file_count,mermaid_block_count,missing_documents,package_id,required_document_count,snapshot_count,snapshot_failures,snapshot_total_bytes |
| `M0_SOURCE_SNAPSHOT_MANIFEST.json` | 16815 | `4e09ca1f82c97cf82b4b63c0f03884a0e4338237e5fec21af0fbc6811532ba07` | 机器JSON；顶层字段=failures,large_manifest_policy,package_id,snapshot_count,snapshot_total_bytes,snapshots,status |
| `M0_SOURCE_SNAPSHOT_MANIFEST_V2.json` | 17655 | `ec86d163cfd78b46d024f8ad875cc6fb4c51dd01c89cacb3ce847da88b12f551` | 机器JSON；顶层字段=failures,large_manifest_policy,package_id,snapshot_count,snapshot_total_bytes,snapshots,status |
| `M0_SOURCE_SNAPSHOT_MANIFEST_V3.json` | 18432 | `5225b2653112c4595541bc8d72226d43dd44e6a1ea6962ddddd8517dff28cc7b` | 机器JSON；顶层字段=base_manifest,failures,package_id,snapshot_count,snapshot_total_bytes,snapshots,status |
| `README.md` | 6152 | `1571c8ba8b19f7afc41bce1aae011b61840cdb55ac9f81dc49810c5b20b1de2f` | Markdown《M0 历史 Motion Artifact、降噪、检测器与动态 HR 归档》 |
| `evidence/EARLY_MULTICLASS_SEARCH_AUDIT.json` | 1780 | `a7f9e6e9431cc6d7921a3081927ea1cacbf4814233d1ac4fb1b541cd0a667ad8` | 机器JSON；顶层字段=audit_date,status,scope,three_class_datasets,svm_models,verified_qualitative_history,not_found,paper_wording |
| `evidence/MOTION29_DATA_AUDIT.json` | 2465 | `14cac2f40806b8d53d2438186799de2a1558e1b2e38e94a11b7f9f5b120b1c5b` | 机器JSON；顶层字段=audit_date,status,sampling_rate_assumed_by_project_hz,datasets,combined,duration_seconds,supervision,sequence |
| `evidence/current_binary_detector_balanced_v2/A_external_motion_confusion_matrix.png` | 39752 | `4151bbe6a161078f01560958afe9d038165ec26d2b2b827222983ceacdfb04de` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_balanced_v2/A_holdout_motion_confusion_matrix.png` | 33922 | `1db8c723594c0953be349786514f509bb3c7e417bfe57032bf4f62a21e20115d` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_balanced_v2/B_external_motion_confusion_matrix.png` | 39449 | `bcdb9e8d3a293ea6f52cc20bda715be44ca368fdac312b4d749ef375059ba956` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_balanced_v2/B_holdout_motion_confusion_matrix.png` | 32856 | `8980dfb801ec2cca3f6c3247aeaacd0af29f936dbd2ca27a47bced22b9c9a74c` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_balanced_v2/config_summary.json` | 55902 | `3b6d6cf3f4f0e0cef1c7d33ce3e01dc4379bfeb761d4408d6fd716cc02b1e3ba` | 机器JSON；顶层字段=results_root,run_name,cv_folds,final_train_epochs,target_fs,win_sec,hop_sec,batch_size |
| `evidence/current_binary_detector_balanced_v2/detector_ab_f1_comparison.png` | 31849 | `4f75d3969a25e7dd0b526ae76f55d520b83551e9663756e46c44a59a7a23cd6b` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_balanced_v2/detector_benchmark_summary.json` | 29140 | `0a34ae2656aac00a3cdeb789e582981745c5322ba90d3cf36d108ae925c8f7a0` | 机器JSON；顶层字段=status,reason,detector_input,split_counts,models |
| `evidence/current_binary_detector_smoke/A_external_motion_confusion_matrix.png` | 38584 | `8b6df160dc7695b92c9cc057cda8f39c019fc8e58bddea25c0dee3f5f79471a4` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_smoke/B_external_motion_confusion_matrix.png` | 32815 | `dea2591f62ae90555a8b035f47898b36ba19ca1debf6aaedc2df29fc27155744` | 历史结果PNG证据；内容与来源见专题和证据索引 |
| `evidence/current_binary_detector_smoke/config_summary.json` | 5518 | `80fe885e246e3381dca0d6b0977e596417b3c7e62259ed4502aa5f47d885ce3b` | 机器JSON；顶层字段=results_root,run_name,cv_folds,final_train_epochs,target_fs,win_sec,hop_sec,batch_size |
| `evidence/current_binary_detector_smoke/detector_benchmark_summary.json` | 26326 | `8eeeab19041d9738db9c9e976cbdf02f4dd3aaf3b245406da21f15137a27dabc` | 机器JSON；顶层字段=status,reason,detector_input,split_counts,models |
| `snapshots/algorithm_diagrams/00_PROJECT_HISTORICAL_SIGNAL_FLOW.md` | 2684 | `1ed28714c1f7f46a779f259a31f940dc26e18deaf4c08e3b20e5aea1caf5a9e4` | Markdown《项目历史信号处理总图 / Historical Signal-Processing Map》；Mermaid图块=1 |
| `snapshots/algorithm_diagrams/01_FOUNDATION_FUNCS_PPG.md` | 1807 | `c8586eb04bfe58fc3ff56f09e1d0221da600aa853312144ad3cc03638018f431` | Markdown《M0 基础函数与 Dash 算法图 / Foundation Functions and Dash Flow》；Mermaid图块=2 |
| `snapshots/algorithm_diagrams/02_V7_TO_STAGE2_EVOLUTION.md` | 3068 | `53d16384e83b8ce98ffafd2dc3107be803f5dee76a19a5a2f0175d5e981efb88` | Markdown《v7 至 Stage-2 演化图 / v7-to-Stage-2 Evolution》；Mermaid图块=4 |
| `snapshots/algorithm_diagrams/03_HYBRID_SUITE.md` | 2256 | `de5bb69fc9fe0d2513986380f2c9d87b9ef88aa53b631af56a6311a1b671944f` | Markdown《Hybrid 去噪、导出与运行图 / Hybrid Denoiser Suite》；Mermaid图块=3 |
| `snapshots/algorithm_diagrams/04_HEARTBEAT_AND_MOTION_AB.md` | 2515 | `1d9a75b196e673209d6867468a90e96bb6c99c4893deb62e6690822582cae835` | Markdown《Heartbeat 与 PPG+IMU Motion A/B 图 / Heartbeat and Motion A/B》；Mermaid图块=3 |
| `snapshots/algorithm_diagrams/05_SCRIPT_ALGORITHM_ATLAS.md` | 6934 | `d6681b04fc2c0235be1669ef026001774ddee3f681056e47b77e3629963685d4` | Markdown《M0 逐脚本算法结构图册 / Per-script Algorithm Atlas》；Mermaid图块=16 |
| `snapshots/algorithm_diagrams/06_FIVE_METHOD_FAMILIES_IMPLEMENTATION_AND_BENCHMARK.md` | 5558 | `dec93e8d4901361ec78c512658ca08ce9bc17657851a7a0d9d3036a40aac1a12` | Markdown《M0 五类方法、统一实现与 Benchmark 算法图》；Mermaid图块=6 |
| `snapshots/algorithm_diagrams/07_MADENOISER_ROUTE_TO_FRAILTY_FEATURE_SELECTION.md` | 3042 | `1cb90ecec9db32867f424c3db82aac91bac53281e075067a26807119690c5905` | Markdown《MAdenoiser 已确认路线到 Frailty 特征选择算法图》；Mermaid图块=4 |
| `snapshots/algorithm_diagrams/08_ACTIVITY_MOTION_TRANSFER_AND_RECOVERY_FEATURES.md` | 2776 | `3dafee689821990a7318cb2b4c5510a99623e8c4d2855107324bef59da2b7e3a` | Markdown《Activity/Motion 迁移重训、SQI 与恢复特征流程图》；Mermaid图块=3 |
| `snapshots/records/ARCHIVED_CODE_IO_INVENTORY.md` | 6837 | `cf81c62086fa40812905f28ec7c27e52ad67409e69e699b390e7183345ded08a` | Markdown《非根归档代码逐文件 I/O 与版本关系 / Archived Code I/O and Lineage Inventory》 |
| `snapshots/records/CODE_IO_MASTER_INDEX.md` | 3761 | `78e60dda2d6d166b5e90e0e861ee47a2e1137f066adbc192bfcbd867fce95938` | Markdown《52份代码/Notebook I/O 总索引 / Master Code and Notebook I/O Index》 |
| `snapshots/records/HUMAN_DECISION_GATES.md` | 3170 | `093a37653cde9afba230c8dcd1a4a3e8f2cb1aa2ff81b500620d8a9f66eff4dc` | Markdown《人工决策门 / Human Decision Gates》 |
| `snapshots/records/M0_ARCHIVED_LINEAGE_EVIDENCE.md` | 3106 | `17e5f4b33e2d225dc72a2e9a42ae5c7e592030023debbcce2ced7182ac6cc331` | Markdown《M0 归档版本与实际输出生产关系 / Archived Lineage and Output Provenance》 |
| `snapshots/records/M0_CODE_OUTPUT_CROSSWALK.md` | 9419 | `432718ff0f46c140e3a0ca6729d6e9548fb822be2520d00c0232f65e4d89a159` | Markdown《M0 代码—输入—输出对应表 / Code–Input–Output Crosswalk》 |
| `snapshots/records/M0_EXECUTIVE_REPORT.md` | 9014 | `cc297c4e12f68d0858f0b53591d299708993f402085f8a5fba809c00d4bda0d6` | Markdown《M0 执行、算法与结果总报告 / M0 Execution, Algorithm, and Results Report》 |
| `snapshots/records/M0_METHOD_REGISTRY.md` | 17119 | `7b4e4f13bebc964fcf2ebe87a258d08629c2922c9ad7895756712d63944f3cd6` | Markdown《M0 Motion Processing Method Registry》 |
| `snapshots/records/M0_PAPER_EVIDENCE.md` | 5977 | `c2e6c988bd827a500dbdca97b1dd6be13cc036088def9d45167200b13b5a70f1` | Markdown《M0 论文证据、结果评价与表述边界 / Paper Evidence and Claim Boundaries》 |
| `snapshots/records/M0_RISK_REGISTER.md` | 7541 | `14d212d20bff712802b55696918cdb34fb7d37da94366055f645ec83c192e540` | Markdown《M0 风险登记 / Risk Register》 |
| `snapshots/records/PROJECT_WIDE_SCAN_FINDINGS.md` | 4315 | `a38e5401967d3b56034ea283dc9bc069f70fa2afc9fd3ad3dfbfe3b090dcf2d6` | Markdown《Workspace 全项目扫描发现 / Project-wide Scan Findings》 |
| `snapshots/records/ROOT_FILE_IO_INVENTORY.md` | 11314 | `9a1550d161a251839187d41f00ccfe0d2022be19f3f0f6e79c872655effdae2e` | Markdown《根目录逐文件 I/O 与内容清单 / Root-file I/O and Content Inventory》 |
| `snapshots/records/SCAN_PROTOCOL.md` | 2143 | `82444b736ceedac4293f62ebee2d2e59c01394986a367faef0b8fca13e749180` | Markdown《扫描协议与证据要求 / Scan Protocol and Evidence Requirements》 |
| `snapshots/records/decisions/20260803_m0_activity_motion_supervision.md` | 1905 | `f4f10640e4926a70b9e470bcf33f65aafa22b2ca3534f0b499abbc1495e75580` | Markdown《M0-MOT-001 — 29-subject Activity/Motion 监督与时序特征决定》 |
| `snapshots/records/decisions/20260803_m0_madenoiser_route.md` | 1834 | `ca93169b3a0167598ddebbaa7c8d0d95ee8ff50f30703c2ae6791d0653ac57a3` | Markdown《M0-MAD-001 — MAdenoiser 后续路线决定》 |
| `snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION.json` | 2850 | `c5b8d606ddaee50c3c366d2f720fe70f724ea2c97823f4ab5e38b5acd7b3fec1` | 机器JSON；顶层字段=status,diagram_file_count,mermaid_block_count,expected_m0_script_count,missing_m0_scripts,failures,files |
| `snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V2.json` | 3112 | `9aa45ec2a7d7ea11a3d1ff5240bbfe00dcea578bbb34f5eb46d482b6cbf17517` | 机器JSON；顶层字段=status,diagram_file_count,mermaid_block_count,expected_m0_script_count,missing_m0_scripts,failures,files |
| `snapshots/verification/ALGORITHM_DIAGRAM_VERIFICATION_ROUTE_V3.json` | 3375 | `5d275bc757e44fa145ca466d2e20684d5b2124cc9d25407eb5b368e7f9bc8983` | 机器JSON；顶层字段=status,diagram_file_count,mermaid_block_count,expected_m0_script_count,missing_m0_scripts,failures,files |
| `snapshots/verification/BASELINE_SUMMARY.json` | 375 | `8cb2a6ead19c107cefb2dd1f4e1f65fc8533576a1d87749e360a58aa252af881` | 机器JSON；顶层字段=code_full_read_count,error_count,path_reference_count,root_file_count,root_full_read_count,security,timestamp_utc,workspace_byte_count |
| `snapshots/verification/CODE_DIAGRAM_COVERAGE.json` | 6904 | `44f522b54d717a099543298506094b2fb4c02713678a3bfc44faacd0f3b965ff` | 机器JSON；顶层字段=status,code_manifest_count,group_counts,expected_group_counts,covered_count,missing_count,failures,files |
| `snapshots/verification/CODE_FILES.jsonl` | 57269 | `92f9b0328fced5a72922cfc8eea8c0bd8fcc44f97090d5b0ac4f0a09d50ec2d0` | 逐行机器证据；记录数=52 |
| `snapshots/verification/CODE_PATH_REFERENCES.jsonl` | 767183 | `af487aca29c73018988af74e0805ff4e84c15142cfb61775b8d6d76e714b8132` | 逐行机器证据；记录数=2387 |
| `snapshots/verification/ROOT_FILES.jsonl` | 37555 | `e2843bac57e526da14c8cd7ed9fce604ceb48591997008ba732e58d1e1956bd1` | 逐行机器证据；记录数=45 |
| `snapshots/verification/SCAN_RUNS.jsonl` | 8377 | `c43da31b5de360796888bbc61b51539664490000d62bc98ab8202b8af66c9fbf` | 逐行机器证据；记录数=25 |
| `snapshots/verification/SCAN_VERIFICATION.json` | 4259 | `e562b99a17854050defefef45c4caacc52470420706fc27429d697fe0c52e0df` | 机器JSON；顶层字段=baseline,failures,inputs,ledger,outputs,status |
| `snapshots/verification/TOP_LEVEL_DIRECTORIES.json` | 7081 | `6b79fd6b4f78a71e8cb4ed7a67292e3917162c530c5e3e2e7157e492c4c72ebf` | 机器JSON |
| `snapshots/verification/inputs/PPG_Testing_05_01_2026.summary.json` | 277 | `7f16b8e6c23016576fe547476813cd34980255caf3388527a4e89d225415e135` | 机器JSON；顶层字段=byte_count,error_count,file_count,format_counts,head_bytes_per_file,target,timestamp_utc |
| `snapshots/verification/inputs/physionet.org.summary.json` | 288 | `1875d00e52c9329d04d81b8819b366ae5a5a1716b234876118b25c9ea570c151` | 机器JSON；顶层字段=byte_count,error_count,file_count,format_counts,head_bytes_per_file,target,timestamp_utc |
| `snapshots/verification/outputs/CNN_RESULTS.summary.json` | 352 | `e2e864632044318a4ceb8773dcf136528192e5fc8d1602970ba1180d3f6a63ba` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/denoiser_preview_output.summary.json` | 258 | `f2ca400b7ca021c7df49f046cf4a71978da2f0a7eadff6aba9454dfba78e58cb` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results.summary.json` | 240 | `d642178e58439226705345a0bf88044dc90c395d65f8d487ae48c52f8e1c323f` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_denoiser_v8.summary.json` | 231 | `ebf1f8d3f80e47c68f689c58251d9b03833330ad45c88e8f6e0454f061ca9e59` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_frailty3.summary.json` | 319 | `e4432c91530a63009313e819cbdb0dabc40e9fa3af8468ee039b93b02fbec241` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_hybrid_denoiser.summary.json` | 273 | `0d5f98bcf7072f0fa7ce3cf0323b6fe502f2a5941858ae03b3183c0c17f6a7c9` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_hybrid_denoiser_raw_imu.summary.json` | 313 | `ef876dc94e26df82cf3fd0eadaa715092220143e97c42b5ac123720698d83f4d` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_hybrid_denoiser_raw_imu_baseline.summary.json` | 322 | `5649e32d09cbd3502540065e95a8f9ee03e55dfe44186fce5016f9be93112cab` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_stage1.summary.json` | 282 | `490692f73a7ec03885bfb020efe3c2917643c365b2009286302f3a78209f92be` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_stage2.summary.json` | 226 | `869828aa6fe291ab74b98ec13fc54e630c04318850d1b6914eb20db606e8e6bd` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_v72_noleak.summary.json` | 285 | `ee4a88a2c718e76533f189d81e02340490e6c1917f81fca311e818578674134f` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_v7_4.summary.json` | 327 | `72e65eceddad71d9089aff63758a7d1f5b562cfe10e8ab878f47c640e5c57d6f` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `snapshots/verification/outputs/results_v8_audit.summary.json` | 284 | `f032ac9f5d97d53d2c5c2c6bf5ff20d07fee8b2e30e377176f612dafa4d6a994` | 机器JSON；顶层字段=binary_metadata_only_count,byte_count,error_count,file_count,suffix_counts,target,text_full_read_count,timestamp_utc |
| `10_M0_PACKAGE_TREE_V3.md` | self | intentionally omitted | 自动生成的v3包树自身。 |

- 永久文件数（含本索引）：**80**。
- v1/v2历史文件保持原字节；v3只追加新决定、算法图、验证和证据。
- 总项目树由 `final_v0/FINAL_V0_TREE.md` 维护。
