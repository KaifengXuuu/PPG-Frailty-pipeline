# M1 包文件树与逐文件说明 / M1 Package Tree

> 由 `tools/validate_m1_contracts.py --write-report` 生成；验证报告与本树不自哈希。

## Tree

```text
M1_end_to_end_architecture_contract/
├── examples
│   ├── pipeline_accelerated_arm64.json
│   ├── pipeline_high_performance_x86.json
│   └── pipeline_value_arm64.json
├── examples_v2
│   ├── pipeline_accelerated_arm64_v2.json
│   ├── pipeline_high_performance_x86_v2.json
│   └── pipeline_value_arm64_v2.json
├── examples_v3
│   ├── pipeline_accelerated_arm64_v3.json
│   ├── pipeline_high_performance_x86_v3.json
│   └── pipeline_value_arm64_v3.json
├── registries
│   ├── classifier_registry.json
│   ├── feature_extractor_registry.json
│   ├── platform_profiles.json
│   └── quality_policy_registry.json
├── registries_v2
│   ├── classifier_registry_v2.json
│   ├── feature_extractor_registry_v2.json
│   ├── platform_profiles_v2.json
│   └── quality_policy_registry_v2.json
├── registries_v3
│   ├── quality_routing_registry_v3.json
│   └── quality_routing_registry_v3_active.json
├── schemas
│   ├── inference_output.schema.json
│   ├── pipeline_config.schema.json
│   └── signal_input.schema.json
├── schemas_v2
│   ├── inference_output_v2.schema.json
│   ├── pipeline_config_v2.schema.json
│   └── signal_input_v2.schema.json
├── schemas_v3
│   ├── inference_output_v3.schema.json
│   └── pipeline_config_v3.schema.json
├── tools
│   ├── bootstrap_m1_contract_report.py
│   ├── validate_m1_contracts.py
│   ├── validate_m1_contracts_v2.py
│   ├── validate_m1_contracts_v3.py
│   ├── validate_m1_contracts_v3_current.py
│   ├── validate_m1_v2_semantic_invariants.py
│   └── validate_m1_v3_routing_invariants.py
├── 00_CURRENT_STATUS_V2.md
├── 00_CURRENT_STATUS_V3.md
├── 00_CURRENT_STATUS_V3_1.md
├── 01_END_TO_END_ARCHITECTURE_AND_API.md
├── 02_MOBILE_PLATFORM_PROFILES.md
├── 03_TRAINING_VS_MOBILE_INFERENCE_BOUNDARY.md
├── 04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md
├── 05_VALIDATION_LIMITATIONS_AND_SEMANTIC_GATES.md
├── 06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md
├── M1_CONTRACT_VERIFICATION.json
├── M1_CONTRACT_VERIFICATION_V2.json
├── M1_CONTRACT_VERIFICATION_V3_CURRENT.json
├── M1_PACKAGE_TREE.md
├── M1_PACKAGE_TREE_V2.md
├── M1_PACKAGE_TREE_V3_CURRENT.md
├── M1_ROUTING_INVARIANTS_V3.json
├── M1_SEMANTIC_INVARIANTS_V2.json
└── README.md
```

## Integrity

| File | Bytes | SHA-256 | Content |
|---|---:|---|---|
| `00_CURRENT_STATUS_V2.md` | 4302 | `781734d3b24cbfd814fba017085d2fe341286b8e25629b2690be8535ddb91e2b` | Markdown《M1 当前权威合同 V2 / Current Authoritative Contract V2》；Mermaid=0 |
| `00_CURRENT_STATUS_V3.md` | 4921 | `5e057aaa7aa7c9daeae9b0a750ac3f2fb530158aca3cedf0190c65f915cf063d` | Markdown《M1 当前权威合同 V3 / Current Authoritative Contract V3》；Mermaid=0 |
| `00_CURRENT_STATUS_V3_1.md` | 1615 | `db4856cb44db5cef133c32d7e670fe06687423ad44afc4d159943a7486636c6d` | Markdown《M1 V3 当前验证入口修正 / Current V3 Validation Entry》；Mermaid=0 |
| `01_END_TO_END_ARCHITECTURE_AND_API.md` | 7282 | `a1043a67fa0fd6b688948f2374852bb4c4df30f72787ce0ffad9b34b8528a725` | Markdown《M1 端到端模块架构与统一 Python API》；Mermaid=0 |
| `02_MOBILE_PLATFORM_PROFILES.md` | 4887 | `492f5cbdfb50ff09b74e4dd65c9ccc4f5387391b2e5ab8f731d183d615803150` | Markdown《M1 血压仪大小中心屏显处理设备：平台分档与工程预算》；Mermaid=0 |
| `03_TRAINING_VS_MOBILE_INFERENCE_BOUNDARY.md` | 4218 | `5b04465ca5496b9d5ddf363134eea0de180756e552cb1e2506a3f7cb908a87d8` | Markdown《M1 训练/评估 Pipeline 与移动推理 Runtime 边界》；Mermaid=0 |
| `04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md` | 4722 | `27f52db07505ea4dcf9fc26189b4c088d7fdff678d2eb3651707a77f13a1696d` | Markdown《M1 现有实现接口审计与移动部署风险》；Mermaid=0 |
| `05_VALIDATION_LIMITATIONS_AND_SEMANTIC_GATES.md` | 2295 | `ab0e5bb8c8f48318579e0d98f7bc4801cd489356f4d69053e22762e6bc3e1c7c` | Markdown《M1 V2 验证限制与补充语义门》；Mermaid=0 |
| `06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md` | 7000 | `45959509dce135a506f5cae360118513932d475eb6ab1bc205a393d8636a5faf` | Markdown《M1 V3 顺序 SQI、可选 Motion 与 Denoiser 路由合同》；Mermaid=0 |
| `M1_CONTRACT_VERIFICATION_V2.json` | 779 | `b42a04f32aab347fb1b3354ef4ed51a4adb5215c0c6f9a1a0370530ce1f73032` | 机器JSON；顶层=allowed_dependencies,canonical_channel_order,classifier_count,contract_version,example_config_count,failure_count |
| `M1_CONTRACT_VERIFICATION_V3_CURRENT.json` | 920 | `2bb9e0be3833b35bc3ce838a52d900d118fd1455a149295912aa6a25ac553d85` | 机器JSON；顶层=canonical_channel_order,classifier_count,contract_version,denoiser_frontend_count,example_config_count,failure_count |
| `M1_PACKAGE_TREE_V2.md` | 3267 | `c78a8896b1e0689bb2fc1fe0af8d4c50004b142adcd6676fee6338d4475c34db` | Markdown《M1 V2 权威文件树与完整性 / M1 V2 Integrity Tree》；Mermaid=0 |
| `M1_PACKAGE_TREE_V3_CURRENT.md` | 3821 | `8968dd7180d0a9595ea0678f97726d1206ac1934f82357a11646e7548407db8b` | Markdown《M1 V3 CURRENT 权威文件树与完整性》；Mermaid=0 |
| `M1_ROUTING_INVARIANTS_V3.json` | 4836 | `58769053d31dfd611a64e6317d25f310e277e65022e5fe49e5b5c1d2653b04d1` | 机器JSON；顶层=contract_version,failure_count,failures,model_execution,passed_test_count,scope |
| `M1_SEMANTIC_INVARIANTS_V2.json` | 2857 | `a937775022748a0d49188ee6cb7d9f638389d305045799ccbce1ed5848112a5b` | 机器JSON；顶层=contract_version,failure_count,failures,model_execution,passed_test_count,scope |
| `README.md` | 2324 | `bae17fcf9c610768a6d8eefbc8ddb7fe2caead1df72e18801a2fa15d207ee1c0` | Markdown《M1 端到端架构、数据契约与移动处理中心约束》；Mermaid=0 |
| `examples/pipeline_accelerated_arm64.json` | 1530 | `7c1a5d35a9a82f5dde4de6ca116b01ec46757b51e09edd24cc8adda394121959` | 机器JSON；顶层=schema_version,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime |
| `examples/pipeline_high_performance_x86.json` | 1535 | `92f51376b0f1c96df9193d97b84e2140532476d9fe3dcb917313c30c8104e1e0` | 机器JSON；顶层=schema_version,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime |
| `examples/pipeline_value_arm64.json` | 1473 | `0beaafb1ad36aa926257ebc5322b63ec7c04cdffd78d017839cf4d4408dd3c8a` | 机器JSON；顶层=schema_version,config_id,platform_profile_id,input_contract_ref,output_contract_ref,runtime |
| `examples_v2/pipeline_accelerated_arm64_v2.json` | 1928 | `b37961cc016d083860a37df6145eec995461f336f02cd66ca1e7f257c2511e5a` | 机器JSON；顶层=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v2/pipeline_high_performance_x86_v2.json` | 1876 | `0e32946c35b9a914af10424bf814b7a455707acb9b6c2ed20de1b2db284918aa` | 机器JSON；顶层=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v2/pipeline_value_arm64_v2.json` | 1789 | `18505c10fcc5851af3ab022ea4a40e8cb07bdbd6494c8b85ede317850b83de9b` | 机器JSON；顶层=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v3/pipeline_accelerated_arm64_v3.json` | 3220 | `61a9bbf212817e1a93e69e5c3b5f4f8cd530e66b9ba1c1e4066da7a21786abf2` | 机器JSON；顶层=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v3/pipeline_high_performance_x86_v3.json` | 3169 | `1c2936ac0d2cd0a9da85122cb15e17cf231b3a357627bb2ee969895db28da6ea` | 机器JSON；顶层=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v3/pipeline_value_arm64_v3.json` | 2967 | `93e876ad843957b15811c768696c37ee14b68c42ff3687e3b07ff152323da496` | 机器JSON；顶层=schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `registries/classifier_registry.json` | 2053 | `a3f0b0a58813af7a0ccaef9a3397d0833f1f855df0c60040d245c8d571f6761d` | 机器JSON；顶层=registry_version,output_adapter,label_order,classifiers |
| `registries/feature_extractor_registry.json` | 1065 | `db274e89b735aee89cc964004ff804bad52f56a4a7db6e9e5e8e3cb36c2291fe` | 机器JSON；顶层=registry_version,extractors |
| `registries/platform_profiles.json` | 2201 | `a7f1e6fd921487061ecff2de794f68d9b4e19c52c18b5edf79f323abcddf2a62` | 机器JSON；顶层=registry_version,status,allowed_dependencies,profiles |
| `registries/quality_policy_registry.json` | 705 | `706f858406e5461f6b8637e8b0daf9bc41e3814cb7c8a1bfb35eac205631fa6e` | 机器JSON；顶层=registry_version,sqi_monitor_required,action_mode_cardinality,policies |
| `registries_v2/classifier_registry_v2.json` | 2945 | `a5b0ccb4103eee8de2983c9e19a5cb3103c82f2dc283d70555b9337455c29c18` | 机器JSON；顶层=registry_version,output_adapter,label_order,entry_contract,classifiers |
| `registries_v2/feature_extractor_registry_v2.json` | 1446 | `04d9c6759f04f9423493e709039318d42dc5b51b6194146e9c5fe26e719582dd` | 机器JSON；顶层=registry_version,entry_contract,extractors |
| `registries_v2/platform_profiles_v2.json` | 2419 | `108c800e6ccefa4dbe2f65fc78062828cd4f097b9c5662299f08c6395a27c6a5` | 机器JSON；顶层=registry_version,status,budgets_are_measured_results,allowed_dependencies,streaming_contract,power_scope |
| `registries_v2/quality_policy_registry_v2.json` | 1049 | `968cbb3e5bcc5776bf2a5b3439acda6360ac23490f92f7e88f246fc5139d4a21` | 机器JSON；顶层=registry_version,sqi_monitor_required,action_mode_cardinality,diagnostic_candidates_may_run_in_parallel,allowed_final_action_codes,policies |
| `registries_v3/quality_routing_registry_v3.json` | 3488 | `32d257cbc89c0eb690d1cd53eea13a3af4e99243d504fd61ca9ca97f9919ee34` | 机器JSON；顶层=registry_version,architecture_version,sqi_required,motion_detector_optional,first_stage_parallelism,state_axes |
| `registries_v3/quality_routing_registry_v3_active.json` | 3163 | `0d5643de7a19f9c612fdbd29afa31273a37fc91fbc716727092a82a67fa6411e` | 机器JSON；顶层=registry_version,architecture_version,sqi_required,motion_detector_optional,first_stage_parallelism,state_axes |
| `schemas/inference_output.schema.json` | 1960 | `217f6b8f8ba36c6d083704d40ca49025753994793dcd0e3676cdcb080026e3f8` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas/pipeline_config.schema.json` | 3608 | `b5afc25c5e2888d6a5f80c802062eba12095da4e47b55884bb7834eb9317e145` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas/signal_input.schema.json` | 2691 | `447ddc7684a0825efe37f04abbc3887417be99b7f26e0ff874ac28fc70c4c149` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas_v2/inference_output_v2.schema.json` | 5086 | `c41dab571d544d340cac3290215b3312cf53123218b6095770acf4e3d19cce67` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas_v2/pipeline_config_v2.schema.json` | 6028 | `c15e854bca5a0d40cd98f183cba93237036c9be8408240083c73ed17a14a644c` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas_v2/signal_input_v2.schema.json` | 2813 | `6c4cbff49ee39ff04a385b5bc2a106926217f1bf591361fd39f8bce84f7dbfba` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas_v3/inference_output_v3.schema.json` | 9023 | `e9f6870c36e3e82c14dca17cc27eab90e6001558719d8b2d67e7c847e0a7486f` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `schemas_v3/pipeline_config_v3.schema.json` | 11504 | `1a06d416e312507997d13be440e0aca3712957f9d6d7ffa05f0ae602a51ea5ce` | 机器JSON；顶层=$schema,$id,title,type,additionalProperties,required |
| `tools/bootstrap_m1_contract_report.py` | 1981 | `06086c237c3ec698b5cc21cf6d65760c2e034058b0ff8e2da830618df3b7145c` | 双语合同验证与索引工具 |
| `tools/validate_m1_contracts.py` | 18601 | `21230f930c08c281585eb985d42a6d005522b0dddceed4f4a7b6b7b2975a57eb` | 双语合同验证与索引工具 |
| `tools/validate_m1_contracts_v2.py` | 19439 | `451fe054cc21e8e6f8f767feea37bf409b3f60a7819d42c61a05d48c90da031b` | 双语合同验证与索引工具 |
| `tools/validate_m1_contracts_v3.py` | 27634 | `659e91ef58b2b4fa6fa5513bf4b5efe2ef9b6a342c8328497386a649d2a3f1e8` | 双语合同验证与索引工具 |
| `tools/validate_m1_contracts_v3_current.py` | 6129 | `6b3724dfcc367c3b6802e0ed7b3b7fd0558989e89fddc4b991d8d0f7d3882e58` | 双语合同验证与索引工具 |
| `tools/validate_m1_v2_semantic_invariants.py` | 12064 | `88a03099fd1cf26eb2555cf018a41a1b514128d72d1d162041f8daa3f3e48351` | 双语合同验证与索引工具 |
| `tools/validate_m1_v3_routing_invariants.py` | 21392 | `7241e213195a2ec1ae63b65d829871202ad2eafb801ae8693d9812914fb14f85` | 双语合同验证与索引工具 |
| `M1_CONTRACT_VERIFICATION.json` | self | intentionally omitted | 自动生成验证报告 |
| `M1_PACKAGE_TREE.md` | self | intentionally omitted | 自动生成包树 |

- Permanent files including generated indexes: **52**.
- 所有写入均位于 `final_v0/M1_end_to_end_architecture_contract/`。
