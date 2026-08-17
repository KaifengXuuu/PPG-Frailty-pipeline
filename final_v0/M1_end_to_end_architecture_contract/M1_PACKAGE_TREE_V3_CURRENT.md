# M1 V3 CURRENT 权威文件树与完整性

> 由 `tools/validate_m1_contracts_v3_current.py --write-report` 生成。

| File | Bytes | SHA-256 | Content |
|---|---:|---|---|
| `00_CURRENT_STATUS_V3.md` | 4921 | `5e057aaa7aa7c9daeae9b0a750ac3f2fb530158aca3cedf0190c65f915cf063d` | Markdown《M1 当前权威合同 V3 / Current Authoritative Contract V3》 |
| `00_CURRENT_STATUS_V3_1.md` | 1615 | `db4856cb44db5cef133c32d7e670fe06687423ad44afc4d159943a7486636c6d` | Markdown《M1 V3 当前验证入口修正 / Current V3 Validation Entry》 |
| `06_SEQUENTIAL_SQI_MOTION_ROUTING_V3.md` | 7000 | `45959509dce135a506f5cae360118513932d475eb6ab1bc205a393d8636a5faf` | Markdown《M1 V3 顺序 SQI、可选 Motion 与 Denoiser 路由合同》 |
| `schemas_v2/signal_input_v2.schema.json` | 2813 | `6c4cbff49ee39ff04a385b5bc2a106926217f1bf591361fd39f8bce84f7dbfba` | Machine JSON: $schema,$id,title,type,additionalProperties,required |
| `schemas_v3/pipeline_config_v3.schema.json` | 11504 | `1a06d416e312507997d13be440e0aca3712957f9d6d7ffa05f0ae602a51ea5ce` | Machine JSON: $schema,$id,title,type,additionalProperties,required |
| `schemas_v3/inference_output_v3.schema.json` | 9023 | `e9f6870c36e3e82c14dca17cc27eab90e6001558719d8b2d67e7c847e0a7486f` | Machine JSON: $schema,$id,title,type,additionalProperties,required |
| `registries_v2/platform_profiles_v2.json` | 2419 | `108c800e6ccefa4dbe2f65fc78062828cd4f097b9c5662299f08c6395a27c6a5` | Machine JSON: registry_version,status,budgets_are_measured_results,allowed_dependencies,streaming_contract,power_scope |
| `registries_v3/quality_routing_registry_v3_active.json` | 3163 | `0d5643de7a19f9c612fdbd29afa31273a37fc91fbc716727092a82a67fa6411e` | Machine JSON: registry_version,architecture_version,sqi_required,motion_detector_optional,first_stage_parallelism,state_axes |
| `registries_v2/feature_extractor_registry_v2.json` | 1446 | `04d9c6759f04f9423493e709039318d42dc5b51b6194146e9c5fe26e719582dd` | Machine JSON: registry_version,entry_contract,extractors |
| `registries_v2/classifier_registry_v2.json` | 2945 | `a5b0ccb4103eee8de2983c9e19a5cb3103c82f2dc283d70555b9337455c29c18` | Machine JSON: registry_version,output_adapter,label_order,entry_contract,classifiers |
| `examples_v3/pipeline_high_performance_x86_v3.json` | 3169 | `1c2936ac0d2cd0a9da85122cb15e17cf231b3a357627bb2ee969895db28da6ea` | Machine JSON: schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v3/pipeline_accelerated_arm64_v3.json` | 3220 | `61a9bbf212817e1a93e69e5c3b5f4f8cd530e66b9ba1c1e4066da7a21786abf2` | Machine JSON: schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v3/pipeline_value_arm64_v3.json` | 2967 | `93e876ad843957b15811c768696c37ee14b68c42ff3687e3b07ff152323da496` | Machine JSON: schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `tools/validate_m1_contracts_v3.py` | 27634 | `659e91ef58b2b4fa6fa5513bf4b5efe2ef9b6a342c8328497386a649d2a3f1e8` | Bilingual V3 contract or semantic validator |
| `tools/validate_m1_contracts_v3_current.py` | 6129 | `6b3724dfcc367c3b6802e0ed7b3b7fd0558989e89fddc4b991d8d0f7d3882e58` | Bilingual V3 contract or semantic validator |
| `tools/validate_m1_v3_routing_invariants.py` | 21392 | `7241e213195a2ec1ae63b65d829871202ad2eafb801ae8693d9812914fb14f85` | Bilingual V3 contract or semantic validator |
| `M1_CONTRACT_VERIFICATION_V3_CURRENT.json` | self | intentionally omitted | CURRENT machine verification |
| `M1_PACKAGE_TREE_V3_CURRENT.md` | self | intentionally omitted | CURRENT integrity tree |

- CURRENT authority files including generated indexes: **18**.
- 首版 V3 validator/迁移 registry 保留为历史，CURRENT 使用 active registry。
