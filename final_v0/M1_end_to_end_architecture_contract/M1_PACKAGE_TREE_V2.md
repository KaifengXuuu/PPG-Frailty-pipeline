# M1 V2 权威文件树与完整性 / M1 V2 Integrity Tree

> 由 `tools/validate_m1_contracts_v2.py --write-report` 生成；V1 历史层见 `M1_PACKAGE_TREE.md`。

| File | Bytes | SHA-256 | Content |
|---|---:|---|---|
| `00_CURRENT_STATUS_V2.md` | 4302 | `781734d3b24cbfd814fba017085d2fe341286b8e25629b2690be8535ddb91e2b` | Markdown《M1 当前权威合同 V2 / Current Authoritative Contract V2》 |
| `04_EXISTING_CODE_AUDIT_AND_MOBILE_RISKS.md` | 4722 | `27f52db07505ea4dcf9fc26189b4c088d7fdff678d2eb3651707a77f13a1696d` | Markdown《M1 现有实现接口审计与移动部署风险》 |
| `schemas_v2/signal_input_v2.schema.json` | 2813 | `6c4cbff49ee39ff04a385b5bc2a106926217f1bf591361fd39f8bce84f7dbfba` | Machine JSON: $schema,$id,title,type,additionalProperties,required |
| `schemas_v2/pipeline_config_v2.schema.json` | 6028 | `c15e854bca5a0d40cd98f183cba93237036c9be8408240083c73ed17a14a644c` | Machine JSON: $schema,$id,title,type,additionalProperties,required |
| `schemas_v2/inference_output_v2.schema.json` | 5086 | `c41dab571d544d340cac3290215b3312cf53123218b6095770acf4e3d19cce67` | Machine JSON: $schema,$id,title,type,additionalProperties,required |
| `registries_v2/platform_profiles_v2.json` | 2419 | `108c800e6ccefa4dbe2f65fc78062828cd4f097b9c5662299f08c6395a27c6a5` | Machine JSON: registry_version,status,budgets_are_measured_results,allowed_dependencies,streaming_contract,power_scope |
| `registries_v2/quality_policy_registry_v2.json` | 1049 | `968cbb3e5bcc5776bf2a5b3439acda6360ac23490f92f7e88f246fc5139d4a21` | Machine JSON: registry_version,sqi_monitor_required,action_mode_cardinality,diagnostic_candidates_may_run_in_parallel,allowed_final_action_codes,policies |
| `registries_v2/feature_extractor_registry_v2.json` | 1446 | `04d9c6759f04f9423493e709039318d42dc5b51b6194146e9c5fe26e719582dd` | Machine JSON: registry_version,entry_contract,extractors |
| `registries_v2/classifier_registry_v2.json` | 2945 | `a5b0ccb4103eee8de2983c9e19a5cb3103c82f2dc283d70555b9337455c29c18` | Machine JSON: registry_version,output_adapter,label_order,entry_contract,classifiers |
| `examples_v2/pipeline_high_performance_x86_v2.json` | 1876 | `0e32946c35b9a914af10424bf814b7a455707acb9b6c2ed20de1b2db284918aa` | Machine JSON: schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v2/pipeline_accelerated_arm64_v2.json` | 1928 | `b37961cc016d083860a37df6145eec995461f336f02cd66ca1e7f257c2511e5a` | Machine JSON: schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `examples_v2/pipeline_value_arm64_v2.json` | 1789 | `18505c10fcc5851af3ab022ea4a40e8cb07bdbd6494c8b85ede317850b83de9b` | Machine JSON: schema_version,deployment_state,config_id,platform_profile_id,input_contract_ref,output_contract_ref |
| `tools/validate_m1_contracts_v2.py` | 19439 | `451fe054cc21e8e6f8f767feea37bf409b3f60a7819d42c61a05d48c90da031b` | Bilingual V2 contract validator |
| `M1_CONTRACT_VERIFICATION_V2.json` | self | intentionally omitted | V2 machine verification |
| `M1_PACKAGE_TREE_V2.md` | self | intentionally omitted | V2 integrity tree |

- V2 authoritative files including generated indexes: **15**.
- All V2 writes remain under `final_v0/`.
