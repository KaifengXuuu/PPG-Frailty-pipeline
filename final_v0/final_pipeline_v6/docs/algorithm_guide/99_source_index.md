# Appendix: Source Locations and Snapshot

This table lists the main algorithm files, configurations, and entry points discussed in the guide. Listing an entry point does not mean that the guide explains all of its training/reporting logic. Paths are relative to the V6 root. Line counts include comments and blank lines; their sum is not a count of added production code: this guide is documentation only.

If a line reference no longer matches, search for the function name in the text, then compare the source digest below. SHA-256 identifies the documented source snapshot only; these digests do not participate in pipeline execution. The English V6 snapshot has refreshed file counts and digests; historical inline line ranges may shift as comments are translated, so function and class names remain the primary navigation anchors.

| ID | Source file | Lines |
|---|---|---:|
| S01 | `configs/presets/finalcase.yaml` | 377 |
| S02 | `configs/studies/finalcase.yaml` | 94 |
| S03 | `src/ppg_frailty/artifacts/base.py` | 211 |
| S04 | `src/ppg_frailty/artifacts/bss.py` | 444 |
| S05 | `src/ppg_frailty/artifacts/decomposition.py` | 139 |
| S06 | `src/ppg_frailty/artifacts/identity.py` | 41 |
| S07 | `src/ppg_frailty/artifacts/legacy.py` | 400 |
| S08 | `src/ppg_frailty/artifacts/nlms.py` | 136 |
| S09 | `src/ppg_frailty/artifacts/router.py` | 169 |
| S10 | `src/ppg_frailty/artifacts/spectral.py` | 222 |
| S11 | `src/ppg_frailty/contracts.py` | 386 |
| S12 | `src/ppg_frailty/data/preprocessing_cache.py` | 842 |
| S13 | `src/ppg_frailty/data/qc.py` | 334 |
| S14 | `src/ppg_frailty/data/recording_cache.py` | 546 |
| S15 | `src/ppg_frailty/data/windows.py` | 221 |
| S16 | `src/ppg_frailty/experiment.py` | 4145 |
| S17 | `src/ppg_frailty/features/engineering.py` | 414 |
| S18 | `src/ppg_frailty/features/prv_backend_compare.py` | 249 |
| S19 | `src/ppg_frailty/features/registry.py` | 534 |
| S20 | `src/ppg_frailty/features/vector_transform.py` | 272 |
| S21 | `src/ppg_frailty/features/window_matrix.py` | 620 |
| S22 | `src/ppg_frailty/legacy_bridge.py` | 666 |
| S23 | `src/ppg_frailty/models/motion.py` | 219 |
| S24 | `src/ppg_frailty/models/time_scale.py` | 294 |
| S25 | `src/ppg_frailty/module_registry.py` | 1213 |
| S26 | `src/ppg_frailty/normalization.py` | 220 |
| S27 | `src/ppg_frailty/peaks/aboy_project.py` | 560 |
| S28 | `src/ppg_frailty/peaks/aboy_project_v2.py` | 410 |
| S29 | `src/ppg_frailty/peaks/msptdfast_v2.py` | 282 |
| S30 | `src/ppg_frailty/peaks/pairing.py` | 342 |
| S31 | `src/ppg_frailty/peaks/resolver.py` | 207 |
| S32 | `src/ppg_frailty/pipeline.py` | 353 |
| S33 | `src/ppg_frailty/provenance.py` | 121 |
| S34 | `src/ppg_frailty/quality/motion.py` | 589 |
| S35 | `src/ppg_frailty/quality/motion_adapters.py` | 454 |
| S36 | `src/ppg_frailty/quality/motion_bundle_adapter.py` | 528 |
| S37 | `src/ppg_frailty/quality/motion_reference.py` | 982 |
| S38 | `src/ppg_frailty/quality/motion_runner.py` | 1220 |
| S39 | `src/ppg_frailty/quality/routing.py` | 294 |
| S40 | `src/ppg_frailty/quality/routing_timeline.py` | 379 |
| S41 | `src/ppg_frailty/quality/stage5_pre.py` | 1056 |
| S42 | `src/ppg_frailty/quality/window_selection.py` | 342 |
| S43 | `src/ppg_frailty/representations/feature_matrix.py` | 43 |
| S44 | `src/ppg_frailty/representations/feature_vector.py` | 34 |
| S45 | `src/ppg_frailty/representations/fusion.py` | 14 |
| S46 | `src/ppg_frailty/representations/imu_transform.py` | 290 |
| S47 | `src/ppg_frailty/representations/motion.py` | 368 |
| S48 | `src/ppg_frailty/representations/raw.py` | 244 |
| S49 | `src/ppg_frailty/signal/imu.py` | 879 |
| S50 | `src/ppg_frailty/signal/morphology.py` | 166 |
| S51 | `src/ppg_frailty/signal/motion_imu.py` | 816 |
| S52 | `src/ppg_frailty/signal/optical.py` | 379 |
| S53 | `src/ppg_frailty/signal/peaks.py` | 225 |
| S54 | `src/ppg_frailty/signal/preprocess.py` | 808 |
| S55 | `src/ppg_frailty/signal/prv.py` | 530 |
| S56 | `src/ppg_frailty/signal/resample.py` | 275 |
| S57 | `src/ppg_frailty/signal/sqi.py` | 1189 |
| S58 | `src/ppg_frailty/signal/views.py` | 157 |
| S59 | `src/ppg_frailty/study/schema.py` | 799 |
| S60 | `src/ppg_frailty/v5/cli.py` | 447 |
| S61 | `src/ppg_frailty/models/feature_baselines.py` | 232 |
| S62 | `src/ppg_frailty/models/fusion.py` | 117 |
| S63 | `src/ppg_frailty/__init__.py` | 41 |
| S64 | `src/ppg_frailty/config.py` | 1057 |
| S65 | `src/ppg_frailty/v5/sweep.py` | 98 |

## Full SHA-256 Digests

```text
S01 100bc71467ca12674081856fe1b8ab388bdee215f39bcd2b1cea1a6e1bbbe128
S02 c924411fb84fc19fd11603c5653cac1fd55057701d075d64491248a8e9728c02
S03 890e843436877578ef9804f8a3da7a565eb159543d8a2ca6c84e9c69590147e1
S04 435defb90dc59ec85306627fb9eb4fa8f3e2fd4fd21795f2499386752442cc0b
S05 46b085075567ceaa32c874a5bb60e6b2866762c53a7daae896e7782c602d5f15
S06 c14fc0a55183626b90a44503445d4946e68f242700d7b52aa89ad40fe2bc031f
S07 1298d791d0538effa0b9d44d7bf211a209d84da376bd915ad999ed6b0e59e110
S08 599939e824699c00b72d139ef5233ab3460fd557728bd5c7c4d1f95a926b8824
S09 a6cd7bf4d5436a148fc9154ac10b18abcd307b6d9fff0366e00b96800f4bdc02
S10 7cdc1559a636500966ce33c1812524ac4065ad7c77637094dba45bec9b8e3142
S11 68126fd69c425e333d50a5aa2f1c8f776c901a9a25d86aace1859a77760b8d46
S12 e857c79432a4ab10d71d6f82db1d105df8b429e09db82d17719e1c414d88b6a1
S13 6bde49f9ae383f53f8ab928b4c4a8e22c0c1ae8405844e272669b7efcedf94b2
S14 2f83521f00228fe10cdbb64304c5e23cf17611b8c3681ff7a51cc5b69a3c737e
S15 eda30e3c3b35e069183355a05fd5507cb37813c5a92c94ca08986160b14545e7
S16 7171470540ad1ea378cc2e42914a1a4d4fdcbcb7746878973fd383b4091f7720
S17 817adc7deee315951bf01eaa4ed68d8d2d2dc5d9a16eb412b53a281ac9a1b036
S18 e56057685072ad007d56ec9090fb89051ba96dffe2dda4b255914ac1564f1938
S19 c7b4cad2fd9de637993789eb4b3ca8e6b0bc4fbc83a977f3e266c7005fe30f3d
S20 5599c7c96171bbd1d067d581cfee8d5174f40df5cc06a1ac5489af81587e8949
S21 4c3bb7513182b258bb5a1705e30878d7bbfe04668d1221ec9f63025380ff6110
S22 b8bc7ebfabb565505416c53d995ceef65d98323f602013d70573e55842a1e93d
S23 48d54b10a6d5eb0a4a15214949821a0a65fd884a9ce1bb55f254b0b59d06a80e
S24 7d64a38dfadccb33de2835aefe905c9915b70fffa67880904890ad103aa28d10
S25 cd6a16bcc0f1b2f619011665016973cf872782a15fa6c111bc6d749cf20893dc
S26 571772713d0cc78a3aa72f7eca298a95cd2cd45005233284b001cb69b7160525
S27 865ffd1a93409899a64ef813f610af46c1382456a051dc416a2da7fdd9075ed1
S28 67fe5d50274171e3cdc5383f4c38072da0e23432885d083aa87035b465de46f3
S29 7e8dcf6ef62c117b696a21c1e55f87055bcaadddb481e399232910e275b2517e
S30 4e5e6aa8f8a278bf32941b84582fed0eb4d39cba650cf83a15f4e685ba1507b6
S31 563d744807816bf48f1026b229eed8d04834512ab383f8cc7c579a54e787307e
S32 490c321c35c891f30af7109646afffa0e023e29325e361fc5c7d9f794151f22d
S33 24caa858db0f525dce95d2de87746224b90ad6eda16b44915eacbc95f920fca5
S34 f5adeaac15e6f4d7cced15b84d3150df8875dd097b595f54ce0cd8d095c77c3b
S35 e8801aa1309dd505df4ace859c2328cbc5c0c6a604a6bca31f1930446ca90863
S36 e4f74a2599c4c6b3e24f743618851d788123011631b6bd2b283ab4a3b5fb4e62
S37 3a39dcb742aca8e648bd2023591cc9f0932f1940ed6cd731edbe31d33532ff1c
S38 ec0e6eb99453eb2e2f20a69ec2764dfeb3f092826fbb4f2148174603125a2917
S39 2b10ac176e8515ae256a45c770c22b797af4862ffa7c2719b6b2899ca48572af
S40 61c70c87587ad01e382d629d3cec0c9d385feafd9e8e7938358daa9d3213dc0c
S41 835c9f2b46c9d5ae46b50ba360f08220e312081f1f526c51d518d3d23b9c92f1
S42 9fd2d1a64a40d21dbcee97150e59933f64b9e6d073e841ba171c02ffbdb4ac9c
S43 e74d8517e852fd31df67e91ca691d5c8f454e4b4036f9a7da33dc15ec8a17129
S44 1fd97c4c23dccec4d423c12028346fbd88a8c6fc38a96dc760c0a2ae3612f046
S45 e6be1cb4f094ebe0de0354def9afad1b69c245b167346d78c39e713806a60f32
S46 9b15379f99c8a79e1df0f65c6250925df9028a2efcba643e7395b0a024aa2987
S47 9eb0dca0905fe1b55ded8dd9f8a1721b922b7d70e1b9446c75cb0050dc58e547
S48 15b7a0dfe26b5964dd1fd7b824a885dd59123b67f5eecc91521dbbaae5162620
S49 fcdd0178f0925c43adceee5f9906c0161b0c42c0ea152f36c9fd224be283cf65
S50 14ef761f8569456e6d2cac313473ed46329feb3c3f99a51802e746bf840f0665
S51 53d8b35cd6d4480bcf54cf8a73eaaa40a010145ccb363f2bcbfed13a5b62fe9f
S52 fff69a047ad4a26571e5a8ae79b63516ed18ecc0d7a3ae105077a345f9a98659
S53 20a03be155b6cd2aa075e74b7b4c52c83d6341eeb08310edf89a6cf2c15b7302
S54 d80a2bc6154f68b338877da4a3e5dc61d725ef207ca34e29f79873707c6df087
S55 3f27031d26d2c571bb8563d1b290ee9802057c4a36d393ec5eeae2dd2f4e5e11
S56 49f24ab532bddbcf008b1ef6418d5b6e2a9bacbdc9eedcdb5d2a006b6d295b3c
S57 c033c8e853ed9ba7f31907f93b894825ca969f0f07b7b272b10be96d8e36c6d4
S58 12db9c60b1f41e5bf35ab4dadf9b55d1abd5b8e9310ce2804c83c515e2572dc1
S59 2d9e954ca03ed6081f30aad0c2a539e69108db551911e10ae18d15b4e6180ef7
S60 2d38eb1be298bd12aded68b45846a084b5987385101e31813513d7f8e5081f02
S61 085829b438ab2b6266b425a4316134098993345e320166a8afcea2c556b2fdc8
S62 64976aa0e6a8b8574e2e7332b5166db7e08f379a26e7b844d863190c445dc6e3
S63 a9f1f5ce38e6820ba74b49e9cc9aa0b546505271050f829a37d9fe54a90353a3
S64 ea6844b95ff6fd26d15fd087721352e23704bba7fcad5deb8ac2b9ff7b5e9cf9
S65 84b662773185e4bf47df354c8c881f48ec7c4809c2ac5c1e572a1bcbc460da5d
```

## Review Order and Validation Boundary

1. Read each section's intuitive explanation first to identify what the module aims to preserve and remove.
2. Check input units, axis order, validity masks, and defaults to confirm that the data represent the intended processing stage.
3. Check equations against the named functions and contiguous line ranges, especially filter direction, quantile constants, missing-value propagation, adjacency between peaks, boundary slices, and failure branches.
4. Check the entry point: existence of a low-level function, registration in the registry, and activation in a particular run are three different facts. Historical/specialized paths must not be confused with the current finalcase.
5. For transforms involving training-data statistics, inspect the fitting population separately. A function name containing `fit` or `cache` does not by itself establish whether leakage occurs.

This guide explains the implemented algorithms; it neither proves their effectiveness nor reproduces every internal iteration of third-party libraries line by line. Unimplemented learned denoisers, external-backend internals, and unexecuted tests of optional branches do not become validated merely because they are described. Before changing an algorithm, separately confirm the research objective and numerical-equivalence requirements. Reading this guide does not change configurations or results.
