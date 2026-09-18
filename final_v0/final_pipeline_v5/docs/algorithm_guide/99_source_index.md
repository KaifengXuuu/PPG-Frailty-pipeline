# 附录：源文件定位与快照

本表列出正文涉及的主要算法文件、配置和调用入口；包含调用入口不表示本文解释其全部训练/报告逻辑。路径相对 V5 根目录。行数包括注释与空行，不能用本表合计衡量新增生产代码：本文只新增文档。

行号定位失败时，先搜索正文中的函数名，再与下列源文件摘要核对。SHA-256 仅用于识别文档所说明的源码快照，不参与 pipeline 执行。

| 编号 | 源文件 | 行数 |
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
| S16 | `src/ppg_frailty/experiment.py` | 4112 |
| S17 | `src/ppg_frailty/features/engineering.py` | 414 |
| S18 | `src/ppg_frailty/features/prv_backend_compare.py` | 249 |
| S19 | `src/ppg_frailty/features/registry.py` | 534 |
| S20 | `src/ppg_frailty/features/vector_transform.py` | 272 |
| S21 | `src/ppg_frailty/features/window_matrix.py` | 606 |
| S22 | `src/ppg_frailty/legacy_bridge.py` | 666 |
| S23 | `src/ppg_frailty/models/motion.py` | 219 |
| S24 | `src/ppg_frailty/models/time_scale.py` | 294 |
| S25 | `src/ppg_frailty/module_registry.py` | 1220 |
| S26 | `src/ppg_frailty/normalization.py` | 220 |
| S27 | `src/ppg_frailty/peaks/aboy_project.py` | 560 |
| S28 | `src/ppg_frailty/peaks/aboy_project_v2.py` | 410 |
| S29 | `src/ppg_frailty/peaks/msptdfast_v2.py` | 282 |
| S30 | `src/ppg_frailty/peaks/pairing.py` | 342 |
| S31 | `src/ppg_frailty/peaks/resolver.py` | 207 |
| S32 | `src/ppg_frailty/pipeline.py` | 352 |
| S33 | `src/ppg_frailty/provenance.py` | 121 |
| S34 | `src/ppg_frailty/quality/motion.py` | 589 |
| S35 | `src/ppg_frailty/quality/motion_adapters.py` | 454 |
| S36 | `src/ppg_frailty/quality/motion_bundle_adapter.py` | 521 |
| S37 | `src/ppg_frailty/quality/motion_reference.py` | 984 |
| S38 | `src/ppg_frailty/quality/motion_runner.py` | 1220 |
| S39 | `src/ppg_frailty/quality/routing.py` | 294 |
| S40 | `src/ppg_frailty/quality/routing_timeline.py` | 379 |
| S41 | `src/ppg_frailty/quality/stage5_pre.py` | 1044 |
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
| S54 | `src/ppg_frailty/signal/preprocess.py` | 806 |
| S55 | `src/ppg_frailty/signal/prv.py` | 530 |
| S56 | `src/ppg_frailty/signal/resample.py` | 275 |
| S57 | `src/ppg_frailty/signal/sqi.py` | 1189 |
| S58 | `src/ppg_frailty/signal/views.py` | 157 |
| S59 | `src/ppg_frailty/study/schema.py` | 799 |
| S60 | `src/ppg_frailty/v5/cli.py` | 445 |
| S61 | `src/ppg_frailty/models/feature_baselines.py` | 232 |
| S62 | `src/ppg_frailty/models/fusion.py` | 117 |
| S63 | `src/ppg_frailty/__init__.py` | 41 |
| S64 | `src/ppg_frailty/config.py` | 1057 |
| S65 | `src/ppg_frailty/v5/sweep.py` | 98 |

## 完整 SHA-256

```text
S01 100bc71467ca12674081856fe1b8ab388bdee215f39bcd2b1cea1a6e1bbbe128
S02 c924411fb84fc19fd11603c5653cac1fd55057701d075d64491248a8e9728c02
S03 9f04a31c3248d2a008bfb6e7deabcf8d4c9b7ab7463089933591b1a5ef3199df
S04 f270b8ab06d81e1559df37cd79acbaabc1e6f9dd7ae7d6b4d8ae14d502131104
S05 9e2a2301efabece0d0a4a25b9680a3563570336012a3a9b600ceea9d47e0d83d
S06 b5430f73f563acef415077c946d8681369af3b288e60934d4f3899f54a283ec0
S07 7ab4dc64951f39bbca08a88f30799f77546cebdc679485065a37cf5e0a4da47d
S08 4d9230c1efdf6f9fdc0ef5944eb8185f9f7fa2ca28fb85b34fe4d799ce39c28a
S09 864d00073e24473fe278f2c220ac8e53b98a842ae87ccae4e27e5fcda1154775
S10 eb1497f88ec92d3e64e08c391e94caf3c098ce5bb119501e57d33b9dbedb8266
S11 370e1c5705eec3ceeaab1370573c5e27744339d1769f56d7d79d2f58b2abf47d
S12 e857c79432a4ab10d71d6f82db1d105df8b429e09db82d17719e1c414d88b6a1
S13 071843dec6251f0b9eb31625c3f499e5156e5c94e6f3b3a4ab2f66796c539038
S14 2f83521f00228fe10cdbb64304c5e23cf17611b8c3681ff7a51cc5b69a3c737e
S15 7895f45559e61aa59cc4bbd1c329336040d16db4b17a169e0ba6ea396430fcfb
S16 772fdc91ac8724a83e1865be34a80d165e27f50c3f518e041c8a740c55c72857
S17 5e58390015a7a9cbe6e7e50baf3d406221ec1ee70ffde7d0d49e73ab244065f6
S18 e5da1efc674fdfa78b6211299821331dccd954cf3bcc4c7ca4e5d0a0948e508e
S19 701eb514647516956eace3c55b29bf50085ae8751e23e57895e54e4d605bb66c
S20 8e37f3815b3ba211c1b9a5f3867fc9bc7f44f50c70da4311a0b213e4d3d3a8de
S21 dac44e1d3ad506540410fc7d14c328bd0fdb553a8a8b468b25c582e6ced09931
S22 b8bc7ebfabb565505416c53d995ceef65d98323f602013d70573e55842a1e93d
S23 48d54b10a6d5eb0a4a15214949821a0a65fd884a9ce1bb55f254b0b59d06a80e
S24 f4b29b9d6d7fdb52e0402669c68ce897f9c61a859acce4ad466a3a92f040f5ea
S25 6e859a7a82ef34cb41a2621b71954207dc02786a70a9a1fbbf1fa8d52ee12b3d
S26 571772713d0cc78a3aa72f7eca298a95cd2cd45005233284b001cb69b7160525
S27 865ffd1a93409899a64ef813f610af46c1382456a051dc416a2da7fdd9075ed1
S28 67fe5d50274171e3cdc5383f4c38072da0e23432885d083aa87035b465de46f3
S29 7e8dcf6ef62c117b696a21c1e55f87055bcaadddb481e399232910e275b2517e
S30 af8d0e8ef5fc0c7cca76e27928a6d2ed4565088d1b8196c85563f69651f28c10
S31 563d744807816bf48f1026b229eed8d04834512ab383f8cc7c579a54e787307e
S32 d2f7c4b94073522a1e097a97da365ef459a9ad7f043399da4b4344f5677ece89
S33 da29f1fcf789fe0034e718245e398c454b3517244ae0af82c0818f9cc5caf34e
S34 f5adeaac15e6f4d7cced15b84d3150df8875dd097b595f54ce0cd8d095c77c3b
S35 e8801aa1309dd505df4ace859c2328cbc5c0c6a604a6bca31f1930446ca90863
S36 08dd47d8b91211824f58d8c7a0ca87486bd8d649cc2d57fb3164ac8bf1342492
S37 bac1d3c562c480a4036aa91213e9513e8ba178807f16a04caa7e3b0d405fe63f
S38 ec0e6eb99453eb2e2f20a69ec2764dfeb3f092826fbb4f2148174603125a2917
S39 2b10ac176e8515ae256a45c770c22b797af4862ffa7c2719b6b2899ca48572af
S40 61c70c87587ad01e382d629d3cec0c9d385feafd9e8e7938358daa9d3213dc0c
S41 d583a07efd3c61a4e6a99c523d2d1343cf58aec7e223e0e161557f1519121ac0
S42 9fd2d1a64a40d21dbcee97150e59933f64b9e6d073e841ba171c02ffbdb4ac9c
S43 cea879607f1dab3b95ac39f1b074775d303c1983843eca3719c468f2d8886f65
S44 e0f9f4a815e731f6fd7043b03209cc4af7b5bcd38245291fc1b8d128dd4b4f9d
S45 dfb6199500c059fc30083fdf88e353bb87d99f525c0859774882f67d4ffd0e35
S46 93cc839fd9dabf2026e743326d0230f8c1ffabfab102c1d09e37c91f5b4bf4ae
S47 9eb0dca0905fe1b55ded8dd9f8a1721b922b7d70e1b9446c75cb0050dc58e547
S48 5967a1e116e7101c7dfadc78aaf01bdda0552bf5bba044c9e22f622b560f526b
S49 e23613c3fbc699c999799e77545e5f16b5be26f647dcb51200d6a471321ab7a9
S50 baf6d68b023fe8f540636e465e3852663b56add04f7c92d3acc9a23fe7c8eab3
S51 53d8b35cd6d4480bcf54cf8a73eaaa40a010145ccb363f2bcbfed13a5b62fe9f
S52 fff69a047ad4a26571e5a8ae79b63516ed18ecc0d7a3ae105077a345f9a98659
S53 bcca25ea6685d42273b0bba692289f7967bba83f8c22955a48335f181d8ddad5
S54 939f06742a6e3c160d2995389e8b6128eefc94eeac86b86a5078e7876ad19494
S55 c8788a3550ae6acb2a963f0a87d01a184b0ec9b928ae4f2d1db34fce483a4789
S56 8ccd654a886abd7629ab7a72f6bdee793af851cd1b8ea245b3055c46d494b1b3
S57 e2a6c2233ae00d8842e4d6472330a717f828912e0f60f98c7943c598a2fa6c04
S58 71ae264f99d1f6ced5deec8e6652004dc4c3c9d3724e824d1cab1e4db1eaf4ed
S59 2d9e954ca03ed6081f30aad0c2a539e69108db551911e10ae18d15b4e6180ef7
S60 6155a86bca5d63628f0785e222e607657f4614892a3be8546093b672015a0180
S61 0912e392c93a32931b9bf6d0ff541a394431ae13c3cdc0bab6057925e1834b06
S62 c17de27a20a6fbbd4a395a53f2c18e03092a18b5f58ba09ea87f9b203ab40750
S63 932d3a12c247f281219432cba5f54e111a2f2b052d8762bc7b81e0e992ed52d8
S64 70d653ad0b6d5c1659bce8db9c8f511692cb16272a8b86ae621fd2d40a21ac50
S65 de8149563f409006991e82c88cba1bc9f14e712ee0a5665c23d07158dd824cec
```

## 审阅顺序与验证边界

1. 先看各节“直觉说明”，确定模块想保留什么、想去掉什么。
2. 对照输入单位、轴顺序、有效掩码和默认值，确认手中的数据处于正确阶段。
3. 按给出的函数及连续行范围检查公式，尤其是滤波方向、分位数常数、缺失值传播、峰间邻接关系、边界切片和失败分支。
4. 检查调用入口：底层函数存在、registry 注册、实际运行启用是三个不同事实。历史/专项路径与当前 finalcase 不应混为一谈。
5. 涉及训练数据统计的变换单独检查拟合人群；仅凭函数名带有 `fit` 或 `cache` 不能判断是否泄漏。

本文是当前实现的算法说明，不是算法有效性证明，也不是所有第三方库内部迭代代码的逐行复刻。未实现的 learned denoiser、外部后端的内部细节和未执行的可选分支测试，不通过文字说明视作已经验证。修改算法前，应另行确认研究目的和数值等价要求；阅读文档本身不会改变现有配置与结果。
