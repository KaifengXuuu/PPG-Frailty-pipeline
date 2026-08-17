# 扫描协议与证据要求 / Scan Protocol and Evidence Requirements

## 1. 扫描范围 / Scope

1. 根目录代码和文本文件逐文件、逐字节完整读取。
2. 从代码中提取并记录全部可识别的输入路径、输出路径、输出内容及输出结构。
3. 输入路径逐文件读取头部内容，识别数据结构、格式、列、shape、采样相关字段和异常。
4. 输出路径中的文本文件逐字节完整读取；非文本文件登记文件名、格式、大小及可安全取得的结构元数据。
5. 对比代码声明的输出 schema 与实际输出，建立代码—输入—输出对应关系。
6. 从结果中提取论文所需关键指标、实验条件、失败信息、可比性和方法评价。
7. 保留 workspace 文件树、输出目录树和主要内容说明。

## 2. 完整读取证据 / Full-read evidence

- 完整文本读取记录 `byte_count`、`sha256`、编码状态与行数。
- 输入头部读取记录读取字节数、头部 SHA-256、格式识别结果与脱敏预览。
- 二进制输出默认不读取载荷；记录路径、扩展名、大小和必要结构元数据。
- `.env` 可参与不可逆 SHA-256 校验，但任何值均不得输出或写入记录。
- 读取失败必须记录异常类型，不得默认为扫描成功。

## 3. 分类与状态 / Classification and status

方法状态统一使用：

- `strictly_validated`：已实现并有严格、无明显泄漏的验证证据。
- `implemented_unverified`：已实现，但未找到足够验证证据。
- `smoke_only`：仅证明流程可运行。
- `failed_or_deprecated`：已有失败证据或已明确废弃。
- `not_implemented`：只存在计划或接口描述。
- `unknown`：证据不足，必须人工确认。

记录来源统一区分 `code`、`output`、`project_record`、`user_confirmed` 与 `inferred`。

## 4. 隐私与安全 / Privacy and security

原始生理数据的头部预览只保留理解 schema 所需的最小片段；可能包含身份信息、凭据或私密字段时必须脱敏。不得把密钥、token、账号或完整原始数据复制到 `final_v0`。

