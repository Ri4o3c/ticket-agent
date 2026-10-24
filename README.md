# 技术支持工单 Agent

一个Windows本地单用户学习项目：Python、LangChain、MCP、Chroma和SQLite，附带原生HTML/CSS/JavaScript页面。

2026-10-24更新：增加可靠性修复、执行记录存储与查询页面，以及E04/E05/E08自动评估。
当前186项软件测试通过；3条案例已实际运行聊天模型，其中2条人工复核通过，E08草稿语义仍待复核。
这是当前验证范围，不代表已完成企业生产验收；每位使用者仍需自行填写服务配置。

2026-10-20可靠性修复见FIXES_20261020.md；项目进度见PROGRESS.md；可观测性与评估的解释和下一阶段目标见OBSERVABILITY_AND_EVALUATION.md。

2026-10-21已实现执行记录存储与Agent自动采集，见[存储练习](TRACE_STORE_STEP1.md)和[Agent接入说明](TRACE_AGENT_STEP2.md)。可运行`uv run python trace_demo.py`独立练习1条run与3条step；真实CLI/Web执行自动写入记录库。

2026-10-22增加[历史执行查询](TRACE_QUERY_STEP3.md)：页头“执行记录”查看近期记录或按run_id查询，每个问题的处理过程也有对应入口。模型、工具和人工等待分别展示；执行结束不等于评估通过。该步全套132项功能测试通过，尚未完成真实模型质量评估。

同日增加[首条自动评估练习](EVALUATION_STEP1.md)：`uv run python eval_one_case.py`自动准备临时数据，执行E05并输出逐项评分。默认使用固定模型验证评分流程；加入`--scenario no-query`可看到“未查询就回答”如何被判为失败。当时新增13项测试后全套145项通过，不代表真实模型质量已测得。

已增加[真实模型运行入口](EVALUATION_STEP2.md)：配置当前工程的聊天API后，运行`uv run python eval_one_case.py --live`。不需要Embedding或MinerU；默认命令仍是固定模型模式。新增7项测试后全套152项通过。入口代码使用受控HTTP离线验证，真实模型结果需实际运行报告确认。

2026-10-23增加[E04已存在工单评估](EVALUATION_STEP3.md)：`uv run python eval_one_case.py --case E04 --live`。评分标准来自调用前准备的临时工单。新增14项后全套166项测试通过；E04/E05均已真实运行，单次结果和人工复核情况见PROGRESS.md。

同日增加[E08取消后继续对话评估](EVALUATION_STEP4.md)：`uv run python eval_one_case.py --case E08 --live`。18项新增测试后，隔离回归184项通过。真实E08完成取消和后续查询，自动结果needs_review，等待草稿语义人工复核；软件测试数量不等于真实模型案例数量。

**本版本不提供默认API地址、模型名称、向量维度或Key。所有服务参数均由使用者填写。**

## 可以做什么

- 创建并命名多个独立知识库，上传TXT、Markdown、文字型PDF或JSONL。
- 每库独立Chroma collection及文件清单；聊天会话绑定所选库，切库开启新对话。
- 根据知识资料回答并查看引用原文。
- 准备工单草稿，经用户明确确认后写入正式工单，再查询真实状态。
- 通过本地工作人员命令推进工单状态，保留人工审批和幂等创建规则。
- 分别配置聊天、Embedding及MinerU服务。

MinerU当前只保存地址和Token，尚未调用解析接口；扫描件和复杂PDF请先在外部解析，再上传Markdown。示例产品“知库助手1.0”为虚构教学内容。

## 环境与启动

当前验证环境为Windows、Python 3.14。网页保存Key使用Windows账户DPAPI，因此该功能需要Windows。用户自行安装Python及uv。

在仓库根目录运行：

```powershell
uv sync --locked
uv run python web_app.py
```

然后打开浏览器中的本地地址 `http://127.0.0.1:8765`。首次启动自动打开空白“服务配置”表单，未配置时不会启动模型会话或发送API请求。

也可在自己创建的虚拟环境中安装pyproject.toml所列依赖后运行 `python web_app.py`，无需Node.js或前端构建。

## 首次配置

填写以下参数并保存：

| 服务 | 必填参数 |
| --- | --- |
| 聊天模型 | 自己的Base URL、支持工具调用的模型名称、API Key |
| Embedding | 自己的Base URL、模型名称、API Key；维度可留空使用模型默认值 |
| MinerU | 可选地址与Token；本版仅保存参数 |

模型接口须兼容OpenAI Chat Completions或Embeddings格式。Base URL填写API根地址，不是聊天网页链接，也不是完整的/chat/completions或/embeddings请求地址。无需鉴权的本地服务可用local作为占位Key。

地址未变时Key留空保留原值；更换地址且已有Key时，需要同时填写对应Key。保存后开启新会话。更换Embedding模型、地址或维度会使用独立索引，需重新导入资料，旧索引不删除。

Key加密保存在本机data/service_settings.json，网页不回填明文。也可自行将.env.example复制为.env，填写通用变量；代码不读取旧版DASHSCOPE变量，也没有服务商配置回退。加密文件不可跨电脑直接复用。

## 上传并提问

1. 在“知识库”面板新建或选择目标库。
2. 上传自己的资料；单文件最大10MB，最多2000块，每块最多4000字。
3. 显示“可检索”后，选择“使用此库聊天”，或在主页面选择对应库。
4. 输入问题。Enter发送，Shift+Enter换行；三条快捷提示只填入问题。

可先上传examples/knowledge.md体验虚构产品问答。示例资料不自动入库，不会在启动时调用付费服务。

命令行可显式把100条教学分块导入默认知识库。全新仓库读取examples/knowledge_chunks.jsonl；若自己准备了data/knowledge_chunks.jsonl，则优先读取该文件：

```powershell
uv run python rag_index.py --build
uv run python agent_step1.py
```

命令行单独输入END提交问题，EXIT退出。评估问题位于eval_cases.json，不应上传进知识库。

## 工单确认与工作人员操作

Agent的模型负责选择工具，Python经MCP执行，再把结果交回模型。Embedding只负责向量化。D-编号是草稿，T-编号是正式工单；只有创建工具返回成功和正式编号才能确认已创建。

网页确认按钮由程序记录审批，模型不能自行批准。创建重试沿用原草稿和请求ID；结果未知时先用find_ticket_by_request核查。

工作人员命令见TOOLS_GUIDE.md。本项目为单用户本地学习版，身份来自本机配置，不包含真实用户登录系统。

## 自动测试

在仓库根目录运行：

```powershell
uv run python -B -m unittest discover -s tests -v
```

自动测试使用临时数据、固定模型回复或本地计算向量，不使用真实付费API。软件测试范围和发布包检查见TEST_REPORT.md。

## 主要文件

| 文件 | 用途 |
| --- | --- |
| web_app.py、web/ | 本地Web接口和页面 |
| agent_step1.py | Agent工具循环、会话和人工确认 |
| mcp_tools.py | 五个MCP工具，按会话绑定知识库 |
| service_settings.py | 用户服务参数、就绪检查、Key加密 |
| knowledge_bases.py | 知识库名称与ID清单 |
| knowledge_upload.py、knowledge_preprocess.py | 上传、清洗分块、去重、入库 |
| rag_index.py、rag_answer.py | 向量检索和独立RAG问答 |
| ticket_store.py、ticket_cli.py | 工单存储、审批、状态流转 |
| trace_store.py、agent_trace.py、trace_demo.py | 执行记录存储、Agent采集适配及独立模拟练习 |
| eval_cases.json、eval_one_case.py、eval_cancellation.py | 10例评估定义与E04/E05/E08执行和独立规则评分 |
| examples/ | 无私人路径的虚构教学资料 |

## 上传GitHub

仓库不包含原作者的.env、加密配置、上传资料、工单数据库、向量库、虚拟环境和IDE配置。请保留.gitignore；后续自己运行生成的data/、chroma_db/与.env也应继续忽略。examples/只有虚构教学资料。

本项目默认只监听127.0.0.1，适合本机学习使用。
