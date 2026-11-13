# 技术支持工单 Agent

在本地对话中查询产品资料、登记故障并跟踪工单。项目使用 Python、LangChain、MCP、Chroma 和 SQLite，提供 Windows 单用户网页与命令行入口。

**[下载演示视频（MP4 · 44 秒 · 无解说）](https://github.com/Ri4o3c/ticket-agent/releases/download/demo-video-v1/ticket-agent-demo.mp4)**

视频展示 PDF 报错问答、查看引用、人工确认建单、查询状态和查看执行记录。录制使用模拟模型和检索，审批、写入与查询走项目代码和独立的演示数据库。

## 功能与执行流程

- **知识库问答** · 支持 TXT、Markdown、文字型 PDF 和 JSONL。每个知识库有独立的 Chroma collection 和文件清单。会话绑定所选知识库，切库时开启新对话。回答可查看检索原文。
- **工单审批** · 模型选择 MCP 工具并准备 `D-` 草稿，Python 记录用户的确认，SQLite 校验审批后写入 `T-` 正式工单。重试沿用原请求 ID，创建结果未知时先查询原请求。
- **状态查询与执行记录** · 工单进度从 SQLite 读取。工作人员可通过本地命令更新状态。网页可查看历史执行及模型、工具、程序校验和人工等待步骤。

## 本地运行

已验证环境为 Windows、Python 3.14 和 uv。网页使用 Windows DPAPI 保存 API Key，默认监听 `127.0.0.1`。

```powershell
uv sync --locked
uv run python web_app.py
```

打开 [本地工作台](http://127.0.0.1:8765)，在「服务配置」分别填写聊天与 Embedding 服务的 API 根地址、模型和 Key。接口需兼容 OpenAI Chat Completions 或 Embeddings 格式，聊天模型需支持工具调用。Embedding 维度可留空，使用模型默认值。可参考 [.env.example](.env.example) 配置环境变量。

在「知识库」上传 [示例资料](examples/knowledge.md)，选择「使用此库聊天」，然后输入「上传 PDF 时提示 E103，应该怎么处理？」。单文件上限 10 MB、2,000 个分块，每块最多 4,000 字。扫描件先在外部做 OCR。当前 MinerU 设置支持保存地址和 Token，解析接口仍待接入。

<details>
<summary>命令行演示</summary>

以下命令将教学分块显式导入默认知识库，再启动对话。若存在自备的 `data/knowledge_chunks.jsonl`，入库时优先使用该文件。输入 `END` 提交，输入 `EXIT` 退出。

```powershell
uv run python rag_index.py --build
uv run python agent_step1.py
```

工作人员操作见 [工具指南](TOOLS_GUIDE.md)。

</details>

## 验证记录

- **软件测试** · 2026-10-24 记录 186 项通过，使用临时数据、固定模型回复或本地计算向量。见 [测试报告](TEST_REPORT.md)。
- **真实模型案例** · E04（已有工单的状态查询）和 E05（查询不存在或无权访问的工单）通过人工复核。E08（取消建单后继续对话）完成取消和后续查询，草稿语义仍待复核。见 [评估说明](EVALUATION.md) 和 [项目进度](PROGRESS.md)。
- **运行范围** · 公开版本面向本机单用户学习。多用户登录、生产运行和整体模型质量尚待验证。

```powershell
uv run python -B -m unittest discover -s tests -v
```

## 代码入口

| 入口 | 职责 |
| --- | --- |
| [web_app.py](web_app.py)、[web/](web/) | 本地接口和网页 |
| [agent_step1.py](agent_step1.py)、[mcp_tools.py](mcp_tools.py) | 模型工具循环、审批与 MCP 工具 |
| [knowledge_upload.py](knowledge_upload.py)、[rag_index.py](rag_index.py) | 文件入库、分块和检索 |
| [ticket_store.py](ticket_store.py)、[ticket_cli.py](ticket_cli.py) | SQLite 工单、审批和状态流转 |
| [agent_trace.py](agent_trace.py)、[trace_store.py](trace_store.py) | 执行采集和查询 |
| [eval_one_case.py](eval_one_case.py)、[eval_cases.json](eval_cases.json) | 单例评估入口与案例定义 |

可靠性修复见 [修复记录](FIXES_20261020.md)，执行记录设计见 [可观测性与评估](OBSERVABILITY_AND_EVALUATION.md)。仓库提供虚构教学资料和配置模板。运行生成的配置、资料、工单数据库和向量库保留在本机。
