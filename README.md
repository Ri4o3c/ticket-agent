# 技术支持工单 Agent

一个Windows本地单用户学习项目：Python、LangChain、MCP、Chroma和SQLite，附带原生HTML/CSS/JavaScript页面。

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

命令行可显式把100条教学分块导入默认知识库：

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
| examples/ | 无私人路径的虚构教学资料 |

## 上传GitHub

此目录已排除原作者的.env、加密配置、上传资料、工单数据库、向量库、虚拟环境、IDE配置及Git历史。请上传本目录内的源码文件，保留.gitignore；后续自己运行生成的data/、chroma_db/与.env也应继续忽略。

本项目默认只监听127.0.0.1，适合本机学习使用。
