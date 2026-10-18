# 工具与工作人员命令

模型可调用search_docs、prepare_ticket、create_ticket、get_ticket、find_ticket_by_request。知识库归属由程序绑定，模型不能传入另一个kb_id跨库检索。工单审批和工作人员更新状态不开放为模型工具。

使用下列命令查看本机入口参数：

```powershell
uv run python ticket_cli.py --help
uv run python mcp_demo.py --help
```

工作人员通过当前进程中的TICKET_USER_ID、TICKET_WORKSPACE_ID、TICKET_ROLE确定本地身份。必须使用与目标工单相同的空间；角色admin才能修改状态。

```powershell
$env:TICKET_ROLE = 'admin'
uv run python ticket_cli.py set-status T-000001 处理中
```

具体子命令以ticket_cli.py --help输出为准。状态只允许待处理→处理中→已解决，同一状态重复提交允许，不能跳过或倒退。操作后恢复自己的普通角色设置。网页Agent需重新调用get_ticket获取实时状态。

本版不附带任何已有工单。先在网页准备草稿并确认创建，取得自己的正式工单编号后再进行演示。
