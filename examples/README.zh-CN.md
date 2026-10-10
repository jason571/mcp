# MCP 连接示例

[English](README.md) | 简体中文

每个 JSON 文件都是独立的连接模板，不会启动服务：

- [FastVideo H3](fastvideo-h3.example.json)
- [MiniMax H3](minimax-h3.example.json)
- [Media Workflow](media-workflow.example.json)
- [YuE2](yue2.example.json)
- [Z-Image](z-image.example.json)
- [Chrome 桥接](chrome.example.json)
- [Tavily 托管服务](tavily.example.json)

URL 改为实际可访问的 MCP 地址，在客户端凭据配置中填写真实密钥。`YOUR_MCP_TOKEN` 和 `YOUR_TAVILY_API_KEY` 只是占位值。填写完成的配置保存在本地，不提交到 Git。认证 Header 必须与服务要求一致，需要 Bearer 的服务只保留一个 `Bearer ` 前缀。

FastVideo 和 MiniMax 示例使用相同端口作为示意；部署时选择其中一个，或分别设置不同端口。实际监听地址以组件配置为准。使用反向代理时同步核对 Host 与来源白名单。

Tavily 托管地址与认证方式见[官方文档](https://docs.tavily.com/documentation/mcp)。示例通过占位 Authorization Header 表达密钥，不把真实密钥写入 URL。

天气组件的 stdio 传输及独立 Workspace Broker 按各自文档和客户端能力配置，不套用统一 HTTP 示例。
