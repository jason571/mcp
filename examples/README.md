# MCP connection examples

English | [简体中文](README.zh-CN.md)

Each JSON file is a separate connection template, not a running service:

- [FastVideo H3](fastvideo-h3.example.json)
- [MiniMax H3](minimax-h3.example.json)
- [Media Workflow](media-workflow.example.json)
- [YuE2](yue2.example.json)
- [Z-Image](z-image.example.json)
- [Chrome bridge](chrome.example.json)
- [Hosted Tavily](tavily.example.json)

Set the URL to the actual reachable MCP endpoint and replace the token through your client's credential configuration. `YOUR_MCP_TOKEN` and `YOUR_TAVILY_API_KEY` are not usable credentials. Keep completed configurations private. For an authenticated service, the header must match its expected format, including a single `Bearer ` prefix where required.

FastVideo and MiniMax examples use the same illustrative port: deploy either service there or assign distinct ports. The component configuration determines the actual listening port. Check `Host`/origin allowlists when using a reverse proxy.

Tavily's remote endpoint and authentication are described in its [official documentation](https://docs.tavily.com/documentation/mcp). The example uses a placeholder Authorization header rather than embedding a real key in a URL.

For weather/stdin-stdout transport and the optional workspace broker, follow the component documentation and the client's transport/runtime requirements; there is no one universal HTTP configuration for every component.
