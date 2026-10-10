# Licensing and upstream sources

Licenses apply per component. Included upstream license files and notices remain authoritative; this repository does not relicense the collection as a whole.

| Component / resource | Source / license reference |
| --- | --- |
| Chrome MCP bridge | [hangwin/mcp-chrome](https://github.com/hangwin/mcp-chrome); included MIT licenses |
| Chrome ONNX Runtime worker assets | [Microsoft ONNX Runtime](https://github.com/microsoft/onnxruntime); [MIT license](mcp-chrome/app/chrome-extension/workers/ONNX_RUNTIME_LICENSE.txt) |
| Weather Forecast Server | [shibing624/weather-forecast-server](https://github.com/shibing624/weather-forecast-server); included Apache-2.0 license |
| Z-Image MCP wrapper | Included [MIT license](z-image-mcp/LICENSE); model terms are separate |
| MiniMax H3 model | [MiniMax H3 model license](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE); weights are not distributed here |
| YuE2 model | [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B); inference package, main model and VAE terms are separate |
| Z-Image model | [Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo); weights are not distributed here |

For components without an explicit license file, this publication does not imply an additional redistribution grant. Check the relevant rights before reusing or redistributing them. Dependency licenses are supplied by their own packages; installing a dependency or model does not change its terms.

Private deployment configs, credentials, model files, state databases and generated outputs are excluded. Public examples use placeholders. The Chrome worker WebAssembly files are retained runtime resources, not generated task artifacts.

## 中文说明

本仓库按组件保留许可证，不用统一许可证替代第三方条款。模型、依赖包与服务包装代码分别适用自己的许可；模型权重不随仓库分发。没有明确许可证的组件，不因上传仓库而获得额外的再分发授权。实际部署配置、密钥、数据库与生成文件均不进入仓库。
