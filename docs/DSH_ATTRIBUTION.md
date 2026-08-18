# DSH Attribution

本项目的以下设计参考/迁移自 [deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness)
("Everything is a Plugin",MIT License):

- 交互模式:可观测轨迹/会话式指令框/工作区管理/配置面板的 UI 结构参考其 Web 前端
- 架构模式:工具注册(manifest)、事件驱动轨迹、后台任务、沙箱三档权限分级
  等 harness 工程模式

未直接复制其源码(前端为 React/TypeScript,本项目用原生 JS 单页实现;
后端为独立 Python 实现)。

```
MIT License

Copyright (c) 2025-2026 DeepSeek-AI (deepseek-harness)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```
