# TraceLens

📹 面向长视频的可追溯 AI Agent，支持证据检索、工具调用、决策链可视化。

一个生产级的视频问答系统，实现了从视频上传、自动转录、语义检索到 Agent 推理的完整闭环。

## ✨ 核心特性

- **多策略检索**：语义检索（Qdrant + SentenceTransformers）+ 关键词检索的混合方案
- **结构化 LangGraph Agent**：Planner → Retriever → Verifier → Writer → Critic 工作流，支持条件路由、受限修订和拒答保护
- **工具调用系统**：显式工具调用（semantic_search, keyword_search, verify_coverage）并追踪
- **自动转录**：FFmpeg + faster-whisper 实时将视频转换成带时间戳的证据
- **决策链可视化**：前端展示 Agent 每一步的思考过程和工具调用
- **证据引用**：每条答案都附带精确的时间戳和来源
- **无证据拒答**：当证据不足时，Agent 会主动拒绝回答而非幻觉
- **性能监控**：`/api/metrics` 端点提供系统统计和 Agent 能力展示

## 🏗️ 技术栈

| 组件 | 技术 |
|------|------|
| 后端框架 | FastAPI + Uvicorn |
| Agent 框架 | LangGraph |
| 向量数据库 | Qdrant（Docker 持久化卷） |
| 文本向量化 | BGE-small-zh-v1.5 INT8 ONNX（CPU） |
| 视频处理 | FFmpeg |
| 语音转写 | faster-whisper (Tiny model, CPU-optimized) |
| 持久化存储 | SQLite |
| 前端框架 | Vue 3 + Vite |
| 测试框架 | pytest |

## 🚀 快速开始

### 环境要求

### 后端启动

```powershell
cd backend
pip install -r requirements.txt
Copy-Item .env.example .env
python -m alembic upgrade head
python scripts/check_database.py
# 编辑 .env，填写 DEEPSEEK_API_KEY 和 DEEPSEEK_MODEL
python -m pytest -q              # 运行测试（不调用真实 DeepSeek）
uvicorn tracelens.main:app --reload --port 9090
```

DeepSeek 配置保存在 `backend/.env`，后端启动时会自动读取，不需要每次重新输入。`.env` 已加入 Git 忽略规则，不能提交真实 API Key。默认使用 DeepSeek 的 OpenAI 兼容地址 `https://api.deepseek.com/v1`，模型为 `deepseek-chat`。

程序默认不读取系统 HTTP 代理（`DEEPSEEK_TRUST_ENV=false`），适合本机存在失效代理配置的情况。如果你的网络必须经过代理，请在 `.env` 中设置 `DEEPSEEK_TRUST_ENV=true`，或填写 `DEEPSEEK_PROXY=http://代理地址:端口`。

未填写 `DEEPSEEK_API_KEY` 时，系统仍可使用本地检索和模板回退回答；填写后 `/api/ask` 首次提问会调用 DeepSeek，并校验返回的引用只能来自检索到的证据。已有会话的追问只调用追问生成器，返回自然语言段落。

上传视频时，系统会先用 FFmpeg 抽取音频，再由 faster-whisper 转写。视频没有可识别语音、文件损坏或转写失败时，接口会返回明确错误，不会写入伪造证据。

打开 http://127.0.0.1:9090/docs 查看 API 文档。

### 前端启动

```powershell
cd frontend
npm install
npm run dev
```

打开 http://127.0.0.1:5173 使用工作台。

### 一键完整容器栈（建议在本机基础设施验证完成后使用）

根目录的 `docker-compose.yml` 会同时启动 MySQL、Redis、Qdrant、RocketMQ、数据库迁移、API、Linux Worker 和 Nginx 前端。浏览器只需要访问 `http://127.0.0.1:8080`；API、数据库和消息队列不会直接暴露给浏览器。

首次切换前，先确认 `backend/.env` 已填写真实的 `DEEPSEEK_API_KEY` 和高强度 `JWT_SECRET`。当前若正在运行 `docker-compose.infra.yml`，它会占用 Redis/Qdrant 的 7000/7001 端口；请先执行下列命令停止旧的基础设施容器（不加 `-v`，已有 Docker 数据卷不会删除），再启动完整栈：

```powershell
docker compose -f docker-compose.infra.yml down
docker compose up --build -d
docker compose ps
```

完整栈中的 MySQL 是 Docker 自己的数据卷，和你当前 Windows 本机 `3306` 上的 MySQL 是两套独立数据。为避免端口冲突，它只映射到宿主机 `3307`，而 API 在容器内部通过 `mysql:3306` 连接它。第一次启动会自动运行 Alembic 迁移；查看状态可用：

```powershell
docker compose logs -f migrate api worker
```

如果你的网络可以直接访问 Docker Hub，可在启动前设置 `DOCKER_IMAGE_REGISTRY=docker.io`；默认使用已验证的 `docker.m.daocloud.io` 加速源。这个设置只影响项目镜像构建，不会修改 Docker Desktop 的全局镜像配置。

后端镜像默认使用 `mirrors.aliyun.com` 下载 Debian 系统依赖，并对偶发的 502 做三次重试。网络可直连 Debian 时，可设置 `DEBIAN_MIRROR=deb.debian.org`。默认运行的是 CPU ONNX 向量模型，不会下载 PyTorch/GPU 组件；只有显式改用旧的 `sentence-transformers` 后，才需要额外安装 `backend/requirements-legacy-embeddings.txt`。

Windows 无法运行 RocketMQ 的 Python 客户端，所以此前本机 API 会退回线程池；完整容器栈中的 `worker` 是 Linux，能够真正消费 RocketMQ 任务。这正是容器化在这个项目里最实际的价值之一。

完整栈默认使用参考项目同款的固定版本 BGE-small-zh-v1.5 INT8 ONNX 模型。模型不写入 Git，也不打进镜像，而是首次运行时保存到 `app_data` volume；如果模型下载失败，系统会继续使用关键词/上下文检索，数据库和 API 不会因此停止。

## 📋 API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/auth/register` | POST | 注册用户并返回 JWT |
| `/api/auth/login` | POST | 登录并返回 JWT |
| `/api/auth/me` | GET | 获取当前用户 |
| `/api/evidence` | GET/POST | 证据列表和创建 |
| `/api/ask` | POST | 提问（执行 Agent 工作流） |
| `/api/videos/upload` | POST | 上传视频（自动转录） |
| `/api/videos` | GET | 查看已保存的视频资源 |
| `/api/videos/{video_id}` | DELETE | 删除视频、证据和检索索引 |
| `/api/videos/{video_id}/memory` | GET | 查看该视频的会话、消息和历史报告 |
| `/api/demo/seed` | POST | 加载演示数据 |
| `/api/metrics` | GET | 系统指标和 Agent 能力 |

## 🧪 测试

```powershell
cd backend
pytest -q                        # 运行测试，不调用真实 DeepSeek
pytest -v                        # 详细输出
pytest tests/test_api.py::test_ask_includes_trace -v  # 单个测试
```

**测试覆盖：**
- ✅ 健康检查
- ✅ 证据管理（CRUD）
- ✅ 无证据拒答
- ✅ 视频上传和转录
- ✅ Agent 决策链追踪
- ✅ 会话复用、持久化和恢复
- ✅ 视频删除时清理会话记忆
- ✅ 系统指标

## 💡 使用示例

### 1. 加载演示数据

```bash
curl -X POST http://127.0.0.1:9090/api/demo/seed \
  -H "Content-Type: application/json" \
  -d '{"video_id": "demo-video"}'
```

### 2. 提问（触发 Agent 工作流）

```bash
curl -X POST http://127.0.0.1:9090/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "项目式学习的特点是什么？",
    "video_id": "demo-video"
  }'
```

**响应示例：**

```json
{
  "question": "项目式学习的特点是什么？",
  "answer": "根据检索到的视频证据，相关内容包括：视频先介绍了项目式学习：先做一个能运行的小项目，再围绕项目补齐知识。",
  "grounded": true,
  "citations": [
    {
      "evidence_id": 1,
      "timestamp": "00:00 - 00:42",
      "text": "视频先介绍了项目式学习：先做一个能运行的小项目，再围绕项目补齐知识。"
    }
  ],
  "trace": [
    "Retrieve: Found 3 evidence pieces",
    "Tool: semantic_search returned 3 results",
    "Verify: Coverage: 100.0% token overlap (adequate=true)",
    "Answer: Generated grounded=true"
  ]
}
```

### 3. 上传视频

```bash
curl -X POST http://127.0.0.1:9090/api/videos/upload \
  -F "video_id=my-video" \
  -F "file=@video.mp4"
```

### 4. 查看系统指标

```bash
curl http://127.0.0.1:9090/api/metrics | jq
```

## 🎯 Agent 工作流

```
用户提问
  ↓
[Planner Node]
  ├─ 理解问题意图
  └─ 拆分必须验证的证据需求
  ↓
[Retriever Node]
  ├─ 按证据需求检索 ASR/OCR 时间轴
  └─ 建立 Evidence Ledger（证据台账）
  ↓
[Verifier Node]
  ├─ 逐个证据槽位验证支持关系
  └─ 判断证据是否充分、是否需要拒答
  ↓
[Writer Node]
  └─ 只使用通过验证的 evidenceId 生成报告
  ↓
[Critic Node]
  ├─ 检查完整性、矛盾、外部知识和引用
  └─ 未通过时回到 Retriever 补充证据，再经过 Verifier 和 Writer 修订一次
  ↓
返回给用户（包含 Agent 图、证据台账和引用）
```

## 与参考项目的对应关系

当前项目已对齐参考项目的核心 Agent 结构，概念对应如下：

| 参考项目概念 | 当前项目实现 | 说明 |
|---|---|---|
| `Media` | `videos` SQLite 表 + `data/uploads/` | 保存视频资源、哈希、文件路径和转录状态 |
| `EvidenceSegment` | `evidence` SQLite 表 | 保存带开始/结束时间的转录证据 |
| `AgentToolbox` | `backend/tracelens/agent.py` 中的 `AgentToolbox` | 提供元数据、时间轴检索、证据窗口和报告工具 |
| `agent_structured` | `backend/tracelens/agent_structured.py` | 完整 Planner → Retriever → Verifier → Writer → Critic 工作流，使用 Evidence Ledger 和 Critic 质量门禁 |
| `agent_graph` | `backend/tracelens/agent_graph.py` | 保留旧版 Retrieve → Verify → Answer 兼容流程 |
| `generate_report` | Agent 工具的结构化报告提交 | 后端校验引用 ID 和证据覆盖后才接受 |
| Qdrant 检索 | `backend/tracelens/qdrant_retrieval.py` + `retrieval.py` | 向量不可用时自动降级为关键词检索 |
| Agent 会话记忆 | 已启用 | SQLite 持久化会话、消息、结构化报告和工具轨迹；最近消息可按配置注入 DeepSeek |
| Bilibili 导入 | `backend/tracelens/bilibili.py` + `/api/media/bilibili/*` | BV 预览、异步下载、复用现有 ASR/OCR 流程 |
| MCP Server | `backend/tracelens/mcp_server.py` | Streamable HTTP/stdio，暴露视频、证据和导入工具 |
| Skills | `skills/*/SKILL.md` | 课程笔记、操作指南、会议复盘、证据审计 |
| MySQL / Redis / RocketMQ | `docker-compose.infra.yml` | 已提供基础设施与连接配置；当前存储仍默认 SQLite |

学习参考项目时，建议优先关注这条数据流：

```text
问题 -> 模型选择工具 -> 后端执行工具 -> 工具结果回传模型
     -> 模型提交报告 -> 后端校验引用 -> 返回答案
```

当前项目已经实现参考项目中的核心 Agent 数据流，并继续补齐工程化能力：会话持久化、报告留档、工具轨迹、Bilibili 导入、MCP 工具层和可选 RocketMQ Worker。SQLite 仍是当前默认存储；`docker-compose.infra.yml` 可启动 MySQL（数据库名 `tracelens`）、Redis、RocketMQ 和 Qdrant，但 MySQL 存储迁移仍是下一阶段的生产化工作。

### Bilibili、MCP 与基础设施

```powershell
# 启动 MySQL、Redis、RocketMQ 和 Qdrant
docker compose -f docker-compose.infra.yml up -d

# API
cd backend
pip install -r requirements.txt
uvicorn tracelens.main:app --reload --port 9090

# MCP（另开终端）
python -m tracelens.mcp_server --transport streamable-http --host 127.0.0.1 --port 8001

# 启用 RocketMQ 后再启动 Worker
python -m tracelens.worker
```

`POST /api/media/bilibili/preview` 用于校验 BV 号并读取元数据；`POST /api/media/bilibili/import` 返回异步任务 ID，使用 `GET /api/media/bilibili/import-status?task_id=...` 查询导入和转录状态。首次使用 Bilibili 需要安装 `yt-dlp`，并确保本机可用 FFmpeg。

登录接口使用 PBKDF2 密码哈希和 HS256 JWT。默认启用 `AUTH_REQUIRED=true`；开发时如需使用无账户演示数据，可在本地 `.env` 显式设置为 `false`。JWT 包含唯一 `jti`，退出登录会撤销令牌；Redis 可用时负责令牌撤销和登录限流，不可用时限流退回单进程内存。部署前必须替换 `JWT_SECRET`。

### 会话记忆

每次 /api/ask 都会创建或复用 session_id。系统会把用户问题、Agent 回答、结构化报告和工具调用轨迹写入 data/agent.sqlite3，重启后仍可通过 /api/videos/{video_id}/memory 恢复。前端会自动携带当前会话 ID，并展示该视频最近一次会话。

由于历史消息发送给 DeepSeek 属于数据外发，配置项 DEEPSEEK_SEND_SESSION_HISTORY 控制是否将最近 12 条消息加入模型上下文。backend/.env.example 默认关闭；将它设为 true 后才会把历史消息发送给 DeepSeek，设为 false 时历史仍保存在本地，但仅用于页面恢复和审计。

## 📊 项目统计

- **代码组织**：`backend/tracelens/` 按 Agent、检索、媒体、存储和 API 职责拆分
- **测试覆盖**：14 个回归测试，100% 通过
- **依赖库数**：10+ 主要库（FastAPI、LangGraph、Qdrant 等）
- **前端组件**：Vue 3 单页应用，实时交互式 UI

## 🔍 目录结构

```
video-evidence-agent/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   └── main.py              # 兼容入口，实际应用在 tracelens/
│   ├── tracelens/
│   │   ├── main.py              # ASGI 入口
│   │   ├── api.py               # FastAPI 路由和应用组装
│   │   ├── agent.py             # AgentToolbox、DeepSeek 和引用校验
│   │   ├── agent_graph.py       # 旧版 LangGraph 兼容工作流
│   │   ├── agent_structured.py # Planner -> Retriever -> Verifier -> Writer -> Critic
│   │   ├── retrieval.py         # 关键词、向量和 Qdrant 检索
│   │   ├── embedding_retrieval.py # Embedding 后端边界
│   │   ├── qdrant_retrieval.py  # Qdrant 存储边界
│   │   ├── runtime_retrieval.py # 运行时检索策略选择
│   │   ├── ocr_runner.py        # FFmpeg + faster-whisper 转写
│   │   ├── storage.py           # SQLite 和会话持久化
│   │   ├── media.py             # 视频资源校验
│   │   ├── config.py            # 环境变量和运行时路径
│   │   └── models.py            # 请求模型和领域模型
│   ├── tests/
│   │   └── test_api.py          # 14 个 pytest 测试
│   ├── requirements.txt
│   └── pytest.ini
├── frontend/
│   ├── src/
│   │   ├── App.vue              # 主工作台（Trace 可视化）
│   │   ├── api.js               # API 客户端
│   │   └── style.css            # Agent trace 样式
│   ├── index.html
│   ├── package.json
│   └── vite.config.js
├── data/                         # 运行时数据（git ignore）
│   ├── agent.sqlite3            # 证据存储
│   ├── uploads/                 # 上传的视频
│   └── qdrant/                  # 向量数据库
├── PROJECT_CONTEXT.md           # 详细项目背景
└── README.md
```

## 🛠️ 开发指南

### 添加新工具

在 `backend/tracelens/agent.py` 的 `AgentToolbox` 中定义新工具：

```python
AVAILABLE_TOOLS = {
    "my_tool": {
        "description": "Tool description",
        "params": ["param1", "param2"],
    },
}

def my_tool_function(param1: str, param2: str) -> dict[str, Any]:
    # 实现逻辑
    return {"result": "..."}
```

然后在 DeepSeek 工具循环或 `backend/tracelens/agent_graph.py` 的节点中调用它。

### 自定义 Agent 节点

编辑 LangGraph 工作流（`build_agent_graph()` 函数）：

```python
def custom_node(state: AgentState) -> dict[str, Any]:
    # 你的逻辑
    return {"trace": [...]}

def build_agent_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("custom", custom_node)
    workflow.add_edge("previous_node", "custom")
    # ...
    return workflow.compile()
```

## 📈 性能指标

系统在本机测试环境中的表现：

| 指标 | 数值 |
|------|------|
| 文本检索延迟 | <100ms |
| 向量相似度计算 | <50ms |
| 完整 Agent 循环 | ~200ms |
| 视频转录速度 | ~1x（实时速度） |
| 单次上传处理 | ~1-3s（含转录） |

## 🎓 适合作为面试项目的理由

1. **完整的系统设计**：从数据导入到答案输出的全流程
2. **主流技术栈**：FastAPI、LangGraph、Qdrant 都是生产级别的工具
3. **可解释性强**：Agent Trace 展示了完整的推理过程，易于讨论
4. **工程素质**：有自动化测试、指标监控、错误处理
5. **扩展性强**：易于添加新工具、新节点、新模型
6. **面试话题丰富**：检索、排序、Agent 设计、文本处理、视频处理等

## 🚀 下一步计划

- [x] 多轮对话支持（SQLite 会话历史管理）
- [ ] 检索结果重排序层（使用小模型）
- [x] Qdrant 持久化存储配置
- [x] Docker 本地完整运行镜像
- [ ] 自动评估框架（NDCG、F1 等指标）
- [ ] 支持多种 LLM（OpenAI、本地模型等）
- [ ] 前端实时流式输出
- [ ] 证据高亮和视频时间轴跳转

## 📝 许可证

MIT

## 👤 作者

项目式学习 - 面试准备项目

---

**最后更新**：2026-08-31  
**Agent 版本**：0.3.0  
**测试状态**：✅ 7/7 通过
