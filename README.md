# 轻量 RAG 知识库问答系统

一个本地运行的文档问答系统：上传 PDF / Markdown / TXT / Word 文档，系统自动**解析 → 切片 → 向量化入库**，提问时**向量检索召回相关片段**，再由大模型（或检索摘要模式）组织成带引用的流畅回答。

- 后端：FastAPI + ChromaDB + sentence-transformers（本地嵌入模型，数据不出本机）
- 前端：原生 HTML/CSS/JS 单页应用，流式打字机输出，引用片段可展开核对
- 数据：SQLite 持久化会话与消息，刷新不丢失

## 目录结构

```
rag-kb/
├── backend/
│   ├── app.py               # FastAPI 入口与全部接口
│   ├── requirements.txt
│   └── rag/
│       ├── config.py        # 配置与本地设置持久化
│       ├── document_loader.py  # 文档解析（pdf/md/txt/docx）
│       ├── chunker.py       # 切片策略（段落 + 滑动窗口）
│       ├── embeddings.py    # 本地嵌入模型（懒加载）
│       ├── vector_store.py  # Chroma 向量库 + SQLite 元数据
│       ├── llm.py           # OpenAI 兼容接口流式调用
│       └── pipeline.py      # RAG 编排（检索 + 生成）
├── frontend/index.html      # 前端单页
├── start.bat                # Windows 一键启动
└── README.md
```

## 快速开始

环境要求：Windows / macOS / Linux，Python 3.10+。

### 方式一：Windows 一键启动

双击 `start.bat`。首次运行会自动创建虚拟环境并安装依赖（含约 100MB 的本地嵌入模型下载，耐心等待），随后自动打开浏览器访问 `http://127.0.0.1:8000`。

### 方式二：手动启动

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 建议先装 CPU 版 torch（体积小）
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r backend/requirements.txt

python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

浏览器打开 `http://127.0.0.1:8000`。

## 使用流程

1. **上传文档**：左侧知识库拖拽或点击上传（支持多文件）。
2. **提问**：主区输入问题，Enter 发送。系统召回 Top-K 相关片段。
3. **查看引用**：答案下方出现引用卡片（来源 · 相似度），点击展开片段原文核对。

> 双击文档卡片可预览该文档的切片（已入库的片段内容）。

## 两种回答模式

| 模式 | 触发条件 | 效果 |
|---|---|---|
| **LLM 生成** | 在「设置」中填入 OpenAI 兼容接口的 Base URL + API Key + 模型名 | 大模型基于召回片段流式生成答案，带打字机效果 |
| **检索摘要** | 未配置 Key | 透明展示召回片段（来源 / 相似度 / 原文），即开即用 |

## 配置大模型

点击右上角「设置」：

- **接口地址**：任何 OpenAI 兼容服务。例：
  - OpenAI：`https://api.openai.com/v1`
  - DeepSeek：`https://api.deepseek.com/v1`
  - Kimi：`https://api.moonshot.cn/v1`
  - 本地 vLLM / Ollama 网关：`http://127.0.0.1:xxxx/v1`
- **API Key**：仅保存在本机 `backend/data/settings.json`，不会外发。
- **模型名称**：按服务商填写，如 `gpt-4o-mini`、`deepseek-chat`、`moonshot-v1-8k`。
- **Top-K**：召回片段数量（2–8）。

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 前端页面 |
| POST | `/api/documents/upload` | 上传文档（multipart），解析切片入库 |
| GET | `/api/documents` | 文档列表与总片段数 |
| DELETE | `/api/documents/{id}` | 删除文档及其切片 |
| GET | `/api/documents/{id}/chunks` | 文档切片预览 |
| GET | `/api/conversations` | 会话列表 |
| POST | `/api/conversations` | 新建会话 |
| DELETE | `/api/conversations/{id}` | 删除会话 |
| GET | `/api/conversations/{id}/messages` | 会话消息 |
| POST | `/api/chat/stream` | 问答（SSE 流式） |
| GET/POST | `/api/settings` | 读取/保存 LLM 设置 |

## 技术要点

- **切片策略**：先按空行分段，再对超长段落做 500 字符滑动窗口（80 字符重叠），合并过短碎片，兼顾语义完整与检索精度。
- **检索**：`BAAI/bge-small-zh-v1.5` 本地嵌入（中文友好），ChromaDB cosine 相似度召回，支持按单文档过滤。
- **流式**：SSE 协议逐 token 推送，前端 `ReadableStream` 解析，打字机渲染。
- **持久化**：SQLite 存会话 / 消息 / 引用；Chroma 存向量；上传原文件保留在 `backend/data/uploads`。

## 常见问题

- **首次上传/提问很慢**：首次会加载本地嵌入模型（约 100MB），之后常驻内存，不再等待。
- **PDF 解析为空**：扫描件 / 图片型 PDF 无文本层，请先 OCR 或改用文本类文档。
- **端口被占用**：修改 `start.bat` / 启动命令中的 `--port 8000`。
- **LLM 报错 401/404**：检查 Base URL 是否以 `/v1` 结尾、模型名与 Key 是否匹配该服务商。
