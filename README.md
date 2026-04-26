# Developer Docs Copilot

一个面向开发者文档问答场景的 RAG 项目。  
当前版本聚焦 SQLAlchemy 官方文档，能够基于检索结果回答问题，并给出引用内容与原始链接，强调答案可追溯而不是“纯生成”。

## Recruiter Snapshot

- **项目定位：** 面向开发者效率的 AI 应用 / RAG Demo
- **核心能力：** 文档抓取、结构化切分、向量检索、基于上下文生成回答、引用回溯
- **展示价值：** 体现了我把 LLM 能力落到真实工具场景中的能力，而不只是调用一个聊天接口

## 功能

- 抓取 SQLAlchemy 官方文档的少量核心页面
- 解析标题、正文和代码示例
- 用 `text-embedding-v4` 建立 Chroma 向量索引
- 用 `qwen3.6-flash` 基于检索结果生成回答
- 在 Streamlit 页面里展示答案、示例代码和 citations

当前实现使用 `DashScope / Qwen` 的 OpenAI 兼容接口。

## 项目结构

```text
developer-docs-copilot/
├── app.py
├── config.py
├── DESIGN.md
├── eval.py
├── ingest.py
├── prompts.py
├── rag.py
├── requirements.txt
├── .env.example
├── README.md
└── data/
    ├── raw/
    ├── processed/
    └── chroma/
```

## 快速开始

1. 创建虚拟环境并安装依赖

```bash
cd /root/project/developer-docs-copilot
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. 配置环境变量

```bash
cp .env.example .env
```

至少填写：

```bash
OPENAI_API_KEY=...
```

默认还会读取：

```bash
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_CHAT_MODEL=qwen3.6-flash
OPENAI_EMBEDDING_MODEL=text-embedding-v4
MAX_SEED_PAGES=0
```

其中 `MAX_SEED_PAGES=0` 表示默认索引全部内置种子页；如果想缩小抓取范围，可以改成具体数字。

3. 启动应用

```bash
streamlit run app.py
```

4. 在左侧点击 `Build / Refresh Index`

应用会抓取并索引这些页面：

- SQLAlchemy Tutorial
- `engine`
- `data_select`
- `session_basics`
- ORM Querying Guide
- `query`
- `select` querying guide
- `relationships`
- `columns`
- `api`
- `dml`
- `inheritance`
- `asyncio`
- `migration_20`

## 技术亮点页

Streamlit 页面里除了问答页，还增加了一个 `Technical Highlights` 视图，用来展示：

- `纯向量` vs `BM25 + 向量` vs `rerank` 的 mini eval 对比
- 某个官方文档页面被切成了哪些 chunk
- 一次提问的实时检索轨迹：向量召回、BM25、融合结果、最终送给模型的上下文

这部分主要是为了让面试官能在几分钟内快速看懂项目深度，而不是只看到一个聊天框。

## 评测

项目内置了一个 5 题的小型评测集，位于 [data/eval_questions.json](data/eval_questions.json)。

运行方式：

```bash
python3 eval.py
```

如果只想看检索指标，不跑生成：

```bash
python3 eval.py --skip-generation
```

当前评测会输出：

- `Recall@3 / Recall@6`
- 回答包含预期关键词的比例
- 平均检索耗时

## 设计决策

关键设计说明见 [DESIGN.md](DESIGN.md)：

- 为什么选择 `BM25 + 向量` 混合检索
- 为什么 chunk 采用 section 粒度而不是固定 token 窗口
- 为什么 citation 要加 section anchor

## 推荐 demo 问题

- `SQLAlchemy 2.0 推荐怎么写 select 查询？`
- `Session 的推荐使用方式是什么？`
- `SQLAlchemy 2.0 和旧版查询风格有什么区别？`

## 说明

- 文档范围只覆盖少量核心页面，不是全站问答。
- 检索阶段使用向量召回和 BM25 词项相关性融合，再结合章节规则做轻量重排。
- 如果没有足够证据，应用会明确说未找到答案，而不是自由发挥。
- `qwen3-vl-rerank` 这类 rerank 模型不负责生成向量，不能直接替代 embedding 模型。

## 已知限制

- 当前只索引了 14 个 SQLAlchemy 核心页面，不是全站问答系统。
- 对于“如何调试 SQLAlchemy 性能问题”这类更偏操作经验、排障经验的问题，效果会受到文档覆盖范围限制。
- 当前没有做多轮对话记忆，每次提问都按单轮检索与回答处理。
- mini eval 规模还比较小，更适合作为 demo 和优化对比，而不是严格学术评测。
