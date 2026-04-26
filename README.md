# Developer Docs Copilot: SQLAlchemy RAG Assistant

一个面向开发者文档问答场景的 RAG 项目。  
当前版本聚焦 SQLAlchemy 官方文档，能够基于检索结果回答问题，并给出引用内容与原始链接，强调答案可追溯而不是“纯生成”。

## Recruiter Snapshot

- **项目定位：** 面向开发者效率的 AI 应用 / RAG Demo
- **核心能力：** 文档抓取、结构化切分、向量检索、基于上下文生成回答、引用回溯
- **展示价值：** 体现了我把 LLM 能力落到真实工具场景中的能力，而不只是调用一个聊天接口

## Highlights

- **文档 ETL Pipeline：** 自动抓取 SQLAlchemy 官方文档，解析标题层级、正文与代码块，并生成结构化 chunk。
- **Hybrid Retrieval：** 结合向量检索、BM25、关键词匹配、文档类型加权和章节匹配，提高中文问题到英文技术文档的召回效果。
- **Grounded Generation：** 答案仅基于检索上下文生成，并返回官方文档 citation，降低幻觉风险。
- **Groundedness Verifier：** 回答生成后会再做一次规则型校验，检查关键结论和代码符号是否能在 citation 片段中找到依据。
- **Low-confidence Refusal：** 当检索相关性低于阈值时拒答，避免无依据生成。
- **Production-oriented Design：** 支持索引刷新、接口重试、环境变量配置和本地持久化向量库。

## Architecture

```text
User Question
    ↓
Query Expansion
    ↓
Vector Search + BM25 Search
    ↓
Score Fusion / Rerank
    ↓
Context Builder
    ↓
LLM Answer Generator
    ↓
Groundedness Verifier
    ↓
Citation Display + Refusal Check
```

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
├── eval/
│   ├── eval_answer.py
│   ├── eval_retrieval.py
│   ├── golden_qa.jsonl
│   └── report.md
├── evals.py
├── ingest.py
├── prompts.py
├── rag.py
├── verifier.py
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
MAX_SEED_PAGES=14
```

默认会索引 14 个内置 SQLAlchemy 核心页面；如果想缩小抓取范围，可以把 `MAX_SEED_PAGES` 改成更小的数字。

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

项目内置了两层评测数据：

- [data/eval_questions.json](data/eval_questions.json)：给页面里的 mini eval 用，适合快速演示。
- [eval/golden_qa.jsonl](eval/golden_qa.jsonl)：更正式的 golden set，包含支持回答问题和超出覆盖范围的问题。

运行方式：

```bash
python3 eval/eval_retrieval.py
```

如果只想看检索指标，不跑生成：

```bash
python3 eval/eval_retrieval.py --skip-generation
```

如果想看回答质量和拒答表现：

```bash
python3 eval/eval_answer.py
```

当前评测会输出：

- `Recall@3 / Recall@6`
- 回答包含预期关键词的比例
- grounded answer ratio
- grounding coverage
- refusal accuracy
- 平均检索耗时

这部分可以直接支撑面试里的几个关键问题：

- 怎么判断 hybrid retrieval 是否比纯向量更好
- 怎么看 citation 是否来自更相关的章节
- 怎么在低证据场景下拒答而不是自由发挥
- 怎么在生成后再检查回答是否真的被引用片段支撑

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
- 回答生成后还会做一次 groundedness verification，证据不足时会提示用户核对引用，严重不足时会直接拒答。
- `qwen3-vl-rerank` 这类 rerank 模型不负责生成向量，不能直接替代 embedding 模型。

## 已知限制

- 当前只索引了 14 个 SQLAlchemy 核心页面，不是全站问答系统。
- 对于“如何调试 SQLAlchemy 性能问题”这类更偏操作经验、排障经验的问题，效果会受到文档覆盖范围限制。
- 当前没有做多轮对话记忆，每次提问都按单轮检索与回答处理。
- mini eval 规模还比较小，更适合作为 demo 和优化对比，而不是严格学术评测。
