# SQLAlchemy Upgrade Copilot

基于 SQLAlchemy 官方文档回答问题，并附上引用来源和原始链接。

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
```

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

## 推荐 demo 问题

- `SQLAlchemy 2.0 推荐怎么写 select 查询？`
- `Session 的推荐使用方式是什么？`
- `SQLAlchemy 2.0 和旧版查询风格有什么区别？`

## 说明

- 文档范围只覆盖少量核心页面，不是全站问答。
- 如果没有足够证据，应用会明确说未找到答案，而不是自由发挥。
- `qwen3-vl-rerank` 这类 rerank 模型不负责生成向量，不能直接替代 embedding 模型。
