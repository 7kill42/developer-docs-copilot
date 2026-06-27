# Developer Docs Copilot Recruiter Brief

## 一句话定位

面向开发文档和垂直知识库的 RAG 问答助手，覆盖文档采集、结构化切分、混合检索、引用溯源、低置信拒答和安全索引刷新。

## 为什么适合 Agent / RAG 岗位

- 证明了模型回答不能只依赖参数记忆，需要接入外部知识源并返回可追溯证据。
- 同一套 RAG 架构支持 SQLAlchemy 开发文档和金融/保险资料，体现领域迁移能力。
- 检索层结合向量召回、BM25、章节命中和关键词扩展，适合高术语密度文档。
- 生成层通过 citation、grounded prompt 和低置信拒答控制无依据回答。

## 核心链路

```text
Documents
  -> Crawl / parse
  -> Section-level chunks
  -> Chroma vector index + BM25 index
  -> Hybrid retrieval / rerank
  -> Grounded answer generation
  -> Citation display / refusal
```

## 技术亮点

- **结构化解析：** 使用 httpx、BeautifulSoup、lxml 抽取标题层级、正文、代码示例和来源信息。
- **Section 粒度切分：** 尽量保留文档章节语义，避免固定窗口切断条款或代码解释。
- **混合检索：** 向量检索负责语义召回，BM25 和章节命中增强专有名词、API 名称和条款关键词。
- **低置信拒答：** 检索证据不足时不强行生成，降低幻觉风险。
- **安全索引刷新：** 先写临时 Chroma collection，校验成功后再替换正式索引，避免失败时旧知识库不可用。

## 面试可讲问题

- 为什么选择 section 粒度 chunk，而不是固定 token 窗口？
- BM25 和向量检索在开发文档 / 金融资料中的互补点是什么？
- citation 如何帮助判断回答是否 grounded？
- 低置信拒答阈值如何设计，可能带来什么误伤？
- 索引刷新失败时如何保证服务可用？

## 建议展示方式

1. 展示一次文档索引构建 summary。
2. 提一个 SQLAlchemy 或金融/保险问题。
3. 展示召回片段、最终回答和 citation。
4. 展示一个证据不足问题触发拒答。
