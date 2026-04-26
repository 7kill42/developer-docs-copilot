from __future__ import annotations

import json
from typing import Any

import streamlit as st

from config import PROCESSED_DIR
from eval import load_eval_questions, run_retrieval_eval
from ingest import build_index, estimate_tokens
from rag import answer_question, has_index, load_index_summary


st.set_page_config(page_title="SQLAlchemy Upgrade Copilot", page_icon=":books:", layout="wide")


DOC_TYPE_LABELS = {
    "tutorial": "教程",
    "orm_querying_guide": "ORM 查询指南",
    "migration_guide": "迁移指南",
    "asyncio_guide": "AsyncIO 指南",
}


def _format_doc_type(doc_type: str) -> str:
    return DOC_TYPE_LABELS.get(doc_type, doc_type)


def _preview(text: str, *, limit: int = 180) -> str:
    snippet = text.strip().replace("\n", " ")
    return snippet if len(snippet) <= limit else snippet[: limit - 3] + "..."


@st.cache_data(show_spinner=False)
def _load_chunk_rows(cache_key: int) -> list[dict[str, Any]]:
    del cache_key
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    for chunk in chunks:
        rows.append(
            {
                "page_url": chunk.get("url", "").split("#", 1)[0],
                "title": chunk.get("title", ""),
                "section_path": chunk.get("section_path", ""),
                "section_id": chunk.get("section_id", ""),
                "doc_type": _format_doc_type(chunk.get("doc_type", "")),
                "token_count": estimate_tokens(chunk.get("content", "")),
                "has_code_example": "Yes" if chunk.get("has_code_example") else "No",
                "preview": _preview(chunk.get("content", ""), limit=150),
            }
        )
    return rows


@st.cache_data(show_spinner=False)
def _run_cached_mini_eval(index_signature: str, questions_signature: str) -> dict[str, Any]:
    del index_signature
    questions = json.loads(questions_signature)
    return run_retrieval_eval(questions, include_generation=False)


def _stage_table(items: list[dict[str, Any]], *, score_key: str = "score") -> list[dict[str, Any]]:
    rows = []
    for item in items:
        rows.append(
            {
                "title": item.get("title", ""),
                "doc_type": _format_doc_type(item.get("doc_type", "")),
                "score": round(float(item.get(score_key, 0.0)), 3),
                "vector": round(float(item.get("vector_score", 0.0)), 3),
                "bm25": round(float(item.get("bm25_score", 0.0)), 3),
                "section_path": item.get("section_path", ""),
                "url": item.get("url", ""),
            }
        )
    return rows


def _context_table(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "title": item.get("title", ""),
            "doc_type": _format_doc_type(item.get("doc_type", "")),
            "chars": len(item.get("content", "")),
            "excerpt": _preview(item.get("content", ""), limit=220),
            "url": item.get("url", ""),
        }
        for item in items
    ]


def _render_answer_block(result: dict[str, Any]) -> None:
    st.subheader("回答")
    st.write(result["answer"])

    if result["example_code"]:
        st.subheader("示例代码")
        st.code(result["example_code"], language="python")

    st.subheader("官方依据")
    st.caption("下面这些是当前回答主要参考的 SQLAlchemy 官方文档章节，你可以点开原文继续核对。")
    for idx, citation in enumerate(result["citations"], start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}. {citation['title']}**")
            st.caption(_format_doc_type(citation["doc_type"]))
            st.caption(f"章节路径：{citation['section_path']}")
            score = citation.get("score")
            if score is not None:
                st.caption(f"匹配度：{score}")
            st.markdown("**相关原文摘录**")
            st.write(citation["snippet"])
            st.link_button("查看官方原文", citation["url"], key=f"citation-{idx}-{citation['url']}")


st.title("SQLAlchemy Upgrade Copilot")
st.caption("基于 SQLAlchemy 官方文档的 RAG 问答助手，回答附带引用来源与原始链接。")

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

with st.sidebar:
    st.subheader("Index")
    summary = load_index_summary()
    if summary:
        st.json(summary)
    else:
        st.info("还没有索引，先点击下方按钮。")

    if st.button("Build / Refresh Index", use_container_width=True):
        with st.spinner("正在抓取 SQLAlchemy 文档并重建向量索引..."):
            try:
                result = build_index()
            except Exception as exc:
                message = str(exc)
                if "502" in message or "模型服务暂时不可用" in message:
                    st.error(f"构建索引失败：百炼接口暂时异常（可能是瞬时 502），可以稍后重试。原始错误：{message}")
                else:
                    st.error(f"构建索引失败：{message}")
            else:
                st.success("索引构建完成。")
                st.json(result)

ask_tab, highlights_tab = st.tabs(["Ask", "Technical Highlights"])

with ask_tab:
    st.markdown(
        """
请直接输入 SQLAlchemy 问题，例如：
- SQLAlchemy 2.0 推荐怎么写 select 查询？
- Session 的推荐使用方式是什么？
- SQLAlchemy 2.0 和旧版查询风格有什么区别？
"""
    )

    question = st.text_area(
        "输入你的问题",
        key="question_input",
        height=120,
        placeholder="例如：SQLAlchemy 2.0 推荐怎么写 select 查询？",
    )

    submit = st.button("Ask", type="primary", use_container_width=True)

    if submit:
        if not question.strip():
            st.warning("先输入一个问题。")
        elif not has_index():
            st.warning("还没有可用索引。先点击左侧的 Build / Refresh Index。")
        else:
            with st.spinner("正在检索 SQLAlchemy 官方文档并生成回答，复杂问题通常需要 5-15 秒..."):
                try:
                    st.session_state["last_result"] = answer_question(question.strip(), include_debug=True)
                except Exception as exc:
                    message = str(exc)
                    if "502" in message or "模型服务暂时不可用" in message:
                        st.error(f"回答失败：百炼接口暂时异常（可能是瞬时 502），请再试一次。原始错误：{message}")
                    else:
                        st.error(f"回答失败：{message}")

    if st.session_state["last_result"] is not None:
        _render_answer_block(st.session_state["last_result"])

with highlights_tab:
    st.subheader("技术亮点总览")
    st.markdown(
        """
- 混合检索：同一套问题同时展示 `纯向量`、`BM25 + 向量`、`rerank` 三个阶段的差异。
- 结构化 chunk：按 section 切分文档，并把 token 数、代码示例标记和 section anchor 一起暴露出来。
- 可解释 RAG：把实时检索过程拆成可视化步骤，而不是只给最终回答。
"""
    )

    st.divider()
    st.subheader("1. 检索策略对比")
    if not has_index():
        st.info("先构建索引，才能运行 mini eval。")
    else:
        questions = load_eval_questions()
        questions_signature = json.dumps(questions, ensure_ascii=False, sort_keys=True)
        index_signature = json.dumps(load_index_summary(), ensure_ascii=False, sort_keys=True)
        if st.button("Run 5-question mini eval", key="run-mini-eval"):
            with st.spinner("正在跑 mini eval，对比不同检索策略..."):
                st.session_state["mini_eval_summary"] = _run_cached_mini_eval(index_signature, questions_signature)

        mini_eval_summary = st.session_state.get("mini_eval_summary")
        if mini_eval_summary:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.metric("评测问题数", mini_eval_summary["questions_evaluated"])
            with col2:
                st.metric("平均检索耗时", f"{mini_eval_summary['avg_retrieval_ms']} ms")

            st.caption("这个表用 5 个固定测试问题对比三种策略的 Recall 指标。")
            st.dataframe(mini_eval_summary["strategy_rows"], use_container_width=True, hide_index=True)
            with st.expander("查看每个问题的细项结果"):
                st.dataframe(mini_eval_summary["question_rows"], use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("2. Chunk 策略可视化")
    chunks_path = PROCESSED_DIR / "chunks.json"
    chunk_cache_key = chunks_path.stat().st_mtime_ns if chunks_path.exists() else 0
    chunk_rows = _load_chunk_rows(chunk_cache_key)
    if not chunk_rows:
        st.info("先构建索引，才能查看 chunk 拆分情况。")
    else:
        page_urls = sorted({row["page_url"] for row in chunk_rows})
        selected_page = st.selectbox("选择一个文档页面", page_urls)
        page_rows = [row for row in chunk_rows if row["page_url"] == selected_page]
        page_rows.sort(key=lambda row: (row["section_path"], row["title"]))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunks", len(page_rows))
        with col2:
            st.metric("Avg Tokens", round(sum(row["token_count"] for row in page_rows) / len(page_rows), 1))
        with col3:
            st.metric("Code Chunks", sum(row["has_code_example"] == "Yes" for row in page_rows))

        st.caption("下面能直接看到某个页面被切成了哪些 chunk、每个 chunk 多大、有没有代码示例。")
        st.dataframe(page_rows, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("3. 实时检索过程")
    last_result = st.session_state.get("last_result")
    if not last_result or "debug" not in last_result:
        st.info("先在 Ask 页问一个问题，这里就会显示完整的检索轨迹。")
    else:
        debug = last_result["debug"]
        st.caption(f"当前问题：{debug['question']}")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("候选池大小", debug["candidate_pool_size"])
        with col2:
            st.metric("检索耗时", f"{debug['retrieval_ms']} ms")

        st.markdown("**向量检索 Top-6**")
        st.dataframe(_stage_table(debug["vector_hits"]), use_container_width=True, hide_index=True)

        st.markdown("**BM25 Top-6**")
        st.dataframe(_stage_table(debug["bm25_hits"], score_key="bm25_score"), use_container_width=True, hide_index=True)

        st.markdown("**融合后 Top-3**")
        st.dataframe(_stage_table(debug["hybrid_hits"][:3]), use_container_width=True, hide_index=True)

        st.markdown("**rerank 后 Top-3**")
        st.dataframe(_stage_table(debug["reranked_hits"][:3]), use_container_width=True, hide_index=True)

        st.markdown("**最终送给模型的上下文**")
        st.dataframe(_context_table(debug["model_context"]), use_container_width=True, hide_index=True)
