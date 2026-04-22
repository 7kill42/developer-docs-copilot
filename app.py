from __future__ import annotations

import streamlit as st

from ingest import build_index
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


st.title("SQLAlchemy Upgrade Copilot")
st.caption("基于 SQLAlchemy 官方文档的 RAG 问答助手，回答附带引用来源与原始链接。")

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
                result = answer_question(question.strip())
            except Exception as exc:
                message = str(exc)
                if "502" in message or "模型服务暂时不可用" in message:
                    st.error(f"回答失败：百炼接口暂时异常（可能是瞬时 502），请再试一次。原始错误：{message}")
                else:
                    st.error(f"回答失败：{message}")
            else:
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
                        doc_type = _format_doc_type(citation["doc_type"])
                        st.caption(f"{doc_type}")
                        st.caption(f"章节路径：{citation['section_path']}")
                        score = citation.get("score")
                        if score is not None:
                            st.caption(f"匹配度：{score}")
                        st.markdown("**相关原文摘录**")
                        st.write(citation["snippet"])
                        st.link_button("查看官方原文", citation["url"])
