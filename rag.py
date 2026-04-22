from __future__ import annotations

import json
import re
import time
from functools import lru_cache
from typing import Any

import chromadb
from openai import APIConnectionError, APIStatusError, OpenAI

from config import PROCESSED_DIR, settings
from prompts import SYSTEM_PROMPT, build_user_prompt


def _get_openai_client() -> OpenAI:
    if not settings.has_openai_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法调用 OpenAI。")
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
        max_retries=1,
    )


def _get_collection():
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))
    return chroma_client.get_collection(name=settings.chroma_collection_name)


def has_index() -> bool:
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))
    try:
        chroma_client.get_collection(name=settings.chroma_collection_name)
        return True
    except Exception:
        return False


QUERY_HINTS = {
    "查询": ["query", "select", "statement", "execute", "scalars"],
    "怎么写": ["how", "example", "select", "query"],
    "如何": ["how", "usage", "example"],
    "推荐": ["recommended", "best practice", "2.0 style"],
    "会话": ["session", "transaction"],
    "session": ["session", "transaction"],
    "区别": ["difference", "migration", "legacy", "2.0", "1.4"],
    "旧版": ["legacy", "1.4", "query", "migration"],
    "新版": ["2.0", "modern", "select"],
    "迁移": ["migration", "2.0", "legacy", "query"],
    "orm": ["orm", "mapped", "entity"],
    "执行": ["execute", "scalars", "session"],
    "示例": ["example", "sample", "code"],
    "关系": ["relationship", "orm", "joinedload"],
    "延迟加载": ["lazy loading", "lazyload", "relationship loading"],
    "joinedload": ["joinedload", "joined eager loading", "relationship loading"],
    "预加载": ["eager loading", "joined eager loading", "selectinload"],
    "异步": ["asyncio", "asyncsession", "async engine"],
    "asyncio": ["asyncio", "asyncsession", "async engine"],
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [token for token in normalized.split() if len(token) > 1]


def _expand_question(question: str) -> str:
    expanded_parts = [question]
    lowered = question.lower()
    for key, hints in QUERY_HINTS.items():
        if key in lowered or key in question:
            expanded_parts.extend(hints)
    return " ".join(dict.fromkeys(" ".join(expanded_parts).split()))


@lru_cache(maxsize=1)
def _load_local_chunks() -> list[dict[str, Any]]:
    chunks_path = PROCESSED_DIR / "chunks.json"
    if not chunks_path.exists():
        return []
    try:
        return json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _keyword_score(question: str, chunk: dict[str, Any]) -> float:
    terms = _extract_terms(_expand_question(question))
    if not terms:
        return 0.0

    title = _normalize_text(chunk.get("title", ""))
    section = _normalize_text(chunk.get("section_path", ""))
    content = _normalize_text(chunk.get("content", ""))

    score = 0.0
    for term in terms:
        if term in title:
            score += 0.22
        if term in section:
            score += 0.18
        if term in content:
            score += 0.07
    return min(score, 1.0)


def _doc_type_boost(question: str, doc_type: str) -> float:
    lowered = question.lower()
    boost = 0.0
    if any(token in lowered for token in ["区别", "迁移", "1.4", "2.0", "legacy"]):
        if doc_type == "migration_guide":
            boost += 0.12
    if any(token in lowered for token in ["session", "会话", "事务"]):
        if doc_type == "tutorial":
            boost += 0.06
    if any(token in lowered for token in ["query", "select", "查询", "怎么写", "如何"]):
        if doc_type in {"tutorial", "orm_querying_guide"}:
            boost += 0.08
    return boost


def _is_relationship_loading_question(question: str) -> bool:
    lowered = question.lower()
    signals = [
        "joinedload",
        "relationship",
        "lazy",
        "eager",
        "延迟加载",
        "预加载",
        "加载策略",
        "关联加载",
        "relationship loading",
    ]
    return any(signal in lowered or signal in question for signal in signals)


def _is_legacy_query_question(question: str) -> bool:
    lowered = question.lower()
    signals = ["legacy", "旧版", "老", "query 风格", "query风格", "1.4", "2.0", "迁移", "区别"]
    return any(signal in lowered or signal in question for signal in signals)


def _section_match_boost(question: str, title: str, section_path: str) -> float:
    lowered_question = question.lower()
    lowered_title = title.lower()
    lowered_section = section_path.lower()
    blob = f"{lowered_title} {lowered_section}"
    boost = 0.0

    if _is_relationship_loading_question(question):
        if any(term in blob for term in [
            "relationship loading",
            "joined eager loading",
            "joinedload",
            "loader api",
            "loading styles",
            "lazy loading",
        ]):
            boost += 0.28
        if "query object" in blob or "legacy query" in blob:
            boost -= 0.22

    if "joinedload" in lowered_question and any(term in blob for term in ["joined eager loading", "joinedload"]):
        boost += 0.18
    if any(term in question for term in ["延迟加载"]) or "lazy" in lowered_question:
        if "lazy" in blob or "relationship loading" in blob:
            boost += 0.16

    if any(term in lowered_question for term in ["asyncio", "async", "asyncsession"]) or "异步" in question:
        if any(term in blob for term in ["asynchronous i/o", "asyncio", "asyncsession"]):
            boost += 0.2

    if _is_legacy_query_question(question):
        if "legacy query" in blob or "query object" in blob:
            boost += 0.18
    else:
        if "legacy query" in blob or "query object" in blob:
            boost -= 0.12

    return boost


def _call_with_retries(func, *, retries: int = 3, base_sleep: float = 1.0):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return func()
        except APIConnectionError as exc:
            last_error = exc
        except APIStatusError as exc:
            last_error = exc
            if exc.status_code < 500 and exc.status_code != 429:
                raise
        except Exception as exc:
            last_error = exc
            message = str(exc).lower()
            if "status 502" not in message and "connection error" not in message:
                raise

        if attempt < retries:
            time.sleep(base_sleep * attempt)

    raise RuntimeError(
        f"模型服务暂时不可用，已重试 {retries} 次仍失败：{last_error}"
    ) from last_error


def search_docs(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    top_k = top_k or settings.top_k
    client = _get_openai_client()
    collection = _get_collection()
    expanded_question = _expand_question(question)

    embedding_response = _call_with_retries(
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=expanded_question,
        )
    )
    query_embedding = embedding_response.data[0].embedding

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=settings.retrieve_k,
        include=["documents", "metadatas", "distances"],
    )

    ranked_by_id: dict[str, dict[str, Any]] = {}
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    wants_examples = any(token in question.lower() for token in ["example", "示例", "怎么写", "如何"])
    for document, metadata, distance in zip(documents, metadatas, distances):
        content_hash = f"{metadata.get('url', '')}::{metadata.get('section_path', '')}::{document[:120]}"
        score = 1 / (1 + float(distance or 0))
        if metadata.get("has_code_example") == "true" and wants_examples:
            score += 0.15
        score += _doc_type_boost(question, metadata.get("doc_type", ""))
        score += _section_match_boost(
            question,
            metadata.get("title", ""),
            metadata.get("section_path", ""),
        )
        candidate = {
            "id": content_hash,
            "vector_score": score,
            "keyword_score": 0.0,
            "score": score,
            "content": document,
            "title": metadata.get("title", ""),
            "section_path": metadata.get("section_path", ""),
            "doc_type": metadata.get("doc_type", ""),
            "url": metadata.get("url", ""),
            "has_code_example": metadata.get("has_code_example") == "true",
        }
        ranked_by_id[candidate["id"]] = candidate

    for chunk in _load_local_chunks():
        keyword = _keyword_score(question, chunk)
        if keyword <= 0:
            continue
        chunk_id = f"{chunk.get('url', '')}::{chunk.get('section_path', '')}::{chunk.get('content', '')[:120]}"
        candidate = ranked_by_id.get(
            chunk_id,
            {
                "id": chunk_id,
                "vector_score": 0.0,
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "section_path": chunk.get("section_path", ""),
                "doc_type": chunk.get("doc_type", ""),
                "url": chunk.get("url", ""),
                "has_code_example": bool(chunk.get("has_code_example")),
            },
        )
        candidate["keyword_score"] = max(candidate.get("keyword_score", 0.0), keyword)
        candidate["score"] = (
            candidate.get("vector_score", 0.0) * 0.65
            + candidate["keyword_score"] * 0.35
            + _doc_type_boost(question, candidate.get("doc_type", ""))
            + _section_match_boost(question, candidate.get("title", ""), candidate.get("section_path", ""))
        )
        if candidate.get("has_code_example") and wants_examples:
            candidate["score"] += 0.1
        ranked_by_id[chunk_id] = candidate

    ranked = list(ranked_by_id.values())
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:top_k]


def _parse_json_response(raw_text: str) -> dict[str, str]:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw_text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"answer": raw_text.strip(), "example_code": ""}


def _trim_context_for_model(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for item in citations[: settings.model_context_k]:
        content = item["content"][: settings.model_context_char_limit].strip()
        if len(item["content"]) > len(content):
            content += "\n\n[truncated]"
        trimmed.append({**item, "content": content})
    return trimmed


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in citations:
        key = (
            item.get("title", ""),
            item.get("section_path", ""),
            item.get("url", ""),
        )
        existing = deduped.get(key)
        if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
            deduped[key] = item

    unique = list(deduped.values())
    unique.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return unique


def answer_question(question: str) -> dict[str, Any]:
    citations = search_docs(question)
    citations = _dedupe_citations(citations)
    if not citations:
        return {
            "answer": "未在已索引文档中找到明确答案。",
            "example_code": "",
            "citations": [],
        }

    top_score = citations[0].get("score", 0.0)
    if top_score < settings.min_relevance_score:
        return {
            "answer": "当前检索到的文档与问题相关性不够高，暂时不能基于官方文档给出可靠答案。你可以换一种更具体的问法，比如带上 `Session`、`select()`、`migration 2.0` 这类关键词。",
            "example_code": "",
            "citations": [
                {
                    "title": item["title"],
                    "section_path": item["section_path"],
                    "doc_type": item["doc_type"],
                    "url": item["url"],
                    "snippet": item["content"][:260].strip(),
                    "score": round(item.get("score", 0.0), 3),
                }
                for item in citations[:3]
            ],
        }

    client = _get_openai_client()
    model_context = _trim_context_for_model(citations)
    response = _call_with_retries(
        lambda: client.responses.create(
            model=settings.openai_chat_model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(question, model_context)},
            ],
        )
    )
    payload = _parse_json_response(response.output_text)

    trimmed_citations = []
    for item in citations[: max(2, min(len(citations), 4))]:
        snippet = item["content"][:420].strip()
        trimmed_citations.append(
            {
                "title": item["title"],
                "section_path": item["section_path"],
                "doc_type": item["doc_type"],
                "url": item["url"],
                "score": round(item.get("score", 0.0), 3),
                "snippet": snippet + ("..." if len(item["content"]) > len(snippet) else ""),
            }
        )

    return {
        "answer": payload.get("answer", "未在已索引文档中找到明确答案。").strip(),
        "example_code": payload.get("example_code", "").strip(),
        "citations": trimmed_citations,
    }


def load_index_summary() -> dict[str, Any]:
    summary_path = PROCESSED_DIR / "index_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
