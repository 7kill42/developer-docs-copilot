from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from pathlib import Path
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

BM25_K1 = 1.5
BM25_B = 0.75
BM25_SCORE_SCALE = 5.0
RETRIEVAL_STRATEGY_LABELS = {
    "vector": "纯向量检索",
    "hybrid": "BM25 + 向量融合",
    "rerank": "BM25 + 向量 + rerank",
}
_LOCAL_INDEX_CACHE: dict[str, Any] = {
    "mtime_ns": None,
    "chunks": [],
    "documents": [],
    "doc_freq": {},
    "avg_doc_len": 1.0,
    "corpus_size": 0,
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


def _load_json_chunks(chunks_path: Path) -> list[dict[str, Any]]:
    if not chunks_path.exists():
        return []
    try:
        return json.loads(chunks_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def _chunk_id(chunk: dict[str, Any]) -> str:
    return f"{chunk.get('url', '')}::{chunk.get('section_path', '')}::{chunk.get('content', '')[:120]}"


def _chunk_search_text(chunk: dict[str, Any]) -> str:
    title = chunk.get("title", "")
    section = chunk.get("section_path", "")
    content = chunk.get("content", "")
    # Repeat title / section once to let BM25 see heading terms more clearly.
    return " ".join([title, title, section, content])


def _load_local_index() -> dict[str, Any]:
    chunks_path = PROCESSED_DIR / "chunks.json"
    mtime_ns = chunks_path.stat().st_mtime_ns if chunks_path.exists() else None
    if _LOCAL_INDEX_CACHE["mtime_ns"] == mtime_ns:
        return _LOCAL_INDEX_CACHE

    chunks = _load_json_chunks(chunks_path)
    documents: list[dict[str, Any]] = []
    doc_freq: Counter[str] = Counter()
    total_doc_len = 0

    for chunk in chunks:
        terms = _extract_terms(_chunk_search_text(chunk))
        term_counts = Counter(terms)
        doc_len = sum(term_counts.values()) or 1
        total_doc_len += doc_len
        doc_freq.update(term_counts.keys())
        documents.append(
            {
                "id": _chunk_id(chunk),
                "chunk": chunk,
                "term_counts": term_counts,
                "doc_len": doc_len,
            }
        )

    corpus_size = len(documents)
    avg_doc_len = total_doc_len / corpus_size if corpus_size else 1.0
    _LOCAL_INDEX_CACHE.update(
        {
            "mtime_ns": mtime_ns,
            "chunks": chunks,
            "documents": documents,
            "doc_freq": dict(doc_freq),
            "avg_doc_len": avg_doc_len,
            "corpus_size": corpus_size,
        }
    )
    return _LOCAL_INDEX_CACHE


def _load_local_chunks() -> list[dict[str, Any]]:
    return _load_local_index()["chunks"]


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


def _bm25_score(query_terms: list[str], document: dict[str, Any], index: dict[str, Any]) -> float:
    if not query_terms or index["corpus_size"] <= 0:
        return 0.0

    score = 0.0
    doc_len = document["doc_len"]
    avg_doc_len = index["avg_doc_len"] or 1.0
    term_counts: Counter[str] = document["term_counts"]
    doc_freq: dict[str, int] = index["doc_freq"]
    corpus_size = index["corpus_size"]

    for term in query_terms:
        tf = term_counts.get(term, 0)
        if tf <= 0:
            continue
        df = doc_freq.get(term, 0)
        idf = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
        denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * doc_len / avg_doc_len)
        score += idf * (tf * (BM25_K1 + 1.0) / denom)
    return score


def _search_bm25(question: str, *, limit: int) -> list[dict[str, Any]]:
    index = _load_local_index()
    query_terms = _question_terms(question)
    if not query_terms:
        return []

    hits: list[dict[str, Any]] = []
    for document in index["documents"]:
        bm25_score = _bm25_score(query_terms, document, index)
        if bm25_score <= 0:
            continue
        chunk = document["chunk"]
        hits.append(
            {
                "id": document["id"],
                "bm25_score": bm25_score,
                "keyword_score": _keyword_score(question, chunk),
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "section_path": chunk.get("section_path", ""),
                "section_id": chunk.get("section_id", ""),
                "doc_type": chunk.get("doc_type", ""),
                "url": chunk.get("url", ""),
                "has_code_example": bool(chunk.get("has_code_example")),
            }
        )

    hits.sort(key=lambda item: (item["bm25_score"], item["keyword_score"]), reverse=True)
    return hits[:limit]


def _wants_examples(question: str) -> bool:
    return any(token in question.lower() for token in ["example", "示例", "怎么写", "如何"])


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


def _question_terms(question: str) -> list[str]:
    terms = _extract_terms(_expand_question(question))
    return list(dict.fromkeys(terms))


def _split_content_units(content: str) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n{2,}", content):
        block = block.strip()
        if not block:
            continue
        if block.startswith("Code example:"):
            units.append(block)
            continue

        sentences = [
            _cleaned
            for _cleaned in (
                _candidate.strip()
                for _candidate in re.split(r"(?<=[。！？!?；;:：.])\s+|\n+", block)
            )
            if _cleaned
        ]
        if len(sentences) <= 1:
            units.append(block)
        else:
            units.extend(sentences)
    return units


def _unit_relevance_score(unit: str, question_terms: list[str], wants_examples: bool) -> float:
    normalized_unit = _normalize_text(unit)
    score = 0.0
    for term in question_terms:
        if term in normalized_unit:
            score += 1.6 if len(term) >= 6 else 1.0
    if wants_examples and unit.startswith("Code example:"):
        score += 1.2
    return score


def _extract_relevant_window(question: str, content: str, char_limit: int) -> str:
    units = _split_content_units(content)
    if not units:
        return content[:char_limit].strip()

    question_terms = _question_terms(question)
    wants_examples = any(token in question.lower() for token in ["example", "示例", "怎么写", "如何"])
    ranked_indices = sorted(
        range(len(units)),
        key=lambda idx: (_unit_relevance_score(units[idx], question_terms, wants_examples), -idx),
        reverse=True,
    )

    selected_indices: set[int] = set()
    current_length = 0

    def try_add(index: int) -> bool:
        nonlocal current_length
        if index < 0 or index >= len(units) or index in selected_indices:
            return False
        addition = len(units[index]) + (2 if selected_indices else 0)
        if selected_indices and current_length + addition > char_limit:
            return False
        if not selected_indices and len(units[index]) > char_limit:
            trimmed = units[index][:char_limit].strip()
            selected_indices.add(index)
            current_length = len(trimmed)
            units[index] = trimmed
            return True
        if current_length + addition > char_limit:
            return False
        selected_indices.add(index)
        current_length += addition
        return True

    for index in ranked_indices:
        if _unit_relevance_score(units[index], question_terms, wants_examples) <= 0 and selected_indices:
            break
        for candidate in (index - 1, index, index + 1):
            try_add(candidate)
        if current_length >= char_limit:
            break

    if not selected_indices:
        for index in range(len(units)):
            if not try_add(index):
                break

    ordered_units = [units[index] for index in sorted(selected_indices)]
    packed = "\n\n".join(ordered_units).strip()
    if len(packed) < len(content):
        packed += "\n\n[truncated]"
    return packed


def _candidate_bonus(question: str, candidate: dict[str, Any], wants_examples: bool) -> float:
    bonus = _doc_type_boost(question, candidate.get("doc_type", ""))
    bonus += _section_match_boost(
        question,
        candidate.get("title", ""),
        candidate.get("section_path", ""),
    )
    if wants_examples and candidate.get("has_code_example"):
        bonus += 0.12
    return bonus


def _strategy_score(question: str, candidate: dict[str, Any], strategy: str) -> float:
    wants_examples = _wants_examples(question)
    vector_strength = candidate.get("vector_score", 0.0)
    bm25_raw = candidate.get("bm25_score", 0.0)
    bm25_strength = bm25_raw / (bm25_raw + BM25_SCORE_SCALE) if bm25_raw > 0 else 0.0
    keyword_strength = candidate.get("keyword_score", 0.0)

    if strategy == "vector":
        return vector_strength
    if strategy == "hybrid":
        return vector_strength * 0.6 + bm25_strength * 0.28 + keyword_strength * 0.12
    if strategy == "rerank":
        return (
            vector_strength * 0.52
            + bm25_strength * 0.28
            + keyword_strength * 0.12
            + _candidate_bonus(question, candidate, wants_examples)
        )
    raise ValueError(f"未知检索策略：{strategy}")


def _copy_candidate(candidate: dict[str, Any], *, score: float) -> dict[str, Any]:
    copied = dict(candidate)
    copied["score"] = score
    return copied


def _rank_candidates(
    question: str,
    candidates_by_id: dict[str, dict[str, Any]],
    *,
    strategy: str,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = [
        _copy_candidate(candidate, score=_strategy_score(question, candidate, strategy))
        for candidate in candidates_by_id.values()
    ]
    ranked.sort(
        key=lambda item: (
            item["score"],
            item.get("bm25_score", 0.0),
            item.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def _dedupe_ranked_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        key = (
            item.get("title", ""),
            item.get("section_path", ""),
            item.get("url", ""),
        )
        existing = deduped.get(key)
        if existing is None or item.get("score", 0.0) > existing.get("score", 0.0):
            deduped[key] = item

    unique = list(deduped.values())
    unique.sort(
        key=lambda item: (
            item.get("score", 0.0),
            item.get("bm25_score", 0.0),
            item.get("vector_score", 0.0),
        ),
        reverse=True,
    )
    return unique


def get_retrieval_debug(
    question: str,
    *,
    top_k: int | None = None,
    stage_k: int = 6,
) -> dict[str, Any]:
    top_k = top_k or settings.top_k
    client = _get_openai_client()
    collection = _get_collection()
    expanded_question = _expand_question(question)

    started_at = time.perf_counter()
    embedding_response = _call_with_retries(
        lambda: client.embeddings.create(
            model=settings.openai_embedding_model,
            input=expanded_question,
        )
    )
    query_embedding = embedding_response.data[0].embedding

    n_results = max(settings.retrieve_k, stage_k, top_k)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    ranked_by_id: dict[str, dict[str, Any]] = {}
    vector_hits: list[dict[str, Any]] = []
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for document, metadata, distance in zip(documents, metadatas, distances):
        content_hash = f"{metadata.get('url', '')}::{metadata.get('section_path', '')}::{document[:120]}"
        candidate = {
            "id": content_hash,
            "vector_score": 1 / (1 + float(distance or 0)),
            "bm25_score": 0.0,
            "keyword_score": 0.0,
            "score": 0.0,
            "content": document,
            "title": metadata.get("title", ""),
            "section_path": metadata.get("section_path", ""),
            "section_id": metadata.get("section_id", ""),
            "doc_type": metadata.get("doc_type", ""),
            "url": metadata.get("url", ""),
            "has_code_example": metadata.get("has_code_example") == "true",
        }
        ranked_by_id[candidate["id"]] = candidate
        vector_hits.append(_copy_candidate(candidate, score=candidate["vector_score"]))

    lexical_hits = _search_bm25(question, limit=max(n_results * 2, stage_k * 2, top_k * 3))
    for hit in lexical_hits:
        chunk_id = hit["id"]
        candidate = ranked_by_id.get(
            chunk_id,
            {
                "id": chunk_id,
                "vector_score": 0.0,
                "bm25_score": 0.0,
                "keyword_score": 0.0,
                "score": 0.0,
                "content": hit["content"],
                "title": hit["title"],
                "section_path": hit["section_path"],
                "section_id": hit["section_id"],
                "doc_type": hit["doc_type"],
                "url": hit["url"],
                "has_code_example": hit["has_code_example"],
            },
        )
        candidate["bm25_score"] = max(candidate.get("bm25_score", 0.0), hit["bm25_score"])
        candidate["keyword_score"] = max(candidate.get("keyword_score", 0.0), hit["keyword_score"])
        ranked_by_id[chunk_id] = candidate

    vector_hits = _dedupe_ranked_items(sorted(vector_hits, key=lambda item: item["score"], reverse=True))[:stage_k]
    bm25_hits = _dedupe_ranked_items(
        [_copy_candidate(hit, score=hit.get("bm25_score", 0.0)) for hit in lexical_hits]
    )[:stage_k]
    hybrid_hits = _dedupe_ranked_items(
        _rank_candidates(question, ranked_by_id, strategy="hybrid", top_k=max(stage_k, top_k))
    )
    reranked_hits = _dedupe_ranked_items(
        _rank_candidates(question, ranked_by_id, strategy="rerank", top_k=max(stage_k, top_k))
    )
    model_context = _trim_context_for_model(question, reranked_hits[:top_k])

    return {
        "question": question,
        "expanded_question": expanded_question,
        "candidate_pool_size": len(ranked_by_id),
        "retrieval_ms": round((time.perf_counter() - started_at) * 1000, 1),
        "vector_hits": vector_hits,
        "bm25_hits": bm25_hits,
        "hybrid_hits": hybrid_hits[:stage_k],
        "reranked_hits": reranked_hits[:top_k],
        "model_context": model_context,
    }


def search_docs(question: str, top_k: int | None = None) -> list[dict[str, Any]]:
    debug = get_retrieval_debug(question, top_k=top_k)
    return debug["reranked_hits"]


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


def _trim_context_for_model(question: str, citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    trimmed: list[dict[str, Any]] = []
    for item in citations[: settings.model_context_k]:
        content = _extract_relevant_window(
            question,
            item["content"],
            settings.model_context_char_limit,
        )
        trimmed.append({**item, "content": content})
    return trimmed


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _dedupe_ranked_items(citations)


def answer_question(question: str, *, include_debug: bool = False) -> dict[str, Any]:
    debug = get_retrieval_debug(question)
    citations = debug["reranked_hits"]
    citations = _dedupe_citations(citations)
    if not citations:
        result = {
            "answer": "未在已索引文档中找到明确答案。",
            "example_code": "",
            "citations": [],
        }
        if include_debug:
            result["debug"] = debug
        return result

    top_score = citations[0].get("score", 0.0)
    if top_score < settings.min_relevance_score:
        result = {
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
        if include_debug:
            result["debug"] = debug
        return result

    client = _get_openai_client()
    model_context = debug["model_context"]
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

    result = {
        "answer": payload.get("answer", "未在已索引文档中找到明确答案。").strip(),
        "example_code": payload.get("example_code", "").strip(),
        "citations": trimmed_citations,
    }
    if include_debug:
        result["debug"] = debug
    return result


def load_index_summary() -> dict[str, Any]:
    summary_path = PROCESSED_DIR / "index_summary.json"
    if not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
