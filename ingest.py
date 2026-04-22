from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

import chromadb
import httpx
import tiktoken
from bs4 import BeautifulSoup, Tag
from openai import APIConnectionError, APIStatusError, OpenAI

from config import PROCESSED_DIR, RAW_DIR, settings


SEED_PAGES = [
    {
        "url": "https://docs.sqlalchemy.org/en/20/tutorial/index.html",
        "doc_type": "tutorial",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/tutorial/engine.html",
        "doc_type": "tutorial",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/tutorial/data_select.html",
        "doc_type": "tutorial",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/session_basics.html",
        "doc_type": "tutorial",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/index.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/query.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/select.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/relationships.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/columns.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/api.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/dml.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/queryguide/inheritance.html",
        "doc_type": "orm_querying_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html",
        "doc_type": "asyncio_guide",
    },
    {
        "url": "https://docs.sqlalchemy.org/en/20/changelog/migration_20.html",
        "doc_type": "migration_guide",
    },
]


def _slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower() or "doc"


def _content_hash(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]


def _get_token_encoder():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


ENCODER = _get_token_encoder()


def _estimate_tokens(text: str) -> int:
    if ENCODER is None:
        return max(1, len(text) // 4)
    return len(ENCODER.encode(text))


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


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", text)).strip()


def _extract_main_content(soup: BeautifulSoup) -> Tag | None:
    selectors = [
        "article[role='main']",
        "div[role='main']",
        "main",
        "div.body[role='main']",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node:
            return node
    return soup.body


def _iter_sections(main_node: Tag, page_title: str, base_url: str, doc_type: str):
    heading_stack: list[str] = [page_title]
    current_heading = page_title
    current_parts: list[str] = []
    has_code_example = False

    def flush():
        nonlocal current_heading, current_parts, has_code_example
        content = _clean_text("\n\n".join(current_parts))
        if content:
            yield {
                "title": current_heading,
                "section_path": " > ".join(heading_stack),
                "doc_type": doc_type,
                "url": base_url,
                "content": content,
                "has_code_example": has_code_example,
            }
        current_parts = []
        has_code_example = False

    for node in main_node.descendants:
        if not isinstance(node, Tag):
            continue

        if node.name in {"h1", "h2", "h3"}:
            yield from flush()
            level = int(node.name[1])
            heading_text = _clean_text(node.get_text(" ", strip=True))
            if not heading_text:
                continue
            heading_stack = heading_stack[:level - 1] + [heading_text]
            current_heading = heading_text
            continue

        if node.name == "p":
            text = _clean_text(node.get_text(" ", strip=True))
            if text:
                current_parts.append(text)
            continue

        if node.name in {"pre", "code"}:
            code_text = node.get_text("\n", strip=False).strip()
            if code_text and len(code_text.splitlines()) >= 2:
                current_parts.append(f"Code example:\n{code_text}")
                has_code_example = True
            continue

        if node.name == "li":
            text = _clean_text(node.get_text(" ", strip=True))
            if text:
                current_parts.append(f"- {text}")

    yield from flush()


def _split_large_chunk(chunk: dict) -> Iterable[dict]:
    token_count = _estimate_tokens(chunk["content"])
    if token_count <= settings.chunk_token_target:
        yield chunk
        return

    paragraphs = [p for p in chunk["content"].split("\n\n") if p.strip()]
    buffer: list[str] = []
    for paragraph in paragraphs:
        candidate = "\n\n".join(buffer + [paragraph])
        if buffer and _estimate_tokens(candidate) > settings.chunk_token_target:
            yield {**chunk, "content": "\n\n".join(buffer)}
            buffer = [paragraph]
        else:
            buffer.append(paragraph)

    if buffer:
        yield {**chunk, "content": "\n\n".join(buffer)}


def crawl_and_parse_docs() -> list[dict]:
    selected_pages = SEED_PAGES[: settings.max_seed_pages]
    chunks: list[dict] = []
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for seed in selected_pages:
            response = client.get(seed["url"])
            response.raise_for_status()

            slug = _slugify(seed["url"].split("/en/20/")[-1])
            _save_text(RAW_DIR / f"{slug}.html", response.text)

            soup = BeautifulSoup(response.text, "lxml")
            main_node = _extract_main_content(soup)
            page_title = _clean_text((soup.title.string if soup.title else "").split("—")[0])
            if not page_title:
                page_title = _clean_text(main_node.get_text(" ", strip=True)[:80]) if main_node else slug

            if not main_node:
                continue

            page_sections = list(
                _iter_sections(
                    main_node=main_node,
                    page_title=page_title,
                    base_url=seed["url"],
                    doc_type=seed["doc_type"],
                )
            )

            for section in page_sections:
                for final_chunk in _split_large_chunk(section):
                    content = final_chunk["content"].strip()
                    if len(content) < 80:
                        continue
                    chunk_id = _content_hash(f"{final_chunk['url']}::{final_chunk['section_path']}::{content[:120]}")
                    chunks.append(
                        {
                            "id": chunk_id,
                            "title": final_chunk["title"],
                            "section_path": final_chunk["section_path"],
                            "doc_type": final_chunk["doc_type"],
                            "url": final_chunk["url"],
                            "has_code_example": final_chunk["has_code_example"],
                            "content": content,
                        }
                    )

    unique_chunks = list({chunk["id"]: chunk for chunk in chunks}.values())
    _save_text(PROCESSED_DIR / "chunks.json", json.dumps(unique_chunks, ensure_ascii=False, indent=2))
    return unique_chunks


def build_index() -> dict:
    if not settings.has_openai_key:
        raise RuntimeError("OPENAI_API_KEY 未配置，无法构建向量索引。")

    chunks = crawl_and_parse_docs()
    if not chunks:
        raise RuntimeError("没有解析出可索引的文档块。")

    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.openai_timeout_seconds,
        max_retries=1,
    )
    chroma_client = chromadb.PersistentClient(path=str(settings.chroma_path))

    try:
        chroma_client.delete_collection(settings.chroma_collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=settings.chroma_collection_name)

    documents = [chunk["content"] for chunk in chunks]
    embeddings: list[list[float]] = []
    # DashScope text embedding models accept at most 10 inputs per batch.
    batch_size = 10
    for start in range(0, len(documents), batch_size):
        batch = documents[start : start + batch_size]
        response = _call_with_retries(
            lambda batch=batch: client.embeddings.create(
                model=settings.openai_embedding_model,
                input=batch,
            )
        )
        embeddings.extend(item.embedding for item in response.data)

    collection.add(
        ids=[chunk["id"] for chunk in chunks],
        documents=documents,
        metadatas=[
            {
                "title": chunk["title"],
                "section_path": chunk["section_path"],
                "doc_type": chunk["doc_type"],
                "url": chunk["url"],
                "has_code_example": str(chunk["has_code_example"]).lower(),
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )

    summary = {
        "pages_indexed": len(SEED_PAGES[: settings.max_seed_pages]),
        "chunks_indexed": len(chunks),
        "collection_name": settings.chroma_collection_name,
    }
    _save_text(PROCESSED_DIR / "index_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    result = build_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))
