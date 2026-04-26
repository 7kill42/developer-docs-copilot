from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterable

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


def estimate_tokens(text: str) -> int:
    return _estimate_tokens(text)


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


def _clean_heading_text(text: str) -> str:
    return _clean_text(text.replace("¶", " "))


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


def _is_heading_tag(node: Tag) -> bool:
    return bool(node.name and re.fullmatch(r"h[1-6]", node.name))


def _extract_heading(node: Tag | None, fallback: str) -> str:
    if not node:
        return fallback
    return _clean_heading_text(node.get_text(" ", strip=True)) or fallback


def _extract_section_id(section_node: Tag, heading_node: Tag | None) -> str:
    if section_node.get("id"):
        return str(section_node["id"])
    if heading_node and heading_node.get("id"):
        return str(heading_node["id"])

    for child in section_node.children:
        if isinstance(child, Tag) and child.name == "span" and child.get("id"):
            return str(child["id"])

    if heading_node:
        headerlink = heading_node.select_one("a.headerlink[href^='#']")
        if headerlink and headerlink.get("href"):
            return str(headerlink["href"]).lstrip("#")

    return ""


def _build_section_url(base_url: str, section_id: str) -> str:
    return f"{base_url}#{section_id}" if section_id else base_url


def _is_code_container(node: Tag) -> bool:
    classes = set(node.get("class", []))
    if node.name == "pre":
        return True
    if node.name == "div" and "literal-block-wrapper" in classes:
        return node.find("pre") is not None
    if node.name == "div" and any(cls == "highlight" or cls.startswith("highlight-") for cls in classes):
        return node.find("pre") is not None
    return False


def _extract_code_text(node: Tag) -> str:
    pre = node if node.name == "pre" else node.find("pre")
    if pre is None:
        return ""
    code_text = pre.get_text("\n", strip=False).strip()
    return code_text if len(code_text.splitlines()) >= 2 else ""


def _append_unique(parts: list[str], seen: set[str], values: Iterable[str]) -> None:
    for value in values:
        cleaned = value.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        parts.append(cleaned)


def _extract_block_parts(node: Tag) -> tuple[list[str], bool]:
    if node.name in {"script", "style", "nav", "section"} or _is_heading_tag(node):
        return [], False

    if _is_code_container(node):
        code_text = _extract_code_text(node)
        if code_text:
            return [f"Code example:\n{code_text}"], True
        return [], False

    if node.name in {"ul", "ol"}:
        items = []
        for li in node.find_all("li", recursive=False):
            text = _clean_text(li.get_text(" ", strip=True))
            if text:
                items.append(f"- {text}")
        return items, False

    if node.name == "dl":
        items = []
        current_term = ""
        for child in node.find_all(["dt", "dd"], recursive=False):
            text = _clean_text(child.get_text(" ", strip=True))
            if not text:
                continue
            if child.name == "dt":
                current_term = text
            elif current_term:
                items.append(f"- {current_term}: {text}")
                current_term = ""
            else:
                items.append(f"- {text}")
        return items, False

    if node.name in {"p", "blockquote"}:
        text = _clean_text(node.get_text(" ", strip=True))
        return ([text] if text else []), False

    parts: list[str] = []
    seen_parts: set[str] = set()
    has_code_example = False
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        child_parts, child_has_code = _extract_block_parts(child)
        _append_unique(parts, seen_parts, child_parts)
        has_code_example = has_code_example or child_has_code

    if parts:
        return parts, has_code_example

    text = _clean_text(node.get_text(" ", strip=True))
    return ([text] if text else []), has_code_example


def _extract_container_parts(node: Tag, *, include_sections: bool) -> tuple[list[str], bool]:
    parts: list[str] = []
    seen_parts: set[str] = set()
    has_code_example = False
    for child in node.children:
        if not isinstance(child, Tag):
            continue
        if not include_sections and child.name == "section":
            continue
        child_parts, child_has_code = _extract_block_parts(child)
        _append_unique(parts, seen_parts, child_parts)
        has_code_example = has_code_example or child_has_code
    return parts, has_code_example


def _iter_section_nodes(
    section_node: Tag,
    *,
    page_title: str,
    base_url: str,
    doc_type: str,
    heading_stack: list[str],
):
    heading_node = section_node.find(re.compile(r"^h[1-6]$"), recursive=False)
    heading_text = _extract_heading(heading_node, page_title)
    current_path = heading_stack if heading_stack and heading_stack[-1] == heading_text else heading_stack + [heading_text]
    section_id = _extract_section_id(section_node, heading_node)
    parts, has_code_example = _extract_container_parts(section_node, include_sections=False)
    content = _clean_text("\n\n".join(parts))
    if content:
        yield {
            "title": heading_text,
            "section_path": " > ".join(current_path),
            "section_id": section_id,
            "doc_type": doc_type,
            "url": _build_section_url(base_url, section_id),
            "content": content,
            "has_code_example": has_code_example,
        }

    for child_section in section_node.find_all("section", recursive=False):
        yield from _iter_section_nodes(
            child_section,
            page_title=page_title,
            base_url=base_url,
            doc_type=doc_type,
            heading_stack=current_path,
        )


def _iter_sections(main_node: Tag, page_title: str, base_url: str, doc_type: str):
    intro_parts, intro_has_code = _extract_container_parts(main_node, include_sections=False)
    intro_content = _clean_text("\n\n".join(intro_parts))
    if intro_content:
        yield {
            "title": page_title,
            "section_path": page_title,
            "section_id": "",
            "doc_type": doc_type,
            "url": base_url,
            "content": intro_content,
            "has_code_example": intro_has_code,
        }

    top_level_sections = main_node.find_all("section", recursive=False)
    if top_level_sections:
        for section_node in top_level_sections:
            yield from _iter_section_nodes(
                section_node,
                page_title=page_title,
                base_url=base_url,
                doc_type=doc_type,
                heading_stack=[page_title],
            )
        return

    parts, has_code_example = _extract_container_parts(main_node, include_sections=True)
    content = _clean_text("\n\n".join(parts))
    if content:
        yield {
            "title": page_title,
            "section_path": page_title,
            "section_id": "",
            "doc_type": doc_type,
            "url": base_url,
            "content": content,
            "has_code_example": has_code_example,
        }


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


def _selected_seed_pages() -> list[dict[str, str]]:
    if settings.max_seed_pages <= 0:
        return list(SEED_PAGES)
    return SEED_PAGES[: settings.max_seed_pages]


def crawl_and_parse_docs() -> list[dict]:
    selected_pages = _selected_seed_pages()
    chunks: list[dict] = []
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for seed in selected_pages:
            response = client.get(seed["url"])
            response.raise_for_status()

            slug = _slugify(seed["url"].split("/en/20/")[-1])
            _save_text(RAW_DIR / f"{slug}.html", response.text)

            soup = BeautifulSoup(response.text, "lxml")
            main_node = _extract_main_content(soup)
            page_title = _clean_heading_text((soup.title.string if soup.title else "").split("—")[0])
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
                            "section_id": final_chunk["section_id"],
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
                "section_id": chunk.get("section_id", ""),
                "doc_type": chunk["doc_type"],
                "url": chunk["url"],
                "has_code_example": str(chunk["has_code_example"]).lower(),
            }
            for chunk in chunks
        ],
        embeddings=embeddings,
    )

    summary = {
        "pages_indexed": len(selected_pages),
        "chunks_indexed": len(chunks),
        "collection_name": settings.chroma_collection_name,
    }
    _save_text(PROCESSED_DIR / "index_summary.json", json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    result = build_index()
    print(json.dumps(result, ensure_ascii=False, indent=2))
