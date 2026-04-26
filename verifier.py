from __future__ import annotations

import re
from typing import Any


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "then",
    "than",
    "your",
    "when",
    "where",
    "have",
    "has",
    "had",
    "using",
    "used",
    "only",
    "just",
    "into",
    "also",
    "可以",
    "需要",
    "当前",
    "这个",
    "那个",
    "通过",
    "以及",
    "然后",
    "如果",
    "因为",
    "所以",
    "就是",
    "建议",
    "官方",
    "文档",
    "根据",
    "回答",
}


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_keywords(text: str) -> list[str]:
    normalized = _normalize_text(text)
    keywords: list[str] = []
    for token in normalized.split():
        if token in _STOPWORDS:
            continue
        if len(token) >= 3 or any(char.isdigit() for char in token):
            keywords.append(token)
    return list(dict.fromkeys(keywords))


def _extract_code_symbols(code: str) -> list[str]:
    symbols = re.findall(r"[A-Za-z_][A-Za-z0-9_\.]{2,}", code)
    cleaned = []
    for symbol in symbols:
        lowered = symbol.lower()
        if lowered in _STOPWORDS:
            continue
        cleaned.append(lowered)
    return list(dict.fromkeys(cleaned))


def _split_claims(answer: str) -> list[str]:
    claims = []
    for segment in re.split(r"[\n。！？!?；;]+", answer):
        cleaned = segment.strip(" -:")
        if len(cleaned) >= 8:
            claims.append(cleaned)
    return claims


def _claim_support_ratio(claim: str, evidence_blob: str) -> float:
    keywords = _extract_keywords(claim)
    if not keywords:
        return 1.0 if _normalize_text(claim) in evidence_blob else 0.0

    exact_hits = sum(1 for keyword in keywords if keyword in evidence_blob)
    return exact_hits / len(keywords)


def verify_answer(
    answer: str,
    citations: list[dict[str, Any]],
    *,
    example_code: str = "",
    min_claim_ratio: float = 0.5,
    min_grounded_coverage: float = 0.6,
) -> dict[str, Any]:
    evidence_blob = _normalize_text(
        "\n".join(
            " ".join(
                [
                    citation.get("title", ""),
                    citation.get("section_path", ""),
                    citation.get("snippet", ""),
                ]
            )
            for citation in citations
        )
    )

    claims = _split_claims(answer)
    supported_claims: list[str] = []
    unsupported_claims: list[str] = []
    claim_scores: list[float] = []

    for claim in claims:
        ratio = _claim_support_ratio(claim, evidence_blob)
        claim_scores.append(ratio)
        if ratio >= min_claim_ratio:
            supported_claims.append(claim)
        else:
            unsupported_claims.append(claim)

    if not claims:
        coverage_ratio = 1.0 if answer.strip() else 0.0
    else:
        coverage_ratio = len(supported_claims) / len(claims)

    unsupported_code_tokens: list[str] = []
    if example_code.strip():
        for symbol in _extract_code_symbols(example_code):
            if symbol not in evidence_blob:
                unsupported_code_tokens.append(symbol)

    unsupported_code_tokens = unsupported_code_tokens[:6]
    is_grounded = (
        bool(citations)
        and coverage_ratio >= min_grounded_coverage
        and len(unsupported_code_tokens) <= 2
    )

    if not citations:
        suggestion = "当前没有可核对的引用片段，应该直接拒答。"
    elif not is_grounded:
        suggestion = "当前回答里有部分结论或代码符号缺少引用支撑，建议收紧表述或直接拒答。"
    else:
        suggestion = "当前回答的大部分关键结论都能在引用片段中找到依据。"

    return {
        "is_grounded": is_grounded,
        "coverage_ratio": round(coverage_ratio, 3),
        "claims_checked": len(claims),
        "supported_claims": supported_claims,
        "unsupported_claims": unsupported_claims[:3],
        "unsupported_code_tokens": unsupported_code_tokens,
        "suggestion": suggestion,
        "claim_scores": [round(score, 3) for score in claim_scores[:6]],
    }
