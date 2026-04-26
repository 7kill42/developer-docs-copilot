from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config import DATA_DIR
from rag import RETRIEVAL_STRATEGY_LABELS, answer_question, get_retrieval_debug


EVAL_QUESTIONS_PATH = DATA_DIR / "eval_questions.json"
STAGE_KEY_BY_STRATEGY = {
    "vector": "vector_hits",
    "hybrid": "hybrid_hits",
    "rerank": "reranked_hits",
}


def load_eval_questions(path: Path | None = None) -> list[dict[str, Any]]:
    questions_path = path or EVAL_QUESTIONS_PATH
    return json.loads(questions_path.read_text(encoding="utf-8"))


def _expected_match(item: dict[str, Any], case: dict[str, Any]) -> bool:
    expected_doc_types = set(case.get("expected_doc_types", []))
    expected_url_keywords = case.get("expected_url_keywords", [])

    doc_type_match = not expected_doc_types or item.get("doc_type", "") in expected_doc_types
    if not doc_type_match:
        return False

    if not expected_url_keywords:
        return True

    haystack = " ".join(
        [
            item.get("url", ""),
            item.get("title", ""),
            item.get("section_path", ""),
        ]
    ).lower()
    return any(keyword.lower() in haystack for keyword in expected_url_keywords)


def _hit_at_k(items: list[dict[str, Any]], case: dict[str, Any], k: int) -> bool:
    return any(_expected_match(item, case) for item in items[:k])


def _keyword_hit_ratio(text: str, expected_keywords: list[str]) -> float:
    if not expected_keywords:
        return 1.0
    lowered = text.lower()
    matched = sum(1 for keyword in expected_keywords if keyword.lower() in lowered)
    return matched / len(expected_keywords)


def run_retrieval_eval(
    questions: list[dict[str, Any]] | None = None,
    *,
    top_ks: tuple[int, int] = (3, 6),
    include_generation: bool = True,
) -> dict[str, Any]:
    cases = questions or load_eval_questions()
    strategy_rows: list[dict[str, Any]] = []
    question_rows: list[dict[str, Any]] = []
    retrieval_times: list[float] = []
    answer_keyword_ratios: list[float] = []

    for case in cases:
        question = case["question"]
        debug = get_retrieval_debug(question, top_k=max(top_ks), stage_k=max(top_ks))
        retrieval_times.append(float(debug["retrieval_ms"]))

        question_row: dict[str, Any] = {
            "question": question,
            "retrieval_ms": debug["retrieval_ms"],
        }
        for strategy, stage_key in STAGE_KEY_BY_STRATEGY.items():
            stage_items = debug[stage_key]
            for k in top_ks:
                question_row[f"{strategy}_recall_at_{k}"] = int(_hit_at_k(stage_items, case, k))

        if include_generation:
            answer_payload = answer_question(question)
            answer_text = " ".join(
                [answer_payload.get("answer", ""), answer_payload.get("example_code", "")]
            ).strip()
            keyword_ratio = _keyword_hit_ratio(answer_text, case.get("expected_keywords", []))
            question_row["answer_keyword_ratio"] = round(keyword_ratio, 3)
            question_row["answer_preview"] = answer_payload.get("answer", "")[:120]
            answer_keyword_ratios.append(keyword_ratio)

        question_rows.append(question_row)

    for strategy in STAGE_KEY_BY_STRATEGY:
        row = {"strategy": RETRIEVAL_STRATEGY_LABELS[strategy]}
        for k in top_ks:
            hits = sum(item[f"{strategy}_recall_at_{k}"] for item in question_rows)
            row[f"Recall@{k}"] = round(hits / len(question_rows), 3) if question_rows else 0.0
        strategy_rows.append(row)

    return {
        "questions_evaluated": len(question_rows),
        "avg_retrieval_ms": round(sum(retrieval_times) / len(retrieval_times), 1)
        if retrieval_times
        else 0.0,
        "avg_answer_keyword_ratio": round(sum(answer_keyword_ratios) / len(answer_keyword_ratios), 3)
        if answer_keyword_ratios
        else None,
        "strategy_rows": strategy_rows,
        "question_rows": question_rows,
    }


def _format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    widths = {
        column: max(len(column), *(len(str(row.get(column, ""))) for row in rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(str(row.get(column, "")).ljust(widths[column]) for column in columns)
        for row in rows
    ]
    return "\n".join([header, separator, *body])


def print_eval_report(summary: dict[str, Any]) -> None:
    print(f"Questions evaluated: {summary['questions_evaluated']}")
    print(f"Average retrieval latency: {summary['avg_retrieval_ms']} ms")
    if summary.get("avg_answer_keyword_ratio") is not None:
        print(f"Average answer keyword ratio: {summary['avg_answer_keyword_ratio']}")
    print()
    print("Strategy comparison")
    print(
        _format_table(
            summary["strategy_rows"],
            ["strategy", "Recall@3", "Recall@6"],
        )
    )
    print()
    print("Per-question details")
    detail_columns = [
        "question",
        "retrieval_ms",
        "vector_recall_at_3",
        "hybrid_recall_at_3",
        "rerank_recall_at_3",
    ]
    if summary.get("avg_answer_keyword_ratio") is not None:
        detail_columns.append("answer_keyword_ratio")
    print(_format_table(summary["question_rows"], detail_columns))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mini retrieval evaluation for the docs copilot.")
    parser.add_argument(
        "--questions-path",
        type=Path,
        default=EVAL_QUESTIONS_PATH,
        help="Path to eval_questions.json",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only evaluate retrieval metrics without calling answer generation.",
    )
    args = parser.parse_args()

    questions = load_eval_questions(args.questions_path)
    summary = run_retrieval_eval(questions, include_generation=not args.skip_generation)
    print_eval_report(summary)


if __name__ == "__main__":
    main()
