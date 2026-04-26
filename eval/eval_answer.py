from __future__ import annotations

import argparse
from pathlib import Path

from evals import load_golden_qa, run_answer_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answer/refusal evaluation on the golden QA set.")
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_qa.jsonl",
        help="Path to golden_qa.jsonl",
    )
    args = parser.parse_args()

    summary = run_answer_eval(load_golden_qa(args.golden_path))
    print(f"Questions evaluated: {summary['questions_evaluated']}")
    print(f"Supported questions: {summary['supported_questions']}")
    print(f"Refusal questions: {summary['refusal_questions']}")
    if summary["avg_answer_keyword_ratio"] is not None:
        print(f"Average answer keyword ratio: {summary['avg_answer_keyword_ratio']}")
    if summary["grounded_answer_ratio"] is not None:
        print(f"Grounded answer ratio: {summary['grounded_answer_ratio']}")
    if summary["refusal_accuracy"] is not None:
        print(f"Refusal accuracy: {summary['refusal_accuracy']}")
    print()
    print("Per-question details")
    for row in summary["question_rows"]:
        print(
            f"- {row['question_type']}: refuse={row['did_refuse']} expected_refuse={row['should_refuse']} "
            f"keyword_ratio={row['keyword_ratio']} citations={row['citation_count']}"
        )


if __name__ == "__main__":
    main()
