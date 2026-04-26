from __future__ import annotations

import argparse
from pathlib import Path

from evals import load_golden_qa, print_eval_report, run_retrieval_eval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation on the golden QA set.")
    parser.add_argument(
        "--golden-path",
        type=Path,
        default=Path(__file__).resolve().parent / "golden_qa.jsonl",
        help="Path to golden_qa.jsonl",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Only evaluate retrieval metrics without calling answer generation.",
    )
    args = parser.parse_args()

    questions = load_golden_qa(args.golden_path)
    supported_questions = [item for item in questions if not item.get("should_refuse", False)]
    summary = run_retrieval_eval(supported_questions, include_generation=not args.skip_generation)
    print_eval_report(summary)


if __name__ == "__main__":
    main()
