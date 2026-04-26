# Evaluation Notes

This directory turns the project from a demo-only RAG app into a more discussable, measurable system.

## Dataset

- `golden_qa.jsonl` contains supported and out-of-scope SQLAlchemy questions.
- Supported questions are used to compare retrieval strategies.
- Out-of-scope questions are used to inspect refusal behavior.

## Scripts

- `python3 eval/eval_retrieval.py --skip-generation`
  Measures retrieval-focused metrics such as `Recall@3`, `Recall@6`, and retrieval latency.
- `python3 eval/eval_retrieval.py`
  Runs the same retrieval evaluation and also checks whether generated answers contain expected keywords.
- `python3 eval/eval_answer.py`
  Measures answer keyword coverage, grounded answer ratio, and refusal accuracy.

## Current Metrics

- `Recall@K`
  Checks whether the expected SQLAlchemy page/section appears in the top-K retrieved chunks.
- `Average Answer Keyword Ratio`
  Uses expected keywords as a light-weight proxy for answer completeness.
- `Grounded Answer Ratio`
  Counts answer keyword coverage only when citations are returned, as a simple groundedness proxy.
- `Refusal Accuracy`
  Checks whether out-of-scope questions trigger a refusal instead of unsupported generation.
- `Average Retrieval Latency`
  Reports the average retrieval time for the evaluated question set.

## Interview Framing

This evaluation setup supports several strong discussion points:

- Hybrid retrieval can be compared directly against vector-only and reranked stages.
- Refusal behavior is explicitly tested instead of being left to prompt wording alone.
- The project has a path toward larger-scale golden set evaluation without changing the app architecture.
