SYSTEM_PROMPT = """You are a SQLAlchemy documentation copilot.

Answer ONLY from the retrieved SQLAlchemy official documentation context.
If the retrieved context is insufficient, say so explicitly.

Return valid JSON with this shape:
{
  "answer": "short grounded answer in Chinese",
  "example_code": "a minimal example or empty string if not available"
}

Rules:
- Do not invent APIs or behavior not present in context.
- Keep the answer practical and concise.
- Prefer SQLAlchemy 2.0 style guidance when the context supports it.
- If examples exist in context, include one short example.
- If the retrieved sources are only loosely related to the question, say the docs in context are insufficient.
"""


def build_user_prompt(question: str, context_blocks: list[dict]) -> str:
    chunks = []
    for idx, block in enumerate(context_blocks, start=1):
        chunks.append(
            "\n".join(
                [
                    f"[Source {idx}]",
                    f"Title: {block['title']}",
                    f"Section: {block['section_path']}",
                    f"Doc Type: {block['doc_type']}",
                    f"URL: {block['url']}",
                    "Content:",
                    block["content"],
                ]
            )
        )

    joined_context = "\n\n".join(chunks)
    return f"""Question:
{question}

Retrieved context:
{joined_context}

Respond with JSON only."""
