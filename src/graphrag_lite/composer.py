"""Answer composition: extractive and optional LLM-backed.

The default extractive composer stitches the top-ranked passage texts together
with inline citation tags ``[Pn]`` so the caller can trace every sentence back
to a source passage.

Optional LLM path: ``compose_answer_llm`` lazy-imports ``anthropic`` and uses it
to synthesise a coherent response.  Only available with ``pip install graphrag-lite[llm]``.
"""

from __future__ import annotations

from graphrag_lite.corpus import Passage
from graphrag_lite.retriever import RetrievalResult


def compose_answer(
    query: str,
    results: list[RetrievalResult],
    passages: dict[str, Passage],
) -> str:
    """Build an extractive cited answer from retrieval results.

    Each passage text is prefixed with its citation tag ``[<passage_id>]``.
    The final string is headed by a restatement of the query.

    Args:
        query: The original user question.
        results: Ordered list of RetrievalResult from HybridRetriever.retrieve().
        passages: Mapping of passage_id â†’ Passage for text lookup.

    Returns:
        A multi-paragraph string with inline citation tags.
    """
    if not results:
        return f"Query: {query}\n\nNo relevant passages found."

    parts: list[str] = [f"Query: {query}\n"]
    for r in results:
        p = passages.get(r.passage_id)
        if p is None:
            continue
        citation = f"[{r.passage_id}]"
        entities_note = ""
        if r.matched_entities:
            entities_note = f" (entities: {', '.join(r.matched_entities)})"
        parts.append(f"{citation}{entities_note}\n{p.text}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Optional LLM composer (lazy-imports anthropic)
# ---------------------------------------------------------------------------


def compose_answer_llm(
    query: str,
    results: list[RetrievalResult],
    passages: dict[str, Passage],
    client: object,
    model: str = "claude-haiku-4-5",
    max_tokens: int = 512,
) -> str:  # pragma: no cover
    """Synthesise a fluent answer using an Anthropic model.

    Requires ``pip install graphrag-lite[llm]``.

    Args:
        query: The user question.
        results: Retrieval results to use as context.
        passages: Passage lookup dict.
        client: An ``anthropic.Anthropic`` instance.
        model: Claude model id.
        max_tokens: Maximum response tokens.

    Returns:
        Synthesised answer string (not extractive).

    Raises:
        ImportError: if ``anthropic`` is not installed.
    """
    try:
        import anthropic  # noqa: F401  # lazy
    except ImportError as exc:
        raise ImportError(
            "Install graphrag-lite[llm] to use LLM-based answer composition."
        ) from exc

    context_parts: list[str] = []
    for r in results:
        p = passages.get(r.passage_id)
        if p:
            context_parts.append(f"[{r.passage_id}] {p.text}")

    context = "\n\n".join(context_parts)
    prompt = (
        f"Answer the following question using ONLY the provided context passages. "
        f"Cite passage ids inline like [P0].\n\n"
        f"Question: {query}\n\nContext:\n{context}"
    )

    message = client.messages.create(  # type: ignore[attr-defined]
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    text_block = next((b for b in message.content if getattr(b, "type", None) == "text"), None)
    return text_block.text if text_block is not None else ""
