# CLAUDE.md — graphrag-lite

## Project overview
Compact GraphRAG library: text corpus → knowledge graph → hybrid graph+vector retrieval → cited answer.

## Key design decisions

### HashEmbedder (embeddings.py)
- Uses `hashlib.md5` (NOT Python `hash()` which is PYTHONHASHSEED-randomised).
- Bag-of-hashed-tokens accumulator, L2-normalised. Dim=256 default.
- Fully deterministic: same input → same vector across any Python process.

### Hybrid scoring formula (retriever.py)
```
score = alpha * vector_score + (1 - alpha) * graph_score
vector_score = cosine_similarity(embed(query), embed(passage))
graph_score  = min(1.0, |matched_entities| / max(1, |query_entities|))
matched_entities = passage_entities ∩ (query_entities ∪ 2-hop_graph_neighbours)
```

### Entity extraction (extractor.py)
- Heuristic regex: multi-word Title-Case phrases first, then single Title-Case words not at sentence start.
- Optional LLM path via `extract_entities_llm()` — lazy-imports `anthropic`.

### Optional dependencies
- `sentence-transformers`: lazy-imported inside `SentenceTransformerEmbedder.embed()`.
- `anthropic`: lazy-imported inside `extract_entities_llm()` and `compose_answer_llm()`.

## Dev workflow
```bash
pip install -e ".[dev]"
pytest          # all tests offline
ruff check src  # lint (line-length 100, E/F/I/UP/B, ignore E501)
```

## File layout
```
src/graphrag_lite/
  __init__.py      # version, public exports
  corpus.py        # Passage, split_corpus
  extractor.py     # extract_entities, extract_relations, extract_entities_llm
  graph.py         # KnowledgeGraph, build_graph
  embeddings.py    # Embedder ABC, HashEmbedder, SentenceTransformerEmbedder, cosine_similarity
  index.py         # VectorIndex
  retriever.py     # HybridRetriever, RetrievalResult
  composer.py      # compose_answer, compose_answer_llm
  eval.py          # recall_at_k, hit_rate, evaluate, FIXTURE_EXAMPLES
  pipeline.py      # GraphRAGPipeline (build, ask, save, load)
  cli.py           # grl build / grl ask CLI

tests/
  conftest.py      # small_corpus, built_pipeline, two_hop_pipeline fixtures
  test_corpus.py
  test_extractor.py
  test_graph.py
  test_embeddings.py
  test_index.py
  test_retriever.py
  test_eval.py
  test_composer.py
  test_pipeline.py
  test_cli.py
```
