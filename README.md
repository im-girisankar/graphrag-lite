# graphrag-lite

> Answers that follow the connections, not just keyword matches.

A compact, dependency-light GraphRAG implementation: raw text corpus → knowledge graph → hybrid graph+vector retrieval → cited, extractive answer.

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| M1 | Corpus split, heuristic entity/relation extraction, in-memory knowledge graph | ✅ Complete |
| M2 | Hashing embedder fallback, vector index, hybrid retriever (graph + vector), answer composer | ✅ Complete |
| M3 | Evaluation harness (recall@k, hit-rate), CLI (`grl build` / `grl ask`), save/load | ✅ Complete |

## Why GraphRAG?

Traditional RAG retrieves by vector similarity: *which passage looks most like the query?*  
GraphRAG also asks: *which entities appear in the query, who are their neighbours, and what do those neighbours talk about?*

If a passage never mentions your query terms directly but is 2 hops away in the entity co-occurrence graph, pure vector search misses it. Hybrid scoring catches it.

## Installation

```bash
# Core (no external deps — HashEmbedder is pure Python)
pip install -e .

# With real sentence embeddings
pip install -e ".[embeddings]"

# With LLM-composed answers (requires ANTHROPIC_API_KEY)
pip install -e ".[llm]"

# Dev (pytest + ruff)
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
from graphrag_lite import GraphRAGPipeline

pipeline = GraphRAGPipeline()
pipeline.build([
    "Machine Learning is a subfield of Artificial Intelligence.",
    "Neural Networks learn through Backpropagation.",
    "Backpropagation uses gradient descent to minimize the loss function.",
])

answer = pipeline.ask("How do Neural Networks learn?")
print(answer)
```

### CLI

```bash
# Build index from a JSON array of strings
grl build --corpus corpus.json --output ./my_index

# Ask a question
grl ask --index ./my_index --query "How do Neural Networks learn?"

# Tune the vector/graph balance (0=pure graph, 1=pure vector, default 0.5)
grl ask --index ./my_index --query "..." --alpha 0.3
```

## Architecture

```
docs[] ──► split_corpus ──► Passage[]
                                │
                    extract_entities / extract_relations
                                │
                          KnowledgeGraph
                         (nodes, edges, passage_entities)
                                │
                    ┌───────────┴────────────┐
              HashEmbedder              VectorIndex
               (MD5-based,            (cosine flat)
               pure Python)
                    └───────────┬────────────┘
                          HybridRetriever
                   score = α·vector + (1-α)·graph
                                │
                         compose_answer
                      (extractive + [Pn] citations)
```

### Hybrid Scoring

```
score(passage) = alpha * vector_score + (1 - alpha) * graph_score

vector_score = cosine_similarity(embed(query), embed(passage))   ∈ [0, 1]

graph_score  = min(1.0, |matched_entities| / max(1, |query_entities|))

matched_entities = passage_entities ∩ (query_entities ∪ 2-hop_neighbours)
```

`alpha=0.5` (default) blends both signals equally.

## Evaluation

```python
from graphrag_lite.eval import FIXTURE_EXAMPLES, evaluate
from graphrag_lite.retriever import HybridRetriever

# ... build pipeline ...
retriever = HybridRetriever(pipeline.graph, pipeline.vector_index,
                             pipeline.passages, pipeline.embedder)
metrics = evaluate(retriever, FIXTURE_EXAMPLES, k=4)
print(metrics)  # {'recall@k': 1.0, 'hit_rate': 1.0}
```

## Tests

```bash
pytest          # runs all tests (offline, no network/keys required)
ruff check src  # lint
```

## License

MIT © 2026 Girisankar G
