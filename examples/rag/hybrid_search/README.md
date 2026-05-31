# Hybrid Search Demo

Demonstrates end-to-end hybrid retrieval using `SimpleVectorStore` with `QueryModeSparse` and `QueryModeHybrid`, comparing results across all three retrieval modes.

## What This Demonstrates

- `QueryModeDefault` — pure dense cosine-similarity ranking
- `QueryModeSparse` — BM25 keyword ranking (no embeddings required)
- `QueryModeHybrid` — weighted combination of dense and sparse scores via alpha
- `VectorRetriever` forwarding `QueryStr` automatically in hybrid mode

## Prerequisites

No environment variables or API keys required. The demo uses deterministic in-memory embeddings and BM25.

## Running

```bash
cd golang
go run ./examples/rag/hybrid_search/
```

## Expected Output

The demo indexes 6 nodes about ML, neural networks, databases, NLP, and computer vision, then issues the query `"machine learning neural networks"` in all three modes. Results differ across modes:

- **Dense** ranks nodes by embedding vector proximity.
- **Sparse** ranks nodes by keyword overlap (BM25).
- **Hybrid** blends both signals, surfacing nodes that are both lexically and semantically relevant.
