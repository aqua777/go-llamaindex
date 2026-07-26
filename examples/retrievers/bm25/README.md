# BM25 Retriever Example

Demonstrates `BM25Retriever` — a standalone, in-memory sparse retriever that
ranks nodes using the BM25 algorithm without requiring an external vector
database.

## What This Example Demonstrates

- Creating a `BM25Retriever` over a fixed node corpus
- Retrieving top-K results ranked by BM25 score
- Customizing BM25 parameters (`k1`, `b`) via `WithBM25Options`
- Applying metadata filters before scoring (`query.Filters`)
- Supplying a pre-fitted model via `WithBM25Model`
- The lexical vs. semantic gap: semantically related but lexically different
  nodes score zero

## Prerequisites

No external services required. All retrieval is in-memory.

## How to Run

```bash
cd golang
go run ./examples/retrievers/bm25/
```

## Expected Output

The demo prints query results for several queries, showing how BM25 ranks
nodes that share exact tokens with the query above nodes that do not.  The
final section demonstrates that a GPU/tensor query returns zero scores for
nodes about neural networks — illustrating why hybrid search (BM25 + dense
embeddings) is useful.

## Key Concepts

| Concept | Description |
|---------|-------------|
| `BM25Retriever` | Fits BM25 on the corpus at construction; immutable after that. |
| `WithBM25TopK` | Limits the number of results returned by `Retrieve`. |
| `WithBM25Options` | Forwards parameters (`k1`, `b`, stopwords, tokenizer) to the BM25 constructor. |
| `WithBM25Model` | Injects a pre-fitted `*embedding.BM25`; skips internal fitting. |
| `query.Filters` | Metadata filters applied before BM25 scoring; excluded nodes never score. |

## BM25 Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `k1` | `1.5` | Term-frequency saturation. Higher values reward repeated terms more. |
| `b` | `0.75` | Document-length normalization. `0` = no normalization; `1` = full normalization. |
