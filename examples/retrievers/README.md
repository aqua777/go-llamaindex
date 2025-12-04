# Retriever Examples

This directory contains examples demonstrating various retrieval strategies in `go-llamaindex`.

## Examples

### 1. Auto-Merging Retriever (`auto_merging/`)

Demonstrates hierarchical document retrieval with automatic merging of child nodes into parent nodes.

**Key Concepts:**
- Hierarchical document structure (parent-child relationships)
- Automatic merging when threshold of children is retrieved
- Gap filling between consecutive nodes
- Score aggregation across merged nodes

**Use Cases:**
- Documents with natural hierarchies (chapters → sections → paragraphs)
- When broader context is needed for related chunks
- Reducing fragmentation in retrieval results

**Run:**
```bash
cd auto_merging && go run main.go
```

### 2. Router Retriever (`router/`)

Shows query routing to specialized retrievers based on query classification.

**Key Concepts:**
- Multiple domain-specific retrievers
- Query-based retriever selection
- Selector interface for routing logic
- Result deduplication across retrievers

**Use Cases:**
- Multi-domain knowledge bases
- Specialized retrieval strategies per topic
- Optimizing retrieval for different query types

**Run:**
```bash
cd router && go run main.go
```

### 3. Composable Retrievers (`composable/`)

Demonstrates combining multiple retrievers in flexible configurations.

**Key Concepts:**
- Retriever composition patterns
- Sequential and parallel retrieval
- Result filtering and transformation
- Building complex retrieval pipelines

**Use Cases:**
- Multi-stage retrieval pipelines
- Combining dense and sparse retrieval
- Custom retrieval workflows

**Run:**
```bash
cd composable && go run main.go
```

### 4. BM25 Retriever (`bm25/`)

Shows sparse retrieval using the BM25 algorithm for keyword-based search.

**Key Concepts:**
- BM25 (Best Matching 25) algorithm
- Sparse embeddings vs dense embeddings
- Term frequency and inverse document frequency
- Hybrid retrieval (combining with dense retrieval)

**Use Cases:**
- Keyword-heavy queries
- When exact term matching is important
- Hybrid search combining semantic and lexical matching

**Run:**
```bash
cd bm25 && go run main.go
```

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable) - for examples using dense embeddings

## Retriever Types

| Retriever | Description | Best For |
|-----------|-------------|----------|
| **VectorRetriever** | Dense semantic search | General semantic similarity |
| **FusionRetriever** | Combines multiple retrievers | Ensemble retrieval |
| **AutoMergingRetriever** | Merges hierarchical nodes | Hierarchical documents |
| **RouterRetriever** | Routes to specialized retrievers | Multi-domain knowledge |
| **BM25** | Sparse keyword matching | Exact term matching |

## Related Python Examples

These examples correspond to the following Python notebooks:
- `retrievers/auto_merging_retriever.ipynb`
- `retrievers/router_retriever.ipynb`
- `retrievers/composable_retrievers.ipynb`
- `retrievers/bm25_retriever.ipynb`

## Components Used

- `rag/retriever.AutoMergingRetriever` - Hierarchical merging
- `rag/retriever.RouterRetriever` - Query routing
- `rag/retriever.FusionRetriever` - Retriever fusion
- `embedding.BM25` - Sparse BM25 embeddings
- `storage.StorageContext` - Document storage for parent nodes
