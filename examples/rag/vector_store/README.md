# Vector Store Operations Example

This example demonstrates vector store operations in Go, corresponding to the Python `low_level/vector_store.ipynb` example.

## Overview

Learn how to:
1. Create and configure a vector store
2. Add nodes with embeddings
3. Query for similar nodes
4. Apply metadata filters
5. Delete nodes
6. Use persistence

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/rag/vector_store
go run main.go
```

## Components Used

- `rag/store/chromem.ChromemStore` - Vector store implementation
- `embedding.OpenAIEmbedding` - Embedding generation
- `schema.VectorStoreQuery` - Query configuration
- `schema.MetadataFilters` - Metadata filtering
