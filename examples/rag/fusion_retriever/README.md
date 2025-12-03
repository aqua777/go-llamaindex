# Fusion Retriever Example

This example demonstrates fusion retrieval strategies in Go, corresponding to the Python `low_level/fusion_retriever.ipynb` example.

## Overview

Learn how to combine results from multiple retrievers using different fusion strategies:
1. **Simple Fusion** - Takes max score for duplicates
2. **Reciprocal Rank Fusion (RRF)** - Combines rankings across retrievers
3. **Relative Score Fusion** - Normalizes and weights scores
4. **Distance-Based Score Fusion** - Uses statistical normalization

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/rag/fusion_retriever
go run main.go
```

## Components Used

- `rag/retriever.FusionRetriever` - Combines multiple retrievers
- `rag/retriever.FusionMode` - Fusion strategy selection
- `rag.VectorRetriever` - Base vector retriever
