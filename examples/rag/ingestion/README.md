# Document Ingestion Pipeline Example

This example demonstrates the document ingestion pipeline in Go, corresponding to the Python `low_level/ingestion.ipynb` example.

## Overview

Learn how to:
1. Create an ingestion pipeline with transformations
2. Use text splitters and node parsers
3. Apply embedding transformations
4. Handle document deduplication
5. Use pipeline caching

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/rag/ingestion
go run main.go
```

## Components Used

- `ingestion.IngestionPipeline` - Main pipeline orchestrator
- `textsplitter.SentenceSplitter` - Text chunking
- `nodeparser.SentenceNodeParser` - Node parsing with relationships
- `embedding.OpenAIEmbedding` - Embedding transformation
- `ingestion.IngestionCache` - Transformation caching
