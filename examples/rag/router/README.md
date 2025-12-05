# Query Routing Example

This example demonstrates query routing in Go, corresponding to the Python `low_level/router.ipynb` example.

## Overview

Learn how to:
1. Create multiple query engines for different data sources
2. Use selectors to route queries to appropriate engines
3. Combine responses from multiple engines

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Usage

```bash
export OPENAI_API_KEY=your-api-key
cd examples/rag/router
go run main.go
```

## Components Used

- `rag/queryengine.RouterQueryEngine` - Routes queries to appropriate engines
- `rag/queryengine.QueryEngineTool` - Wraps query engines with metadata
- `selector.LLMSingleSelector` - LLM-based single selection
- `selector.LLMMultiSelector` - LLM-based multi selection
