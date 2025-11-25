# Metadata Filtering and Streaming Example

This example demonstrates two key production-ready features of the Go LlamaIndex implementation:

## Features Demonstrated

### 1. Metadata Filtering
Query documents with metadata-based filters to scope your retrieval. This is useful for:
- Multi-tenant applications (filter by user_id, tenant_id)
- Document categorization (filter by category, type, author)
- Time-based filtering (filter by date, version)

**Example:**
```go
query := schema.QueryBundle{
    QueryString: "What programming languages are mentioned?",
    Filters: &schema.MetadataFilters{
        Filters: []schema.MetadataFilter{
            {
                Key:      "category",
                Value:    "technology",
                Operator: schema.FilterOperatorEq,
            },
        },
    },
}
response, err := queryEngine.Query(ctx, query)
```

### 2. Streaming Responses
Get real-time token-by-token responses from the LLM instead of waiting for the complete response. This provides:
- Better user experience with immediate feedback
- Lower perceived latency
- Ability to show progress indicators

**Example:**
```go
streamResponse, err := queryEngine.QueryStream(ctx, query)
if err != nil {
    log.Fatal(err)
}

for token := range streamResponse.ResponseStream {
    fmt.Print(token)
}
```

## Running the Example

1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

2. Run the example:
```bash
go run main.go
```

## What the Example Does

1. Creates sample documents with metadata (category and author)
2. Loads documents from a directory
3. Splits documents into chunks
4. Generates embeddings for all chunks
5. Stores chunks in a persistent ChromemStore vector database
6. Demonstrates 5 different query scenarios:
   - Query without filters
   - Query with category filter
   - Query with author filter
   - Streaming query
   - Streaming query with filter

## Architecture

The implementation follows the LlamaIndex architecture:

```
┌─────────────┐
│ QueryEngine │
└──────┬──────┘
       │
       ├─────────────┐
       │             │
┌──────▼──────┐ ┌───▼────────┐
│  Retriever  │ │Synthesizer │
└──────┬──────┘ └───┬────────┘
       │            │
┌──────▼──────┐ ┌──▼─────┐
│VectorStore  │ │  LLM   │
│(Chromem)    │ │(OpenAI)│
└─────────────┘ └────────┘
```

## Key Components

- **Schema Package**: Core data structures (Node, QueryBundle, MetadataFilters, etc.)
- **VectorStore**: Interface for vector storage with filtering support
- **ChromemStore**: Persistent vector store implementation using chromem-go
- **LLM Interface**: Supports both regular and streaming completions
- **Retriever**: Handles embedding generation and vector search with filters
- **Synthesizer**: Generates responses from retrieved context (streaming or non-streaming)
- **QueryEngine**: Orchestrates the full RAG pipeline

## Production Readiness

This example showcases production-ready features:
- ✅ Metadata filtering for scoped retrieval
- ✅ Streaming responses for better UX
- ✅ Persistent vector storage
- ✅ Configurable components
- ✅ Error handling
- ✅ Clean architecture with interfaces

