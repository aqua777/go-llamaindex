# Document Ingestion Pipeline Examples

This directory contains examples demonstrating document ingestion pipelines in go-llamaindex.

## Examples

### 1. Basic Ingestion (`main.go`)

Demonstrates the core ingestion pipeline functionality.

**Features:**
- Document loading with SimpleDirectoryReader
- Text splitting with SentenceSplitter
- Node creation with relationships
- Embedding generation
- Metadata-aware splitting

**Run:**
```bash
go run main.go
```

### 2. Document Management (`document_management/`)

Demonstrates document lifecycle management with deduplication.

**Features:**
- Docstore strategies (Upserts, UpsertsAndDelete, DuplicatesOnly)
- Hash-based deduplication
- Document updates and deletions
- Vector store synchronization
- Transformation caching

**Run:**
```bash
cd document_management && go run main.go
```

## Key Concepts

### Ingestion Pipeline

```go
pipeline := ingestion.NewIngestionPipeline(
    ingestion.WithPipelineName("my_pipeline"),
    ingestion.WithDocstore(docStore),
    ingestion.WithVectorStore(vectorStore),
    ingestion.WithDocstoreStrategy(ingestion.DocstoreStrategyUpserts),
    ingestion.WithTransformations([]ingestion.TransformComponent{
        textSplitter,
        embeddingTransform,
    }),
)

nodes, err := pipeline.Run(ctx, documents, nil)
```

### Docstore Strategies

| Strategy | Description |
|----------|-------------|
| `Upserts` | Update existing, add new documents |
| `UpsertsAndDelete` | Also delete documents not in current batch |
| `DuplicatesOnly` | Skip exact duplicates by hash |

### Transform Components

```go
type TransformComponent interface {
    Transform(ctx context.Context, nodes []schema.Node) ([]schema.Node, error)
    Name() string
}
```

### Pipeline Caching

```go
cache := ingestion.NewIngestionCache()
pipeline := ingestion.NewIngestionPipeline(
    ingestion.WithPipelineCache(cache),
    // ...
)
```

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Components Used

- `ingestion.IngestionPipeline` - Main pipeline orchestrator
- `textsplitter.SentenceSplitter` - Text chunking
- `embedding.OpenAIEmbedding` - Embedding transformation
- `ingestion.IngestionCache` - Transformation caching
- `ingestion.DocstoreStrategy` - Deduplication strategies
