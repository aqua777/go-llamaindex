# Index Examples

This directory contains examples demonstrating different index types in `go-llamaindex`.

## Examples

### 1. Knowledge Graph Index (`knowledge_graph/`)

Builds a knowledge graph by extracting triplets from documents.

**Features:**
- Automatic triplet extraction using LLM
- Manual triplet insertion
- Graph store for triplet storage
- Keyword-based retrieval
- Graph traversal with configurable depth
- Custom triplet extraction functions

**Run:**
```bash
cd knowledge_graph && go run main.go
```

### 2. Summary Index (`summary_index/`)

Stores documents in a list and synthesizes answers from all nodes.

**Features:**
- Simple list-based document storage
- Default retrieval returns all nodes
- Optional embedding-based retrieval
- Multiple response synthesis modes
- Document insert, delete, and refresh

**Run:**
```bash
cd summary_index && go run main.go
```

## Key Concepts

### Index Types

| Index Type | Description | Best For |
|------------|-------------|----------|
| **VectorStoreIndex** | Stores embeddings for semantic search | Similarity-based retrieval |
| **SummaryIndex** | Stores all nodes in a list | Comprehensive summarization |
| **KnowledgeGraphIndex** | Extracts and stores triplets | Entity relationships |
| **KeywordTableIndex** | Keyword-based indexing | Exact keyword matching |
| **TreeIndex** | Hierarchical tree structure | Hierarchical summarization |

### Knowledge Graph Index

```go
// Create KG index with LLM extraction
kgIndex, err := index.NewKnowledgeGraphIndexFromDocuments(
    ctx,
    documents,
    index.WithKGIndexLLM(llm),
    index.WithKGIndexMaxTripletsPerChunk(10),
)

// Query the graph
queryEngine := kgIndex.AsQueryEngine()
response, _ := queryEngine.Query(ctx, "What did Einstein develop?")

// Access graph store directly
graphStore := kgIndex.GraphStore()
relations, _ := graphStore.Get(ctx, "Einstein")
```

### Summary Index

```go
// Create summary index
summaryIndex, err := index.NewSummaryIndexFromDocuments(
    ctx,
    documents,
    index.WithSummaryIndexEmbedModel(embedModel),
)

// Query with tree summarization
queryEngine := summaryIndex.AsQueryEngine(
    index.WithQueryEngineLLM(llm),
    index.WithResponseMode(synthesizer.ResponseModeTreeSummarize),
)
response, _ := queryEngine.Query(ctx, "Summarize the documents")

// Get all nodes
nodes, _ := summaryIndex.GetNodes(ctx)
```

### Triplet Structure

```go
type Triplet struct {
    Subject  string  // Entity (e.g., "Einstein")
    Relation string  // Relationship (e.g., "developed")
    Object   string  // Target (e.g., "Theory of Relativity")
}

// Example triplets:
// (Einstein, was born in, Germany)
// (Einstein, developed, Theory of Relativity)
// (Marie Curie, discovered, Radium)
```

### Response Modes

| Mode | Description |
|------|-------------|
| `Compact` | Combines nodes and generates single response |
| `Refine` | Iteratively refines answer through each node |
| `TreeSummarize` | Recursively summarizes nodes |
| `Simple` | Direct response from context |

## Retriever Modes

### Knowledge Graph Retriever

```go
// Keyword mode (default)
retriever := kgIndex.AsRetriever()

// Embedding mode (requires embeddings)
retriever, _ := kgIndex.AsRetrieverWithMode(
    index.KGRetrieverModeEmbedding,
)

// Hybrid mode
retriever, _ := kgIndex.AsRetrieverWithMode(
    index.KGRetrieverModeHybrid,
)
```

### Summary Index Retriever

```go
// Default mode (returns all nodes)
retriever := summaryIndex.AsRetriever()

// With top-k limit
retriever := summaryIndex.AsRetriever(
    index.WithSimilarityTopK(5),
)
```

## Document Operations

```go
// Insert nodes
err := idx.InsertNodes(ctx, nodes)

// Delete nodes
err := idx.DeleteNodes(ctx, nodeIDs)

// Refresh documents (update existing, add new)
refreshed, err := idx.RefreshDocuments(ctx, documents)
```

## Environment Variables

All examples require:
- `OPENAI_API_KEY` - OpenAI API key for LLM and embedding operations
