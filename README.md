# `go-llamaindex`

**⚠️ EXPERIMENTAL AI-GENERATED CODE ⚠️**

This is an AI-generated conversion of [LlamaIndex](https://github.com/run-llama/llamaindex) concepts to Go. It has not been thoroughly tested, audited, or optimized for production use. Use at your own risk and thoroughly test any functionality before deploying in production environments.

## Overview

`go-llamaindex` is a Go implementation of core LlamaIndex abstractions, providing a framework for building Retrieval-Augmented Generation (RAG) applications. It includes components for document loading, text splitting, embedding generation, vector storage, and query processing.

## Features

### Core Components
- **Schema**: Data structures for nodes, queries, and responses with metadata support
- **LLM Integration**: Interface for Large Language Models (OpenAI GPT models supported)
- **Embedding Models**: Text embedding generation (OpenAI embeddings supported)
- **Vector Stores**: Persistent vector storage with metadata filtering (ChromaDB via chromem-go)
- **Text Splitting**: Sentence-aware text chunking with configurable parameters

### RAG Pipeline
- **Retrievers**: Vector similarity search with metadata filtering
- **Synthesizers**: Context stuffing for LLM prompting
- **Query Engines**: End-to-end RAG orchestration (retrieve → synthesize)
- **Streaming Support**: Real-time token streaming for responses

### Data Ingestion
- **Directory Reader**: Load documents from filesystem directories
- **Metadata Support**: Preserve document metadata through the pipeline

## Installation

```bash
go get github.com/aqua777/go-llamaindex
```

## Quick Start

Here's a basic RAG pipeline example:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/aqua777/go-llamaindex/embedding"
    "github.com/aqua777/go-llamaindex/llm"
    "github.com/aqua777/go-llamaindex/rag"
    "github.com/aqua777/go-llamaindex/rag/reader"
    "github.com/aqua777/go-llamaindex/rag/store/chromem"
    "github.com/aqua777/go-llamaindex/schema"
    "github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
    ctx := context.Background()

    // Initialize components
    embedModel := embedding.NewOpenAIEmbedding("", "text-embedding-ada-002")
    llmModel := llm.NewOpenAILLM("", "gpt-3.5-turbo", "")
    vectorStore, _ := chromem.NewChromemStore("./db", "collection")

    // Load and process documents
    dirReader := reader.NewSimpleDirectoryReader("./data")
    docs, _ := dirReader.LoadData()

    splitter := textsplitter.NewSentenceSplitter(512, 50, nil, nil)

    var nodes []schema.Node
    for _, doc := range docs {
        chunks := splitter.SplitText(doc.Text)
        for i, chunk := range chunks {
            node := schema.Node{
                ID:       fmt.Sprintf("%s-chunk-%d", doc.ID, i),
                Text:     chunk,
                Type:     schema.ObjectTypeText,
                Metadata: doc.Metadata,
            }

            // Generate embedding
            embedding, _ := embedModel.GetTextEmbedding(ctx, chunk)
            node.Embedding = embedding
            nodes = append(nodes, node)
        }
    }

    // Store in vector database
    vectorStore.Add(ctx, nodes)

    // Setup RAG query engine
    retriever := rag.NewVectorRetriever(vectorStore, embedModel, 3)
    synthesizer := rag.NewSimpleSynthesizer(llmModel)
    queryEngine := rag.NewRetrieverQueryEngine(retriever, synthesizer)

    // Query
    response, _ := queryEngine.Query(ctx, schema.QueryBundle{
        QueryString: "What is this document about?",
    })

    fmt.Println(response.Response)
}
```

## Advanced Usage

### Metadata Filtering

Filter retrieved documents by metadata:

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

### Streaming Responses

Get streaming responses for real-time output:

```go
streamResponse, err := queryEngine.QueryStream(ctx, schema.QueryBundle{
    QueryString: "Explain this concept",
})

for token := range streamResponse.ResponseStream {
    fmt.Print(token)
}
```

### Custom LLM Integration

Implement your own LLM:

```go
type MyLLM struct{}

func (m *MyLLM) Complete(ctx context.Context, prompt string) (string, error) {
    // Your LLM implementation
    return "response", nil
}

func (m *MyLLM) Chat(ctx context.Context, messages []llm.ChatMessage) (string, error) {
    // Chat implementation
    return "response", nil
}

func (m *MyLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
    // Streaming implementation
    ch := make(chan string)
    go func() {
        defer close(ch)
        // Send tokens...
    }()
    return ch, nil
}
```

## Architecture

The library follows LlamaIndex's modular architecture:

```
Documents → Nodes → Embeddings → Vector Store
                                    ↓
Query → Retriever → Synthesizer → Response
```

### Key Interfaces

- `llm.LLM`: Language model interface
- `embedding.EmbeddingModel`: Text embedding interface
- `store.VectorStore`: Vector storage interface
- `rag.Retriever`: Document retrieval interface
- `rag.Synthesizer`: Response synthesis interface
- `textsplitter.TextSplitter`: Text chunking interface

## Examples

See the `examples/` directory for complete implementations:

- `examples/rag/basic_rag/`: Basic RAG pipeline with document ingestion
- `examples/rag/metadata_and_streaming/`: Advanced features with metadata filtering and streaming

## Dependencies

- [go-openai](https://github.com/sashabaranov/go-openai): OpenAI API client
- [chromem-go](https://github.com/philippgille/chromem-go): Vector database
- [sentences](https://github.com/neurosnap/sentences): Sentence tokenization
- [tiktoken-go](https://github.com/pkoukk/tiktoken-go): Token counting

## Limitations

As an AI-generated conversion, this implementation may have:

- Incomplete feature coverage compared to LlamaIndex
- Potential bugs or edge cases not yet discovered
- Performance characteristics not optimized
- Limited testing and validation

## Contributing

This is experimental code. Contributions are welcome but expect significant changes as the codebase matures.

## License

See individual dependency licenses.
