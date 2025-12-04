# CLAUDE.md - `go-llamaindex`

This file provides guidance to AI coding agents when working with the `go-llamaindex` codebase. It documents architectural decisions, design patterns, and conventions derived from the Python and TypeScript LlamaIndex implementations.

## Development Commands

```bash
# Run all tests
go test ./...

# Run tests with verbose output
go test -v ./...

# Run specific package tests
go test -v ./rag/...
go test -v ./textsplitter/...

# Run tests with coverage
go test -cover ./...

# Build the module
go build ./...

# Tidy dependencies
go mod tidy

# Vendor dependencies (if using vendoring)
go mod vendor
```

## Architecture Overview

`go-llamaindex` is a Go implementation of core LlamaIndex abstractions for building Retrieval-Augmented Generation (RAG) applications. The architecture follows the same conceptual model as Python and TypeScript versions while embracing Go idioms.

### Core Data Flow

```
Document → TextSplitter → Nodes → EmbeddingModel → VectorStore
                                                        ↓
QueryBundle → Retriever → NodeWithScore[] → Synthesizer → EngineResponse
```

### Package Structure

```
golang/
├── schema/          # Core data structures (Node, Document, QueryBundle, etc.)
├── llm/             # LLM interface and implementations (OpenAI)
├── embedding/       # Embedding interface and implementations (OpenAI)
├── textsplitter/    # Text chunking (SentenceSplitter)
├── settings/        # Global configuration management
├── rag/             # RAG pipeline components
│   ├── store/       # VectorStore interface and implementations
│   │   └── chromem/ # Chromem-go vector store adapter
│   └── reader/      # Document readers (SimpleDirectoryReader)
├── examples/        # Usage examples
└── vendor/          # Vendored dependencies
```

## Key Architectural Concepts

### 1. Interface-Driven Design

All major components are defined as interfaces, enabling easy swapping of implementations. This mirrors the abstract base class pattern in Python/TypeScript.

**Core Interfaces:**

| Package | Interface | Purpose |
|---------|-----------|---------|
| `llm` | `LLM` | Language model operations (Complete, Chat, Stream) |
| `embedding` | `EmbeddingModel` | Text embedding generation |
| `rag/store` | `VectorStore` | Vector storage and similarity search |
| `rag` | `Retriever` | Document retrieval from query |
| `rag` | `Synthesizer` | Response generation from context |
| `rag` | `QueryEngine` | End-to-end query orchestration |
| `textsplitter` | `TextSplitter` | Text chunking |
| `textsplitter` | `Tokenizer` | Token counting |

### 2. Schema Types (Equivalent to Python/TS BaseNode)

The `schema` package defines core data structures that flow through the pipeline:

- **`Node`**: The fundamental unit of data (equivalent to `TextNode` in Python/TS)
  - Contains: ID, Text, Type, Metadata, Embedding
  - Types: `ObjectTypeText`, `ObjectTypeImage`, `ObjectTypeIndex`, `ObjectTypeDocument`

- **`Document`**: Source document before chunking
  - Contains: ID, Text, Metadata

- **`NodeWithScore`**: Node with similarity score from retrieval

- **`QueryBundle`**: Encapsulates query with optional filters
  - Contains: QueryString, Filters (MetadataFilters)

- **`EngineResponse`**: Final response with source nodes
  - Contains: Response string, SourceNodes

### 3. Provider Pattern

LLM and Embedding providers implement common interfaces:

```go
// llm/interface.go
type LLM interface {
    Complete(ctx context.Context, prompt string) (string, error)
    Chat(ctx context.Context, messages []ChatMessage) (string, error)
    Stream(ctx context.Context, prompt string) (<-chan string, error)
}

// embedding/interface.go
type EmbeddingModel interface {
    GetTextEmbedding(ctx context.Context, text string) ([]float64, error)
    GetQueryEmbedding(ctx context.Context, query string) ([]float64, error)
}
```

### 4. Global Settings

The `settings` package provides thread-safe global configuration (similar to `Settings` in Python/TS):

```go
settings.SetLLM(myLLM)
settings.SetEmbedModel(myEmbedding)
settings.SetChunkSize(1024)
settings.SetChunkOverlap(200)
```

### 5. RAG Pipeline Components

**Retriever**: Converts query to relevant nodes
```go
type Retriever interface {
    Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error)
}
```

**Synthesizer**: Generates response from query + context nodes
```go
type Synthesizer interface {
    Synthesize(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.EngineResponse, error)
    SynthesizeStream(ctx context.Context, query schema.QueryBundle, nodes []schema.NodeWithScore) (schema.StreamingEngineResponse, error)
}
```

**QueryEngine**: Orchestrates retrieve → synthesize flow
```go
type QueryEngine interface {
    Query(ctx context.Context, query schema.QueryBundle) (schema.EngineResponse, error)
    QueryStream(ctx context.Context, query schema.QueryBundle) (schema.StreamingEngineResponse, error)
}
```

### 6. VectorStore Interface

```go
type VectorStore interface {
    Add(ctx context.Context, nodes []schema.Node) ([]string, error)
    Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error)
}
```

Implementations:
- `store.SimpleVectorStore`: In-memory store for testing
- `chromem.ChromemStore`: Persistent store using chromem-go

### 7. Metadata Filtering

Supports filtering during retrieval (aligned with Python/TS):

```go
type MetadataFilter struct {
    Key      string
    Value    interface{}
    Operator FilterOperator  // ==, >, <, !=, >=, <=, in, nin
}

type MetadataFilters struct {
    Filters []MetadataFilter
}
```

## Design Patterns & Conventions

### Go-Specific Patterns

1. **Context Propagation**: All operations accept `context.Context` as first parameter for cancellation and timeouts.

2. **Error Handling**: Return errors explicitly; no exceptions. Wrap errors with context using `fmt.Errorf("operation failed: %w", err)`.

3. **Functional Options**: Use for complex constructors (not yet fully adopted, but recommended for new code):
   ```go
   func NewSentenceSplitter(chunkSize, chunkOverlap int, opts ...Option) *SentenceSplitter
   ```

4. **Interface Segregation**: Keep interfaces small and focused. Prefer multiple small interfaces over one large interface.

5. **Dependency Injection**: Pass dependencies via constructors, not global state (except for `settings` package defaults).

### Naming Conventions

| Concept | Python | TypeScript | Go |
|---------|--------|------------|-----|
| Base node | `BaseNode` | `BaseNode` | `Node` |
| Text node | `TextNode` | `TextNode` | `Node` (with Type field) |
| Node types | `ObjectType.TEXT` | `ObjectType.TEXT` | `ObjectTypeText` |
| Query input | `QueryBundle` | `QueryBundle` | `QueryBundle` |
| Response | `Response` | `EngineResponse` | `EngineResponse` |
| Vector store | `VectorStore` | `BaseVectorStore` | `VectorStore` |

### File Organization

- One interface per file when the interface is the primary export (e.g., `interface.go`)
- Implementation files named after the provider (e.g., `openai.go`)
- Test files use `_test.go` suffix
- Mock implementations in `mock.go` files

## Alignment with Python/TypeScript Implementations

### Concepts to Maintain

1. **Node Relationships**: Python/TS support `SOURCE`, `PREVIOUS`, `NEXT`, `PARENT`, `CHILD` relationships. Consider adding to Go `Node` struct.

2. **MetadataMode**: Python/TS have `ALL`, `EMBED`, `LLM`, `NONE` modes for metadata inclusion. Consider adding.

3. **TransformComponent**: Python/TS have composable transform pipelines. The `TextSplitter` interface is a start.

4. **Ingestion Pipeline**: Python/TS have `IngestionPipeline` for document processing. `RAGSystem` partially implements this.

5. **Index Abstractions**: Python/TS have `VectorStoreIndex`, `SummaryIndex`, etc. Currently only vector retrieval is implemented.

### Features to Consider Adding

- **Chat Engine**: Conversational interface with memory
- **Response Synthesizers**: Multiple strategies (refine, tree_summarize, compact)
- **Node Parsers**: Beyond sentence splitting (semantic, hierarchical)
- **Postprocessors**: Reranking, filtering, metadata replacement
- **Callbacks/Instrumentation**: Event hooks for observability
- **Async Operations**: Go's goroutines for concurrent embedding/retrieval

## Testing

### Test Setup Convention

**Always use `testing` + `stretchr/testify/suite` for unit tests:**

```go
package mypackage

import (
    "testing"
    
    "github.com/stretchr/testify/suite"
)

type MyComponentTestSuite struct {
    suite.Suite
    // Add shared test fixtures here
}

func TestMyComponentTestSuite(t *testing.T) {
    suite.Run(t, new(MyComponentTestSuite))
}

func (s *MyComponentTestSuite) SetupTest() {
    // Setup before each test
}

func (s *MyComponentTestSuite) TearDownTest() {
    // Cleanup after each test
}

func (s *MyComponentTestSuite) TestFeature_Scenario() {
    // Arrange
    input := "test input"
    
    // Act
    result := MyFunction(input)
    
    // Assert
    s.NoError(err)
    s.Equal(expected, result)
    s.Len(items, 3)
    s.Contains(str, "substring")
    s.NotNil(obj)
}
```

### Test File Structure

```go
// mycomponent_test.go

package mypackage

import (
    "context"
    "testing"
    
    "github.com/stretchr/testify/suite"
    
    // Internal imports
    "github.com/aqua777/go-llamaindex/schema"
)

// Mock implementations for testing
type MockDependency struct {
    ReturnValue string
    ReturnError error
}

func (m *MockDependency) DoSomething(ctx context.Context) (string, error) {
    return m.ReturnValue, m.ReturnError
}

// Test suite
type MyComponentTestSuite struct {
    suite.Suite
    ctx context.Context
}

func TestMyComponentTestSuite(t *testing.T) {
    suite.Run(t, new(MyComponentTestSuite))
}

func (s *MyComponentTestSuite) SetupTest() {
    s.ctx = context.Background()
}

// Test methods follow Test<Method>_<Scenario> naming
func (s *MyComponentTestSuite) TestProcess_Success() {
    // Test implementation
}

func (s *MyComponentTestSuite) TestProcess_ErrorHandling() {
    // Test error cases
}
```

### Testify Suite Assertions

Common assertions to use:

```go
s.NoError(err)                    // Assert no error
s.Error(err)                      // Assert error occurred
s.Equal(expected, actual)         // Assert equality
s.NotEqual(a, b)                  // Assert inequality
s.Nil(obj)                        // Assert nil
s.NotNil(obj)                     // Assert not nil
s.True(condition)                 // Assert true
s.False(condition)                // Assert false
s.Len(slice, expectedLen)         // Assert slice/map length
s.Empty(slice)                    // Assert empty
s.NotEmpty(slice)                 // Assert not empty
s.Contains(str, substring)        // Assert contains
s.ElementsMatch(expected, actual) // Assert same elements (any order)
s.Greater(a, b)                   // Assert a > b
s.GreaterOrEqual(a, b)            // Assert a >= b
```

### Mock Pattern

Create mocks that implement interfaces for testing:

```go
// In test file or mock.go
type MockLLM struct {
    Response string
    Err      error
}

func (m *MockLLM) Complete(ctx context.Context, prompt string) (string, error) {
    return m.Response, m.Err
}

func (m *MockLLM) Chat(ctx context.Context, messages []ChatMessage) (string, error) {
    return m.Response, m.Err
}

func (m *MockLLM) Stream(ctx context.Context, prompt string) (<-chan string, error) {
    ch := make(chan string, 1)
    ch <- m.Response
    close(ch)
    return ch, m.Err
}
```

### Integration Test Pattern

For tests requiring external services:

```go
func (s *IntegrationTestSuite) TestWithRealAPI() {
    apiKey := os.Getenv("OPENAI_API_KEY")
    if apiKey == "" {
        s.T().Skip("Skipping integration test: OPENAI_API_KEY not set")
    }
    
    // Test with real API
}
```

## Development Guidelines

### Example Applications

When developing example/demo applications:

1. **Location**: All examples must be placed in `golang/examples/` in subdirectories corresponding to groups of use cases (e.g., `examples/rag/`, `examples/embedding/`, `examples/agent/`).

2. **Structure**: Each example must have a `main()` function defined in `main.go` file and be executable independently from other examples.

3. **Data Files**: If an example requires data files, it should expect to find them in a `.data` directory in the same location as `main.go`. The `.data` directory is excluded from Git (via `.gitignore`).

4. **Documentation**: Each example must provide a `README.md` file explaining:
   - What the example demonstrates
   - Prerequisites and setup instructions
   - How to run the example
   - Expected output or behavior

### When Adding New Providers

1. Create interface in appropriate package if not exists
2. Implement the interface in a new file (e.g., `anthropic.go`)
3. Add mock implementation for testing
4. Write comprehensive tests using testify/suite
5. Update this CLAUDE.md if new patterns emerge

### When Extending Core Types

1. Consider backward compatibility
2. Use pointer fields for optional data
3. Add JSON tags for serialization
4. Update related interfaces if needed

### When Adding New RAG Components

1. Define interface in `rag/interfaces.go`
2. Implement in separate file
3. Ensure context propagation
4. Support both sync and streaming where applicable
5. Add to `RAGSystem` if it's a common component

## Dependencies

| Package | Purpose |
|---------|---------|
| `github.com/sashabaranov/go-openai` | OpenAI API client |
| `github.com/philippgille/chromem-go` | Vector database |
| `github.com/neurosnap/sentences` | Sentence tokenization |
| `github.com/pkoukk/tiktoken-go` | Token counting |
| `github.com/stretchr/testify` | Testing assertions and suites |
| `github.com/google/uuid` | UUID generation |

## Module Information

- **Module Path**: `github.com/aqua777/go-llamaindex`
- **Go Version**: 1.24+
- **License**: See LICENSE file
