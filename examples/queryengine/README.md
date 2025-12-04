# Query Engine Examples

This directory contains examples demonstrating various query engine patterns in go-llamaindex.

## Examples

### 1. Sub-Question Query Engine (`sub_question/`)

Demonstrates decomposing complex queries into simpler sub-questions, routing each to appropriate query engines, and synthesizing a final response.

**Key Concepts:**
- Question generation with LLM
- Multi-source query routing
- Response aggregation

**Run:**
```bash
cd sub_question && go run main.go
```

### 2. Retriever Router (`retriever_router/`)

Shows how to route queries to different retrievers based on query content using LLM-based selection.

**Key Concepts:**
- Retriever tools with descriptions
- LLM-based selector
- Query routing

**Run:**
```bash
cd retriever_router && go run main.go
```

### 3. Custom Query Engine (`custom_query_engine/`)

Demonstrates implementing a custom query engine by implementing the `QueryEngine` interface.

**Key Concepts:**
- QueryEngine interface
- Custom retrieval logic
- Custom synthesis

**Run:**
```bash
cd custom_query_engine && go run main.go
```

### 4. Custom Retriever (`custom_retriever/`)

Shows how to implement custom retrievers for specialized data sources.

**Key Concepts:**
- Retriever interface
- Custom retrieval strategies
- Integration with query engines

**Run:**
```bash
cd custom_retriever && go run main.go
```

### 5. Knowledge Graph Query Engine (`knowledge_graph/`)

Demonstrates building and querying a knowledge graph index.

**Key Concepts:**
- Triplet extraction
- Graph-based retrieval
- KG query modes (keyword, embedding, hybrid)

**Run:**
```bash
cd knowledge_graph && go run main.go
```

## Prerequisites

- Go 1.21+
- OpenAI API key (set `OPENAI_API_KEY` environment variable)

## Related Python Examples

These examples correspond to the following Python notebooks:
- `query_engine/sub_question_query_engine.ipynb`
- `query_engine/RetrieverRouterQueryEngine.ipynb`
- `query_engine/custom_query_engine.ipynb`
- `query_engine/CustomRetrievers.ipynb`
- `query_engine/knowledge_graph_query_engine.ipynb`
