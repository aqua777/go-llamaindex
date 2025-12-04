# Evaluation Examples

This directory contains examples demonstrating RAG evaluation capabilities in go-llamaindex.

## Examples

### 1. Semantic Similarity (`semantic_similarity/`)

Demonstrates embedding-based semantic similarity evaluation between responses and references.

**Features:**
- Cosine, dot product, and Euclidean similarity modes
- Configurable similarity thresholds
- Direct similarity computation helpers
- Batch semantic evaluation

**Run:**
```bash
cd semantic_similarity && go run main.go
```

### 2. Batch Evaluation (`batch_eval/`)

Shows how to run multiple evaluators in parallel on batches of data.

**Features:**
- Concurrent evaluation with configurable workers
- Multiple evaluators (faithfulness, relevancy, correctness)
- Aggregate statistics (average score, pass rate)
- Result comparison between runs

**Run:**
```bash
cd batch_eval && go run main.go
```

### 3. Question Generation (`question_generation/`)

Demonstrates automatic question generation for RAG evaluation datasets.

**Features:**
- Query decomposition into sub-questions
- Tool/index routing for sub-questions
- Custom prompt templates
- Evaluation dataset generation

**Run:**
```bash
cd question_generation && go run main.go
```

### 4. Retry Query (`retry_query/`)

Shows how to handle transient failures with automatic retries.

**Features:**
- Configurable retry count and delay
- Context cancellation support
- Rate limit handling (429 errors)
- Different retry strategies

**Run:**
```bash
cd retry_query && go run main.go
```

## Related Examples

The basic evaluation example in `examples/rag/evaluation/` covers:
- Faithfulness evaluation
- Relevancy evaluation
- Correctness evaluation
- Context relevancy evaluation

## Key Concepts

### Evaluation Types

| Type | Purpose | Requires |
|------|---------|----------|
| Faithfulness | Check if response is grounded in context | Context, Response |
| Relevancy | Check if response answers the query | Query, Context, Response |
| Correctness | Compare response to reference answer | Query, Response, Reference |
| Semantic Similarity | Embedding-based comparison | Response, Reference, Embedding Model |
| Context Relevancy | Check if retrieved contexts are relevant | Query, Contexts |

### Batch Evaluation

```go
evaluators := map[string]evaluation.Evaluator{
    "faithfulness": faithfulnessEval,
    "relevancy":    relevancyEval,
}

runner := evaluation.NewBatchEvalRunner(evaluators,
    evaluation.WithBatchWorkers(4),
)

result, _ := runner.EvaluateResponseStrs(ctx, queries, responses, contexts, nil)
fmt.Printf("Average score: %.2f\n", result.GetAverageScore("faithfulness"))
```

### Question Generation

```go
generator := questiongen.NewLLMQuestionGenerator(llm)

tools := []selector.ToolMetadata{
    {Name: "docs", Description: "Documentation"},
}

subQuestions, _ := generator.Generate(ctx, tools, "Complex query here")
```

### Retry Query Engine

```go
retryEngine := queryengine.NewRetryQueryEngine(
    baseEngine,
    queryengine.WithMaxRetries(3),
    queryengine.WithRetryDelay(time.Second),
)

response, err := retryEngine.Query(ctx, "query")
```

## Environment Variables

All examples require:
- `OPENAI_API_KEY` - OpenAI API key for LLM and embeddings
