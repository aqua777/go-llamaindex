# Postprocessor Examples

This directory contains examples demonstrating node postprocessing capabilities in `go-llamaindex`.

## Examples

### 1. LLM Reranking (`llm_rerank/`)

Demonstrates LLM-based reranking of retrieved nodes.

**Features:**
- LLMRerank: Score-based relevance ranking
- RankGPT: Pairwise comparison reranking
- Custom reranking prompts
- Batch processing

**Run:**
```bash
cd llm_rerank && go run main.go
```

**Requires:** `OPENAI_API_KEY`

### 2. Long Context Reorder (`long_context_reorder/`)

Optimizes document order for long context LLMs.

**Features:**
- Places important documents at context boundaries
- Based on research showing LLMs struggle with middle context
- Combines with other postprocessors

**Run:**
```bash
cd long_context_reorder && go run main.go
```

### 3. Metadata Replacement (`metadata_replacement/`)

Replaces node content with metadata values.

**Features:**
- Sentence window retrieval expansion
- Hierarchical document context
- Multiple metadata key support

**Run:**
```bash
cd metadata_replacement && go run main.go
```

### 4. PII Masking (`pii_masking/`)

Detects and masks personally identifiable information.

**Features:**
- Email, phone, SSN, credit card detection
- Mask or filter modes
- Custom PII patterns
- Preserve original text option

**Run:**
```bash
cd pii_masking && go run main.go
```

### 5. Recency Postprocessor (`recency_postprocessor/`)

Filters and reweights nodes based on timestamps.

**Features:**
- Max age filtering
- Linear/exponential/step decay modes
- Date sorting
- Multiple date format support

**Run:**
```bash
cd recency_postprocessor && go run main.go
```

### 6. Sentence Optimizer (`sentence_optimizer/`)

Optimizes content by selecting relevant sentences.

**Features:**
- Query-based sentence selection
- Similarity threshold filtering
- Context window inclusion
- Text compression and stopword removal

**Run:**
```bash
cd sentence_optimizer && go run main.go
```

**Requires:** `OPENAI_API_KEY`

## Key Concepts

### NodePostprocessor Interface

All postprocessors implement:

```go
type NodePostprocessor interface {
    PostprocessNodes(
        ctx context.Context,
        nodes []schema.NodeWithScore,
        queryBundle *schema.QueryBundle,
    ) ([]schema.NodeWithScore, error)
    Name() string
}
```

### Postprocessor Chain

Combine multiple postprocessors:

```go
chain := postprocessor.NewPostprocessorChain(
    postprocessor.NewLongContextReorder(),
    postprocessor.NewPIIPostprocessor(),
)

result, _ := chain.PostprocessNodes(ctx, nodes, query)
```

### Common Postprocessors

| Postprocessor | Purpose |
|---------------|---------|
| LLMRerank | LLM-based relevance scoring |
| RankGPTRerank | Pairwise comparison ranking |
| LongContextReorder | Optimize for long context |
| MetadataReplacement | Expand content from metadata |
| PIIPostprocessor | Mask/filter PII |
| NodeRecencyPostprocessor | Time-based filtering |
| SentenceOptimizerPostprocessor | Select relevant sentences |
| TextCompressorPostprocessor | Compress/truncate text |

## Usage Patterns

### Reranking Pipeline

```go
// Retrieve -> Rerank -> Use
nodes := retriever.Retrieve(ctx, query)
reranker := postprocessor.NewLLMRerank(
    postprocessor.WithLLMRerankLLM(llm),
    postprocessor.WithLLMRerankTopN(5),
)
reranked, _ := reranker.PostprocessNodes(ctx, nodes, query)
```

### Privacy-Safe Retrieval

```go
piiMasker := postprocessor.NewPIIPostprocessor(
    postprocessor.WithPIIMask(true),
    postprocessor.WithPIITypes(postprocessor.PIITypeEmail, postprocessor.PIITypePhone),
)
safeNodes, _ := piiMasker.PostprocessNodes(ctx, nodes, nil)
```

### Time-Sensitive Retrieval

```go
recency := postprocessor.NewNodeRecencyPostprocessor(
    postprocessor.WithRecencyDateKey("published_date"),
    postprocessor.WithRecencyMaxAge(30 * 24 * time.Hour),
    postprocessor.WithRecencyTimeWeightMode(postprocessor.TimeWeightModeLinear),
)
recentNodes, _ := recency.PostprocessNodes(ctx, nodes, nil)
```

## Environment Variables

- `OPENAI_API_KEY` - Required for LLM reranking and sentence optimization examples
