# Extractor Examples

This directory contains examples demonstrating metadata extraction in `go-llamaindex`.

## Examples

### 1. Metadata Extraction (`metadata_extraction/`)

Demonstrates extracting metadata from documents using LLM-based extractors.

**Features:**
- TitleExtractor for document titles
- KeywordsExtractor for key terms
- SummaryExtractor with context (self, prev, next)
- ExtractorChain for combining extractors
- ProcessNodes for in-place updates
- Custom extraction templates
- Concurrent extraction

**Run:**
```bash
cd metadata_extraction && go run main.go
```

### 2. Entity Extraction (`entity_extraction/`)

Demonstrates named entity recognition from climate-related documents.

**Features:**
- Custom EntityExtractor implementation
- Multiple entity types (person, org, location, date, metric)
- Domain-specific extraction templates
- Climate/environmental focus
- Combined extraction chains
- Metadata enrichment

**Run:**
```bash
cd entity_extraction && go run main.go
```

## Key Concepts

### MetadataExtractor Interface

```go
type MetadataExtractor interface {
    Extract(ctx context.Context, nodes []*schema.Node) ([]ExtractedMetadata, error)
    ProcessNodes(ctx context.Context, nodes []*schema.Node) ([]*schema.Node, error)
    Name() string
}
```

### Built-in Extractors

| Extractor | Output Field | Description |
|-----------|--------------|-------------|
| `TitleExtractor` | `document_title` | Extracts document titles |
| `KeywordsExtractor` | `excerpt_keywords` | Extracts keywords/entities |
| `SummaryExtractor` | `section_summary`, `prev_section_summary`, `next_section_summary` | Generates summaries |
| `QuestionsAnsweredExtractor` | `questions_this_excerpt_can_answer` | Generates answerable questions |

### TitleExtractor

```go
extractor := extractors.NewTitleExtractor(
    extractors.WithTitleLLM(llm),
    extractors.WithTitleNodes(5),  // nodes to use for title
)

metadata, err := extractor.Extract(ctx, nodes)
// metadata[i]["document_title"] = "Extracted Title"
```

### KeywordsExtractor

```go
extractor := extractors.NewKeywordsExtractor(
    extractors.WithKeywordsLLM(llm),
    extractors.WithKeywordsCount(5),
    extractors.WithKeywordsPromptTemplate(customTemplate),
)

metadata, err := extractor.Extract(ctx, nodes)
// metadata[i]["excerpt_keywords"] = "keyword1, keyword2, ..."

// Parse keywords
keywords := extractors.ParseKeywords(metadata[0]["excerpt_keywords"].(string))
```

### SummaryExtractor

```go
extractor := extractors.NewSummaryExtractor(
    extractors.WithSummaryLLM(llm),
    extractors.WithSummaryTypes(
        extractors.SummaryTypeSelf,  // current node
        extractors.SummaryTypePrev,  // previous node
        extractors.SummaryTypeNext,  // next node
    ),
)

metadata, err := extractor.Extract(ctx, nodes)
// metadata[i]["section_summary"] = "Summary of current node"
// metadata[i]["prev_section_summary"] = "Summary of previous node"
// metadata[i]["next_section_summary"] = "Summary of next node"
```

### ExtractorChain

```go
chain := extractors.NewExtractorChain(
    extractors.NewTitleExtractor(extractors.WithTitleLLM(llm)),
    extractors.NewKeywordsExtractor(extractors.WithKeywordsLLM(llm)),
    extractors.NewSummaryExtractor(extractors.WithSummaryLLM(llm)),
)

// Extract all metadata at once
metadata, err := chain.Extract(ctx, nodes)

// Or process nodes in place
enrichedNodes, err := chain.ProcessNodes(ctx, nodes)
```

### Custom Extractor

```go
type MyExtractor struct {
    *extractors.LLMExtractor
}

func NewMyExtractor(llm llm.LLM) *MyExtractor {
    return &MyExtractor{
        LLMExtractor: extractors.NewLLMExtractor(
            []extractors.BaseExtractorOption{
                extractors.WithExtractorName("MyExtractor"),
                extractors.WithNumWorkers(4),
            },
            extractors.WithLLM(llm),
        ),
    }
}

func (e *MyExtractor) Extract(ctx context.Context, nodes []*schema.Node) ([]extractors.ExtractedMetadata, error) {
    // Custom extraction logic
}
```

### Configuration Options

```go
// Base options
extractors.WithExtractorName("name")
extractors.WithTextNodeOnly(true)
extractors.WithMetadataMode(extractors.MetadataModeAll)
extractors.WithInPlace(true)
extractors.WithNumWorkers(4)
extractors.WithShowProgress(true)
```

## Metadata Modes

| Mode | Description |
|------|-------------|
| `MetadataModeAll` | Include all metadata |
| `MetadataModeNone` | Exclude all metadata |
| `MetadataModeEmbed` | Include only embed metadata |
| `MetadataModeLLM` | Include only LLM metadata |

## Environment Variables

All examples require:
- `OPENAI_API_KEY` - OpenAI API key for LLM operations
