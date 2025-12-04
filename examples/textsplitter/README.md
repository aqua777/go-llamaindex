# Text Splitter Examples

This directory contains examples demonstrating text splitting strategies in go-llamaindex.

## Examples

### 1. Sentence Splitter (`sentence-splitter/`)

Demonstrates splitting text into chunks using sentence boundaries.

**Key Concepts:**
- Sentence-aware text chunking
- Configurable chunk size and overlap
- Preserving sentence integrity
- Metadata propagation

**Run:**
```bash
cd sentence-splitter && go run main.go
```

## Prerequisites

- Go 1.21+

## Text Splitting Strategies

| Splitter | Description | Best For |
|----------|-------------|----------|
| **SentenceSplitter** | Splits on sentence boundaries | General text, articles |
| **TokenSplitter** | Splits by token count | LLM context management |
| **CodeSplitter** | Splits code by syntax | Source code files |
| **MarkdownSplitter** | Splits by markdown structure | Documentation |

## Key Parameters

### Chunk Size
The target size for each chunk in characters or tokens.
- **Smaller chunks** (256-512): More precise retrieval, more chunks
- **Larger chunks** (1024-2048): More context per chunk, fewer chunks

### Chunk Overlap
The number of characters/tokens to overlap between chunks.
- **Purpose**: Prevents information loss at chunk boundaries
- **Typical values**: 10-20% of chunk size

## Example Usage

```go
import "github.com/aqua777/go-llamaindex/textsplitter"

// Create sentence splitter
splitter := textsplitter.NewSentenceSplitter(
    1024,  // chunk size
    200,   // chunk overlap
    nil,   // tokenizer (nil for default)
    nil,   // sentence splitter (nil for default)
)

// Split text into chunks
chunks := splitter.SplitText(document.Text)
```

## Components Used

- `textsplitter.SentenceSplitter` - Sentence-aware splitting
- `textsplitter.TokenSplitter` - Token-based splitting
- `settings.GetChunkSize()` - Default chunk size from settings
- `settings.GetChunkOverlap()` - Default overlap from settings

## Related Python Examples

These examples correspond to the following Python functionality:
- `llama_index.core.node_parser.SentenceSplitter`
- `llama_index.core.node_parser.TokenTextSplitter`
