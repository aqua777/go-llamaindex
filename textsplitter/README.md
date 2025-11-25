# TextSplitter for Go

A robust, recursive text splitting library for Go, designed to break down large texts into semantically meaningful chunks. This package is a port of the `SentenceSplitter` from LlamaIndex (Python), tailored for the Go ecosystem.

It supports:
- **Recursive Splitting**: Paragraphs -> Sentences -> Words -> Characters.
- **Greedy Merging**: Combines small chunks up to `ChunkSize` with precise `ChunkOverlap`.
- **Pluggable Components**: Swap out tokenizers (e.g., simple whitespace, OpenAI's TikToken) and sentence splitting strategies (e.g., Regex, Neurosnap).

## Installation

```bash
go get github.com/aqua777/go-llamaindex/textsplitter
```

## Usage

### Basic Usage

The default configuration uses a simple whitespace tokenizer and a regex-based sentence splitter.

```go
package main

import (
	"fmt"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	text := "Hello world. This is a test text that we want to split into chunks."

	// Create a splitter with:
	// - ChunkSize: 20 tokens (approx)
	// - ChunkOverlap: 5 tokens
	// - Tokenizer: nil (defaults to SimpleTokenizer)
	// - Strategy: nil (defaults to RegexSplitterStrategy)
	splitter := textsplitter.NewSentenceSplitter(20, 5, nil, nil)

	chunks := splitter.SplitText(text)

	for i, chunk := range chunks {
		fmt.Printf("Chunk %d: %s\n", i+1, chunk)
	}
}
```

### Advanced Usage: OpenAI TikToken

For LLM applications, you often want to count tokens exactly as the model does. You can use the built-in `TikTokenTokenizer`.

```go
package main

import (
	"fmt"
	"log"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	text := "Your long text here..."

	// Initialize TikToken for GPT-3.5/4
	tokenizer, err := textsplitter.NewTikTokenTokenizer("gpt-3.5-turbo")
	if err != nil {
		log.Fatal(err)
	}

	// Use the custom tokenizer
	splitter := textsplitter.NewSentenceSplitter(1024, 200, tokenizer, nil)

	chunks := splitter.SplitText(text)
	fmt.Printf("Split into %d chunks using TikToken.\n", len(chunks))
}
```

### Advanced Usage: Neurosnap Sentence Splitting

For higher quality sentence segmentation (handling abbreviations, etc.), you can use the `NeurosnapSplitterStrategy`. This package **embeds** the English training data, so it works out of the box with zero configuration.

```go
package main

import (
	"fmt"
	"log"
	"github.com/aqua777/go-llamaindex/textsplitter"
)

func main() {
	// Pass nil to use the embedded English training data.
	strategy, err := textsplitter.NewNeurosnapSplitterStrategy(nil)
	if err != nil {
		log.Fatal("Could not load training data:", err)
	}

	splitter := textsplitter.NewSentenceSplitter(1024, 200, nil, strategy)
	
	text := "Mr. Smith went to Washington. He bought a 5.5 in. display."
	chunks := splitter.SplitText(text)
	
	// Should correctly handle "Mr." and "5.5 in." without splitting
	fmt.Println(chunks)
}
```

## Configuration

### `NewSentenceSplitter`

```go
func NewSentenceSplitter(
    chunkSize int, 
    chunkOverlap int, 
    tokenizer Tokenizer, 
    splitterStrategy SentenceSplitterStrategy,
) *SentenceSplitter
```

- **chunkSize**: Target size of each chunk in tokens.
- **chunkOverlap**: Number of tokens to overlap between chunks to maintain context.
- **tokenizer**: Implementation of `Tokenizer` interface. Defaults to `SimpleTokenizer` (whitespace) if nil.
- **splitterStrategy**: Implementation of `SentenceSplitterStrategy` interface. Defaults to `RegexSplitterStrategy` if nil.

## Interfaces

You can implement your own components by satisfying these interfaces defined in `iface.go`:

```go
// Tokenizer encodes text into a list of string tokens (or proxy tokens for counting).
type Tokenizer interface {
	Encode(text string) []string
}

// SentenceSplitterStrategy defines how to split a large text into primary sentences.
type SentenceSplitterStrategy interface {
	Split(text string) []string
}
```

## License

MIT
