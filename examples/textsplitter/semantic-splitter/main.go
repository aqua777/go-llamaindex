package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/textsplitter"
)

// topicMockEmbedding returns orthogonal vectors so pairwise dissimilarity reflects topic mix
// in each combined-sentence window (LlamaIndex semantic chunking).
type topicMockEmbedding struct{}

func (topicMockEmbedding) GetTextEmbedding(_ context.Context, text string) ([]float32, error) {
	lower := strings.ToLower(text)
	switch {
	case strings.Contains(lower, "weather") || strings.Contains(lower, "rain") || strings.Contains(lower, "forecast"):
		return []float32{1, 0, 0}, nil
	case strings.Contains(lower, "stock") || strings.Contains(lower, "market") || strings.Contains(lower, "invest"):
		return []float32{0, 1, 0}, nil
	default:
		return []float32{0, 0, 1}, nil
	}
}

func (topicMockEmbedding) GetQueryEmbedding(ctx context.Context, text string) ([]float32, error) {
	return topicMockEmbedding{}.GetTextEmbedding(ctx, text)
}

func main() {
	doc := `The weather is rainy today. Forecast shows more rain tomorrow. Stock markets rallied on news. Investors bought tech shares.`
	strategy, err := textsplitter.NewNeurosnapSplitterStrategy(nil)
	if err != nil {
		fmt.Printf("neurosnap: %v\n", err)
		return
	}
	splitter := textsplitter.NewSemanticSplitterNodeParser(
		topicMockEmbedding{},
		1,
		75,
		strategy,
	)
	chunks := splitter.SplitText(doc)
	fmt.Printf("SemanticSplitterNodeParser (mock embeddings): %d chunk(s)\n", len(chunks))
	for i, c := range chunks {
		fmt.Printf("--- Chunk %d (%d runes) ---\n%s\n", i+1, len([]rune(c)), strings.TrimSpace(c))
	}
	fmt.Println("SUCCESS: semantic split completed")
}
