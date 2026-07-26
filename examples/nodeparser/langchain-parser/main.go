package main

import (
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/nodeparser"
	"github.com/aqua777/go-llamaindex/schema"
)

// mockSplitter splits on " | " to simulate a Langchain-style text splitter without external deps.
type mockSplitter struct{}

func (mockSplitter) SplitText(text string) ([]string, error) {
	parts := strings.Split(text, " | ")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	if len(out) == 0 {
		return []string{text}, nil
	}
	return out, nil
}

func main() {
	sample := `First segment | Second segment | Third segment`

	parser := nodeparser.NewLangchainNodeParser(mockSplitter{})
	docs := []schema.Document{{ID: "langchain-demo", Text: sample}}
	nodes := parser.GetNodesFromDocuments(docs)

	fmt.Printf("Extracted %d node(s) via mock Langchain SplitText:\n", len(nodes))
	for i, n := range nodes {
		fmt.Printf("\n--- Node %d ---\n", i+1)
		fmt.Printf("chunk_index: %v / chunk_count: %v\n", n.Metadata["chunk_index"], n.Metadata["chunk_count"])
		fmt.Printf("source_doc_id: %v\n", n.Metadata["source_doc_id"])
		fmt.Printf("text:\n%s\n", n.Text)
	}
}
