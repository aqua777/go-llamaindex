package main

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/nodeparser"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	sample := `# Report

This is an introductory paragraph.

## Data

| Name  | Value |
|-------|-------|
| alpha | 100   |
| beta  | 200   |

Closing paragraph.
`

	parser := nodeparser.NewMarkdownElementNodeParser()
	docs := []schema.Document{{ID: "sample-md", Text: sample}}
	nodes := parser.GetNodesFromDocuments(docs)

	fmt.Printf("Extracted %d node(s):\n", len(nodes))
	for i, n := range nodes {
		kind, _ := n.Metadata[nodeparser.MetadataKeyMarkdownElement].(string)
		fmt.Printf("\n--- Node %d ---\n", i+1)
		fmt.Printf("markdown_element: %s\n", kind)
		fmt.Printf("text:\n%s\n", n.Text)
	}
}
