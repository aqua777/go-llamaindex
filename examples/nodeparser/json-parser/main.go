package main

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/nodeparser"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	sample := `{
  "title": "JSON parser demo",
  "nested": { "count": 42, "label": "inner" },
  "tags": ["alpha", "beta"],
  "flag": true
}`

	parser := nodeparser.NewJSONNodeParser()
	docs := []schema.Document{{ID: "sample-json", Text: sample}}
	nodes := parser.GetNodesFromDocuments(docs)

	fmt.Printf("Extracted %d node(s):\n", len(nodes))
	for i, n := range nodes {
		path, _ := n.Metadata[nodeparser.MetadataKeyJSONPath].(string)
		fmt.Printf("\n--- Node %d ---\n", i+1)
		fmt.Printf("json_path: %s\n", path)
		fmt.Printf("text: %q\n", n.Text)
	}
}
