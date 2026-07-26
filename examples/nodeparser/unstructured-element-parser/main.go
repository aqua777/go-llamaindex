package main

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/nodeparser"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	sample := `This is the opening paragraph of plain text.

The second paragraph adds more detail without any markup.

A final paragraph closes the sample.`

	parser := nodeparser.NewUnstructuredElementNodeParser()
	docs := []schema.Document{{ID: "sample-plain", Text: sample}}
	nodes := parser.GetNodesFromDocuments(docs)

	fmt.Printf("Extracted %d node(s):\n", len(nodes))
	for i, n := range nodes {
		kind, _ := n.Metadata[nodeparser.MetadataKeyUnstructuredElement].(string)
		fmt.Printf("\n--- Node %d ---\n", i+1)
		fmt.Printf("unstructured_element: %s\n", kind)
		fmt.Printf("text:\n%s\n", n.Text)
	}
}
