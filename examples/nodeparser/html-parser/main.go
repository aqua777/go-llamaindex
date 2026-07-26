package main

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/nodeparser"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	sample := `<!DOCTYPE html>
<html><head><title>Demo</title></head>
<body>
<section><h1>Overview</h1><p>First paragraph.</p></section>
<p>Second paragraph with <b>bold</b> and <i>italic</i>.</p>
<ul><li>Item one</li><li>Item two</li></ul>
</body></html>`

	parser := nodeparser.NewHTMLNodeParser()
	docs := []schema.Document{{ID: "sample-html", Text: sample}}
	nodes := parser.GetNodesFromDocuments(docs)

	fmt.Printf("Extracted %d node(s):\n", len(nodes))
	for i, n := range nodes {
		tag, _ := n.Metadata[nodeparser.MetadataKeyHTMLTag].(string)
		fmt.Printf("\n--- Node %d ---\n", i+1)
		fmt.Printf("html_tag: %s\n", tag)
		fmt.Printf("text: %q\n", n.Text)
		if len(n.Metadata) > 0 {
			fmt.Printf("metadata keys: ")
			first := true
			for k := range n.Metadata {
				if !first {
					fmt.Print(", ")
				}
				first = false
				fmt.Print(k)
			}
			fmt.Println()
		}
	}
}
