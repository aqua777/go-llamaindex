package main

import (
	"fmt"

	"github.com/aqua777/go-llamaindex/nodeparser"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	text := ""
	for i := 0; i < 120; i++ {
		text += fmt.Sprintf("Paragraph %d. Some content to fill the chunk ladder. ", i)
	}

	parser := nodeparser.NewHierarchicalNodeParser()
	docs := []schema.Document{{ID: "long-doc", Text: text}}
	nodes := parser.GetNodesFromDocuments(docs)

	fmt.Printf("Total nodes: %d\n\n", len(nodes))
	for i, n := range nodes {
		lvl, _ := n.Metadata[nodeparser.MetadataKeyHierarchyLevel].(int)
		fmt.Printf("--- Node %d (hierarchy_level=%d) ---\n", i+1, lvl)
		fmt.Printf("id: %s\n", n.ID)
		fmt.Printf("text_len: %d\n", len(n.Text))
		if p := n.Relationships.GetParent(); p != nil {
			fmt.Printf("PARENT -> %s\n", p.NodeID)
		} else {
			fmt.Println("PARENT -> (none)")
		}
		ch := n.Relationships.GetChildren()
		if len(ch) > 0 {
			fmt.Print("CHILD -> ")
			for j, c := range ch {
				if j > 0 {
					fmt.Print(", ")
				}
				fmt.Print(c.NodeID)
			}
			fmt.Println()
		} else {
			fmt.Println("CHILD -> (none)")
		}
		fmt.Println()
	}
}
