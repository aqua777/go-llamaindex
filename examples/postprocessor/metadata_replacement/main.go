// Package main demonstrates metadata replacement postprocessing.
// This example corresponds to Python's node_postprocessor/MetadataReplacementDemo.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/postprocessor"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Metadata Replacement Demo ===")
	fmt.Println("\nReplaces node content with a value from metadata.")
	fmt.Println("Useful for sentence window retrieval and context expansion.")

	separator := strings.Repeat("=", 60)

	// 1. Create nodes with metadata containing window context
	fmt.Println("\n" + separator)
	fmt.Println("=== Sentence Window Retrieval ===")
	fmt.Println(separator)

	fmt.Println("\nScenario: Retrieved sentences with surrounding context in metadata")

	windowNodes := createWindowNodes()

	fmt.Println("\nOriginal nodes (individual sentences):")
	for i, node := range windowNodes {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	fmt.Println("\nMetadata 'window' contains surrounding sentences:")
	for i, node := range windowNodes {
		if window, ok := node.Node.Metadata["window"].(string); ok {
			fmt.Printf("  %d. %s\n", i+1, truncate(window, 60))
		}
	}

	// 2. Apply metadata replacement
	fmt.Println("\n" + separator)
	fmt.Println("=== Applying Metadata Replacement ===")
	fmt.Println(separator)

	windowReplacer := postprocessor.NewMetadataReplacementPostprocessor("window")

	fmt.Printf("\nTarget metadata key: '%s'\n", windowReplacer.TargetMetadataKey())
	fmt.Println("Replacing node content with window context...")

	replacedNodes, err := windowReplacer.PostprocessNodes(ctx, windowNodes, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("\nAfter replacement (expanded context):")
	for i, node := range replacedNodes {
		fmt.Printf("  %d. %s\n", i+1, truncate(node.Node.Text, 70))
	}

	// 3. Different metadata keys
	fmt.Println("\n" + separator)
	fmt.Println("=== Different Metadata Keys ===")
	fmt.Println(separator)

	multiMetaNodes := createMultiMetadataNodes()

	fmt.Println("\nNodes with multiple metadata fields:")
	for i, node := range multiMetaNodes {
		fmt.Printf("  %d. Text: %s\n", i+1, node.Node.Text)
		fmt.Printf("     - summary: %v\n", node.Node.Metadata["summary"])
		fmt.Printf("     - full_text: %v\n", truncate(fmt.Sprintf("%v", node.Node.Metadata["full_text"]), 40))
	}

	// Replace with summary
	summaryReplacer := postprocessor.NewMetadataReplacementPostprocessor("summary")
	summaryNodes, _ := summaryReplacer.PostprocessNodes(ctx, multiMetaNodes, nil)

	fmt.Println("\nAfter replacing with 'summary':")
	for i, node := range summaryNodes {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// Replace with full_text
	fullTextReplacer := postprocessor.NewMetadataReplacementPostprocessor("full_text")
	fullTextNodes, _ := fullTextReplacer.PostprocessNodes(ctx, multiMetaNodes, nil)

	fmt.Println("\nAfter replacing with 'full_text':")
	for i, node := range fullTextNodes {
		fmt.Printf("  %d. %s\n", i+1, truncate(node.Node.Text, 60))
	}

	// 4. Handling missing metadata
	fmt.Println("\n" + separator)
	fmt.Println("=== Handling Missing Metadata ===")
	fmt.Println(separator)

	mixedNodes := createMixedNodes()

	fmt.Println("\nNodes with mixed metadata availability:")
	for i, node := range mixedNodes {
		hasKey := "no"
		if _, ok := node.Node.Metadata["expanded"]; ok {
			hasKey = "yes"
		}
		fmt.Printf("  %d. Has 'expanded': %s - %s\n", i+1, hasKey, node.Node.Text)
	}

	expandedReplacer := postprocessor.NewMetadataReplacementPostprocessor("expanded")
	mixedResult, _ := expandedReplacer.PostprocessNodes(ctx, mixedNodes, nil)

	fmt.Println("\nAfter replacement (missing keys keep original):")
	for i, node := range mixedResult {
		fmt.Printf("  %d. %s\n", i+1, node.Node.Text)
	}

	// 5. Use case: Hierarchical document structure
	fmt.Println("\n" + separator)
	fmt.Println("=== Hierarchical Document Structure ===")
	fmt.Println(separator)

	fmt.Println("\nScenario: Paragraphs with section context")

	hierarchicalNodes := createHierarchicalNodes()

	fmt.Println("\nOriginal (paragraph level):")
	for i, node := range hierarchicalNodes {
		fmt.Printf("  %d. %s\n", i+1, truncate(node.Node.Text, 50))
	}

	sectionReplacer := postprocessor.NewMetadataReplacementPostprocessor("section_context")
	sectionNodes, _ := sectionReplacer.PostprocessNodes(ctx, hierarchicalNodes, nil)

	fmt.Println("\nWith section context:")
	for i, node := range sectionNodes {
		fmt.Printf("  %d. %s\n", i+1, truncate(node.Node.Text, 70))
	}

	// 6. Combining with other postprocessors
	fmt.Println("\n" + separator)
	fmt.Println("=== Postprocessor Chain ===")
	fmt.Println(separator)

	chain := postprocessor.NewPostprocessorChain(
		postprocessor.NewMetadataReplacementPostprocessor("window"),
		postprocessor.NewLongContextReorder(),
	)

	fmt.Println("\nChain: MetadataReplacement -> LongContextReorder")

	chainedResult, _ := chain.PostprocessNodes(ctx, windowNodes, nil)

	fmt.Println("\nResult after chain:")
	for i, node := range chainedResult {
		fmt.Printf("  %d. (score: %.2f) %s\n", i+1, node.Score, truncate(node.Node.Text, 50))
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nMetadata Replacement Use Cases:")
	fmt.Println("  1. Sentence Window Retrieval:")
	fmt.Println("     - Retrieve by sentence, expand to surrounding context")
	fmt.Println()
	fmt.Println("  2. Hierarchical Documents:")
	fmt.Println("     - Retrieve paragraphs, expand to section context")
	fmt.Println()
	fmt.Println("  3. Summary Expansion:")
	fmt.Println("     - Retrieve by summary, replace with full text")
	fmt.Println()
	fmt.Println("  4. Multi-level Indexing:")
	fmt.Println("     - Index at fine granularity, retrieve at coarse granularity")

	fmt.Println("\n=== Metadata Replacement Demo Complete ===")
}

// createWindowNodes creates nodes with sentence window metadata.
func createWindowNodes() []schema.NodeWithScore {
	data := []struct {
		sentence string
		window   string
		score    float64
	}{
		{
			sentence: "Machine learning enables computers to learn from data.",
			window:   "Artificial intelligence is transforming technology. Machine learning enables computers to learn from data. This has applications in many fields.",
			score:    0.92,
		},
		{
			sentence: "Neural networks are inspired by the human brain.",
			window:   "Deep learning uses layered architectures. Neural networks are inspired by the human brain. They consist of interconnected nodes.",
			score:    0.85,
		},
		{
			sentence: "Natural language processing handles text data.",
			window:   "NLP is a key AI application. Natural language processing handles text data. It enables chatbots and translation.",
			score:    0.78,
		},
	}

	nodes := make([]schema.NodeWithScore, len(data))
	for i, d := range data {
		node := schema.NewTextNode(d.sentence)
		node.ID = fmt.Sprintf("sent_%d", i+1)
		node.Metadata["window"] = d.window
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: d.score,
		}
	}

	return nodes
}

// createMultiMetadataNodes creates nodes with multiple metadata fields.
func createMultiMetadataNodes() []schema.NodeWithScore {
	data := []struct {
		text     string
		summary  string
		fullText string
	}{
		{
			text:     "Introduction paragraph",
			summary:  "Overview of the topic",
			fullText: "This document provides a comprehensive introduction to the topic. It covers fundamental concepts and practical applications.",
		},
		{
			text:     "Methods section",
			summary:  "Research methodology",
			fullText: "The methodology section describes the experimental approach, data collection procedures, and analysis techniques used in this study.",
		},
	}

	nodes := make([]schema.NodeWithScore, len(data))
	for i, d := range data {
		node := schema.NewTextNode(d.text)
		node.ID = fmt.Sprintf("doc_%d", i+1)
		node.Metadata["summary"] = d.summary
		node.Metadata["full_text"] = d.fullText
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: 0.8,
		}
	}

	return nodes
}

// createMixedNodes creates nodes with varying metadata availability.
func createMixedNodes() []schema.NodeWithScore {
	nodes := make([]schema.NodeWithScore, 3)

	// Node with expanded metadata
	node1 := schema.NewTextNode("Short text")
	node1.ID = "node_1"
	node1.Metadata["expanded"] = "This is the expanded version of the short text with more context."
	nodes[0] = schema.NodeWithScore{Node: *node1, Score: 0.9}

	// Node without expanded metadata
	node2 := schema.NewTextNode("Another short text")
	node2.ID = "node_2"
	// No "expanded" key
	nodes[1] = schema.NodeWithScore{Node: *node2, Score: 0.85}

	// Node with expanded metadata
	node3 := schema.NewTextNode("Brief content")
	node3.ID = "node_3"
	node3.Metadata["expanded"] = "Brief content is now expanded with additional information and context."
	nodes[2] = schema.NodeWithScore{Node: *node3, Score: 0.8}

	return nodes
}

// createHierarchicalNodes creates nodes with hierarchical context.
func createHierarchicalNodes() []schema.NodeWithScore {
	data := []struct {
		paragraph      string
		sectionContext string
	}{
		{
			paragraph:      "The algorithm processes input data efficiently.",
			sectionContext: "## Algorithm Design\n\nThe algorithm processes input data efficiently. It uses a divide-and-conquer approach for optimal performance.",
		},
		{
			paragraph:      "Results show significant improvement.",
			sectionContext: "## Experimental Results\n\nResults show significant improvement. The new method outperforms baselines by 15%.",
		},
	}

	nodes := make([]schema.NodeWithScore, len(data))
	for i, d := range data {
		node := schema.NewTextNode(d.paragraph)
		node.ID = fmt.Sprintf("para_%d", i+1)
		node.Metadata["section_context"] = d.sectionContext
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: 0.88,
		}
	}

	return nodes
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
