// Package main demonstrates long context reordering for optimal LLM performance.
// This example corresponds to Python's node_postprocessor/LongContextReorder.ipynb
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

	fmt.Println("=== Long Context Reorder Demo ===")
	fmt.Println("\nBased on research showing LLMs struggle with information in the middle of long contexts.")
	fmt.Println("Paper: https://arxiv.org/abs/2307.03172")

	separator := strings.Repeat("=", 60)

	// 1. Create sample nodes with varying relevance scores
	nodes := createSampleNodes()

	fmt.Printf("\nCreated %d sample nodes\n", len(nodes))
	fmt.Println("\nOriginal order (by retrieval score, highest first):")
	for i, node := range nodes {
		fmt.Printf("  %d. (score: %.2f) %s\n", i+1, node.Score, truncate(node.Node.Text, 40))
	}

	// 2. Apply long context reordering
	fmt.Println("\n" + separator)
	fmt.Println("=== Long Context Reordering ===")
	fmt.Println(separator)

	reorderer := postprocessor.NewLongContextReorder()

	fmt.Println("\nReordering to place most relevant documents at start and end...")
	fmt.Println("(LLMs perform better when key info is at context boundaries)")

	reorderedNodes, err := reorderer.PostprocessNodes(ctx, nodes, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("\nReordered nodes:")
	for i, node := range reorderedNodes {
		position := "middle"
		if i == 0 {
			position = "START"
		} else if i == len(reorderedNodes)-1 {
			position = "END"
		}
		fmt.Printf("  %d. [%s] (score: %.2f) %s\n", i+1, position, node.Score, truncate(node.Node.Text, 35))
	}

	// 3. Visualize the reordering pattern
	fmt.Println("\n" + separator)
	fmt.Println("=== Reordering Pattern Visualization ===")
	fmt.Println(separator)

	fmt.Println("\nOriginal order by score:")
	printScoreBar(nodes)

	fmt.Println("\nAfter long context reorder:")
	printScoreBar(reorderedNodes)

	fmt.Println("\nPattern: High-scoring nodes are placed at the START and END")
	fmt.Println("         Lower-scoring nodes are placed in the MIDDLE")

	// 4. Demonstrate with different input sizes
	fmt.Println("\n" + separator)
	fmt.Println("=== Different Input Sizes ===")
	fmt.Println(separator)

	sizes := []int{3, 5, 7, 10}

	for _, size := range sizes {
		testNodes := createNodesWithSize(size)
		reordered, _ := reorderer.PostprocessNodes(ctx, testNodes, nil)

		fmt.Printf("\n%d nodes: ", size)
		for i, n := range reordered {
			if i > 0 {
				fmt.Print(" -> ")
			}
			fmt.Printf("%.1f", n.Score)
		}
		fmt.Println()
	}

	// 5. Combining with other postprocessors
	fmt.Println("\n" + separator)
	fmt.Println("=== Combining with Other Postprocessors ===")
	fmt.Println(separator)

	// Create a chain: Top-K filter -> Long Context Reorder
	topKFilter := postprocessor.NewTopKPostprocessor(5)

	chain := postprocessor.NewPostprocessorChain(
		topKFilter,
		reorderer,
	)

	fmt.Println("\nPostprocessor chain: TopK(5) -> LongContextReorder")
	fmt.Printf("Input: %d nodes\n", len(nodes))

	chainedNodes, err := chain.PostprocessNodes(ctx, nodes, nil)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Printf("Output: %d nodes\n", len(chainedNodes))
		fmt.Println("\nChained result:")
		for i, node := range chainedNodes {
			fmt.Printf("  %d. (score: %.2f) %s\n", i+1, node.Score, truncate(node.Node.Text, 40))
		}
	}

	// 6. When to use long context reorder
	fmt.Println("\n" + separator)
	fmt.Println("=== When to Use Long Context Reorder ===")
	fmt.Println(separator)

	fmt.Println("\nUse Cases:")
	fmt.Println("  ✓ When passing many retrieved documents to an LLM")
	fmt.Println("  ✓ When using models with long context windows (8K+)")
	fmt.Println("  ✓ When document order affects answer quality")
	fmt.Println()
	fmt.Println("Not Needed When:")
	fmt.Println("  ✗ Only a few documents (< 5)")
	fmt.Println("  ✗ Using models with attention mechanisms that handle middle context well")
	fmt.Println("  ✗ Documents are already ordered by importance")
	fmt.Println()
	fmt.Println("Best Practices:")
	fmt.Println("  1. Apply after relevance-based filtering (TopK, similarity threshold)")
	fmt.Println("  2. Use before passing context to LLM for synthesis")
	fmt.Println("  3. Combine with other postprocessors in a chain")

	fmt.Println("\n=== Long Context Reorder Demo Complete ===")
}

// createSampleNodes creates sample nodes with decreasing scores.
func createSampleNodes() []schema.NodeWithScore {
	texts := []struct {
		text  string
		score float64
	}{
		{"Document about machine learning fundamentals and neural networks.", 0.95},
		{"Deep learning architectures including CNNs and transformers.", 0.88},
		{"Natural language processing techniques and applications.", 0.82},
		{"Computer vision and image recognition methods.", 0.75},
		{"Reinforcement learning and decision making algorithms.", 0.68},
		{"Data preprocessing and feature engineering best practices.", 0.62},
		{"Model evaluation metrics and validation strategies.", 0.55},
	}

	nodes := make([]schema.NodeWithScore, len(texts))
	for i, t := range texts {
		node := schema.NewTextNode(t.text)
		node.ID = fmt.Sprintf("doc_%d", i+1)
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: t.score,
		}
	}

	return nodes
}

// createNodesWithSize creates nodes with sequential scores.
func createNodesWithSize(size int) []schema.NodeWithScore {
	nodes := make([]schema.NodeWithScore, size)
	for i := 0; i < size; i++ {
		node := schema.NewTextNode(fmt.Sprintf("Document %d", i+1))
		node.ID = fmt.Sprintf("doc_%d", i+1)
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: float64(size-i) / float64(size), // Decreasing scores
		}
	}
	return nodes
}

// printScoreBar prints a visual representation of scores.
func printScoreBar(nodes []schema.NodeWithScore) {
	fmt.Print("  [")
	for i, node := range nodes {
		if i > 0 {
			fmt.Print("|")
		}
		bars := int(node.Score * 5)
		fmt.Print(strings.Repeat("█", bars))
		fmt.Print(strings.Repeat("░", 5-bars))
	}
	fmt.Println("]")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// TopKPostprocessor is a simple top-k filter.
type TopKPostprocessor struct {
	k int
}

// NewTopKPostprocessor creates a new TopKPostprocessor.
func NewTopKPostprocessor(k int) *TopKPostprocessor {
	return &TopKPostprocessor{k: k}
}

// PostprocessNodes returns the top k nodes.
func (p *TopKPostprocessor) PostprocessNodes(
	ctx context.Context,
	nodes []schema.NodeWithScore,
	queryBundle *schema.QueryBundle,
) ([]schema.NodeWithScore, error) {
	if len(nodes) <= p.k {
		return nodes, nil
	}
	return nodes[:p.k], nil
}

// Name returns the postprocessor name.
func (p *TopKPostprocessor) Name() string {
	return "TopKPostprocessor"
}
