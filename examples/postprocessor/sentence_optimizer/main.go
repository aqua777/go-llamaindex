// Package main demonstrates sentence optimization for retrieved content.
// This example corresponds to Python's node_postprocessor/OptimizerDemo.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/postprocessor"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create embedding model
	embedModel := embedding.NewOpenAIEmbedding("", "")
	fmt.Println("=== Sentence Optimizer Demo ===")
	fmt.Println("\nOptimizes node content by selecting most relevant sentences.")

	separator := strings.Repeat("=", 60)

	// 2. Create nodes with long content
	nodes := createLongContentNodes()

	fmt.Println("\nOriginal nodes (long content):")
	for i, node := range nodes {
		fmt.Printf("  %d. (%d chars) %s\n", i+1, len(node.Node.Text), truncate(node.Node.Text, 60))
	}

	// 3. Basic sentence optimization
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Sentence Optimization ===")
	fmt.Println(separator)

	optimizer := postprocessor.NewSentenceOptimizerPostprocessor(
		embedModel,
		postprocessor.WithOptimizerTopK(3), // Keep top 3 sentences
	)

	query := &schema.QueryBundle{
		QueryString: "What are the benefits of machine learning?",
	}

	fmt.Printf("\nQuery: %s\n", query.QueryString)
	fmt.Println("Selecting top 3 most relevant sentences per node...")

	optimizedNodes, err := optimizer.PostprocessNodes(ctx, nodes, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	fmt.Println("\nOptimized nodes:")
	for i, node := range optimizedNodes {
		fmt.Printf("  %d. (%d chars) %s\n", i+1, len(node.Node.Text), truncate(node.Node.Text, 70))
		if optimized, ok := node.Node.Metadata["optimized"]; ok && optimized.(bool) {
			fmt.Println("     [Content was optimized]")
		}
	}

	// 4. With similarity threshold
	fmt.Println("\n" + separator)
	fmt.Println("=== Similarity Threshold ===")
	fmt.Println(separator)

	thresholdOptimizer := postprocessor.NewSentenceOptimizerPostprocessor(
		embedModel,
		postprocessor.WithOptimizerTopK(5),
		postprocessor.WithOptimizerThreshold(0.3), // Only sentences with similarity > 0.3
	)

	fmt.Println("\nOnly keeping sentences with similarity > 0.3...")

	thresholdNodes, _ := thresholdOptimizer.PostprocessNodes(ctx, nodes, query)

	fmt.Println("\nThreshold-filtered results:")
	for i, node := range thresholdNodes {
		fmt.Printf("  %d. (%d chars) %s\n", i+1, len(node.Node.Text), truncate(node.Node.Text, 70))
	}

	// 5. Context window
	fmt.Println("\n" + separator)
	fmt.Println("=== Context Window ===")
	fmt.Println(separator)

	contextOptimizer := postprocessor.NewSentenceOptimizerPostprocessor(
		embedModel,
		postprocessor.WithOptimizerTopK(2),
		postprocessor.WithOptimizerContextWindow(1), // Include 1 sentence before/after
	)

	fmt.Println("\nKeeping top 2 sentences plus 1 surrounding sentence each...")

	contextNodes, _ := contextOptimizer.PostprocessNodes(ctx, nodes, query)

	fmt.Println("\nWith context window:")
	for i, node := range contextNodes {
		fmt.Printf("  %d. (%d chars) %s\n", i+1, len(node.Node.Text), truncate(node.Node.Text, 70))
	}

	// 6. Text compression
	fmt.Println("\n" + separator)
	fmt.Println("=== Text Compression ===")
	fmt.Println(separator)

	compressor := postprocessor.NewTextCompressorPostprocessor(
		postprocessor.WithCompressorMaxLength(200),
	)

	fmt.Println("\nCompressing text to max 200 characters...")

	compressedNodes, _ := compressor.PostprocessNodes(ctx, nodes, nil)

	fmt.Println("\nCompressed results:")
	for i, node := range compressedNodes {
		fmt.Printf("  %d. (%d chars) %s\n", i+1, len(node.Node.Text), node.Node.Text)
	}

	// 7. Stopword removal
	fmt.Println("\n" + separator)
	fmt.Println("=== Stopword Removal ===")
	fmt.Println(separator)

	stopwordCompressor := postprocessor.NewTextCompressorPostprocessor(
		postprocessor.WithCompressorMaxLength(500),
		postprocessor.WithCompressorRemoveStopwords(true),
	)

	fmt.Println("\nRemoving common stopwords...")

	stopwordNodes, _ := stopwordCompressor.PostprocessNodes(ctx, nodes[:1], nil)

	fmt.Println("\nOriginal:")
	fmt.Printf("  %s\n", truncate(nodes[0].Node.Text, 80))
	fmt.Println("\nAfter stopword removal:")
	fmt.Printf("  %s\n", truncate(stopwordNodes[0].Node.Text, 80))

	// 8. Combining optimizers
	fmt.Println("\n" + separator)
	fmt.Println("=== Combining Optimizers ===")
	fmt.Println(separator)

	chain := postprocessor.NewPostprocessorChain(
		postprocessor.NewSentenceOptimizerPostprocessor(
			embedModel,
			postprocessor.WithOptimizerTopK(3),
		),
		postprocessor.NewTextCompressorPostprocessor(
			postprocessor.WithCompressorMaxLength(300),
		),
	)

	fmt.Println("\nChain: SentenceOptimizer(top 3) -> TextCompressor(max 300)")

	chainedNodes, _ := chain.PostprocessNodes(ctx, nodes, query)

	fmt.Println("\nChained result:")
	for i, node := range chainedNodes {
		fmt.Printf("  %d. (%d chars) %s\n", i+1, len(node.Node.Text), truncate(node.Node.Text, 70))
	}

	// 9. Preserve paragraphs
	fmt.Println("\n" + separator)
	fmt.Println("=== Preserve Paragraphs ===")
	fmt.Println(separator)

	paragraphOptimizer := postprocessor.NewSentenceOptimizerPostprocessor(
		embedModel,
		postprocessor.WithOptimizerTopK(3),
		postprocessor.WithOptimizerPreserveParagraphs(true),
	)

	fmt.Println("\nOptimizing while preserving paragraph structure...")

	paragraphNodes, _ := paragraphOptimizer.PostprocessNodes(ctx, nodes, query)

	fmt.Println("\nWith paragraph preservation:")
	for i, node := range paragraphNodes {
		fmt.Printf("  %d. %s\n", i+1, truncate(node.Node.Text, 70))
	}

	// 10. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nSentence Optimizer Features:")
	fmt.Println("  - Select most relevant sentences based on query")
	fmt.Println("  - Configurable top-K sentences per node")
	fmt.Println("  - Similarity threshold filtering")
	fmt.Println("  - Context window for surrounding sentences")
	fmt.Println("  - Paragraph structure preservation")
	fmt.Println()
	fmt.Println("Text Compressor Features:")
	fmt.Println("  - Maximum length truncation")
	fmt.Println("  - Stopword removal")
	fmt.Println("  - Sentence boundary aware truncation")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Reducing context length for LLM calls")
	fmt.Println("  - Focusing on query-relevant content")
	fmt.Println("  - Improving response quality by removing noise")

	fmt.Println("\n=== Sentence Optimizer Demo Complete ===")
}

// createLongContentNodes creates nodes with long text content.
func createLongContentNodes() []schema.NodeWithScore {
	texts := []string{
		`Machine learning is a subset of artificial intelligence. It enables computers to learn from data without explicit programming. The field has grown rapidly in recent years. Deep learning is a popular approach within machine learning. Neural networks form the foundation of deep learning systems. These systems can recognize patterns in large datasets. Machine learning has many practical applications. It powers recommendation systems and voice assistants. The technology continues to evolve and improve.`,

		`Natural language processing deals with text and speech. It allows computers to understand human language. NLP applications include translation and sentiment analysis. Chatbots use NLP to communicate with users. The field combines linguistics and computer science. Recent advances have improved NLP capabilities significantly. Transformer models have revolutionized the field. BERT and GPT are examples of transformer-based models. These models can generate human-like text.`,

		`Computer vision enables machines to interpret images. It has applications in autonomous vehicles and medical imaging. Object detection identifies items within images. Image classification assigns labels to pictures. Facial recognition is a common computer vision application. The technology raises privacy concerns. Deep learning has improved computer vision accuracy. Convolutional neural networks are widely used. Real-time processing is now possible on mobile devices.`,
	}

	nodes := make([]schema.NodeWithScore, len(texts))
	for i, text := range texts {
		node := schema.NewTextNode(text)
		node.ID = fmt.Sprintf("doc_%d", i+1)
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: 0.9 - float64(i)*0.05,
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
