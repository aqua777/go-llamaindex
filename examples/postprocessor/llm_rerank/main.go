// Package main demonstrates LLM-based reranking of retrieved nodes.
// This example corresponds to Python's node_postprocessor/LLMReranker-Gatsby.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/postprocessor"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM for reranking
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== LLM Reranking Demo ===")
	fmt.Println("\nLLM initialized for reranking")

	separator := strings.Repeat("=", 60)

	// 2. Create sample nodes (simulating retrieval results)
	nodes := createSampleNodes()

	fmt.Printf("\nCreated %d sample nodes for reranking\n", len(nodes))
	fmt.Println("\nOriginal order (by initial retrieval score):")
	for i, node := range nodes {
		fmt.Printf("  %d. (score: %.2f) %s\n", i+1, node.Score, truncate(node.Node.Text, 50))
	}

	// 3. Basic LLM Reranking
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic LLM Reranking ===")
	fmt.Println(separator)

	llmReranker := postprocessor.NewLLMRerank(
		postprocessor.WithLLMRerankLLM(llmInstance),
		postprocessor.WithLLMRerankTopN(3),
	)

	query := &schema.QueryBundle{
		QueryString: "What are the main themes in The Great Gatsby?",
	}

	fmt.Printf("\nQuery: %s\n", query.QueryString)
	fmt.Println("\nReranking nodes based on relevance to query...")

	rerankedNodes, err := llmReranker.PostprocessNodes(ctx, nodes, query)
	if err != nil {
		log.Printf("Reranking error: %v", err)
	} else {
		fmt.Printf("\nReranked results (top %d):\n", len(rerankedNodes))
		for i, node := range rerankedNodes {
			fmt.Printf("  %d. (relevance: %.1f) %s\n", i+1, node.Score, truncate(node.Node.Text, 50))
		}
	}

	// 4. RankGPT Reranking
	fmt.Println("\n" + separator)
	fmt.Println("=== RankGPT Reranking ===")
	fmt.Println(separator)

	rankGPTReranker := postprocessor.NewRankGPTRerank(
		postprocessor.WithRankGPTLLM(llmInstance),
		postprocessor.WithRankGPTTopN(3),
		postprocessor.WithRankGPTVerbose(true),
	)

	fmt.Printf("\nQuery: %s\n", query.QueryString)
	fmt.Println("\nUsing RankGPT for pairwise comparison reranking...")

	rankGPTNodes, err := rankGPTReranker.PostprocessNodes(ctx, nodes, query)
	if err != nil {
		log.Printf("RankGPT error: %v", err)
	} else {
		fmt.Printf("\nRankGPT results (top %d):\n", len(rankGPTNodes))
		for i, node := range rankGPTNodes {
			fmt.Printf("  %d. %s\n", i+1, truncate(node.Node.Text, 60))
		}
	}

	// 5. Custom reranking prompt
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Reranking Prompt ===")
	fmt.Println(separator)

	customPrompt := `Given the following documents and a question, identify which documents are most relevant.
Rate each document's relevance from 1-10.

{context_str}

Question: {query_str}

Respond with document numbers and relevance scores:
Doc: <number>, Relevance: <score>
`

	customReranker := postprocessor.NewLLMRerank(
		postprocessor.WithLLMRerankLLM(llmInstance),
		postprocessor.WithLLMRerankTopN(3),
		postprocessor.WithLLMRerankPrompt(customPrompt),
	)

	fmt.Println("\nUsing custom prompt for domain-specific reranking...")

	customNodes, err := customReranker.PostprocessNodes(ctx, nodes, query)
	if err != nil {
		log.Printf("Custom reranking error: %v", err)
	} else {
		fmt.Printf("\nCustom reranked results:\n")
		for i, node := range customNodes {
			fmt.Printf("  %d. (score: %.1f) %s\n", i+1, node.Score, truncate(node.Node.Text, 50))
		}
	}

	// 6. Batch processing
	fmt.Println("\n" + separator)
	fmt.Println("=== Batch Processing ===")
	fmt.Println(separator)

	batchReranker := postprocessor.NewLLMRerank(
		postprocessor.WithLLMRerankLLM(llmInstance),
		postprocessor.WithLLMRerankTopN(5),
		postprocessor.WithLLMRerankBatchSize(3), // Process 3 nodes at a time
	)

	fmt.Printf("\nProcessing %d nodes in batches of 3...\n", len(nodes))

	batchNodes, err := batchReranker.PostprocessNodes(ctx, nodes, query)
	if err != nil {
		log.Printf("Batch reranking error: %v", err)
	} else {
		fmt.Printf("\nBatch reranked results:\n")
		for i, node := range batchNodes {
			fmt.Printf("  %d. (score: %.1f) %s\n", i+1, node.Score, truncate(node.Node.Text, 50))
		}
	}

	// 7. Comparing reranking methods
	fmt.Println("\n" + separator)
	fmt.Println("=== Comparison Summary ===")
	fmt.Println(separator)

	fmt.Println("\nReranking Methods:")
	fmt.Println("  1. LLMRerank:")
	fmt.Println("     - Uses LLM to score document relevance")
	fmt.Println("     - Returns relevance scores (1-10)")
	fmt.Println("     - Good for general-purpose reranking")
	fmt.Println()
	fmt.Println("  2. RankGPT:")
	fmt.Println("     - Uses pairwise comparison approach")
	fmt.Println("     - More robust for relative ordering")
	fmt.Println("     - Better for complex queries")
	fmt.Println()
	fmt.Println("Configuration Options:")
	fmt.Println("  - TopN: Number of results to return")
	fmt.Println("  - BatchSize: Documents processed per LLM call")
	fmt.Println("  - Custom prompts for domain-specific reranking")

	fmt.Println("\n=== LLM Reranking Demo Complete ===")
}

// createSampleNodes creates sample nodes for demonstration.
func createSampleNodes() []schema.NodeWithScore {
	texts := []struct {
		text  string
		score float64
	}{
		{
			text:  "The Great Gatsby is a novel by F. Scott Fitzgerald set in the Jazz Age. It explores themes of wealth, class, and the American Dream.",
			score: 0.85,
		},
		{
			text:  "Jay Gatsby is the mysterious millionaire protagonist who throws lavish parties hoping to reunite with his lost love, Daisy Buchanan.",
			score: 0.82,
		},
		{
			text:  "The novel critiques the moral decay of the 1920s American society and the corruption of the American Dream.",
			score: 0.78,
		},
		{
			text:  "Nick Carraway serves as the narrator, providing an outsider's perspective on the wealthy elite of Long Island.",
			score: 0.75,
		},
		{
			text:  "The green light at the end of Daisy's dock symbolizes Gatsby's hopes and dreams for the future.",
			score: 0.72,
		},
		{
			text:  "F. Scott Fitzgerald was born in 1896 in Minnesota and attended Princeton University.",
			score: 0.65, // Less relevant biographical info
		},
		{
			text:  "The novel was published in 1925 and initially received mixed reviews but is now considered a classic.",
			score: 0.60, // Publication info, less about themes
		},
	}

	nodes := make([]schema.NodeWithScore, len(texts))
	for i, t := range texts {
		node := schema.NewTextNode(t.text)
		node.ID = fmt.Sprintf("node_%d", i+1)
		nodes[i] = schema.NodeWithScore{
			Node:  *node,
			Score: t.score,
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
