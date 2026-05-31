// Package main demonstrates BM25 sparse retrieval using BM25Retriever.
// This example corresponds to Python's retrievers/bm25_retriever.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== BM25 Retriever Demo ===")
	fmt.Println()
	fmt.Println("BM25 (Best Matching 25) is a sparse retrieval algorithm that ranks")
	fmt.Println("documents based on term frequency and inverse document frequency.")
	fmt.Println()

	sep := strings.Repeat("=", 70)

	// 1. Build a node corpus.
	fmt.Println(sep)
	fmt.Println("=== Creating Node Corpus ===")
	fmt.Println(sep)
	fmt.Println()

	nodes := buildCorpus()
	fmt.Printf("Created corpus with %d nodes\n\n", len(nodes))
	for i, n := range nodes {
		fmt.Printf("Node %d: %s\n", i+1, truncate(n.Text, 70))
	}
	fmt.Println()

	// 2. Create BM25Retriever with default settings.
	fmt.Println(sep)
	fmt.Println("=== BM25Retriever (default settings, topK=3) ===")
	fmt.Println(sep)
	fmt.Println()

	r := retriever.NewBM25Retriever(nodes, retriever.WithBM25TopK(3))
	fmt.Println("BM25Retriever created with default BM25 parameters (k1=1.5, b=0.75).")
	fmt.Println()

	queries := []string{
		"machine learning algorithms",
		"natural language processing",
		"database optimization",
	}

	for _, q := range queries {
		fmt.Printf("Query: %q\n", q)
		results, err := r.Retrieve(ctx, schema.QueryBundle{QueryString: q})
		if err != nil {
			fmt.Printf("  Error: %v\n\n", err)
			continue
		}
		printResults(results)
	}

	// 3. BM25Retriever with custom BM25 parameters.
	fmt.Println(sep)
	fmt.Println("=== BM25Retriever with Custom BM25 Parameters ===")
	fmt.Println(sep)
	fmt.Println()

	rHighK1 := retriever.NewBM25Retriever(nodes,
		retriever.WithBM25TopK(3),
		retriever.WithBM25Options(embedding.WithBM25K1(2.0)),
	)
	fmt.Println("k1=2.0 (higher term-frequency weight) — query: \"machine learning\"")
	results, err := rHighK1.Retrieve(ctx, schema.QueryBundle{QueryString: "machine learning"})
	if err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	} else {
		printResults(results)
	}

	rLowB := retriever.NewBM25Retriever(nodes,
		retriever.WithBM25TopK(3),
		retriever.WithBM25Options(embedding.WithBM25B(0.3)),
	)
	fmt.Println("b=0.3 (less document-length normalization) — query: \"machine learning\"")
	results, err = rLowB.Retrieve(ctx, schema.QueryBundle{QueryString: "machine learning"})
	if err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	} else {
		printResults(results)
	}

	// 4. BM25Retriever with metadata filtering.
	fmt.Println(sep)
	fmt.Println("=== BM25Retriever with Metadata Filtering ===")
	fmt.Println(sep)
	fmt.Println()

	filteredR := retriever.NewBM25Retriever(nodes, retriever.WithBM25TopK(3))
	filters := &schema.MetadataFilters{
		Filters: []schema.MetadataFilter{
			{Key: "category", Value: "ml", Operator: schema.FilterOperatorEq},
		},
	}
	fmt.Println("Filter: category == \"ml\"")
	fmt.Printf("Query: %q\n", "learning algorithms")
	results, err = filteredR.Retrieve(ctx, schema.QueryBundle{
		QueryString: "learning algorithms",
		Filters:     filters,
	})
	if err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	} else {
		printResults(results)
	}

	// 5. Demonstrate lexical vs. semantic gap: semantically related but
	//    lexically different nodes score lower.
	fmt.Println(sep)
	fmt.Println("=== Lexical vs. Semantic Gap ===")
	fmt.Println(sep)
	fmt.Println()

	fmt.Println("BM25 is a lexical model: nodes that share no tokens with the query")
	fmt.Println("receive a score of zero regardless of their semantic meaning.")
	fmt.Println()

	lexR := retriever.NewBM25Retriever(nodes, retriever.WithBM25TopK(len(nodes)))
	fmt.Printf("Query: %q\n", "GPU accelerated tensor operations")
	results, err = lexR.Retrieve(ctx, schema.QueryBundle{QueryString: "GPU accelerated tensor operations"})
	if err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	} else {
		fmt.Printf("Results (%d nodes returned):\n", len(results))
		printResults(results)
		fmt.Println("Note: nodes about ML use neural networks (semantically related) but")
		fmt.Println("      contain no exact tokens from the query, so they score zero or low.")
	}
	fmt.Println()

	// 6. Pre-fitted model via WithBM25Model.
	fmt.Println(sep)
	fmt.Println("=== Pre-fitted Model via WithBM25Model ===")
	fmt.Println(sep)
	fmt.Println()

	preFitted := embedding.NewBM25(embedding.WithBM25K1(1.2))
	corpusTexts := make([]string, len(nodes))
	for i, n := range nodes {
		corpusTexts[i] = n.Text
	}
	preFitted.Fit(corpusTexts)

	rPreFitted := retriever.NewBM25Retriever(nodes,
		retriever.WithBM25TopK(3),
		retriever.WithBM25Model(preFitted),
	)
	fmt.Println("Using a pre-fitted BM25 model (k1=1.2) supplied via WithBM25Model.")
	fmt.Printf("Query: %q\n", "neural networks deep learning")
	results, err = rPreFitted.Retrieve(ctx, schema.QueryBundle{QueryString: "neural networks deep learning"})
	if err != nil {
		fmt.Printf("  Error: %v\n\n", err)
	} else {
		printResults(results)
	}

	fmt.Println("=== BM25 Retriever Demo Complete ===")
}

// buildCorpus creates a set of nodes with varied text and metadata categories.
func buildCorpus() []schema.Node {
	entries := []struct {
		text     string
		category string
	}{
		{"Machine learning algorithms can automatically learn patterns from data without explicit programming.", "ml"},
		{"Deep learning is a subset of machine learning that uses neural networks with multiple layers.", "ml"},
		{"Natural language processing enables computers to understand and generate human language.", "nlp"},
		{"Database systems store and retrieve data efficiently using indexing and query optimization.", "db"},
		{"Data preprocessing is essential for machine learning model training and performance.", "ml"},
		{"Reinforcement learning trains agents through rewards and penalties in an environment.", "ml"},
		{"Computer vision algorithms process and analyze visual data from images and videos.", "cv"},
		{"Big data systems handle large volumes of data using distributed processing frameworks.", "db"},
		{"Neural networks learn representations through backpropagation and gradient descent.", "ml"},
		{"Information retrieval systems rank documents based on relevance to user queries.", "ir"},
	}

	nodes := make([]schema.Node, len(entries))
	for i, e := range entries {
		n := schema.NewTextNode(e.text)
		n.Metadata = map[string]interface{}{"category": e.category}
		nodes[i] = *n
	}
	return nodes
}

func printResults(results []schema.NodeWithScore) {
	if len(results) == 0 {
		fmt.Println("  (no results)")
		fmt.Println()
		return
	}
	for i, r := range results {
		cat := r.Node.Metadata["category"]
		fmt.Printf("  %d. [score=%.4f, cat=%v] %s\n", i+1, r.Score, cat, truncate(r.Node.Text, 60))
	}
	fmt.Println()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
