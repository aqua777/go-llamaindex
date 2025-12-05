// Package main demonstrates vector store operations.
// This example corresponds to Python's low_level/vector_store.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/store/chromem"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create embedding model (empty strings use defaults/env vars)
	embedModel := embedding.NewOpenAIEmbedding("", "")
	fmt.Println("Embedding model initialized")

	// 2. Create in-memory vector store
	fmt.Println("\n=== Creating In-Memory Vector Store ===")
	store, err := chromem.NewChromemStore("", "demo_collection")
	if err != nil {
		log.Fatalf("Failed to create vector store: %v", err)
	}
	fmt.Println("Vector store created successfully")

	// 3. Create sample nodes with embeddings
	fmt.Println("\n=== Creating Sample Nodes ===")
	sampleTexts := []struct {
		text     string
		category string
		year     int
	}{
		{"Machine learning is a subset of artificial intelligence.", "technology", 2023},
		{"Deep learning uses neural networks with many layers.", "technology", 2023},
		{"Natural language processing enables computers to understand text.", "technology", 2022},
		{"The stock market showed strong gains this quarter.", "finance", 2023},
		{"Investment strategies should consider risk tolerance.", "finance", 2022},
		{"Climate change affects global weather patterns.", "science", 2023},
		{"Renewable energy sources are becoming more affordable.", "science", 2022},
	}

	nodes := make([]schema.Node, len(sampleTexts))
	for i, sample := range sampleTexts {
		// Generate embedding
		emb, err := embedModel.GetTextEmbedding(ctx, sample.text)
		if err != nil {
			log.Fatalf("Failed to generate embedding: %v", err)
		}

		nodes[i] = schema.Node{
			ID:        fmt.Sprintf("node_%d", i),
			Text:      sample.text,
			Type:      schema.ObjectTypeText,
			Embedding: emb,
			Metadata: map[string]interface{}{
				"category": sample.category,
				"year":     sample.year,
			},
		}
		fmt.Printf("Created node %d: %s (category: %s)\n", i, truncate(sample.text, 50), sample.category)
	}

	// 4. Add nodes to vector store
	fmt.Println("\n=== Adding Nodes to Vector Store ===")
	ids, err := store.Add(ctx, nodes)
	if err != nil {
		log.Fatalf("Failed to add nodes: %v", err)
	}
	fmt.Printf("Added %d nodes with IDs: %v\n", len(ids), ids)

	// 5. Query the vector store
	fmt.Println("\n=== Basic Query ===")
	queryText := "How does AI learn from data?"
	queryEmb, err := embedModel.GetTextEmbedding(ctx, queryText)
	if err != nil {
		log.Fatalf("Failed to generate query embedding: %v", err)
	}

	query := schema.VectorStoreQuery{
		Embedding: queryEmb,
		TopK:      3,
	}

	results, err := store.Query(ctx, query)
	if err != nil {
		log.Fatalf("Query failed: %v", err)
	}

	fmt.Printf("Query: %s\n", queryText)
	fmt.Println("Results:")
	for i, result := range results {
		fmt.Printf("  %d. [Score: %.4f] %s\n", i+1, result.Score, result.Node.Text)
		fmt.Printf("     Metadata: %v\n", result.Node.Metadata)
	}

	// 6. Query with metadata filter
	fmt.Println("\n=== Query with Metadata Filter (category=technology) ===")
	queryWithFilter := schema.VectorStoreQuery{
		Embedding: queryEmb,
		TopK:      5,
		Filters: schema.NewMetadataFilters(
			schema.NewMetadataFilter("category", "technology"),
		),
	}

	filteredResults, err := store.Query(ctx, queryWithFilter)
	if err != nil {
		log.Fatalf("Filtered query failed: %v", err)
	}

	fmt.Printf("Query: %s (filtered by category=technology)\n", queryText)
	fmt.Println("Results:")
	for i, result := range filteredResults {
		fmt.Printf("  %d. [Score: %.4f] %s\n", i+1, result.Score, result.Node.Text)
	}

	// 7. Different query - finance related
	fmt.Println("\n=== Finance Query ===")
	financeQuery := "What are good investment strategies?"
	financeEmb, err := embedModel.GetTextEmbedding(ctx, financeQuery)
	if err != nil {
		log.Fatalf("Failed to generate query embedding: %v", err)
	}

	financeResults, err := store.Query(ctx, schema.VectorStoreQuery{
		Embedding: financeEmb,
		TopK:      3,
	})
	if err != nil {
		log.Fatalf("Finance query failed: %v", err)
	}

	fmt.Printf("Query: %s\n", financeQuery)
	fmt.Println("Results:")
	for i, result := range financeResults {
		fmt.Printf("  %d. [Score: %.4f] %s\n", i+1, result.Score, result.Node.Text)
	}

	// 8. Delete a node
	fmt.Println("\n=== Deleting a Node ===")
	nodeToDelete := "node_0"
	err = store.Delete(ctx, nodeToDelete)
	if err != nil {
		log.Printf("Warning: Delete failed: %v", err)
	} else {
		fmt.Printf("Deleted node: %s\n", nodeToDelete)
	}

	// 9. Demonstrate persistence
	fmt.Println("\n=== Persistent Vector Store ===")
	persistPath := "./chromem_persist"
	defer os.RemoveAll(persistPath) // Clean up after demo

	persistentStore, err := chromem.NewChromemStore(persistPath, "persistent_collection")
	if err != nil {
		log.Fatalf("Failed to create persistent store: %v", err)
	}
	fmt.Printf("Created persistent store at: %s\n", persistPath)

	// Add a node to persistent store
	_, err = persistentStore.Add(ctx, []schema.Node{nodes[0]})
	if err != nil {
		log.Fatalf("Failed to add to persistent store: %v", err)
	}
	fmt.Println("Added node to persistent store")

	// Query persistent store
	persistResults, err := persistentStore.Query(ctx, schema.VectorStoreQuery{
		Embedding: queryEmb,
		TopK:      1,
	})
	if err != nil {
		log.Fatalf("Persistent store query failed: %v", err)
	}
	fmt.Printf("Persistent store query returned %d results\n", len(persistResults))

	fmt.Println("\n=== Vector Store Operations Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
