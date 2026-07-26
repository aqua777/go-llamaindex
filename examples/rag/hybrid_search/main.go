// Package main demonstrates end-to-end hybrid search using SimpleVectorStore.
//
// Demo: Hybrid Search
// Sprint: bm25-hybrid-search
// Phase: 2
//
// This example requires no environment variables or API keys. It uses
// MockEmbeddingModel from the embedding package to produce deterministic
// dense vectors, combined with BM25 sparse scoring inside SimpleVectorStore.
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/store"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	if err := run(); err != nil {
		fmt.Printf("ERROR: %v\n", err)
		panic(err)
	}
}

func run() error {
	ctx := context.Background()

	sep := strings.Repeat("=", 70)
	fmt.Println(sep)
	fmt.Println("Demo: Hybrid Search with SimpleVectorStore")
	fmt.Println(sep)
	fmt.Println()
	fmt.Println("This demo indexes 6 nodes and queries them in three modes:")
	fmt.Println("  default  — pure dense cosine similarity")
	fmt.Println("  sparse   — BM25 keyword-based ranking")
	fmt.Println("  hybrid   — weighted combination of dense + sparse")
	fmt.Println()

	// Build corpus.
	nodes := buildNodes()
	fmt.Printf("Corpus: %d nodes\n\n", len(nodes))
	for _, n := range nodes {
		fmt.Printf("  [%s] %s\n", n.ID, truncate(n.Text, 65))
	}
	fmt.Println()

	// Index nodes into SimpleVectorStore.
	vs := store.NewSimpleVectorStore()
	_, err := vs.Add(ctx, nodes)
	if err != nil {
		return fmt.Errorf("failed to add nodes: %w", err)
	}

	query := "machine learning neural networks"
	// Use a fixed embedding that points toward the ML cluster (nodes 1, 2, 4).
	queryEmb := []float32{1, 1, 0, 0, 0, 0}

	fmt.Printf("Query: %q\n", query)
	fmt.Println()

	// Default (dense) mode.
	fmt.Println(sep)
	fmt.Println("Mode: default (pure dense cosine similarity)")
	fmt.Println(sep)
	denseResults, err := vs.Query(ctx, schema.VectorStoreQuery{
		QueryEmbedding: queryEmb,
		SimilarityTopK: 4,
		Mode:           schema.QueryModeDefault,
	})
	if err != nil {
		return fmt.Errorf("dense query: %w", err)
	}
	printResults(denseResults)

	// Sparse (BM25) mode.
	fmt.Println(sep)
	fmt.Println("Mode: sparse (BM25 keyword ranking)")
	fmt.Println(sep)
	sparseResults, err := vs.Query(ctx, schema.VectorStoreQuery{
		QueryStr:       query,
		SimilarityTopK: 4,
		Mode:           schema.QueryModeSparse,
	})
	if err != nil {
		return fmt.Errorf("sparse query: %w", err)
	}
	printResults(sparseResults)

	// Hybrid mode (default alpha = 0.5).
	fmt.Println(sep)
	fmt.Println("Mode: hybrid (alpha=0.5, equal weight dense+sparse)")
	fmt.Println(sep)
	hybridResults, err := vs.Query(ctx, schema.VectorStoreQuery{
		QueryStr:       query,
		QueryEmbedding: queryEmb,
		SimilarityTopK: 4,
		Mode:           schema.QueryModeHybrid,
	})
	if err != nil {
		return fmt.Errorf("hybrid query: %w", err)
	}
	printResults(hybridResults)

	// End-to-end with VectorRetriever in hybrid mode.
	// VectorRetriever does not expose alpha directly, so we wrap the store
	// with a thin alphaStore that injects the desired alpha into every Query.
	fmt.Println(sep)
	fmt.Println("VectorRetriever — hybrid mode (alpha=0.7, denser weight)")
	fmt.Println(sep)
	a := 0.7
	mockEmb := embedding.NewMockEmbeddingModel(queryEmb)
	vr := retriever.NewVectorRetriever(
		&alphaStore{VectorStore: vs, alpha: a},
		mockEmb,
		retriever.WithTopK(4),
		retriever.WithQueryMode(schema.QueryModeHybrid),
	)
	retrieverResults, err := vr.Retrieve(ctx, schema.QueryBundle{QueryString: query})
	if err != nil {
		return fmt.Errorf("retriever hybrid query: %w", err)
	}
	printResults(retrieverResults)

	fmt.Println(sep)
	fmt.Println("SUCCESS: Hybrid search demo completed.")
	fmt.Println(sep)
	return nil
}

// buildNodes creates 6 nodes with deterministic embeddings and varied text.
// Embedding dimensions correspond to: [ml, nn, db, nlp, cv, other].
func buildNodes() []schema.Node {
	entries := []struct {
		id   string
		text string
		emb  []float32
	}{
		{"n1", "Machine learning algorithms learn patterns from training data.", []float32{1, 0.5, 0, 0, 0, 0}},
		{"n2", "Neural networks use backpropagation to optimize weights.", []float32{0.8, 1, 0, 0, 0, 0}},
		{"n3", "Database indexing improves query performance significantly.", []float32{0, 0, 1, 0, 0, 0}},
		{"n4", "Deep learning is a subset of machine learning using neural networks.", []float32{0.9, 0.9, 0, 0, 0, 0}},
		{"n5", "Natural language processing enables text understanding.", []float32{0, 0, 0, 1, 0, 0}},
		{"n6", "Computer vision processes images for object detection.", []float32{0, 0, 0, 0, 1, 0}},
	}

	nodes := make([]schema.Node, len(entries))
	for i, e := range entries {
		n := schema.NewTextNode(e.text)
		n.ID = e.id
		n.Embedding = e.emb
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
		fmt.Printf("  %d. [score=%.4f] [%s] %s\n", i+1, r.Score, r.Node.ID, truncate(r.Node.Text, 55))
	}
	fmt.Println()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// alphaStore wraps a VectorStore and injects a fixed alpha value into every
// hybrid Query call. VectorRetriever does not expose alpha directly, so this
// thin wrapper is used in the demo to demonstrate a specific alpha value.
type alphaStore struct {
	store.VectorStore
	alpha float64
}

func (a *alphaStore) Query(ctx context.Context, query schema.VectorStoreQuery) ([]schema.NodeWithScore, error) {
	query.Alpha = &a.alpha
	return a.VectorStore.Query(ctx, query)
}
