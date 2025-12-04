// Package main demonstrates BM25 sparse retrieval.
// This example corresponds to Python's retrievers/bm25_retriever.ipynb
package main

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/aqua777/go-llamaindex/embedding"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== BM25 Retriever Demo ===")
	fmt.Println()
	fmt.Println("BM25 (Best Matching 25) is a sparse retrieval algorithm that ranks")
	fmt.Println("documents based on term frequency and inverse document frequency.")
	fmt.Println()

	separator := strings.Repeat("=", 70)

	// 1. Create document corpus
	fmt.Println(separator)
	fmt.Println("=== Creating Document Corpus ===")
	fmt.Println(separator)
	fmt.Println()

	documents := getDocumentCorpus()
	fmt.Printf("Created corpus with %d documents\n\n", len(documents))

	for i, doc := range documents {
		fmt.Printf("Doc %d: %s\n", i+1, truncate(doc, 70))
	}
	fmt.Println()

	// 2. Initialize and fit BM25 model
	fmt.Println(separator)
	fmt.Println("=== Initializing BM25 Model ===")
	fmt.Println(separator)
	fmt.Println()

	// Create BM25 with default parameters
	bm25 := embedding.NewBM25()
	fmt.Println("Created BM25 model with default parameters:")
	fmt.Println("  k1 = 1.5 (term frequency saturation)")
	fmt.Println("  b = 0.75 (document length normalization)")
	fmt.Println()

	// Fit on corpus
	fmt.Println("Fitting model on corpus...")
	bm25.Fit(documents)
	fmt.Printf("Vocabulary size: %d terms\n\n", bm25.GetVocabularySize())

	// 3. Basic BM25 retrieval
	fmt.Println(separator)
	fmt.Println("=== Basic BM25 Retrieval ===")
	fmt.Println(separator)
	fmt.Println()

	queries := []string{
		"machine learning algorithms",
		"natural language processing",
		"database optimization",
	}

	for _, query := range queries {
		fmt.Printf("Query: %s\n", query)
		results := retrieveWithBM25(ctx, bm25, documents, query, 3)
		printResults(results)
	}

	// 4. BM25 with custom parameters
	fmt.Println(separator)
	fmt.Println("=== BM25 with Custom Parameters ===")
	fmt.Println(separator)
	fmt.Println()

	// Higher k1 = more weight on term frequency
	bm25HighK1 := embedding.NewBM25(embedding.WithBM25K1(2.0))
	bm25HighK1.Fit(documents)
	fmt.Println("BM25 with k1=2.0 (higher term frequency weight):")
	results1 := retrieveWithBM25(ctx, bm25HighK1, documents, "machine learning", 3)
	printResults(results1)

	// Lower b = less document length normalization
	bm25LowB := embedding.NewBM25(embedding.WithBM25B(0.3))
	bm25LowB.Fit(documents)
	fmt.Println("BM25 with b=0.3 (less length normalization):")
	results2 := retrieveWithBM25(ctx, bm25LowB, documents, "machine learning", 3)
	printResults(results2)

	// 5. BM25 with custom stopwords
	fmt.Println(separator)
	fmt.Println("=== BM25 with Custom Stopwords ===")
	fmt.Println(separator)
	fmt.Println()

	customStopwords := []string{"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
		"have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
		"data", "system", "systems"} // Added domain-specific stopwords

	bm25Custom := embedding.NewBM25(embedding.WithBM25Stopwords(customStopwords))
	bm25Custom.Fit(documents)
	fmt.Println("BM25 with custom stopwords (including 'data', 'system'):")
	fmt.Printf("Query: %s\n", "data processing system")
	results3 := retrieveWithBM25(ctx, bm25Custom, documents, "data processing system", 3)
	printResults(results3)

	// 6. Sparse embedding inspection
	fmt.Println(separator)
	fmt.Println("=== Sparse Embedding Inspection ===")
	fmt.Println(separator)
	fmt.Println()

	testDoc := "Machine learning models process data efficiently"
	fmt.Printf("Document: %s\n\n", testDoc)

	sparseEmb, _ := bm25.GetSparseEmbedding(ctx, testDoc)
	fmt.Printf("Sparse embedding dimension: %d\n", sparseEmb.Dimension)
	fmt.Printf("Non-zero entries: %d\n", len(sparseEmb.Indices))
	fmt.Println()

	// Show top terms by BM25 score
	fmt.Println("Top terms by BM25 score:")
	type termScore struct {
		idx   int
		score float64
	}
	var terms []termScore
	for i, idx := range sparseEmb.Indices {
		terms = append(terms, termScore{idx, sparseEmb.Values[i]})
	}
	sort.Slice(terms, func(i, j int) bool {
		return terms[i].score > terms[j].score
	})
	for i, t := range terms[:min(5, len(terms))] {
		fmt.Printf("  %d. Index %d: %.4f\n", i+1, t.idx, t.score)
	}
	fmt.Println()

	// 7. Query embedding vs document embedding
	fmt.Println(separator)
	fmt.Println("=== Query vs Document Embeddings ===")
	fmt.Println(separator)
	fmt.Println()

	testQuery := "machine learning"
	fmt.Printf("Query: %s\n", testQuery)
	fmt.Printf("Document: %s\n\n", testDoc)

	queryEmb, _ := bm25.GetSparseQueryEmbedding(ctx, testQuery)
	docEmb, _ := bm25.GetSparseEmbedding(ctx, testDoc)

	fmt.Printf("Query embedding non-zero entries: %d\n", len(queryEmb.Indices))
	fmt.Printf("Document embedding non-zero entries: %d\n", len(docEmb.Indices))
	fmt.Printf("Dot product (relevance score): %.4f\n\n", queryEmb.DotProduct(docEmb))

	// 8. BM25+ variant
	fmt.Println(separator)
	fmt.Println("=== BM25+ Variant ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("BM25+ adds a delta parameter to address the issue where")
	fmt.Println("long documents may be unfairly penalized.")
	fmt.Println()

	bm25Plus := embedding.NewBM25Plus(1.0) // delta = 1.0
	bm25Plus.Fit(documents)

	fmt.Println("Comparing BM25 vs BM25+ on same query:")
	fmt.Printf("Query: %s\n\n", "efficient data processing algorithms")

	resultsStd := retrieveWithBM25(ctx, bm25, documents, "efficient data processing algorithms", 3)
	fmt.Println("BM25 (standard):")
	printResults(resultsStd)

	resultsPlus := retrieveWithBM25Plus(ctx, bm25Plus, documents, "efficient data processing algorithms", 3)
	fmt.Println("BM25+ (with delta=1.0):")
	printResults(resultsPlus)

	// 9. Hybrid retrieval concept
	fmt.Println(separator)
	fmt.Println("=== Hybrid Retrieval (BM25 + Dense) ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("Hybrid retrieval combines BM25 (sparse) with dense embeddings:")
	fmt.Println()
	fmt.Println("1. BM25 Strengths:")
	fmt.Println("   - Exact keyword matching")
	fmt.Println("   - No training required")
	fmt.Println("   - Interpretable scores")
	fmt.Println("   - Fast for large vocabularies")
	fmt.Println()
	fmt.Println("2. Dense Embedding Strengths:")
	fmt.Println("   - Semantic similarity")
	fmt.Println("   - Handles synonyms")
	fmt.Println("   - Context-aware matching")
	fmt.Println()
	fmt.Println("3. Hybrid Approach:")
	fmt.Println("   - Run both retrievers")
	fmt.Println("   - Normalize scores")
	fmt.Println("   - Weighted combination (e.g., 0.5 * BM25 + 0.5 * Dense)")
	fmt.Println("   - Or use Reciprocal Rank Fusion (RRF)")
	fmt.Println()

	// 10. IDF inspection
	fmt.Println(separator)
	fmt.Println("=== Term IDF Values ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("IDF (Inverse Document Frequency) measures term rarity:")
	fmt.Println("Higher IDF = rarer term = more discriminative")
	fmt.Println()

	testTerms := []string{"machine", "learning", "data", "the", "algorithms", "optimization"}
	for _, term := range testTerms {
		idf := bm25.GetTermIDF(term)
		rarity := "common"
		if idf > 1.0 {
			rarity = "rare"
		} else if idf > 0.5 {
			rarity = "moderate"
		}
		fmt.Printf("  '%s': IDF=%.4f (%s)\n", term, idf, rarity)
	}
	fmt.Println()

	fmt.Println("=== BM25 Retriever Demo Complete ===")
}

// retrieveWithBM25 retrieves documents using BM25 scoring.
func retrieveWithBM25(ctx context.Context, bm25 *embedding.BM25, documents []string, query string, topK int) []schema.NodeWithScore {
	var results []schema.NodeWithScore

	for i, doc := range documents {
		score := bm25.Score(query, doc)
		results = append(results, schema.NodeWithScore{
			Node: schema.Node{
				ID:   fmt.Sprintf("doc-%d", i+1),
				Text: doc,
				Type: schema.ObjectTypeText,
			},
			Score: score,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(topK, len(results))]
}

// retrieveWithBM25Plus retrieves documents using BM25+ scoring.
func retrieveWithBM25Plus(ctx context.Context, bm25Plus *embedding.BM25Plus, documents []string, query string, topK int) []schema.NodeWithScore {
	var results []schema.NodeWithScore

	queryEmb, _ := bm25Plus.GetSparseEmbedding(ctx, query)

	for i, doc := range documents {
		docEmb, _ := bm25Plus.GetSparseEmbedding(ctx, doc)
		score := queryEmb.DotProduct(docEmb)

		results = append(results, schema.NodeWithScore{
			Node: schema.Node{
				ID:   fmt.Sprintf("doc-%d", i+1),
				Text: doc,
				Type: schema.ObjectTypeText,
			},
			Score: score,
		})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(topK, len(results))]
}

// getDocumentCorpus returns a sample document corpus.
func getDocumentCorpus() []string {
	return []string{
		"Machine learning algorithms can automatically learn patterns from data without explicit programming.",
		"Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
		"Natural language processing enables computers to understand and generate human language.",
		"Database systems store and retrieve data efficiently using indexing and query optimization.",
		"Data preprocessing is essential for machine learning model training and performance.",
		"Reinforcement learning trains agents through rewards and penalties in an environment.",
		"Computer vision algorithms process and analyze visual data from images and videos.",
		"Big data systems handle large volumes of data using distributed processing frameworks.",
		"Neural networks learn representations through backpropagation and gradient descent.",
		"Information retrieval systems rank documents based on relevance to user queries.",
	}
}

func printResults(results []schema.NodeWithScore) {
	for i, r := range results {
		fmt.Printf("  %d. [Score: %.4f] %s\n", i+1, r.Score, truncate(r.Node.Text, 60))
	}
	fmt.Println()
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
