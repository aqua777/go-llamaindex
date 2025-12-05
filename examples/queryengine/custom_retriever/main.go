// Package main demonstrates implementing custom retrievers.
// This example corresponds to Python's query_engine/CustomRetrievers.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Custom Retriever Demo ===")
	fmt.Println("\nLLM initialized")

	// Sample documents
	documents := getSampleDocuments()

	separator := strings.Repeat("=", 70)

	// 2. Demonstrate BM25-style Retriever
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Retriever 1: BM25-Style Retriever ===")
	fmt.Println(separator)

	bm25Retriever := NewBM25Retriever(documents, 3)
	testRetriever(ctx, bm25Retriever, "machine learning algorithms", llmInstance)

	// 3. Demonstrate Metadata Filter Retriever
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Retriever 2: Metadata Filter Retriever ===")
	fmt.Println(separator)

	metadataRetriever := NewMetadataFilterRetriever(documents, map[string]interface{}{
		"category": "technology",
	})
	testRetriever(ctx, metadataRetriever, "What is cloud computing?", llmInstance)

	// 4. Demonstrate Hybrid Retriever (combines multiple strategies)
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Retriever 3: Hybrid Retriever ===")
	fmt.Println(separator)

	hybridRetriever := NewHybridRetriever(documents, 5)
	testRetriever(ctx, hybridRetriever, "artificial intelligence applications", llmInstance)

	// 5. Demonstrate Time-Weighted Retriever
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Retriever 4: Time-Weighted Retriever ===")
	fmt.Println(separator)

	timeWeightedRetriever := NewTimeWeightedRetriever(documents, 0.5)
	testRetriever(ctx, timeWeightedRetriever, "latest technology trends", llmInstance)

	// 6. Demonstrate Composable Retriever (chains retrievers)
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Retriever 5: Composable Retriever ===")
	fmt.Println(separator)

	// First filter by metadata, then apply BM25
	composableRetriever := NewComposableRetriever(
		NewMetadataFilterRetriever(documents, map[string]interface{}{"category": "technology"}),
		NewBM25Retriever(nil, 3), // Will receive filtered docs
	)
	testRetriever(ctx, composableRetriever, "programming languages", llmInstance)

	// 7. Show retriever comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Retriever Comparison ===")
	fmt.Println(separator)

	query := "What are the benefits of cloud computing?"
	queryBundle := schema.QueryBundle{QueryString: query}
	fmt.Printf("\nQuery: %s\n\n", query)

	retrievers := map[string]retriever.Retriever{
		"BM25":         NewBM25Retriever(documents, 3),
		"Metadata":     NewMetadataFilterRetriever(documents, map[string]interface{}{"category": "technology"}),
		"Hybrid":       NewHybridRetriever(documents, 3),
		"TimeWeighted": NewTimeWeightedRetriever(documents, 0.5),
	}

	for name, ret := range retrievers {
		nodes, err := ret.Retrieve(ctx, queryBundle)
		if err != nil {
			log.Printf("%s retriever failed: %v\n", name, err)
			continue
		}
		fmt.Printf("%s Retriever: %d results\n", name, len(nodes))
		for i, node := range nodes {
			if i >= 2 {
				break
			}
			fmt.Printf("  - Score: %.3f, Content: %s\n", node.Score, truncate(node.Node.GetContent(schema.MetadataModeNone), 50))
		}
	}

	fmt.Println("\n=== Custom Retriever Demo Complete ===")
}

// testRetriever tests a retriever with a query and synthesizes a response.
func testRetriever(ctx context.Context, ret retriever.Retriever, query string, llmModel llm.LLM) {
	fmt.Printf("\nQuery: %s\n", query)

	queryBundle := schema.QueryBundle{QueryString: query}
	nodes, err := ret.Retrieve(ctx, queryBundle)
	if err != nil {
		log.Printf("Retrieval failed: %v\n", err)
		return
	}

	fmt.Printf("Retrieved %d nodes:\n", len(nodes))
	for i, node := range nodes {
		if i >= 3 {
			fmt.Printf("  ... and %d more\n", len(nodes)-3)
			break
		}
		fmt.Printf("  %d. Score: %.3f, Content: %s\n", i+1, node.Score, truncate(node.Node.GetContent(schema.MetadataModeNone), 60))
	}

	// Create query engine and get response
	synth := synthesizer.NewSimpleSynthesizer(llmModel)
	engine := queryengine.NewRetrieverQueryEngine(ret, synth)

	response, err := engine.Query(ctx, query)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
		return
	}

	fmt.Printf("\nResponse: %s\n", truncate(response.Response, 200))
}

// ============================================================================
// Custom Retriever 1: BM25-Style Retriever
// ============================================================================

// BM25Retriever implements BM25-style term frequency scoring.
type BM25Retriever struct {
	*retriever.BaseRetriever
	documents []schema.Document
	topK      int
	k1        float64 // term frequency saturation parameter
	b         float64 // length normalization parameter
	avgDocLen float64
	docFreqs  map[string]int // document frequency for each term
}

// NewBM25Retriever creates a new BM25Retriever.
func NewBM25Retriever(documents []schema.Document, topK int) *BM25Retriever {
	r := &BM25Retriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		documents:     documents,
		topK:          topK,
		k1:            1.2,
		b:             0.75,
		docFreqs:      make(map[string]int),
	}

	// Calculate average document length and document frequencies
	if len(documents) > 0 {
		totalLen := 0
		for _, doc := range documents {
			words := strings.Fields(strings.ToLower(doc.Text))
			totalLen += len(words)

			// Count unique terms per document
			seen := make(map[string]bool)
			for _, word := range words {
				if !seen[word] {
					r.docFreqs[word]++
					seen[word] = true
				}
			}
		}
		r.avgDocLen = float64(totalLen) / float64(len(documents))
	}

	return r
}

// Retrieve implements the Retriever interface using BM25 scoring.
func (r *BM25Retriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryTerms := strings.Fields(strings.ToLower(query.QueryString))
	n := float64(len(r.documents))

	var results []schema.NodeWithScore

	for _, doc := range r.documents {
		docTerms := strings.Fields(strings.ToLower(doc.Text))
		docLen := float64(len(docTerms))

		// Count term frequencies in document
		termFreqs := make(map[string]int)
		for _, term := range docTerms {
			termFreqs[term]++
		}

		// Calculate BM25 score
		score := 0.0
		for _, term := range queryTerms {
			tf := float64(termFreqs[term])
			df := float64(r.docFreqs[term])

			if tf > 0 && df > 0 {
				// IDF component
				idf := math.Log((n - df + 0.5) / (df + 0.5))

				// TF component with length normalization
				tfNorm := (tf * (r.k1 + 1)) / (tf + r.k1*(1-r.b+r.b*(docLen/r.avgDocLen)))

				score += idf * tfNorm
			}
		}

		if score > 0 {
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			results = append(results, schema.NodeWithScore{
				Node:  *node,
				Score: score,
			})
		}
	}

	// Sort by score descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit to topK
	if len(results) > r.topK {
		results = results[:r.topK]
	}

	return results, nil
}

// ============================================================================
// Custom Retriever 2: Metadata Filter Retriever
// ============================================================================

// MetadataFilterRetriever filters documents by metadata before retrieval.
type MetadataFilterRetriever struct {
	*retriever.BaseRetriever
	documents []schema.Document
	filters   map[string]interface{}
}

// NewMetadataFilterRetriever creates a new MetadataFilterRetriever.
func NewMetadataFilterRetriever(documents []schema.Document, filters map[string]interface{}) *MetadataFilterRetriever {
	return &MetadataFilterRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		documents:     documents,
		filters:       filters,
	}
}

// Retrieve implements the Retriever interface with metadata filtering.
func (r *MetadataFilterRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	var results []schema.NodeWithScore
	queryLower := strings.ToLower(query.QueryString)

	for _, doc := range r.documents {
		// Check if document matches all filters
		if !r.matchesFilters(doc.Metadata) {
			continue
		}

		// Simple keyword scoring for matched documents
		docLower := strings.ToLower(doc.Text)
		score := 0.0

		queryWords := strings.Fields(queryLower)
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(docLower, word) {
				score += 0.2
			}
		}

		// Give base score for matching metadata
		if score == 0 {
			score = 0.1
		}

		node := schema.NewTextNode(doc.Text)
		node.Metadata = doc.Metadata
		results = append(results, schema.NodeWithScore{
			Node:  *node,
			Score: score,
		})
	}

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// matchesFilters checks if metadata matches all filters.
func (r *MetadataFilterRetriever) matchesFilters(metadata map[string]interface{}) bool {
	for key, value := range r.filters {
		docValue, exists := metadata[key]
		if !exists {
			return false
		}
		if docValue != value {
			return false
		}
	}
	return true
}

// ============================================================================
// Custom Retriever 3: Hybrid Retriever
// ============================================================================

// HybridRetriever combines keyword and semantic-like scoring.
type HybridRetriever struct {
	*retriever.BaseRetriever
	documents      []schema.Document
	topK           int
	keywordWeight  float64
	semanticWeight float64
}

// NewHybridRetriever creates a new HybridRetriever.
func NewHybridRetriever(documents []schema.Document, topK int) *HybridRetriever {
	return &HybridRetriever{
		BaseRetriever:  retriever.NewBaseRetriever(),
		documents:      documents,
		topK:           topK,
		keywordWeight:  0.5,
		semanticWeight: 0.5,
	}
}

// Retrieve implements the Retriever interface with hybrid scoring.
func (r *HybridRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	queryWords := strings.Fields(queryLower)

	var results []schema.NodeWithScore

	for _, doc := range r.documents {
		docLower := strings.ToLower(doc.Text)
		docWords := strings.Fields(docLower)

		// Keyword score (Jaccard-like)
		keywordScore := r.calculateKeywordScore(queryWords, docWords)

		// Semantic score (simulated with n-gram overlap)
		semanticScore := r.calculateSemanticScore(queryLower, docLower)

		// Combined score
		score := r.keywordWeight*keywordScore + r.semanticWeight*semanticScore

		if score > 0 {
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			node.Metadata["keyword_score"] = keywordScore
			node.Metadata["semantic_score"] = semanticScore
			results = append(results, schema.NodeWithScore{
				Node:  *node,
				Score: score,
			})
		}
	}

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Limit to topK
	if len(results) > r.topK {
		results = results[:r.topK]
	}

	return results, nil
}

// calculateKeywordScore calculates keyword overlap score.
func (r *HybridRetriever) calculateKeywordScore(queryWords, docWords []string) float64 {
	docWordSet := make(map[string]bool)
	for _, w := range docWords {
		docWordSet[w] = true
	}

	matches := 0
	for _, w := range queryWords {
		if len(w) > 2 && docWordSet[w] {
			matches++
		}
	}

	if len(queryWords) == 0 {
		return 0
	}

	return float64(matches) / float64(len(queryWords))
}

// calculateSemanticScore simulates semantic similarity with n-gram overlap.
func (r *HybridRetriever) calculateSemanticScore(query, doc string) float64 {
	// Generate bigrams
	queryBigrams := r.generateNGrams(query, 2)
	docBigrams := r.generateNGrams(doc, 2)

	if len(queryBigrams) == 0 {
		return 0
	}

	docBigramSet := make(map[string]bool)
	for _, bg := range docBigrams {
		docBigramSet[bg] = true
	}

	matches := 0
	for _, bg := range queryBigrams {
		if docBigramSet[bg] {
			matches++
		}
	}

	return float64(matches) / float64(len(queryBigrams))
}

// generateNGrams generates n-grams from text.
func (r *HybridRetriever) generateNGrams(text string, n int) []string {
	words := strings.Fields(text)
	if len(words) < n {
		return nil
	}

	var ngrams []string
	for i := 0; i <= len(words)-n; i++ {
		ngram := strings.Join(words[i:i+n], " ")
		ngrams = append(ngrams, ngram)
	}

	return ngrams
}

// ============================================================================
// Custom Retriever 4: Time-Weighted Retriever
// ============================================================================

// TimeWeightedRetriever weights results by recency.
type TimeWeightedRetriever struct {
	*retriever.BaseRetriever
	documents   []schema.Document
	decayFactor float64 // How much to weight recency (0-1)
}

// NewTimeWeightedRetriever creates a new TimeWeightedRetriever.
func NewTimeWeightedRetriever(documents []schema.Document, decayFactor float64) *TimeWeightedRetriever {
	return &TimeWeightedRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		documents:     documents,
		decayFactor:   decayFactor,
	}
}

// Retrieve implements the Retriever interface with time weighting.
func (r *TimeWeightedRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for i, doc := range r.documents {
		docLower := strings.ToLower(doc.Text)

		// Base relevance score
		relevanceScore := 0.0
		queryWords := strings.Fields(queryLower)
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(docLower, word) {
				relevanceScore += 0.2
			}
		}

		if relevanceScore == 0 {
			continue
		}

		// Time weight (newer documents have higher index in our sample)
		// In real scenarios, you'd use actual timestamps
		recencyScore := float64(i+1) / float64(len(r.documents))

		// Combined score
		score := (1-r.decayFactor)*relevanceScore + r.decayFactor*recencyScore

		node := schema.NewTextNode(doc.Text)
		node.Metadata = doc.Metadata
		node.Metadata["relevance_score"] = relevanceScore
		node.Metadata["recency_score"] = recencyScore
		results = append(results, schema.NodeWithScore{
			Node:  *node,
			Score: score,
		})
	}

	// Sort by score
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results, nil
}

// ============================================================================
// Custom Retriever 5: Composable Retriever
// ============================================================================

// ComposableRetriever chains multiple retrievers.
type ComposableRetriever struct {
	*retriever.BaseRetriever
	firstRetriever  retriever.Retriever
	secondRetriever retriever.Retriever
}

// NewComposableRetriever creates a new ComposableRetriever.
func NewComposableRetriever(first, second retriever.Retriever) *ComposableRetriever {
	return &ComposableRetriever{
		BaseRetriever:   retriever.NewBaseRetriever(),
		firstRetriever:  first,
		secondRetriever: second,
	}
}

// Retrieve implements the Retriever interface by chaining retrievers.
func (r *ComposableRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// First retrieval
	firstResults, err := r.firstRetriever.Retrieve(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("first retriever failed: %w", err)
	}

	fmt.Printf("  First retriever returned %d results\n", len(firstResults))

	// Convert results to documents for second retriever
	var docs []schema.Document
	for _, node := range firstResults {
		docs = append(docs, schema.Document{
			Text:     node.Node.GetContent(schema.MetadataModeNone),
			Metadata: node.Node.Metadata,
		})
	}

	// If second retriever is BM25, update its documents
	if bm25, ok := r.secondRetriever.(*BM25Retriever); ok {
		bm25.documents = docs
	}

	// Second retrieval
	secondResults, err := r.secondRetriever.Retrieve(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("second retriever failed: %w", err)
	}

	fmt.Printf("  Second retriever returned %d results\n", len(secondResults))

	return secondResults, nil
}

// ============================================================================
// Sample Data
// ============================================================================

func getSampleDocuments() []schema.Document {
	return []schema.Document{
		{
			Text:     "Machine learning is a subset of artificial intelligence that enables systems to learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2023},
		},
		{
			Text:     "Cloud computing provides on-demand computing resources over the internet. Major providers include AWS, Azure, and Google Cloud Platform.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2023},
		},
		{
			Text:     "Python is a popular programming language known for its simplicity and readability. It's widely used in data science, web development, and automation.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2022},
		},
		{
			Text:     "The Renaissance was a period of cultural rebirth in Europe from the 14th to 17th century. It saw advances in art, science, and philosophy.",
			Metadata: map[string]interface{}{"category": "history", "year": 2021},
		},
		{
			Text:     "Quantum computing uses quantum mechanical phenomena to perform computations. It has potential applications in cryptography and drug discovery.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2024},
		},
		{
			Text:     "Deep learning is a type of machine learning using neural networks with multiple layers. It powers applications like image recognition and natural language processing.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2024},
		},
		{
			Text:     "Artificial intelligence applications include chatbots, recommendation systems, autonomous vehicles, and medical diagnosis tools.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2024},
		},
		{
			Text:     "The benefits of cloud computing include scalability, cost savings, flexibility, and disaster recovery capabilities.",
			Metadata: map[string]interface{}{"category": "technology", "year": 2023},
		},
	}
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
