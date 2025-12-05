// Package main demonstrates composable retriever patterns.
// This example corresponds to Python's retrievers/composable_retrievers.ipynb
package main

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Composable Retrievers Demo ===")
	fmt.Println()
	fmt.Println("Composable retrievers allow building complex retrieval pipelines")
	fmt.Println("by combining multiple retrievers in various configurations.")
	fmt.Println()

	separator := strings.Repeat("=", 70)

	// 1. Create base retrievers
	fmt.Println(separator)
	fmt.Println("=== Creating Base Retrievers ===")
	fmt.Println(separator)
	fmt.Println()

	// Semantic retriever (simulates dense embeddings)
	semanticRetriever := NewSemanticRetriever(getSemanticDocs())
	fmt.Println("Created SemanticRetriever (dense embedding simulation)")

	// Keyword retriever (simulates BM25/sparse)
	keywordRetriever := NewKeywordRetriever(getKeywordDocs())
	fmt.Println("Created KeywordRetriever (keyword matching simulation)")

	// Metadata retriever (filters by metadata)
	metadataRetriever := NewMetadataRetriever(getMetadataDocs())
	fmt.Println("Created MetadataRetriever (metadata filtering)")
	fmt.Println()

	query := schema.QueryBundle{QueryString: "machine learning model training optimization"}

	// 2. Sequential Composition (Pipeline)
	fmt.Println(separator)
	fmt.Println("=== Sequential Composition (Pipeline) ===")
	fmt.Println(separator)
	fmt.Println()
	fmt.Println("Pattern: Retriever1 -> Filter -> Retriever2 -> Rerank")
	fmt.Println()

	// First stage: broad semantic retrieval
	fmt.Println("Stage 1: Semantic Retrieval (broad)")
	semanticResults, _ := semanticRetriever.Retrieve(ctx, query)
	printResults("Semantic", semanticResults)

	// Second stage: filter by score threshold
	fmt.Println("Stage 2: Score Filtering (threshold > 0.3)")
	filteredResults := filterByScore(semanticResults, 0.3)
	printResults("Filtered", filteredResults)

	// Third stage: rerank by keyword relevance
	fmt.Println("Stage 3: Keyword Reranking")
	rerankedResults := rerankByKeywords(filteredResults, query.QueryString)
	printResults("Reranked", rerankedResults)

	// 3. Parallel Composition (Ensemble)
	fmt.Println(separator)
	fmt.Println("=== Parallel Composition (Ensemble) ===")
	fmt.Println(separator)
	fmt.Println()
	fmt.Println("Pattern: [Retriever1, Retriever2, Retriever3] -> Merge -> Dedupe")
	fmt.Println()

	// Run all retrievers in parallel (simulated)
	fmt.Println("Running 3 retrievers in parallel...")
	results1, _ := semanticRetriever.Retrieve(ctx, query)
	results2, _ := keywordRetriever.Retrieve(ctx, query)
	results3, _ := metadataRetriever.Retrieve(ctx, query)

	fmt.Printf("  Semantic: %d results\n", len(results1))
	fmt.Printf("  Keyword: %d results\n", len(results2))
	fmt.Printf("  Metadata: %d results\n", len(results3))
	fmt.Println()

	// Merge and deduplicate
	fmt.Println("Merging and deduplicating...")
	merged := mergeResults(results1, results2, results3)
	printResults("Merged", merged)

	// 4. Conditional Composition (Router)
	fmt.Println(separator)
	fmt.Println("=== Conditional Composition (Router) ===")
	fmt.Println(separator)
	fmt.Println()
	fmt.Println("Pattern: Classify Query -> Route to Appropriate Retriever")
	fmt.Println()

	queries := []schema.QueryBundle{
		{QueryString: "what is the API rate limit"},           // keyword-heavy
		{QueryString: "explain how neural networks learn"},    // semantic
		{QueryString: "documents from 2024 about compliance"}, // metadata
	}

	for _, q := range queries {
		queryType := classifyQuery(q.QueryString)
		fmt.Printf("Query: %s\n", q.QueryString)
		fmt.Printf("Classification: %s\n", queryType)

		var results []schema.NodeWithScore
		switch queryType {
		case "keyword":
			results, _ = keywordRetriever.Retrieve(ctx, q)
		case "semantic":
			results, _ = semanticRetriever.Retrieve(ctx, q)
		case "metadata":
			results, _ = metadataRetriever.Retrieve(ctx, q)
		}
		printResults("Routed", results)
	}

	// 5. Hierarchical Composition
	fmt.Println(separator)
	fmt.Println("=== Hierarchical Composition ===")
	fmt.Println(separator)
	fmt.Println()
	fmt.Println("Pattern: Coarse Retrieval -> Fine Retrieval -> Final Selection")
	fmt.Println()

	// Coarse: get many candidates
	fmt.Println("Level 1: Coarse retrieval (top 10)")
	coarseRetriever := NewCoarseRetriever(getAllDocs(), 10)
	coarseResults, _ := coarseRetriever.Retrieve(ctx, query)
	fmt.Printf("  Retrieved %d candidates\n", len(coarseResults))

	// Fine: re-retrieve from candidates
	fmt.Println("Level 2: Fine retrieval (top 5 from candidates)")
	fineResults := fineRetrieve(coarseResults, query.QueryString, 5)
	fmt.Printf("  Refined to %d results\n", len(fineResults))

	// Final: select best
	fmt.Println("Level 3: Final selection (top 3)")
	finalResults := fineResults[:min(3, len(fineResults))]
	printResults("Final", finalResults)

	// 6. Composition Patterns Summary
	fmt.Println(separator)
	fmt.Println("=== Composition Patterns Summary ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. Sequential (Pipeline):")
	fmt.Println("   Retriever -> Transform -> Filter -> Rerank")
	fmt.Println("   Use: Multi-stage refinement")
	fmt.Println()
	fmt.Println("2. Parallel (Ensemble):")
	fmt.Println("   [R1, R2, R3] -> Merge -> Dedupe")
	fmt.Println("   Use: Combining different retrieval strategies")
	fmt.Println()
	fmt.Println("3. Conditional (Router):")
	fmt.Println("   Classify -> Route to best retriever")
	fmt.Println("   Use: Query-dependent retrieval")
	fmt.Println()
	fmt.Println("4. Hierarchical (Coarse-to-Fine):")
	fmt.Println("   Coarse -> Fine -> Final")
	fmt.Println("   Use: Large-scale retrieval with refinement")
	fmt.Println()
	fmt.Println("5. Hybrid (Dense + Sparse):")
	fmt.Println("   Semantic + Keyword -> Weighted Merge")
	fmt.Println("   Use: Best of both worlds")
	fmt.Println()

	fmt.Println("=== Composable Retrievers Demo Complete ===")
}

// SemanticRetriever simulates dense embedding retrieval.
type SemanticRetriever struct {
	*retriever.BaseRetriever
	docs []schema.NodeWithScore
}

func NewSemanticRetriever(docs []schema.NodeWithScore) *SemanticRetriever {
	return &SemanticRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		docs:          docs,
	}
}

func (r *SemanticRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Simulate semantic similarity scoring
	var results []schema.NodeWithScore
	queryWords := strings.Fields(strings.ToLower(query.QueryString))

	for _, doc := range r.docs {
		textLower := strings.ToLower(doc.Node.Text)
		score := 0.0

		// Simulate embedding similarity with word overlap + base score
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(textLower, word) {
				score += 0.1
			}
		}
		score += doc.Score * 0.3 // Add base relevance

		if score > 0.2 {
			results = append(results, schema.NodeWithScore{
				Node:  doc.Node,
				Score: score,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(5, len(results))], nil
}

// KeywordRetriever simulates BM25/keyword retrieval.
type KeywordRetriever struct {
	*retriever.BaseRetriever
	docs []schema.NodeWithScore
}

func NewKeywordRetriever(docs []schema.NodeWithScore) *KeywordRetriever {
	return &KeywordRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		docs:          docs,
	}
}

func (r *KeywordRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Simulate keyword matching (BM25-like)
	var results []schema.NodeWithScore
	queryWords := strings.Fields(strings.ToLower(query.QueryString))

	for _, doc := range r.docs {
		textLower := strings.ToLower(doc.Node.Text)
		matchCount := 0

		for _, word := range queryWords {
			if strings.Contains(textLower, word) {
				matchCount++
			}
		}

		if matchCount > 0 {
			score := float64(matchCount) / float64(len(queryWords))
			results = append(results, schema.NodeWithScore{
				Node:  doc.Node,
				Score: score,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(5, len(results))], nil
}

// MetadataRetriever filters by metadata.
type MetadataRetriever struct {
	*retriever.BaseRetriever
	docs []schema.NodeWithScore
}

func NewMetadataRetriever(docs []schema.NodeWithScore) *MetadataRetriever {
	return &MetadataRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		docs:          docs,
	}
}

func (r *MetadataRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Filter by metadata extracted from query
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, doc := range r.docs {
		score := doc.Score

		// Check for year mentions
		if strings.Contains(queryLower, "2024") {
			if year, ok := doc.Node.Metadata["year"]; ok && year == "2024" {
				score += 0.3
			}
		}

		// Check for category mentions
		if strings.Contains(queryLower, "compliance") {
			if cat, ok := doc.Node.Metadata["category"]; ok && cat == "compliance" {
				score += 0.3
			}
		}

		if score > doc.Score {
			results = append(results, schema.NodeWithScore{
				Node:  doc.Node,
				Score: score,
			})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	return results[:min(5, len(results))], nil
}

// CoarseRetriever retrieves many candidates.
type CoarseRetriever struct {
	*retriever.BaseRetriever
	docs []schema.NodeWithScore
	topK int
}

func NewCoarseRetriever(docs []schema.NodeWithScore, topK int) *CoarseRetriever {
	return &CoarseRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		docs:          docs,
		topK:          topK,
	}
}

func (r *CoarseRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	// Return top K by base score
	sorted := make([]schema.NodeWithScore, len(r.docs))
	copy(sorted, r.docs)

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})

	return sorted[:min(r.topK, len(sorted))], nil
}

// Helper functions

func filterByScore(results []schema.NodeWithScore, threshold float64) []schema.NodeWithScore {
	var filtered []schema.NodeWithScore
	for _, r := range results {
		if r.Score >= threshold {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

func rerankByKeywords(results []schema.NodeWithScore, query string) []schema.NodeWithScore {
	queryWords := strings.Fields(strings.ToLower(query))
	reranked := make([]schema.NodeWithScore, len(results))
	copy(reranked, results)

	for i := range reranked {
		textLower := strings.ToLower(reranked[i].Node.Text)
		bonus := 0.0
		for _, word := range queryWords {
			if len(word) > 4 && strings.Contains(textLower, word) {
				bonus += 0.05
			}
		}
		reranked[i].Score += bonus
	}

	sort.Slice(reranked, func(i, j int) bool {
		return reranked[i].Score > reranked[j].Score
	})

	return reranked
}

func mergeResults(resultSets ...[]schema.NodeWithScore) []schema.NodeWithScore {
	seen := make(map[string]schema.NodeWithScore)

	for _, results := range resultSets {
		for _, r := range results {
			if existing, ok := seen[r.Node.ID]; ok {
				// Keep higher score
				if r.Score > existing.Score {
					seen[r.Node.ID] = r
				}
			} else {
				seen[r.Node.ID] = r
			}
		}
	}

	var merged []schema.NodeWithScore
	for _, r := range seen {
		merged = append(merged, r)
	}

	sort.Slice(merged, func(i, j int) bool {
		return merged[i].Score > merged[j].Score
	})

	return merged[:min(5, len(merged))]
}

func classifyQuery(query string) string {
	queryLower := strings.ToLower(query)

	// Check for metadata indicators
	if strings.Contains(queryLower, "from 2024") || strings.Contains(queryLower, "about compliance") {
		return "metadata"
	}

	// Check for keyword-heavy queries
	keywordIndicators := []string{"what is", "rate limit", "api", "how to"}
	for _, ind := range keywordIndicators {
		if strings.Contains(queryLower, ind) {
			return "keyword"
		}
	}

	// Default to semantic
	return "semantic"
}

func fineRetrieve(candidates []schema.NodeWithScore, query string, topK int) []schema.NodeWithScore {
	queryWords := strings.Fields(strings.ToLower(query))
	refined := make([]schema.NodeWithScore, len(candidates))
	copy(refined, candidates)

	// Re-score based on detailed matching
	for i := range refined {
		textLower := strings.ToLower(refined[i].Node.Text)
		newScore := 0.0

		for _, word := range queryWords {
			if strings.Contains(textLower, word) {
				newScore += 0.15
			}
		}
		refined[i].Score = newScore
	}

	sort.Slice(refined, func(i, j int) bool {
		return refined[i].Score > refined[j].Score
	})

	return refined[:min(topK, len(refined))]
}

// Document collections

func getSemanticDocs() []schema.NodeWithScore {
	return createDocSet([]docData{
		{"sem-1", "Neural networks learn through backpropagation, adjusting weights based on error gradients.", 0.8},
		{"sem-2", "Machine learning models require training data to learn patterns and make predictions.", 0.75},
		{"sem-3", "Deep learning architectures like transformers have revolutionized NLP tasks.", 0.7},
		{"sem-4", "Model optimization techniques include learning rate scheduling and regularization.", 0.65},
	}, "semantic")
}

func getKeywordDocs() []schema.NodeWithScore {
	return createDocSet([]docData{
		{"kw-1", "API rate limit: 1000 requests per minute. Use exponential backoff for retries.", 0.8},
		{"kw-2", "Training a machine learning model requires labeled data and compute resources.", 0.75},
		{"kw-3", "Model optimization: batch size, learning rate, and epochs are key hyperparameters.", 0.7},
		{"kw-4", "Neural network training involves forward pass, loss calculation, and backpropagation.", 0.65},
	}, "keyword")
}

func getMetadataDocs() []schema.NodeWithScore {
	docs := []schema.NodeWithScore{
		{Node: schema.Node{ID: "meta-1", Text: "2024 Compliance Report: Updated data privacy regulations.", Type: schema.ObjectTypeText, Metadata: map[string]interface{}{"year": "2024", "category": "compliance"}}, Score: 0.8},
		{Node: schema.Node{ID: "meta-2", Text: "2023 Annual Review: Performance metrics and goals.", Type: schema.ObjectTypeText, Metadata: map[string]interface{}{"year": "2023", "category": "review"}}, Score: 0.75},
		{Node: schema.Node{ID: "meta-3", Text: "2024 Security Guidelines: Best practices for data protection.", Type: schema.ObjectTypeText, Metadata: map[string]interface{}{"year": "2024", "category": "security"}}, Score: 0.7},
		{Node: schema.Node{ID: "meta-4", Text: "2024 Compliance Checklist: Required documentation and audits.", Type: schema.ObjectTypeText, Metadata: map[string]interface{}{"year": "2024", "category": "compliance"}}, Score: 0.65},
	}
	return docs
}

func getAllDocs() []schema.NodeWithScore {
	var all []schema.NodeWithScore
	all = append(all, getSemanticDocs()...)
	all = append(all, getKeywordDocs()...)
	all = append(all, getMetadataDocs()...)
	return all
}

type docData struct {
	id    string
	text  string
	score float64
}

func createDocSet(docs []docData, source string) []schema.NodeWithScore {
	results := make([]schema.NodeWithScore, len(docs))
	for i, d := range docs {
		results[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   d.id,
				Text: d.text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source": source,
				},
			},
			Score: d.score,
		}
	}
	return results
}

func printResults(label string, results []schema.NodeWithScore) {
	fmt.Printf("%s Results (%d):\n", label, len(results))
	for i, r := range results {
		fmt.Printf("  %d. [%.3f] %s: %s\n", i+1, r.Score, r.Node.ID, truncate(r.Node.Text, 55))
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
