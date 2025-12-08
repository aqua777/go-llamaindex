// Package main demonstrates fusion retrieval strategies.
// This example corresponds to Python's low_level/fusion_retriever.ipynb
package main

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create mock retrievers with different "perspectives"
	fmt.Println("Creating mock retrievers...")

	// Retriever 1: Focuses on technical aspects
	techRetriever := NewMockRetriever("tech", getTechDocuments())

	// Retriever 2: Focuses on business aspects
	businessRetriever := NewMockRetriever("business", getBusinessDocuments())

	// Retriever 3: Focuses on practical examples
	examplesRetriever := NewMockRetriever("examples", getExampleDocuments())

	retrievers := []retriever.Retriever{techRetriever, businessRetriever, examplesRetriever}
	fmt.Printf("Created %d retrievers\n", len(retrievers))

	separator := strings.Repeat("=", 60)

	// Query to test
	query := schema.QueryBundle{QueryString: "How can AI improve business operations?"}
	fmt.Printf("\nQuery: %s\n", query.QueryString)

	// 2. Simple Fusion (max score for duplicates)
	fmt.Println("\n" + separator)
	fmt.Println("=== Simple Fusion ===")
	fmt.Println(separator + "\n")

	simpleFusion := retriever.NewFusionRetriever(
		retrievers,
		retriever.WithFusionMode(retriever.FusionModeSimple),
		retriever.WithSimilarityTopK(5),
	)

	simpleResults, err := simpleFusion.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Simple fusion failed: %v\n", err)
	} else {
		fmt.Println("Results (Simple Fusion - max score for duplicates):")
		printResults(simpleResults)
	}

	// 3. Reciprocal Rank Fusion (RRF)
	fmt.Println("\n" + separator)
	fmt.Println("=== Reciprocal Rank Fusion (RRF) ===")
	fmt.Println(separator + "\n")

	rrfFusion := retriever.NewFusionRetriever(
		retrievers,
		retriever.WithFusionMode(retriever.FusionModeReciprocalRank),
		retriever.WithSimilarityTopK(5),
	)

	rrfResults, err := rrfFusion.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("RRF fusion failed: %v\n", err)
	} else {
		fmt.Println("Results (RRF - combines rankings across retrievers):")
		printResults(rrfResults)
	}

	// 4. Relative Score Fusion
	fmt.Println("\n" + separator)
	fmt.Println("=== Relative Score Fusion ===")
	fmt.Println(separator + "\n")

	relativeScoreFusion := retriever.NewFusionRetriever(
		retrievers,
		retriever.WithFusionMode(retriever.FusionModeRelativeScore),
		retriever.WithSimilarityTopK(5),
	)

	relativeResults, err := relativeScoreFusion.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Relative score fusion failed: %v\n", err)
	} else {
		fmt.Println("Results (Relative Score - normalizes scores to [0,1]):")
		printResults(relativeResults)
	}

	// 5. Distance-Based Score Fusion
	fmt.Println("\n" + separator)
	fmt.Println("=== Distance-Based Score Fusion ===")
	fmt.Println(separator + "\n")

	distBasedFusion := retriever.NewFusionRetriever(
		retrievers,
		retriever.WithFusionMode(retriever.FusionModeDistBasedScore),
		retriever.WithSimilarityTopK(5),
	)

	distResults, err := distBasedFusion.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Distance-based fusion failed: %v\n", err)
	} else {
		fmt.Println("Results (Distance-Based - uses mean +/- 3 std dev):")
		printResults(distResults)
	}

	// 6. Weighted Fusion
	fmt.Println("\n" + separator)
	fmt.Println("=== Weighted Fusion ===")
	fmt.Println(separator + "\n")

	// Give more weight to technical retriever
	weightedFusion := retriever.NewFusionRetriever(
		retrievers,
		retriever.WithFusionMode(retriever.FusionModeRelativeScore),
		retriever.WithRetrieverWeights([]float64{0.5, 0.3, 0.2}), // tech: 50%, business: 30%, examples: 20%
		retriever.WithSimilarityTopK(5),
	)

	weightedResults, err := weightedFusion.Retrieve(ctx, query)
	if err != nil {
		fmt.Printf("Weighted fusion failed: %v\n", err)
	} else {
		fmt.Println("Results (Weighted - tech:50%, business:30%, examples:20%):")
		printResults(weightedResults)
	}

	// 7. Fusion Mode Comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Fusion Mode Comparison ===")
	fmt.Println(separator + "\n")

	fmt.Println("Fusion Modes:")
	fmt.Println()
	fmt.Println("1. Simple Fusion:")
	fmt.Println("   - Takes maximum score when same document appears in multiple retrievers")
	fmt.Println("   - Fast and simple")
	fmt.Println("   - Best for: Quick combination without score manipulation")
	fmt.Println()
	fmt.Println("2. Reciprocal Rank Fusion (RRF):")
	fmt.Println("   - Score = sum(1 / (k + rank)) across retrievers")
	fmt.Println("   - Robust to outliers and score scale differences")
	fmt.Println("   - Best for: Combining retrievers with different scoring scales")
	fmt.Println()
	fmt.Println("3. Relative Score Fusion:")
	fmt.Println("   - Normalizes scores to [0,1] using min-max scaling")
	fmt.Println("   - Applies retriever weights")
	fmt.Println("   - Best for: When scores are meaningful and comparable")
	fmt.Println()
	fmt.Println("4. Distance-Based Score Fusion:")
	fmt.Println("   - Uses mean +/- 3 standard deviations for normalization")
	fmt.Println("   - More robust to outliers than relative score")
	fmt.Println("   - Best for: Datasets with score outliers")

	fmt.Println("\n=== Fusion Retriever Demo Complete ===")
}

// MockRetriever simulates a retriever with predefined documents.
type MockRetriever struct {
	*retriever.BaseRetriever
	name      string
	documents []schema.NodeWithScore
}

// NewMockRetriever creates a new MockRetriever.
func NewMockRetriever(name string, docs []schema.NodeWithScore) *MockRetriever {
	return &MockRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		name:          name,
		documents:     docs,
	}
}

// Retrieve returns documents matching the query.
func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, doc := range m.documents {
		// Simple keyword matching for demo
		textLower := strings.ToLower(doc.Node.Text)
		score := 0.0

		keywords := []string{"ai", "business", "operations", "improve", "machine learning", "automation"}
		for _, kw := range keywords {
			if strings.Contains(queryLower, kw) && strings.Contains(textLower, kw) {
				score += 0.2
			}
		}

		if score > 0 {
			results = append(results, schema.NodeWithScore{
				Node:  doc.Node,
				Score: score * doc.Score, // Combine with base score
			})
		}
	}

	// If no matches, return top documents
	if len(results) == 0 && len(m.documents) > 0 {
		results = m.documents[:min(3, len(m.documents))]
	}

	return results, nil
}

// Document collections for different retrievers

func getTechDocuments() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{"tech-1", "AI and machine learning algorithms can automate repetitive tasks, reducing human error and improving efficiency in business operations.", 0.95},
		{"tech-2", "Natural Language Processing (NLP) enables businesses to analyze customer feedback at scale, extracting insights from unstructured text data.", 0.88},
		{"tech-3", "Computer vision systems can automate quality control in manufacturing, detecting defects faster than human inspectors.", 0.82},
		{"tech-4", "Predictive analytics uses historical data to forecast future trends, helping businesses make data-driven decisions.", 0.78},
	}

	results := make([]schema.NodeWithScore, len(docs))
	for i, d := range docs {
		results[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   d.id,
				Text: d.text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source": "tech",
				},
			},
			Score: d.score,
		}
	}
	return results
}

func getBusinessDocuments() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{"biz-1", "AI-powered automation can reduce operational costs by up to 30% while improving service quality and customer satisfaction.", 0.92},
		{"biz-2", "Businesses implementing AI see an average ROI of 250% within the first year of deployment.", 0.85},
		{"biz-3", "AI chatbots can handle 80% of routine customer inquiries, freeing human agents for complex issues.", 0.80},
		{"biz-4", "Supply chain optimization through AI can reduce inventory costs by 20-50%.", 0.75},
	}

	results := make([]schema.NodeWithScore, len(docs))
	for i, d := range docs {
		results[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   d.id,
				Text: d.text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source": "business",
				},
			},
			Score: d.score,
		}
	}
	return results
}

func getExampleDocuments() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{"ex-1", "Example: Amazon uses AI to optimize warehouse operations, reducing delivery times by 25%.", 0.90},
		{"ex-2", "Case Study: A retail company improved inventory accuracy from 65% to 95% using AI-powered demand forecasting.", 0.84},
		{"ex-3", "Example: Banks use AI fraud detection to prevent $billions in losses annually.", 0.79},
		{"ex-4", "Case Study: Healthcare providers reduced patient wait times by 40% using AI scheduling systems.", 0.73},
	}

	results := make([]schema.NodeWithScore, len(docs))
	for i, d := range docs {
		results[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   d.id,
				Text: d.text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"source": "examples",
				},
			},
			Score: d.score,
		}
	}
	return results
}

// printResults prints retrieval results.
func printResults(results []schema.NodeWithScore) {
	for i, r := range results {
		source := "unknown"
		if s, ok := r.Node.Metadata["source"]; ok {
			source = fmt.Sprintf("%v", s)
		}
		fmt.Printf("  %d. [Score: %.4f] [Source: %s] %s\n", i+1, r.Score, source, truncate(r.Node.Text, 70))
	}
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
