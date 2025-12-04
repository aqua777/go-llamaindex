// Package main demonstrates implementing a custom query engine.
// This example corresponds to Python's query_engine/custom_query_engine.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Custom Query Engine Demo ===")
	fmt.Println("\nLLM initialized")

	separator := strings.Repeat("=", 70)

	// 2. Create a simple mock retriever with sample documents
	documents := getSampleDocuments()
	mockRetriever := NewSimpleRetriever(documents)

	// 3. Demonstrate different custom query engine implementations
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Query Engine 1: RAG with Custom Prompt ===")
	fmt.Println(separator)

	// Custom prompt template
	customPrompt := `You are a helpful assistant that answers questions based on the provided context.
Always cite the source of your information when possible.
If the context doesn't contain relevant information, say so clearly.

Context:
{context_str}

Question: {query_str}

Answer (with citations):`

	customEngine1 := NewRAGQueryEngineWithCustomPrompt(
		mockRetriever,
		llmInstance,
		customPrompt,
	)

	testQuery := "What are the benefits of exercise?"
	fmt.Printf("\nQuery: %s\n", testQuery)

	response, err := customEngine1.Query(ctx, testQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Response)
	}

	// 4. Custom Query Engine 2: Query with preprocessing
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Query Engine 2: Query Preprocessing ===")
	fmt.Println(separator)

	preprocessingEngine := NewPreprocessingQueryEngine(
		mockRetriever,
		llmInstance,
	)

	// Test with a poorly formatted query
	messyQuery := "   WHAT ARE the BENEFITS of   exercise???   "
	fmt.Printf("\nOriginal Query: '%s'\n", messyQuery)

	response, err = preprocessingEngine.Query(ctx, messyQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Response)
	}

	// 5. Custom Query Engine 3: Multi-step reasoning
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Query Engine 3: Multi-Step Reasoning ===")
	fmt.Println(separator)

	multiStepEngine := NewMultiStepQueryEngine(
		mockRetriever,
		llmInstance,
	)

	complexQuery := "How do diet and exercise work together for health?"
	fmt.Printf("\nComplex Query: %s\n", complexQuery)

	response, err = multiStepEngine.Query(ctx, complexQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Response)
	}

	// 6. Custom Query Engine 4: Confidence-based response
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Query Engine 4: Confidence-Based Response ===")
	fmt.Println(separator)

	confidenceEngine := NewConfidenceQueryEngine(
		mockRetriever,
		llmInstance,
		0.5, // minimum confidence threshold
	)

	// Test with a query that should have good matches
	goodQuery := "What foods are good for health?"
	fmt.Printf("\nQuery (should match): %s\n", goodQuery)

	response, err = confidenceEngine.Query(ctx, goodQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Response)
	}

	// Test with a query that might not match well
	poorQuery := "What is quantum physics?"
	fmt.Printf("\nQuery (might not match): %s\n", poorQuery)

	response, err = confidenceEngine.Query(ctx, poorQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Response)
	}

	fmt.Println("\n=== Custom Query Engine Demo Complete ===")
}

// ============================================================================
// Custom Query Engine 1: RAG with Custom Prompt
// ============================================================================

// RAGQueryEngineWithCustomPrompt is a query engine with a custom prompt template.
type RAGQueryEngineWithCustomPrompt struct {
	retriever retriever.Retriever
	llm       llm.LLM
	prompt    *prompts.PromptTemplate
}

// NewRAGQueryEngineWithCustomPrompt creates a new RAGQueryEngineWithCustomPrompt.
func NewRAGQueryEngineWithCustomPrompt(ret retriever.Retriever, llmModel llm.LLM, promptTemplate string) *RAGQueryEngineWithCustomPrompt {
	return &RAGQueryEngineWithCustomPrompt{
		retriever: ret,
		llm:       llmModel,
		prompt:    prompts.NewPromptTemplate(promptTemplate, prompts.PromptTypeCustom),
	}
}

// Query implements the QueryEngine interface.
func (e *RAGQueryEngineWithCustomPrompt) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	// Retrieve relevant nodes
	queryBundle := schema.QueryBundle{QueryString: query}
	nodes, err := e.retriever.Retrieve(ctx, queryBundle)
	if err != nil {
		return nil, fmt.Errorf("retrieval failed: %w", err)
	}

	// Build context from nodes
	var contextParts []string
	for i, node := range nodes {
		contextParts = append(contextParts, fmt.Sprintf("[%d] %s", i+1, node.Node.GetContent(schema.MetadataModeNone)))
	}
	contextStr := strings.Join(contextParts, "\n\n")

	// Format prompt
	formattedPrompt := e.prompt.Format(map[string]string{
		"context_str": contextStr,
		"query_str":   query,
	})

	// Get LLM response
	responseText, err := e.llm.Complete(ctx, formattedPrompt)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %w", err)
	}

	return &synthesizer.Response{
		Response:    responseText,
		SourceNodes: nodes,
	}, nil
}

// ============================================================================
// Custom Query Engine 2: Query Preprocessing
// ============================================================================

// PreprocessingQueryEngine preprocesses queries before retrieval.
type PreprocessingQueryEngine struct {
	retriever retriever.Retriever
	llm       llm.LLM
	synth     synthesizer.Synthesizer
}

// NewPreprocessingQueryEngine creates a new PreprocessingQueryEngine.
func NewPreprocessingQueryEngine(ret retriever.Retriever, llmModel llm.LLM) *PreprocessingQueryEngine {
	return &PreprocessingQueryEngine{
		retriever: ret,
		llm:       llmModel,
		synth:     synthesizer.NewSimpleSynthesizer(llmModel),
	}
}

// Query implements the QueryEngine interface with preprocessing.
func (e *PreprocessingQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	// Preprocess the query
	processedQuery := e.preprocessQuery(query)
	fmt.Printf("Preprocessed Query: '%s'\n", processedQuery)

	// Retrieve with processed query
	queryBundle := schema.QueryBundle{QueryString: processedQuery}
	nodes, err := e.retriever.Retrieve(ctx, queryBundle)
	if err != nil {
		return nil, fmt.Errorf("retrieval failed: %w", err)
	}

	// Synthesize response
	return e.synth.Synthesize(ctx, processedQuery, nodes)
}

// preprocessQuery cleans and normalizes the query.
func (e *PreprocessingQueryEngine) preprocessQuery(query string) string {
	// Trim whitespace
	query = strings.TrimSpace(query)

	// Normalize case
	query = strings.ToLower(query)

	// Remove excessive punctuation
	query = strings.ReplaceAll(query, "???", "?")
	query = strings.ReplaceAll(query, "!!!", "!")

	// Normalize whitespace
	words := strings.Fields(query)
	query = strings.Join(words, " ")

	// Capitalize first letter
	if len(query) > 0 {
		query = strings.ToUpper(query[:1]) + query[1:]
	}

	return query
}

// ============================================================================
// Custom Query Engine 3: Multi-Step Reasoning
// ============================================================================

// MultiStepQueryEngine breaks down complex queries into steps.
type MultiStepQueryEngine struct {
	retriever retriever.Retriever
	llm       llm.LLM
}

// NewMultiStepQueryEngine creates a new MultiStepQueryEngine.
func NewMultiStepQueryEngine(ret retriever.Retriever, llmModel llm.LLM) *MultiStepQueryEngine {
	return &MultiStepQueryEngine{
		retriever: ret,
		llm:       llmModel,
	}
}

// Query implements the QueryEngine interface with multi-step reasoning.
func (e *MultiStepQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	// Step 1: Decompose the query
	subQueries := e.decomposeQuery(query)
	fmt.Printf("Decomposed into %d sub-queries:\n", len(subQueries))
	for i, sq := range subQueries {
		fmt.Printf("  %d. %s\n", i+1, sq)
	}

	// Step 2: Retrieve for each sub-query
	var allNodes []schema.NodeWithScore
	for _, sq := range subQueries {
		queryBundle := schema.QueryBundle{QueryString: sq}
		nodes, err := e.retriever.Retrieve(ctx, queryBundle)
		if err != nil {
			continue
		}
		allNodes = append(allNodes, nodes...)
	}

	// Step 3: Deduplicate nodes
	allNodes = deduplicateNodes(allNodes)
	fmt.Printf("Retrieved %d unique nodes\n", len(allNodes))

	// Step 4: Synthesize final response
	synth := synthesizer.NewSimpleSynthesizer(e.llm)
	return synth.Synthesize(ctx, query, allNodes)
}

// decomposeQuery breaks a complex query into simpler sub-queries.
func (e *MultiStepQueryEngine) decomposeQuery(query string) []string {
	// Simple heuristic: split on "and" or identify key concepts
	query = strings.ToLower(query)

	// Check for "and" to split
	if strings.Contains(query, " and ") {
		parts := strings.Split(query, " and ")
		var subQueries []string
		for _, part := range parts {
			part = strings.TrimSpace(part)
			if len(part) > 0 {
				// Make it a question if it isn't
				if !strings.HasSuffix(part, "?") {
					part = "What is " + part + "?"
				}
				subQueries = append(subQueries, part)
			}
		}
		if len(subQueries) > 0 {
			return subQueries
		}
	}

	// Default: return original query
	return []string{query}
}

// ============================================================================
// Custom Query Engine 4: Confidence-Based Response
// ============================================================================

// ConfidenceQueryEngine only responds if retrieval confidence is high enough.
type ConfidenceQueryEngine struct {
	retriever           retriever.Retriever
	llm                 llm.LLM
	confidenceThreshold float64
}

// NewConfidenceQueryEngine creates a new ConfidenceQueryEngine.
func NewConfidenceQueryEngine(ret retriever.Retriever, llmModel llm.LLM, threshold float64) *ConfidenceQueryEngine {
	return &ConfidenceQueryEngine{
		retriever:           ret,
		llm:                 llmModel,
		confidenceThreshold: threshold,
	}
}

// Query implements the QueryEngine interface with confidence checking.
func (e *ConfidenceQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	// Retrieve nodes
	queryBundle := schema.QueryBundle{QueryString: query}
	nodes, err := e.retriever.Retrieve(ctx, queryBundle)
	if err != nil {
		return nil, fmt.Errorf("retrieval failed: %w", err)
	}

	// Calculate average confidence
	avgConfidence := e.calculateAverageConfidence(nodes)
	fmt.Printf("Average retrieval confidence: %.2f (threshold: %.2f)\n", avgConfidence, e.confidenceThreshold)

	// Check confidence threshold
	if avgConfidence < e.confidenceThreshold {
		return &synthesizer.Response{
			Response: fmt.Sprintf("I don't have enough confident information to answer this question. (Confidence: %.2f)", avgConfidence),
			Metadata: map[string]interface{}{
				"confidence": avgConfidence,
				"threshold":  e.confidenceThreshold,
			},
		}, nil
	}

	// Synthesize response
	synth := synthesizer.NewSimpleSynthesizer(e.llm)
	response, err := synth.Synthesize(ctx, query, nodes)
	if err != nil {
		return nil, err
	}

	response.Metadata = map[string]interface{}{
		"confidence": avgConfidence,
	}

	return response, nil
}

// calculateAverageConfidence calculates the average score of retrieved nodes.
func (e *ConfidenceQueryEngine) calculateAverageConfidence(nodes []schema.NodeWithScore) float64 {
	if len(nodes) == 0 {
		return 0
	}

	var total float64
	for _, node := range nodes {
		total += node.Score
	}

	return total / float64(len(nodes))
}

// ============================================================================
// Helper Types and Functions
// ============================================================================

// SimpleRetriever is a basic retriever for demonstration.
type SimpleRetriever struct {
	*retriever.BaseRetriever
	documents []schema.Document
}

// NewSimpleRetriever creates a new SimpleRetriever.
func NewSimpleRetriever(documents []schema.Document) *SimpleRetriever {
	return &SimpleRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		documents:     documents,
	}
}

// Retrieve returns documents matching the query.
func (r *SimpleRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, doc := range r.documents {
		docLower := strings.ToLower(doc.Text)
		score := 0.0

		// Simple keyword matching
		queryWords := strings.Fields(queryLower)
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(docLower, word) {
				score += 0.25
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

	// Sort by score
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Limit results
	if len(results) > 5 {
		results = results[:5]
	}

	return results, nil
}

// deduplicateNodes removes duplicate nodes based on content hash.
func deduplicateNodes(nodes []schema.NodeWithScore) []schema.NodeWithScore {
	seen := make(map[string]bool)
	var unique []schema.NodeWithScore

	for _, node := range nodes {
		hash := node.Node.GenerateHash()
		if !seen[hash] {
			seen[hash] = true
			unique = append(unique, node)
		}
	}

	return unique
}

// getSampleDocuments returns sample documents for the demo.
func getSampleDocuments() []schema.Document {
	return []schema.Document{
		{Text: "Regular exercise has numerous health benefits including improved cardiovascular health, stronger muscles, better mental health, and increased energy levels. Aim for at least 150 minutes of moderate activity per week.", Metadata: map[string]interface{}{"topic": "exercise"}},
		{Text: "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Proper nutrition supports immune function, energy levels, and overall health.", Metadata: map[string]interface{}{"topic": "nutrition"}},
		{Text: "Sleep is essential for physical and mental health. Adults should aim for 7-9 hours of quality sleep per night. Poor sleep can affect mood, cognitive function, and immune system.", Metadata: map[string]interface{}{"topic": "sleep"}},
		{Text: "Stress management techniques include meditation, deep breathing, regular exercise, and maintaining social connections. Chronic stress can negatively impact both physical and mental health.", Metadata: map[string]interface{}{"topic": "stress"}},
		{Text: "Hydration is crucial for health. Water helps regulate body temperature, transport nutrients, and remove waste. Most adults should drink about 8 glasses of water daily.", Metadata: map[string]interface{}{"topic": "hydration"}},
		{Text: "Combining regular exercise with a healthy diet is the most effective approach to weight management and overall health. Exercise burns calories while proper nutrition provides essential nutrients.", Metadata: map[string]interface{}{"topic": "health"}},
	}
}
