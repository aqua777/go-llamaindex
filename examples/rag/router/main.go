// Package main demonstrates query routing across multiple query engines.
// This example corresponds to Python's low_level/router.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/selector"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("LLM initialized")

	// 2. Create mock query engines for different domains
	techEngine := NewMockQueryEngine("tech", getTechKnowledge())
	financeEngine := NewMockQueryEngine("finance", getFinanceKnowledge())
	healthEngine := NewMockQueryEngine("health", getHealthKnowledge())

	// 3. Wrap engines as tools with descriptions
	tools := []*queryengine.QueryEngineTool{
		queryengine.NewQueryEngineTool(
			techEngine,
			"tech_engine",
			"Useful for answering questions about technology, programming, AI, machine learning, and software development.",
		),
		queryengine.NewQueryEngineTool(
			financeEngine,
			"finance_engine",
			"Useful for answering questions about finance, investing, stocks, economics, and money management.",
		),
		queryengine.NewQueryEngineTool(
			healthEngine,
			"health_engine",
			"Useful for answering questions about health, fitness, nutrition, and medical topics.",
		),
	}

	fmt.Printf("Created %d query engine tools\n", len(tools))
	for _, tool := range tools {
		fmt.Printf("  - %s: %s\n", tool.Name, truncate(tool.Description, 60))
	}

	separator := strings.Repeat("=", 60)

	// 4. Create Router Query Engine with Single Selector
	fmt.Println("\n" + separator)
	fmt.Println("=== Router with Single Selector ===")
	fmt.Println(separator + "\n")

	// Create LLM-based selector
	singleSelector := selector.NewLLMSingleSelector(llmInstance)

	// Create adapter to use selector with router
	selectorAdapter := &SelectorAdapter{selector: singleSelector, tools: tools}

	routerEngine := queryengine.NewRouterQueryEngine(
		tools,
		queryengine.WithRouterSelector(selectorAdapter),
	)

	// Test queries
	queries := []string{
		"What is machine learning and how does it work?",
		"How should I diversify my investment portfolio?",
		"What are the benefits of regular exercise?",
	}

	for _, query := range queries {
		fmt.Printf("Query: %s\n", query)
		response, err := routerEngine.Query(ctx, query)
		if err != nil {
			log.Printf("Query failed: %v\n", err)
			continue
		}
		fmt.Printf("Response: %s\n\n", truncate(response.Response, 200))
	}

	// 5. Create Router with Multi Selector
	fmt.Println("\n" + separator)
	fmt.Println("=== Router with Multi Selector ===")
	fmt.Println(separator + "\n")

	multiSelector := selector.NewLLMMultiSelector(llmInstance, selector.WithMaxOutputs(2))
	multiSelectorAdapter := &SelectorAdapter{selector: multiSelector, tools: tools}

	multiRouterEngine := queryengine.NewRouterQueryEngine(
		tools,
		queryengine.WithRouterSelector(multiSelectorAdapter),
		queryengine.WithRouterSummarizer(synthesizer.NewSimpleSynthesizer(llmInstance)),
	)

	// Query that might benefit from multiple sources
	crossDomainQuery := "How can technology help improve personal health and financial wellness?"
	fmt.Printf("Cross-domain Query: %s\n", crossDomainQuery)

	response, err := multiRouterEngine.Query(ctx, crossDomainQuery)
	if err != nil {
		log.Printf("Multi-router query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", response.Response)
	}

	// 6. Demonstrate manual routing
	fmt.Println("\n" + separator)
	fmt.Println("=== Manual Query Routing ===")
	fmt.Println(separator + "\n")

	manualQueries := map[string]string{
		"tech":    "Explain the concept of neural networks.",
		"finance": "What is compound interest?",
		"health":  "How much water should I drink daily?",
	}

	for domain, query := range manualQueries {
		fmt.Printf("[%s] Query: %s\n", domain, query)
		var engine queryengine.QueryEngine
		switch domain {
		case "tech":
			engine = techEngine
		case "finance":
			engine = financeEngine
		case "health":
			engine = healthEngine
		}

		response, err := engine.Query(ctx, query)
		if err != nil {
			log.Printf("Query failed: %v\n", err)
			continue
		}
		fmt.Printf("Response: %s\n\n", truncate(response.Response, 150))
	}

	fmt.Println("\n=== Query Routing Demo Complete ===")
}

// MockQueryEngine is a simple query engine that returns predefined responses.
type MockQueryEngine struct {
	domain    string
	knowledge map[string]string
}

// NewMockQueryEngine creates a new MockQueryEngine.
func NewMockQueryEngine(domain string, knowledge map[string]string) *MockQueryEngine {
	return &MockQueryEngine{
		domain:    domain,
		knowledge: knowledge,
	}
}

// Query returns a response based on keyword matching.
func (m *MockQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	queryLower := strings.ToLower(query)

	// Find best matching response
	bestMatch := ""
	bestScore := 0

	for keywords, response := range m.knowledge {
		score := 0
		for _, keyword := range strings.Split(keywords, ",") {
			if strings.Contains(queryLower, strings.TrimSpace(keyword)) {
				score++
			}
		}
		if score > bestScore {
			bestScore = score
			bestMatch = response
		}
	}

	if bestMatch == "" {
		bestMatch = fmt.Sprintf("I'm the %s engine. I don't have specific information about that query, but I can help with %s-related questions.", m.domain, m.domain)
	}

	return &synthesizer.Response{
		Response: bestMatch,
		Metadata: map[string]interface{}{
			"source": m.domain,
		},
	}, nil
}

// SelectorAdapter adapts the selector.Selector interface to queryengine.QueryEngineSelector.
type SelectorAdapter struct {
	selector selector.Selector
	tools    []*queryengine.QueryEngineTool
}

// Select implements QueryEngineSelector.
func (a *SelectorAdapter) Select(ctx context.Context, tools []*queryengine.QueryEngineTool, query schema.QueryBundle) (*queryengine.SelectorResult, error) {
	// Convert tools to selector.ToolMetadata
	choices := make([]selector.ToolMetadata, len(tools))
	for i, tool := range tools {
		choices[i] = selector.ToolMetadata{
			Name:        tool.Name,
			Description: tool.Description,
		}
	}

	// Use selector
	result, err := a.selector.Select(ctx, choices, query.QueryString)
	if err != nil {
		return nil, err
	}

	// Convert result
	indices := make([]int, len(result.Selections))
	reasons := make([]string, len(result.Selections))
	for i, sel := range result.Selections {
		indices[i] = sel.Index
		reasons[i] = sel.Reason
	}

	return &queryengine.SelectorResult{
		Indices: indices,
		Reasons: reasons,
	}, nil
}

// Knowledge bases for mock engines

func getTechKnowledge() map[string]string {
	return map[string]string{
		"machine learning,ml,ai":       "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, learn patterns, and make predictions or decisions.",
		"neural network,deep learning": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers that process information using connectionist approaches to computation.",
		"programming,code,software":    "Programming is the process of creating instructions for computers to follow. It involves writing code in programming languages like Python, Go, Java, or JavaScript to build software applications.",
	}
}

func getFinanceKnowledge() map[string]string {
	return map[string]string{
		"invest,portfolio,diversify": "Portfolio diversification involves spreading investments across various asset classes (stocks, bonds, real estate) to reduce risk. A well-diversified portfolio can help protect against market volatility.",
		"compound interest,savings":  "Compound interest is interest calculated on both the initial principal and accumulated interest. It's a powerful wealth-building tool - the earlier you start saving, the more your money can grow.",
		"stock,market,trading":       "The stock market is where shares of publicly traded companies are bought and sold. Investors can profit from price appreciation and dividends, but should be aware of market risks.",
	}
}

func getHealthKnowledge() map[string]string {
	return map[string]string{
		"exercise,fitness,workout": "Regular exercise provides numerous benefits including improved cardiovascular health, stronger muscles and bones, better mental health, and increased energy levels. Aim for at least 150 minutes of moderate activity per week.",
		"nutrition,diet,food":      "A balanced diet includes fruits, vegetables, whole grains, lean proteins, and healthy fats. Proper nutrition supports immune function, energy levels, and overall health.",
		"water,hydration,drink":    "Adequate hydration is essential for health. Most adults should aim for 8-10 glasses (about 2 liters) of water daily, though needs vary based on activity level, climate, and individual factors.",
	}
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
