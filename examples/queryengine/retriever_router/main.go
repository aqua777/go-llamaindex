// Package main demonstrates the RouterRetriever for routing queries to different retrievers.
// This example corresponds to Python's query_engine/RetrieverRouterQueryEngine.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/selector"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Retriever Router Query Engine Demo ===")
	fmt.Println("LLM initialized")

	// 2. Create domain-specific retrievers
	// In production, these would be backed by actual vector stores
	scienceRetriever := NewMockRetriever("science", getScienceDocuments())
	historyRetriever := NewMockRetriever("history", getHistoryDocuments())
	techRetriever := NewMockRetriever("technology", getTechnologyDocuments())

	// 3. Create retriever tools with descriptions
	retrieverTools := []*retriever.RetrieverTool{
		retriever.NewRetrieverTool(
			scienceRetriever,
			"science_retriever",
			"Retrieves documents about scientific topics including physics, chemistry, biology, and astronomy. Use for questions about natural phenomena, scientific discoveries, and research.",
		),
		retriever.NewRetrieverTool(
			historyRetriever,
			"history_retriever",
			"Retrieves documents about historical events, figures, and periods. Use for questions about past events, civilizations, and historical analysis.",
		),
		retriever.NewRetrieverTool(
			techRetriever,
			"technology_retriever",
			"Retrieves documents about technology, computing, and engineering. Use for questions about software, hardware, and technological innovations.",
		),
	}

	fmt.Printf("\nCreated %d retriever tools:\n", len(retrieverTools))
	for _, tool := range retrieverTools {
		fmt.Printf("  - %s: %s\n", tool.Name, truncate(tool.Description, 60))
	}

	separator := strings.Repeat("=", 70)

	// 4. Create LLM-based selector for routing
	fmt.Println("\n" + separator)
	fmt.Println("=== Router Retriever with LLM Selector ===")
	fmt.Println(separator)

	llmSelector := NewLLMRetrieverSelector(llmInstance)

	routerRetriever := retriever.NewRouterRetriever(
		retrieverTools,
		retriever.WithSelector(llmSelector),
	)

	// 5. Create query engine with router retriever
	synth := synthesizer.NewCompactAndRefineSynthesizer(llmInstance)
	queryEngine := queryengine.NewRetrieverQueryEngine(routerRetriever, synth)

	// 6. Test queries for different domains
	testQueries := []struct {
		query    string
		expected string
	}{
		{"What is quantum entanglement?", "science"},
		{"Who was Julius Caesar and what did he accomplish?", "history"},
		{"How does machine learning work?", "technology"},
		{"What caused the fall of the Roman Empire?", "history"},
		{"Explain the theory of relativity", "science"},
		{"What is cloud computing?", "technology"},
	}

	for _, tc := range testQueries {
		fmt.Printf("\nQuery: %s\n", tc.query)
		fmt.Printf("Expected domain: %s\n", tc.expected)

		response, err := queryEngine.Query(ctx, tc.query)
		if err != nil {
			log.Printf("Query failed: %v\n", err)
			continue
		}

		fmt.Printf("Response: %s\n", truncate(response.Response, 200))
		if len(response.SourceNodes) > 0 {
			fmt.Printf("Sources: %d nodes\n", len(response.SourceNodes))
		}
	}

	// 7. Demonstrate multi-selector (selecting multiple retrievers)
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Retriever Selection ===")
	fmt.Println(separator)

	// Use SimpleSelector which selects all retrievers
	allSelector := &retriever.SimpleSelector{}
	multiRouterRetriever := retriever.NewRouterRetriever(
		retrieverTools,
		retriever.WithSelector(allSelector),
	)

	multiQueryEngine := queryengine.NewRetrieverQueryEngine(multiRouterRetriever, synth)

	crossDomainQuery := "How has technology influenced historical research and scientific discoveries?"
	fmt.Printf("\nCross-domain Query: %s\n", crossDomainQuery)

	response, err := multiQueryEngine.Query(ctx, crossDomainQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", truncate(response.Response, 300))
		fmt.Printf("Total sources: %d nodes from all retrievers\n", len(response.SourceNodes))
	}

	// 8. Direct retrieval comparison
	fmt.Println("\n" + separator)
	fmt.Println("=== Direct Retrieval Comparison ===")
	fmt.Println(separator)

	directQuery := "What is photosynthesis?"
	queryBundle := schema.QueryBundle{QueryString: directQuery}

	fmt.Printf("\nQuery: %s\n\n", directQuery)

	// Retrieve from each retriever directly
	for _, tool := range retrieverTools {
		nodes, err := tool.Retriever.Retrieve(ctx, queryBundle)
		if err != nil {
			log.Printf("Retrieval from %s failed: %v\n", tool.Name, err)
			continue
		}
		fmt.Printf("%s: %d nodes retrieved\n", tool.Name, len(nodes))
		for i, node := range nodes {
			if i >= 2 {
				fmt.Printf("  ... and %d more\n", len(nodes)-2)
				break
			}
			fmt.Printf("  - Score: %.2f, Content: %s\n", node.Score, truncate(node.Node.GetContent(schema.MetadataModeNone), 60))
		}
	}

	fmt.Println("\n=== Retriever Router Demo Complete ===")
}

// MockRetriever is a simple retriever that returns documents based on keyword matching.
type MockRetriever struct {
	*retriever.BaseRetriever
	domain    string
	documents []schema.Document
}

// NewMockRetriever creates a new MockRetriever.
func NewMockRetriever(domain string, documents []schema.Document) *MockRetriever {
	return &MockRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		domain:        domain,
		documents:     documents,
	}
}

// Retrieve returns documents that match the query.
func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, doc := range m.documents {
		// Simple keyword matching
		docLower := strings.ToLower(doc.Text)
		score := 0.0

		// Count matching words
		queryWords := strings.Fields(queryLower)
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(docLower, word) {
				score += 0.2
			}
		}

		if score > 0 {
			node := schema.NewTextNode(doc.Text)
			node.Metadata = doc.Metadata
			node.Metadata["domain"] = m.domain
			results = append(results, schema.NodeWithScore{
				Node:  *node,
				Score: score,
			})
		}
	}

	// Sort by score (simple bubble sort for demo)
	for i := 0; i < len(results); i++ {
		for j := i + 1; j < len(results); j++ {
			if results[j].Score > results[i].Score {
				results[i], results[j] = results[j], results[i]
			}
		}
	}

	// Limit to top 3
	if len(results) > 3 {
		results = results[:3]
	}

	return results, nil
}

// LLMRetrieverSelector uses an LLM to select the best retriever.
type LLMRetrieverSelector struct {
	llm      llm.LLM
	selector selector.Selector
}

// NewLLMRetrieverSelector creates a new LLMRetrieverSelector.
func NewLLMRetrieverSelector(llmModel llm.LLM) *LLMRetrieverSelector {
	return &LLMRetrieverSelector{
		llm:      llmModel,
		selector: selector.NewLLMSingleSelector(llmModel),
	}
}

// Select chooses the best retriever for the query.
func (s *LLMRetrieverSelector) Select(ctx context.Context, tools []*retriever.RetrieverTool, query schema.QueryBundle) (*retriever.SelectorResult, error) {
	// Convert retriever tools to selector metadata
	choices := make([]selector.ToolMetadata, len(tools))
	for i, tool := range tools {
		choices[i] = selector.ToolMetadata{
			Name:        tool.Name,
			Description: tool.Description,
		}
	}

	// Use the selector
	result, err := s.selector.Select(ctx, choices, query.QueryString)
	if err != nil {
		return nil, err
	}

	// Convert to retriever selector result
	indices := make([]int, len(result.Selections))
	reasons := make([]string, len(result.Selections))
	for i, sel := range result.Selections {
		indices[i] = sel.Index
		reasons[i] = sel.Reason
	}

	return &retriever.SelectorResult{
		Indices: indices,
		Reasons: reasons,
	}, nil
}

// Document collections

func getScienceDocuments() []schema.Document {
	return []schema.Document{
		{Text: "Quantum entanglement is a phenomenon where two particles become interconnected and the quantum state of one instantly influences the other, regardless of distance. Einstein called it 'spooky action at a distance'.", Metadata: map[string]interface{}{"topic": "physics"}},
		{Text: "Photosynthesis is the process by which plants convert sunlight, water, and carbon dioxide into glucose and oxygen. It occurs in the chloroplasts of plant cells.", Metadata: map[string]interface{}{"topic": "biology"}},
		{Text: "The theory of relativity, developed by Einstein, consists of special relativity (dealing with objects moving at constant speeds) and general relativity (describing gravity as curvature of spacetime).", Metadata: map[string]interface{}{"topic": "physics"}},
		{Text: "DNA (deoxyribonucleic acid) is the molecule that carries genetic information in living organisms. It has a double helix structure discovered by Watson and Crick.", Metadata: map[string]interface{}{"topic": "biology"}},
		{Text: "Black holes are regions of spacetime where gravity is so strong that nothing, not even light, can escape. They form when massive stars collapse at the end of their life cycle.", Metadata: map[string]interface{}{"topic": "astronomy"}},
	}
}

func getHistoryDocuments() []schema.Document {
	return []schema.Document{
		{Text: "Julius Caesar was a Roman general and statesman who played a critical role in transforming the Roman Republic into the Roman Empire. He was assassinated on the Ides of March, 44 BCE.", Metadata: map[string]interface{}{"era": "ancient"}},
		{Text: "The fall of the Roman Empire was caused by multiple factors including economic troubles, military overspending, government corruption, and invasions by barbarian tribes.", Metadata: map[string]interface{}{"era": "ancient"}},
		{Text: "The Renaissance was a cultural movement that began in Italy in the 14th century and spread throughout Europe. It marked a rebirth of interest in classical learning and values.", Metadata: map[string]interface{}{"era": "medieval"}},
		{Text: "World War II (1939-1945) was the deadliest conflict in human history. It involved most of the world's nations and resulted in an estimated 70-85 million fatalities.", Metadata: map[string]interface{}{"era": "modern"}},
		{Text: "The Industrial Revolution began in Britain in the late 18th century and transformed society from agrarian to industrial, introducing new manufacturing processes and technologies.", Metadata: map[string]interface{}{"era": "modern"}},
	}
}

func getTechnologyDocuments() []schema.Document {
	return []schema.Document{
		{Text: "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to identify patterns in data.", Metadata: map[string]interface{}{"field": "AI"}},
		{Text: "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software. Major providers include AWS, Azure, and Google Cloud.", Metadata: map[string]interface{}{"field": "infrastructure"}},
		{Text: "The internet was developed from ARPANET in the 1960s and became publicly available in the 1990s. It has revolutionized communication, commerce, and access to information.", Metadata: map[string]interface{}{"field": "networking"}},
		{Text: "Blockchain is a distributed ledger technology that records transactions across multiple computers. It provides transparency, security, and decentralization.", Metadata: map[string]interface{}{"field": "distributed systems"}},
		{Text: "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to perform computations. It has the potential to solve problems intractable for classical computers.", Metadata: map[string]interface{}{"field": "computing"}},
	}
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
