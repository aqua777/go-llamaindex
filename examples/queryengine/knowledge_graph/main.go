// Package main demonstrates the Knowledge Graph Query Engine.
// This example corresponds to Python's query_engine/knowledge_graph_query_engine.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/graphstore"
	"github.com/aqua777/go-llamaindex/index"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Knowledge Graph Query Engine Demo ===")
	fmt.Println("\nLLM initialized")

	separator := strings.Repeat("=", 70)

	// 2. Create sample documents about a fictional company
	documents := getCompanyDocuments()
	fmt.Printf("\nLoaded %d documents about TechCorp\n", len(documents))

	// 3. Build Knowledge Graph Index
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Knowledge Graph Index ===")
	fmt.Println(separator)

	// Create KG index with LLM for triplet extraction
	kgIndex, err := index.NewKnowledgeGraphIndexFromDocuments(
		ctx,
		documents,
		index.WithKGIndexLLM(llmInstance),
		index.WithKGIndexMaxTripletsPerChunk(10),
	)
	if err != nil {
		log.Fatalf("Failed to create KG index: %v", err)
	}

	fmt.Println("Knowledge Graph Index created successfully")

	// 4. Explore the graph
	fmt.Println("\n" + separator)
	fmt.Println("=== Exploring the Knowledge Graph ===")
	fmt.Println(separator)

	// Get all keywords (entities) in the graph
	keywords := kgIndex.GetAllKeywords()
	fmt.Printf("\nEntities in the graph (%d total):\n", len(keywords))
	for i, kw := range keywords {
		if i >= 10 {
			fmt.Printf("  ... and %d more\n", len(keywords)-10)
			break
		}
		fmt.Printf("  - %s\n", kw)
	}

	// Get triplets from the graph store
	graphStore := kgIndex.GraphStore()
	fmt.Println("\nSample triplets from the graph:")
	displayGraphTriplets(ctx, graphStore, 5)

	// 5. Query the Knowledge Graph
	fmt.Println("\n" + separator)
	fmt.Println("=== Querying the Knowledge Graph ===")
	fmt.Println(separator)

	// Create query engine from the index
	queryEngine := kgIndex.AsQueryEngine(
		index.WithQueryEngineLLM(llmInstance),
	)

	// Test queries
	queries := []string{
		"Who is the CEO of TechCorp?",
		"What products does TechCorp make?",
		"Where is TechCorp headquartered?",
		"What is the relationship between TechCorp and CloudAI?",
	}

	for _, query := range queries {
		fmt.Printf("\nQuery: %s\n", query)

		response, err := queryEngine.Query(ctx, query)
		if err != nil {
			log.Printf("Query failed: %v\n", err)
			continue
		}

		fmt.Printf("Response: %s\n", truncate(response.Response, 200))
	}

	// 6. Demonstrate manual triplet insertion
	fmt.Println("\n" + separator)
	fmt.Println("=== Manual Triplet Insertion ===")
	fmt.Println(separator)

	// Add new triplets manually
	newTriplets := []graphstore.Triplet{
		{Subject: "TechCorp", Relation: "acquired", Object: "DataViz Inc"},
		{Subject: "DataViz Inc", Relation: "specializes in", Object: "Data Visualization"},
		{Subject: "TechCorp", Relation: "revenue", Object: "$5 billion"},
	}

	fmt.Println("\nAdding new triplets:")
	for _, triplet := range newTriplets {
		fmt.Printf("  (%s, %s, %s)\n", triplet.Subject, triplet.Relation, triplet.Object)
		err := kgIndex.UpsertTriplet(ctx, triplet, false)
		if err != nil {
			log.Printf("Failed to insert triplet: %v\n", err)
		}
	}

	// Query with new information
	fmt.Println("\nQuerying with new information:")
	newQuery := "What company did TechCorp acquire?"
	fmt.Printf("Query: %s\n", newQuery)

	response, err := queryEngine.Query(ctx, newQuery)
	if err != nil {
		log.Printf("Query failed: %v\n", err)
	} else {
		fmt.Printf("Response: %s\n", truncate(response.Response, 200))
	}

	// 7. Demonstrate different retrieval modes
	fmt.Println("\n" + separator)
	fmt.Println("=== Different Retrieval Modes ===")
	fmt.Println(separator)

	// Keyword mode (default)
	fmt.Println("\n--- Keyword Mode ---")
	keywordRetriever, err := kgIndex.AsRetrieverWithMode(index.KGRetrieverModeKeyword)
	if err != nil {
		log.Printf("Failed to create keyword retriever: %v\n", err)
	} else {
		testRetriever(ctx, keywordRetriever, "TechCorp products", llmInstance)
	}

	// 8. Show graph traversal
	fmt.Println("\n" + separator)
	fmt.Println("=== Graph Traversal Demo ===")
	fmt.Println(separator)

	fmt.Println("\nTraversing from 'TechCorp':")
	traverseFromEntity(ctx, graphStore, "TechCorp", 2)

	// 9. Custom triplet extraction demo
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Triplet Extraction ===")
	fmt.Println(separator)

	// Create a new index with custom extraction function
	customExtractor := func(text string) ([]graphstore.Triplet, error) {
		// Simple rule-based extraction for demo
		var triplets []graphstore.Triplet

		// Look for "X is Y" patterns
		text = strings.ToLower(text)
		if strings.Contains(text, " is ") {
			parts := strings.SplitN(text, " is ", 2)
			if len(parts) == 2 {
				subject := strings.TrimSpace(parts[0])
				object := strings.TrimSpace(parts[1])
				if len(subject) > 0 && len(object) > 0 && len(object) < 100 {
					triplets = append(triplets, graphstore.Triplet{
						Subject:  capitalize(subject),
						Relation: "is",
						Object:   capitalize(strings.Split(object, ".")[0]),
					})
				}
			}
		}

		return triplets, nil
	}

	simpleDoc := []schema.Document{
		{Text: "Go is a programming language. Go is developed by Google."},
	}

	customKGIndex, err := index.NewKnowledgeGraphIndexFromDocuments(
		ctx,
		simpleDoc,
		index.WithKGIndexTripletExtractFn(customExtractor),
	)
	if err != nil {
		log.Printf("Failed to create custom KG index: %v\n", err)
	} else {
		fmt.Println("Custom extraction results:")
		customKeywords := customKGIndex.GetAllKeywords()
		for _, kw := range customKeywords {
			fmt.Printf("  - %s\n", kw)
		}
	}

	fmt.Println("\n=== Knowledge Graph Query Engine Demo Complete ===")
}

// displayGraphTriplets displays sample triplets from the graph store.
func displayGraphTriplets(ctx context.Context, gs graphstore.GraphStore, limit int) {
	// Get triplets for common subjects
	subjects := []string{"TechCorp", "Alice Chen", "CloudAI", "SmartAssist"}

	count := 0
	for _, subj := range subjects {
		// Get returns [][]string where each element is [relation, object]
		relObjs, err := gs.Get(ctx, subj)
		if err != nil {
			continue
		}

		for _, relObj := range relObjs {
			if len(relObj) >= 2 {
				fmt.Printf("  (%s, %s, %s)\n", subj, relObj[0], relObj[1])
				count++
				if count >= limit {
					return
				}
			}
		}
	}
}

// traverseFromEntity traverses the graph from a starting entity.
func traverseFromEntity(ctx context.Context, gs graphstore.GraphStore, entity string, depth int) {
	visited := make(map[string]bool)
	traverseHelper(ctx, gs, entity, depth, 0, visited, "")
}

func traverseHelper(ctx context.Context, gs graphstore.GraphStore, entity string, maxDepth, currentDepth int, visited map[string]bool, indent string) {
	if currentDepth > maxDepth || visited[entity] {
		return
	}
	visited[entity] = true

	fmt.Printf("%s%s\n", indent, entity)

	// Get returns [][]string where each element is [relation, object]
	relObjs, err := gs.Get(ctx, entity)
	if err != nil {
		return
	}

	for _, relObj := range relObjs {
		if len(relObj) >= 2 {
			relation := relObj[0]
			object := relObj[1]
			if !visited[object] {
				fmt.Printf("%s  -[%s]-> %s\n", indent, relation, object)
				if currentDepth < maxDepth {
					traverseHelper(ctx, gs, object, maxDepth, currentDepth+1, visited, indent+"    ")
				}
			}
		}
	}
}

// testRetriever tests a retriever with a query.
func testRetriever(ctx context.Context, ret interface {
	Retrieve(context.Context, schema.QueryBundle) ([]schema.NodeWithScore, error)
}, query string, llmModel llm.LLM) {
	queryBundle := schema.QueryBundle{QueryString: query}
	nodes, err := ret.Retrieve(ctx, queryBundle)
	if err != nil {
		log.Printf("Retrieval failed: %v\n", err)
		return
	}

	fmt.Printf("Query: %s\n", query)
	fmt.Printf("Retrieved %d nodes\n", len(nodes))
	for i, node := range nodes {
		if i >= 3 {
			break
		}
		fmt.Printf("  %d. Score: %.2f, Content: %s\n", i+1, node.Score, truncate(node.Node.GetContent(schema.MetadataModeNone), 60))
	}

	// Synthesize response
	synth := synthesizer.NewSimpleSynthesizer(llmModel)
	response, err := synth.Synthesize(ctx, query, nodes)
	if err != nil {
		log.Printf("Synthesis failed: %v\n", err)
		return
	}
	fmt.Printf("Response: %s\n", truncate(response.Response, 150))
}

// getCompanyDocuments returns sample documents about a fictional company.
func getCompanyDocuments() []schema.Document {
	return []schema.Document{
		{
			Text:     "TechCorp is a technology company founded in 2010. The company is headquartered in San Francisco, California. TechCorp specializes in artificial intelligence and cloud computing solutions.",
			Metadata: map[string]interface{}{"type": "company_info"},
		},
		{
			Text:     "Alice Chen is the CEO of TechCorp. She joined the company in 2015 and became CEO in 2020. Alice Chen previously worked at Google as a senior engineer.",
			Metadata: map[string]interface{}{"type": "leadership"},
		},
		{
			Text:     "TechCorp's main products include CloudAI, an enterprise AI platform, and SmartAssist, a customer service automation tool. CloudAI was launched in 2018 and has over 500 enterprise customers.",
			Metadata: map[string]interface{}{"type": "products"},
		},
		{
			Text:     "TechCorp has partnerships with Microsoft and Amazon Web Services. The company uses AWS for its cloud infrastructure. Microsoft is a strategic partner for enterprise sales.",
			Metadata: map[string]interface{}{"type": "partnerships"},
		},
		{
			Text:     "The engineering team at TechCorp is led by Bob Smith, the CTO. Bob Smith has a PhD in Computer Science from MIT. The team consists of over 200 engineers working on AI and cloud technologies.",
			Metadata: map[string]interface{}{"type": "team"},
		},
		{
			Text:     "TechCorp went public in 2022 with an IPO price of $45 per share. The company is listed on NASDAQ under the ticker symbol TECH. Current market cap is approximately $10 billion.",
			Metadata: map[string]interface{}{"type": "financial"},
		},
	}
}

// capitalize capitalizes the first letter of a string.
func capitalize(s string) string {
	if len(s) == 0 {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
