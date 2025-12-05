// Package main demonstrates the RouterRetriever for query-based retriever selection.
// This example corresponds to Python's retrievers/router_retriever.ipynb
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

	fmt.Println("=== Router Retriever Demo ===")
	fmt.Println()
	fmt.Println("The RouterRetriever routes queries to specialized retrievers based on")
	fmt.Println("query classification, enabling domain-specific retrieval strategies.")
	fmt.Println()

	separator := strings.Repeat("=", 70)

	// 1. Create domain-specific retrievers
	fmt.Println(separator)
	fmt.Println("=== Creating Domain-Specific Retrievers ===")
	fmt.Println(separator)
	fmt.Println()

	// Technical documentation retriever
	techRetriever := NewDomainRetriever("technical", getTechnicalDocs())
	techTool := retriever.NewRetrieverTool(
		techRetriever,
		"technical_docs",
		"Use for technical questions about APIs, code, architecture, and implementation details.",
	)

	// Business documentation retriever
	bizRetriever := NewDomainRetriever("business", getBusinessDocs())
	bizTool := retriever.NewRetrieverTool(
		bizRetriever,
		"business_docs",
		"Use for business questions about pricing, policies, contracts, and company information.",
	)

	// Support documentation retriever
	supportRetriever := NewDomainRetriever("support", getSupportDocs())
	supportTool := retriever.NewRetrieverTool(
		supportRetriever,
		"support_docs",
		"Use for support questions about troubleshooting, FAQs, and how-to guides.",
	)

	tools := []*retriever.RetrieverTool{techTool, bizTool, supportTool}

	fmt.Println("Created 3 domain-specific retrievers:")
	for _, tool := range tools {
		fmt.Printf("  - %s: %s\n", tool.Name, truncate(tool.Description, 60))
	}
	fmt.Println()

	// 2. Test with SimpleSelector (selects all)
	fmt.Println(separator)
	fmt.Println("=== SimpleSelector (Select All) ===")
	fmt.Println(separator)
	fmt.Println()

	simpleRouter := retriever.NewRouterRetriever(
		tools,
		retriever.WithSelector(&retriever.SimpleSelector{}),
	)

	query1 := schema.QueryBundle{QueryString: "How do I authenticate with the API?"}
	fmt.Printf("Query: %s\n\n", query1.QueryString)

	results1, err := simpleRouter.Retrieve(ctx, query1)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Results (from all retrievers):")
		printResults(results1)
	}

	// 3. Test with SingleSelector (selects first)
	fmt.Println(separator)
	fmt.Println("=== SingleSelector (Select First) ===")
	fmt.Println(separator)
	fmt.Println()

	singleRouter := retriever.NewRouterRetriever(
		tools,
		retriever.WithSelector(&retriever.SingleSelector{}),
	)

	query2 := schema.QueryBundle{QueryString: "What is the pricing for enterprise plans?"}
	fmt.Printf("Query: %s\n\n", query2.QueryString)

	results2, err := singleRouter.Retrieve(ctx, query2)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		fmt.Println("Results (from first retriever only):")
		printResults(results2)
	}

	// 4. Test with custom KeywordSelector
	fmt.Println(separator)
	fmt.Println("=== KeywordSelector (Custom Routing) ===")
	fmt.Println(separator)
	fmt.Println()

	keywordRouter := retriever.NewRouterRetriever(
		tools,
		retriever.WithSelector(&KeywordSelector{}),
	)

	queries := []schema.QueryBundle{
		{QueryString: "How do I call the REST API endpoint?"},
		{QueryString: "What are the pricing tiers?"},
		{QueryString: "My application is crashing, how do I fix it?"},
		{QueryString: "Tell me about machine learning features"},
	}

	for _, q := range queries {
		fmt.Printf("Query: %s\n", q.QueryString)

		results, err := keywordRouter.Retrieve(ctx, q)
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}

		fmt.Printf("Routed to: %s\n", detectDomain(q.QueryString))
		printResults(results)
	}

	// 5. Explain routing strategies
	fmt.Println(separator)
	fmt.Println("=== Routing Strategies ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. SimpleSelector:")
	fmt.Println("   - Selects ALL retrievers")
	fmt.Println("   - Results are deduplicated by node ID")
	fmt.Println("   - Best for: Comprehensive search across all domains")
	fmt.Println()
	fmt.Println("2. SingleSelector:")
	fmt.Println("   - Selects only the FIRST retriever")
	fmt.Println("   - Useful for testing or default fallback")
	fmt.Println("   - Best for: Simple single-source retrieval")
	fmt.Println()
	fmt.Println("3. Custom Selector (e.g., KeywordSelector):")
	fmt.Println("   - Implements Selector interface")
	fmt.Println("   - Can use keywords, LLM classification, or embeddings")
	fmt.Println("   - Best for: Domain-specific routing logic")
	fmt.Println()
	fmt.Println("4. LLM-based Selector (not shown):")
	fmt.Println("   - Uses LLM to classify query intent")
	fmt.Println("   - Most flexible but adds latency")
	fmt.Println("   - Best for: Complex multi-domain systems")
	fmt.Println()

	// 6. Use cases
	fmt.Println(separator)
	fmt.Println("=== Use Cases ===")
	fmt.Println(separator)
	fmt.Println()

	fmt.Println("1. Multi-Department Knowledge Base:")
	fmt.Println("   Route to HR, Engineering, Sales, or Legal docs based on query")
	fmt.Println()
	fmt.Println("2. Multi-Language Documentation:")
	fmt.Println("   Route to language-specific retrievers based on detected language")
	fmt.Println()
	fmt.Println("3. Tiered Retrieval:")
	fmt.Println("   Route simple queries to fast retriever, complex to thorough one")
	fmt.Println()
	fmt.Println("4. Hybrid Search:")
	fmt.Println("   Route keyword queries to BM25, semantic queries to dense retriever")
	fmt.Println()

	fmt.Println("=== Router Retriever Demo Complete ===")
}

// KeywordSelector routes queries based on keyword matching.
type KeywordSelector struct{}

// Select chooses retrievers based on keywords in the query.
func (s *KeywordSelector) Select(ctx context.Context, tools []*retriever.RetrieverTool, query schema.QueryBundle) (*retriever.SelectorResult, error) {
	domain := detectDomain(query.QueryString)

	var indices []int
	var reasons []string

	for i, tool := range tools {
		if strings.Contains(tool.Name, domain) {
			indices = append(indices, i)
			reasons = append(reasons, fmt.Sprintf("matched domain: %s", domain))
		}
	}

	// Fallback to all if no match
	if len(indices) == 0 {
		for i := range tools {
			indices = append(indices, i)
			reasons = append(reasons, "fallback: no domain match")
		}
	}

	return &retriever.SelectorResult{
		Indices: indices,
		Reasons: reasons,
	}, nil
}

// detectDomain detects the domain based on keywords.
func detectDomain(query string) string {
	queryLower := strings.ToLower(query)

	techKeywords := []string{"api", "code", "function", "endpoint", "sdk", "library", "implementation"}
	bizKeywords := []string{"price", "pricing", "cost", "contract", "enterprise", "plan", "subscription"}
	supportKeywords := []string{"error", "crash", "fix", "troubleshoot", "help", "issue", "problem"}

	techScore := countKeywords(queryLower, techKeywords)
	bizScore := countKeywords(queryLower, bizKeywords)
	supportScore := countKeywords(queryLower, supportKeywords)

	if techScore >= bizScore && techScore >= supportScore && techScore > 0 {
		return "technical"
	}
	if bizScore >= techScore && bizScore >= supportScore && bizScore > 0 {
		return "business"
	}
	if supportScore > 0 {
		return "support"
	}
	return "all"
}

func countKeywords(text string, keywords []string) int {
	count := 0
	for _, kw := range keywords {
		if strings.Contains(text, kw) {
			count++
		}
	}
	return count
}

// DomainRetriever is a mock retriever for a specific domain.
type DomainRetriever struct {
	*retriever.BaseRetriever
	domain string
	docs   []schema.NodeWithScore
}

// NewDomainRetriever creates a new domain-specific retriever.
func NewDomainRetriever(domain string, docs []schema.NodeWithScore) *DomainRetriever {
	return &DomainRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		domain:        domain,
		docs:          docs,
	}
}

// Retrieve returns documents from this domain.
func (d *DomainRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for _, doc := range d.docs {
		textLower := strings.ToLower(doc.Node.Text)
		score := 0.0

		// Simple keyword overlap scoring
		queryWords := strings.Fields(queryLower)
		for _, word := range queryWords {
			if len(word) > 3 && strings.Contains(textLower, word) {
				score += 0.1
			}
		}

		if score > 0 {
			results = append(results, schema.NodeWithScore{
				Node:  doc.Node,
				Score: score + doc.Score*0.5,
			})
		}
	}

	// Return top results or all if few matches
	if len(results) == 0 {
		return d.docs[:min(2, len(d.docs))], nil
	}

	return results[:min(3, len(results))], nil
}

// Document collections

func getTechnicalDocs() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{"tech-1", "REST API Authentication: Use Bearer tokens in the Authorization header. Obtain tokens via OAuth2 flow or API key exchange.", 0.9},
		{"tech-2", "SDK Installation: Install via pip (Python), npm (Node.js), or go get (Go). Initialize with your API key.", 0.85},
		{"tech-3", "Rate Limiting: API calls are limited to 1000/minute. Use exponential backoff for retries.", 0.8},
		{"tech-4", "Webhook Integration: Configure webhooks in the dashboard. Events are sent as POST requests with JSON payload.", 0.75},
	}

	return createDocs(docs, "technical")
}

func getBusinessDocs() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{"biz-1", "Enterprise Pricing: Custom pricing based on usage. Contact sales for quotes. Volume discounts available.", 0.9},
		{"biz-2", "Subscription Plans: Free tier (1000 calls/month), Pro ($99/month, 50K calls), Enterprise (custom).", 0.85},
		{"biz-3", "SLA Terms: 99.9% uptime guarantee for Enterprise. 24/7 support included. Dedicated account manager.", 0.8},
		{"biz-4", "Contract Terms: Annual contracts with monthly billing. 30-day cancellation notice required.", 0.75},
	}

	return createDocs(docs, "business")
}

func getSupportDocs() []schema.NodeWithScore {
	docs := []struct {
		id    string
		text  string
		score float64
	}{
		{"sup-1", "Troubleshooting Connection Errors: Check network connectivity, verify API endpoint URL, ensure firewall allows outbound HTTPS.", 0.9},
		{"sup-2", "Common Error Codes: 401 (unauthorized), 429 (rate limited), 500 (server error). See docs for full list.", 0.85},
		{"sup-3", "Application Crashes: Enable debug logging, check memory usage, verify input validation. Contact support with logs.", 0.8},
		{"sup-4", "FAQ: How to reset API key, change billing info, add team members, export data.", 0.75},
	}

	return createDocs(docs, "support")
}

func createDocs(docs []struct {
	id    string
	text  string
	score float64
}, domain string) []schema.NodeWithScore {
	results := make([]schema.NodeWithScore, len(docs))
	for i, d := range docs {
		results[i] = schema.NodeWithScore{
			Node: schema.Node{
				ID:   d.id,
				Text: d.text,
				Type: schema.ObjectTypeText,
				Metadata: map[string]interface{}{
					"domain": domain,
				},
			},
			Score: d.score,
		}
	}
	return results
}

func printResults(results []schema.NodeWithScore) {
	for i, r := range results {
		domain := "unknown"
		if d, ok := r.Node.Metadata["domain"]; ok {
			domain = fmt.Sprintf("%v", d)
		}
		fmt.Printf("  %d. [Score: %.3f] [Domain: %s] %s\n", i+1, r.Score, domain, truncate(r.Node.Text, 60))
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
