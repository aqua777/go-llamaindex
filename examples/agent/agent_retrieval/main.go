// Package main demonstrates an agent with retriever tool integration.
// This example corresponds to Python's agent/openai_agent_retrieval.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"

	"github.com/aqua777/go-llamaindex/agent"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/tools"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Agent with Retrieval Demo ===")
	fmt.Println("\nLLM initialized")

	separator := strings.Repeat("=", 70)

	// 2. Create document collections
	techDocs := getTechDocuments()
	hrDocs := getHRDocuments()
	financeDocs := getFinanceDocuments()

	fmt.Printf("\nDocument collections:\n")
	fmt.Printf("  - Tech docs: %d documents\n", len(techDocs))
	fmt.Printf("  - HR docs: %d documents\n", len(hrDocs))
	fmt.Printf("  - Finance docs: %d documents\n", len(financeDocs))

	// 3. Create retrievers for each collection
	techRetriever := NewSimpleRetriever("tech", techDocs, 3)
	hrRetriever := NewSimpleRetriever("hr", hrDocs, 3)
	financeRetriever := NewSimpleRetriever("finance", financeDocs, 3)

	// 4. Create retriever tools
	techTool := tools.NewRetrieverToolFromDefaults(
		techRetriever,
		"search_tech_docs",
		"Search technical documentation including API guides, architecture docs, and troubleshooting guides. Use for technical questions.",
	)

	hrTool := tools.NewRetrieverToolFromDefaults(
		hrRetriever,
		"search_hr_docs",
		"Search HR documentation including policies, benefits, and employee handbook. Use for HR-related questions.",
	)

	financeTool := tools.NewRetrieverToolFromDefaults(
		financeRetriever,
		"search_finance_docs",
		"Search financial documentation including expense policies, budgets, and procurement guides. Use for finance-related questions.",
	)

	retrieverTools := []tools.Tool{techTool, hrTool, financeTool}

	fmt.Printf("\nCreated %d retriever tools:\n", len(retrieverTools))
	for _, tool := range retrieverTools {
		meta := tool.Metadata()
		fmt.Printf("  - %s\n", meta.Name)
	}

	// 5. Create agent with retriever tools
	fmt.Println("\n" + separator)
	fmt.Println("=== Agent with Document Retrieval ===")
	fmt.Println(separator)

	retrievalAgent := agent.NewFunctionCallingReActAgent(
		agent.WithAgentLLM(llmInstance),
		agent.WithAgentTools(retrieverTools),
		agent.WithAgentVerbose(true),
		agent.WithAgentMaxIterations(5),
		agent.WithAgentSystemPrompt(`You are a helpful company assistant with access to various document collections.
Use the appropriate search tool to find relevant information:
- search_tech_docs: For technical questions
- search_hr_docs: For HR and policy questions
- search_finance_docs: For finance and expense questions

Always cite the source of your information.`),
	)

	// Test queries
	queries := []string{
		"How do I authenticate with the API?",
		"What is the vacation policy?",
		"What's the process for expense reimbursement?",
		"How do I set up the development environment?",
	}

	for _, query := range queries {
		fmt.Printf("\nUser: %s\n", query)

		response, err := retrievalAgent.Chat(ctx, query)
		if err != nil {
			log.Printf("Agent error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", truncate(response.Response, 300))
		if len(response.ToolCalls) > 0 {
			fmt.Printf("Documents retrieved from: %s\n", response.ToolCalls[0].ToolName)
		}

		// Reset for next query
		retrievalAgent.Reset(ctx)
	}

	// 6. Multi-retriever query
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Retriever Query ===")
	fmt.Println(separator)

	complexQuery := "I'm a new employee. What do I need to know about setting up my dev environment and what benefits do I get?"
	fmt.Printf("\nUser: %s\n", complexQuery)

	response, err := retrievalAgent.Chat(ctx, complexQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", truncate(response.Response, 400))
		fmt.Printf("\nRetrievers used: %d\n", len(response.ToolCalls))
		for _, tc := range response.ToolCalls {
			fmt.Printf("  - %s\n", tc.ToolName)
		}
	}

	// 7. Agent with combined retriever and function tools
	fmt.Println("\n" + separator)
	fmt.Println("=== Combined Retriever + Function Tools ===")
	fmt.Println(separator)

	// Add a calculator tool
	calcTool, _ := tools.NewFunctionToolFromDefaults(
		func(a, b float64, operation string) (string, error) {
			var result float64
			switch strings.ToLower(operation) {
			case "add", "+":
				result = a + b
			case "subtract", "-":
				result = a - b
			case "multiply", "*":
				result = a * b
			case "divide", "/":
				if b == 0 {
					return "Error: division by zero", nil
				}
				result = a / b
			default:
				return fmt.Sprintf("Unknown operation: %s", operation), nil
			}
			return fmt.Sprintf("%.2f %s %.2f = %.2f", a, operation, b, result), nil
		},
		"calculator",
		"Perform basic math operations. Input: two numbers and operation (add, subtract, multiply, divide).",
		tools.WithFunctionToolParameters(map[string]interface{}{
			"type": "object",
			"properties": map[string]interface{}{
				"a":         map[string]interface{}{"type": "number", "description": "First number"},
				"b":         map[string]interface{}{"type": "number", "description": "Second number"},
				"operation": map[string]interface{}{"type": "string", "description": "Operation: add, subtract, multiply, divide"},
			},
			"required": []string{"a", "b", "operation"},
		}),
	)

	combinedTools := append(retrieverTools, calcTool)

	combinedAgent := agent.NewFunctionCallingReActAgent(
		agent.WithAgentLLM(llmInstance),
		agent.WithAgentTools(combinedTools),
		agent.WithAgentVerbose(true),
		agent.WithAgentSystemPrompt(`You are a helpful assistant with access to company documents and a calculator.
Search documents for information and use the calculator for any math.`),
	)

	calcQuery := "What's the daily expense limit for meals, and if I have 5 days of travel, what's my total meal budget?"
	fmt.Printf("\nUser: %s\n", calcQuery)

	response, err = combinedAgent.Chat(ctx, calcQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", response.Response)
	}

	// 8. Demonstrate return_direct behavior
	fmt.Println("\n" + separator)
	fmt.Println("=== Return Direct Tool ===")
	fmt.Println(separator)

	// Create a tool that returns directly without further processing
	directTool := tools.NewRetrieverToolFromDefaults(
		techRetriever,
		"quick_search",
		"Quick search that returns results directly.",
	)
	// Note: In a real scenario, you'd set ReturnDirect on the tool metadata

	directAgent := agent.NewFunctionCallingReActAgent(
		agent.WithAgentLLM(llmInstance),
		agent.WithAgentTools([]tools.Tool{directTool}),
	)

	directQuery := "Find API authentication docs"
	fmt.Printf("\nUser: %s\n", directQuery)

	response, err = directAgent.Chat(ctx, directQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", truncate(response.Response, 200))
	}

	fmt.Println("\n=== Agent with Retrieval Demo Complete ===")
}

// SimpleRetriever is a basic keyword-based retriever for demonstration.
type SimpleRetriever struct {
	name     string
	docs     []schema.Document
	topK     int
}

// NewSimpleRetriever creates a new simple retriever.
func NewSimpleRetriever(name string, docs []schema.Document, topK int) *SimpleRetriever {
	return &SimpleRetriever{
		name: name,
		docs: docs,
		topK: topK,
	}
}

// Retrieve implements retriever.Retriever.
func (r *SimpleRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryTerms := strings.Fields(strings.ToLower(query.QueryString))

	type scoredDoc struct {
		doc   schema.Document
		score float64
	}

	var scored []scoredDoc

	for _, doc := range r.docs {
		docText := strings.ToLower(doc.Text)
		score := 0.0

		for _, term := range queryTerms {
			if strings.Contains(docText, term) {
				score += 1.0
				// Bonus for title match
				if title, ok := doc.Metadata["title"].(string); ok {
					if strings.Contains(strings.ToLower(title), term) {
						score += 0.5
					}
				}
			}
		}

		if score > 0 {
			// Normalize by query length
			score = score / float64(len(queryTerms))
			scored = append(scored, scoredDoc{doc: doc, score: score})
		}
	}

	// Sort by score descending
	sort.Slice(scored, func(i, j int) bool {
		return scored[i].score > scored[j].score
	})

	// Take top K
	results := make([]schema.NodeWithScore, 0, r.topK)
	for i := 0; i < len(scored) && i < r.topK; i++ {
		node := schema.NewTextNode(scored[i].doc.Text)
		node.ID = fmt.Sprintf("%s_%d", r.name, i)
		for k, v := range scored[i].doc.Metadata {
			node.Metadata[k] = v
		}
		results = append(results, schema.NodeWithScore{
			Node:  *node,
			Score: scored[i].score,
		})
	}

	return results, nil
}

// Ensure SimpleRetriever implements Retriever interface
var _ retriever.Retriever = (*SimpleRetriever)(nil)

// getTechDocuments returns technical documentation.
func getTechDocuments() []schema.Document {
	return []schema.Document{
		{
			Text: "API Authentication: All API requests require authentication using Bearer tokens. " +
				"To get a token, make a POST request to /auth/token with your client_id and client_secret. " +
				"Tokens expire after 1 hour and should be refreshed using the refresh_token.",
			Metadata: map[string]interface{}{"title": "API Authentication Guide", "category": "api"},
		},
		{
			Text: "Development Environment Setup: 1. Install Go 1.21+. 2. Clone the repository. " +
				"3. Run 'make setup' to install dependencies. 4. Copy .env.example to .env and configure. " +
				"5. Run 'make dev' to start the development server.",
			Metadata: map[string]interface{}{"title": "Dev Environment Setup", "category": "setup"},
		},
		{
			Text: "API Rate Limits: Free tier: 100 requests/minute. Pro tier: 1000 requests/minute. " +
				"Enterprise: Custom limits. Rate limit headers are included in all responses. " +
				"Exceeding limits returns HTTP 429.",
			Metadata: map[string]interface{}{"title": "API Rate Limits", "category": "api"},
		},
		{
			Text: "Error Handling: All errors return JSON with 'error' and 'message' fields. " +
				"Common errors: 400 (Bad Request), 401 (Unauthorized), 403 (Forbidden), 404 (Not Found), " +
				"500 (Internal Server Error). Always check the error message for details.",
			Metadata: map[string]interface{}{"title": "Error Handling Guide", "category": "api"},
		},
		{
			Text: "Database Architecture: We use PostgreSQL for primary data and Redis for caching. " +
				"All database migrations are in the /migrations folder. Run 'make migrate' to apply. " +
				"Connection pooling is configured with max 100 connections.",
			Metadata: map[string]interface{}{"title": "Database Architecture", "category": "architecture"},
		},
	}
}

// getHRDocuments returns HR documentation.
func getHRDocuments() []schema.Document {
	return []schema.Document{
		{
			Text: "Vacation Policy: Full-time employees receive 20 days of paid vacation per year. " +
				"Vacation accrues monthly at 1.67 days/month. Unused vacation can be carried over up to 5 days. " +
				"Vacation requests should be submitted at least 2 weeks in advance.",
			Metadata: map[string]interface{}{"title": "Vacation Policy", "category": "benefits"},
		},
		{
			Text: "Health Benefits: Company provides comprehensive health insurance including medical, dental, and vision. " +
				"Coverage begins on the first day of employment. Family coverage is available at subsidized rates. " +
				"Annual wellness benefit of $500 for gym memberships or fitness equipment.",
			Metadata: map[string]interface{}{"title": "Health Benefits", "category": "benefits"},
		},
		{
			Text: "Remote Work Policy: Employees may work remotely up to 3 days per week. " +
				"Core hours are 10am-3pm in your local timezone. Home office stipend of $500 is provided. " +
				"VPN must be used when accessing company systems remotely.",
			Metadata: map[string]interface{}{"title": "Remote Work Policy", "category": "policy"},
		},
		{
			Text: "Performance Reviews: Annual performance reviews are conducted in December. " +
				"Mid-year check-ins occur in June. Reviews include self-assessment, manager feedback, and peer feedback. " +
				"Compensation adjustments are based on performance ratings.",
			Metadata: map[string]interface{}{"title": "Performance Reviews", "category": "policy"},
		},
		{
			Text: "Onboarding: New employees complete a 2-week onboarding program. " +
				"Week 1 covers company culture, tools, and processes. Week 2 is team-specific training. " +
				"Each new hire is assigned a buddy for their first 90 days.",
			Metadata: map[string]interface{}{"title": "Onboarding Guide", "category": "onboarding"},
		},
	}
}

// getFinanceDocuments returns finance documentation.
func getFinanceDocuments() []schema.Document {
	return []schema.Document{
		{
			Text: "Expense Reimbursement: Submit expenses within 30 days of incurring them. " +
				"Use the expense portal to upload receipts. Reimbursements are processed bi-weekly. " +
				"Direct deposit is available for faster processing.",
			Metadata: map[string]interface{}{"title": "Expense Reimbursement", "category": "expenses"},
		},
		{
			Text: "Travel Policy: Book travel through the company portal for best rates. " +
				"Economy class for flights under 6 hours, business class for longer flights. " +
				"Daily meal allowance: $75. Hotel limit: $200/night in most cities, $300 in high-cost cities.",
			Metadata: map[string]interface{}{"title": "Travel Policy", "category": "travel"},
		},
		{
			Text: "Procurement Process: Purchases under $500 can be made with manager approval. " +
				"$500-$5000 requires department head approval. Over $5000 requires VP approval. " +
				"All software purchases must go through IT review.",
			Metadata: map[string]interface{}{"title": "Procurement Process", "category": "procurement"},
		},
		{
			Text: "Budget Planning: Annual budgets are set in Q4 for the following year. " +
				"Quarterly budget reviews occur in months 3, 6, 9. Budget transfers between categories " +
				"require finance approval. Unused budget does not roll over.",
			Metadata: map[string]interface{}{"title": "Budget Planning", "category": "budget"},
		},
		{
			Text: "Corporate Card Policy: Corporate cards are available for employees with frequent expenses. " +
				"Monthly limit is $5000. All charges must be reconciled within 5 business days. " +
				"Personal charges on corporate cards are prohibited.",
			Metadata: map[string]interface{}{"title": "Corporate Card Policy", "category": "expenses"},
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

// Unused but kept for potential future use
var _ = math.Sqrt
