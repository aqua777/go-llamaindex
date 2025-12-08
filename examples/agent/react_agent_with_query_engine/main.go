// Package main demonstrates a ReAct agent with query engine integration.
// This example corresponds to Python's agent/react_agent_with_query_engine.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/agent"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
	"github.com/aqua777/go-llamaindex/tools"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== ReAct Agent with Query Engine Demo ===")
	fmt.Println("\nLLM initialized")

	separator := strings.Repeat("=", 70)

	// 2. Create mock query engines for different knowledge domains
	// In production, these would be backed by actual vector stores
	companyQE := NewMockQueryEngine(
		"company_knowledge",
		getCompanyKnowledge(),
		llmInstance,
	)

	productQE := NewMockQueryEngine(
		"product_knowledge",
		getProductKnowledge(),
		llmInstance,
	)

	policyQE := NewMockQueryEngine(
		"policy_knowledge",
		getPolicyKnowledge(),
		llmInstance,
	)

	// 3. Create query engine tools
	companyTool := tools.NewQueryEngineToolFromDefaults(
		companyQE,
		"company_info",
		"Useful for answering questions about company information, history, leadership, and organization structure.",
	)

	productTool := tools.NewQueryEngineToolFromDefaults(
		productQE,
		"product_info",
		"Useful for answering questions about products, features, pricing, and specifications.",
	)

	policyTool := tools.NewQueryEngineToolFromDefaults(
		policyQE,
		"policy_info",
		"Useful for answering questions about company policies, procedures, and guidelines.",
	)

	queryEngineTools := []tools.Tool{companyTool, productTool, policyTool}

	fmt.Printf("\nCreated %d query engine tools:\n", len(queryEngineTools))
	for _, tool := range queryEngineTools {
		meta := tool.Metadata()
		fmt.Printf("  - %s: %s\n", meta.Name, truncate(meta.Description, 60))
	}

	// 4. Create ReAct Agent with query engine tools
	fmt.Println("\n" + separator)
	fmt.Println("=== ReAct Agent with Knowledge Base ===")
	fmt.Println(separator)

	ragAgent := agent.NewReActAgentFromDefaults(
		llmInstance,
		queryEngineTools,
		agent.WithAgentVerbose(true),
		agent.WithAgentMaxIterations(5),
		agent.WithAgentSystemPrompt(`You are a helpful customer service agent for TechCorp.
You have access to tools that can query different knowledge bases:
- company_info: For company-related questions
- product_info: For product-related questions
- policy_info: For policy-related questions

Use the appropriate tool to find information before answering.
If you can't find the answer, say so honestly.`),
	)

	// Test queries
	queries := []string{
		"Who is the CEO of TechCorp?",
		"What are the main features of CloudAI?",
		"What is the return policy?",
		"How much does the Enterprise plan cost?",
	}

	for _, query := range queries {
		fmt.Printf("\nUser: %s\n", query)

		response, err := ragAgent.Chat(ctx, query)
		if err != nil {
			log.Printf("Agent error: %v\n", err)
			continue
		}

		fmt.Printf("Agent: %s\n", response.Response)
		if len(response.ToolCalls) > 0 {
			fmt.Printf("Tools used:\n")
			for _, tc := range response.ToolCalls {
				fmt.Printf("  - %s: %s\n", tc.ToolName, truncate(tc.ToolOutput.Content, 50))
			}
		}

		// Reset for next query
		ragAgent.Reset(ctx)
	}

	// 5. Multi-tool query
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Tool Query ===")
	fmt.Println(separator)

	complexQuery := "I want to know about TechCorp's leadership and what products they offer. Also, what's their refund policy?"
	fmt.Printf("\nUser: %s\n", complexQuery)

	response, err := ragAgent.Chat(ctx, complexQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", response.Response)
		fmt.Printf("\nTools used: %d\n", len(response.ToolCalls))
		for _, tc := range response.ToolCalls {
			fmt.Printf("  - %s\n", tc.ToolName)
		}
	}

	// 6. Agent with calculator + query engine
	fmt.Println("\n" + separator)
	fmt.Println("=== Combined Tools (Calculator + Knowledge) ===")
	fmt.Println(separator)

	// Add calculator tool
	calcTool, _ := tools.NewFunctionToolFromDefaults(
		func(expression string) (string, error) {
			// Simple calculator for demo
			return evaluateExpression(expression), nil
		},
		"calculator",
		"Useful for mathematical calculations. Input should be a math expression like '100 * 0.15' or '500 - 200'.",
	)

	combinedTools := append(queryEngineTools, calcTool)

	combinedAgent := agent.NewReActAgentFromDefaults(
		llmInstance,
		combinedTools,
		agent.WithAgentVerbose(true),
		agent.WithAgentSystemPrompt(`You are a helpful assistant with access to company knowledge and a calculator.
Use the knowledge tools to find information and the calculator for any math.`),
	)

	calcQuery := "What is the price of the Enterprise plan and how much would 5 licenses cost?"
	fmt.Printf("\nUser: %s\n", calcQuery)

	response, err = combinedAgent.Chat(ctx, calcQuery)
	if err != nil {
		log.Printf("Agent error: %v\n", err)
	} else {
		fmt.Printf("Agent: %s\n", response.Response)
	}

	fmt.Println("\n=== ReAct Agent with Query Engine Demo Complete ===")
}

// MockQueryEngine is a simple query engine for demonstration.
type MockQueryEngine struct {
	name      string
	knowledge map[string]string
	llm       llm.LLM
	synth     *synthesizer.SimpleSynthesizer
}

// NewMockQueryEngine creates a new mock query engine.
func NewMockQueryEngine(name string, knowledge map[string]string, llmModel llm.LLM) *MockQueryEngine {
	return &MockQueryEngine{
		name:      name,
		knowledge: knowledge,
		llm:       llmModel,
		synth:     synthesizer.NewSimpleSynthesizer(llmModel),
	}
}

// Query implements queryengine.QueryEngine.
func (m *MockQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	// Find relevant knowledge
	var relevantDocs []schema.NodeWithScore
	queryLower := strings.ToLower(query)

	for topic, content := range m.knowledge {
		// Simple keyword matching
		if strings.Contains(queryLower, strings.ToLower(topic)) ||
			strings.Contains(strings.ToLower(content), queryLower) {
			node := schema.NewTextNode(content)
			node.ID = topic
			node.Metadata["topic"] = topic
			relevantDocs = append(relevantDocs, schema.NodeWithScore{
				Node:  *node,
				Score: 0.9,
			})
		}
	}

	// If no exact match, return all knowledge
	if len(relevantDocs) == 0 {
		for topic, content := range m.knowledge {
			node := schema.NewTextNode(content)
			node.ID = topic
			node.Metadata["topic"] = topic
			relevantDocs = append(relevantDocs, schema.NodeWithScore{
				Node:  *node,
				Score: 0.5,
			})
		}
	}

	// Synthesize response
	synthResponse, err := m.synth.Synthesize(ctx, query, relevantDocs)
	if err != nil {
		return nil, err
	}

	return &synthesizer.Response{
		Response:    synthResponse.Response,
		SourceNodes: relevantDocs,
		Metadata:    map[string]interface{}{"engine": m.name},
	}, nil
}

// getCompanyKnowledge returns company-related knowledge.
func getCompanyKnowledge() map[string]string {
	return map[string]string{
		"CEO": "Alice Chen is the CEO of TechCorp. She joined in 2015 and became CEO in 2020. She previously worked at Google.",
		"CTO": "Bob Smith is the CTO of TechCorp. He has a PhD from MIT and leads the engineering team of 200+ engineers.",
		"company": "TechCorp is a technology company founded in 2010, headquartered in San Francisco. They specialize in AI and cloud computing.",
		"history": "TechCorp was founded in 2010 by a group of Stanford graduates. The company went public in 2022 on NASDAQ under ticker TECH.",
		"leadership": "TechCorp's leadership includes CEO Alice Chen, CTO Bob Smith, and CFO Carol Davis. The board has 7 members.",
	}
}

// getProductKnowledge returns product-related knowledge.
func getProductKnowledge() map[string]string {
	return map[string]string{
		"CloudAI": "CloudAI is TechCorp's flagship enterprise AI platform. Features include: ML model training, deployment, monitoring, and auto-scaling. Launched in 2018 with 500+ enterprise customers.",
		"SmartAssist": "SmartAssist is a customer service automation tool. It provides: chatbot creation, ticket routing, sentiment analysis, and analytics dashboard.",
		"pricing": "TechCorp offers three pricing tiers: Starter ($99/month), Professional ($299/month), and Enterprise ($999/month). Volume discounts available.",
		"features": "Key platform features: Real-time analytics, API access, custom integrations, 24/7 support, and 99.9% uptime SLA.",
		"Enterprise": "The Enterprise plan costs $999/month and includes: unlimited users, priority support, custom SLA, dedicated account manager, and on-premise deployment option.",
	}
}

// getPolicyKnowledge returns policy-related knowledge.
func getPolicyKnowledge() map[string]string {
	return map[string]string{
		"return": "TechCorp offers a 30-day money-back guarantee on all plans. Refunds are processed within 5-7 business days.",
		"refund": "Refund policy: Full refund within 30 days, 50% refund within 60 days, no refund after 60 days. Enterprise contracts have custom terms.",
		"support": "Support policy: Starter gets email support (48h response), Professional gets chat support (4h response), Enterprise gets phone support (1h response).",
		"privacy": "TechCorp is GDPR and SOC2 compliant. Customer data is encrypted at rest and in transit. Data retention is 90 days after account closure.",
		"SLA": "Service Level Agreement: 99.9% uptime guarantee. Credits provided for downtime: 10% for <99.9%, 25% for <99.5%, 50% for <99%.",
	}
}

// evaluateExpression is a simple expression evaluator for demo purposes.
func evaluateExpression(expr string) string {
	// This is a simplified calculator for demo
	// In production, use a proper expression parser
	expr = strings.TrimSpace(expr)

	// Handle simple operations
	if strings.Contains(expr, "*") {
		parts := strings.Split(expr, "*")
		if len(parts) == 2 {
			a := parseFloat(parts[0])
			b := parseFloat(parts[1])
			return fmt.Sprintf("%.2f", a*b)
		}
	}
	if strings.Contains(expr, "/") {
		parts := strings.Split(expr, "/")
		if len(parts) == 2 {
			a := parseFloat(parts[0])
			b := parseFloat(parts[1])
			if b != 0 {
				return fmt.Sprintf("%.2f", a/b)
			}
		}
	}
	if strings.Contains(expr, "+") {
		parts := strings.Split(expr, "+")
		if len(parts) == 2 {
			a := parseFloat(parts[0])
			b := parseFloat(parts[1])
			return fmt.Sprintf("%.2f", a+b)
		}
	}
	if strings.Contains(expr, "-") {
		parts := strings.Split(expr, "-")
		if len(parts) == 2 {
			a := parseFloat(parts[0])
			b := parseFloat(parts[1])
			return fmt.Sprintf("%.2f", a-b)
		}
	}

	return "Could not evaluate: " + expr
}

func parseFloat(s string) float64 {
	s = strings.TrimSpace(s)
	f := 0.0
	fmt.Sscanf(s, "%f", &f)
	return f
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
