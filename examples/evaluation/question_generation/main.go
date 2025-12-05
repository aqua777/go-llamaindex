// Package main demonstrates question generation for RAG evaluation.
// This example corresponds to Python's evaluation/QuestionGeneration.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/questiongen"
	"github.com/aqua777/go-llamaindex/selector"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM for question generation
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Question Generation Demo ===")
	fmt.Println("\nLLM initialized for question generation")

	separator := strings.Repeat("=", 60)

	// 2. Create question generator
	fmt.Println("\n" + separator)
	fmt.Println("=== Basic Question Generation ===")
	fmt.Println(separator)

	questionGenerator := questiongen.NewLLMQuestionGenerator(llmInstance)

	// Define tools (knowledge sources)
	tools := []selector.ToolMetadata{
		{
			Name:        "company_financials",
			Description: "Provides information about company financial reports, revenue, and earnings",
		},
		{
			Name:        "product_catalog",
			Description: "Contains product information, specifications, and pricing",
		},
		{
			Name:        "customer_support",
			Description: "Handles customer inquiries, FAQs, and support tickets",
		},
	}

	fmt.Println("\nAvailable tools:")
	for _, tool := range tools {
		fmt.Printf("  - %s: %s\n", tool.Name, truncate(tool.Description, 50))
	}

	// Generate sub-questions for a complex query
	complexQuery := "What are the company's top-selling products and how did they contribute to last quarter's revenue?"

	fmt.Printf("\nComplex Query: %s\n", complexQuery)
	fmt.Println("\nGenerating sub-questions...")

	subQuestions, err := questionGenerator.Generate(ctx, tools, complexQuery)
	if err != nil {
		log.Printf("Question generation failed: %v", err)
	} else {
		fmt.Printf("\nGenerated %d sub-questions:\n", len(subQuestions))
		for i, sq := range subQuestions {
			fmt.Printf("  %d. [%s] %s\n", i+1, sq.ToolName, sq.SubQuestion)
		}
	}

	// 3. Multi-domain question decomposition
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-Domain Question Decomposition ===")
	fmt.Println(separator)

	// Define tools for different domains
	multiDomainTools := []selector.ToolMetadata{
		{
			Name:        "uber_10k",
			Description: "Provides information about Uber financials for year 2023",
		},
		{
			Name:        "lyft_10k",
			Description: "Provides information about Lyft financials for year 2023",
		},
		{
			Name:        "industry_analysis",
			Description: "Contains ride-sharing industry trends and market analysis",
		},
	}

	comparisonQuery := "Compare Uber and Lyft's revenue growth, market share, and profitability in 2023. What are the key industry trends affecting both companies?"

	fmt.Printf("\nComparison Query: %s\n", comparisonQuery)
	fmt.Println("\nGenerating sub-questions...")

	subQuestions, err = questionGenerator.Generate(ctx, multiDomainTools, comparisonQuery)
	if err != nil {
		log.Printf("Question generation failed: %v", err)
	} else {
		fmt.Printf("\nGenerated %d sub-questions:\n", len(subQuestions))
		for i, sq := range subQuestions {
			fmt.Printf("  %d. [%s] %s\n", i+1, sq.ToolName, sq.SubQuestion)
		}

		// Group by tool
		fmt.Println("\nGrouped by tool:")
		toolQuestions := make(map[string][]string)
		for _, sq := range subQuestions {
			toolQuestions[sq.ToolName] = append(toolQuestions[sq.ToolName], sq.SubQuestion)
		}
		for tool, questions := range toolQuestions {
			fmt.Printf("\n  %s:\n", tool)
			for _, q := range questions {
				fmt.Printf("    - %s\n", q)
			}
		}
	}

	// 4. Technical documentation queries
	fmt.Println("\n" + separator)
	fmt.Println("=== Technical Documentation Queries ===")
	fmt.Println(separator)

	techTools := []selector.ToolMetadata{
		{
			Name:        "api_docs",
			Description: "API documentation including endpoints, parameters, and examples",
		},
		{
			Name:        "architecture_docs",
			Description: "System architecture, design patterns, and infrastructure details",
		},
		{
			Name:        "troubleshooting_guide",
			Description: "Common issues, error codes, and debugging procedures",
		},
	}

	techQuery := "How do I authenticate with the API, what's the rate limit, and how do I handle 429 errors?"

	fmt.Printf("\nTechnical Query: %s\n", techQuery)
	fmt.Println("\nGenerating sub-questions...")

	subQuestions, err = questionGenerator.Generate(ctx, techTools, techQuery)
	if err != nil {
		log.Printf("Question generation failed: %v", err)
	} else {
		fmt.Printf("\nGenerated %d sub-questions:\n", len(subQuestions))
		for i, sq := range subQuestions {
			fmt.Printf("  %d. [%s] %s\n", i+1, sq.ToolName, sq.SubQuestion)
		}
	}

	// 5. Custom prompt template
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Prompt Template ===")
	fmt.Println(separator)

	customPrompt := `You are a question decomposition expert. Given a complex question and available tools, break it down into simpler sub-questions.

Tools available:
%s

User Question: %s

Generate sub-questions in JSON format:
` + "```json" + `
{
    "items": [
        {"sub_question": "...", "tool_name": "..."}
    ]
}
` + "```" + `
`

	customGenerator := questiongen.NewLLMQuestionGenerator(
		llmInstance,
		questiongen.WithQuestionGenPrompt(customPrompt),
	)

	simpleQuery := "What products does the company sell and what are their prices?"

	fmt.Printf("\nUsing custom prompt for: %s\n", simpleQuery)

	subQuestions, err = customGenerator.Generate(ctx, tools[:2], simpleQuery)
	if err != nil {
		log.Printf("Question generation failed: %v", err)
	} else {
		fmt.Printf("\nGenerated %d sub-questions:\n", len(subQuestions))
		for i, sq := range subQuestions {
			fmt.Printf("  %d. [%s] %s\n", i+1, sq.ToolName, sq.SubQuestion)
		}
	}

	// 6. Evaluation dataset generation
	fmt.Println("\n" + separator)
	fmt.Println("=== Generating Evaluation Dataset ===")
	fmt.Println(separator)

	// Generate questions for different topics to create an evaluation dataset
	topics := []struct {
		query string
		tools []selector.ToolMetadata
	}{
		{
			query: "What are the main features and pricing of the enterprise plan?",
			tools: []selector.ToolMetadata{
				{Name: "pricing_docs", Description: "Pricing information and plan comparisons"},
				{Name: "feature_docs", Description: "Product features and capabilities"},
			},
		},
		{
			query: "How do I set up SSO and what security certifications does the platform have?",
			tools: []selector.ToolMetadata{
				{Name: "security_docs", Description: "Security features and certifications"},
				{Name: "integration_docs", Description: "Integration guides and SSO setup"},
			},
		},
	}

	fmt.Println("\nGenerating evaluation questions for multiple topics:")

	var allQuestions []questiongen.SubQuestion
	for i, topic := range topics {
		fmt.Printf("\n  Topic %d: %s\n", i+1, truncate(topic.query, 50))

		subQuestions, err = questionGenerator.Generate(ctx, topic.tools, topic.query)
		if err != nil {
			log.Printf("    Generation failed: %v", err)
			continue
		}

		fmt.Printf("    Generated %d questions\n", len(subQuestions))
		allQuestions = append(allQuestions, subQuestions...)
	}

	fmt.Printf("\nTotal evaluation questions generated: %d\n", len(allQuestions))

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nQuestion Generation Use Cases:")
	fmt.Println("  1. Query Decomposition:")
	fmt.Println("     - Break complex queries into simpler sub-questions")
	fmt.Println("     - Route sub-questions to appropriate tools/indexes")
	fmt.Println()
	fmt.Println("  2. Evaluation Dataset Creation:")
	fmt.Println("     - Generate diverse questions from documents")
	fmt.Println("     - Create test cases for RAG system evaluation")
	fmt.Println()
	fmt.Println("  3. Sub-Question Query Engine:")
	fmt.Println("     - Power multi-step reasoning")
	fmt.Println("     - Enable cross-document queries")
	fmt.Println()
	fmt.Println("  4. Query Understanding:")
	fmt.Println("     - Identify information needs")
	fmt.Println("     - Map queries to knowledge sources")

	fmt.Println("\n=== Question Generation Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
