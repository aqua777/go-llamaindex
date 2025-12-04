// Package main demonstrates a router workflow for query routing.
// This example corresponds to Python's workflow/router_query_engine.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/workflow"
)

// Define custom event types for router workflow
const (
	ClassifyEventType     workflow.EventType = "router.classify"
	RouteEventType        workflow.EventType = "router.route"
	TechQueryEventType    workflow.EventType = "router.tech_query"
	FinanceQueryEventType workflow.EventType = "router.finance_query"
	GeneralQueryEventType workflow.EventType = "router.general_query"
	AnswerEventType       workflow.EventType = "router.answer"
)

// Query categories
type QueryCategory string

const (
	CategoryTech    QueryCategory = "technology"
	CategoryFinance QueryCategory = "finance"
	CategoryGeneral QueryCategory = "general"
)

// Event data structures
type ClassifyData struct {
	Query string
}

type RouteData struct {
	Query    string
	Category QueryCategory
}

type QueryData struct {
	Query    string
	Category QueryCategory
}

type AnswerData struct {
	Query    string
	Category QueryCategory
	Answer   string
	Source   string
}

// Event factories
var (
	ClassifyEvent     = workflow.NewEventFactory[ClassifyData](ClassifyEventType)
	RouteEvent        = workflow.NewEventFactory[RouteData](RouteEventType)
	TechQueryEvent    = workflow.NewEventFactory[QueryData](TechQueryEventType)
	FinanceQueryEvent = workflow.NewEventFactory[QueryData](FinanceQueryEventType)
	GeneralQueryEvent = workflow.NewEventFactory[QueryData](GeneralQueryEventType)
	AnswerEvent       = workflow.NewEventFactory[AnswerData](AnswerEventType)
)

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Router Workflow Demo ===")
	fmt.Println("\nRoutes queries to specialized handlers based on classification.")

	separator := strings.Repeat("=", 60)

	// 1. Build the router workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Router Workflow ===")
	fmt.Println(separator)

	// Simulated knowledge bases
	techKB := map[string]string{
		"programming": "Programming involves writing code to create software applications.",
		"ai":          "Artificial Intelligence enables machines to simulate human intelligence.",
		"cloud":       "Cloud computing provides on-demand computing resources over the internet.",
	}

	financeKB := map[string]string{
		"stocks":    "Stocks represent ownership shares in a company.",
		"investing": "Investing involves allocating money with the expectation of returns.",
		"budget":    "A budget is a financial plan for managing income and expenses.",
	}

	routerWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Router Workflow"),
		workflow.WithWorkflowTimeout(30*time.Second),
	)

	// Step 1: Handle start event -> Classify query
	routerWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			query := data.Input.(string)
			fmt.Printf("\n[Start] Query: %s\n", query)
			wfCtx.Set("query", query)
			return []workflow.Event{ClassifyEvent.With(ClassifyData{Query: query})}, nil
		},
	)

	// Step 2: Handle classify event -> Determine category
	routerWorkflow.Handle(
		[]workflow.EventType{ClassifyEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := ClassifyEvent.Extract(event)
			fmt.Printf("[Classify] Analyzing query category...\n")

			// Use LLM to classify
			prompt := fmt.Sprintf(`Classify the following query into one of these categories: technology, finance, general.
Only respond with the category name.

Query: %s

Category:`, data.Query)

			response, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				response = "general"
			}

			// Parse category
			category := CategoryGeneral
			responseLower := strings.ToLower(strings.TrimSpace(response))
			if strings.Contains(responseLower, "tech") {
				category = CategoryTech
			} else if strings.Contains(responseLower, "finance") {
				category = CategoryFinance
			}

			fmt.Printf("[Classify] Category: %s\n", category)
			wfCtx.Set("category", string(category))

			return []workflow.Event{
				RouteEvent.With(RouteData{Query: data.Query, Category: category}),
			}, nil
		},
	)

	// Step 3: Handle route event -> Route to appropriate handler
	routerWorkflow.Handle(
		[]workflow.EventType{RouteEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := RouteEvent.Extract(event)
			fmt.Printf("[Route] Routing to %s handler\n", data.Category)

			queryData := QueryData{Query: data.Query, Category: data.Category}

			switch data.Category {
			case CategoryTech:
				return []workflow.Event{TechQueryEvent.With(queryData)}, nil
			case CategoryFinance:
				return []workflow.Event{FinanceQueryEvent.With(queryData)}, nil
			default:
				return []workflow.Event{GeneralQueryEvent.With(queryData)}, nil
			}
		},
	)

	// Step 4a: Handle tech query
	routerWorkflow.Handle(
		[]workflow.EventType{TechQueryEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := TechQueryEvent.Extract(event)
			fmt.Printf("[Tech] Processing technology query\n")

			// Search tech KB
			var context string
			queryLower := strings.ToLower(data.Query)
			for topic, info := range techKB {
				if strings.Contains(queryLower, topic) {
					context = info
					break
				}
			}

			// Generate answer
			prompt := fmt.Sprintf(`You are a technology expert. Answer the following question.
Context: %s

Question: %s

Answer:`, context, data.Query)

			answer, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				answer = "Unable to process technology query."
			}

			return []workflow.Event{
				AnswerEvent.With(AnswerData{
					Query:    data.Query,
					Category: data.Category,
					Answer:   answer,
					Source:   "Technology Knowledge Base",
				}),
			}, nil
		},
	)

	// Step 4b: Handle finance query
	routerWorkflow.Handle(
		[]workflow.EventType{FinanceQueryEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := FinanceQueryEvent.Extract(event)
			fmt.Printf("[Finance] Processing finance query\n")

			// Search finance KB
			var context string
			queryLower := strings.ToLower(data.Query)
			for topic, info := range financeKB {
				if strings.Contains(queryLower, topic) {
					context = info
					break
				}
			}

			// Generate answer
			prompt := fmt.Sprintf(`You are a finance expert. Answer the following question.
Context: %s

Question: %s

Answer:`, context, data.Query)

			answer, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				answer = "Unable to process finance query."
			}

			return []workflow.Event{
				AnswerEvent.With(AnswerData{
					Query:    data.Query,
					Category: data.Category,
					Answer:   answer,
					Source:   "Finance Knowledge Base",
				}),
			}, nil
		},
	)

	// Step 4c: Handle general query
	routerWorkflow.Handle(
		[]workflow.EventType{GeneralQueryEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := GeneralQueryEvent.Extract(event)
			fmt.Printf("[General] Processing general query\n")

			// Generate answer without specific context
			prompt := fmt.Sprintf(`Answer the following question:

Question: %s

Answer:`, data.Query)

			answer, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				answer = "Unable to process query."
			}

			return []workflow.Event{
				AnswerEvent.With(AnswerData{
					Query:    data.Query,
					Category: data.Category,
					Answer:   answer,
					Source:   "General Knowledge",
				}),
			}, nil
		},
	)

	// Step 5: Handle answer event -> Stop
	routerWorkflow.Handle(
		[]workflow.EventType{AnswerEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := AnswerEvent.Extract(event)
			fmt.Printf("[Answer] Generated response from %s\n", data.Source)
			wfCtx.Set("answer", data.Answer)
			wfCtx.Set("source", data.Source)
			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	fmt.Println("\nWorkflow structure:")
	fmt.Println("  1. Start -> Classify (determine query category)")
	fmt.Println("  2. Classify -> Route (select handler)")
	fmt.Println("  3. Route -> TechQuery | FinanceQuery | GeneralQuery")
	fmt.Println("  4. *Query -> Answer (generate response)")
	fmt.Println("  5. Answer -> Stop")

	// 2. Run the workflow with different queries
	fmt.Println("\n" + separator)
	fmt.Println("=== Running Router Workflow ===")
	fmt.Println(separator)

	queries := []string{
		"What is cloud computing?",
		"How do I start investing in stocks?",
		"What's the weather like today?",
		"Explain artificial intelligence",
	}

	for i, query := range queries {
		fmt.Printf("\n--- Query %d ---\n", i+1)

		result, err := routerWorkflow.Run(ctx, workflow.NewStartEvent(query))
		if err != nil {
			fmt.Printf("Workflow error: %v\n", err)
			continue
		}

		// Extract result
		if workflow.StopEvent.Include(result.FinalEvent) {
			stopData, _ := workflow.StopEvent.Extract(result.FinalEvent)
			if answerData, ok := stopData.Result.(AnswerData); ok {
				fmt.Printf("\nQuery: %s\n", answerData.Query)
				fmt.Printf("Category: %s\n", answerData.Category)
				fmt.Printf("Source: %s\n", answerData.Source)
				fmt.Printf("Answer: %s\n", truncate(answerData.Answer, 150))
			}
		}

		fmt.Printf("Duration: %v\n", result.Duration)
	}

	// 3. Visualize routing pattern
	fmt.Println("\n" + separator)
	fmt.Println("=== Router Pattern Visualization ===")
	fmt.Println(separator)

	fmt.Println(`
                    ┌─────────┐
                    │  Start  │
                    └────┬────┘
                         │
                         ▼
                    ┌──────────┐
                    │ Classify │
                    └────┬─────┘
                         │
                         ▼
                    ┌─────────┐
                    │  Route  │
                    └────┬────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌──────────┐    ┌─────────┐
    │  Tech   │    │ Finance  │    │ General │
    │ Handler │    │ Handler  │    │ Handler │
    └────┬────┘    └────┬─────┘    └────┬────┘
         │               │               │
         └───────────────┼───────────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ Answer  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │  Stop   │
                    └─────────┘
`)

	// 4. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nRouter Workflow Features:")
	fmt.Println("  - Query classification using LLM")
	fmt.Println("  - Dynamic routing to specialized handlers")
	fmt.Println("  - Domain-specific knowledge bases")
	fmt.Println("  - Unified response format")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Multi-domain chatbots")
	fmt.Println("  - Query routing in RAG systems")
	fmt.Println("  - Intent-based conversation handling")
	fmt.Println("  - Specialized expert systems")

	fmt.Println("\n=== Router Workflow Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
