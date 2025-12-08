// Package main demonstrates a RAG (Retrieval-Augmented Generation) workflow.
// This example corresponds to Python's workflow/rag.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/workflow"
)

// Define custom event types for RAG workflow
const (
	QueryEventType      workflow.EventType = "rag.query"
	RetrieveEventType   workflow.EventType = "rag.retrieve"
	SynthesizeEventType workflow.EventType = "rag.synthesize"
	ResponseEventType   workflow.EventType = "rag.response"
)

// Event data structures
type QueryData struct {
	Query string
}

type RetrieveData struct {
	Query    string
	Contexts []string
}

type SynthesizeData struct {
	Query    string
	Contexts []string
}

type ResponseData struct {
	Query    string
	Response string
	Sources  []string
}

// Event factories
var (
	QueryEvent      = workflow.NewEventFactory[QueryData](QueryEventType)
	RetrieveEvent   = workflow.NewEventFactory[RetrieveData](RetrieveEventType)
	SynthesizeEvent = workflow.NewEventFactory[SynthesizeData](SynthesizeEventType)
	ResponseEvent   = workflow.NewEventFactory[ResponseData](ResponseEventType)
)

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== RAG Workflow Demo ===")

	separator := strings.Repeat("=", 60)

	// 1. Build the RAG workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Building RAG Workflow ===")
	fmt.Println(separator)

	// Simulated document store
	documents := map[string][]string{
		"machine learning": {
			"Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
			"Deep learning uses neural networks with multiple layers to learn complex patterns.",
			"Supervised learning uses labeled data to train models for prediction tasks.",
		},
		"golang": {
			"Go is a statically typed, compiled programming language designed at Google.",
			"Go features garbage collection, structural typing, and CSP-style concurrency.",
			"Go is popular for building web services, cloud infrastructure, and CLI tools.",
		},
		"default": {
			"This is general information that may be relevant to your query.",
		},
	}

	ragWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("RAG Workflow"),
		workflow.WithWorkflowTimeout(30*time.Second),
	)

	// Step 1: Handle start event -> Query event
	ragWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(ctx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			query := data.Input.(string)
			fmt.Printf("\n[Start] Received query: %s\n", query)
			ctx.Set("query", query)
			return []workflow.Event{QueryEvent.With(QueryData{Query: query})}, nil
		},
	)

	// Step 2: Handle query event -> Retrieve event
	ragWorkflow.Handle(
		[]workflow.EventType{QueryEventType},
		func(ctx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := QueryEvent.Extract(event)
			fmt.Printf("[Query] Processing query: %s\n", data.Query)

			// Simulate retrieval
			var contexts []string
			queryLower := strings.ToLower(data.Query)
			for topic, docs := range documents {
				if strings.Contains(queryLower, topic) {
					contexts = append(contexts, docs...)
				}
			}
			if len(contexts) == 0 {
				contexts = documents["default"]
			}

			fmt.Printf("[Query] Retrieved %d contexts\n", len(contexts))
			return []workflow.Event{
				RetrieveEvent.With(RetrieveData{Query: data.Query, Contexts: contexts}),
			}, nil
		},
	)

	// Step 3: Handle retrieve event -> Synthesize event
	ragWorkflow.Handle(
		[]workflow.EventType{RetrieveEventType},
		func(ctx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := RetrieveEvent.Extract(event)
			fmt.Printf("[Retrieve] Preparing synthesis with %d contexts\n", len(data.Contexts))
			ctx.Set("contexts", data.Contexts)
			return []workflow.Event{
				SynthesizeEvent.With(SynthesizeData{Query: data.Query, Contexts: data.Contexts}),
			}, nil
		},
	)

	// Step 4: Handle synthesize event -> Response event (uses LLM)
	ragWorkflow.Handle(
		[]workflow.EventType{SynthesizeEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := SynthesizeEvent.Extract(event)
			fmt.Printf("[Synthesize] Generating response with LLM...\n")

			// Build prompt
			contextStr := strings.Join(data.Contexts, "\n\n")
			prompt := fmt.Sprintf(`Based on the following context, answer the question.

Context:
%s

Question: %s

Answer:`, contextStr, data.Query)

			// Call LLM
			response, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				return []workflow.Event{
					workflow.NewErrorEvent(err, "synthesize", event),
				}, nil
			}

			fmt.Printf("[Synthesize] Generated response\n")
			return []workflow.Event{
				ResponseEvent.With(ResponseData{
					Query:    data.Query,
					Response: response,
					Sources:  data.Contexts,
				}),
			}, nil
		},
	)

	// Step 5: Handle response event -> Stop event
	ragWorkflow.Handle(
		[]workflow.EventType{ResponseEventType},
		func(ctx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := ResponseEvent.Extract(event)
			fmt.Printf("[Response] Workflow complete\n")
			ctx.Set("response", data.Response)
			ctx.Set("sources", data.Sources)
			return []workflow.Event{
				workflow.NewStopEvent(data),
			}, nil
		},
	)

	fmt.Println("\nWorkflow steps:")
	fmt.Println("  1. Start -> Query")
	fmt.Println("  2. Query -> Retrieve (find relevant documents)")
	fmt.Println("  3. Retrieve -> Synthesize (prepare for LLM)")
	fmt.Println("  4. Synthesize -> Response (generate answer)")
	fmt.Println("  5. Response -> Stop")

	// 2. Run the workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Running RAG Workflow ===")
	fmt.Println(separator)

	queries := []string{
		"What is machine learning?",
		"Tell me about Golang programming",
	}

	for i, query := range queries {
		fmt.Printf("\n--- Query %d ---\n", i+1)

		result, err := ragWorkflow.Run(ctx, workflow.NewStartEvent(query))
		if err != nil {
			fmt.Printf("Workflow error: %v\n", err)
			continue
		}

		// Extract result
		if workflow.StopEvent.Include(result.FinalEvent) {
			stopData, _ := workflow.StopEvent.Extract(result.FinalEvent)
			if respData, ok := stopData.Result.(ResponseData); ok {
				fmt.Printf("\nQuery: %s\n", respData.Query)
				fmt.Printf("Response: %s\n", truncate(respData.Response, 200))
				fmt.Printf("Sources: %d documents\n", len(respData.Sources))
			}
		}

		fmt.Printf("Duration: %v\n", result.Duration)
	}

	// 3. Streaming workflow execution
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming Workflow Execution ===")
	fmt.Println(separator)

	fmt.Println("\nStreaming events as they occur...")

	stream := ragWorkflow.RunStream(ctx, workflow.NewStartEvent("What is deep learning?"))

	for event := range stream.Events() {
		fmt.Printf("  Event: %s\n", event.Type())
	}

	if err := stream.Err(); err != nil {
		fmt.Printf("Stream error: %v\n", err)
	}

	// 4. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nRAG Workflow Features:")
	fmt.Println("  - Event-driven architecture")
	fmt.Println("  - Clear separation of concerns (retrieve, synthesize)")
	fmt.Println("  - State management across steps")
	fmt.Println("  - Error handling with error events")
	fmt.Println("  - Streaming execution support")
	fmt.Println()
	fmt.Println("Workflow Components:")
	fmt.Println("  - Events: Typed messages between steps")
	fmt.Println("  - Handlers: Functions that process events")
	fmt.Println("  - Context: Shared state and event queue")
	fmt.Println("  - Result: Final output with state and duration")

	fmt.Println("\n=== RAG Workflow Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
