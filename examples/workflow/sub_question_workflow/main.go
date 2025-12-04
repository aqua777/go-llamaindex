// Package main demonstrates a sub-question decomposition workflow.
// This example corresponds to Python's workflow/sub_question_query_engine.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/questiongen"
	"github.com/aqua777/go-llamaindex/selector"
	"github.com/aqua777/go-llamaindex/workflow"
)

// Define custom event types for sub-question workflow
const (
	DecomposeEventType   workflow.EventType = "subq.decompose"
	SubQuestionEventType workflow.EventType = "subq.sub_question"
	SubAnswerEventType   workflow.EventType = "subq.sub_answer"
	SynthesizeEventType  workflow.EventType = "subq.synthesize"
	FinalAnswerEventType workflow.EventType = "subq.final_answer"
)

// Event data structures
type DecomposeData struct {
	Query        string
	SubQuestions []questiongen.SubQuestion
}

type SubQuestionData struct {
	Index       int
	SubQuestion string
	ToolName    string
	Total       int
}

type SubAnswerData struct {
	Index       int
	SubQuestion string
	Answer      string
	ToolName    string
}

type SynthesizeData struct {
	OriginalQuery string
	SubAnswers    []SubAnswerData
}

type FinalAnswerData struct {
	Query      string
	Answer     string
	SubAnswers []SubAnswerData
}

// Event factories
var (
	DecomposeEvent   = workflow.NewEventFactory[DecomposeData](DecomposeEventType)
	SubQuestionEvent = workflow.NewEventFactory[SubQuestionData](SubQuestionEventType)
	SubAnswerEvent   = workflow.NewEventFactory[SubAnswerData](SubAnswerEventType)
	SynthesizeEvent  = workflow.NewEventFactory[SynthesizeData](SynthesizeEventType)
	FinalAnswerEvent = workflow.NewEventFactory[FinalAnswerData](FinalAnswerEventType)
)

func main() {
	ctx := context.Background()

	// Create LLM and question generator
	llmInstance := llm.NewOpenAILLM("", "", "")
	questionGenerator := questiongen.NewLLMQuestionGenerator(llmInstance)

	fmt.Println("=== Sub-Question Workflow Demo ===")
	fmt.Println("\nDecomposes complex queries into sub-questions for better answers.")

	separator := strings.Repeat("=", 60)

	// 1. Define available tools/knowledge sources
	tools := []selector.ToolMetadata{
		{
			Name:        "uber_financials",
			Description: "Contains Uber's financial reports, revenue, and earnings data",
		},
		{
			Name:        "lyft_financials",
			Description: "Contains Lyft's financial reports, revenue, and earnings data",
		},
		{
			Name:        "industry_analysis",
			Description: "Contains ride-sharing industry trends and market analysis",
		},
	}

	// Simulated knowledge bases for each tool
	knowledgeBases := map[string]string{
		"uber_financials":   "Uber reported $31.8 billion in revenue for 2023, with a net income of $1.9 billion. The company has 131 million monthly active users.",
		"lyft_financials":   "Lyft reported $4.4 billion in revenue for 2023. The company has 22 million active riders and operates primarily in the US and Canada.",
		"industry_analysis": "The ride-sharing market is expected to grow at 15% CAGR through 2028. Key trends include autonomous vehicles, electric fleets, and subscription models.",
	}

	// 2. Build the sub-question workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Sub-Question Workflow ===")
	fmt.Println(separator)

	// Track collected answers
	var collectedAnswers []SubAnswerData

	subqWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Sub-Question Workflow"),
		workflow.WithWorkflowTimeout(60*time.Second),
	)

	// Step 1: Handle start event -> Decompose into sub-questions
	subqWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			query := data.Input.(string)
			fmt.Printf("\n[Start] Complex query: %s\n", query)

			wfCtx.Set("original_query", query)
			collectedAnswers = nil // Reset

			// Generate sub-questions
			fmt.Printf("[Decompose] Generating sub-questions...\n")
			subQuestions, err := questionGenerator.Generate(ctx, tools, query)
			if err != nil {
				// Fallback to single question
				subQuestions = []questiongen.SubQuestion{
					{SubQuestion: query, ToolName: "industry_analysis"},
				}
			}

			fmt.Printf("[Decompose] Generated %d sub-questions\n", len(subQuestions))
			wfCtx.Set("total_subquestions", len(subQuestions))

			return []workflow.Event{
				DecomposeEvent.With(DecomposeData{Query: query, SubQuestions: subQuestions}),
			}, nil
		},
	)

	// Step 2: Handle decompose event -> Create sub-question events
	subqWorkflow.Handle(
		[]workflow.EventType{DecomposeEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := DecomposeEvent.Extract(event)

			var events []workflow.Event
			for i, sq := range data.SubQuestions {
				fmt.Printf("[SubQ %d] %s (tool: %s)\n", i+1, sq.SubQuestion, sq.ToolName)
				events = append(events, SubQuestionEvent.With(SubQuestionData{
					Index:       i,
					SubQuestion: sq.SubQuestion,
					ToolName:    sq.ToolName,
					Total:       len(data.SubQuestions),
				}))
			}

			return events, nil
		},
	)

	// Step 3: Handle sub-question event -> Answer using appropriate tool
	subqWorkflow.Handle(
		[]workflow.EventType{SubQuestionEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := SubQuestionEvent.Extract(event)
			fmt.Printf("[Answer %d] Processing with %s...\n", data.Index+1, data.ToolName)

			// Get context from knowledge base
			context := knowledgeBases[data.ToolName]
			if context == "" {
				context = "No specific information available."
			}

			// Generate answer
			prompt := fmt.Sprintf(`Based on the following context, answer the question concisely.

Context: %s

Question: %s

Answer:`, context, data.SubQuestion)

			answer, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				answer = fmt.Sprintf("Unable to answer: %v", err)
			}

			return []workflow.Event{
				SubAnswerEvent.With(SubAnswerData{
					Index:       data.Index,
					SubQuestion: data.SubQuestion,
					Answer:      answer,
					ToolName:    data.ToolName,
				}),
			}, nil
		},
	)

	// Step 4: Handle sub-answer event -> Collect and check if all done
	subqWorkflow.Handle(
		[]workflow.EventType{SubAnswerEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := SubAnswerEvent.Extract(event)

			collectedAnswers = append(collectedAnswers, data)
			total, _ := wfCtx.GetInt("total_subquestions")

			fmt.Printf("[Collect] Received answer %d/%d\n", len(collectedAnswers), total)

			if len(collectedAnswers) >= total {
				originalQuery, _ := wfCtx.GetString("original_query")
				fmt.Printf("[Collect] All sub-questions answered, synthesizing...\n")

				return []workflow.Event{
					SynthesizeEvent.With(SynthesizeData{
						OriginalQuery: originalQuery,
						SubAnswers:    collectedAnswers,
					}),
				}, nil
			}

			return nil, nil
		},
	)

	// Step 5: Handle synthesize event -> Combine sub-answers
	subqWorkflow.Handle(
		[]workflow.EventType{SynthesizeEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := SynthesizeEvent.Extract(event)
			fmt.Printf("[Synthesize] Combining %d sub-answers...\n", len(data.SubAnswers))

			// Build context from sub-answers
			var subAnswerText strings.Builder
			for _, sa := range data.SubAnswers {
				subAnswerText.WriteString(fmt.Sprintf("Q: %s\nA: %s\n\n", sa.SubQuestion, sa.Answer))
			}

			// Synthesize final answer
			prompt := fmt.Sprintf(`Based on the following sub-questions and answers, provide a comprehensive answer to the original question.

Original Question: %s

Sub-questions and Answers:
%s

Comprehensive Answer:`, data.OriginalQuery, subAnswerText.String())

			finalAnswer, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				finalAnswer = "Unable to synthesize final answer."
			}

			return []workflow.Event{
				FinalAnswerEvent.With(FinalAnswerData{
					Query:      data.OriginalQuery,
					Answer:     finalAnswer,
					SubAnswers: data.SubAnswers,
				}),
			}, nil
		},
	)

	// Step 6: Handle final answer event -> Stop
	subqWorkflow.Handle(
		[]workflow.EventType{FinalAnswerEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := FinalAnswerEvent.Extract(event)
			fmt.Printf("[Final] Workflow complete\n")
			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	fmt.Println("\nWorkflow structure:")
	fmt.Println("  1. Start -> Decompose (generate sub-questions)")
	fmt.Println("  2. Decompose -> SubQuestion[] (fan-out)")
	fmt.Println("  3. SubQuestion -> SubAnswer (answer each)")
	fmt.Println("  4. SubAnswer -> Synthesize (when all done)")
	fmt.Println("  5. Synthesize -> FinalAnswer")
	fmt.Println("  6. FinalAnswer -> Stop")

	// 3. Run the workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Running Sub-Question Workflow ===")
	fmt.Println(separator)

	complexQueries := []string{
		"Compare Uber and Lyft's revenue and market position in 2023",
		"What are the key trends in the ride-sharing industry?",
	}

	for i, query := range complexQueries {
		fmt.Printf("\n--- Complex Query %d ---\n", i+1)

		result, err := subqWorkflow.Run(ctx, workflow.NewStartEvent(query))
		if err != nil {
			fmt.Printf("Workflow error: %v\n", err)
			continue
		}

		// Extract result
		if workflow.StopEvent.Include(result.FinalEvent) {
			stopData, _ := workflow.StopEvent.Extract(result.FinalEvent)
			if finalData, ok := stopData.Result.(FinalAnswerData); ok {
				fmt.Printf("\nOriginal Query: %s\n", finalData.Query)
				fmt.Printf("\nSub-questions answered:\n")
				for _, sa := range finalData.SubAnswers {
					fmt.Printf("  - [%s] %s\n", sa.ToolName, truncate(sa.SubQuestion, 40))
				}
				fmt.Printf("\nFinal Answer:\n%s\n", truncate(finalData.Answer, 300))
			}
		}

		fmt.Printf("\nDuration: %v\n", result.Duration)
	}

	// 4. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nSub-Question Workflow Features:")
	fmt.Println("  - Automatic query decomposition")
	fmt.Println("  - Tool/source routing for sub-questions")
	fmt.Println("  - Parallel sub-question processing")
	fmt.Println("  - Answer synthesis from multiple sources")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Complex multi-part questions")
	fmt.Println("  - Cross-document queries")
	fmt.Println("  - Comparative analysis")
	fmt.Println("  - Research questions requiring multiple sources")

	fmt.Println("\n=== Sub-Question Workflow Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
