// Package main demonstrates a reflection workflow with iterative improvement.
// This example corresponds to Python's workflow/reflection.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/workflow"
)

// Define custom event types for reflection workflow
const (
	GenerateEventType workflow.EventType = "reflection.generate"
	CritiqueEventType workflow.EventType = "reflection.critique"
	RefineEventType   workflow.EventType = "reflection.refine"
	FinalEventType    workflow.EventType = "reflection.final"
)

// Event data structures
type GenerateData struct {
	Task  string
	Draft string
}

type CritiqueData struct {
	Task     string
	Draft    string
	Critique string
	Score    int
}

type RefineData struct {
	Task      string
	Draft     string
	Critique  string
	Iteration int
}

type FinalData struct {
	Task       string
	FinalDraft string
	Iterations int
}

// Event factories
var (
	GenerateEvent = workflow.NewEventFactory[GenerateData](GenerateEventType)
	CritiqueEvent = workflow.NewEventFactory[CritiqueData](CritiqueEventType)
	RefineEvent   = workflow.NewEventFactory[RefineData](RefineEventType)
	FinalEvent    = workflow.NewEventFactory[FinalData](FinalEventType)
)

const (
	MaxIterations = 3
	PassingScore  = 8
)

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Reflection Workflow Demo ===")
	fmt.Println("\nIteratively improves output through self-critique and refinement.")

	separator := strings.Repeat("=", 60)

	// 1. Build the reflection workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Reflection Workflow ===")
	fmt.Println(separator)

	reflectionWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Reflection Workflow"),
		workflow.WithWorkflowTimeout(120*time.Second),
	)

	// Step 1: Handle start event -> Generate initial draft
	reflectionWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			task := data.Input.(string)
			fmt.Printf("\n[Start] Task: %s\n", truncate(task, 50))

			wfCtx.Set("task", task)
			wfCtx.Set("iteration", 0)

			// Generate initial draft
			prompt := fmt.Sprintf(`Write a response for the following task:

Task: %s

Response:`, task)

			draft, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				return []workflow.Event{workflow.NewErrorEvent(err, "generate", event)}, nil
			}

			fmt.Printf("[Generate] Created initial draft (%d chars)\n", len(draft))
			return []workflow.Event{
				GenerateEvent.With(GenerateData{Task: task, Draft: draft}),
			}, nil
		},
	)

	// Step 2: Handle generate event -> Critique the draft
	reflectionWorkflow.Handle(
		[]workflow.EventType{GenerateEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := GenerateEvent.Extract(event)
			iteration, _ := wfCtx.GetInt("iteration")
			fmt.Printf("[Critique] Evaluating draft (iteration %d)\n", iteration+1)

			// Critique the draft
			prompt := fmt.Sprintf(`Evaluate the following response to a task. 
Provide a critique and a score from 1-10 (10 being perfect).

Task: %s

Response: %s

Format your response as:
CRITIQUE: <your critique>
SCORE: <number 1-10>`, data.Task, data.Draft)

			critique, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				return []workflow.Event{workflow.NewErrorEvent(err, "critique", event)}, nil
			}

			// Parse score (simple extraction)
			score := extractScore(critique)
			fmt.Printf("[Critique] Score: %d/10\n", score)

			return []workflow.Event{
				CritiqueEvent.With(CritiqueData{
					Task:     data.Task,
					Draft:    data.Draft,
					Critique: critique,
					Score:    score,
				}),
			}, nil
		},
	)

	// Step 3: Handle critique event -> Decide to refine or finalize
	reflectionWorkflow.Handle(
		[]workflow.EventType{CritiqueEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := CritiqueEvent.Extract(event)
			iteration, _ := wfCtx.GetInt("iteration")

			// Check if we should stop
			if data.Score >= PassingScore {
				fmt.Printf("[Decision] Score %d >= %d, finalizing\n", data.Score, PassingScore)
				return []workflow.Event{
					FinalEvent.With(FinalData{
						Task:       data.Task,
						FinalDraft: data.Draft,
						Iterations: iteration + 1,
					}),
				}, nil
			}

			if iteration >= MaxIterations-1 {
				fmt.Printf("[Decision] Max iterations (%d) reached, finalizing\n", MaxIterations)
				return []workflow.Event{
					FinalEvent.With(FinalData{
						Task:       data.Task,
						FinalDraft: data.Draft,
						Iterations: iteration + 1,
					}),
				}, nil
			}

			// Continue refining
			fmt.Printf("[Decision] Score %d < %d, refining...\n", data.Score, PassingScore)
			return []workflow.Event{
				RefineEvent.With(RefineData{
					Task:      data.Task,
					Draft:     data.Draft,
					Critique:  data.Critique,
					Iteration: iteration + 1,
				}),
			}, nil
		},
	)

	// Step 4: Handle refine event -> Generate improved draft
	reflectionWorkflow.Handle(
		[]workflow.EventType{RefineEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := RefineEvent.Extract(event)
			fmt.Printf("[Refine] Improving draft (iteration %d)\n", data.Iteration+1)

			wfCtx.Set("iteration", data.Iteration)

			// Refine based on critique
			prompt := fmt.Sprintf(`Improve the following response based on the critique.

Task: %s

Current Response: %s

Critique: %s

Improved Response:`, data.Task, data.Draft, data.Critique)

			improved, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				return []workflow.Event{workflow.NewErrorEvent(err, "refine", event)}, nil
			}

			fmt.Printf("[Refine] Created improved draft (%d chars)\n", len(improved))
			return []workflow.Event{
				GenerateEvent.With(GenerateData{Task: data.Task, Draft: improved}),
			}, nil
		},
	)

	// Step 5: Handle final event -> Stop
	reflectionWorkflow.Handle(
		[]workflow.EventType{FinalEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := FinalEvent.Extract(event)
			fmt.Printf("[Final] Workflow complete after %d iterations\n", data.Iterations)
			wfCtx.Set("final_draft", data.FinalDraft)
			wfCtx.Set("iterations", data.Iterations)
			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	fmt.Println("\nWorkflow steps:")
	fmt.Println("  1. Start -> Generate (create initial draft)")
	fmt.Println("  2. Generate -> Critique (evaluate quality)")
	fmt.Println("  3. Critique -> Refine OR Final (based on score)")
	fmt.Println("  4. Refine -> Generate (improve and loop)")
	fmt.Println("  5. Final -> Stop")
	fmt.Printf("\nSettings: MaxIterations=%d, PassingScore=%d\n", MaxIterations, PassingScore)

	// 2. Run the workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Running Reflection Workflow ===")
	fmt.Println(separator)

	tasks := []string{
		"Write a haiku about programming",
		"Explain quantum computing in one paragraph for a beginner",
	}

	for i, task := range tasks {
		fmt.Printf("\n--- Task %d ---\n", i+1)

		result, err := reflectionWorkflow.Run(ctx, workflow.NewStartEvent(task))
		if err != nil {
			fmt.Printf("Workflow error: %v\n", err)
			continue
		}

		// Extract result
		if workflow.StopEvent.Include(result.FinalEvent) {
			stopData, _ := workflow.StopEvent.Extract(result.FinalEvent)
			if finalData, ok := stopData.Result.(FinalData); ok {
				fmt.Printf("\nTask: %s\n", truncate(finalData.Task, 50))
				fmt.Printf("Final Draft:\n%s\n", truncate(finalData.FinalDraft, 300))
				fmt.Printf("Iterations: %d\n", finalData.Iterations)
			}
		}

		fmt.Printf("Duration: %v\n", result.Duration)
	}

	// 3. Visualize the reflection loop
	fmt.Println("\n" + separator)
	fmt.Println("=== Reflection Loop Visualization ===")
	fmt.Println(separator)

	fmt.Println(`
    ┌─────────────┐
    │   Start     │
    └──────┬──────┘
           │
           ▼
    ┌─────────────┐
    │  Generate   │◄──────────┐
    └──────┬──────┘           │
           │                  │
           ▼                  │
    ┌─────────────┐           │
    │  Critique   │           │
    └──────┬──────┘           │
           │                  │
           ▼                  │
    ┌─────────────┐    No     │
    │ Score >= 8? ├───────────┤
    │ or Max Iter │   Refine  │
    └──────┬──────┘           │
           │ Yes              │
           ▼                  │
    ┌─────────────┐           │
    │   Final     │           │
    └──────┬──────┘           │
           │                  │
           ▼                  │
    ┌─────────────┐           │
    │    Stop     │           │
    └─────────────┘           │
`)
	// 4. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nReflection Workflow Features:")
	fmt.Println("  - Self-critique and improvement loop")
	fmt.Println("  - Quality-based termination (score threshold)")
	fmt.Println("  - Iteration limit for safety")
	fmt.Println("  - State tracking across iterations")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Content generation with quality control")
	fmt.Println("  - Code generation with self-review")
	fmt.Println("  - Essay writing with iterative refinement")
	fmt.Println("  - Any task benefiting from self-improvement")

	fmt.Println("\n=== Reflection Workflow Demo Complete ===")
}

// extractScore extracts a numeric score from critique text.
func extractScore(critique string) int {
	// Simple extraction - look for "SCORE: N" pattern
	lower := strings.ToLower(critique)
	if idx := strings.Index(lower, "score:"); idx != -1 {
		remaining := strings.TrimSpace(critique[idx+6:])
		for _, c := range remaining {
			if c >= '0' && c <= '9' {
				return int(c - '0')
			}
		}
	}
	// Default to middle score if not found
	return 5
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
