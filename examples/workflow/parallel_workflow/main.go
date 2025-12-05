// Package main demonstrates parallel execution in workflows.
// This example corresponds to Python's workflow/parallel_execution.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/workflow"
)

// Define custom event types for parallel workflow
const (
	FanOutEventType    workflow.EventType = "parallel.fanout"
	TaskEventType      workflow.EventType = "parallel.task"
	TaskDoneEventType  workflow.EventType = "parallel.task_done"
	FanInEventType     workflow.EventType = "parallel.fanin"
	AggregateEventType workflow.EventType = "parallel.aggregate"
)

// Event data structures
type FanOutData struct {
	Tasks []string
}

type TaskData struct {
	TaskID int
	Task   string
	Total  int
}

type TaskDoneData struct {
	TaskID int
	Task   string
	Result string
}

type FanInData struct {
	Results []TaskDoneData
}

type AggregateData struct {
	Summary string
	Results []TaskDoneData
}

// Event factories
var (
	FanOutEvent    = workflow.NewEventFactory[FanOutData](FanOutEventType)
	TaskEvent      = workflow.NewEventFactory[TaskData](TaskEventType)
	TaskDoneEvent  = workflow.NewEventFactory[TaskDoneData](TaskDoneEventType)
	FanInEvent     = workflow.NewEventFactory[FanInData](FanInEventType)
	AggregateEvent = workflow.NewEventFactory[AggregateData](AggregateEventType)
)

func main() {
	ctx := context.Background()

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Parallel Workflow Demo ===")
	fmt.Println("\nDemonstrates fan-out/fan-in pattern for parallel task execution.")

	separator := strings.Repeat("=", 60)

	// 1. Build the parallel workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Parallel Workflow ===")
	fmt.Println(separator)

	parallelWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Parallel Workflow"),
		workflow.WithWorkflowTimeout(60*time.Second),
	)

	// Mutex for collecting results
	var resultsMu sync.Mutex
	var collectedResults []TaskDoneData

	// Step 1: Handle start event -> Fan out to multiple tasks
	parallelWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			tasks := data.Input.([]string)
			fmt.Printf("\n[Start] Received %d tasks to process in parallel\n", len(tasks))

			wfCtx.Set("total_tasks", len(tasks))
			wfCtx.Set("completed_tasks", 0)

			// Reset results
			resultsMu.Lock()
			collectedResults = nil
			resultsMu.Unlock()

			return []workflow.Event{FanOutEvent.With(FanOutData{Tasks: tasks})}, nil
		},
	)

	// Step 2: Handle fan-out -> Create individual task events
	parallelWorkflow.Handle(
		[]workflow.EventType{FanOutEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := FanOutEvent.Extract(event)
			fmt.Printf("[FanOut] Distributing %d tasks\n", len(data.Tasks))

			var events []workflow.Event
			for i, task := range data.Tasks {
				events = append(events, TaskEvent.With(TaskData{
					TaskID: i,
					Task:   task,
					Total:  len(data.Tasks),
				}))
			}

			return events, nil
		},
	)

	// Step 3: Handle task event -> Process task (simulated parallel)
	parallelWorkflow.Handle(
		[]workflow.EventType{TaskEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := TaskEvent.Extract(event)
			fmt.Printf("[Task %d] Processing: %s\n", data.TaskID, truncate(data.Task, 30))

			// Simulate processing with LLM
			prompt := fmt.Sprintf("Briefly answer: %s", data.Task)
			result, err := llmInstance.Complete(ctx, prompt)
			if err != nil {
				result = fmt.Sprintf("Error: %v", err)
			}

			fmt.Printf("[Task %d] Completed\n", data.TaskID)

			return []workflow.Event{
				TaskDoneEvent.With(TaskDoneData{
					TaskID: data.TaskID,
					Task:   data.Task,
					Result: result,
				}),
			}, nil
		},
	)

	// Step 4: Handle task done -> Collect results and check if all done
	parallelWorkflow.Handle(
		[]workflow.EventType{TaskDoneEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := TaskDoneEvent.Extract(event)

			// Collect result
			resultsMu.Lock()
			collectedResults = append(collectedResults, data)
			count := len(collectedResults)
			resultsMu.Unlock()

			totalTasks, _ := wfCtx.GetInt("total_tasks")
			fmt.Printf("[Collect] Received result %d/%d\n", count, totalTasks)

			// Check if all tasks are done
			if count >= totalTasks {
				resultsMu.Lock()
				results := make([]TaskDoneData, len(collectedResults))
				copy(results, collectedResults)
				resultsMu.Unlock()

				fmt.Printf("[Collect] All tasks complete, triggering fan-in\n")
				return []workflow.Event{FanInEvent.With(FanInData{Results: results})}, nil
			}

			return nil, nil
		},
	)

	// Step 5: Handle fan-in -> Aggregate results
	parallelWorkflow.Handle(
		[]workflow.EventType{FanInEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := FanInEvent.Extract(event)
			fmt.Printf("[FanIn] Aggregating %d results\n", len(data.Results))

			// Create summary
			var summaryParts []string
			for _, r := range data.Results {
				summaryParts = append(summaryParts, fmt.Sprintf("Q: %s\nA: %s", r.Task, truncate(r.Result, 100)))
			}
			summary := strings.Join(summaryParts, "\n\n")

			return []workflow.Event{
				AggregateEvent.With(AggregateData{
					Summary: summary,
					Results: data.Results,
				}),
			}, nil
		},
	)

	// Step 6: Handle aggregate -> Stop
	parallelWorkflow.Handle(
		[]workflow.EventType{AggregateEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := AggregateEvent.Extract(event)
			fmt.Printf("[Aggregate] Workflow complete with %d results\n", len(data.Results))
			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	fmt.Println("\nWorkflow pattern:")
	fmt.Println("  1. Start -> FanOut (distribute tasks)")
	fmt.Println("  2. FanOut -> Task[] (create parallel tasks)")
	fmt.Println("  3. Task -> TaskDone (process each task)")
	fmt.Println("  4. TaskDone -> FanIn (when all complete)")
	fmt.Println("  5. FanIn -> Aggregate (combine results)")
	fmt.Println("  6. Aggregate -> Stop")

	// 2. Run the workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Running Parallel Workflow ===")
	fmt.Println(separator)

	tasks := []string{
		"What is the capital of France?",
		"What is 2 + 2?",
		"Name a programming language",
	}

	fmt.Printf("\nProcessing %d tasks in parallel:\n", len(tasks))
	for i, task := range tasks {
		fmt.Printf("  %d. %s\n", i+1, task)
	}

	startTime := time.Now()
	result, err := parallelWorkflow.Run(ctx, workflow.NewStartEvent(tasks))
	if err != nil {
		fmt.Printf("Workflow error: %v\n", err)
	} else {
		fmt.Printf("\nTotal duration: %v\n", result.Duration)

		// Extract results
		if workflow.StopEvent.Include(result.FinalEvent) {
			stopData, _ := workflow.StopEvent.Extract(result.FinalEvent)
			if aggData, ok := stopData.Result.(AggregateData); ok {
				fmt.Printf("\nResults:\n")
				for _, r := range aggData.Results {
					fmt.Printf("  Task %d: %s\n", r.TaskID, truncate(r.Result, 60))
				}
			}
		}
	}
	fmt.Printf("Wall clock time: %v\n", time.Since(startTime))

	// 3. Visualize fan-out/fan-in pattern
	fmt.Println("\n" + separator)
	fmt.Println("=== Fan-Out/Fan-In Pattern ===")
	fmt.Println(separator)

	fmt.Println(`
                    ┌─────────┐
                    │  Start  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │ Fan-Out │
                    └────┬────┘
                         │
           ┌─────────────┼─────────────┐
           │             │             │
           ▼             ▼             ▼
      ┌────────┐    ┌────────┐    ┌────────┐
      │ Task 1 │    │ Task 2 │    │ Task 3 │
      └────┬───┘    └────┬───┘    └────┬───┘
           │             │             │
           └─────────────┼─────────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ Fan-In  │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │Aggregate│
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │  Stop   │
                    └─────────────┘
`)

	// 4. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nParallel Workflow Features:")
	fmt.Println("  - Fan-out: Distribute work to multiple tasks")
	fmt.Println("  - Parallel execution: Tasks run concurrently")
	fmt.Println("  - Fan-in: Collect and aggregate results")
	fmt.Println("  - Synchronization: Wait for all tasks to complete")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Batch processing with multiple LLM calls")
	fmt.Println("  - Multi-source data retrieval")
	fmt.Println("  - Parallel document analysis")
	fmt.Println("  - Map-reduce style computations")

	fmt.Println("\n=== Parallel Workflow Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
