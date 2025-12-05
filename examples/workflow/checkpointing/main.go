// Package main demonstrates workflow checkpointing and state persistence.
// This example corresponds to Python's workflow/checkpointing_workflows.ipynb
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/workflow"
)

// Define custom event types for checkpointing workflow
const (
	ProcessEventType    workflow.EventType = "checkpoint.process"
	CheckpointEventType workflow.EventType = "checkpoint.save"
	ResumeEventType     workflow.EventType = "checkpoint.resume"
	CompleteEventType   workflow.EventType = "checkpoint.complete"
)

// Event data structures
type ProcessData struct {
	Step  int
	Data  string
	Total int
}

type CheckpointData struct {
	Step      int
	State     map[string]interface{}
	Timestamp time.Time
}

type ResumeData struct {
	FromStep int
	State    map[string]interface{}
}

type CompleteData struct {
	Steps    int
	Results  []string
	Duration time.Duration
}

// Event factories
var (
	ProcessEvent    = workflow.NewEventFactory[ProcessData](ProcessEventType)
	CheckpointEvent = workflow.NewEventFactory[CheckpointData](CheckpointEventType)
	ResumeEvent     = workflow.NewEventFactory[ResumeData](ResumeEventType)
	CompleteEvent   = workflow.NewEventFactory[CompleteData](CompleteEventType)
)

// Checkpoint represents a saved workflow state
type Checkpoint struct {
	WorkflowName string                 `json:"workflow_name"`
	Step         int                    `json:"step"`
	State        map[string]interface{} `json:"state"`
	Timestamp    time.Time              `json:"timestamp"`
}

// CheckpointStore manages checkpoint persistence
type CheckpointStore struct {
	checkpoints map[string]Checkpoint
	filePath    string
}

// NewCheckpointStore creates a new checkpoint store
func NewCheckpointStore(filePath string) *CheckpointStore {
	return &CheckpointStore{
		checkpoints: make(map[string]Checkpoint),
		filePath:    filePath,
	}
}

// Save saves a checkpoint
func (s *CheckpointStore) Save(name string, checkpoint Checkpoint) error {
	s.checkpoints[name] = checkpoint
	return s.persist()
}

// Load loads a checkpoint
func (s *CheckpointStore) Load(name string) (Checkpoint, bool) {
	cp, ok := s.checkpoints[name]
	return cp, ok
}

// Delete removes a checkpoint
func (s *CheckpointStore) Delete(name string) {
	delete(s.checkpoints, name)
	s.persist()
}

// persist saves checkpoints to file
func (s *CheckpointStore) persist() error {
	if s.filePath == "" {
		return nil
	}
	data, err := json.MarshalIndent(s.checkpoints, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(s.filePath, data, 0644)
}

// loadFromFile loads checkpoints from file
func (s *CheckpointStore) loadFromFile() error {
	if s.filePath == "" {
		return nil
	}
	data, err := os.ReadFile(s.filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	return json.Unmarshal(data, &s.checkpoints)
}

func main() {
	ctx := context.Background()

	fmt.Println("=== Checkpointing Workflow Demo ===")
	fmt.Println("\nDemonstrates saving and resuming workflow state.")

	separator := strings.Repeat("=", 60)

	// 1. Create checkpoint store
	checkpointStore := NewCheckpointStore("") // In-memory for demo

	// 2. Build the checkpointing workflow
	fmt.Println("\n" + separator)
	fmt.Println("=== Building Checkpointing Workflow ===")
	fmt.Println(separator)

	checkpointWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Checkpointing Workflow"),
		workflow.WithWorkflowTimeout(30*time.Second),
	)

	// Step 1: Handle start event -> Begin processing
	checkpointWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			items := data.Input.([]string)
			fmt.Printf("\n[Start] Processing %d items\n", len(items))

			wfCtx.Set("items", items)
			wfCtx.Set("total", len(items))
			wfCtx.Set("results", []string{})
			wfCtx.Set("start_time", time.Now())

			return []workflow.Event{
				ProcessEvent.With(ProcessData{Step: 0, Data: items[0], Total: len(items)}),
			}, nil
		},
	)

	// Step 2: Handle resume event -> Continue from checkpoint
	checkpointWorkflow.Handle(
		[]workflow.EventType{ResumeEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := ResumeEvent.Extract(event)
			fmt.Printf("\n[Resume] Resuming from step %d\n", data.FromStep)

			// Restore state
			for k, v := range data.State {
				wfCtx.Set(k, v)
			}

			items, _ := wfCtx.Get("items")
			itemList := items.([]string)

			if data.FromStep >= len(itemList) {
				return []workflow.Event{
					CompleteEvent.With(CompleteData{
						Steps:   data.FromStep,
						Results: wfCtx.State().Keys(),
					}),
				}, nil
			}

			return []workflow.Event{
				ProcessEvent.With(ProcessData{
					Step:  data.FromStep,
					Data:  itemList[data.FromStep],
					Total: len(itemList),
				}),
			}, nil
		},
	)

	// Step 3: Handle process event -> Process item and checkpoint
	checkpointWorkflow.Handle(
		[]workflow.EventType{ProcessEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := ProcessEvent.Extract(event)
			fmt.Printf("[Process] Step %d/%d: %s\n", data.Step+1, data.Total, data.Data)

			// Simulate processing
			result := fmt.Sprintf("Processed: %s", strings.ToUpper(data.Data))

			// Update results
			results, _ := wfCtx.Get("results")
			resultList := results.([]string)
			resultList = append(resultList, result)
			wfCtx.Set("results", resultList)
			wfCtx.Set("current_step", data.Step+1)

			// Save checkpoint after each step
			checkpoint := Checkpoint{
				WorkflowName: "checkpointing_demo",
				Step:         data.Step + 1,
				State: map[string]interface{}{
					"items":        wfCtx.State().Keys(),
					"results":      resultList,
					"current_step": data.Step + 1,
				},
				Timestamp: time.Now(),
			}
			checkpointStore.Save("checkpointing_demo", checkpoint)
			fmt.Printf("[Checkpoint] Saved at step %d\n", data.Step+1)

			// Check if done
			if data.Step+1 >= data.Total {
				startTime, _ := wfCtx.Get("start_time")
				duration := time.Since(startTime.(time.Time))

				return []workflow.Event{
					CompleteEvent.With(CompleteData{
						Steps:    data.Step + 1,
						Results:  resultList,
						Duration: duration,
					}),
				}, nil
			}

			// Continue to next step
			items, _ := wfCtx.Get("items")
			itemList := items.([]string)

			return []workflow.Event{
				ProcessEvent.With(ProcessData{
					Step:  data.Step + 1,
					Data:  itemList[data.Step+1],
					Total: data.Total,
				}),
			}, nil
		},
	)

	// Step 4: Handle complete event -> Stop
	checkpointWorkflow.Handle(
		[]workflow.EventType{CompleteEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := CompleteEvent.Extract(event)
			fmt.Printf("[Complete] Processed %d steps in %v\n", data.Steps, data.Duration)

			// Clean up checkpoint
			checkpointStore.Delete("checkpointing_demo")
			fmt.Printf("[Checkpoint] Cleaned up\n")

			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	fmt.Println("\nWorkflow features:")
	fmt.Println("  - Checkpoint after each processing step")
	fmt.Println("  - Resume from last checkpoint on failure")
	fmt.Println("  - State persistence across restarts")

	// 3. Run the workflow normally
	fmt.Println("\n" + separator)
	fmt.Println("=== Normal Workflow Execution ===")
	fmt.Println(separator)

	items := []string{"apple", "banana", "cherry", "date"}

	fmt.Printf("\nProcessing items: %v\n", items)

	result, err := checkpointWorkflow.Run(ctx, workflow.NewStartEvent(items))
	if err != nil {
		fmt.Printf("Workflow error: %v\n", err)
	} else {
		if workflow.StopEvent.Include(result.FinalEvent) {
			stopData, _ := workflow.StopEvent.Extract(result.FinalEvent)
			if completeData, ok := stopData.Result.(CompleteData); ok {
				fmt.Printf("\nResults:\n")
				for i, r := range completeData.Results {
					fmt.Printf("  %d. %s\n", i+1, r)
				}
				fmt.Printf("Total duration: %v\n", completeData.Duration)
			}
		}
	}

	// 4. Demonstrate checkpoint and resume
	fmt.Println("\n" + separator)
	fmt.Println("=== Simulated Failure and Resume ===")
	fmt.Println(separator)

	// Create a workflow that "fails" after step 2
	failingWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Failing Workflow"),
		workflow.WithWorkflowTimeout(30*time.Second),
	)

	var failStep = 2
	var savedCheckpoint Checkpoint

	failingWorkflow.Handle(
		[]workflow.EventType{workflow.StartEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := workflow.StartEvent.Extract(event)
			items := data.Input.([]string)
			fmt.Printf("\n[Start] Processing %d items (will fail at step %d)\n", len(items), failStep+1)

			wfCtx.Set("items", items)
			wfCtx.Set("results", []string{})

			return []workflow.Event{
				ProcessEvent.With(ProcessData{Step: 0, Data: items[0], Total: len(items)}),
			}, nil
		},
	)

	failingWorkflow.Handle(
		[]workflow.EventType{ProcessEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := ProcessEvent.Extract(event)
			fmt.Printf("[Process] Step %d/%d: %s\n", data.Step+1, data.Total, data.Data)

			// Simulate processing
			result := fmt.Sprintf("Processed: %s", strings.ToUpper(data.Data))

			results, _ := wfCtx.Get("results")
			resultList := results.([]string)
			resultList = append(resultList, result)
			wfCtx.Set("results", resultList)

			// Save checkpoint
			items, _ := wfCtx.Get("items")
			savedCheckpoint = Checkpoint{
				WorkflowName: "failing_demo",
				Step:         data.Step + 1,
				State: map[string]interface{}{
					"items":   items,
					"results": resultList,
				},
				Timestamp: time.Now(),
			}
			fmt.Printf("[Checkpoint] Saved at step %d\n", data.Step+1)

			// Simulate failure
			if data.Step+1 == failStep {
				fmt.Printf("[ERROR] Simulated failure at step %d!\n", data.Step+1)
				return []workflow.Event{
					workflow.NewErrorEvent(fmt.Errorf("simulated failure"), "process", event),
				}, nil
			}

			// Continue
			if data.Step+1 >= data.Total {
				return []workflow.Event{
					CompleteEvent.With(CompleteData{Steps: data.Step + 1, Results: resultList}),
				}, nil
			}

			itemList := items.([]string)
			return []workflow.Event{
				ProcessEvent.With(ProcessData{
					Step:  data.Step + 1,
					Data:  itemList[data.Step+1],
					Total: data.Total,
				}),
			}, nil
		},
	)

	failingWorkflow.Handle(
		[]workflow.EventType{CompleteEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := CompleteEvent.Extract(event)
			fmt.Printf("[Complete] Finished %d steps\n", data.Steps)
			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	// Run and expect failure
	fmt.Println("\nRunning workflow (expecting failure)...")
	_, err = failingWorkflow.Run(ctx, workflow.NewStartEvent(items))
	if err != nil {
		fmt.Printf("Workflow failed as expected: %v\n", err)
	}

	// Show saved checkpoint
	fmt.Printf("\nSaved checkpoint:\n")
	fmt.Printf("  Step: %d\n", savedCheckpoint.Step)
	fmt.Printf("  Timestamp: %v\n", savedCheckpoint.Timestamp)

	// Resume from checkpoint
	fmt.Println("\nResuming from checkpoint...")

	// Create resume workflow
	resumeWorkflow := workflow.NewWorkflow(
		workflow.WithWorkflowName("Resume Workflow"),
		workflow.WithWorkflowTimeout(30*time.Second),
	)

	resumeWorkflow.Handle(
		[]workflow.EventType{ResumeEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := ResumeEvent.Extract(event)
			fmt.Printf("[Resume] Continuing from step %d\n", data.FromStep)

			// Restore state
			for k, v := range data.State {
				wfCtx.Set(k, v)
			}

			items := data.State["items"].([]string)
			results := data.State["results"].([]string)

			// Process remaining items
			for i := data.FromStep; i < len(items); i++ {
				result := fmt.Sprintf("Processed: %s", strings.ToUpper(items[i]))
				results = append(results, result)
				fmt.Printf("[Process] Step %d/%d: %s -> %s\n", i+1, len(items), items[i], result)
			}

			return []workflow.Event{
				CompleteEvent.With(CompleteData{Steps: len(items), Results: results}),
			}, nil
		},
	)

	resumeWorkflow.Handle(
		[]workflow.EventType{CompleteEventType},
		func(wfCtx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
			data, _ := CompleteEvent.Extract(event)
			fmt.Printf("[Complete] Finished all %d steps\n", data.Steps)
			return []workflow.Event{workflow.NewStopEvent(data)}, nil
		},
	)

	// Run resume
	resumeResult, err := resumeWorkflow.Run(ctx, ResumeEvent.With(ResumeData{
		FromStep: savedCheckpoint.Step,
		State:    savedCheckpoint.State,
	}))

	if err != nil {
		fmt.Printf("Resume error: %v\n", err)
	} else if workflow.StopEvent.Include(resumeResult.FinalEvent) {
		stopData, _ := workflow.StopEvent.Extract(resumeResult.FinalEvent)
		if completeData, ok := stopData.Result.(CompleteData); ok {
			fmt.Printf("\nFinal results after resume:\n")
			for i, r := range completeData.Results {
				fmt.Printf("  %d. %s\n", i+1, r)
			}
		}
	}

	// 5. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nCheckpointing Features:")
	fmt.Println("  - Save workflow state at any point")
	fmt.Println("  - Resume from last checkpoint on failure")
	fmt.Println("  - Persist checkpoints to storage")
	fmt.Println("  - Clean up checkpoints on completion")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Long-running workflows")
	fmt.Println("  - Fault-tolerant processing")
	fmt.Println("  - Batch processing with recovery")
	fmt.Println("  - Workflows with external dependencies")

	fmt.Println("\n=== Checkpointing Workflow Demo Complete ===")
}
