# Workflow Examples

This directory contains examples demonstrating event-driven workflow orchestration in `go-llamaindex`.

## Examples

### 1. RAG Workflow (`rag_workflow/`)

Demonstrates a complete RAG pipeline as a workflow.

**Features:**
- Query -> Retrieve -> Synthesize -> Response flow
- State management across steps
- LLM-based response generation
- Streaming execution support

**Run:**
```bash
cd rag_workflow && go run main.go
```

### 2. Reflection Workflow (`reflection_workflow/`)

Shows iterative self-improvement through critique and refinement.

**Features:**
- Generate -> Critique -> Refine loop
- Quality-based termination (score threshold)
- Iteration limits for safety
- Self-critique using LLM

**Run:**
```bash
cd reflection_workflow && go run main.go
```

### 3. Parallel Workflow (`parallel_workflow/`)

Demonstrates fan-out/fan-in pattern for parallel execution.

**Features:**
- Fan-out: Distribute tasks
- Parallel task processing
- Fan-in: Collect and aggregate results
- Synchronization handling

**Run:**
```bash
cd parallel_workflow && go run main.go
```

### 4. Router Workflow (`router_workflow/`)

Shows query routing to specialized handlers.

**Features:**
- LLM-based query classification
- Dynamic routing to domain handlers
- Multiple knowledge bases
- Unified response format

**Run:**
```bash
cd router_workflow && go run main.go
```

### 5. Sub-Question Workflow (`sub_question_workflow/`)

Demonstrates query decomposition for complex questions.

**Features:**
- Automatic sub-question generation
- Tool/source routing
- Parallel sub-question processing
- Answer synthesis

**Run:**
```bash
cd sub_question_workflow && go run main.go
```

### 6. Checkpointing (`checkpointing/`)

Shows workflow state persistence and recovery.

**Features:**
- Save checkpoints during execution
- Resume from last checkpoint
- State serialization
- Failure recovery

**Run:**
```bash
cd checkpointing && go run main.go
```

## Key Concepts

### Workflow Structure

```go
// Create workflow
wf := workflow.NewWorkflow(
    workflow.WithWorkflowName("My Workflow"),
    workflow.WithWorkflowTimeout(30*time.Second),
)

// Register handlers
wf.Handle(
    []workflow.EventType{MyEventType},
    func(ctx *workflow.Context, event workflow.Event) ([]workflow.Event, error) {
        // Process event
        return []workflow.Event{NextEvent.With(data)}, nil
    },
)

// Run workflow
result, err := wf.Run(ctx, workflow.NewStartEvent(input))
```

### Event Types

```go
// Define custom events
const MyEventType workflow.EventType = "my.event"

type MyEventData struct {
    Field string
}

var MyEvent = workflow.NewEventFactory[MyEventData](MyEventType)

// Create event
event := MyEvent.With(MyEventData{Field: "value"})

// Extract data
data, ok := MyEvent.Extract(event)
```

### Built-in Events

| Event | Purpose |
|-------|---------|
| StartEvent | Begin workflow execution |
| StopEvent | End workflow with result |
| ErrorEvent | Signal error condition |
| InputRequiredEvent | Request human input |
| HumanResponseEvent | Provide human response |

### Context and State

```go
// Store state
ctx.Set("key", value)

// Retrieve state
value, ok := ctx.Get("key")
str, ok := ctx.GetString("key")
num, ok := ctx.GetInt("key")

// Send events
ctx.SendEvent(event)
```

### Workflow Patterns

#### Sequential
```
Start -> Step1 -> Step2 -> Step3 -> Stop
```

#### Branching
```
Start -> Classify -> RouteA | RouteB | RouteC -> Stop
```

#### Loop
```
Start -> Generate -> Critique -> (Refine -> Generate) | Stop
```

#### Fan-Out/Fan-In
```
Start -> FanOut -> [Task1, Task2, Task3] -> FanIn -> Stop
```

## Handler Decorators

```go
// Logging
handler := workflow.LoggingHandler("step_name", myHandler)

// Timing
handler := workflow.TimingHandler("step_name", myHandler)

// Conditional
handler := workflow.ConditionalHandler(condition, myHandler)

// Fallback
handler := workflow.FallbackHandler(myHandler, fallbackHandler)

// Chain
handler := workflow.ChainHandlers(handler1, handler2, handler3)
```

## Streaming Execution

```go
stream := wf.RunStream(ctx, workflow.NewStartEvent(input))

for event := range stream.Events() {
    fmt.Printf("Event: %s\n", event.Type())
}

if err := stream.Err(); err != nil {
    log.Printf("Error: %v", err)
}
```

## Environment Variables

All examples require:
- `OPENAI_API_KEY` - OpenAI API key for LLM operations
