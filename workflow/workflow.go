package workflow

import (
	"context"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"time"
)

// Workflow is the main workflow orchestration engine.
// It manages event handlers and executes workflows based on events.
type Workflow struct {
	name     string
	steps    []*Step
	handlers map[EventType][]*Step
	timeout  time.Duration
	logger   *slog.Logger
	mu       sync.RWMutex
}

// WorkflowOption configures a Workflow.
type WorkflowOption func(*Workflow)

// WithWorkflowName sets the workflow name.
func WithWorkflowName(name string) WorkflowOption {
	return func(w *Workflow) {
		w.name = name
	}
}

// WithWorkflowTimeout sets the workflow timeout.
func WithWorkflowTimeout(timeout time.Duration) WorkflowOption {
	return func(w *Workflow) {
		w.timeout = timeout
	}
}

// WithWorkflowLogger sets the workflow logger.
func WithWorkflowLogger(logger *slog.Logger) WorkflowOption {
	return func(w *Workflow) {
		w.logger = logger
	}
}

// NewWorkflow creates a new workflow.
func NewWorkflow(opts ...WorkflowOption) *Workflow {
	w := &Workflow{
		name:     "workflow",
		steps:    make([]*Step, 0),
		handlers: make(map[EventType][]*Step),
		timeout:  60 * time.Second,
		logger:   slog.New(slog.NewJSONHandler(os.Stdout, nil)),
	}

	for _, opt := range opts {
		opt(w)
	}

	return w
}

// Handle registers a handler for the given event types.
func (w *Workflow) Handle(eventTypes []EventType, handler Handler, config ...StepConfig) *Workflow {
	cfg := DefaultStepConfig()
	if len(config) > 0 {
		cfg = config[0]
	}

	step := &Step{
		AcceptedEvents: eventTypes,
		Handler:        handler,
		Config:         cfg,
	}

	w.mu.Lock()
	defer w.mu.Unlock()

	w.steps = append(w.steps, step)
	for _, eventType := range eventTypes {
		w.handlers[eventType] = append(w.handlers[eventType], step)
	}

	return w
}

// HandleTyped registers a typed handler for a specific event type.
func HandleTyped[T any](w *Workflow, factory *EventFactory[T], handler TypedHandler[T], config ...StepConfig) *Workflow {
	wrappedHandler := WrapTypedHandler(factory, handler)
	return w.Handle([]EventType{factory.Type()}, wrappedHandler, config...)
}

// CreateContext creates a new execution context for this workflow.
func (w *Workflow) CreateContext(ctx context.Context) *Context {
	return NewContext(ctx, w, w.timeout)
}

// Run executes the workflow with the given start event and returns the final result.
func (w *Workflow) Run(ctx context.Context, startEvent Event) (*WorkflowResult, error) {
	startTime := time.Now()
	wfCtx := w.CreateContext(ctx)

	// Send the start event
	wfCtx.SendEvent(startEvent)

	// Process events until done
	var finalEvent Event
	var finalErr error

	for !wfCtx.IsDone() {
		select {
		case event := <-wfCtx.eventQueue:
			if event == nil {
				continue
			}

			// Check if this is a stop event
			if StopEvent.Include(event) {
				finalEvent = event
				wfCtx.markDone()
				continue
			}

			// Check if this is an error event
			if ErrorEvent.Include(event) {
				data, _ := ErrorEvent.Extract(event)
				finalErr = data.Error
				finalEvent = event
				wfCtx.markDone()
				continue
			}

			// Process the event
			events, err := w.processEvent(wfCtx, event)
			if err != nil {
				w.logger.Error("Error processing event", "event_type", event.Type(), "error", err)
				finalErr = err
				wfCtx.markDone()
				continue
			}

			// Send resulting events
			for _, e := range events {
				wfCtx.SendEvent(e)
			}

		case <-ctx.Done():
			finalErr = ctx.Err()
			wfCtx.markDone()

		case <-time.After(100 * time.Millisecond):
			// Check for timeout
			if wfCtx.IsTimedOut() {
				finalErr = fmt.Errorf("workflow timed out after %v", w.timeout)
				wfCtx.markDone()
			}
			// Check if queue is empty and no more events expected
			if len(wfCtx.eventQueue) == 0 {
				// Give a small grace period for any pending events
				select {
				case event := <-wfCtx.eventQueue:
					if event != nil {
						wfCtx.SendEvent(event)
					}
				case <-time.After(50 * time.Millisecond):
					// No more events, we're done
					wfCtx.markDone()
				}
			}
		}
	}

	return &WorkflowResult{
		FinalEvent: finalEvent,
		State:      wfCtx.State(),
		Error:      finalErr,
		Duration:   time.Since(startTime),
	}, finalErr
}

// RunStream executes the workflow and returns a stream of events.
func (w *Workflow) RunStream(ctx context.Context, startEvent Event) *WorkflowStream {
	wfCtx := w.CreateContext(ctx)
	stream := NewWorkflowStream(wfCtx)

	go func() {
		defer stream.close(nil)

		// Send the start event
		wfCtx.SendEvent(startEvent)

		for !wfCtx.IsDone() {
			select {
			case event := <-wfCtx.eventQueue:
				if event == nil {
					continue
				}

				// Emit the event to the stream
				stream.emit(event)

				// Check if we should stop
				if stream.shouldStop(event) {
					wfCtx.markDone()
					continue
				}

				// Check if this is a stop event
				if StopEvent.Include(event) {
					wfCtx.markDone()
					continue
				}

				// Check if this is an error event
				if ErrorEvent.Include(event) {
					data, _ := ErrorEvent.Extract(event)
					stream.close(data.Error)
					wfCtx.markDone()
					return
				}

				// Process the event
				events, err := w.processEvent(wfCtx, event)
				if err != nil {
					w.logger.Error("Error processing event", "event_type", event.Type(), "error", err)
					stream.close(err)
					wfCtx.markDone()
					return
				}

				// Send resulting events
				for _, e := range events {
					wfCtx.SendEvent(e)
				}

			case <-ctx.Done():
				stream.close(ctx.Err())
				wfCtx.markDone()
				return

			case <-time.After(100 * time.Millisecond):
				// Check for timeout
				if wfCtx.IsTimedOut() {
					stream.close(fmt.Errorf("workflow timed out after %v", w.timeout))
					wfCtx.markDone()
					return
				}
				// Check if queue is empty
				if len(wfCtx.eventQueue) == 0 {
					select {
					case event := <-wfCtx.eventQueue:
						if event != nil {
							wfCtx.SendEvent(event)
						}
					case <-time.After(50 * time.Millisecond):
						wfCtx.markDone()
					}
				}
			}
		}
	}()

	return stream
}

// processEvent processes a single event through the workflow.
func (w *Workflow) processEvent(ctx *Context, event Event) ([]Event, error) {
	// Check timeout before processing
	if ctx.IsTimedOut() {
		return nil, fmt.Errorf("workflow timed out after %v", w.timeout)
	}

	w.mu.RLock()
	steps := w.handlers[event.Type()]
	w.mu.RUnlock()

	if len(steps) == 0 {
		w.logger.Debug("No handlers for event", "event_type", event.Type())
		return nil, nil
	}

	var allEvents []Event
	for _, step := range steps {
		// Check timeout before each step
		if ctx.IsTimedOut() {
			return nil, fmt.Errorf("workflow timed out after %v", w.timeout)
		}
		events, err := w.executeStep(ctx, step, event)
		if err != nil {
			return nil, err
		}
		allEvents = append(allEvents, events...)
	}

	return allEvents, nil
}

// executeStep executes a single step with retry logic.
func (w *Workflow) executeStep(ctx *Context, step *Step, event Event) ([]Event, error) {
	if step.Config.RetryPolicy == nil {
		return step.Handler(ctx, event)
	}

	policy := step.Config.RetryPolicy
	var lastErr error
	delay := policy.InitialDelay

	for attempt := 0; attempt <= policy.MaxRetries; attempt++ {
		events, err := step.Handler(ctx, event)
		if err == nil {
			return events, nil
		}

		lastErr = err
		if !policy.RetryOn(err) {
			return nil, err
		}

		if attempt < policy.MaxRetries {
			w.logger.Warn("Step failed, retrying",
				"step", step.Config.Name,
				"attempt", attempt+1,
				"max_retries", policy.MaxRetries,
				"error", err,
			)
			time.Sleep(delay)
			delay = time.Duration(float64(delay) * policy.Multiplier)
			if delay > policy.MaxDelay {
				delay = policy.MaxDelay
			}
		}
	}

	return nil, fmt.Errorf("step failed after %d retries: %w", policy.MaxRetries, lastErr)
}

// GetSteps returns all registered steps.
func (w *Workflow) GetSteps() []*Step {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return append([]*Step{}, w.steps...)
}

// GetHandlersForEvent returns all handlers for a specific event type.
func (w *Workflow) GetHandlersForEvent(eventType EventType) []*Step {
	w.mu.RLock()
	defer w.mu.RUnlock()
	return append([]*Step{}, w.handlers[eventType]...)
}

// WorkflowBuilder provides a fluent API for building workflows.
type WorkflowBuilder struct {
	workflow *Workflow
}

// NewWorkflowBuilder creates a new workflow builder.
func NewWorkflowBuilder(opts ...WorkflowOption) *WorkflowBuilder {
	return &WorkflowBuilder{
		workflow: NewWorkflow(opts...),
	}
}

// On registers a handler for the given event types.
func (b *WorkflowBuilder) On(eventTypes []EventType, handler Handler, config ...StepConfig) *WorkflowBuilder {
	b.workflow.Handle(eventTypes, handler, config...)
	return b
}

// OnEvent registers a handler for a single event type.
func (b *WorkflowBuilder) OnEvent(eventType EventType, handler Handler, config ...StepConfig) *WorkflowBuilder {
	return b.On([]EventType{eventType}, handler, config...)
}

// OnStart registers a handler for start events.
func (b *WorkflowBuilder) OnStart(handler TypedHandler[StartEventData], config ...StepConfig) *WorkflowBuilder {
	wrappedHandler := WrapTypedHandler(StartEvent, handler)
	return b.OnEvent(StartEventType, wrappedHandler, config...)
}

// Build returns the constructed workflow.
func (b *WorkflowBuilder) Build() *Workflow {
	return b.workflow
}

// OnTyped registers a typed handler using the builder pattern.
func OnTyped[T any](b *WorkflowBuilder, factory *EventFactory[T], handler TypedHandler[T], config ...StepConfig) *WorkflowBuilder {
	HandleTyped(b.workflow, factory, handler, config...)
	return b
}
