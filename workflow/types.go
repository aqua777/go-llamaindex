// Package workflow provides an event-driven workflow orchestration system.
// It allows defining workflows as a series of steps that react to events,
// enabling complex, stateful processing pipelines.
package workflow

import (
	"context"
	"sync"
	"time"
)

// EventType is a unique identifier for an event type.
type EventType string

// Event is the interface that all workflow events must implement.
type Event interface {
	// Type returns the unique type identifier for this event.
	Type() EventType
	// Data returns the event's payload data.
	Data() interface{}
}

// BaseEvent provides a basic implementation of the Event interface.
type BaseEvent struct {
	eventType EventType
	data      interface{}
}

// Type returns the event type.
func (e *BaseEvent) Type() EventType {
	return e.eventType
}

// Data returns the event data.
func (e *BaseEvent) Data() interface{} {
	return e.data
}

// NewEvent creates a new event with the given type and data.
func NewEvent(eventType EventType, data interface{}) *BaseEvent {
	return &BaseEvent{
		eventType: eventType,
		data:      data,
	}
}

// EventFactory creates typed events.
type EventFactory[T any] struct {
	eventType  EventType
	debugLabel string
}

// NewEventFactory creates a new event factory for a specific event type.
func NewEventFactory[T any](eventType EventType) *EventFactory[T] {
	return &EventFactory[T]{
		eventType:  eventType,
		debugLabel: string(eventType),
	}
}

// NewEventFactoryWithLabel creates a new event factory with a debug label.
func NewEventFactoryWithLabel[T any](eventType EventType, debugLabel string) *EventFactory[T] {
	return &EventFactory[T]{
		eventType:  eventType,
		debugLabel: debugLabel,
	}
}

// Type returns the event type for this factory.
func (f *EventFactory[T]) Type() EventType {
	return f.eventType
}

// With creates a new event with the given data.
func (f *EventFactory[T]) With(data T) *TypedEvent[T] {
	return &TypedEvent[T]{
		BaseEvent: BaseEvent{
			eventType: f.eventType,
			data:      data,
		},
		typedData: data,
	}
}

// Include checks if an event matches this factory's type.
func (f *EventFactory[T]) Include(event Event) bool {
	if event == nil {
		return false
	}
	return event.Type() == f.eventType
}

// Extract extracts the typed data from an event if it matches.
func (f *EventFactory[T]) Extract(event Event) (T, bool) {
	var zero T
	if !f.Include(event) {
		return zero, false
	}
	if typed, ok := event.(*TypedEvent[T]); ok {
		return typed.typedData, true
	}
	if data, ok := event.Data().(T); ok {
		return data, true
	}
	return zero, false
}

// TypedEvent is an event with typed data.
type TypedEvent[T any] struct {
	BaseEvent
	typedData T
}

// TypedData returns the typed event data.
func (e *TypedEvent[T]) TypedData() T {
	return e.typedData
}

// Context provides the execution context for workflow steps.
// It allows steps to access shared state, send events, and manage the workflow lifecycle.
type Context struct {
	ctx        context.Context
	cancel     context.CancelFunc
	workflow   *Workflow
	state      *StateStore
	eventQueue chan Event
	mu         sync.RWMutex
	done       bool
	timeout    time.Duration
	startTime  time.Time
}

// NewContext creates a new workflow context.
func NewContext(ctx context.Context, workflow *Workflow, timeout time.Duration) *Context {
	ctx, cancel := context.WithCancel(ctx)
	return &Context{
		ctx:        ctx,
		cancel:     cancel,
		workflow:   workflow,
		state:      NewStateStore(),
		eventQueue: make(chan Event, 1000),
		timeout:    timeout,
		startTime:  time.Now(),
	}
}

// Context returns the underlying context.Context.
func (c *Context) Context() context.Context {
	return c.ctx
}

// SendEvent sends an event to the workflow for processing.
func (c *Context) SendEvent(event Event) {
	c.mu.RLock()
	done := c.done
	c.mu.RUnlock()

	if done {
		return
	}

	select {
	case c.eventQueue <- event:
	case <-c.ctx.Done():
	}
}

// State returns the state store for this context.
func (c *Context) State() *StateStore {
	return c.state
}

// Get retrieves a value from the context state.
func (c *Context) Get(key string) (interface{}, bool) {
	return c.state.Get(key)
}

// Set stores a value in the context state.
func (c *Context) Set(key string, value interface{}) {
	c.state.Set(key, value)
}

// Delete removes a value from the context state.
func (c *Context) Delete(key string) {
	c.state.Delete(key)
}

// GetString retrieves a string value from the context state.
func (c *Context) GetString(key string) (string, bool) {
	return c.state.GetString(key)
}

// GetInt retrieves an int value from the context state.
func (c *Context) GetInt(key string) (int, bool) {
	return c.state.GetInt(key)
}

// GetBool retrieves a bool value from the context state.
func (c *Context) GetBool(key string) (bool, bool) {
	return c.state.GetBool(key)
}

// Cancel cancels the workflow execution.
func (c *Context) Cancel() {
	c.mu.Lock()
	c.done = true
	c.mu.Unlock()
	c.cancel()
}

// IsDone returns true if the workflow has completed.
func (c *Context) IsDone() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.done
}

// IsTimedOut returns true if the workflow has exceeded its timeout.
func (c *Context) IsTimedOut() bool {
	if c.timeout <= 0 {
		return false
	}
	return time.Since(c.startTime) > c.timeout
}

// markDone marks the context as done.
func (c *Context) markDone() {
	c.mu.Lock()
	c.done = true
	c.mu.Unlock()
}

// StateStore provides thread-safe state storage for workflows.
type StateStore struct {
	data map[string]interface{}
	mu   sync.RWMutex
}

// NewStateStore creates a new state store.
func NewStateStore() *StateStore {
	return &StateStore{
		data: make(map[string]interface{}),
	}
}

// Get retrieves a value from the store.
func (s *StateStore) Get(key string) (interface{}, bool) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	val, ok := s.data[key]
	return val, ok
}

// Set stores a value in the store.
func (s *StateStore) Set(key string, value interface{}) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data[key] = value
}

// Delete removes a value from the store.
func (s *StateStore) Delete(key string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.data, key)
}

// GetString retrieves a string value from the store.
func (s *StateStore) GetString(key string) (string, bool) {
	val, ok := s.Get(key)
	if !ok {
		return "", false
	}
	str, ok := val.(string)
	return str, ok
}

// GetInt retrieves an int value from the store.
func (s *StateStore) GetInt(key string) (int, bool) {
	val, ok := s.Get(key)
	if !ok {
		return 0, false
	}
	i, ok := val.(int)
	return i, ok
}

// GetBool retrieves a bool value from the store.
func (s *StateStore) GetBool(key string) (bool, bool) {
	val, ok := s.Get(key)
	if !ok {
		return false, false
	}
	b, ok := val.(bool)
	return b, ok
}

// Keys returns all keys in the store.
func (s *StateStore) Keys() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	keys := make([]string, 0, len(s.data))
	for k := range s.data {
		keys = append(keys, k)
	}
	return keys
}

// Clear removes all values from the store.
func (s *StateStore) Clear() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.data = make(map[string]interface{})
}

// Clone creates a copy of the store.
func (s *StateStore) Clone() *StateStore {
	s.mu.RLock()
	defer s.mu.RUnlock()
	clone := NewStateStore()
	for k, v := range s.data {
		clone.data[k] = v
	}
	return clone
}

// Handler is a function that handles events in a workflow.
// It receives the context and the triggering event, and returns
// zero or more events to emit, or an error.
type Handler func(ctx *Context, event Event) ([]Event, error)

// TypedHandler is a handler for a specific event type.
type TypedHandler[T any] func(ctx *Context, data T) ([]Event, error)

// WrapTypedHandler wraps a typed handler into a generic Handler.
func WrapTypedHandler[T any](factory *EventFactory[T], handler TypedHandler[T]) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		data, ok := factory.Extract(event)
		if !ok {
			return nil, nil
		}
		return handler(ctx, data)
	}
}

// StepConfig contains configuration for a workflow step.
type StepConfig struct {
	// Name is the name of the step (for debugging/logging).
	Name string
	// NumWorkers is the number of concurrent workers for this step.
	NumWorkers int
	// RetryPolicy configures retry behavior for this step.
	RetryPolicy *RetryPolicy
}

// DefaultStepConfig returns the default step configuration.
func DefaultStepConfig() StepConfig {
	return StepConfig{
		NumWorkers: 1,
	}
}

// RetryPolicy configures retry behavior for workflow steps.
type RetryPolicy struct {
	// MaxRetries is the maximum number of retry attempts.
	MaxRetries int
	// InitialDelay is the initial delay between retries.
	InitialDelay time.Duration
	// MaxDelay is the maximum delay between retries.
	MaxDelay time.Duration
	// Multiplier is the multiplier for exponential backoff.
	Multiplier float64
	// RetryOn is a function that determines if an error should be retried.
	RetryOn func(error) bool
}

// DefaultRetryPolicy returns a default retry policy.
func DefaultRetryPolicy() *RetryPolicy {
	return &RetryPolicy{
		MaxRetries:   3,
		InitialDelay: 100 * time.Millisecond,
		MaxDelay:     5 * time.Second,
		Multiplier:   2.0,
		RetryOn:      func(err error) bool { return true },
	}
}

// Step represents a registered workflow step.
type Step struct {
	// AcceptedEvents is the list of event types this step handles.
	AcceptedEvents []EventType
	// Handler is the function that processes events.
	Handler Handler
	// Config is the step configuration.
	Config StepConfig
}

// WorkflowResult represents the result of a workflow execution.
type WorkflowResult struct {
	// FinalEvent is the event that caused the workflow to stop.
	FinalEvent Event
	// State is the final state of the workflow.
	State *StateStore
	// Error is any error that occurred during execution.
	Error error
	// Duration is how long the workflow took to execute.
	Duration time.Duration
}

// WorkflowStream provides streaming access to workflow events.
type WorkflowStream struct {
	events  chan Event
	done    chan struct{}
	err     error
	mu      sync.RWMutex
	stopOn  []EventType
	context *Context
}

// NewWorkflowStream creates a new workflow stream.
func NewWorkflowStream(ctx *Context) *WorkflowStream {
	return &WorkflowStream{
		events:  make(chan Event, 100),
		done:    make(chan struct{}),
		context: ctx,
	}
}

// Events returns the channel of events.
func (s *WorkflowStream) Events() <-chan Event {
	return s.events
}

// Done returns a channel that's closed when the stream is complete.
func (s *WorkflowStream) Done() <-chan struct{} {
	return s.done
}

// Err returns any error that occurred.
func (s *WorkflowStream) Err() error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.err
}

// Until configures the stream to stop when an event of the given type is received.
func (s *WorkflowStream) Until(eventTypes ...EventType) *WorkflowStream {
	s.stopOn = eventTypes
	return s
}

// UntilFactory configures the stream to stop when an event from the given factory is received.
func (s *WorkflowStream) UntilFactory(factories ...interface{ Type() EventType }) *WorkflowStream {
	types := make([]EventType, len(factories))
	for i, f := range factories {
		types[i] = f.Type()
	}
	return s.Until(types...)
}

// ToArray collects all events into a slice.
func (s *WorkflowStream) ToArray() ([]Event, error) {
	var events []Event
	for event := range s.events {
		events = append(events, event)
	}
	return events, s.Err()
}

// shouldStop checks if the stream should stop on this event.
func (s *WorkflowStream) shouldStop(event Event) bool {
	if len(s.stopOn) == 0 {
		return false
	}
	for _, t := range s.stopOn {
		if event.Type() == t {
			return true
		}
	}
	return false
}

// emit sends an event to the stream.
func (s *WorkflowStream) emit(event Event) {
	select {
	case s.events <- event:
	default:
		// Buffer full, drop event
	}
}

// close closes the stream.
func (s *WorkflowStream) close(err error) {
	s.mu.Lock()
	s.err = err
	s.mu.Unlock()
	close(s.events)
	close(s.done)
}
