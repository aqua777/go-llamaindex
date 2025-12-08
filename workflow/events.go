package workflow

// Common event types used in workflows.
const (
	// StartEventType is the event type for workflow start events.
	StartEventType EventType = "workflow.start"
	// StopEventType is the event type for workflow stop events.
	StopEventType EventType = "workflow.stop"
	// ErrorEventType is the event type for error events.
	ErrorEventType EventType = "workflow.error"
	// InputRequiredEventType is the event type for human input required events.
	InputRequiredEventType EventType = "workflow.input_required"
	// HumanResponseEventType is the event type for human response events.
	HumanResponseEventType EventType = "workflow.human_response"
)

// StartEventData contains data for a start event.
type StartEventData struct {
	// Input is the initial input to the workflow.
	Input interface{}
	// Metadata contains optional metadata for the workflow.
	Metadata map[string]interface{}
}

// StopEventData contains data for a stop event.
type StopEventData struct {
	// Result is the final result of the workflow.
	Result interface{}
	// Reason is an optional reason for stopping.
	Reason string
}

// ErrorEventData contains data for an error event.
type ErrorEventData struct {
	// Error is the error that occurred.
	Error error
	// Step is the name of the step where the error occurred.
	Step string
	// Event is the event that was being processed when the error occurred.
	Event Event
}

// InputRequiredEventData contains data for an input required event.
type InputRequiredEventData struct {
	// Prompt is the prompt to display to the user.
	Prompt string
	// Prefix is an optional prefix for the input.
	Prefix string
}

// HumanResponseEventData contains data for a human response event.
type HumanResponseEventData struct {
	// Response is the human's response.
	Response string
}

// Pre-defined event factories for common events.
var (
	// StartEvent is the factory for start events.
	StartEvent = NewEventFactoryWithLabel[StartEventData](StartEventType, "start")
	// StopEvent is the factory for stop events.
	StopEvent = NewEventFactoryWithLabel[StopEventData](StopEventType, "stop")
	// ErrorEvent is the factory for error events.
	ErrorEvent = NewEventFactoryWithLabel[ErrorEventData](ErrorEventType, "error")
	// InputRequiredEvent is the factory for input required events.
	InputRequiredEvent = NewEventFactoryWithLabel[InputRequiredEventData](InputRequiredEventType, "input_required")
	// HumanResponseEvent is the factory for human response events.
	HumanResponseEvent = NewEventFactoryWithLabel[HumanResponseEventData](HumanResponseEventType, "human_response")
)

// NewStartEvent creates a new start event with the given input.
func NewStartEvent(input interface{}) *TypedEvent[StartEventData] {
	return StartEvent.With(StartEventData{Input: input})
}

// NewStartEventWithMetadata creates a new start event with input and metadata.
func NewStartEventWithMetadata(input interface{}, metadata map[string]interface{}) *TypedEvent[StartEventData] {
	return StartEvent.With(StartEventData{Input: input, Metadata: metadata})
}

// NewStopEvent creates a new stop event with the given result.
func NewStopEvent(result interface{}) *TypedEvent[StopEventData] {
	return StopEvent.With(StopEventData{Result: result})
}

// NewStopEventWithReason creates a new stop event with a result and reason.
func NewStopEventWithReason(result interface{}, reason string) *TypedEvent[StopEventData] {
	return StopEvent.With(StopEventData{Result: result, Reason: reason})
}

// NewErrorEvent creates a new error event.
func NewErrorEvent(err error, step string, event Event) *TypedEvent[ErrorEventData] {
	return ErrorEvent.With(ErrorEventData{Error: err, Step: step, Event: event})
}

// NewInputRequiredEvent creates a new input required event.
func NewInputRequiredEvent(prompt string) *TypedEvent[InputRequiredEventData] {
	return InputRequiredEvent.With(InputRequiredEventData{Prompt: prompt})
}

// NewHumanResponseEvent creates a new human response event.
func NewHumanResponseEvent(response string) *TypedEvent[HumanResponseEventData] {
	return HumanResponseEvent.With(HumanResponseEventData{Response: response})
}

// CustomEventFactory creates a custom event factory with a unique type.
func CustomEventFactory[T any](name string) *EventFactory[T] {
	return NewEventFactory[T](EventType("custom." + name))
}
