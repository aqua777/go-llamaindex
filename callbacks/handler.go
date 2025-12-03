package callbacks

// CallbackHandler is the interface for callback handlers.
type CallbackHandler interface {
	// OnEventStart is called when an event starts.
	// Returns the event ID.
	OnEventStart(
		eventType CBEventType,
		payload map[string]interface{},
		eventID string,
		parentID string,
	) string

	// OnEventEnd is called when an event ends.
	OnEventEnd(
		eventType CBEventType,
		payload map[string]interface{},
		eventID string,
	)

	// StartTrace is called when an overall trace is launched.
	StartTrace(traceID string)

	// EndTrace is called when an overall trace is exited.
	EndTrace(traceID string, traceMap map[string][]string)

	// EventStartsToIgnore returns event types to ignore on start.
	EventStartsToIgnore() []CBEventType

	// EventEndsToIgnore returns event types to ignore on end.
	EventEndsToIgnore() []CBEventType
}

// BaseCallbackHandler provides a base implementation of CallbackHandler.
type BaseCallbackHandler struct {
	eventStartsToIgnore []CBEventType
	eventEndsToIgnore   []CBEventType
}

// BaseCallbackHandlerOption configures a BaseCallbackHandler.
type BaseCallbackHandlerOption func(*BaseCallbackHandler)

// WithEventStartsToIgnore sets event types to ignore on start.
func WithEventStartsToIgnore(events []CBEventType) BaseCallbackHandlerOption {
	return func(h *BaseCallbackHandler) {
		h.eventStartsToIgnore = events
	}
}

// WithEventEndsToIgnore sets event types to ignore on end.
func WithEventEndsToIgnore(events []CBEventType) BaseCallbackHandlerOption {
	return func(h *BaseCallbackHandler) {
		h.eventEndsToIgnore = events
	}
}

// NewBaseCallbackHandler creates a new BaseCallbackHandler.
func NewBaseCallbackHandler(opts ...BaseCallbackHandlerOption) *BaseCallbackHandler {
	h := &BaseCallbackHandler{
		eventStartsToIgnore: []CBEventType{},
		eventEndsToIgnore:   []CBEventType{},
	}

	for _, opt := range opts {
		opt(h)
	}

	return h
}

// EventStartsToIgnore returns event types to ignore on start.
func (h *BaseCallbackHandler) EventStartsToIgnore() []CBEventType {
	return h.eventStartsToIgnore
}

// EventEndsToIgnore returns event types to ignore on end.
func (h *BaseCallbackHandler) EventEndsToIgnore() []CBEventType {
	return h.eventEndsToIgnore
}

// OnEventStart is a no-op implementation.
func (h *BaseCallbackHandler) OnEventStart(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
	parentID string,
) string {
	return eventID
}

// OnEventEnd is a no-op implementation.
func (h *BaseCallbackHandler) OnEventEnd(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
) {
}

// StartTrace is a no-op implementation.
func (h *BaseCallbackHandler) StartTrace(traceID string) {
}

// EndTrace is a no-op implementation.
func (h *BaseCallbackHandler) EndTrace(traceID string, traceMap map[string][]string) {
}

// ShouldIgnoreEventStart checks if an event type should be ignored on start.
func (h *BaseCallbackHandler) ShouldIgnoreEventStart(eventType CBEventType) bool {
	for _, e := range h.eventStartsToIgnore {
		if e == eventType {
			return true
		}
	}
	return false
}

// ShouldIgnoreEventEnd checks if an event type should be ignored on end.
func (h *BaseCallbackHandler) ShouldIgnoreEventEnd(eventType CBEventType) bool {
	for _, e := range h.eventEndsToIgnore {
		if e == eventType {
			return true
		}
	}
	return false
}

// Ensure BaseCallbackHandler implements CallbackHandler.
var _ CallbackHandler = (*BaseCallbackHandler)(nil)
