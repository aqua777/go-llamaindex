package callbacks

import (
	"sync"

	"github.com/google/uuid"
)

// CallbackManager handles callbacks for events within LlamaIndex.
type CallbackManager struct {
	handlers   []CallbackHandler
	traceMap   map[string][]string
	traceStack []string
	traceIDs   []string
	mu         sync.RWMutex
}

// CallbackManagerOption configures a CallbackManager.
type CallbackManagerOption func(*CallbackManager)

// WithHandlers sets the handlers.
func WithHandlers(handlers []CallbackHandler) CallbackManagerOption {
	return func(m *CallbackManager) {
		m.handlers = handlers
	}
}

// NewCallbackManager creates a new CallbackManager.
func NewCallbackManager(opts ...CallbackManagerOption) *CallbackManager {
	m := &CallbackManager{
		handlers:   []CallbackHandler{},
		traceMap:   make(map[string][]string),
		traceStack: []string{BaseTraceEvent},
		traceIDs:   []string{},
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// OnEventStart runs handlers when an event starts and returns the event ID.
func (m *CallbackManager) OnEventStart(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
	parentID string,
) string {
	m.mu.Lock()
	defer m.mu.Unlock()

	if eventID == "" {
		eventID = uuid.New().String()
	}

	// Get parent ID from trace stack if not provided
	if parentID == "" {
		if len(m.traceStack) > 0 {
			parentID = m.traceStack[len(m.traceStack)-1]
		} else {
			// Start a default trace if none is running
			m.startTraceInternal("llama-index")
			parentID = m.traceStack[len(m.traceStack)-1]
		}
	}

	// Add to trace map
	m.traceMap[parentID] = append(m.traceMap[parentID], eventID)

	// Call handlers
	for _, handler := range m.handlers {
		if !m.shouldIgnoreEventStart(handler, eventType) {
			handler.OnEventStart(eventType, payload, eventID, parentID)
		}
	}

	// Push to trace stack if not a leaf event
	if !IsLeafEvent(eventType) {
		m.traceStack = append(m.traceStack, eventID)
	}

	return eventID
}

// OnEventEnd runs handlers when an event ends.
func (m *CallbackManager) OnEventEnd(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if eventID == "" {
		eventID = uuid.New().String()
	}

	// Call handlers
	for _, handler := range m.handlers {
		if !m.shouldIgnoreEventEnd(handler, eventType) {
			handler.OnEventEnd(eventType, payload, eventID)
		}
	}

	// Pop from trace stack if not a leaf event
	if !IsLeafEvent(eventType) && len(m.traceStack) > 0 {
		m.traceStack = m.traceStack[:len(m.traceStack)-1]
	}
}

// AddHandler adds a handler to the callback manager.
func (m *CallbackManager) AddHandler(handler CallbackHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.handlers = append(m.handlers, handler)
}

// RemoveHandler removes a handler from the callback manager.
func (m *CallbackManager) RemoveHandler(handler CallbackHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()

	for i, h := range m.handlers {
		if h == handler {
			m.handlers = append(m.handlers[:i], m.handlers[i+1:]...)
			return
		}
	}
}

// SetHandlers sets handlers as the only handlers on the callback manager.
func (m *CallbackManager) SetHandlers(handlers []CallbackHandler) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.handlers = handlers
}

// Handlers returns the current handlers.
func (m *CallbackManager) Handlers() []CallbackHandler {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.handlers
}

// StartTrace starts an overall trace.
func (m *CallbackManager) StartTrace(traceID string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.startTraceInternal(traceID)
}

// startTraceInternal is the internal implementation of StartTrace (must hold lock).
func (m *CallbackManager) startTraceInternal(traceID string) {
	if traceID != "" {
		if len(m.traceIDs) == 0 {
			m.resetTraceEvents()

			for _, handler := range m.handlers {
				handler.StartTrace(traceID)
			}

			m.traceIDs = []string{traceID}
		} else {
			m.traceIDs = append(m.traceIDs, traceID)
		}
	}
}

// EndTrace ends an overall trace.
func (m *CallbackManager) EndTrace(traceID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if traceID != "" && len(m.traceIDs) > 0 {
		m.traceIDs = m.traceIDs[:len(m.traceIDs)-1]
		if len(m.traceIDs) == 0 {
			for _, handler := range m.handlers {
				handler.EndTrace(traceID, m.traceMap)
			}
			m.traceIDs = []string{}
		}
	}
}

// TraceMap returns the current trace map.
func (m *CallbackManager) TraceMap() map[string][]string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.traceMap
}

// resetTraceEvents resets the current trace.
func (m *CallbackManager) resetTraceEvents() {
	m.traceMap = make(map[string][]string)
	m.traceStack = []string{BaseTraceEvent}
}

// shouldIgnoreEventStart checks if an event should be ignored on start.
func (m *CallbackManager) shouldIgnoreEventStart(handler CallbackHandler, eventType CBEventType) bool {
	for _, e := range handler.EventStartsToIgnore() {
		if e == eventType {
			return true
		}
	}
	return false
}

// shouldIgnoreEventEnd checks if an event should be ignored on end.
func (m *CallbackManager) shouldIgnoreEventEnd(handler CallbackHandler, eventType CBEventType) bool {
	for _, e := range handler.EventEndsToIgnore() {
		if e == eventType {
			return true
		}
	}
	return false
}

// EventContext is a wrapper to call callbacks on event starts and ends.
type EventContext struct {
	manager   *CallbackManager
	eventType CBEventType
	eventID   string
	started   bool
	finished  bool
}

// NewEventContext creates a new EventContext.
func NewEventContext(manager *CallbackManager, eventType CBEventType, eventID string) *EventContext {
	if eventID == "" {
		eventID = uuid.New().String()
	}
	return &EventContext{
		manager:   manager,
		eventType: eventType,
		eventID:   eventID,
		started:   false,
		finished:  false,
	}
}

// OnStart triggers the event start.
func (e *EventContext) OnStart(payload map[string]interface{}) {
	if !e.started {
		e.started = true
		e.manager.OnEventStart(e.eventType, payload, e.eventID, "")
	}
}

// OnEnd triggers the event end.
func (e *EventContext) OnEnd(payload map[string]interface{}) {
	if !e.finished {
		e.finished = true
		e.manager.OnEventEnd(e.eventType, payload, e.eventID)
	}
}

// EventID returns the event ID.
func (e *EventContext) EventID() string {
	return e.eventID
}

// IsStarted returns whether the event has started.
func (e *EventContext) IsStarted() bool {
	return e.started
}

// IsFinished returns whether the event has finished.
func (e *EventContext) IsFinished() bool {
	return e.finished
}

// Event creates an EventContext for the given event type.
func (m *CallbackManager) Event(eventType CBEventType, eventID string) *EventContext {
	return NewEventContext(m, eventType, eventID)
}

// WithEvent executes a function within an event context.
func (m *CallbackManager) WithEvent(
	eventType CBEventType,
	startPayload map[string]interface{},
	fn func() (map[string]interface{}, error),
) error {
	ctx := m.Event(eventType, "")
	ctx.OnStart(startPayload)

	endPayload, err := fn()
	if err != nil {
		// Add exception to payload
		if endPayload == nil {
			endPayload = make(map[string]interface{})
		}
		endPayload[string(EventPayloadException)] = err
	}

	ctx.OnEnd(endPayload)
	return err
}

// WithTrace executes a function within a trace context.
func (m *CallbackManager) WithTrace(traceID string, fn func() error) error {
	m.StartTrace(traceID)
	defer m.EndTrace(traceID)

	err := fn()
	if err != nil {
		m.OnEventStart(CBEventTypeException, map[string]interface{}{
			string(EventPayloadException): err,
		}, "", "")
	}

	return err
}
