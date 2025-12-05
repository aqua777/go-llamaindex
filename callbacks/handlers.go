package callbacks

import (
	"fmt"
	"io"
	"os"
	"sync"
	"time"
)

// LoggingHandler is a callback handler that logs events.
type LoggingHandler struct {
	*BaseCallbackHandler
	writer    io.Writer
	verbose   bool
	mu        sync.Mutex
	startTime map[string]time.Time
}

// LoggingHandlerOption configures a LoggingHandler.
type LoggingHandlerOption func(*LoggingHandler)

// WithWriter sets the writer for logging.
func WithWriter(w io.Writer) LoggingHandlerOption {
	return func(h *LoggingHandler) {
		h.writer = w
	}
}

// WithVerbose sets verbose logging.
func WithVerbose(verbose bool) LoggingHandlerOption {
	return func(h *LoggingHandler) {
		h.verbose = verbose
	}
}

// NewLoggingHandler creates a new LoggingHandler.
func NewLoggingHandler(opts ...LoggingHandlerOption) *LoggingHandler {
	h := &LoggingHandler{
		BaseCallbackHandler: NewBaseCallbackHandler(),
		writer:              os.Stdout,
		verbose:             false,
		startTime:           make(map[string]time.Time),
	}

	for _, opt := range opts {
		opt(h)
	}

	return h
}

// OnEventStart logs the event start.
func (h *LoggingHandler) OnEventStart(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
	parentID string,
) string {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.startTime[eventID] = time.Now()

	if h.verbose {
		fmt.Fprintf(h.writer, "[%s] Event START: %s (id=%s, parent=%s)\n",
			time.Now().Format(TimestampFormat), eventType, eventID, parentID)
		for k, v := range payload {
			fmt.Fprintf(h.writer, "  %s: %v\n", k, v)
		}
	} else {
		fmt.Fprintf(h.writer, "[%s] %s started\n",
			time.Now().Format(TimestampFormat), eventType)
	}

	return eventID
}

// OnEventEnd logs the event end.
func (h *LoggingHandler) OnEventEnd(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
) {
	h.mu.Lock()
	defer h.mu.Unlock()

	duration := time.Duration(0)
	if start, ok := h.startTime[eventID]; ok {
		duration = time.Since(start)
		delete(h.startTime, eventID)
	}

	if h.verbose {
		fmt.Fprintf(h.writer, "[%s] Event END: %s (id=%s, duration=%v)\n",
			time.Now().Format(TimestampFormat), eventType, eventID, duration)
		for k, v := range payload {
			fmt.Fprintf(h.writer, "  %s: %v\n", k, v)
		}
	} else {
		fmt.Fprintf(h.writer, "[%s] %s completed (%v)\n",
			time.Now().Format(TimestampFormat), eventType, duration)
	}
}

// StartTrace logs the trace start.
func (h *LoggingHandler) StartTrace(traceID string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	fmt.Fprintf(h.writer, "[%s] Trace START: %s\n",
		time.Now().Format(TimestampFormat), traceID)
}

// EndTrace logs the trace end.
func (h *LoggingHandler) EndTrace(traceID string, traceMap map[string][]string) {
	h.mu.Lock()
	defer h.mu.Unlock()

	fmt.Fprintf(h.writer, "[%s] Trace END: %s\n",
		time.Now().Format(TimestampFormat), traceID)

	if h.verbose && len(traceMap) > 0 {
		fmt.Fprintf(h.writer, "  Trace map:\n")
		for parent, children := range traceMap {
			fmt.Fprintf(h.writer, "    %s -> %v\n", parent, children)
		}
	}
}

// Ensure LoggingHandler implements CallbackHandler.
var _ CallbackHandler = (*LoggingHandler)(nil)

// TokenCountingHandler tracks token usage.
type TokenCountingHandler struct {
	*BaseCallbackHandler
	mu               sync.Mutex
	totalLLMTokens   int
	promptTokens     int
	completionTokens int
	totalEmbedTokens int
	llmEventCount    int
	embedEventCount  int
}

// NewTokenCountingHandler creates a new TokenCountingHandler.
func NewTokenCountingHandler() *TokenCountingHandler {
	return &TokenCountingHandler{
		BaseCallbackHandler: NewBaseCallbackHandler(),
	}
}

// OnEventStart handles event start for token counting.
func (h *TokenCountingHandler) OnEventStart(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
	parentID string,
) string {
	return eventID
}

// OnEventEnd handles event end for token counting.
func (h *TokenCountingHandler) OnEventEnd(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if payload == nil {
		return
	}

	switch eventType {
	case CBEventTypeLLM:
		h.llmEventCount++
		if tokens, ok := payload["prompt_tokens"].(int); ok {
			h.promptTokens += tokens
			h.totalLLMTokens += tokens
		}
		if tokens, ok := payload["completion_tokens"].(int); ok {
			h.completionTokens += tokens
			h.totalLLMTokens += tokens
		}
		if tokens, ok := payload["total_tokens"].(int); ok {
			h.totalLLMTokens = tokens
		}

	case CBEventTypeEmbedding:
		h.embedEventCount++
		if tokens, ok := payload["total_tokens"].(int); ok {
			h.totalEmbedTokens += tokens
		}
	}
}

// StartTrace is a no-op for token counting.
func (h *TokenCountingHandler) StartTrace(traceID string) {}

// EndTrace is a no-op for token counting.
func (h *TokenCountingHandler) EndTrace(traceID string, traceMap map[string][]string) {}

// TotalLLMTokens returns the total LLM tokens.
func (h *TokenCountingHandler) TotalLLMTokens() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.totalLLMTokens
}

// PromptTokens returns the prompt tokens.
func (h *TokenCountingHandler) PromptTokens() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.promptTokens
}

// CompletionTokens returns the completion tokens.
func (h *TokenCountingHandler) CompletionTokens() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.completionTokens
}

// TotalEmbedTokens returns the total embedding tokens.
func (h *TokenCountingHandler) TotalEmbedTokens() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.totalEmbedTokens
}

// LLMEventCount returns the LLM event count.
func (h *TokenCountingHandler) LLMEventCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.llmEventCount
}

// EmbedEventCount returns the embedding event count.
func (h *TokenCountingHandler) EmbedEventCount() int {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.embedEventCount
}

// Reset resets all counters.
func (h *TokenCountingHandler) Reset() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.totalLLMTokens = 0
	h.promptTokens = 0
	h.completionTokens = 0
	h.totalEmbedTokens = 0
	h.llmEventCount = 0
	h.embedEventCount = 0
}

// Ensure TokenCountingHandler implements CallbackHandler.
var _ CallbackHandler = (*TokenCountingHandler)(nil)

// EventCollectorHandler collects events for later inspection.
type EventCollectorHandler struct {
	*BaseCallbackHandler
	mu          sync.Mutex
	startEvents []CollectedEvent
	endEvents   []CollectedEvent
}

// CollectedEvent represents a collected event.
type CollectedEvent struct {
	EventType CBEventType
	Payload   map[string]interface{}
	EventID   string
	ParentID  string
	Time      time.Time
}

// NewEventCollectorHandler creates a new EventCollectorHandler.
func NewEventCollectorHandler() *EventCollectorHandler {
	return &EventCollectorHandler{
		BaseCallbackHandler: NewBaseCallbackHandler(),
		startEvents:         []CollectedEvent{},
		endEvents:           []CollectedEvent{},
	}
}

// OnEventStart collects the event start.
func (h *EventCollectorHandler) OnEventStart(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
	parentID string,
) string {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.startEvents = append(h.startEvents, CollectedEvent{
		EventType: eventType,
		Payload:   payload,
		EventID:   eventID,
		ParentID:  parentID,
		Time:      time.Now(),
	})

	return eventID
}

// OnEventEnd collects the event end.
func (h *EventCollectorHandler) OnEventEnd(
	eventType CBEventType,
	payload map[string]interface{},
	eventID string,
) {
	h.mu.Lock()
	defer h.mu.Unlock()

	h.endEvents = append(h.endEvents, CollectedEvent{
		EventType: eventType,
		Payload:   payload,
		EventID:   eventID,
		Time:      time.Now(),
	})
}

// StartTrace is a no-op for event collection.
func (h *EventCollectorHandler) StartTrace(traceID string) {}

// EndTrace is a no-op for event collection.
func (h *EventCollectorHandler) EndTrace(traceID string, traceMap map[string][]string) {}

// StartEvents returns the collected start events.
func (h *EventCollectorHandler) StartEvents() []CollectedEvent {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.startEvents
}

// EndEvents returns the collected end events.
func (h *EventCollectorHandler) EndEvents() []CollectedEvent {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.endEvents
}

// Clear clears all collected events.
func (h *EventCollectorHandler) Clear() {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.startEvents = []CollectedEvent{}
	h.endEvents = []CollectedEvent{}
}

// GetEventsByType returns events of a specific type.
func (h *EventCollectorHandler) GetEventsByType(eventType CBEventType) []CollectedEvent {
	h.mu.Lock()
	defer h.mu.Unlock()

	var events []CollectedEvent
	for _, e := range h.startEvents {
		if e.EventType == eventType {
			events = append(events, e)
		}
	}
	return events
}

// Ensure EventCollectorHandler implements CallbackHandler.
var _ CallbackHandler = (*EventCollectorHandler)(nil)
