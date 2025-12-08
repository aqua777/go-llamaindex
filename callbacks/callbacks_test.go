package callbacks

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCBEventType tests the CBEventType enum.
func TestCBEventType(t *testing.T) {
	t.Run("Event type values", func(t *testing.T) {
		assert.Equal(t, CBEventType("chunking"), CBEventTypeChunking)
		assert.Equal(t, CBEventType("llm"), CBEventTypeLLM)
		assert.Equal(t, CBEventType("embedding"), CBEventTypeEmbedding)
		assert.Equal(t, CBEventType("query"), CBEventTypeQuery)
		assert.Equal(t, CBEventType("retrieve"), CBEventTypeRetrieve)
	})

	t.Run("IsLeafEvent", func(t *testing.T) {
		assert.True(t, IsLeafEvent(CBEventTypeChunking))
		assert.True(t, IsLeafEvent(CBEventTypeLLM))
		assert.True(t, IsLeafEvent(CBEventTypeEmbedding))
		assert.False(t, IsLeafEvent(CBEventTypeQuery))
		assert.False(t, IsLeafEvent(CBEventTypeRetrieve))
	})
}

// TestCBEvent tests the CBEvent struct.
func TestCBEvent(t *testing.T) {
	t.Run("NewCBEvent", func(t *testing.T) {
		payload := map[string]interface{}{"key": "value"}
		event := NewCBEvent(CBEventTypeLLM, payload)

		assert.Equal(t, CBEventTypeLLM, event.EventType)
		assert.Equal(t, payload, event.Payload)
		assert.NotEmpty(t, event.ID)
		assert.NotEmpty(t, event.Time)
	})

	t.Run("NewCBEvent with nil payload", func(t *testing.T) {
		event := NewCBEvent(CBEventTypeQuery, nil)

		assert.Equal(t, CBEventTypeQuery, event.EventType)
		assert.Nil(t, event.Payload)
	})
}

// TestEventStats tests the EventStats struct.
func TestEventStats(t *testing.T) {
	t.Run("NewEventStats", func(t *testing.T) {
		stats := NewEventStats(10.0, 5)

		assert.Equal(t, 10.0, stats.TotalSecs)
		assert.Equal(t, 2.0, stats.AverageSecs)
		assert.Equal(t, 5, stats.TotalCount)
	})

	t.Run("NewEventStats with zero count", func(t *testing.T) {
		stats := NewEventStats(0.0, 0)

		assert.Equal(t, 0.0, stats.TotalSecs)
		assert.Equal(t, 0.0, stats.AverageSecs)
		assert.Equal(t, 0, stats.TotalCount)
	})
}

// TestBaseCallbackHandler tests the BaseCallbackHandler.
func TestBaseCallbackHandler(t *testing.T) {
	t.Run("NewBaseCallbackHandler", func(t *testing.T) {
		handler := NewBaseCallbackHandler()

		assert.NotNil(t, handler)
		assert.Empty(t, handler.EventStartsToIgnore())
		assert.Empty(t, handler.EventEndsToIgnore())
	})

	t.Run("WithEventStartsToIgnore", func(t *testing.T) {
		handler := NewBaseCallbackHandler(
			WithEventStartsToIgnore([]CBEventType{CBEventTypeLLM, CBEventTypeEmbedding}),
		)

		assert.Len(t, handler.EventStartsToIgnore(), 2)
		assert.True(t, handler.ShouldIgnoreEventStart(CBEventTypeLLM))
		assert.False(t, handler.ShouldIgnoreEventStart(CBEventTypeQuery))
	})

	t.Run("WithEventEndsToIgnore", func(t *testing.T) {
		handler := NewBaseCallbackHandler(
			WithEventEndsToIgnore([]CBEventType{CBEventTypeChunking}),
		)

		assert.Len(t, handler.EventEndsToIgnore(), 1)
		assert.True(t, handler.ShouldIgnoreEventEnd(CBEventTypeChunking))
		assert.False(t, handler.ShouldIgnoreEventEnd(CBEventTypeLLM))
	})

	t.Run("OnEventStart returns eventID", func(t *testing.T) {
		handler := NewBaseCallbackHandler()
		eventID := handler.OnEventStart(CBEventTypeLLM, nil, "test-id", "parent-id")
		assert.Equal(t, "test-id", eventID)
	})
}

// TestCallbackManager tests the CallbackManager.
func TestCallbackManager(t *testing.T) {
	t.Run("NewCallbackManager", func(t *testing.T) {
		manager := NewCallbackManager()

		assert.NotNil(t, manager)
		assert.Empty(t, manager.Handlers())
	})

	t.Run("AddHandler", func(t *testing.T) {
		manager := NewCallbackManager()
		handler := NewBaseCallbackHandler()

		manager.AddHandler(handler)

		assert.Len(t, manager.Handlers(), 1)
	})

	t.Run("RemoveHandler", func(t *testing.T) {
		manager := NewCallbackManager()
		handler := NewBaseCallbackHandler()

		manager.AddHandler(handler)
		manager.RemoveHandler(handler)

		assert.Empty(t, manager.Handlers())
	})

	t.Run("SetHandlers", func(t *testing.T) {
		manager := NewCallbackManager()
		handler1 := NewBaseCallbackHandler()
		handler2 := NewBaseCallbackHandler()

		manager.SetHandlers([]CallbackHandler{handler1, handler2})

		assert.Len(t, manager.Handlers(), 2)
	})

	t.Run("OnEventStart and OnEventEnd", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		eventID := manager.OnEventStart(CBEventTypeQuery, map[string]interface{}{"query": "test"}, "", "")
		assert.NotEmpty(t, eventID)

		manager.OnEventEnd(CBEventTypeQuery, map[string]interface{}{"response": "result"}, eventID)

		assert.Len(t, collector.StartEvents(), 1)
		assert.Len(t, collector.EndEvents(), 1)
		assert.Equal(t, CBEventTypeQuery, collector.StartEvents()[0].EventType)
	})

	t.Run("StartTrace and EndTrace", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		manager.StartTrace("test-trace")
		manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		manager.OnEventEnd(CBEventTypeLLM, nil, "")
		manager.EndTrace("test-trace")

		assert.NotEmpty(t, manager.TraceMap())
	})

	t.Run("TraceMap", func(t *testing.T) {
		manager := NewCallbackManager()

		manager.StartTrace("test")
		eventID := manager.OnEventStart(CBEventTypeQuery, nil, "", "")
		manager.OnEventEnd(CBEventTypeQuery, nil, eventID)
		manager.EndTrace("test")

		traceMap := manager.TraceMap()
		assert.NotEmpty(t, traceMap)
	})
}

// TestEventContext tests the EventContext.
func TestEventContext(t *testing.T) {
	t.Run("NewEventContext", func(t *testing.T) {
		manager := NewCallbackManager()
		ctx := NewEventContext(manager, CBEventTypeLLM, "")

		assert.NotEmpty(t, ctx.EventID())
		assert.False(t, ctx.IsStarted())
		assert.False(t, ctx.IsFinished())
	})

	t.Run("OnStart and OnEnd", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))
		ctx := NewEventContext(manager, CBEventTypeLLM, "test-event")

		ctx.OnStart(map[string]interface{}{"prompt": "hello"})
		assert.True(t, ctx.IsStarted())
		assert.False(t, ctx.IsFinished())

		ctx.OnEnd(map[string]interface{}{"response": "world"})
		assert.True(t, ctx.IsFinished())

		assert.Len(t, collector.StartEvents(), 1)
		assert.Len(t, collector.EndEvents(), 1)
	})

	t.Run("OnStart called twice", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))
		ctx := NewEventContext(manager, CBEventTypeLLM, "")

		ctx.OnStart(nil)
		ctx.OnStart(nil) // Should be ignored

		assert.Len(t, collector.StartEvents(), 1)
	})

	t.Run("OnEnd called twice", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))
		ctx := NewEventContext(manager, CBEventTypeLLM, "")

		ctx.OnStart(nil)
		ctx.OnEnd(nil)
		ctx.OnEnd(nil) // Should be ignored

		assert.Len(t, collector.EndEvents(), 1)
	})
}

// TestCallbackManagerEvent tests the Event method.
func TestCallbackManagerEvent(t *testing.T) {
	t.Run("Event creates EventContext", func(t *testing.T) {
		manager := NewCallbackManager()
		ctx := manager.Event(CBEventTypeLLM, "")

		assert.NotNil(t, ctx)
		assert.NotEmpty(t, ctx.EventID())
	})

	t.Run("Event with custom ID", func(t *testing.T) {
		manager := NewCallbackManager()
		ctx := manager.Event(CBEventTypeLLM, "custom-id")

		assert.Equal(t, "custom-id", ctx.EventID())
	})
}

// TestCallbackManagerWithEvent tests the WithEvent method.
func TestCallbackManagerWithEvent(t *testing.T) {
	t.Run("WithEvent success", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		err := manager.WithEvent(CBEventTypeLLM, map[string]interface{}{"prompt": "test"}, func() (map[string]interface{}, error) {
			return map[string]interface{}{"response": "result"}, nil
		})

		require.NoError(t, err)
		assert.Len(t, collector.StartEvents(), 1)
		assert.Len(t, collector.EndEvents(), 1)
	})

	t.Run("WithEvent error", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		err := manager.WithEvent(CBEventTypeLLM, nil, func() (map[string]interface{}, error) {
			return nil, assert.AnError
		})

		require.Error(t, err)
		assert.Len(t, collector.EndEvents(), 1)
		// Check that exception was added to payload
		endPayload := collector.EndEvents()[0].Payload
		assert.NotNil(t, endPayload[string(EventPayloadException)])
	})
}

// TestCallbackManagerWithTrace tests the WithTrace method.
func TestCallbackManagerWithTrace(t *testing.T) {
	t.Run("WithTrace success", func(t *testing.T) {
		manager := NewCallbackManager()

		err := manager.WithTrace("test-trace", func() error {
			manager.OnEventStart(CBEventTypeLLM, nil, "", "")
			return nil
		})

		require.NoError(t, err)
	})

	t.Run("WithTrace error", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		err := manager.WithTrace("test-trace", func() error {
			return assert.AnError
		})

		require.Error(t, err)
		// Exception event should be logged
		assert.NotEmpty(t, collector.StartEvents())
	})
}

// TestLoggingHandler tests the LoggingHandler.
func TestLoggingHandler(t *testing.T) {
	t.Run("NewLoggingHandler", func(t *testing.T) {
		handler := NewLoggingHandler()
		assert.NotNil(t, handler)
	})

	t.Run("OnEventStart and OnEventEnd", func(t *testing.T) {
		var buf bytes.Buffer
		handler := NewLoggingHandler(WithWriter(&buf))
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		eventID := manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		time.Sleep(10 * time.Millisecond)
		manager.OnEventEnd(CBEventTypeLLM, nil, eventID)

		output := buf.String()
		assert.Contains(t, output, "llm started")
		assert.Contains(t, output, "llm completed")
	})

	t.Run("Verbose logging", func(t *testing.T) {
		var buf bytes.Buffer
		handler := NewLoggingHandler(WithWriter(&buf), WithVerbose(true))
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		eventID := manager.OnEventStart(CBEventTypeLLM, map[string]interface{}{"prompt": "test"}, "", "")
		manager.OnEventEnd(CBEventTypeLLM, map[string]interface{}{"response": "result"}, eventID)

		output := buf.String()
		assert.Contains(t, output, "Event START")
		assert.Contains(t, output, "Event END")
		assert.Contains(t, output, "prompt")
		assert.Contains(t, output, "response")
	})

	t.Run("StartTrace and EndTrace", func(t *testing.T) {
		var buf bytes.Buffer
		handler := NewLoggingHandler(WithWriter(&buf))
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		manager.StartTrace("test-trace")
		manager.EndTrace("test-trace")

		output := buf.String()
		assert.Contains(t, output, "Trace START")
		assert.Contains(t, output, "Trace END")
	})
}

// TestTokenCountingHandler tests the TokenCountingHandler.
func TestTokenCountingHandler(t *testing.T) {
	t.Run("NewTokenCountingHandler", func(t *testing.T) {
		handler := NewTokenCountingHandler()
		assert.NotNil(t, handler)
		assert.Equal(t, 0, handler.TotalLLMTokens())
	})

	t.Run("Count LLM tokens", func(t *testing.T) {
		handler := NewTokenCountingHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		eventID := manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		manager.OnEventEnd(CBEventTypeLLM, map[string]interface{}{
			"prompt_tokens":     100,
			"completion_tokens": 50,
		}, eventID)

		assert.Equal(t, 150, handler.TotalLLMTokens())
		assert.Equal(t, 100, handler.PromptTokens())
		assert.Equal(t, 50, handler.CompletionTokens())
		assert.Equal(t, 1, handler.LLMEventCount())
	})

	t.Run("Count embedding tokens", func(t *testing.T) {
		handler := NewTokenCountingHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		eventID := manager.OnEventStart(CBEventTypeEmbedding, nil, "", "")
		manager.OnEventEnd(CBEventTypeEmbedding, map[string]interface{}{
			"total_tokens": 200,
		}, eventID)

		assert.Equal(t, 200, handler.TotalEmbedTokens())
		assert.Equal(t, 1, handler.EmbedEventCount())
	})

	t.Run("Reset", func(t *testing.T) {
		handler := NewTokenCountingHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		eventID := manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		manager.OnEventEnd(CBEventTypeLLM, map[string]interface{}{
			"total_tokens": 100,
		}, eventID)

		handler.Reset()

		assert.Equal(t, 0, handler.TotalLLMTokens())
		assert.Equal(t, 0, handler.LLMEventCount())
	})
}

// TestEventCollectorHandler tests the EventCollectorHandler.
func TestEventCollectorHandler(t *testing.T) {
	t.Run("NewEventCollectorHandler", func(t *testing.T) {
		handler := NewEventCollectorHandler()
		assert.NotNil(t, handler)
		assert.Empty(t, handler.StartEvents())
		assert.Empty(t, handler.EndEvents())
	})

	t.Run("Collect events", func(t *testing.T) {
		handler := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		manager.OnEventStart(CBEventTypeLLM, map[string]interface{}{"key": "value"}, "", "")
		manager.OnEventStart(CBEventTypeQuery, nil, "", "")

		assert.Len(t, handler.StartEvents(), 2)
	})

	t.Run("GetEventsByType", func(t *testing.T) {
		handler := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		manager.OnEventStart(CBEventTypeQuery, nil, "", "")
		manager.OnEventStart(CBEventTypeLLM, nil, "", "")

		llmEvents := handler.GetEventsByType(CBEventTypeLLM)
		assert.Len(t, llmEvents, 2)
	})

	t.Run("Clear", func(t *testing.T) {
		handler := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		handler.Clear()

		assert.Empty(t, handler.StartEvents())
	})
}

// TestIgnoreEvents tests event ignoring functionality.
func TestIgnoreEvents(t *testing.T) {
	t.Run("Ignore event starts", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		collector.eventStartsToIgnore = []CBEventType{CBEventTypeLLM}

		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		manager.OnEventStart(CBEventTypeQuery, nil, "", "")

		// LLM event should be ignored
		assert.Len(t, collector.StartEvents(), 1)
		assert.Equal(t, CBEventTypeQuery, collector.StartEvents()[0].EventType)
	})

	t.Run("Ignore event ends", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		collector.eventEndsToIgnore = []CBEventType{CBEventTypeLLM}

		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		eventID1 := manager.OnEventStart(CBEventTypeLLM, nil, "", "")
		eventID2 := manager.OnEventStart(CBEventTypeQuery, nil, "", "")
		manager.OnEventEnd(CBEventTypeLLM, nil, eventID1)
		manager.OnEventEnd(CBEventTypeQuery, nil, eventID2)

		// LLM end event should be ignored
		assert.Len(t, collector.EndEvents(), 1)
		assert.Equal(t, CBEventTypeQuery, collector.EndEvents()[0].EventType)
	})
}

// TestTraceStack tests the trace stack functionality.
func TestTraceStack(t *testing.T) {
	t.Run("Nested events", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		manager.StartTrace("test")

		// Start query (non-leaf, should push to stack)
		queryID := manager.OnEventStart(CBEventTypeQuery, nil, "", "")

		// Start LLM (leaf, should not push to stack)
		llmID := manager.OnEventStart(CBEventTypeLLM, nil, "", "")

		// End LLM
		manager.OnEventEnd(CBEventTypeLLM, nil, llmID)

		// End query
		manager.OnEventEnd(CBEventTypeQuery, nil, queryID)

		manager.EndTrace("test")

		// Check trace map shows correct parent-child relationships
		traceMap := manager.TraceMap()
		assert.Contains(t, traceMap[BaseTraceEvent], queryID)
	})
}

// TestConcurrentAccess tests thread safety.
func TestConcurrentAccess(t *testing.T) {
	t.Run("Concurrent event handling", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		done := make(chan bool)

		// Start multiple goroutines
		for i := 0; i < 10; i++ {
			go func() {
				for j := 0; j < 100; j++ {
					eventID := manager.OnEventStart(CBEventTypeLLM, nil, "", "")
					manager.OnEventEnd(CBEventTypeLLM, nil, eventID)
				}
				done <- true
			}()
		}

		// Wait for all goroutines
		for i := 0; i < 10; i++ {
			<-done
		}

		// Should have 1000 start events and 1000 end events
		assert.Len(t, collector.StartEvents(), 1000)
		assert.Len(t, collector.EndEvents(), 1000)
	})
}

// TestEventPayload tests the EventPayload constants.
func TestEventPayload(t *testing.T) {
	t.Run("Payload constants", func(t *testing.T) {
		assert.Equal(t, EventPayload("documents"), EventPayloadDocuments)
		assert.Equal(t, EventPayload("formatted_prompt"), EventPayloadPrompt)
		assert.Equal(t, EventPayload("response"), EventPayloadResponse)
	})

	t.Run("Use payload in event", func(t *testing.T) {
		collector := NewEventCollectorHandler()
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{collector}))

		payload := map[string]interface{}{
			string(EventPayloadPrompt):   "Hello, world!",
			string(EventPayloadResponse): "Hi there!",
		}

		manager.OnEventStart(CBEventTypeLLM, payload, "", "")

		assert.Equal(t, "Hello, world!", collector.StartEvents()[0].Payload[string(EventPayloadPrompt)])
	})
}

// TestHandlerInterface tests that all handlers implement CallbackHandler.
func TestHandlerInterface(t *testing.T) {
	t.Run("BaseCallbackHandler implements CallbackHandler", func(t *testing.T) {
		var _ CallbackHandler = NewBaseCallbackHandler()
	})

	t.Run("LoggingHandler implements CallbackHandler", func(t *testing.T) {
		var _ CallbackHandler = NewLoggingHandler()
	})

	t.Run("TokenCountingHandler implements CallbackHandler", func(t *testing.T) {
		var _ CallbackHandler = NewTokenCountingHandler()
	})

	t.Run("EventCollectorHandler implements CallbackHandler", func(t *testing.T) {
		var _ CallbackHandler = NewEventCollectorHandler()
	})
}

// TestLoggingHandlerOutput tests the output format of LoggingHandler.
func TestLoggingHandlerOutput(t *testing.T) {
	t.Run("Output contains timestamp", func(t *testing.T) {
		var buf bytes.Buffer
		handler := NewLoggingHandler(WithWriter(&buf))
		manager := NewCallbackManager(WithHandlers([]CallbackHandler{handler}))

		manager.OnEventStart(CBEventTypeLLM, nil, "", "")

		output := buf.String()
		// Check that output contains date-like pattern
		assert.True(t, strings.Contains(output, "/"))
	})
}
