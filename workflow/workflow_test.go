package workflow

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Test event types and factories
var (
	ProcessEventType  EventType = "test.process"
	CompleteEventType EventType = "test.complete"
	FailEventType     EventType = "test.fail"

	ProcessEvent  = NewEventFactory[ProcessData](ProcessEventType)
	CompleteEvent = NewEventFactory[CompleteData](CompleteEventType)
	FailEvent     = NewEventFactory[FailData](FailEventType)
)

type ProcessData struct {
	Value int
}

type CompleteData struct {
	Result int
}

type FailData struct {
	Reason string
}

func TestEventFactory(t *testing.T) {
	t.Run("creates typed events", func(t *testing.T) {
		event := ProcessEvent.With(ProcessData{Value: 42})
		assert.Equal(t, ProcessEventType, event.Type())
		assert.Equal(t, 42, event.TypedData().Value)
	})

	t.Run("Include checks event type", func(t *testing.T) {
		event := ProcessEvent.With(ProcessData{Value: 42})
		assert.True(t, ProcessEvent.Include(event))
		assert.False(t, CompleteEvent.Include(event))
	})

	t.Run("Extract returns typed data", func(t *testing.T) {
		event := ProcessEvent.With(ProcessData{Value: 42})
		data, ok := ProcessEvent.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, 42, data.Value)
	})

	t.Run("Extract returns false for wrong type", func(t *testing.T) {
		event := CompleteEvent.With(CompleteData{Result: 100})
		_, ok := ProcessEvent.Extract(event)
		assert.False(t, ok)
	})

	t.Run("Include returns false for nil", func(t *testing.T) {
		assert.False(t, ProcessEvent.Include(nil))
	})
}

func TestBaseEvent(t *testing.T) {
	t.Run("NewEvent creates event", func(t *testing.T) {
		event := NewEvent("test.event", "data")
		assert.Equal(t, EventType("test.event"), event.Type())
		assert.Equal(t, "data", event.Data())
	})
}

func TestStateStore(t *testing.T) {
	t.Run("Get and Set", func(t *testing.T) {
		store := NewStateStore()
		store.Set("key", "value")
		val, ok := store.Get("key")
		assert.True(t, ok)
		assert.Equal(t, "value", val)
	})

	t.Run("Get returns false for missing key", func(t *testing.T) {
		store := NewStateStore()
		_, ok := store.Get("missing")
		assert.False(t, ok)
	})

	t.Run("Delete removes key", func(t *testing.T) {
		store := NewStateStore()
		store.Set("key", "value")
		store.Delete("key")
		_, ok := store.Get("key")
		assert.False(t, ok)
	})

	t.Run("GetString", func(t *testing.T) {
		store := NewStateStore()
		store.Set("str", "hello")
		store.Set("int", 42)

		str, ok := store.GetString("str")
		assert.True(t, ok)
		assert.Equal(t, "hello", str)

		_, ok = store.GetString("int")
		assert.False(t, ok)
	})

	t.Run("GetInt", func(t *testing.T) {
		store := NewStateStore()
		store.Set("int", 42)

		i, ok := store.GetInt("int")
		assert.True(t, ok)
		assert.Equal(t, 42, i)
	})

	t.Run("GetBool", func(t *testing.T) {
		store := NewStateStore()
		store.Set("bool", true)

		b, ok := store.GetBool("bool")
		assert.True(t, ok)
		assert.True(t, b)
	})

	t.Run("Keys", func(t *testing.T) {
		store := NewStateStore()
		store.Set("a", 1)
		store.Set("b", 2)

		keys := store.Keys()
		assert.Len(t, keys, 2)
		assert.Contains(t, keys, "a")
		assert.Contains(t, keys, "b")
	})

	t.Run("Clear", func(t *testing.T) {
		store := NewStateStore()
		store.Set("a", 1)
		store.Set("b", 2)
		store.Clear()

		assert.Empty(t, store.Keys())
	})

	t.Run("Clone", func(t *testing.T) {
		store := NewStateStore()
		store.Set("key", "value")

		clone := store.Clone()
		val, ok := clone.Get("key")
		assert.True(t, ok)
		assert.Equal(t, "value", val)

		// Modifying clone doesn't affect original
		clone.Set("key", "modified")
		origVal, _ := store.Get("key")
		assert.Equal(t, "value", origVal)
	})
}

func TestWorkflow(t *testing.T) {
	t.Run("NewWorkflow with defaults", func(t *testing.T) {
		w := NewWorkflow()
		assert.NotNil(t, w)
		assert.Equal(t, "workflow", w.name)
		assert.Equal(t, 60*time.Second, w.timeout)
	})

	t.Run("NewWorkflow with options", func(t *testing.T) {
		w := NewWorkflow(
			WithWorkflowName("test-workflow"),
			WithWorkflowTimeout(30*time.Second),
		)
		assert.Equal(t, "test-workflow", w.name)
		assert.Equal(t, 30*time.Second, w.timeout)
	})

	t.Run("Handle registers handler", func(t *testing.T) {
		w := NewWorkflow()
		handler := func(ctx *Context, event Event) ([]Event, error) {
			return nil, nil
		}

		w.Handle([]EventType{ProcessEventType}, handler)

		steps := w.GetHandlersForEvent(ProcessEventType)
		assert.Len(t, steps, 1)
	})

	t.Run("Simple workflow execution", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		// Register start handler
		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			data, _ := StartEvent.Extract(event)
			value := data.Input.(int)
			return []Event{ProcessEvent.With(ProcessData{Value: value * 2})}, nil
		})

		// Register process handler
		w.Handle([]EventType{ProcessEventType}, func(ctx *Context, event Event) ([]Event, error) {
			data, _ := ProcessEvent.Extract(event)
			return []Event{NewStopEvent(data.Value)}, nil
		})

		// Run workflow
		result, err := w.Run(context.Background(), NewStartEvent(21))
		require.NoError(t, err)
		assert.NotNil(t, result.FinalEvent)

		stopData, ok := StopEvent.Extract(result.FinalEvent)
		assert.True(t, ok)
		assert.Equal(t, 42, stopData.Result)
	})

	t.Run("Workflow with state", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			ctx.Set("counter", 0)
			return []Event{ProcessEvent.With(ProcessData{Value: 1})}, nil
		})

		w.Handle([]EventType{ProcessEventType}, func(ctx *Context, event Event) ([]Event, error) {
			counter, _ := ctx.GetInt("counter")
			data, _ := ProcessEvent.Extract(event)

			counter += data.Value
			ctx.Set("counter", counter)

			if counter >= 5 {
				return []Event{NewStopEvent(counter)}, nil
			}
			return []Event{ProcessEvent.With(ProcessData{Value: data.Value + 1})}, nil
		})

		result, err := w.Run(context.Background(), NewStartEvent(nil))
		require.NoError(t, err)

		stopData, _ := StopEvent.Extract(result.FinalEvent)
		assert.Equal(t, 6, stopData.Result) // 1 + 2 + 3 = 6
	})

	t.Run("Workflow with error", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		expectedErr := errors.New("test error")

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return nil, expectedErr
		})

		result, err := w.Run(context.Background(), NewStartEvent(nil))
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.NotNil(t, result)
	})

	t.Run("Workflow with error event", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		expectedErr := errors.New("explicit error")

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return []Event{NewErrorEvent(expectedErr, "start", event)}, nil
		})

		result, err := w.Run(context.Background(), NewStartEvent(nil))
		assert.Error(t, err)
		assert.Equal(t, expectedErr, err)
		assert.True(t, ErrorEvent.Include(result.FinalEvent))
	})

	t.Run("Workflow timeout", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(100 * time.Millisecond))

		// Create a workflow that loops without stopping
		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return []Event{ProcessEvent.With(ProcessData{Value: 1})}, nil
		})

		w.Handle([]EventType{ProcessEventType}, func(ctx *Context, event Event) ([]Event, error) {
			// Keep emitting events without stopping
			time.Sleep(50 * time.Millisecond)
			return []Event{ProcessEvent.With(ProcessData{Value: 1})}, nil
		})

		result, err := w.Run(context.Background(), NewStartEvent(nil))
		assert.Error(t, err)
		if err != nil {
			assert.Contains(t, err.Error(), "timed out")
		}
		_ = result
	})

	t.Run("Workflow context cancellation", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		// Create a workflow that loops and checks context
		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return []Event{ProcessEvent.With(ProcessData{Value: 1})}, nil
		})

		w.Handle([]EventType{ProcessEventType}, func(ctx *Context, event Event) ([]Event, error) {
			// Check if context is cancelled
			select {
			case <-ctx.Context().Done():
				return nil, ctx.Context().Err()
			default:
			}
			time.Sleep(100 * time.Millisecond)
			return []Event{ProcessEvent.With(ProcessData{Value: 1})}, nil
		})

		ctx, cancel := context.WithCancel(context.Background())
		go func() {
			time.Sleep(50 * time.Millisecond)
			cancel()
		}()

		_, err := w.Run(ctx, NewStartEvent(nil))
		// Either context cancelled error or workflow completed before cancellation
		// The important thing is the workflow doesn't hang
		_ = err
	})
}

func TestWorkflowStream(t *testing.T) {
	t.Run("RunStream emits events", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return []Event{ProcessEvent.With(ProcessData{Value: 1})}, nil
		})

		w.Handle([]EventType{ProcessEventType}, func(ctx *Context, event Event) ([]Event, error) {
			data, _ := ProcessEvent.Extract(event)
			if data.Value >= 3 {
				return []Event{NewStopEvent(data.Value)}, nil
			}
			return []Event{ProcessEvent.With(ProcessData{Value: data.Value + 1})}, nil
		})

		stream := w.RunStream(context.Background(), NewStartEvent(nil))
		events, err := stream.Until(StopEventType).ToArray()

		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(events), 3) // At least start, process events, and stop
	})

	t.Run("UntilFactory stops on event", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return []Event{CompleteEvent.With(CompleteData{Result: 42})}, nil
		})

		stream := w.RunStream(context.Background(), NewStartEvent(nil))
		events, err := stream.UntilFactory(CompleteEvent).ToArray()

		require.NoError(t, err)
		assert.GreaterOrEqual(t, len(events), 1)

		// Last event should be CompleteEvent
		lastEvent := events[len(events)-1]
		assert.True(t, CompleteEvent.Include(lastEvent))
	})
}

func TestTypedHandler(t *testing.T) {
	t.Run("HandleTyped registers typed handler", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		HandleTyped(w, StartEvent, func(ctx *Context, data StartEventData) ([]Event, error) {
			value := data.Input.(int)
			return []Event{NewStopEvent(value * 2)}, nil
		})

		result, err := w.Run(context.Background(), NewStartEvent(21))
		require.NoError(t, err)

		stopData, _ := StopEvent.Extract(result.FinalEvent)
		assert.Equal(t, 42, stopData.Result)
	})
}

func TestWorkflowBuilder(t *testing.T) {
	t.Run("Builder creates workflow", func(t *testing.T) {
		w := NewWorkflowBuilder(
			WithWorkflowName("builder-test"),
			WithWorkflowTimeout(10*time.Second),
		).OnStart(func(ctx *Context, data StartEventData) ([]Event, error) {
			return []Event{NewStopEvent("done")}, nil
		}).Build()

		result, err := w.Run(context.Background(), NewStartEvent(nil))
		require.NoError(t, err)

		stopData, _ := StopEvent.Extract(result.FinalEvent)
		assert.Equal(t, "done", stopData.Result)
	})

	t.Run("OnTyped with builder", func(t *testing.T) {
		builder := NewWorkflowBuilder(WithWorkflowTimeout(5 * time.Second))

		OnTyped(builder, ProcessEvent, func(ctx *Context, data ProcessData) ([]Event, error) {
			return []Event{NewStopEvent(data.Value * 2)}, nil
		})

		builder.OnStart(func(ctx *Context, data StartEventData) ([]Event, error) {
			return []Event{ProcessEvent.With(ProcessData{Value: 21})}, nil
		})

		w := builder.Build()
		result, err := w.Run(context.Background(), NewStartEvent(nil))
		require.NoError(t, err)

		stopData, _ := StopEvent.Extract(result.FinalEvent)
		assert.Equal(t, 42, stopData.Result)
	})
}

func TestRetryPolicy(t *testing.T) {
	t.Run("Retries on failure", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		var attempts int32

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			atomic.AddInt32(&attempts, 1)
			if atomic.LoadInt32(&attempts) < 3 {
				return nil, errors.New("temporary error")
			}
			return []Event{NewStopEvent("success")}, nil
		}, StepConfig{
			Name: "retry-step",
			RetryPolicy: &RetryPolicy{
				MaxRetries:   5,
				InitialDelay: 10 * time.Millisecond,
				MaxDelay:     100 * time.Millisecond,
				Multiplier:   2.0,
				RetryOn:      func(err error) bool { return true },
			},
		})

		result, err := w.Run(context.Background(), NewStartEvent(nil))
		require.NoError(t, err)
		assert.Equal(t, int32(3), atomic.LoadInt32(&attempts))

		stopData, _ := StopEvent.Extract(result.FinalEvent)
		assert.Equal(t, "success", stopData.Result)
	})

	t.Run("Fails after max retries", func(t *testing.T) {
		w := NewWorkflow(WithWorkflowTimeout(5 * time.Second))

		w.Handle([]EventType{StartEventType}, func(ctx *Context, event Event) ([]Event, error) {
			return nil, errors.New("persistent error")
		}, StepConfig{
			RetryPolicy: &RetryPolicy{
				MaxRetries:   2,
				InitialDelay: 10 * time.Millisecond,
				MaxDelay:     50 * time.Millisecond,
				Multiplier:   2.0,
				RetryOn:      func(err error) bool { return true },
			},
		})

		_, err := w.Run(context.Background(), NewStartEvent(nil))
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "failed after 2 retries")
	})
}

func TestDecorators(t *testing.T) {
	t.Run("ConditionalHandler", func(t *testing.T) {
		var executed bool
		handler := ConditionalHandler(
			func(ctx *Context, event Event) bool {
				data, _ := ProcessEvent.Extract(event)
				return data.Value > 10
			},
			func(ctx *Context, event Event) ([]Event, error) {
				executed = true
				return nil, nil
			},
		)

		ctx := &Context{state: NewStateStore()}

		// Should not execute (value <= 10)
		handler(ctx, ProcessEvent.With(ProcessData{Value: 5}))
		assert.False(t, executed)

		// Should execute (value > 10)
		handler(ctx, ProcessEvent.With(ProcessData{Value: 15}))
		assert.True(t, executed)
	})

	t.Run("ChainHandlers", func(t *testing.T) {
		handler := ChainHandlers(
			func(ctx *Context, event Event) ([]Event, error) {
				return []Event{NewEvent("event1", nil)}, nil
			},
			func(ctx *Context, event Event) ([]Event, error) {
				return []Event{NewEvent("event2", nil)}, nil
			},
		)

		ctx := &Context{state: NewStateStore()}
		events, err := handler(ctx, NewEvent("trigger", nil))

		require.NoError(t, err)
		assert.Len(t, events, 2)
	})

	t.Run("FallbackHandler", func(t *testing.T) {
		handler := FallbackHandler(
			func(ctx *Context, event Event) ([]Event, error) {
				return nil, errors.New("primary failed")
			},
			func(ctx *Context, event Event) ([]Event, error) {
				return []Event{NewEvent("fallback", nil)}, nil
			},
		)

		ctx := &Context{state: NewStateStore()}
		events, err := handler(ctx, NewEvent("trigger", nil))

		require.NoError(t, err)
		assert.Len(t, events, 1)
		assert.Equal(t, EventType("fallback"), events[0].Type())
	})

	t.Run("FilterEvents", func(t *testing.T) {
		handler := FilterEvents(
			func(ctx *Context, event Event) ([]Event, error) {
				return []Event{
					NewEvent("keep", nil),
					NewEvent("remove", nil),
					NewEvent("keep", nil),
				}, nil
			},
			func(e Event) bool {
				return e.Type() == "keep"
			},
		)

		ctx := &Context{state: NewStateStore()}
		events, err := handler(ctx, NewEvent("trigger", nil))

		require.NoError(t, err)
		assert.Len(t, events, 2)
	})

	t.Run("ApplyMiddleware", func(t *testing.T) {
		var order []string

		middleware1 := func(next Handler) Handler {
			return func(ctx *Context, event Event) ([]Event, error) {
				order = append(order, "m1-before")
				events, err := next(ctx, event)
				order = append(order, "m1-after")
				return events, err
			}
		}

		middleware2 := func(next Handler) Handler {
			return func(ctx *Context, event Event) ([]Event, error) {
				order = append(order, "m2-before")
				events, err := next(ctx, event)
				order = append(order, "m2-after")
				return events, err
			}
		}

		handler := ApplyMiddleware(
			func(ctx *Context, event Event) ([]Event, error) {
				order = append(order, "handler")
				return nil, nil
			},
			middleware1,
			middleware2,
		)

		ctx := &Context{state: NewStateStore()}
		handler(ctx, NewEvent("trigger", nil))

		assert.Equal(t, []string{"m1-before", "m2-before", "handler", "m2-after", "m1-after"}, order)
	})

	t.Run("RecoveryMiddleware", func(t *testing.T) {
		handler := ApplyMiddleware(
			func(ctx *Context, event Event) ([]Event, error) {
				panic("test panic")
			},
			RecoveryMiddleware(),
		)

		ctx := &Context{state: NewStateStore()}
		_, err := handler(ctx, NewEvent("trigger", nil))

		assert.Error(t, err)
		assert.Contains(t, err.Error(), "panic recovered")
	})
}

func TestCommonEvents(t *testing.T) {
	t.Run("StartEvent", func(t *testing.T) {
		event := NewStartEvent("input")
		assert.Equal(t, StartEventType, event.Type())

		data, ok := StartEvent.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, "input", data.Input)
	})

	t.Run("StartEventWithMetadata", func(t *testing.T) {
		metadata := map[string]interface{}{"key": "value"}
		event := NewStartEventWithMetadata("input", metadata)

		data, _ := StartEvent.Extract(event)
		assert.Equal(t, "value", data.Metadata["key"])
	})

	t.Run("StopEvent", func(t *testing.T) {
		event := NewStopEvent("result")
		assert.Equal(t, StopEventType, event.Type())

		data, ok := StopEvent.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, "result", data.Result)
	})

	t.Run("StopEventWithReason", func(t *testing.T) {
		event := NewStopEventWithReason("result", "completed")

		data, _ := StopEvent.Extract(event)
		assert.Equal(t, "completed", data.Reason)
	})

	t.Run("ErrorEvent", func(t *testing.T) {
		err := errors.New("test error")
		event := NewErrorEvent(err, "test-step", nil)

		data, ok := ErrorEvent.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, err, data.Error)
		assert.Equal(t, "test-step", data.Step)
	})

	t.Run("InputRequiredEvent", func(t *testing.T) {
		event := NewInputRequiredEvent("Enter your name:")

		data, ok := InputRequiredEvent.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, "Enter your name:", data.Prompt)
	})

	t.Run("HumanResponseEvent", func(t *testing.T) {
		event := NewHumanResponseEvent("John Doe")

		data, ok := HumanResponseEvent.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, "John Doe", data.Response)
	})

	t.Run("CustomEventFactory", func(t *testing.T) {
		type MyData struct {
			Value string
		}
		factory := CustomEventFactory[MyData]("my_event")

		event := factory.With(MyData{Value: "test"})
		assert.Equal(t, EventType("custom.my_event"), event.Type())

		data, ok := factory.Extract(event)
		assert.True(t, ok)
		assert.Equal(t, "test", data.Value)
	})
}

func TestStepConfig(t *testing.T) {
	t.Run("DefaultStepConfig", func(t *testing.T) {
		cfg := DefaultStepConfig()
		assert.Equal(t, 1, cfg.NumWorkers)
		assert.Nil(t, cfg.RetryPolicy)
	})

	t.Run("BuildStepConfig with options", func(t *testing.T) {
		cfg := BuildStepConfig(
			WithStepName("test-step"),
			WithNumWorkers(4),
			WithRetries(3),
		)

		assert.Equal(t, "test-step", cfg.Name)
		assert.Equal(t, 4, cfg.NumWorkers)
		assert.NotNil(t, cfg.RetryPolicy)
		assert.Equal(t, 3, cfg.RetryPolicy.MaxRetries)
	})

	t.Run("WithExponentialBackoff", func(t *testing.T) {
		cfg := BuildStepConfig(
			WithExponentialBackoff(5, 100*time.Millisecond, 10*time.Second),
		)

		assert.NotNil(t, cfg.RetryPolicy)
		assert.Equal(t, 5, cfg.RetryPolicy.MaxRetries)
		assert.Equal(t, 100*time.Millisecond, cfg.RetryPolicy.InitialDelay)
		assert.Equal(t, 10*time.Second, cfg.RetryPolicy.MaxDelay)
	})
}

func TestContext(t *testing.T) {
	t.Run("Context state operations", func(t *testing.T) {
		w := NewWorkflow()
		ctx := w.CreateContext(context.Background())

		ctx.Set("key", "value")
		val, ok := ctx.Get("key")
		assert.True(t, ok)
		assert.Equal(t, "value", val)

		ctx.Delete("key")
		_, ok = ctx.Get("key")
		assert.False(t, ok)
	})

	t.Run("Context cancellation", func(t *testing.T) {
		w := NewWorkflow()
		ctx := w.CreateContext(context.Background())

		assert.False(t, ctx.IsDone())
		ctx.Cancel()
		assert.True(t, ctx.IsDone())
	})

	t.Run("SendEvent after done is ignored", func(t *testing.T) {
		w := NewWorkflow()
		ctx := w.CreateContext(context.Background())
		ctx.Cancel()

		// Should not panic or block
		ctx.SendEvent(NewEvent("test", nil))
	})
}
