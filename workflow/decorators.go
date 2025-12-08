package workflow

import (
	"time"
)

// StepOption is a function that configures a StepConfig.
type StepOption func(*StepConfig)

// WithStepName sets the step name.
func WithStepName(name string) StepOption {
	return func(c *StepConfig) {
		c.Name = name
	}
}

// WithNumWorkers sets the number of workers for the step.
func WithNumWorkers(n int) StepOption {
	return func(c *StepConfig) {
		c.NumWorkers = n
	}
}

// WithRetryPolicy sets the retry policy for the step.
func WithRetryPolicy(policy *RetryPolicy) StepOption {
	return func(c *StepConfig) {
		c.RetryPolicy = policy
	}
}

// WithRetries configures simple retry behavior.
func WithRetries(maxRetries int) StepOption {
	return func(c *StepConfig) {
		c.RetryPolicy = &RetryPolicy{
			MaxRetries:   maxRetries,
			InitialDelay: 100 * time.Millisecond,
			MaxDelay:     5 * time.Second,
			Multiplier:   2.0,
			RetryOn:      func(err error) bool { return true },
		}
	}
}

// WithExponentialBackoff configures exponential backoff retry behavior.
func WithExponentialBackoff(maxRetries int, initialDelay, maxDelay time.Duration) StepOption {
	return func(c *StepConfig) {
		c.RetryPolicy = &RetryPolicy{
			MaxRetries:   maxRetries,
			InitialDelay: initialDelay,
			MaxDelay:     maxDelay,
			Multiplier:   2.0,
			RetryOn:      func(err error) bool { return true },
		}
	}
}

// BuildStepConfig creates a StepConfig from options.
func BuildStepConfig(opts ...StepOption) StepConfig {
	cfg := DefaultStepConfig()
	for _, opt := range opts {
		opt(&cfg)
	}
	return cfg
}

// Step decorator functions for common patterns.

// LoggingHandler wraps a handler with logging.
func LoggingHandler(name string, handler Handler) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		// Log before
		ctx.workflow.logger.Info("Step started", "step", name, "event_type", event.Type())

		// Execute
		events, err := handler(ctx, event)

		// Log after
		if err != nil {
			ctx.workflow.logger.Error("Step failed", "step", name, "error", err)
		} else {
			ctx.workflow.logger.Info("Step completed", "step", name, "events_emitted", len(events))
		}

		return events, err
	}
}

// TimingHandler wraps a handler with timing measurement.
func TimingHandler(name string, handler Handler) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		start := time.Now()
		events, err := handler(ctx, event)
		duration := time.Since(start)

		ctx.workflow.logger.Debug("Step timing", "step", name, "duration", duration)
		ctx.Set("_timing_"+name, duration)

		return events, err
	}
}

// ConditionalHandler wraps a handler with a condition check.
func ConditionalHandler(condition func(*Context, Event) bool, handler Handler) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		if !condition(ctx, event) {
			return nil, nil
		}
		return handler(ctx, event)
	}
}

// FallbackHandler wraps a handler with a fallback for errors.
func FallbackHandler(handler Handler, fallback Handler) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		events, err := handler(ctx, event)
		if err != nil {
			return fallback(ctx, event)
		}
		return events, nil
	}
}

// ChainHandlers chains multiple handlers together.
// Each handler's output events are collected and returned together.
func ChainHandlers(handlers ...Handler) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		var allEvents []Event
		for _, h := range handlers {
			events, err := h(ctx, event)
			if err != nil {
				return nil, err
			}
			allEvents = append(allEvents, events...)
		}
		return allEvents, nil
	}
}

// PipelineHandlers creates a pipeline where each handler processes
// the events from the previous handler.
func PipelineHandlers(handlers ...Handler) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		currentEvents := []Event{event}

		for _, h := range handlers {
			var nextEvents []Event
			for _, e := range currentEvents {
				events, err := h(ctx, e)
				if err != nil {
					return nil, err
				}
				nextEvents = append(nextEvents, events...)
			}
			currentEvents = nextEvents
		}

		return currentEvents, nil
	}
}

// FilterEvents filters events based on a predicate.
func FilterEvents(handler Handler, predicate func(Event) bool) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		events, err := handler(ctx, event)
		if err != nil {
			return nil, err
		}

		var filtered []Event
		for _, e := range events {
			if predicate(e) {
				filtered = append(filtered, e)
			}
		}
		return filtered, nil
	}
}

// MapEvents transforms events using a mapper function.
func MapEvents(handler Handler, mapper func(Event) Event) Handler {
	return func(ctx *Context, event Event) ([]Event, error) {
		events, err := handler(ctx, event)
		if err != nil {
			return nil, err
		}

		mapped := make([]Event, len(events))
		for i, e := range events {
			mapped[i] = mapper(e)
		}
		return mapped, nil
	}
}

// Middleware is a function that wraps a handler.
type Middleware func(Handler) Handler

// ApplyMiddleware applies middleware to a handler.
func ApplyMiddleware(handler Handler, middleware ...Middleware) Handler {
	for i := len(middleware) - 1; i >= 0; i-- {
		handler = middleware[i](handler)
	}
	return handler
}

// LoggingMiddleware creates logging middleware.
func LoggingMiddleware(name string) Middleware {
	return func(next Handler) Handler {
		return LoggingHandler(name, next)
	}
}

// TimingMiddleware creates timing middleware.
func TimingMiddleware(name string) Middleware {
	return func(next Handler) Handler {
		return TimingHandler(name, next)
	}
}

// RecoveryMiddleware creates panic recovery middleware.
func RecoveryMiddleware() Middleware {
	return func(next Handler) Handler {
		return func(ctx *Context, event Event) (events []Event, err error) {
			defer func() {
				if r := recover(); r != nil {
					if e, ok := r.(error); ok {
						err = e
					} else {
						err = &PanicError{Value: r}
					}
				}
			}()
			return next(ctx, event)
		}
	}
}

// PanicError represents a recovered panic.
type PanicError struct {
	Value interface{}
}

func (e *PanicError) Error() string {
	return "panic recovered: " + toString(e.Value)
}

func toString(v interface{}) string {
	if s, ok := v.(string); ok {
		return s
	}
	if e, ok := v.(error); ok {
		return e.Error()
	}
	return "unknown panic value"
}
