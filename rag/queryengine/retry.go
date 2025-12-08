package queryengine

import (
	"context"
	"time"

	"github.com/aqua777/go-llamaindex/rag/synthesizer"
)

// RetryQueryEngine retries queries on failure.
type RetryQueryEngine struct {
	*BaseQueryEngine
	// QueryEngine is the underlying query engine.
	QueryEngine QueryEngine
	// MaxRetries is the maximum number of retries.
	MaxRetries int
	// RetryDelay is the delay between retries.
	RetryDelay time.Duration
}

// RetryQueryEngineOption is a functional option.
type RetryQueryEngineOption func(*RetryQueryEngine)

// WithMaxRetries sets the maximum number of retries.
func WithMaxRetries(maxRetries int) RetryQueryEngineOption {
	return func(rqe *RetryQueryEngine) {
		rqe.MaxRetries = maxRetries
	}
}

// WithRetryDelay sets the delay between retries.
func WithRetryDelay(delay time.Duration) RetryQueryEngineOption {
	return func(rqe *RetryQueryEngine) {
		rqe.RetryDelay = delay
	}
}

// NewRetryQueryEngine creates a new RetryQueryEngine.
func NewRetryQueryEngine(engine QueryEngine, opts ...RetryQueryEngineOption) *RetryQueryEngine {
	rqe := &RetryQueryEngine{
		BaseQueryEngine: NewBaseQueryEngine(),
		QueryEngine:     engine,
		MaxRetries:      3,
		RetryDelay:      time.Second,
	}

	for _, opt := range opts {
		opt(rqe)
	}

	return rqe
}

// Query executes a query with retries on failure.
func (rqe *RetryQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	var lastErr error

	for attempt := 0; attempt <= rqe.MaxRetries; attempt++ {
		// Check context cancellation
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Try query
		response, err := rqe.QueryEngine.Query(ctx, query)
		if err == nil {
			return response, nil
		}

		lastErr = err

		// Wait before retry (except on last attempt)
		if attempt < rqe.MaxRetries {
			select {
			case <-ctx.Done():
				return nil, ctx.Err()
			case <-time.After(rqe.RetryDelay):
			}
		}
	}

	return nil, lastErr
}

// Ensure RetryQueryEngine implements QueryEngine.
var _ QueryEngine = (*RetryQueryEngine)(nil)
