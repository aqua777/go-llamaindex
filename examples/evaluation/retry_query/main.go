// Package main demonstrates retry query engine for handling failures.
// This example corresponds to Python's evaluation/RetryQuery.ipynb
package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"strings"
	"sync/atomic"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/rag/queryengine"
	"github.com/aqua777/go-llamaindex/rag/synthesizer"
	"github.com/aqua777/go-llamaindex/schema"
)

func main() {
	ctx := context.Background()

	// 1. Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")
	fmt.Println("=== Retry Query Engine Demo ===")
	fmt.Println("\nLLM initialized")

	separator := strings.Repeat("=", 60)

	// 2. Create a flaky query engine that fails intermittently
	fmt.Println("\n" + separator)
	fmt.Println("=== Simulating Unreliable Query Engine ===")
	fmt.Println(separator)

	// Create a mock query engine that fails the first N times
	flakyEngine := NewFlakyQueryEngine(llmInstance, 2) // Fails first 2 attempts

	fmt.Println("\nCreated flaky query engine (fails first 2 attempts)")

	// 3. Wrap with retry query engine
	fmt.Println("\n" + separator)
	fmt.Println("=== Retry Query Engine Configuration ===")
	fmt.Println(separator)

	retryEngine := queryengine.NewRetryQueryEngine(
		flakyEngine,
		queryengine.WithMaxRetries(3),
		queryengine.WithRetryDelay(500*time.Millisecond),
	)

	fmt.Println("\nRetry engine configured:")
	fmt.Println("  - Max retries: 3")
	fmt.Println("  - Retry delay: 500ms")

	// 4. Execute query with retries
	fmt.Println("\n" + separator)
	fmt.Println("=== Executing Query with Retries ===")
	fmt.Println(separator)

	query := "What is the meaning of life?"
	fmt.Printf("\nQuery: %s\n", query)
	fmt.Println("\nAttempting query (watch for retries)...")

	startTime := time.Now()
	response, err := retryEngine.Query(ctx, query)
	elapsed := time.Since(startTime)

	if err != nil {
		log.Printf("Query failed after retries: %v", err)
	} else {
		fmt.Printf("\nSuccess! Response: %s\n", truncate(response.Response, 100))
		fmt.Printf("Total attempts: %d\n", flakyEngine.GetAttemptCount())
		fmt.Printf("Total time: %v\n", elapsed)
	}

	// 5. Test with engine that always fails
	fmt.Println("\n" + separator)
	fmt.Println("=== Testing Persistent Failure ===")
	fmt.Println(separator)

	alwaysFailEngine := NewAlwaysFailQueryEngine()
	retryAlwaysFail := queryengine.NewRetryQueryEngine(
		alwaysFailEngine,
		queryengine.WithMaxRetries(2),
		queryengine.WithRetryDelay(100*time.Millisecond),
	)

	fmt.Println("\nTesting engine that always fails...")
	fmt.Println("  - Max retries: 2")

	startTime = time.Now()
	_, err = retryAlwaysFail.Query(ctx, "test query")
	elapsed = time.Since(startTime)

	if err != nil {
		fmt.Printf("\nExpected failure after retries: %v\n", err)
		fmt.Printf("Total attempts: %d\n", alwaysFailEngine.GetAttemptCount())
		fmt.Printf("Total time: %v\n", elapsed)
	}

	// 6. Context cancellation
	fmt.Println("\n" + separator)
	fmt.Println("=== Context Cancellation ===")
	fmt.Println(separator)

	slowEngine := NewSlowQueryEngine(llmInstance, 2*time.Second)
	retrySlowEngine := queryengine.NewRetryQueryEngine(
		slowEngine,
		queryengine.WithMaxRetries(3),
		queryengine.WithRetryDelay(100*time.Millisecond),
	)

	// Create a context with timeout
	ctxWithTimeout, cancel := context.WithTimeout(ctx, 500*time.Millisecond)
	defer cancel()

	fmt.Println("\nTesting with 500ms timeout on slow engine (2s per query)...")

	startTime = time.Now()
	_, err = retrySlowEngine.Query(ctxWithTimeout, "test query")
	elapsed = time.Since(startTime)

	if err != nil {
		fmt.Printf("\nQuery cancelled: %v\n", err)
		fmt.Printf("Time before cancellation: %v\n", elapsed)
	}

	// 7. Different retry configurations
	fmt.Println("\n" + separator)
	fmt.Println("=== Retry Configuration Comparison ===")
	fmt.Println(separator)

	configs := []struct {
		name       string
		maxRetries int
		delay      time.Duration
		failCount  int
	}{
		{"Aggressive (5 retries, 100ms)", 5, 100 * time.Millisecond, 3},
		{"Conservative (2 retries, 1s)", 2, time.Second, 1},
		{"Balanced (3 retries, 500ms)", 3, 500 * time.Millisecond, 2},
	}

	for _, cfg := range configs {
		fmt.Printf("\n%s:\n", cfg.name)

		testEngine := NewFlakyQueryEngine(llmInstance, cfg.failCount)
		testRetryEngine := queryengine.NewRetryQueryEngine(
			testEngine,
			queryengine.WithMaxRetries(cfg.maxRetries),
			queryengine.WithRetryDelay(cfg.delay),
		)

		startTime = time.Now()
		response, err = testRetryEngine.Query(ctx, "test")
		elapsed = time.Since(startTime)

		if err != nil {
			fmt.Printf("  Result: FAILED after %d attempts\n", testEngine.GetAttemptCount())
		} else {
			fmt.Printf("  Result: SUCCESS after %d attempts\n", testEngine.GetAttemptCount())
		}
		fmt.Printf("  Time: %v\n", elapsed)
	}

	// 8. Practical use case: Rate limiting
	fmt.Println("\n" + separator)
	fmt.Println("=== Practical Use Case: Rate Limiting ===")
	fmt.Println(separator)

	rateLimitedEngine := NewRateLimitedQueryEngine(llmInstance, 3) // Rate limited for first 3 calls
	retryRateLimited := queryengine.NewRetryQueryEngine(
		rateLimitedEngine,
		queryengine.WithMaxRetries(5),
		queryengine.WithRetryDelay(200*time.Millisecond),
	)

	fmt.Println("\nSimulating rate-limited API (429 errors for first 3 calls)...")

	startTime = time.Now()
	response, err = retryRateLimited.Query(ctx, "What is Go programming?")
	elapsed = time.Since(startTime)

	if err != nil {
		fmt.Printf("\nFailed: %v\n", err)
	} else {
		fmt.Printf("\nSuccess after handling rate limits!\n")
		fmt.Printf("Response: %s\n", truncate(response.Response, 80))
		fmt.Printf("Total attempts: %d\n", rateLimitedEngine.GetAttemptCount())
		fmt.Printf("Total time: %v\n", elapsed)
	}

	// 9. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nRetry Query Engine Features:")
	fmt.Println("  - Automatic retry on transient failures")
	fmt.Println("  - Configurable max retries and delay")
	fmt.Println("  - Respects context cancellation")
	fmt.Println("  - Returns last error after all retries exhausted")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Handling API rate limits (429 errors)")
	fmt.Println("  - Network timeout recovery")
	fmt.Println("  - Transient service unavailability")
	fmt.Println("  - LLM provider temporary failures")
	fmt.Println()
	fmt.Println("Best Practices:")
	fmt.Println("  - Use exponential backoff for production")
	fmt.Println("  - Set reasonable timeout with context")
	fmt.Println("  - Log retry attempts for debugging")
	fmt.Println("  - Consider circuit breaker for persistent failures")

	fmt.Println("\n=== Retry Query Engine Demo Complete ===")
}

// FlakyQueryEngine is a mock query engine that fails the first N attempts.
type FlakyQueryEngine struct {
	llm          llm.LLM
	failCount    int
	attemptCount int32
}

// NewFlakyQueryEngine creates a new flaky query engine.
func NewFlakyQueryEngine(llmInstance llm.LLM, failCount int) *FlakyQueryEngine {
	return &FlakyQueryEngine{
		llm:       llmInstance,
		failCount: failCount,
	}
}

// Query implements QueryEngine.
func (e *FlakyQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	attempt := atomic.AddInt32(&e.attemptCount, 1)
	fmt.Printf("  Attempt %d: ", attempt)

	if int(attempt) <= e.failCount {
		fmt.Println("FAILED (simulated transient error)")
		return nil, errors.New("simulated transient failure")
	}

	fmt.Println("SUCCESS")

	// Actually query the LLM
	response, err := e.llm.Complete(ctx, query)
	if err != nil {
		return nil, err
	}

	return &synthesizer.Response{
		Response:    response,
		SourceNodes: []schema.NodeWithScore{},
	}, nil
}

// GetAttemptCount returns the number of attempts made.
func (e *FlakyQueryEngine) GetAttemptCount() int {
	return int(atomic.LoadInt32(&e.attemptCount))
}

// AlwaysFailQueryEngine always fails.
type AlwaysFailQueryEngine struct {
	attemptCount int32
}

// NewAlwaysFailQueryEngine creates a new always-fail query engine.
func NewAlwaysFailQueryEngine() *AlwaysFailQueryEngine {
	return &AlwaysFailQueryEngine{}
}

// Query implements QueryEngine.
func (e *AlwaysFailQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	attempt := atomic.AddInt32(&e.attemptCount, 1)
	fmt.Printf("  Attempt %d: FAILED (always fails)\n", attempt)
	return nil, errors.New("persistent failure")
}

// GetAttemptCount returns the number of attempts made.
func (e *AlwaysFailQueryEngine) GetAttemptCount() int {
	return int(atomic.LoadInt32(&e.attemptCount))
}

// SlowQueryEngine simulates a slow query engine.
type SlowQueryEngine struct {
	llm   llm.LLM
	delay time.Duration
}

// NewSlowQueryEngine creates a new slow query engine.
func NewSlowQueryEngine(llmInstance llm.LLM, delay time.Duration) *SlowQueryEngine {
	return &SlowQueryEngine{
		llm:   llmInstance,
		delay: delay,
	}
}

// Query implements QueryEngine.
func (e *SlowQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case <-time.After(e.delay):
		response, err := e.llm.Complete(ctx, query)
		if err != nil {
			return nil, err
		}
		return &synthesizer.Response{
			Response:    response,
			SourceNodes: []schema.NodeWithScore{},
		}, nil
	}
}

// RateLimitedQueryEngine simulates rate limiting.
type RateLimitedQueryEngine struct {
	llm            llm.LLM
	rateLimitCount int
	attemptCount   int32
}

// NewRateLimitedQueryEngine creates a new rate-limited query engine.
func NewRateLimitedQueryEngine(llmInstance llm.LLM, rateLimitCount int) *RateLimitedQueryEngine {
	return &RateLimitedQueryEngine{
		llm:            llmInstance,
		rateLimitCount: rateLimitCount,
	}
}

// Query implements QueryEngine.
func (e *RateLimitedQueryEngine) Query(ctx context.Context, query string) (*synthesizer.Response, error) {
	attempt := atomic.AddInt32(&e.attemptCount, 1)
	fmt.Printf("  Attempt %d: ", attempt)

	if int(attempt) <= e.rateLimitCount {
		fmt.Println("RATE LIMITED (429)")
		return nil, errors.New("rate limit exceeded (429)")
	}

	fmt.Println("SUCCESS")

	response, err := e.llm.Complete(ctx, query)
	if err != nil {
		return nil, err
	}

	return &synthesizer.Response{
		Response:    response,
		SourceNodes: []schema.NodeWithScore{},
	}, nil
}

// GetAttemptCount returns the number of attempts made.
func (e *RateLimitedQueryEngine) GetAttemptCount() int {
	return int(atomic.LoadInt32(&e.attemptCount))
}

// Ensure interfaces are implemented
var _ queryengine.QueryEngine = (*FlakyQueryEngine)(nil)
var _ queryengine.QueryEngine = (*AlwaysFailQueryEngine)(nil)
var _ queryengine.QueryEngine = (*SlowQueryEngine)(nil)
var _ queryengine.QueryEngine = (*RateLimitedQueryEngine)(nil)

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
