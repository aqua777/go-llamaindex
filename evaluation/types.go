// Package evaluation provides evaluation metrics for RAG systems.
package evaluation

import (
	"context"

	"github.com/aqua777/go-llamaindex/schema"
)

// EvaluationResult represents the result of an evaluation.
type EvaluationResult struct {
	// Query is the original query string.
	Query string `json:"query,omitempty"`
	// Contexts are the context strings used.
	Contexts []string `json:"contexts,omitempty"`
	// Response is the generated response string.
	Response string `json:"response,omitempty"`
	// Reference is the reference/ground truth answer.
	Reference string `json:"reference,omitempty"`
	// Passing indicates if the evaluation passed (binary result).
	Passing *bool `json:"passing,omitempty"`
	// Feedback is the reasoning or feedback for the evaluation.
	Feedback string `json:"feedback,omitempty"`
	// Score is the numerical score for the evaluation.
	Score *float64 `json:"score,omitempty"`
	// InvalidResult indicates if the evaluation result is invalid.
	InvalidResult bool `json:"invalid_result,omitempty"`
	// InvalidReason is the reason for an invalid evaluation.
	InvalidReason string `json:"invalid_reason,omitempty"`
	// Metadata contains additional evaluation metadata.
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// NewEvaluationResult creates a new EvaluationResult.
func NewEvaluationResult() *EvaluationResult {
	return &EvaluationResult{
		Metadata: make(map[string]interface{}),
	}
}

// WithQuery sets the query.
func (r *EvaluationResult) WithQuery(query string) *EvaluationResult {
	r.Query = query
	return r
}

// WithContexts sets the contexts.
func (r *EvaluationResult) WithContexts(contexts []string) *EvaluationResult {
	r.Contexts = contexts
	return r
}

// WithResponse sets the response.
func (r *EvaluationResult) WithResponse(response string) *EvaluationResult {
	r.Response = response
	return r
}

// WithReference sets the reference answer.
func (r *EvaluationResult) WithReference(reference string) *EvaluationResult {
	r.Reference = reference
	return r
}

// WithPassing sets the passing status.
func (r *EvaluationResult) WithPassing(passing bool) *EvaluationResult {
	r.Passing = &passing
	return r
}

// WithFeedback sets the feedback.
func (r *EvaluationResult) WithFeedback(feedback string) *EvaluationResult {
	r.Feedback = feedback
	return r
}

// WithScore sets the score.
func (r *EvaluationResult) WithScore(score float64) *EvaluationResult {
	r.Score = &score
	return r
}

// WithInvalid marks the result as invalid.
func (r *EvaluationResult) WithInvalid(reason string) *EvaluationResult {
	r.InvalidResult = true
	r.InvalidReason = reason
	return r
}

// IsPassing returns true if the evaluation passed.
func (r *EvaluationResult) IsPassing() bool {
	if r.Passing == nil {
		return false
	}
	return *r.Passing
}

// GetScore returns the score or 0 if not set.
func (r *EvaluationResult) GetScore() float64 {
	if r.Score == nil {
		return 0
	}
	return *r.Score
}

// EvaluateInput contains the input for an evaluation.
type EvaluateInput struct {
	// Query is the query string.
	Query string
	// Response is the generated response.
	Response string
	// Contexts are the retrieved contexts.
	Contexts []string
	// Reference is the reference/ground truth answer.
	Reference string
	// Extra contains additional evaluation parameters.
	Extra map[string]interface{}
}

// NewEvaluateInput creates a new EvaluateInput.
func NewEvaluateInput() *EvaluateInput {
	return &EvaluateInput{
		Extra: make(map[string]interface{}),
	}
}

// WithQuery sets the query.
func (i *EvaluateInput) WithQuery(query string) *EvaluateInput {
	i.Query = query
	return i
}

// WithResponse sets the response.
func (i *EvaluateInput) WithResponse(response string) *EvaluateInput {
	i.Response = response
	return i
}

// WithContexts sets the contexts.
func (i *EvaluateInput) WithContexts(contexts []string) *EvaluateInput {
	i.Contexts = contexts
	return i
}

// WithReference sets the reference.
func (i *EvaluateInput) WithReference(reference string) *EvaluateInput {
	i.Reference = reference
	return i
}

// Evaluator is the interface for all evaluators.
type Evaluator interface {
	// Evaluate runs the evaluation with the given input.
	Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error)

	// Name returns the name of the evaluator.
	Name() string
}

// ResponseEvaluator extends Evaluator with response object support.
type ResponseEvaluator interface {
	Evaluator

	// EvaluateResponse evaluates a response with source nodes.
	EvaluateResponse(ctx context.Context, query string, response string, sourceNodes []schema.NodeWithScore) (*EvaluationResult, error)
}

// BaseEvaluator provides common functionality for evaluators.
type BaseEvaluator struct {
	name string
}

// BaseEvaluatorOption configures a BaseEvaluator.
type BaseEvaluatorOption func(*BaseEvaluator)

// WithEvaluatorName sets the evaluator name.
func WithEvaluatorName(name string) BaseEvaluatorOption {
	return func(e *BaseEvaluator) {
		e.name = name
	}
}

// NewBaseEvaluator creates a new BaseEvaluator.
func NewBaseEvaluator(opts ...BaseEvaluatorOption) *BaseEvaluator {
	e := &BaseEvaluator{
		name: "base_evaluator",
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Name returns the evaluator name.
func (e *BaseEvaluator) Name() string {
	return e.name
}

// EvaluateResponse evaluates a response with source nodes.
// This is a helper that extracts contexts from source nodes.
// Concrete implementations should override the Evaluate method.
func (e *BaseEvaluator) EvaluateResponse(ctx context.Context, query string, response string, sourceNodes []schema.NodeWithScore) (*EvaluationResult, error) {
	// Extract contexts from source nodes
	contexts := make([]string, len(sourceNodes))
	for i, node := range sourceNodes {
		contexts[i] = node.Node.GetContent(schema.MetadataModeNone)
	}

	_ = NewEvaluateInput().
		WithQuery(query).
		WithResponse(response).
		WithContexts(contexts)

	// This should be overridden by concrete implementations
	return nil, nil
}

// EvaluatorRegistry holds registered evaluators.
type EvaluatorRegistry struct {
	evaluators map[string]Evaluator
}

// NewEvaluatorRegistry creates a new EvaluatorRegistry.
func NewEvaluatorRegistry() *EvaluatorRegistry {
	return &EvaluatorRegistry{
		evaluators: make(map[string]Evaluator),
	}
}

// Register adds an evaluator to the registry.
func (r *EvaluatorRegistry) Register(evaluator Evaluator) {
	r.evaluators[evaluator.Name()] = evaluator
}

// Get returns an evaluator by name.
func (r *EvaluatorRegistry) Get(name string) (Evaluator, bool) {
	e, ok := r.evaluators[name]
	return e, ok
}

// List returns all registered evaluator names.
func (r *EvaluatorRegistry) List() []string {
	names := make([]string, 0, len(r.evaluators))
	for name := range r.evaluators {
		names = append(names, name)
	}
	return names
}

// All returns all registered evaluators.
func (r *EvaluatorRegistry) All() map[string]Evaluator {
	return r.evaluators
}
