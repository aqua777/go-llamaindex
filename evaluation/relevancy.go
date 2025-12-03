package evaluation

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
)

// DefaultRelevancyTemplate is the default prompt template for relevancy evaluation.
const DefaultRelevancyTemplate = `Your task is to evaluate if the response for the query is in line with the context information provided.
You have two options to answer. Either YES or NO.
Answer YES if the response for the query is in line with context information, otherwise NO.

Query and Response:
{query_response}

Context:
{context}

Answer: `

// RelevancyEvaluator evaluates the relevancy of retrieved contexts and response to a query.
// This evaluator considers the query string, retrieved contexts, and response string.
type RelevancyEvaluator struct {
	*BaseEvaluator
	llm          llm.LLM
	evalTemplate string
	raiseError   bool
}

// RelevancyEvaluatorOption configures a RelevancyEvaluator.
type RelevancyEvaluatorOption func(*RelevancyEvaluator)

// WithRelevancyLLM sets the LLM for evaluation.
func WithRelevancyLLM(l llm.LLM) RelevancyEvaluatorOption {
	return func(e *RelevancyEvaluator) {
		e.llm = l
	}
}

// WithRelevancyTemplate sets the evaluation template.
func WithRelevancyTemplate(template string) RelevancyEvaluatorOption {
	return func(e *RelevancyEvaluator) {
		e.evalTemplate = template
	}
}

// WithRelevancyRaiseError sets whether to raise an error on failure.
func WithRelevancyRaiseError(raise bool) RelevancyEvaluatorOption {
	return func(e *RelevancyEvaluator) {
		e.raiseError = raise
	}
}

// NewRelevancyEvaluator creates a new RelevancyEvaluator.
func NewRelevancyEvaluator(opts ...RelevancyEvaluatorOption) *RelevancyEvaluator {
	e := &RelevancyEvaluator{
		BaseEvaluator: NewBaseEvaluator(WithEvaluatorName("relevancy")),
		evalTemplate:  DefaultRelevancyTemplate,
		raiseError:    false,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Evaluate evaluates whether the contexts and response are relevant to the query.
func (e *RelevancyEvaluator) Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error) {
	if input.Query == "" {
		return NewEvaluationResult().WithInvalid("query must be provided"), nil
	}
	if len(input.Contexts) == 0 {
		return NewEvaluationResult().WithInvalid("contexts must be provided"), nil
	}
	if input.Response == "" {
		return NewEvaluationResult().WithInvalid("response must be provided"), nil
	}
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for relevancy evaluation")
	}

	// Combine contexts
	contextStr := strings.Join(input.Contexts, "\n\n")

	// Format query and response
	queryResponse := fmt.Sprintf("Question: %s\nResponse: %s", input.Query, input.Response)

	// Format the prompt
	prompt := strings.ReplaceAll(e.evalTemplate, "{query_response}", queryResponse)
	prompt = strings.ReplaceAll(prompt, "{context}", contextStr)

	// Get LLM response
	llmResponse, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM evaluation failed: %w", err)
	}

	// Parse the response
	passing := strings.Contains(strings.ToLower(llmResponse), "yes")

	if !passing && e.raiseError {
		return nil, fmt.Errorf("relevancy evaluation failed: response is not relevant to query")
	}

	score := 0.0
	if passing {
		score = 1.0
	}

	return NewEvaluationResult().
		WithQuery(input.Query).
		WithResponse(input.Response).
		WithContexts(input.Contexts).
		WithPassing(passing).
		WithScore(score).
		WithFeedback(llmResponse), nil
}

// ContextRelevancyEvaluator evaluates the relevancy of contexts to a query.
// Unlike RelevancyEvaluator, this focuses only on context-query relevance.
type ContextRelevancyEvaluator struct {
	*BaseEvaluator
	llm          llm.LLM
	evalTemplate string
}

// DefaultContextRelevancyTemplate is the default template for context relevancy.
const DefaultContextRelevancyTemplate = `Your task is to evaluate if the retrieved context is relevant to the query.
You have two options to answer. Either YES or NO.
Answer YES if the context contains information that could help answer the query, otherwise NO.

Query: {query}

Context:
{context}

Answer: `

// ContextRelevancyEvaluatorOption configures a ContextRelevancyEvaluator.
type ContextRelevancyEvaluatorOption func(*ContextRelevancyEvaluator)

// WithContextRelevancyLLM sets the LLM.
func WithContextRelevancyLLM(l llm.LLM) ContextRelevancyEvaluatorOption {
	return func(e *ContextRelevancyEvaluator) {
		e.llm = l
	}
}

// WithContextRelevancyTemplate sets the template.
func WithContextRelevancyTemplate(template string) ContextRelevancyEvaluatorOption {
	return func(e *ContextRelevancyEvaluator) {
		e.evalTemplate = template
	}
}

// NewContextRelevancyEvaluator creates a new ContextRelevancyEvaluator.
func NewContextRelevancyEvaluator(opts ...ContextRelevancyEvaluatorOption) *ContextRelevancyEvaluator {
	e := &ContextRelevancyEvaluator{
		BaseEvaluator: NewBaseEvaluator(WithEvaluatorName("context_relevancy")),
		evalTemplate:  DefaultContextRelevancyTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Evaluate evaluates whether the contexts are relevant to the query.
func (e *ContextRelevancyEvaluator) Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error) {
	if input.Query == "" {
		return NewEvaluationResult().WithInvalid("query must be provided"), nil
	}
	if len(input.Contexts) == 0 {
		return NewEvaluationResult().WithInvalid("contexts must be provided"), nil
	}
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for context relevancy evaluation")
	}

	// Evaluate each context and count relevant ones
	relevantCount := 0
	var feedbacks []string

	for i, context := range input.Contexts {
		prompt := strings.ReplaceAll(e.evalTemplate, "{query}", input.Query)
		prompt = strings.ReplaceAll(prompt, "{context}", context)

		llmResponse, err := e.llm.Complete(ctx, prompt)
		if err != nil {
			return nil, fmt.Errorf("LLM evaluation failed for context %d: %w", i, err)
		}

		if strings.Contains(strings.ToLower(llmResponse), "yes") {
			relevantCount++
		}
		feedbacks = append(feedbacks, fmt.Sprintf("Context %d: %s", i+1, strings.TrimSpace(llmResponse)))
	}

	// Calculate score as ratio of relevant contexts
	score := float64(relevantCount) / float64(len(input.Contexts))
	passing := score > 0.5 // More than half of contexts are relevant

	return NewEvaluationResult().
		WithQuery(input.Query).
		WithContexts(input.Contexts).
		WithPassing(passing).
		WithScore(score).
		WithFeedback(strings.Join(feedbacks, "\n")), nil
}

// AnswerRelevancyEvaluator evaluates if the response answers the query.
type AnswerRelevancyEvaluator struct {
	*BaseEvaluator
	llm          llm.LLM
	evalTemplate string
}

// DefaultAnswerRelevancyTemplate is the default template for answer relevancy.
const DefaultAnswerRelevancyTemplate = `Your task is to evaluate if the response directly answers the query.
You have two options to answer. Either YES or NO.
Answer YES if the response provides a direct answer to the query, otherwise NO.

Query: {query}
Response: {response}

Answer: `

// AnswerRelevancyEvaluatorOption configures an AnswerRelevancyEvaluator.
type AnswerRelevancyEvaluatorOption func(*AnswerRelevancyEvaluator)

// WithAnswerRelevancyLLM sets the LLM.
func WithAnswerRelevancyLLM(l llm.LLM) AnswerRelevancyEvaluatorOption {
	return func(e *AnswerRelevancyEvaluator) {
		e.llm = l
	}
}

// WithAnswerRelevancyTemplate sets the template.
func WithAnswerRelevancyTemplate(template string) AnswerRelevancyEvaluatorOption {
	return func(e *AnswerRelevancyEvaluator) {
		e.evalTemplate = template
	}
}

// NewAnswerRelevancyEvaluator creates a new AnswerRelevancyEvaluator.
func NewAnswerRelevancyEvaluator(opts ...AnswerRelevancyEvaluatorOption) *AnswerRelevancyEvaluator {
	e := &AnswerRelevancyEvaluator{
		BaseEvaluator: NewBaseEvaluator(WithEvaluatorName("answer_relevancy")),
		evalTemplate:  DefaultAnswerRelevancyTemplate,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Evaluate evaluates whether the response answers the query.
func (e *AnswerRelevancyEvaluator) Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error) {
	if input.Query == "" {
		return NewEvaluationResult().WithInvalid("query must be provided"), nil
	}
	if input.Response == "" {
		return NewEvaluationResult().WithInvalid("response must be provided"), nil
	}
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for answer relevancy evaluation")
	}

	// Format the prompt
	prompt := strings.ReplaceAll(e.evalTemplate, "{query}", input.Query)
	prompt = strings.ReplaceAll(prompt, "{response}", input.Response)

	// Get LLM response
	llmResponse, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM evaluation failed: %w", err)
	}

	// Parse the response
	passing := strings.Contains(strings.ToLower(llmResponse), "yes")

	score := 0.0
	if passing {
		score = 1.0
	}

	return NewEvaluationResult().
		WithQuery(input.Query).
		WithResponse(input.Response).
		WithPassing(passing).
		WithScore(score).
		WithFeedback(llmResponse), nil
}
