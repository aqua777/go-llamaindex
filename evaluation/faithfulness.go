package evaluation

import (
	"context"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
)

// DefaultFaithfulnessTemplate is the default prompt template for faithfulness evaluation.
const DefaultFaithfulnessTemplate = `Please tell if a given piece of information is supported by the context.
You need to answer with either YES or NO.
Answer YES if any of the context supports the information, even if most of the context is unrelated.
Some examples are provided below.

Information: Apple pie is generally double-crusted.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
Answer: YES

Information: Apple pies tastes bad.
Context: An apple pie is a fruit pie in which the principal filling ingredient is apples.
Apple pie is often served with whipped cream, ice cream ('apple pie à la mode'), custard or cheddar cheese.
It is generally double-crusted, with pastry both above and below the filling; the upper crust may be solid or latticed (woven of crosswise strips).
Answer: NO

Information: {response}
Context: {context}
Answer: `

// FaithfulnessEvaluator evaluates whether a response is faithful to the contexts.
// It checks if the response is supported by the provided contexts or if it's hallucinated.
type FaithfulnessEvaluator struct {
	*BaseEvaluator
	llm          llm.LLM
	evalTemplate string
	raiseError   bool
}

// FaithfulnessEvaluatorOption configures a FaithfulnessEvaluator.
type FaithfulnessEvaluatorOption func(*FaithfulnessEvaluator)

// WithFaithfulnessLLM sets the LLM for evaluation.
func WithFaithfulnessLLM(l llm.LLM) FaithfulnessEvaluatorOption {
	return func(e *FaithfulnessEvaluator) {
		e.llm = l
	}
}

// WithFaithfulnessTemplate sets the evaluation template.
func WithFaithfulnessTemplate(template string) FaithfulnessEvaluatorOption {
	return func(e *FaithfulnessEvaluator) {
		e.evalTemplate = template
	}
}

// WithFaithfulnessRaiseError sets whether to raise an error on failure.
func WithFaithfulnessRaiseError(raise bool) FaithfulnessEvaluatorOption {
	return func(e *FaithfulnessEvaluator) {
		e.raiseError = raise
	}
}

// NewFaithfulnessEvaluator creates a new FaithfulnessEvaluator.
func NewFaithfulnessEvaluator(opts ...FaithfulnessEvaluatorOption) *FaithfulnessEvaluator {
	e := &FaithfulnessEvaluator{
		BaseEvaluator: NewBaseEvaluator(WithEvaluatorName("faithfulness")),
		evalTemplate:  DefaultFaithfulnessTemplate,
		raiseError:    false,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Evaluate evaluates whether the response is faithful to the contexts.
func (e *FaithfulnessEvaluator) Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error) {
	if len(input.Contexts) == 0 {
		return NewEvaluationResult().WithInvalid("contexts must be provided"), nil
	}
	if input.Response == "" {
		return NewEvaluationResult().WithInvalid("response must be provided"), nil
	}
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for faithfulness evaluation")
	}

	// Combine contexts
	contextStr := strings.Join(input.Contexts, "\n\n")

	// Format the prompt
	prompt := strings.ReplaceAll(e.evalTemplate, "{response}", input.Response)
	prompt = strings.ReplaceAll(prompt, "{context}", contextStr)

	// Get LLM response
	llmResponse, err := e.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM evaluation failed: %w", err)
	}

	// Parse the response
	passing := strings.Contains(strings.ToLower(llmResponse), "yes")

	if !passing && e.raiseError {
		return nil, fmt.Errorf("faithfulness evaluation failed: response is not supported by context")
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

// EvaluateStatements evaluates individual statements for faithfulness.
// This is useful for more granular evaluation.
func (e *FaithfulnessEvaluator) EvaluateStatements(ctx context.Context, statements []string, contexts []string) ([]*EvaluationResult, error) {
	results := make([]*EvaluationResult, len(statements))

	for i, statement := range statements {
		input := NewEvaluateInput().
			WithResponse(statement).
			WithContexts(contexts)

		result, err := e.Evaluate(ctx, input)
		if err != nil {
			return nil, fmt.Errorf("failed to evaluate statement %d: %w", i, err)
		}
		results[i] = result
	}

	return results, nil
}

// AggregateScore calculates the aggregate faithfulness score from multiple results.
func AggregateScore(results []*EvaluationResult) float64 {
	if len(results) == 0 {
		return 0
	}

	var total float64
	for _, r := range results {
		total += r.GetScore()
	}

	return total / float64(len(results))
}
