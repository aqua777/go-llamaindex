package evaluation

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
)

// DefaultCorrectnessSystemTemplate is the default system template for correctness evaluation.
const DefaultCorrectnessSystemTemplate = `You are an expert evaluation system for a question answering chatbot.

You are given the following information:
- a user query, and
- a generated answer

You may also be given a reference answer to use for reference in your evaluation.

Your job is to judge the relevance and correctness of the generated answer.
Output a single score that represents a holistic evaluation.
You must return your response in a line with only the score.
Do not return answers in any other format.
On a separate line provide your reasoning for the score as well.

Follow these guidelines for scoring:
- Your score has to be between 1 and 5, where 1 is the worst and 5 is the best.
- If the generated answer is not relevant to the user query, you should give a score of 1.
- If the generated answer is relevant but contains mistakes, you should give a score between 2 and 3.
- If the generated answer is relevant and fully correct, you should give a score between 4 and 5.

Example Response:
4.0
The generated answer has the exact same metrics as the reference answer, but it is not as concise.`

// DefaultCorrectnessUserTemplate is the default user template for correctness evaluation.
const DefaultCorrectnessUserTemplate = `## User Query
{query}

## Reference Answer
{reference}

## Generated Answer
{response}`

// CorrectnessEvaluator evaluates the correctness of a question answering system.
// This evaluator depends on a reference answer to be provided, in addition to the
// query string and response string.
type CorrectnessEvaluator struct {
	*BaseEvaluator
	llm            llm.LLM
	systemTemplate string
	userTemplate   string
	scoreThreshold float64
	parserFunc     func(string) (float64, string, error)
}

// CorrectnessEvaluatorOption configures a CorrectnessEvaluator.
type CorrectnessEvaluatorOption func(*CorrectnessEvaluator)

// WithCorrectnessLLM sets the LLM for evaluation.
func WithCorrectnessLLM(l llm.LLM) CorrectnessEvaluatorOption {
	return func(e *CorrectnessEvaluator) {
		e.llm = l
	}
}

// WithCorrectnessSystemTemplate sets the system template.
func WithCorrectnessSystemTemplate(template string) CorrectnessEvaluatorOption {
	return func(e *CorrectnessEvaluator) {
		e.systemTemplate = template
	}
}

// WithCorrectnessUserTemplate sets the user template.
func WithCorrectnessUserTemplate(template string) CorrectnessEvaluatorOption {
	return func(e *CorrectnessEvaluator) {
		e.userTemplate = template
	}
}

// WithCorrectnessScoreThreshold sets the score threshold for passing.
func WithCorrectnessScoreThreshold(threshold float64) CorrectnessEvaluatorOption {
	return func(e *CorrectnessEvaluator) {
		e.scoreThreshold = threshold
	}
}

// WithCorrectnessParser sets a custom parser function.
func WithCorrectnessParser(parser func(string) (float64, string, error)) CorrectnessEvaluatorOption {
	return func(e *CorrectnessEvaluator) {
		e.parserFunc = parser
	}
}

// NewCorrectnessEvaluator creates a new CorrectnessEvaluator.
func NewCorrectnessEvaluator(opts ...CorrectnessEvaluatorOption) *CorrectnessEvaluator {
	e := &CorrectnessEvaluator{
		BaseEvaluator:  NewBaseEvaluator(WithEvaluatorName("correctness")),
		systemTemplate: DefaultCorrectnessSystemTemplate,
		userTemplate:   DefaultCorrectnessUserTemplate,
		scoreThreshold: 4.0,
		parserFunc:     defaultCorrectnessParser,
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// Evaluate evaluates the correctness of the response.
func (e *CorrectnessEvaluator) Evaluate(ctx context.Context, input *EvaluateInput) (*EvaluationResult, error) {
	if input.Query == "" {
		return NewEvaluationResult().WithInvalid("query must be provided"), nil
	}
	if input.Response == "" {
		return NewEvaluationResult().WithInvalid("response must be provided"), nil
	}
	if e.llm == nil {
		return nil, fmt.Errorf("LLM must be provided for correctness evaluation")
	}

	// Format the reference (use placeholder if not provided)
	reference := input.Reference
	if reference == "" {
		reference = "(NO REFERENCE ANSWER SUPPLIED)"
	}

	// Format the user prompt
	userPrompt := strings.ReplaceAll(e.userTemplate, "{query}", input.Query)
	userPrompt = strings.ReplaceAll(userPrompt, "{reference}", reference)
	userPrompt = strings.ReplaceAll(userPrompt, "{response}", input.Response)

	// Create chat messages
	messages := []llm.ChatMessage{
		llm.NewSystemMessage(e.systemTemplate),
		llm.NewUserMessage(userPrompt),
	}

	// Get LLM response
	llmResponse, err := e.llm.Chat(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("LLM evaluation failed: %w", err)
	}

	// Parse the response
	score, reasoning, err := e.parserFunc(llmResponse)
	if err != nil {
		return NewEvaluationResult().
			WithQuery(input.Query).
			WithResponse(input.Response).
			WithReference(input.Reference).
			WithInvalid(fmt.Sprintf("failed to parse LLM response: %v", err)).
			WithFeedback(llmResponse), nil
	}

	passing := score >= e.scoreThreshold

	return NewEvaluationResult().
		WithQuery(input.Query).
		WithResponse(input.Response).
		WithReference(input.Reference).
		WithPassing(passing).
		WithScore(score).
		WithFeedback(reasoning), nil
}

// defaultCorrectnessParser parses the LLM response to extract score and reasoning.
func defaultCorrectnessParser(response string) (float64, string, error) {
	lines := strings.Split(strings.TrimSpace(response), "\n")
	if len(lines) == 0 {
		return 0, "", fmt.Errorf("empty response")
	}

	// Try to find a score in the first few lines
	var score float64
	var scoreFound bool
	var reasoningLines []string

	// Pattern to match a score (e.g., "4.0", "4", "4.5")
	scorePattern := regexp.MustCompile(`^(\d+(?:\.\d+)?)\s*$`)

	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if !scoreFound {
			if matches := scorePattern.FindStringSubmatch(line); matches != nil {
				var err error
				score, err = strconv.ParseFloat(matches[1], 64)
				if err != nil {
					continue
				}
				scoreFound = true
				continue
			}

			// Also try to extract score from text like "Score: 4.0"
			scoreTextPattern := regexp.MustCompile(`(?i)score[:\s]+(\d+(?:\.\d+)?)`)
			if matches := scoreTextPattern.FindStringSubmatch(line); matches != nil {
				var err error
				score, err = strconv.ParseFloat(matches[1], 64)
				if err != nil {
					continue
				}
				scoreFound = true
				// Don't skip this line, it might contain reasoning too
				if len(line) > len(matches[0]) {
					reasoningLines = append(reasoningLines, line)
				}
				continue
			}
		}

		// Collect reasoning lines (skip the score line)
		if scoreFound || i > 0 {
			reasoningLines = append(reasoningLines, line)
		}
	}

	if !scoreFound {
		return 0, "", fmt.Errorf("could not find score in response")
	}

	reasoning := strings.Join(reasoningLines, "\n")
	return score, reasoning, nil
}

// NormalizeScore normalizes a score to 0-1 range from 1-5 range.
func NormalizeScore(score float64) float64 {
	return (score - 1) / 4
}

// DenormalizeScore converts a 0-1 score to 1-5 range.
func DenormalizeScore(score float64) float64 {
	return score*4 + 1
}
