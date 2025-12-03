package selector

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
)

// Default prompt templates for selection.
const (
	DefaultSingleSelectPromptTmpl = `Some choices are given below. It is provided in a numbered list (1 to %d), where each item in the list corresponds to a summary.
---------------------
%s
---------------------
Using only the choices above and not prior knowledge, return the choice that is most relevant to the question: '%s'

The output should be ONLY JSON formatted as a JSON instance.

Here is an example:
[
    {
        "choice": 1,
        "reason": "<insert reason for choice>"
    }
]`

	DefaultMultiSelectPromptTmpl = `Some choices are given below. It is provided in a numbered list (1 to %d), where each item in the list corresponds to a summary.
---------------------
%s
---------------------
Using only the choices above and not prior knowledge, return the top choices (no more than %d, but only select what is needed) that are most relevant to the question: '%s'

The output should be ONLY JSON formatted as a JSON instance.

Here is an example:
[
    {
        "choice": 1,
        "reason": "<insert reason for choice>"
    },
    ...
]`
)

// LLMSingleSelector uses an LLM to select one choice from many.
type LLMSingleSelector struct {
	*BaseSelector
	llm            llm.LLM
	promptTemplate string
	outputParser   *SelectionOutputParser
}

// LLMSingleSelectorOption configures an LLMSingleSelector.
type LLMSingleSelectorOption func(*LLMSingleSelector)

// WithSingleSelectPrompt sets the prompt template.
func WithSingleSelectPrompt(template string) LLMSingleSelectorOption {
	return func(s *LLMSingleSelector) {
		s.promptTemplate = template
	}
}

// WithSingleSelectOutputParser sets the output parser.
func WithSingleSelectOutputParser(parser *SelectionOutputParser) LLMSingleSelectorOption {
	return func(s *LLMSingleSelector) {
		s.outputParser = parser
	}
}

// NewLLMSingleSelector creates a new LLMSingleSelector.
func NewLLMSingleSelector(llmInstance llm.LLM, opts ...LLMSingleSelectorOption) *LLMSingleSelector {
	s := &LLMSingleSelector{
		BaseSelector:   NewBaseSelector(WithSelectorName("LLMSingleSelector")),
		llm:            llmInstance,
		promptTemplate: DefaultSingleSelectPromptTmpl,
		outputParser:   NewSelectionOutputParser(),
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Select chooses one option from the choices.
func (s *LLMSingleSelector) Select(ctx context.Context, choices []ToolMetadata, query string) (*SelectorResult, error) {
	choicesText := BuildChoicesText(choices)

	prompt := fmt.Sprintf(s.promptTemplate, len(choices), choicesText, query)

	response, err := s.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %w", err)
	}

	answers, err := s.outputParser.Parse(response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse selection output: %w", err)
	}

	// Convert answers to selections (adjust for zero indexing)
	selections := make([]SingleSelection, len(answers))
	for i, answer := range answers {
		selections[i] = SingleSelection{
			Index:  answer.Choice - 1, // Convert from 1-indexed to 0-indexed
			Reason: answer.Reason,
		}
	}

	return &SelectorResult{Selections: selections}, nil
}

// Ensure LLMSingleSelector implements Selector.
var _ Selector = (*LLMSingleSelector)(nil)

// LLMMultiSelector uses an LLM to select multiple choices from many.
type LLMMultiSelector struct {
	*BaseSelector
	llm            llm.LLM
	promptTemplate string
	outputParser   *SelectionOutputParser
	maxOutputs     int
}

// LLMMultiSelectorOption configures an LLMMultiSelector.
type LLMMultiSelectorOption func(*LLMMultiSelector)

// WithMultiSelectPrompt sets the prompt template.
func WithMultiSelectPrompt(template string) LLMMultiSelectorOption {
	return func(s *LLMMultiSelector) {
		s.promptTemplate = template
	}
}

// WithMultiSelectOutputParser sets the output parser.
func WithMultiSelectOutputParser(parser *SelectionOutputParser) LLMMultiSelectorOption {
	return func(s *LLMMultiSelector) {
		s.outputParser = parser
	}
}

// WithMaxOutputs sets the maximum number of selections.
func WithMaxOutputs(max int) LLMMultiSelectorOption {
	return func(s *LLMMultiSelector) {
		s.maxOutputs = max
	}
}

// NewLLMMultiSelector creates a new LLMMultiSelector.
func NewLLMMultiSelector(llmInstance llm.LLM, opts ...LLMMultiSelectorOption) *LLMMultiSelector {
	s := &LLMMultiSelector{
		BaseSelector:   NewBaseSelector(WithSelectorName("LLMMultiSelector")),
		llm:            llmInstance,
		promptTemplate: DefaultMultiSelectPromptTmpl,
		outputParser:   NewSelectionOutputParser(),
		maxOutputs:     0, // 0 means use number of choices
	}

	for _, opt := range opts {
		opt(s)
	}

	return s
}

// Select chooses multiple options from the choices.
func (s *LLMMultiSelector) Select(ctx context.Context, choices []ToolMetadata, query string) (*SelectorResult, error) {
	choicesText := BuildChoicesText(choices)

	maxOutputs := s.maxOutputs
	if maxOutputs == 0 {
		maxOutputs = len(choices)
	}

	prompt := fmt.Sprintf(s.promptTemplate, len(choices), choicesText, maxOutputs, query)

	response, err := s.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %w", err)
	}

	answers, err := s.outputParser.Parse(response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse selection output: %w", err)
	}

	// Convert answers to selections (adjust for zero indexing)
	selections := make([]SingleSelection, len(answers))
	for i, answer := range answers {
		selections[i] = SingleSelection{
			Index:  answer.Choice - 1, // Convert from 1-indexed to 0-indexed
			Reason: answer.Reason,
		}
	}

	return &SelectorResult{Selections: selections}, nil
}

// MaxOutputs returns the maximum number of outputs.
func (s *LLMMultiSelector) MaxOutputs() int {
	return s.maxOutputs
}

// Ensure LLMMultiSelector implements Selector.
var _ Selector = (*LLMMultiSelector)(nil)

// SelectionOutputParser parses LLM selection output.
type SelectionOutputParser struct{}

// Answer represents a parsed selection answer.
type Answer struct {
	Choice int    `json:"choice"`
	Reason string `json:"reason"`
}

// NewSelectionOutputParser creates a new SelectionOutputParser.
func NewSelectionOutputParser() *SelectionOutputParser {
	return &SelectionOutputParser{}
}

// Parse parses the LLM output into answers.
func (p *SelectionOutputParser) Parse(output string) ([]Answer, error) {
	// Extract JSON from the output
	jsonStr := extractJSON(output)
	if jsonStr == "" {
		return nil, fmt.Errorf("no JSON found in output: %s", output)
	}

	// Parse JSON
	var answers []Answer
	if err := parseJSON(jsonStr, &answers); err != nil {
		// Try parsing as single object
		var answer Answer
		if err2 := parseJSON(jsonStr, &answer); err2 != nil {
			return nil, fmt.Errorf("failed to parse JSON: %w", err)
		}
		answers = []Answer{answer}
	}

	return answers, nil
}

// extractJSON extracts JSON array or object from text.
func extractJSON(text string) string {
	// Find JSON array
	start := strings.Index(text, "[")
	if start != -1 {
		end := strings.LastIndex(text, "]")
		if end > start {
			return text[start : end+1]
		}
	}

	// Find JSON object
	start = strings.Index(text, "{")
	if start != -1 {
		end := strings.LastIndex(text, "}")
		if end > start {
			return text[start : end+1]
		}
	}

	return ""
}

// parseJSON parses JSON into the target.
func parseJSON(jsonStr string, target interface{}) error {
	return json.Unmarshal([]byte(jsonStr), target)
}
