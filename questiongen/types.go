// Package questiongen provides question generation functionality.
package questiongen

import (
	"context"

	"github.com/aqua777/go-llamaindex/selector"
)

// SubQuestion represents a generated sub-question.
type SubQuestion struct {
	SubQuestion string `json:"sub_question"`
	ToolName    string `json:"tool_name"`
}

// SubQuestionList wraps a list of sub-questions.
type SubQuestionList struct {
	Items []SubQuestion `json:"items"`
}

// QuestionGenerator is the interface for question generators.
type QuestionGenerator interface {
	// Generate generates sub-questions from a query and available tools.
	Generate(ctx context.Context, tools []selector.ToolMetadata, query string) ([]SubQuestion, error)
	// Name returns the name of the generator.
	Name() string
}

// BaseQuestionGenerator provides a base implementation.
type BaseQuestionGenerator struct {
	name string
}

// BaseQuestionGeneratorOption configures a BaseQuestionGenerator.
type BaseQuestionGeneratorOption func(*BaseQuestionGenerator)

// WithGeneratorName sets the generator name.
func WithGeneratorName(name string) BaseQuestionGeneratorOption {
	return func(g *BaseQuestionGenerator) {
		g.name = name
	}
}

// NewBaseQuestionGenerator creates a new BaseQuestionGenerator.
func NewBaseQuestionGenerator(opts ...BaseQuestionGeneratorOption) *BaseQuestionGenerator {
	g := &BaseQuestionGenerator{
		name: "BaseQuestionGenerator",
	}

	for _, opt := range opts {
		opt(g)
	}

	return g
}

// Name returns the name of the generator.
func (g *BaseQuestionGenerator) Name() string {
	return g.name
}

// Generate is a no-op implementation.
func (g *BaseQuestionGenerator) Generate(ctx context.Context, tools []selector.ToolMetadata, query string) ([]SubQuestion, error) {
	return []SubQuestion{}, nil
}

// Ensure BaseQuestionGenerator implements QuestionGenerator.
var _ QuestionGenerator = (*BaseQuestionGenerator)(nil)
