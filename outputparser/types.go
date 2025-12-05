// Package outputparser provides output parsing functionality.
package outputparser

import (
	"fmt"
)

// StructuredOutput represents parsed output from an LLM.
type StructuredOutput struct {
	RawOutput    string      `json:"raw_output"`
	ParsedOutput interface{} `json:"parsed_output,omitempty"`
}

// OutputParserError represents an error during output parsing.
type OutputParserError struct {
	Message string
	Output  string
}

func (e *OutputParserError) Error() string {
	return fmt.Sprintf("output parser error: %s (output: %s)", e.Message, e.Output)
}

// NewOutputParserError creates a new OutputParserError.
func NewOutputParserError(message, output string) *OutputParserError {
	return &OutputParserError{
		Message: message,
		Output:  output,
	}
}

// OutputParser is the interface for output parsers.
type OutputParser interface {
	// Parse parses the output string into structured output.
	Parse(output string) (*StructuredOutput, error)
	// Format formats a prompt template with output instructions.
	Format(promptTemplate string) string
	// Name returns the name of the parser.
	Name() string
}

// BaseOutputParser provides a base implementation.
type BaseOutputParser struct {
	name string
}

// BaseOutputParserOption configures a BaseOutputParser.
type BaseOutputParserOption func(*BaseOutputParser)

// WithParserName sets the parser name.
func WithParserName(name string) BaseOutputParserOption {
	return func(p *BaseOutputParser) {
		p.name = name
	}
}

// NewBaseOutputParser creates a new BaseOutputParser.
func NewBaseOutputParser(opts ...BaseOutputParserOption) *BaseOutputParser {
	p := &BaseOutputParser{
		name: "BaseOutputParser",
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Name returns the name of the parser.
func (p *BaseOutputParser) Name() string {
	return p.name
}

// Parse is a no-op implementation that returns the raw output.
func (p *BaseOutputParser) Parse(output string) (*StructuredOutput, error) {
	return &StructuredOutput{
		RawOutput:    output,
		ParsedOutput: output,
	}, nil
}

// Format returns the prompt template unchanged.
func (p *BaseOutputParser) Format(promptTemplate string) string {
	return promptTemplate
}

// Ensure BaseOutputParser implements OutputParser.
var _ OutputParser = (*BaseOutputParser)(nil)
