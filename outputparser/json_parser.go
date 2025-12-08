package outputparser

import (
	"encoding/json"
	"fmt"
	"strings"
)

// JSONOutputParser parses JSON output from LLMs.
type JSONOutputParser struct {
	*BaseOutputParser
	formatInstructions string
}

// JSONOutputParserOption configures a JSONOutputParser.
type JSONOutputParserOption func(*JSONOutputParser)

// WithJSONFormatInstructions sets custom format instructions.
func WithJSONFormatInstructions(instructions string) JSONOutputParserOption {
	return func(p *JSONOutputParser) {
		p.formatInstructions = instructions
	}
}

// DefaultJSONFormatInstructions is the default format instruction.
const DefaultJSONFormatInstructions = `
The output should be formatted as a JSON instance.
`

// NewJSONOutputParser creates a new JSONOutputParser.
func NewJSONOutputParser(opts ...JSONOutputParserOption) *JSONOutputParser {
	p := &JSONOutputParser{
		BaseOutputParser:   NewBaseOutputParser(WithParserName("JSONOutputParser")),
		formatInstructions: DefaultJSONFormatInstructions,
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Parse parses JSON from the output.
func (p *JSONOutputParser) Parse(output string) (*StructuredOutput, error) {
	jsonStr := extractJSON(output)
	if jsonStr == "" {
		return nil, NewOutputParserError("no JSON found in output", output)
	}

	var parsed interface{}
	if err := json.Unmarshal([]byte(jsonStr), &parsed); err != nil {
		return nil, NewOutputParserError(fmt.Sprintf("failed to parse JSON: %v", err), output)
	}

	return &StructuredOutput{
		RawOutput:    output,
		ParsedOutput: parsed,
	}, nil
}

// Format adds JSON format instructions to the prompt.
func (p *JSONOutputParser) Format(promptTemplate string) string {
	return promptTemplate + "\n\n" + p.formatInstructions
}

// Ensure JSONOutputParser implements OutputParser.
var _ OutputParser = (*JSONOutputParser)(nil)

// extractJSON extracts JSON from text.
func extractJSON(text string) string {
	// Look for JSON in code blocks
	codeBlockStart := strings.Index(text, "```json")
	if codeBlockStart != -1 {
		start := codeBlockStart + 7
		end := strings.Index(text[start:], "```")
		if end != -1 {
			return strings.TrimSpace(text[start : start+end])
		}
	}

	// Look for code blocks without language
	codeBlockStart = strings.Index(text, "```")
	if codeBlockStart != -1 {
		start := codeBlockStart + 3
		end := strings.Index(text[start:], "```")
		if end != -1 {
			return strings.TrimSpace(text[start : start+end])
		}
	}

	// Find JSON object
	start := strings.Index(text, "{")
	if start != -1 {
		end := strings.LastIndex(text, "}")
		if end > start {
			return text[start : end+1]
		}
	}

	// Find JSON array
	start = strings.Index(text, "[")
	if start != -1 {
		end := strings.LastIndex(text, "]")
		if end > start {
			return text[start : end+1]
		}
	}

	return ""
}

// ListOutputParser parses list output from LLMs.
type ListOutputParser struct {
	*BaseOutputParser
	separator string
}

// ListOutputParserOption configures a ListOutputParser.
type ListOutputParserOption func(*ListOutputParser)

// WithListSeparator sets the list separator.
func WithListSeparator(sep string) ListOutputParserOption {
	return func(p *ListOutputParser) {
		p.separator = sep
	}
}

// NewListOutputParser creates a new ListOutputParser.
func NewListOutputParser(opts ...ListOutputParserOption) *ListOutputParser {
	p := &ListOutputParser{
		BaseOutputParser: NewBaseOutputParser(WithParserName("ListOutputParser")),
		separator:        "\n",
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Parse parses a list from the output.
func (p *ListOutputParser) Parse(output string) (*StructuredOutput, error) {
	items := strings.Split(output, p.separator)

	// Clean up items
	cleanItems := make([]string, 0, len(items))
	for _, item := range items {
		item = strings.TrimSpace(item)
		if item != "" {
			cleanItems = append(cleanItems, item)
		}
	}

	return &StructuredOutput{
		RawOutput:    output,
		ParsedOutput: cleanItems,
	}, nil
}

// Format returns the prompt template unchanged.
func (p *ListOutputParser) Format(promptTemplate string) string {
	return promptTemplate
}

// Ensure ListOutputParser implements OutputParser.
var _ OutputParser = (*ListOutputParser)(nil)

// BooleanOutputParser parses boolean output from LLMs.
type BooleanOutputParser struct {
	*BaseOutputParser
	trueValues  []string
	falseValues []string
}

// BooleanOutputParserOption configures a BooleanOutputParser.
type BooleanOutputParserOption func(*BooleanOutputParser)

// WithTrueValues sets the values that represent true.
func WithTrueValues(values []string) BooleanOutputParserOption {
	return func(p *BooleanOutputParser) {
		p.trueValues = values
	}
}

// WithFalseValues sets the values that represent false.
func WithFalseValues(values []string) BooleanOutputParserOption {
	return func(p *BooleanOutputParser) {
		p.falseValues = values
	}
}

// NewBooleanOutputParser creates a new BooleanOutputParser.
func NewBooleanOutputParser(opts ...BooleanOutputParserOption) *BooleanOutputParser {
	p := &BooleanOutputParser{
		BaseOutputParser: NewBaseOutputParser(WithParserName("BooleanOutputParser")),
		trueValues:       []string{"yes", "true", "1", "y"},
		falseValues:      []string{"no", "false", "0", "n"},
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// Parse parses a boolean from the output.
func (p *BooleanOutputParser) Parse(output string) (*StructuredOutput, error) {
	cleaned := strings.ToLower(strings.TrimSpace(output))

	for _, v := range p.trueValues {
		if cleaned == v || strings.Contains(cleaned, v) {
			return &StructuredOutput{
				RawOutput:    output,
				ParsedOutput: true,
			}, nil
		}
	}

	for _, v := range p.falseValues {
		if cleaned == v || strings.Contains(cleaned, v) {
			return &StructuredOutput{
				RawOutput:    output,
				ParsedOutput: false,
			}, nil
		}
	}

	return nil, NewOutputParserError("could not parse boolean value", output)
}

// Format returns the prompt template unchanged.
func (p *BooleanOutputParser) Format(promptTemplate string) string {
	return promptTemplate
}

// Ensure BooleanOutputParser implements OutputParser.
var _ OutputParser = (*BooleanOutputParser)(nil)
