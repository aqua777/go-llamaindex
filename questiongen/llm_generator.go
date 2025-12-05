package questiongen

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/selector"
)

// Default prompt template for sub-question generation.
const DefaultSubQuestionPromptTmpl = `Given a user question, and a list of tools, output a list of relevant sub-questions in json markdown that when composed can help answer the full user question:

# Example 1
<Tools>
` + "```json" + `
{
    "uber_10k": "Provides information about Uber financials for year 2021",
    "lyft_10k": "Provides information about Lyft financials for year 2021"
}
` + "```" + `

<User Question>
Compare and contrast the revenue growth and EBITDA of Uber and Lyft for year 2021

<Output>
` + "```json" + `
{
    "items": [
        {"sub_question": "What is the revenue growth of Uber", "tool_name": "uber_10k"},
        {"sub_question": "What is the EBITDA of Uber", "tool_name": "uber_10k"},
        {"sub_question": "What is the revenue growth of Lyft", "tool_name": "lyft_10k"},
        {"sub_question": "What is the EBITDA of Lyft", "tool_name": "lyft_10k"}
    ]
}
` + "```" + `

# Example 2
<Tools>
` + "```json" + `
%s
` + "```" + `

<User Question>
%s

<Output>
`

// LLMQuestionGenerator uses an LLM to generate sub-questions.
type LLMQuestionGenerator struct {
	*BaseQuestionGenerator
	llm            llm.LLM
	promptTemplate string
	outputParser   *SubQuestionOutputParser
}

// LLMQuestionGeneratorOption configures an LLMQuestionGenerator.
type LLMQuestionGeneratorOption func(*LLMQuestionGenerator)

// WithQuestionGenPrompt sets the prompt template.
func WithQuestionGenPrompt(template string) LLMQuestionGeneratorOption {
	return func(g *LLMQuestionGenerator) {
		g.promptTemplate = template
	}
}

// WithQuestionGenOutputParser sets the output parser.
func WithQuestionGenOutputParser(parser *SubQuestionOutputParser) LLMQuestionGeneratorOption {
	return func(g *LLMQuestionGenerator) {
		g.outputParser = parser
	}
}

// NewLLMQuestionGenerator creates a new LLMQuestionGenerator.
func NewLLMQuestionGenerator(llmInstance llm.LLM, opts ...LLMQuestionGeneratorOption) *LLMQuestionGenerator {
	g := &LLMQuestionGenerator{
		BaseQuestionGenerator: NewBaseQuestionGenerator(WithGeneratorName("LLMQuestionGenerator")),
		llm:                   llmInstance,
		promptTemplate:        DefaultSubQuestionPromptTmpl,
		outputParser:          NewSubQuestionOutputParser(),
	}

	for _, opt := range opts {
		opt(g)
	}

	return g
}

// Generate generates sub-questions from the query and tools.
func (g *LLMQuestionGenerator) Generate(ctx context.Context, tools []selector.ToolMetadata, query string) ([]SubQuestion, error) {
	toolsStr := buildToolsText(tools)

	prompt := fmt.Sprintf(g.promptTemplate, toolsStr, query)

	response, err := g.llm.Complete(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("LLM completion failed: %w", err)
	}

	subQuestions, err := g.outputParser.Parse(response)
	if err != nil {
		return nil, fmt.Errorf("failed to parse sub-questions: %w", err)
	}

	return subQuestions, nil
}

// Ensure LLMQuestionGenerator implements QuestionGenerator.
var _ QuestionGenerator = (*LLMQuestionGenerator)(nil)

// buildToolsText converts tools to JSON text.
func buildToolsText(tools []selector.ToolMetadata) string {
	toolsMap := make(map[string]string)
	for _, tool := range tools {
		toolsMap[tool.Name] = tool.Description
	}

	data, err := json.MarshalIndent(toolsMap, "", "    ")
	if err != nil {
		return "{}"
	}
	return string(data)
}

// SubQuestionOutputParser parses LLM output into sub-questions.
type SubQuestionOutputParser struct{}

// NewSubQuestionOutputParser creates a new SubQuestionOutputParser.
func NewSubQuestionOutputParser() *SubQuestionOutputParser {
	return &SubQuestionOutputParser{}
}

// Parse parses the LLM output into sub-questions.
func (p *SubQuestionOutputParser) Parse(output string) ([]SubQuestion, error) {
	// Extract JSON from the output
	jsonStr := extractJSON(output)
	if jsonStr == "" {
		return nil, fmt.Errorf("no JSON found in output: %s", output)
	}

	// Try parsing as SubQuestionList
	var list SubQuestionList
	if err := json.Unmarshal([]byte(jsonStr), &list); err == nil && len(list.Items) > 0 {
		return list.Items, nil
	}

	// Try parsing as array of SubQuestion
	var questions []SubQuestion
	if err := json.Unmarshal([]byte(jsonStr), &questions); err == nil {
		return questions, nil
	}

	return nil, fmt.Errorf("failed to parse sub-questions from: %s", jsonStr)
}

// extractJSON extracts JSON from text (looks for code blocks or raw JSON).
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
