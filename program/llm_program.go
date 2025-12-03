package program

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
)

// LLMProgram uses LLM text generation with output parsing for structured output.
// Unlike FunctionProgram, it doesn't require function calling support.
type LLMProgram struct {
	*BaseProgram
}

// LLMProgramOption configures an LLMProgram.
type LLMProgramOption func(*LLMProgram)

// NewLLMProgram creates a new LLMProgram.
func NewLLMProgram(l llm.LLM, opts ...LLMProgramOption) *LLMProgram {
	p := &LLMProgram{
		BaseProgram: NewBaseProgram(WithProgramLLM(l)),
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// NewLLMProgramWithParser creates an LLMProgram with a specific output parser.
func NewLLMProgramWithParser(l llm.LLM, parser OutputParser) *LLMProgram {
	return &LLMProgram{
		BaseProgram: NewBaseProgram(
			WithProgramLLM(l),
			WithProgramOutputParser(parser),
		),
	}
}

// NewLLMProgramForType creates an LLMProgram for a specific Go type.
func NewLLMProgramForType(l llm.LLM, targetType interface{}) *LLMProgram {
	parser := NewPydanticOutputParser(targetType)
	return NewLLMProgramWithParser(l, parser)
}

// Call executes the LLM program.
func (p *LLMProgram) Call(ctx context.Context, args map[string]interface{}) (*ProgramOutput, error) {
	if p.LLM == nil {
		return nil, fmt.Errorf("LLM is required")
	}

	// Build the prompt
	var promptText string
	if p.Prompt != nil {
		// Convert args to string map
		stringArgs := make(map[string]string)
		for k, v := range args {
			stringArgs[k] = fmt.Sprintf("%v", v)
		}
		promptText = p.Prompt.Format(stringArgs)
	} else {
		// Default prompt from args
		if input, ok := args["input"].(string); ok {
			promptText = input
		} else if query, ok := args["query"].(string); ok {
			promptText = query
		} else {
			// Convert args to string
			argsJSON, _ := json.Marshal(args)
			promptText = string(argsJSON)
		}
	}

	// Add format instructions if we have a parser
	if p.OutputParser != nil {
		instructions := p.OutputParser.GetFormatInstructions()
		if instructions != "" {
			promptText = promptText + "\n\n" + instructions
		}
	}

	// Try to use structured output if available
	if structuredLLM, ok := p.LLM.(llm.LLMWithStructuredOutput); ok && structuredLLM.SupportsStructuredOutput() {
		return p.callWithStructuredOutput(ctx, structuredLLM, promptText)
	}

	// Fall back to regular completion
	return p.callWithCompletion(ctx, promptText)
}

// callWithStructuredOutput uses the LLM's structured output capability.
func (p *LLMProgram) callWithStructuredOutput(ctx context.Context, l llm.LLMWithStructuredOutput, promptText string) (*ProgramOutput, error) {
	messages := []llm.ChatMessage{
		llm.NewUserMessage(promptText),
	}

	format := llm.NewJSONResponseFormat()
	rawOutput, err := l.ChatWithFormat(ctx, messages, format)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	var parsedOutput interface{}
	if p.OutputParser != nil {
		parsedOutput, err = p.OutputParser.Parse(rawOutput)
		if err != nil && p.Verbose {
			fmt.Printf("Warning: failed to parse output: %v\n", err)
		}
	}

	return NewProgramOutput(rawOutput, parsedOutput), nil
}

// callWithCompletion uses regular LLM completion.
func (p *LLMProgram) callWithCompletion(ctx context.Context, promptText string) (*ProgramOutput, error) {
	rawOutput, err := p.LLM.Complete(ctx, promptText)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	var parsedOutput interface{}
	if p.OutputParser != nil {
		parsedOutput, err = p.OutputParser.Parse(rawOutput)
		if err != nil && p.Verbose {
			fmt.Printf("Warning: failed to parse output: %v\n", err)
		}
	}

	return NewProgramOutput(rawOutput, parsedOutput), nil
}

// WithPrompt sets the prompt template (fluent API).
func (p *LLMProgram) WithPrompt(prompt *prompts.PromptTemplate) *LLMProgram {
	p.Prompt = prompt
	return p
}

// WithOutputParser sets the output parser (fluent API).
func (p *LLMProgram) WithOutputParser(parser OutputParser) *LLMProgram {
	p.OutputParser = parser
	return p
}

// WithVerbose enables verbose mode (fluent API).
func (p *LLMProgram) WithVerbose(verbose bool) *LLMProgram {
	p.Verbose = verbose
	return p
}

// Ensure LLMProgram implements Program.
var _ Program = (*LLMProgram)(nil)
var _ ProgramWithPrompt = (*LLMProgram)(nil)

// MultiOutputProgram generates multiple structured outputs.
type MultiOutputProgram struct {
	*BaseProgram
	// NumOutputs is the number of outputs to generate.
	NumOutputs int
}

// NewMultiOutputProgram creates a new MultiOutputProgram.
func NewMultiOutputProgram(l llm.LLM, numOutputs int, parser OutputParser) *MultiOutputProgram {
	return &MultiOutputProgram{
		BaseProgram: NewBaseProgram(
			WithProgramLLM(l),
			WithProgramOutputParser(parser),
		),
		NumOutputs: numOutputs,
	}
}

// Call executes the multi-output program.
func (p *MultiOutputProgram) Call(ctx context.Context, args map[string]interface{}) (*ProgramOutput, error) {
	if p.LLM == nil {
		return nil, fmt.Errorf("LLM is required")
	}

	// Build the prompt
	var promptText string
	if p.Prompt != nil {
		stringArgs := make(map[string]string)
		for k, v := range args {
			stringArgs[k] = fmt.Sprintf("%v", v)
		}
		promptText = p.Prompt.Format(stringArgs)
	} else {
		if input, ok := args["input"].(string); ok {
			promptText = input
		} else {
			argsJSON, _ := json.Marshal(args)
			promptText = string(argsJSON)
		}
	}

	// Add instructions for multiple outputs
	promptText = fmt.Sprintf("%s\n\nPlease generate %d different outputs. Return them as a JSON array.", promptText, p.NumOutputs)

	if p.OutputParser != nil {
		instructions := p.OutputParser.GetFormatInstructions()
		if instructions != "" {
			promptText = promptText + "\n\n" + instructions
		}
	}

	rawOutput, err := p.LLM.Complete(ctx, promptText)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	var parsedOutput interface{}
	if p.OutputParser != nil {
		parsedOutput, err = p.OutputParser.Parse(rawOutput)
		if err != nil && p.Verbose {
			fmt.Printf("Warning: failed to parse output: %v\n", err)
		}
	}

	output := NewProgramOutput(rawOutput, parsedOutput)
	output.Metadata["num_outputs"] = p.NumOutputs

	return output, nil
}

// Ensure MultiOutputProgram implements Program.
var _ Program = (*MultiOutputProgram)(nil)
