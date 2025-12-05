package program

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/prompts"
)

// FunctionProgram uses LLM function calling for structured output.
// It leverages the LLM's native function calling capability to ensure
// structured output that matches a specified schema.
type FunctionProgram struct {
	*BaseProgram
	// FunctionName is the name of the function to call.
	FunctionName string
	// FunctionDescription describes what the function does.
	FunctionDescription string
	// Parameters is the JSON schema for the function parameters.
	Parameters map[string]interface{}
}

// FunctionProgramOption configures a FunctionProgram.
type FunctionProgramOption func(*FunctionProgram)

// WithFunctionName sets the function name.
func WithFunctionName(name string) FunctionProgramOption {
	return func(p *FunctionProgram) {
		p.FunctionName = name
	}
}

// WithFunctionDescription sets the function description.
func WithFunctionDescription(desc string) FunctionProgramOption {
	return func(p *FunctionProgram) {
		p.FunctionDescription = desc
	}
}

// WithFunctionParameters sets the function parameters schema.
func WithFunctionParameters(params map[string]interface{}) FunctionProgramOption {
	return func(p *FunctionProgram) {
		p.Parameters = params
	}
}

// NewFunctionProgram creates a new FunctionProgram.
func NewFunctionProgram(l llm.LLM, opts ...FunctionProgramOption) *FunctionProgram {
	p := &FunctionProgram{
		BaseProgram:         NewBaseProgram(WithProgramLLM(l)),
		FunctionName:        "output",
		FunctionDescription: "Generate structured output",
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
		},
	}

	for _, opt := range opts {
		opt(p)
	}

	return p
}

// NewFunctionProgramFromSchema creates a FunctionProgram from a JSON schema.
func NewFunctionProgramFromSchema(l llm.LLM, name, description string, schema map[string]interface{}) *FunctionProgram {
	return NewFunctionProgram(l,
		WithFunctionName(name),
		WithFunctionDescription(description),
		WithFunctionParameters(schema),
	)
}

// NewFunctionProgramFromType creates a FunctionProgram from a Go type.
func NewFunctionProgramFromType(l llm.LLM, name, description string, targetType interface{}) *FunctionProgram {
	parser := NewPydanticOutputParser(targetType)
	schema := structToSchema(parser.TargetType)

	return &FunctionProgram{
		BaseProgram: NewBaseProgram(
			WithProgramLLM(l),
			WithProgramOutputParser(parser),
		),
		FunctionName:        name,
		FunctionDescription: description,
		Parameters:          schema,
	}
}

// Call executes the function program.
func (p *FunctionProgram) Call(ctx context.Context, args map[string]interface{}) (*ProgramOutput, error) {
	if p.LLM == nil {
		return nil, fmt.Errorf("LLM is required")
	}

	// Check if LLM supports tool calling
	toolLLM, ok := p.LLM.(llm.LLMWithToolCalling)
	if !ok {
		return nil, fmt.Errorf("LLM does not support function calling")
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

	// Create the tool metadata
	tool := llm.NewToolMetadata(p.FunctionName, p.FunctionDescription).
		WithParameters(p.Parameters)

	// Create messages
	messages := []llm.ChatMessage{
		llm.NewUserMessage(promptText),
	}

	// Call LLM with tools
	options := &llm.ChatCompletionOptions{
		ToolChoice: llm.ToolChoiceRequired,
	}

	response, err := toolLLM.ChatWithTools(ctx, messages, []*llm.ToolMetadata{tool}, options)
	if err != nil {
		return nil, fmt.Errorf("LLM call failed: %w", err)
	}

	// Extract the function call result
	rawOutput := response.Text
	var parsedOutput interface{}

	// Look for tool calls in the response message
	if response.Message != nil {
		for _, block := range response.Message.Blocks {
			if block.Type == llm.ContentBlockTypeToolCall && block.ToolCall != nil {
				if block.ToolCall.Name == p.FunctionName {
					rawOutput = block.ToolCall.Arguments
					// Parse the arguments as JSON
					if err := json.Unmarshal([]byte(block.ToolCall.Arguments), &parsedOutput); err != nil {
						return nil, fmt.Errorf("failed to parse function arguments: %w", err)
					}
					break
				}
			}
		}
	}

	// If no tool call found, try to parse the content
	if parsedOutput == nil && p.OutputParser != nil {
		var parseErr error
		parsedOutput, parseErr = p.OutputParser.Parse(rawOutput)
		if parseErr != nil && p.Verbose {
			fmt.Printf("Warning: failed to parse output: %v\n", parseErr)
		}
	}

	output := NewProgramOutput(rawOutput, parsedOutput)
	output.Metadata["function_name"] = p.FunctionName

	return output, nil
}

// WithPrompt sets the prompt template (fluent API).
func (p *FunctionProgram) WithPrompt(prompt *prompts.PromptTemplate) *FunctionProgram {
	p.Prompt = prompt
	return p
}

// WithVerbose enables verbose mode (fluent API).
func (p *FunctionProgram) WithVerbose(verbose bool) *FunctionProgram {
	p.Verbose = verbose
	return p
}

// Ensure FunctionProgram implements Program.
var _ Program = (*FunctionProgram)(nil)
var _ ProgramWithPrompt = (*FunctionProgram)(nil)
