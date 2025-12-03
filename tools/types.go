// Package tools provides tool abstractions for LlamaIndex.
package tools

import (
	"context"
	"encoding/json"
	"fmt"
)

// ToolMetadata contains metadata about a tool.
type ToolMetadata struct {
	// Name is the unique name of the tool.
	Name string `json:"name"`
	// Description describes what the tool does.
	Description string `json:"description"`
	// Parameters is the JSON Schema for the tool's parameters.
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	// ReturnDirect indicates if the tool output should be returned directly.
	ReturnDirect bool `json:"return_direct,omitempty"`
}

// NewToolMetadata creates a new ToolMetadata with the given name and description.
func NewToolMetadata(name, description string) *ToolMetadata {
	return &ToolMetadata{
		Name:        name,
		Description: description,
		Parameters:  DefaultParameters(),
	}
}

// DefaultParameters returns the default parameters schema.
func DefaultParameters() map[string]interface{} {
	return map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"input": map[string]interface{}{
				"title": "input query string",
				"type":  "string",
			},
		},
		"required": []string{"input"},
	}
}

// GetName returns the tool name, or an error if not set.
func (m *ToolMetadata) GetName() string {
	return m.Name
}

// GetParametersDict returns the parameters as a dictionary.
func (m *ToolMetadata) GetParametersDict() map[string]interface{} {
	if m.Parameters == nil {
		return DefaultParameters()
	}
	return m.Parameters
}

// GetParametersJSON returns the parameters as a JSON string.
func (m *ToolMetadata) GetParametersJSON() (string, error) {
	params := m.GetParametersDict()
	data, err := json.Marshal(params)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ToOpenAITool converts the metadata to OpenAI tool format.
func (m *ToolMetadata) ToOpenAITool() map[string]interface{} {
	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        m.Name,
			"description": m.Description,
			"parameters":  m.GetParametersDict(),
		},
	}
}

// ToOpenAIFunction converts the metadata to OpenAI function format (deprecated).
func (m *ToolMetadata) ToOpenAIFunction() map[string]interface{} {
	return map[string]interface{}{
		"name":        m.Name,
		"description": m.Description,
		"parameters":  m.GetParametersDict(),
	}
}

// ToolOutput represents the output of a tool execution.
type ToolOutput struct {
	// Content is the text content of the output.
	Content string `json:"content"`
	// ToolName is the name of the tool that produced this output.
	ToolName string `json:"tool_name"`
	// RawInput is the raw input that was passed to the tool.
	RawInput map[string]interface{} `json:"raw_input,omitempty"`
	// RawOutput is the raw output from the tool.
	RawOutput interface{} `json:"raw_output,omitempty"`
	// IsError indicates if this output represents an error.
	IsError bool `json:"is_error,omitempty"`
	// Error holds the error if IsError is true.
	Error error `json:"-"`
}

// NewToolOutput creates a new ToolOutput.
func NewToolOutput(toolName, content string) *ToolOutput {
	return &ToolOutput{
		Content:  content,
		ToolName: toolName,
		RawInput: make(map[string]interface{}),
	}
}

// NewToolOutputWithInput creates a new ToolOutput with input.
func NewToolOutputWithInput(toolName, content string, rawInput map[string]interface{}, rawOutput interface{}) *ToolOutput {
	return &ToolOutput{
		Content:   content,
		ToolName:  toolName,
		RawInput:  rawInput,
		RawOutput: rawOutput,
	}
}

// NewErrorToolOutput creates a new ToolOutput representing an error.
func NewErrorToolOutput(toolName string, err error) *ToolOutput {
	return &ToolOutput{
		Content:  err.Error(),
		ToolName: toolName,
		IsError:  true,
		Error:    err,
	}
}

// String returns the content of the tool output.
func (o *ToolOutput) String() string {
	return o.Content
}

// Tool is the interface that all tools must implement.
type Tool interface {
	// Metadata returns the tool's metadata.
	Metadata() *ToolMetadata
	// Call executes the tool with the given input.
	Call(ctx context.Context, input interface{}) (*ToolOutput, error)
}

// AsyncTool extends Tool with async capabilities.
type AsyncTool interface {
	Tool
	// CallAsync executes the tool asynchronously.
	CallAsync(ctx context.Context, input interface{}) <-chan *ToolResult
}

// ToolResult wraps a ToolOutput with an error for async operations.
type ToolResult struct {
	Output *ToolOutput
	Error  error
}

// BaseTool provides a base implementation for tools.
type BaseTool struct {
	metadata *ToolMetadata
}

// NewBaseTool creates a new BaseTool with the given metadata.
func NewBaseTool(metadata *ToolMetadata) *BaseTool {
	return &BaseTool{
		metadata: metadata,
	}
}

// Metadata returns the tool's metadata.
func (t *BaseTool) Metadata() *ToolMetadata {
	return t.metadata
}

// ToolSpec defines a specification for creating tools.
type ToolSpec struct {
	// Name is the name of the tool.
	Name string
	// Description is the description of the tool.
	Description string
	// Parameters is the JSON Schema for parameters.
	Parameters map[string]interface{}
	// Handler is the function that handles tool calls.
	Handler func(ctx context.Context, input map[string]interface{}) (interface{}, error)
}

// ToTool converts a ToolSpec to a Tool.
func (s *ToolSpec) ToTool() Tool {
	return &specTool{
		spec: s,
	}
}

// specTool wraps a ToolSpec as a Tool.
type specTool struct {
	spec *ToolSpec
}

func (t *specTool) Metadata() *ToolMetadata {
	return &ToolMetadata{
		Name:        t.spec.Name,
		Description: t.spec.Description,
		Parameters:  t.spec.Parameters,
	}
}

func (t *specTool) Call(ctx context.Context, input interface{}) (*ToolOutput, error) {
	// Convert input to map
	var inputMap map[string]interface{}
	switch v := input.(type) {
	case map[string]interface{}:
		inputMap = v
	case string:
		inputMap = map[string]interface{}{"input": v}
	default:
		// Try to marshal and unmarshal to convert to map
		data, err := json.Marshal(input)
		if err != nil {
			return nil, fmt.Errorf("failed to convert input to map: %w", err)
		}
		if err := json.Unmarshal(data, &inputMap); err != nil {
			return nil, fmt.Errorf("failed to convert input to map: %w", err)
		}
	}

	result, err := t.spec.Handler(ctx, inputMap)
	if err != nil {
		return NewErrorToolOutput(t.spec.Name, err), err
	}

	content := fmt.Sprintf("%v", result)
	return NewToolOutputWithInput(t.spec.Name, content, inputMap, result), nil
}
