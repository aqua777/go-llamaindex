package llm

import (
	"encoding/json"
)

// ToolCall represents a request from the LLM to call a tool/function.
type ToolCall struct {
	// ID is the unique identifier for this tool call.
	ID string `json:"id"`
	// Name is the name of the tool/function to call.
	Name string `json:"name"`
	// Arguments is the JSON-encoded arguments for the tool call.
	Arguments string `json:"arguments"`
}

// NewToolCall creates a new tool call.
func NewToolCall(id, name, arguments string) *ToolCall {
	return &ToolCall{
		ID:        id,
		Name:      name,
		Arguments: arguments,
	}
}

// ParseArguments parses the JSON arguments into a map.
func (tc *ToolCall) ParseArguments() (map[string]interface{}, error) {
	var args map[string]interface{}
	if err := json.Unmarshal([]byte(tc.Arguments), &args); err != nil {
		return nil, err
	}
	return args, nil
}

// ParseArgumentsInto parses the JSON arguments into a provided struct.
func (tc *ToolCall) ParseArgumentsInto(v interface{}) error {
	return json.Unmarshal([]byte(tc.Arguments), v)
}

// ToolResult represents the result of executing a tool call.
type ToolResult struct {
	// ToolCallID is the ID of the tool call this result is for.
	ToolCallID string `json:"tool_call_id"`
	// ToolName is the name of the tool that was called.
	ToolName string `json:"tool_name"`
	// Content is the string content of the result.
	Content string `json:"content"`
	// RawInput is the original input to the tool (for debugging).
	RawInput interface{} `json:"raw_input,omitempty"`
	// RawOutput is the original output from the tool (for debugging).
	RawOutput interface{} `json:"raw_output,omitempty"`
	// IsError indicates if the tool execution resulted in an error.
	IsError bool `json:"is_error,omitempty"`
}

// NewToolResult creates a new tool result.
func NewToolResult(toolCallID, toolName, content string) *ToolResult {
	return &ToolResult{
		ToolCallID: toolCallID,
		ToolName:   toolName,
		Content:    content,
	}
}

// NewToolResultError creates a new tool result indicating an error.
func NewToolResultError(toolCallID, toolName, errorMsg string) *ToolResult {
	return &ToolResult{
		ToolCallID: toolCallID,
		ToolName:   toolName,
		Content:    errorMsg,
		IsError:    true,
	}
}

// ToolMetadata describes a tool that can be called by the LLM.
type ToolMetadata struct {
	// Name is the unique name of the tool.
	Name string `json:"name"`
	// Description is a human-readable description of what the tool does.
	Description string `json:"description"`
	// Parameters is the JSON Schema describing the tool's parameters.
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	// ReturnDirect indicates if the tool output should be returned directly
	// to the user without further LLM processing.
	ReturnDirect bool `json:"return_direct,omitempty"`
}

// NewToolMetadata creates a new tool metadata.
func NewToolMetadata(name, description string) *ToolMetadata {
	return &ToolMetadata{
		Name:        name,
		Description: description,
		Parameters: map[string]interface{}{
			"type":       "object",
			"properties": map[string]interface{}{},
			"required":   []string{},
		},
	}
}

// WithParameters sets the parameters schema for the tool.
func (tm *ToolMetadata) WithParameters(params map[string]interface{}) *ToolMetadata {
	tm.Parameters = params
	return tm
}

// WithReturnDirect sets the return_direct flag.
func (tm *ToolMetadata) WithReturnDirect(returnDirect bool) *ToolMetadata {
	tm.ReturnDirect = returnDirect
	return tm
}

// ToOpenAITool converts the tool metadata to OpenAI function format.
func (tm *ToolMetadata) ToOpenAITool() map[string]interface{} {
	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        tm.Name,
			"description": tm.Description,
			"parameters":  tm.Parameters,
		},
	}
}

// ToOpenAIFunction converts the tool metadata to OpenAI function format (legacy).
func (tm *ToolMetadata) ToOpenAIFunction() map[string]interface{} {
	return map[string]interface{}{
		"name":        tm.Name,
		"description": tm.Description,
		"parameters":  tm.Parameters,
	}
}

// ToolChoice represents how the LLM should handle tool selection.
type ToolChoice string

const (
	// ToolChoiceAuto lets the model decide whether to call tools.
	ToolChoiceAuto ToolChoice = "auto"
	// ToolChoiceNone prevents the model from calling any tools.
	ToolChoiceNone ToolChoice = "none"
	// ToolChoiceRequired forces the model to call at least one tool.
	ToolChoiceRequired ToolChoice = "required"
)

// SpecificToolChoice creates a tool choice that forces a specific tool.
func SpecificToolChoice(toolName string) map[string]interface{} {
	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name": toolName,
		},
	}
}

// ChatCompletionOptions contains options for chat completion requests.
type ChatCompletionOptions struct {
	// Temperature controls randomness (0.0 to 2.0).
	Temperature *float32 `json:"temperature,omitempty"`
	// MaxTokens limits the number of tokens to generate.
	MaxTokens *int `json:"max_tokens,omitempty"`
	// TopP controls nucleus sampling.
	TopP *float32 `json:"top_p,omitempty"`
	// FrequencyPenalty penalizes frequent tokens (-2.0 to 2.0).
	FrequencyPenalty *float32 `json:"frequency_penalty,omitempty"`
	// PresencePenalty penalizes tokens already present (-2.0 to 2.0).
	PresencePenalty *float32 `json:"presence_penalty,omitempty"`
	// Stop sequences that will stop generation.
	Stop []string `json:"stop,omitempty"`
	// Tools available for the model to call.
	Tools []*ToolMetadata `json:"tools,omitempty"`
	// ToolChoice controls how tools are selected.
	ToolChoice interface{} `json:"tool_choice,omitempty"`
	// ResponseFormat specifies the output format (e.g., JSON).
	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`
	// Seed for deterministic sampling.
	Seed *int `json:"seed,omitempty"`
}

// ResponseFormat specifies the format of the LLM response.
type ResponseFormat struct {
	// Type is the format type ("text" or "json_object").
	Type string `json:"type"`
	// JSONSchema is the JSON schema for structured output (if supported).
	JSONSchema map[string]interface{} `json:"json_schema,omitempty"`
}

// NewJSONResponseFormat creates a response format for JSON output.
func NewJSONResponseFormat() *ResponseFormat {
	return &ResponseFormat{Type: "json_object"}
}

// NewJSONSchemaResponseFormat creates a response format with a specific JSON schema.
func NewJSONSchemaResponseFormat(schema map[string]interface{}) *ResponseFormat {
	return &ResponseFormat{
		Type:       "json_schema",
		JSONSchema: schema,
	}
}
