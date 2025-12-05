package llm

import "context"

// LLM is the interface for interacting with Large Language Models.
// This is the basic interface that all LLM implementations must satisfy.
type LLM interface {
	// Complete generates a completion for a given prompt.
	Complete(ctx context.Context, prompt string) (string, error)
	// Chat generates a response for a list of chat messages.
	Chat(ctx context.Context, messages []ChatMessage) (string, error)
	// Stream generates a streaming completion for a given prompt.
	Stream(ctx context.Context, prompt string) (<-chan string, error)
}

// LLMWithMetadata extends LLM with metadata capabilities.
type LLMWithMetadata interface {
	LLM
	// Metadata returns information about the model's capabilities.
	Metadata() LLMMetadata
}

// LLMWithToolCalling extends LLM with tool/function calling capabilities.
type LLMWithToolCalling interface {
	LLM
	// ChatWithTools generates a response that may include tool calls.
	ChatWithTools(ctx context.Context, messages []ChatMessage, tools []*ToolMetadata, opts *ChatCompletionOptions) (CompletionResponse, error)
	// SupportsToolCalling returns true if the model supports tool calling.
	SupportsToolCalling() bool
}

// LLMWithStructuredOutput extends LLM with structured output capabilities.
type LLMWithStructuredOutput interface {
	LLM
	// ChatWithFormat generates a response in the specified format.
	ChatWithFormat(ctx context.Context, messages []ChatMessage, format *ResponseFormat) (string, error)
	// SupportsStructuredOutput returns true if the model supports structured output.
	SupportsStructuredOutput() bool
}

// FullLLM combines all LLM capabilities.
type FullLLM interface {
	LLMWithMetadata
	LLMWithToolCalling
	LLMWithStructuredOutput
	// StreamChat generates a streaming response for chat messages.
	StreamChat(ctx context.Context, messages []ChatMessage) (<-chan StreamToken, error)
}
