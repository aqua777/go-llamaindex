package llm

import (
	"encoding/json"
)

// MessageRole represents the role of a message sender.
type MessageRole string

const (
	// MessageRoleSystem is for system instructions.
	MessageRoleSystem MessageRole = "system"
	// MessageRoleUser is for user messages.
	MessageRoleUser MessageRole = "user"
	// MessageRoleAssistant is for assistant responses.
	MessageRoleAssistant MessageRole = "assistant"
	// MessageRoleTool is for tool/function results.
	MessageRoleTool MessageRole = "tool"
)

// ContentBlockType represents the type of content block.
type ContentBlockType string

const (
	// ContentBlockTypeText is for text content.
	ContentBlockTypeText ContentBlockType = "text"
	// ContentBlockTypeImage is for image content.
	ContentBlockTypeImage ContentBlockType = "image"
	// ContentBlockTypeToolCall is for tool call requests.
	ContentBlockTypeToolCall ContentBlockType = "tool_call"
	// ContentBlockTypeToolResult is for tool call results.
	ContentBlockTypeToolResult ContentBlockType = "tool_result"
)

// ContentBlock represents a block of content in a message.
// Messages can contain multiple content blocks of different types.
type ContentBlock struct {
	// Type is the type of content block.
	Type ContentBlockType `json:"type"`
	// Text is the text content (for text blocks).
	Text string `json:"text,omitempty"`
	// ImageURL is the URL of the image (for image blocks).
	ImageURL string `json:"image_url,omitempty"`
	// ImageBase64 is the base64-encoded image data (for image blocks).
	ImageBase64 string `json:"image_base64,omitempty"`
	// ImageMimeType is the MIME type of the image (for image blocks).
	ImageMimeType string `json:"image_mimetype,omitempty"`
	// ToolCall contains tool call information (for tool_call blocks).
	ToolCall *ToolCall `json:"tool_call,omitempty"`
	// ToolResult contains tool result information (for tool_result blocks).
	ToolResult *ToolResult `json:"tool_result,omitempty"`
}

// NewTextBlock creates a new text content block.
func NewTextBlock(text string) ContentBlock {
	return ContentBlock{
		Type: ContentBlockTypeText,
		Text: text,
	}
}

// NewImageURLBlock creates a new image content block from a URL.
func NewImageURLBlock(url string, mimeType string) ContentBlock {
	return ContentBlock{
		Type:          ContentBlockTypeImage,
		ImageURL:      url,
		ImageMimeType: mimeType,
	}
}

// NewImageBase64Block creates a new image content block from base64 data.
func NewImageBase64Block(base64Data string, mimeType string) ContentBlock {
	return ContentBlock{
		Type:          ContentBlockTypeImage,
		ImageBase64:   base64Data,
		ImageMimeType: mimeType,
	}
}

// NewToolCallBlock creates a new tool call content block.
func NewToolCallBlock(toolCall *ToolCall) ContentBlock {
	return ContentBlock{
		Type:     ContentBlockTypeToolCall,
		ToolCall: toolCall,
	}
}

// NewToolResultBlock creates a new tool result content block.
func NewToolResultBlock(toolResult *ToolResult) ContentBlock {
	return ContentBlock{
		Type:       ContentBlockTypeToolResult,
		ToolResult: toolResult,
	}
}

// ChatMessage represents a message in a chat conversation.
// It supports both simple text content and multi-modal content blocks.
type ChatMessage struct {
	// Role is the role of the message sender.
	Role MessageRole `json:"role"`
	// Content is the simple text content (for backward compatibility).
	Content string `json:"content,omitempty"`
	// Blocks contains structured content blocks for multi-modal messages.
	Blocks []ContentBlock `json:"blocks,omitempty"`
	// Name is an optional name for the message sender.
	Name string `json:"name,omitempty"`
	// ToolCallID is the ID of the tool call this message is responding to.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

// NewChatMessage creates a new chat message with simple text content.
func NewChatMessage(role MessageRole, content string) ChatMessage {
	return ChatMessage{
		Role:    role,
		Content: content,
	}
}

// NewSystemMessage creates a new system message.
func NewSystemMessage(content string) ChatMessage {
	return NewChatMessage(MessageRoleSystem, content)
}

// NewUserMessage creates a new user message.
func NewUserMessage(content string) ChatMessage {
	return NewChatMessage(MessageRoleUser, content)
}

// NewAssistantMessage creates a new assistant message.
func NewAssistantMessage(content string) ChatMessage {
	return NewChatMessage(MessageRoleAssistant, content)
}

// NewToolMessage creates a new tool result message.
func NewToolMessage(toolCallID string, content string) ChatMessage {
	return ChatMessage{
		Role:       MessageRoleTool,
		Content:    content,
		ToolCallID: toolCallID,
	}
}

// NewMultiModalMessage creates a new message with multiple content blocks.
func NewMultiModalMessage(role MessageRole, blocks ...ContentBlock) ChatMessage {
	return ChatMessage{
		Role:   role,
		Blocks: blocks,
	}
}

// GetTextContent returns the text content of the message.
// If the message has blocks, it concatenates all text blocks.
func (m *ChatMessage) GetTextContent() string {
	if m.Content != "" {
		return m.Content
	}
	var text string
	for _, block := range m.Blocks {
		if block.Type == ContentBlockTypeText {
			text += block.Text
		}
	}
	return text
}

// HasToolCalls returns true if the message contains tool calls.
func (m *ChatMessage) HasToolCalls() bool {
	for _, block := range m.Blocks {
		if block.Type == ContentBlockTypeToolCall {
			return true
		}
	}
	return false
}

// GetToolCalls returns all tool calls in the message.
func (m *ChatMessage) GetToolCalls() []*ToolCall {
	var calls []*ToolCall
	for _, block := range m.Blocks {
		if block.Type == ContentBlockTypeToolCall && block.ToolCall != nil {
			calls = append(calls, block.ToolCall)
		}
	}
	return calls
}

// LLMMetadata contains metadata about an LLM model's capabilities.
type LLMMetadata struct {
	// ModelName is the name/identifier of the model.
	ModelName string `json:"model_name"`
	// ContextWindow is the maximum number of tokens the model can process.
	ContextWindow int `json:"context_window"`
	// NumOutputTokens is the maximum number of tokens the model can generate.
	NumOutputTokens int `json:"num_output_tokens"`
	// IsChat indicates if the model supports chat-style interactions.
	IsChat bool `json:"is_chat"`
	// IsFunctionCalling indicates if the model supports function/tool calling.
	IsFunctionCalling bool `json:"is_function_calling"`
	// IsMultiModal indicates if the model supports multi-modal inputs.
	IsMultiModal bool `json:"is_multi_modal"`
	// SystemRole indicates the role name used for system messages.
	SystemRole string `json:"system_role,omitempty"`
}

// DefaultLLMMetadata returns default metadata for unknown models.
func DefaultLLMMetadata(modelName string) LLMMetadata {
	return LLMMetadata{
		ModelName:         modelName,
		ContextWindow:     4096,
		NumOutputTokens:   1024,
		IsChat:            true,
		IsFunctionCalling: false,
		IsMultiModal:      false,
		SystemRole:        "system",
	}
}

// GPT35TurboMetadata returns metadata for GPT-3.5-Turbo.
func GPT35TurboMetadata() LLMMetadata {
	return LLMMetadata{
		ModelName:         "gpt-3.5-turbo",
		ContextWindow:     16385,
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: true,
		IsMultiModal:      false,
		SystemRole:        "system",
	}
}

// GPT4Metadata returns metadata for GPT-4.
func GPT4Metadata() LLMMetadata {
	return LLMMetadata{
		ModelName:         "gpt-4",
		ContextWindow:     8192,
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: true,
		IsMultiModal:      false,
		SystemRole:        "system",
	}
}

// GPT4TurboMetadata returns metadata for GPT-4-Turbo.
func GPT4TurboMetadata() LLMMetadata {
	return LLMMetadata{
		ModelName:         "gpt-4-turbo",
		ContextWindow:     128000,
		NumOutputTokens:   4096,
		IsChat:            true,
		IsFunctionCalling: true,
		IsMultiModal:      true,
		SystemRole:        "system",
	}
}

// GPT4oMetadata returns metadata for GPT-4o.
func GPT4oMetadata() LLMMetadata {
	return LLMMetadata{
		ModelName:         "gpt-4o",
		ContextWindow:     128000,
		NumOutputTokens:   16384,
		IsChat:            true,
		IsFunctionCalling: true,
		IsMultiModal:      true,
		SystemRole:        "system",
	}
}

// CompletionResponse represents a response from an LLM completion.
type CompletionResponse struct {
	// Text is the generated text content.
	Text string `json:"text"`
	// Message is the full chat message (for chat completions).
	Message *ChatMessage `json:"message,omitempty"`
	// Raw is the raw response from the provider (for debugging).
	Raw json.RawMessage `json:"raw,omitempty"`
	// AdditionalKwargs contains additional response metadata.
	AdditionalKwargs map[string]interface{} `json:"additional_kwargs,omitempty"`
}

// NewCompletionResponse creates a new completion response with text.
func NewCompletionResponse(text string) CompletionResponse {
	return CompletionResponse{Text: text}
}

// NewChatCompletionResponse creates a new completion response from a chat message.
func NewChatCompletionResponse(message ChatMessage) CompletionResponse {
	return CompletionResponse{
		Text:    message.GetTextContent(),
		Message: &message,
	}
}

// StreamToken represents a single token in a streaming response.
type StreamToken struct {
	// Delta is the new content added in this token.
	Delta string `json:"delta"`
	// FinishReason indicates why generation stopped (if applicable).
	FinishReason string `json:"finish_reason,omitempty"`
	// ToolCalls contains any tool calls in this token.
	ToolCalls []*ToolCall `json:"tool_calls,omitempty"`
}
