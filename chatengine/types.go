// Package chatengine provides chat engine abstractions for conversational AI.
package chatengine

import (
	"context"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/schema"
)

// ChatMode represents the chat engine mode.
type ChatMode string

const (
	// ChatModeSimple corresponds to SimpleChatEngine.
	ChatModeSimple ChatMode = "simple"
	// ChatModeContext corresponds to ContextChatEngine.
	ChatModeContext ChatMode = "context"
	// ChatModeCondensePlusContext corresponds to CondensePlusContextChatEngine.
	ChatModeCondensePlusContext ChatMode = "condense_plus_context"
)

// ChatResponse represents a chat response.
type ChatResponse struct {
	// Response is the text response.
	Response string
	// SourceNodes are the source nodes used to generate the response.
	SourceNodes []schema.NodeWithScore
	// Sources are tool outputs used in generating the response.
	Sources []ToolSource
	// Metadata contains additional response metadata.
	Metadata map[string]interface{}
}

// NewChatResponse creates a new ChatResponse.
func NewChatResponse(response string) *ChatResponse {
	return &ChatResponse{
		Response:    response,
		SourceNodes: []schema.NodeWithScore{},
		Sources:     []ToolSource{},
		Metadata:    make(map[string]interface{}),
	}
}

// String returns the response text.
func (r *ChatResponse) String() string {
	return r.Response
}

// ToolSource represents a source from a tool.
type ToolSource struct {
	// ToolName is the name of the tool.
	ToolName string
	// Content is the content from the tool.
	Content string
	// RawInput is the raw input to the tool.
	RawInput map[string]interface{}
	// RawOutput is the raw output from the tool.
	RawOutput interface{}
}

// StreamingChatResponse represents a streaming chat response.
type StreamingChatResponse struct {
	// ResponseChan is the channel for streaming response tokens.
	ResponseChan <-chan string
	// SourceNodes are the source nodes used to generate the response.
	SourceNodes []schema.NodeWithScore
	// Sources are tool outputs used in generating the response.
	Sources []ToolSource
	// done indicates if streaming is complete.
	done bool
	// fullResponse accumulates the full response.
	fullResponse string
}

// NewStreamingChatResponse creates a new StreamingChatResponse.
func NewStreamingChatResponse(responseChan <-chan string) *StreamingChatResponse {
	return &StreamingChatResponse{
		ResponseChan: responseChan,
		SourceNodes:  []schema.NodeWithScore{},
		Sources:      []ToolSource{},
	}
}

// Response returns the full accumulated response.
func (r *StreamingChatResponse) Response() string {
	return r.fullResponse
}

// IsDone returns whether streaming is complete.
func (r *StreamingChatResponse) IsDone() bool {
	return r.done
}

// Consume reads all tokens from the stream and returns the full response.
func (r *StreamingChatResponse) Consume() string {
	for token := range r.ResponseChan {
		r.fullResponse += token
	}
	r.done = true
	return r.fullResponse
}

// ChatEngine is the interface for chat engines.
type ChatEngine interface {
	// Chat sends a message and returns a response.
	Chat(ctx context.Context, message string) (*ChatResponse, error)

	// ChatWithHistory sends a message with explicit chat history.
	ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*ChatResponse, error)

	// StreamChat sends a message and returns a streaming response.
	StreamChat(ctx context.Context, message string) (*StreamingChatResponse, error)

	// Reset clears the conversation state.
	Reset(ctx context.Context) error

	// ChatHistory returns the current chat history.
	ChatHistory(ctx context.Context) ([]llm.ChatMessage, error)
}

// BaseChatEngine provides common functionality for chat engines.
type BaseChatEngine struct {
	llm            llm.LLM
	prefixMessages []llm.ChatMessage
}

// BaseChatEngineOption configures a BaseChatEngine.
type BaseChatEngineOption func(*BaseChatEngine)

// WithLLM sets the LLM.
func WithLLM(l llm.LLM) BaseChatEngineOption {
	return func(e *BaseChatEngine) {
		e.llm = l
	}
}

// WithPrefixMessages sets the prefix messages (e.g., system prompt).
func WithPrefixMessages(messages []llm.ChatMessage) BaseChatEngineOption {
	return func(e *BaseChatEngine) {
		e.prefixMessages = messages
	}
}

// WithSystemPrompt sets a system prompt as the first prefix message.
func WithSystemPrompt(prompt string) BaseChatEngineOption {
	return func(e *BaseChatEngine) {
		e.prefixMessages = []llm.ChatMessage{
			{Role: llm.MessageRoleSystem, Content: prompt},
		}
	}
}

// NewBaseChatEngine creates a new BaseChatEngine.
func NewBaseChatEngine(opts ...BaseChatEngineOption) *BaseChatEngine {
	e := &BaseChatEngine{
		prefixMessages: []llm.ChatMessage{},
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// LLM returns the LLM.
func (e *BaseChatEngine) LLM() llm.LLM {
	return e.llm
}

// PrefixMessages returns the prefix messages.
func (e *BaseChatEngine) PrefixMessages() []llm.ChatMessage {
	return e.prefixMessages
}
