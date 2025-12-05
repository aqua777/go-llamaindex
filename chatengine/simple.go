package chatengine

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
)

// SimpleChatEngine is a chat engine that directly uses an LLM without a knowledge base.
type SimpleChatEngine struct {
	*BaseChatEngine
	memory memory.Memory
}

// SimpleChatEngineOption configures a SimpleChatEngine.
type SimpleChatEngineOption func(*SimpleChatEngine)

// WithSimpleChatEngineLLM sets the LLM.
func WithSimpleChatEngineLLM(l llm.LLM) SimpleChatEngineOption {
	return func(e *SimpleChatEngine) {
		e.llm = l
	}
}

// WithSimpleChatEngineMemory sets the memory.
func WithSimpleChatEngineMemory(m memory.Memory) SimpleChatEngineOption {
	return func(e *SimpleChatEngine) {
		e.memory = m
	}
}

// WithSimpleChatEngineSystemPrompt sets the system prompt.
func WithSimpleChatEngineSystemPrompt(prompt string) SimpleChatEngineOption {
	return func(e *SimpleChatEngine) {
		e.prefixMessages = []llm.ChatMessage{
			{Role: llm.MessageRoleSystem, Content: prompt},
		}
	}
}

// WithSimpleChatEnginePrefixMessages sets the prefix messages.
func WithSimpleChatEnginePrefixMessages(messages []llm.ChatMessage) SimpleChatEngineOption {
	return func(e *SimpleChatEngine) {
		e.prefixMessages = messages
	}
}

// NewSimpleChatEngine creates a new SimpleChatEngine.
func NewSimpleChatEngine(opts ...SimpleChatEngineOption) *SimpleChatEngine {
	e := &SimpleChatEngine{
		BaseChatEngine: NewBaseChatEngine(),
		memory:         memory.NewSimpleMemory(),
	}

	for _, opt := range opts {
		opt(e)
	}

	return e
}

// NewSimpleChatEngineFromDefaults creates a SimpleChatEngine with defaults.
func NewSimpleChatEngineFromDefaults(
	llmModel llm.LLM,
	chatHistory []llm.ChatMessage,
	systemPrompt string,
	opts ...SimpleChatEngineOption,
) (*SimpleChatEngine, error) {
	// Create memory with chat history
	mem := memory.NewChatMemoryBuffer()
	if len(chatHistory) > 0 {
		ctx := context.Background()
		if err := mem.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	// Build options
	allOpts := []SimpleChatEngineOption{
		WithSimpleChatEngineLLM(llmModel),
		WithSimpleChatEngineMemory(mem),
	}

	if systemPrompt != "" {
		allOpts = append(allOpts, WithSimpleChatEngineSystemPrompt(systemPrompt))
	}

	allOpts = append(allOpts, opts...)

	return NewSimpleChatEngine(allOpts...), nil
}

// Chat sends a message and returns a response.
func (e *SimpleChatEngine) Chat(ctx context.Context, message string) (*ChatResponse, error) {
	return e.ChatWithHistory(ctx, message, nil)
}

// ChatWithHistory sends a message with explicit chat history.
func (e *SimpleChatEngine) ChatWithHistory(ctx context.Context, message string, chatHistory []llm.ChatMessage) (*ChatResponse, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM not configured")
	}

	// Set chat history if provided
	if chatHistory != nil {
		if err := e.memory.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	// Add user message to memory
	userMessage := llm.ChatMessage{Role: llm.MessageRoleUser, Content: message}
	if err := e.memory.Put(ctx, userMessage); err != nil {
		return nil, err
	}

	// Get all messages for the LLM
	memoryMessages, err := e.memory.Get(ctx, message)
	if err != nil {
		return nil, err
	}

	// Combine prefix messages with memory messages
	allMessages := append(e.prefixMessages, memoryMessages...)

	// Call LLM
	response, err := e.llm.Chat(ctx, allMessages)
	if err != nil {
		return nil, err
	}

	// Add assistant message to memory
	assistantMessage := llm.ChatMessage{Role: llm.MessageRoleAssistant, Content: response}
	if err := e.memory.Put(ctx, assistantMessage); err != nil {
		return nil, err
	}

	return NewChatResponse(response), nil
}

// StreamChat sends a message and returns a streaming response.
func (e *SimpleChatEngine) StreamChat(ctx context.Context, message string) (*StreamingChatResponse, error) {
	if e.llm == nil {
		return nil, fmt.Errorf("LLM not configured")
	}

	// Add user message to memory
	userMessage := llm.ChatMessage{Role: llm.MessageRoleUser, Content: message}
	if err := e.memory.Put(ctx, userMessage); err != nil {
		return nil, err
	}

	// Get all messages for the LLM
	memoryMessages, err := e.memory.Get(ctx, message)
	if err != nil {
		return nil, err
	}

	// Combine prefix messages with memory messages
	allMessages := append(e.prefixMessages, memoryMessages...)

	// Start streaming
	streamChan, err := e.llm.Stream(ctx, formatMessagesForStream(allMessages))
	if err != nil {
		return nil, err
	}

	// Create output channel that also writes to memory
	outputChan := make(chan string)
	go func() {
		defer close(outputChan)
		var fullResponse string
		for token := range streamChan {
			fullResponse += token
			outputChan <- token
		}
		// Add assistant message to memory
		assistantMessage := llm.ChatMessage{Role: llm.MessageRoleAssistant, Content: fullResponse}
		_ = e.memory.Put(ctx, assistantMessage)
	}()

	return NewStreamingChatResponse(outputChan), nil
}

// Reset clears the conversation state.
func (e *SimpleChatEngine) Reset(ctx context.Context) error {
	return e.memory.Reset(ctx)
}

// ChatHistory returns the current chat history.
func (e *SimpleChatEngine) ChatHistory(ctx context.Context) ([]llm.ChatMessage, error) {
	return e.memory.GetAll(ctx)
}

// formatMessagesForStream formats messages for the Stream method.
func formatMessagesForStream(messages []llm.ChatMessage) string {
	var result string
	for _, msg := range messages {
		result += string(msg.Role) + ": " + msg.Content + "\n"
	}
	return result
}

// Ensure SimpleChatEngine implements ChatEngine.
var _ ChatEngine = (*SimpleChatEngine)(nil)
