package memory

import (
	"context"
	"fmt"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/storage/chatstore"
)

const (
	// DefaultTokenLimitRatio is the default ratio of context window to use.
	DefaultTokenLimitRatio = 0.75
	// DefaultTokenLimit is the default token limit.
	DefaultTokenLimit = 3000
)

// ChatMemoryBuffer is a memory buffer with token limit enforcement.
type ChatMemoryBuffer struct {
	*BaseMemory
	tokenLimit  int
	tokenizerFn TokenizerFunc
}

// ChatMemoryBufferOption configures a ChatMemoryBuffer.
type ChatMemoryBufferOption func(*ChatMemoryBuffer)

// WithTokenLimit sets the token limit.
func WithTokenLimit(limit int) ChatMemoryBufferOption {
	return func(m *ChatMemoryBuffer) {
		m.tokenLimit = limit
	}
}

// WithTokenizer sets the tokenizer function.
func WithTokenizer(fn TokenizerFunc) ChatMemoryBufferOption {
	return func(m *ChatMemoryBuffer) {
		m.tokenizerFn = fn
	}
}

// WithBufferChatStore sets the chat store.
func WithBufferChatStore(store chatstore.ChatStore) ChatMemoryBufferOption {
	return func(m *ChatMemoryBuffer) {
		m.chatStore = store
	}
}

// WithBufferChatStoreKey sets the chat store key.
func WithBufferChatStoreKey(key string) ChatMemoryBufferOption {
	return func(m *ChatMemoryBuffer) {
		m.chatStoreKey = key
	}
}

// NewChatMemoryBuffer creates a new ChatMemoryBuffer.
func NewChatMemoryBuffer(opts ...ChatMemoryBufferOption) *ChatMemoryBuffer {
	m := &ChatMemoryBuffer{
		BaseMemory:  NewBaseMemory(),
		tokenLimit:  DefaultTokenLimit,
		tokenizerFn: DefaultTokenizer,
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// NewChatMemoryBufferFromDefaults creates a ChatMemoryBuffer with optional LLM for context window.
func NewChatMemoryBufferFromDefaults(
	chatHistory []llm.ChatMessage,
	llmModel llm.LLM,
	tokenLimit int,
	opts ...ChatMemoryBufferOption,
) (*ChatMemoryBuffer, error) {
	// Determine token limit
	if tokenLimit <= 0 {
		if llmWithMeta, ok := llmModel.(llm.LLMWithMetadata); ok && llmWithMeta != nil {
			metadata := llmWithMeta.Metadata()
			tokenLimit = int(float64(metadata.ContextWindow) * DefaultTokenLimitRatio)
		} else {
			tokenLimit = DefaultTokenLimit
		}
	}

	m := NewChatMemoryBuffer(append(opts, WithTokenLimit(tokenLimit))...)

	// Set initial chat history
	if len(chatHistory) > 0 {
		ctx := context.Background()
		if err := m.Set(ctx, chatHistory); err != nil {
			return nil, err
		}
	}

	return m, nil
}

// TokenLimit returns the token limit.
func (m *ChatMemoryBuffer) TokenLimit() int {
	return m.tokenLimit
}

// Get retrieves chat history within the token limit.
func (m *ChatMemoryBuffer) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	return m.GetWithInitialTokenCount(ctx, input, 0)
}

// GetWithInitialTokenCount retrieves chat history accounting for initial tokens.
func (m *ChatMemoryBuffer) GetWithInitialTokenCount(ctx context.Context, input string, initialTokenCount int) ([]llm.ChatMessage, error) {
	chatHistory, err := m.GetAll(ctx)
	if err != nil {
		return nil, err
	}

	if initialTokenCount > m.tokenLimit {
		return nil, fmt.Errorf("initial token count %d exceeds token limit %d", initialTokenCount, m.tokenLimit)
	}

	if len(chatHistory) == 0 {
		return chatHistory, nil
	}

	messageCount := len(chatHistory)
	curMessages := chatHistory[len(chatHistory)-messageCount:]
	tokenCount := m.tokenCountForMessages(curMessages) + initialTokenCount

	// Trim messages until we're under the token limit
	for tokenCount > m.tokenLimit && messageCount > 1 {
		messageCount--

		// Skip assistant and tool messages at the start
		for messageCount > 0 && (chatHistory[len(chatHistory)-messageCount].Role == llm.MessageRoleAssistant ||
			chatHistory[len(chatHistory)-messageCount].Role == llm.MessageRoleTool) {
			messageCount--
		}

		if messageCount <= 0 {
			break
		}

		curMessages = chatHistory[len(chatHistory)-messageCount:]
		tokenCount = m.tokenCountForMessages(curMessages) + initialTokenCount
	}

	// If still over limit or no messages, return empty
	if tokenCount > m.tokenLimit || messageCount <= 0 {
		return []llm.ChatMessage{}, nil
	}

	return chatHistory[len(chatHistory)-messageCount:], nil
}

// tokenCountForMessages counts tokens in a list of messages.
func (m *ChatMemoryBuffer) tokenCountForMessages(messages []llm.ChatMessage) int {
	if len(messages) == 0 {
		return 0
	}

	var totalContent string
	for _, msg := range messages {
		totalContent += " " + msg.Content
	}

	return m.tokenizerFn(totalContent)
}

// Ensure ChatMemoryBuffer implements Memory.
var _ Memory = (*ChatMemoryBuffer)(nil)
