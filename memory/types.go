// Package memory provides memory abstractions for chat history management.
package memory

import (
	"context"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/storage/chatstore"
)

// DefaultChatStoreKey is the default key for chat history storage.
const DefaultChatStoreKey = "chat_history"

// Memory is the interface for all memory types.
type Memory interface {
	// Get retrieves chat history, optionally filtered by input query.
	Get(ctx context.Context, input string) ([]llm.ChatMessage, error)

	// GetAll retrieves all chat history.
	GetAll(ctx context.Context) ([]llm.ChatMessage, error)

	// Put adds a message to the chat history.
	Put(ctx context.Context, message llm.ChatMessage) error

	// PutMessages adds multiple messages to the chat history.
	PutMessages(ctx context.Context, messages []llm.ChatMessage) error

	// Set replaces the entire chat history.
	Set(ctx context.Context, messages []llm.ChatMessage) error

	// Reset clears all chat history.
	Reset(ctx context.Context) error
}

// BaseMemory provides common functionality for memory implementations.
type BaseMemory struct {
	chatStore    chatstore.ChatStore
	chatStoreKey string
}

// BaseMemoryOption configures a BaseMemory.
type BaseMemoryOption func(*BaseMemory)

// WithChatStore sets the chat store.
func WithChatStore(store chatstore.ChatStore) BaseMemoryOption {
	return func(m *BaseMemory) {
		m.chatStore = store
	}
}

// WithChatStoreKey sets the chat store key.
func WithChatStoreKey(key string) BaseMemoryOption {
	return func(m *BaseMemory) {
		m.chatStoreKey = key
	}
}

// NewBaseMemory creates a new BaseMemory.
func NewBaseMemory(opts ...BaseMemoryOption) *BaseMemory {
	m := &BaseMemory{
		chatStore:    chatstore.NewSimpleChatStore(),
		chatStoreKey: DefaultChatStoreKey,
	}

	for _, opt := range opts {
		opt(m)
	}

	return m
}

// ChatStore returns the underlying chat store.
func (m *BaseMemory) ChatStore() chatstore.ChatStore {
	return m.chatStore
}

// ChatStoreKey returns the chat store key.
func (m *BaseMemory) ChatStoreKey() string {
	return m.chatStoreKey
}

// GetAll retrieves all chat history.
func (m *BaseMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	return m.chatStore.GetMessages(ctx, m.chatStoreKey)
}

// Put adds a message to the chat history.
func (m *BaseMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	return m.chatStore.AddMessage(ctx, m.chatStoreKey, message, chatstore.IndexNotSpecified)
}

// PutMessages adds multiple messages to the chat history.
func (m *BaseMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

// Set replaces the entire chat history.
func (m *BaseMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	return m.chatStore.SetMessages(ctx, m.chatStoreKey, messages)
}

// Reset clears all chat history.
func (m *BaseMemory) Reset(ctx context.Context) error {
	_, err := m.chatStore.DeleteMessages(ctx, m.chatStoreKey)
	return err
}

// TokenizerFunc is a function that counts tokens in a string.
type TokenizerFunc func(text string) int

// DefaultTokenizer is a simple word-based tokenizer.
func DefaultTokenizer(text string) int {
	// Simple approximation: ~4 characters per token
	return len(text) / 4
}

// SimpleMemory is a basic memory implementation that stores all messages.
type SimpleMemory struct {
	*BaseMemory
}

// NewSimpleMemory creates a new SimpleMemory.
func NewSimpleMemory(opts ...BaseMemoryOption) *SimpleMemory {
	return &SimpleMemory{
		BaseMemory: NewBaseMemory(opts...),
	}
}

// Get retrieves all chat history (input is ignored).
func (m *SimpleMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	return m.GetAll(ctx)
}

// Ensure SimpleMemory implements Memory.
var _ Memory = (*SimpleMemory)(nil)
