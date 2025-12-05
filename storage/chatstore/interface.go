// Package chatstore provides chat store interfaces and implementations.
package chatstore

import (
	"context"

	"github.com/aqua777/go-llamaindex/llm"
)

// ChatStore is the interface for storing chat message history.
type ChatStore interface {
	// SetMessages sets the messages for a key, replacing any existing messages.
	SetMessages(ctx context.Context, key string, messages []llm.ChatMessage) error

	// GetMessages retrieves all messages for a key.
	// Returns an empty slice if the key doesn't exist.
	GetMessages(ctx context.Context, key string) ([]llm.ChatMessage, error)

	// AddMessage adds a message to the end of the message list for a key.
	// If idx is provided and >= 0, inserts the message at that index instead.
	AddMessage(ctx context.Context, key string, message llm.ChatMessage, idx int) error

	// DeleteMessages deletes all messages for a key.
	// Returns the deleted messages, or nil if the key didn't exist.
	DeleteMessages(ctx context.Context, key string) ([]llm.ChatMessage, error)

	// DeleteMessage deletes a specific message by index for a key.
	// Returns the deleted message, or nil if the key or index didn't exist.
	DeleteMessage(ctx context.Context, key string, idx int) (*llm.ChatMessage, error)

	// DeleteLastMessage deletes the last message for a key.
	// Returns the deleted message, or nil if the key didn't exist or was empty.
	DeleteLastMessage(ctx context.Context, key string) (*llm.ChatMessage, error)

	// GetKeys returns all keys in the store.
	GetKeys(ctx context.Context) ([]string, error)
}

// IndexNotSpecified is a sentinel value indicating no index was specified.
const IndexNotSpecified = -1
