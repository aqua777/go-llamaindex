package llm

import "context"

// ChatMessage represents a message in a chat conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// LLM is the interface for interacting with Large Language Models.
type LLM interface {
	// Complete generates a completion for a given prompt.
	Complete(ctx context.Context, prompt string) (string, error)
	// Chat generates a response for a list of chat messages.
	Chat(ctx context.Context, messages []ChatMessage) (string, error)
	// Stream generates a streaming completion for a given prompt.
	Stream(ctx context.Context, prompt string) (<-chan string, error)
}
