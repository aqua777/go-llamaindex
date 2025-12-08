package chatstore

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"

	"github.com/aqua777/go-llamaindex/llm"
)

const (
	// DefaultPersistDir is the default directory for persistence.
	DefaultPersistDir = "./storage"
	// DefaultPersistFilename is the default filename for persistence.
	DefaultPersistFilename = "chat_store.json"
)

// SimpleChatStore is an in-memory chat store with optional persistence.
type SimpleChatStore struct {
	mu    sync.RWMutex
	store map[string][]llm.ChatMessage
}

// NewSimpleChatStore creates a new SimpleChatStore.
func NewSimpleChatStore() *SimpleChatStore {
	return &SimpleChatStore{
		store: make(map[string][]llm.ChatMessage),
	}
}

// SetMessages sets the messages for a key.
func (s *SimpleChatStore) SetMessages(ctx context.Context, key string, messages []llm.ChatMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Make a copy to prevent external mutation
	msgCopy := make([]llm.ChatMessage, len(messages))
	copy(msgCopy, messages)
	s.store[key] = msgCopy

	return nil
}

// GetMessages retrieves all messages for a key.
func (s *SimpleChatStore) GetMessages(ctx context.Context, key string) ([]llm.ChatMessage, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	messages, ok := s.store[key]
	if !ok {
		return []llm.ChatMessage{}, nil
	}

	// Return a copy to prevent external mutation
	result := make([]llm.ChatMessage, len(messages))
	copy(result, messages)
	return result, nil
}

// AddMessage adds a message to the message list for a key.
func (s *SimpleChatStore) AddMessage(ctx context.Context, key string, message llm.ChatMessage, idx int) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	messages, ok := s.store[key]
	if !ok {
		messages = []llm.ChatMessage{}
	}

	if idx < 0 || idx >= len(messages) {
		// Append to end
		messages = append(messages, message)
	} else {
		// Insert at index
		messages = append(messages[:idx], append([]llm.ChatMessage{message}, messages[idx:]...)...)
	}

	s.store[key] = messages
	return nil
}

// DeleteMessages deletes all messages for a key.
func (s *SimpleChatStore) DeleteMessages(ctx context.Context, key string) ([]llm.ChatMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	messages, ok := s.store[key]
	if !ok {
		return nil, nil
	}

	delete(s.store, key)
	return messages, nil
}

// DeleteMessage deletes a specific message by index.
func (s *SimpleChatStore) DeleteMessage(ctx context.Context, key string, idx int) (*llm.ChatMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	messages, ok := s.store[key]
	if !ok {
		return nil, nil
	}

	if idx < 0 || idx >= len(messages) {
		return nil, nil
	}

	deleted := messages[idx]
	s.store[key] = append(messages[:idx], messages[idx+1:]...)
	return &deleted, nil
}

// DeleteLastMessage deletes the last message for a key.
func (s *SimpleChatStore) DeleteLastMessage(ctx context.Context, key string) (*llm.ChatMessage, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	messages, ok := s.store[key]
	if !ok || len(messages) == 0 {
		return nil, nil
	}

	deleted := messages[len(messages)-1]
	s.store[key] = messages[:len(messages)-1]
	return &deleted, nil
}

// GetKeys returns all keys in the store.
func (s *SimpleChatStore) GetKeys(ctx context.Context) ([]string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	keys := make([]string, 0, len(s.store))
	for key := range s.store {
		keys = append(keys, key)
	}
	return keys, nil
}

// Persist saves the chat store to disk.
func (s *SimpleChatStore) Persist(ctx context.Context, persistPath string) error {
	if persistPath == "" {
		persistPath = filepath.Join(DefaultPersistDir, DefaultPersistFilename)
	}

	dirPath := filepath.Dir(persistPath)
	if err := os.MkdirAll(dirPath, 0755); err != nil {
		return err
	}

	s.mu.RLock()
	data := s.toDict()
	s.mu.RUnlock()

	jsonData, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(persistPath, jsonData, 0644)
}

// SimpleChatStoreFromPersistPath loads a SimpleChatStore from a persist path.
func SimpleChatStoreFromPersistPath(ctx context.Context, persistPath string) (*SimpleChatStore, error) {
	if persistPath == "" {
		persistPath = filepath.Join(DefaultPersistDir, DefaultPersistFilename)
	}

	store := NewSimpleChatStore()

	// Check if file exists
	if _, err := os.Stat(persistPath); os.IsNotExist(err) {
		return store, nil
	}

	data, err := os.ReadFile(persistPath)
	if err != nil {
		return nil, err
	}

	var storeData map[string][]llm.ChatMessage
	if err := json.Unmarshal(data, &storeData); err != nil {
		return nil, err
	}

	store.store = storeData
	return store, nil
}

// toDict returns a copy of the internal data.
func (s *SimpleChatStore) toDict() map[string][]llm.ChatMessage {
	result := make(map[string][]llm.ChatMessage, len(s.store))
	for k, v := range s.store {
		msgCopy := make([]llm.ChatMessage, len(v))
		copy(msgCopy, v)
		result[k] = msgCopy
	}
	return result
}

// ToDict returns a copy of the internal data (public version).
func (s *SimpleChatStore) ToDict() map[string][]llm.ChatMessage {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.toDict()
}

// SimpleChatStoreFromDict creates a SimpleChatStore from a dictionary.
func SimpleChatStoreFromDict(data map[string][]llm.ChatMessage) *SimpleChatStore {
	store := NewSimpleChatStore()
	for k, v := range data {
		msgCopy := make([]llm.ChatMessage, len(v))
		copy(msgCopy, v)
		store.store[k] = msgCopy
	}
	return store
}

// Ensure SimpleChatStore implements ChatStore.
var _ ChatStore = (*SimpleChatStore)(nil)
