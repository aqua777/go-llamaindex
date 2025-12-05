package chatstore

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/aqua777/go-llamaindex/llm"
)

func TestSimpleChatStoreSetAndGetMessages(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{
		llm.NewUserMessage("Hello"),
		llm.NewAssistantMessage("Hi there!"),
	}

	// Set messages
	err := store.SetMessages(ctx, "user1", messages)
	if err != nil {
		t.Fatalf("SetMessages failed: %v", err)
	}

	// Get messages
	retrieved, err := store.GetMessages(ctx, "user1")
	if err != nil {
		t.Fatalf("GetMessages failed: %v", err)
	}

	if len(retrieved) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(retrieved))
	}

	if retrieved[0].Content != "Hello" {
		t.Errorf("Expected 'Hello', got '%s'", retrieved[0].Content)
	}
}

func TestSimpleChatStoreGetMessagesEmpty(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages, err := store.GetMessages(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetMessages failed: %v", err)
	}

	if len(messages) != 0 {
		t.Errorf("Expected empty slice, got %d messages", len(messages))
	}
}

func TestSimpleChatStoreAddMessage(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	// Add first message
	err := store.AddMessage(ctx, "user1", llm.NewUserMessage("Hello"), IndexNotSpecified)
	if err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	// Add second message
	err = store.AddMessage(ctx, "user1", llm.NewAssistantMessage("Hi!"), IndexNotSpecified)
	if err != nil {
		t.Fatalf("AddMessage failed: %v", err)
	}

	messages, err := store.GetMessages(ctx, "user1")
	if err != nil {
		t.Fatalf("GetMessages failed: %v", err)
	}

	if len(messages) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(messages))
	}
}

func TestSimpleChatStoreAddMessageAtIndex(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	// Add messages
	store.AddMessage(ctx, "user1", llm.NewUserMessage("First"), IndexNotSpecified)
	store.AddMessage(ctx, "user1", llm.NewUserMessage("Third"), IndexNotSpecified)

	// Insert at index 1
	err := store.AddMessage(ctx, "user1", llm.NewUserMessage("Second"), 1)
	if err != nil {
		t.Fatalf("AddMessage at index failed: %v", err)
	}

	messages, _ := store.GetMessages(ctx, "user1")
	if len(messages) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(messages))
	}

	if messages[1].Content != "Second" {
		t.Errorf("Expected 'Second' at index 1, got '%s'", messages[1].Content)
	}
}

func TestSimpleChatStoreDeleteMessages(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{
		llm.NewUserMessage("Hello"),
		llm.NewAssistantMessage("Hi!"),
	}
	store.SetMessages(ctx, "user1", messages)

	// Delete messages
	deleted, err := store.DeleteMessages(ctx, "user1")
	if err != nil {
		t.Fatalf("DeleteMessages failed: %v", err)
	}

	if len(deleted) != 2 {
		t.Errorf("Expected 2 deleted messages, got %d", len(deleted))
	}

	// Verify deletion
	remaining, _ := store.GetMessages(ctx, "user1")
	if len(remaining) != 0 {
		t.Errorf("Expected 0 messages after deletion, got %d", len(remaining))
	}
}

func TestSimpleChatStoreDeleteMessagesNonexistent(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	deleted, err := store.DeleteMessages(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("DeleteMessages failed: %v", err)
	}

	if deleted != nil {
		t.Errorf("Expected nil for nonexistent key, got %v", deleted)
	}
}

func TestSimpleChatStoreDeleteMessage(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{
		llm.NewUserMessage("First"),
		llm.NewUserMessage("Second"),
		llm.NewUserMessage("Third"),
	}
	store.SetMessages(ctx, "user1", messages)

	// Delete middle message
	deleted, err := store.DeleteMessage(ctx, "user1", 1)
	if err != nil {
		t.Fatalf("DeleteMessage failed: %v", err)
	}

	if deleted == nil || deleted.Content != "Second" {
		t.Errorf("Expected deleted message 'Second', got %v", deleted)
	}

	remaining, _ := store.GetMessages(ctx, "user1")
	if len(remaining) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(remaining))
	}
}

func TestSimpleChatStoreDeleteMessageInvalidIndex(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{llm.NewUserMessage("Hello")}
	store.SetMessages(ctx, "user1", messages)

	// Delete with invalid index
	deleted, err := store.DeleteMessage(ctx, "user1", 5)
	if err != nil {
		t.Fatalf("DeleteMessage failed: %v", err)
	}

	if deleted != nil {
		t.Errorf("Expected nil for invalid index, got %v", deleted)
	}
}

func TestSimpleChatStoreDeleteLastMessage(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{
		llm.NewUserMessage("First"),
		llm.NewUserMessage("Last"),
	}
	store.SetMessages(ctx, "user1", messages)

	// Delete last message
	deleted, err := store.DeleteLastMessage(ctx, "user1")
	if err != nil {
		t.Fatalf("DeleteLastMessage failed: %v", err)
	}

	if deleted == nil || deleted.Content != "Last" {
		t.Errorf("Expected deleted message 'Last', got %v", deleted)
	}

	remaining, _ := store.GetMessages(ctx, "user1")
	if len(remaining) != 1 {
		t.Errorf("Expected 1 message, got %d", len(remaining))
	}
}

func TestSimpleChatStoreDeleteLastMessageEmpty(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	deleted, err := store.DeleteLastMessage(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("DeleteLastMessage failed: %v", err)
	}

	if deleted != nil {
		t.Errorf("Expected nil for empty/nonexistent key, got %v", deleted)
	}
}

func TestSimpleChatStoreGetKeys(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	store.SetMessages(ctx, "user1", []llm.ChatMessage{llm.NewUserMessage("Hello")})
	store.SetMessages(ctx, "user2", []llm.ChatMessage{llm.NewUserMessage("Hi")})

	keys, err := store.GetKeys(ctx)
	if err != nil {
		t.Fatalf("GetKeys failed: %v", err)
	}

	if len(keys) != 2 {
		t.Errorf("Expected 2 keys, got %d", len(keys))
	}
}

func TestSimpleChatStorePersist(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "chat_store.json")

	store := NewSimpleChatStore()
	messages := []llm.ChatMessage{
		llm.NewSystemMessage("You are helpful"),
		llm.NewUserMessage("Hello"),
		llm.NewAssistantMessage("Hi there!"),
	}
	store.SetMessages(ctx, "user1", messages)

	// Persist
	err := store.Persist(ctx, persistPath)
	if err != nil {
		t.Fatalf("Persist failed: %v", err)
	}

	// Verify file exists
	if _, err := os.Stat(persistPath); os.IsNotExist(err) {
		t.Error("Persist file was not created")
	}

	// Load from persist path
	loadedStore, err := SimpleChatStoreFromPersistPath(ctx, persistPath)
	if err != nil {
		t.Fatalf("SimpleChatStoreFromPersistPath failed: %v", err)
	}

	// Verify loaded data
	retrieved, err := loadedStore.GetMessages(ctx, "user1")
	if err != nil {
		t.Fatalf("GetMessages failed: %v", err)
	}

	if len(retrieved) != 3 {
		t.Errorf("Expected 3 messages, got %d", len(retrieved))
	}

	if retrieved[0].Role != llm.MessageRoleSystem {
		t.Errorf("Expected system role, got %s", retrieved[0].Role)
	}
}

func TestSimpleChatStoreFromPersistPathNonexistent(t *testing.T) {
	ctx := context.Background()
	tmpDir := t.TempDir()
	persistPath := filepath.Join(tmpDir, "nonexistent.json")

	store, err := SimpleChatStoreFromPersistPath(ctx, persistPath)
	if err != nil {
		t.Fatalf("SimpleChatStoreFromPersistPath failed: %v", err)
	}

	if store == nil {
		t.Error("Expected non-nil store")
	}

	keys, _ := store.GetKeys(ctx)
	if len(keys) != 0 {
		t.Errorf("Expected empty store, got %d keys", len(keys))
	}
}

func TestSimpleChatStoreToAndFromDict(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{
		llm.NewUserMessage("Hello"),
		llm.NewAssistantMessage("Hi!"),
	}
	store.SetMessages(ctx, "user1", messages)

	// ToDict
	data := store.ToDict()

	// FromDict
	loadedStore := SimpleChatStoreFromDict(data)

	retrieved, _ := loadedStore.GetMessages(ctx, "user1")
	if len(retrieved) != 2 {
		t.Errorf("Expected 2 messages, got %d", len(retrieved))
	}
}

func TestSimpleChatStoreIsolation(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	messages := []llm.ChatMessage{llm.NewUserMessage("Hello")}
	store.SetMessages(ctx, "user1", messages)

	// Modify original slice
	messages[0].Content = "Modified"

	// Get messages should return unmodified copy
	retrieved, _ := store.GetMessages(ctx, "user1")
	if retrieved[0].Content != "Hello" {
		t.Errorf("Expected 'Hello', got '%s' - store was mutated", retrieved[0].Content)
	}
}

func TestSimpleChatStoreMultipleUsers(t *testing.T) {
	ctx := context.Background()
	store := NewSimpleChatStore()

	store.SetMessages(ctx, "user1", []llm.ChatMessage{llm.NewUserMessage("User1 message")})
	store.SetMessages(ctx, "user2", []llm.ChatMessage{llm.NewUserMessage("User2 message")})

	user1Messages, _ := store.GetMessages(ctx, "user1")
	user2Messages, _ := store.GetMessages(ctx, "user2")

	if user1Messages[0].Content != "User1 message" {
		t.Errorf("User1 message incorrect")
	}

	if user2Messages[0].Content != "User2 message" {
		t.Errorf("User2 message incorrect")
	}
}
