// Package main demonstrates implementing custom memory types.
// This example corresponds to Python's memory/custom_memory.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Custom Memory Demo ===")

	separator := strings.Repeat("=", 60)

	// 1. Sliding Window Memory
	fmt.Println("\n" + separator)
	fmt.Println("=== Sliding Window Memory ===")
	fmt.Println(separator)

	windowMemory := NewSlidingWindowMemory(4) // Keep last 4 messages

	fmt.Println("\nSliding window memory keeps only the last N messages")
	fmt.Printf("Window size: %d messages\n", windowMemory.windowSize)

	messages := []llm.ChatMessage{
		llm.NewUserMessage("Message 1: Hello!"),
		llm.NewAssistantMessage("Message 2: Hi there!"),
		llm.NewUserMessage("Message 3: How are you?"),
		llm.NewAssistantMessage("Message 4: I'm doing great!"),
		llm.NewUserMessage("Message 5: What's new?"),
		llm.NewAssistantMessage("Message 6: Not much, you?"),
	}

	fmt.Printf("\nAdding %d messages...\n", len(messages))
	for _, msg := range messages {
		windowMemory.Put(ctx, msg)
	}

	history, _ := windowMemory.Get(ctx, "")
	fmt.Printf("\nRetrieved %d messages (window size = 4):\n", len(history))
	for i, msg := range history {
		fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, msg.Content)
	}

	// 2. Timestamped Memory
	fmt.Println("\n" + separator)
	fmt.Println("=== Timestamped Memory ===")
	fmt.Println(separator)

	timestampMemory := NewTimestampedMemory()

	fmt.Println("\nTimestamped memory tracks when each message was added")

	// Add messages with delays
	timestampMemory.Put(ctx, llm.NewUserMessage("First message"))
	time.Sleep(100 * time.Millisecond)
	timestampMemory.Put(ctx, llm.NewAssistantMessage("Response to first"))
	time.Sleep(100 * time.Millisecond)
	timestampMemory.Put(ctx, llm.NewUserMessage("Second message"))

	fmt.Println("\nMessages with timestamps:")
	for i, entry := range timestampMemory.entries {
		fmt.Printf("  %d. [%s] %s (added: %s)\n",
			i+1, entry.Message.Role, truncate(entry.Message.Content, 30),
			entry.Timestamp.Format("15:04:05.000"))
	}

	// Get messages from last 150ms
	recentHistory := timestampMemory.GetSince(ctx, 150*time.Millisecond)
	fmt.Printf("\nMessages from last 150ms: %d\n", len(recentHistory))

	// 3. Tagged Memory
	fmt.Println("\n" + separator)
	fmt.Println("=== Tagged Memory ===")
	fmt.Println(separator)

	taggedMemory := NewTaggedMemory()

	fmt.Println("\nTagged memory allows categorizing and filtering messages")

	// Add messages with tags
	taggedMemory.PutWithTags(ctx, llm.NewUserMessage("What's the weather?"), []string{"weather", "question"})
	taggedMemory.PutWithTags(ctx, llm.NewAssistantMessage("It's sunny today!"), []string{"weather", "answer"})
	taggedMemory.PutWithTags(ctx, llm.NewUserMessage("Tell me a joke"), []string{"humor", "question"})
	taggedMemory.PutWithTags(ctx, llm.NewAssistantMessage("Why did the programmer quit? No arrays!"), []string{"humor", "answer"})
	taggedMemory.PutWithTags(ctx, llm.NewUserMessage("What's 2+2?"), []string{"math", "question"})
	taggedMemory.PutWithTags(ctx, llm.NewAssistantMessage("2+2 equals 4"), []string{"math", "answer"})

	fmt.Printf("\nAdded 6 messages with various tags\n")

	// Get all messages
	allTagged, _ := taggedMemory.GetAll(ctx)
	fmt.Printf("\nAll messages: %d\n", len(allTagged))

	// Get by tag
	weatherMsgs := taggedMemory.GetByTag(ctx, "weather")
	fmt.Printf("\nMessages tagged 'weather': %d\n", len(weatherMsgs))
	for _, msg := range weatherMsgs {
		fmt.Printf("  [%s] %s\n", msg.Role, msg.Content)
	}

	questionMsgs := taggedMemory.GetByTag(ctx, "question")
	fmt.Printf("\nMessages tagged 'question': %d\n", len(questionMsgs))
	for _, msg := range questionMsgs {
		fmt.Printf("  [%s] %s\n", msg.Role, msg.Content)
	}

	// 4. Priority Memory
	fmt.Println("\n" + separator)
	fmt.Println("=== Priority Memory ===")
	fmt.Println(separator)

	priorityMemory := NewPriorityMemory(3) // Keep top 3 priority messages

	fmt.Println("\nPriority memory keeps messages based on importance score")
	fmt.Printf("Max messages: %d\n", priorityMemory.maxMessages)

	// Add messages with priorities
	priorityMemory.PutWithPriority(ctx, llm.NewUserMessage("Casual greeting"), 1)
	priorityMemory.PutWithPriority(ctx, llm.NewAssistantMessage("Casual response"), 1)
	priorityMemory.PutWithPriority(ctx, llm.NewUserMessage("Important question about billing"), 5)
	priorityMemory.PutWithPriority(ctx, llm.NewAssistantMessage("Billing info response"), 5)
	priorityMemory.PutWithPriority(ctx, llm.NewUserMessage("Critical security issue!"), 10)
	priorityMemory.PutWithPriority(ctx, llm.NewAssistantMessage("Security response"), 10)

	fmt.Printf("\nAdded 6 messages with priorities 1, 5, and 10\n")

	priorityHistory, _ := priorityMemory.Get(ctx, "")
	fmt.Printf("\nTop %d priority messages:\n", len(priorityHistory))
	for i, msg := range priorityHistory {
		fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, truncate(msg.Content, 40))
	}

	// 5. Conversation Memory (groups by conversation turns)
	fmt.Println("\n" + separator)
	fmt.Println("=== Conversation Turn Memory ===")
	fmt.Println(separator)

	turnMemory := NewConversationTurnMemory(2) // Keep last 2 turns

	fmt.Println("\nConversation turn memory groups user-assistant pairs")
	fmt.Printf("Max turns: %d\n", turnMemory.maxTurns)

	// Add conversation turns
	turnMemory.Put(ctx, llm.NewUserMessage("Turn 1: Hello"))
	turnMemory.Put(ctx, llm.NewAssistantMessage("Turn 1: Hi!"))
	turnMemory.Put(ctx, llm.NewUserMessage("Turn 2: How are you?"))
	turnMemory.Put(ctx, llm.NewAssistantMessage("Turn 2: Great!"))
	turnMemory.Put(ctx, llm.NewUserMessage("Turn 3: What's up?"))
	turnMemory.Put(ctx, llm.NewAssistantMessage("Turn 3: Not much!"))

	turnHistory, _ := turnMemory.Get(ctx, "")
	fmt.Printf("\nLast %d turns (%d messages):\n", turnMemory.maxTurns, len(turnHistory))
	for i, msg := range turnHistory {
		fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, msg.Content)
	}

	// 6. Implementing the Memory interface
	fmt.Println("\n" + separator)
	fmt.Println("=== Memory Interface Implementation ===")
	fmt.Println(separator)

	fmt.Println("\nThe Memory interface requires these methods:")
	fmt.Println("  - Get(ctx, input) - Retrieve messages (may filter)")
	fmt.Println("  - GetAll(ctx) - Retrieve all messages")
	fmt.Println("  - Put(ctx, message) - Add a single message")
	fmt.Println("  - PutMessages(ctx, messages) - Add multiple messages")
	fmt.Println("  - Set(ctx, messages) - Replace all messages")
	fmt.Println("  - Reset(ctx) - Clear all messages")

	// Verify our custom memories implement the interface
	var _ memory.Memory = (*SlidingWindowMemory)(nil)
	var _ memory.Memory = (*TimestampedMemory)(nil)
	var _ memory.Memory = (*TaggedMemory)(nil)
	var _ memory.Memory = (*PriorityMemory)(nil)
	var _ memory.Memory = (*ConversationTurnMemory)(nil)

	fmt.Println("\nAll custom memory types implement memory.Memory interface ✓")

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nCustom Memory Implementations:")
	fmt.Println("  1. SlidingWindowMemory: Fixed-size window of recent messages")
	fmt.Println("  2. TimestampedMemory: Track message timing, filter by recency")
	fmt.Println("  3. TaggedMemory: Categorize and filter by tags")
	fmt.Println("  4. PriorityMemory: Keep highest priority messages")
	fmt.Println("  5. ConversationTurnMemory: Group by user-assistant turns")
	fmt.Println()
	fmt.Println("Creating Custom Memory:")
	fmt.Println("  1. Implement the memory.Memory interface")
	fmt.Println("  2. Decide on storage strategy (in-memory, persistent)")
	fmt.Println("  3. Define filtering/retrieval logic in Get()")
	fmt.Println("  4. Handle message addition in Put()")
	fmt.Println("  5. Implement Set() and Reset() for state management")

	fmt.Println("\n=== Custom Memory Demo Complete ===")
}

// =============================================================================
// Custom Memory Implementations
// =============================================================================

// SlidingWindowMemory keeps only the last N messages.
type SlidingWindowMemory struct {
	messages   []llm.ChatMessage
	windowSize int
	mu         sync.RWMutex
}

// NewSlidingWindowMemory creates a new sliding window memory.
func NewSlidingWindowMemory(windowSize int) *SlidingWindowMemory {
	return &SlidingWindowMemory{
		messages:   []llm.ChatMessage{},
		windowSize: windowSize,
	}
}

func (m *SlidingWindowMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if len(m.messages) <= m.windowSize {
		result := make([]llm.ChatMessage, len(m.messages))
		copy(result, m.messages)
		return result, nil
	}

	start := len(m.messages) - m.windowSize
	result := make([]llm.ChatMessage, m.windowSize)
	copy(result, m.messages[start:])
	return result, nil
}

func (m *SlidingWindowMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]llm.ChatMessage, len(m.messages))
	copy(result, m.messages)
	return result, nil
}

func (m *SlidingWindowMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages, message)
	return nil
}

func (m *SlidingWindowMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *SlidingWindowMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = make([]llm.ChatMessage, len(messages))
	copy(m.messages, messages)
	return nil
}

func (m *SlidingWindowMemory) Reset(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = []llm.ChatMessage{}
	return nil
}

// TimestampedMemory tracks when messages were added.
type TimestampedMemory struct {
	entries []TimestampedEntry
	mu      sync.RWMutex
}

// TimestampedEntry holds a message with its timestamp.
type TimestampedEntry struct {
	Message   llm.ChatMessage
	Timestamp time.Time
}

// NewTimestampedMemory creates a new timestamped memory.
func NewTimestampedMemory() *TimestampedMemory {
	return &TimestampedMemory{
		entries: []TimestampedEntry{},
	}
}

func (m *TimestampedMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	return m.GetAll(ctx)
}

func (m *TimestampedMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]llm.ChatMessage, len(m.entries))
	for i, entry := range m.entries {
		result[i] = entry.Message
	}
	return result, nil
}

// GetSince returns messages added within the specified duration.
func (m *TimestampedMemory) GetSince(ctx context.Context, duration time.Duration) []llm.ChatMessage {
	m.mu.RLock()
	defer m.mu.RUnlock()

	cutoff := time.Now().Add(-duration)
	var result []llm.ChatMessage
	for _, entry := range m.entries {
		if entry.Timestamp.After(cutoff) {
			result = append(result, entry.Message)
		}
	}
	return result
}

func (m *TimestampedMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, TimestampedEntry{
		Message:   message,
		Timestamp: time.Now(),
	})
	return nil
}

func (m *TimestampedMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *TimestampedMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = make([]TimestampedEntry, len(messages))
	now := time.Now()
	for i, msg := range messages {
		m.entries[i] = TimestampedEntry{Message: msg, Timestamp: now}
	}
	return nil
}

func (m *TimestampedMemory) Reset(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = []TimestampedEntry{}
	return nil
}

// TaggedMemory allows tagging and filtering messages.
type TaggedMemory struct {
	entries []TaggedEntry
	mu      sync.RWMutex
}

// TaggedEntry holds a message with its tags.
type TaggedEntry struct {
	Message llm.ChatMessage
	Tags    []string
}

// NewTaggedMemory creates a new tagged memory.
func NewTaggedMemory() *TaggedMemory {
	return &TaggedMemory{
		entries: []TaggedEntry{},
	}
}

func (m *TaggedMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	return m.GetAll(ctx)
}

func (m *TaggedMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]llm.ChatMessage, len(m.entries))
	for i, entry := range m.entries {
		result[i] = entry.Message
	}
	return result, nil
}

// GetByTag returns messages with the specified tag.
func (m *TaggedMemory) GetByTag(ctx context.Context, tag string) []llm.ChatMessage {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var result []llm.ChatMessage
	for _, entry := range m.entries {
		for _, t := range entry.Tags {
			if t == tag {
				result = append(result, entry.Message)
				break
			}
		}
	}
	return result
}

func (m *TaggedMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	return m.PutWithTags(ctx, message, nil)
}

// PutWithTags adds a message with tags.
func (m *TaggedMemory) PutWithTags(ctx context.Context, message llm.ChatMessage, tags []string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, TaggedEntry{Message: message, Tags: tags})
	return nil
}

func (m *TaggedMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *TaggedMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = make([]TaggedEntry, len(messages))
	for i, msg := range messages {
		m.entries[i] = TaggedEntry{Message: msg, Tags: nil}
	}
	return nil
}

func (m *TaggedMemory) Reset(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = []TaggedEntry{}
	return nil
}

// PriorityMemory keeps messages based on priority.
type PriorityMemory struct {
	entries     []PriorityEntry
	maxMessages int
	mu          sync.RWMutex
}

// PriorityEntry holds a message with its priority.
type PriorityEntry struct {
	Message  llm.ChatMessage
	Priority int
}

// NewPriorityMemory creates a new priority memory.
func NewPriorityMemory(maxMessages int) *PriorityMemory {
	return &PriorityMemory{
		entries:     []PriorityEntry{},
		maxMessages: maxMessages,
	}
}

func (m *PriorityMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Sort by priority (highest first) and take top N
	sorted := make([]PriorityEntry, len(m.entries))
	copy(sorted, m.entries)

	// Simple bubble sort for demo
	for i := 0; i < len(sorted); i++ {
		for j := i + 1; j < len(sorted); j++ {
			if sorted[j].Priority > sorted[i].Priority {
				sorted[i], sorted[j] = sorted[j], sorted[i]
			}
		}
	}

	count := m.maxMessages
	if count > len(sorted) {
		count = len(sorted)
	}

	result := make([]llm.ChatMessage, count)
	for i := 0; i < count; i++ {
		result[i] = sorted[i].Message
	}
	return result, nil
}

func (m *PriorityMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]llm.ChatMessage, len(m.entries))
	for i, entry := range m.entries {
		result[i] = entry.Message
	}
	return result, nil
}

func (m *PriorityMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	return m.PutWithPriority(ctx, message, 0)
}

// PutWithPriority adds a message with a priority score.
func (m *PriorityMemory) PutWithPriority(ctx context.Context, message llm.ChatMessage, priority int) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = append(m.entries, PriorityEntry{Message: message, Priority: priority})
	return nil
}

func (m *PriorityMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *PriorityMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = make([]PriorityEntry, len(messages))
	for i, msg := range messages {
		m.entries[i] = PriorityEntry{Message: msg, Priority: 0}
	}
	return nil
}

func (m *PriorityMemory) Reset(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries = []PriorityEntry{}
	return nil
}

// ConversationTurnMemory groups messages by conversation turns.
type ConversationTurnMemory struct {
	messages []llm.ChatMessage
	maxTurns int
	mu       sync.RWMutex
}

// NewConversationTurnMemory creates a new conversation turn memory.
func NewConversationTurnMemory(maxTurns int) *ConversationTurnMemory {
	return &ConversationTurnMemory{
		messages: []llm.ChatMessage{},
		maxTurns: maxTurns,
	}
}

func (m *ConversationTurnMemory) Get(ctx context.Context, input string) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	// Count turns (user-assistant pairs)
	turns := 0
	startIdx := len(m.messages)

	for i := len(m.messages) - 1; i >= 0; i-- {
		if m.messages[i].Role == llm.MessageRoleUser {
			turns++
			if turns > m.maxTurns {
				break
			}
			startIdx = i
		}
	}

	result := make([]llm.ChatMessage, len(m.messages)-startIdx)
	copy(result, m.messages[startIdx:])
	return result, nil
}

func (m *ConversationTurnMemory) GetAll(ctx context.Context) ([]llm.ChatMessage, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	result := make([]llm.ChatMessage, len(m.messages))
	copy(result, m.messages)
	return result, nil
}

func (m *ConversationTurnMemory) Put(ctx context.Context, message llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = append(m.messages, message)
	return nil
}

func (m *ConversationTurnMemory) PutMessages(ctx context.Context, messages []llm.ChatMessage) error {
	for _, msg := range messages {
		if err := m.Put(ctx, msg); err != nil {
			return err
		}
	}
	return nil
}

func (m *ConversationTurnMemory) Set(ctx context.Context, messages []llm.ChatMessage) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = make([]llm.ChatMessage, len(messages))
	copy(m.messages, messages)
	return nil
}

func (m *ConversationTurnMemory) Reset(ctx context.Context) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.messages = []llm.ChatMessage{}
	return nil
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// Verify interface implementations
var _ memory.Memory = (*SlidingWindowMemory)(nil)
var _ memory.Memory = (*TimestampedMemory)(nil)
var _ memory.Memory = (*TaggedMemory)(nil)
var _ memory.Memory = (*PriorityMemory)(nil)
var _ memory.Memory = (*ConversationTurnMemory)(nil)
