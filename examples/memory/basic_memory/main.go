// Package main demonstrates basic memory usage for chat history management.
// This example corresponds to Python's memory/memory.ipynb
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
)

func main() {
	ctx := context.Background()

	fmt.Println("=== Basic Memory Demo ===")

	separator := strings.Repeat("=", 60)

	// 1. Simple Memory - stores all messages
	fmt.Println("\n" + separator)
	fmt.Println("=== Simple Memory ===")
	fmt.Println(separator)

	simpleMemory := memory.NewSimpleMemory()

	fmt.Println("\nSimple memory stores all messages without any limits.")

	// Add some messages
	messages := []llm.ChatMessage{
		llm.NewUserMessage("Hello, how are you?"),
		llm.NewAssistantMessage("I'm doing well, thank you! How can I help you today?"),
		llm.NewUserMessage("What's the weather like?"),
		llm.NewAssistantMessage("I don't have access to real-time weather data, but I can help you find weather information."),
	}

	for _, msg := range messages {
		if err := simpleMemory.Put(ctx, msg); err != nil {
			log.Printf("Error adding message: %v", err)
		}
	}

	fmt.Printf("\nAdded %d messages to simple memory\n", len(messages))

	// Retrieve all messages
	history, err := simpleMemory.GetAll(ctx)
	if err != nil {
		log.Printf("Error getting history: %v", err)
	} else {
		fmt.Printf("\nRetrieved %d messages:\n", len(history))
		for i, msg := range history {
			fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, truncate(msg.Content, 50))
		}
	}

	// 2. Chat Memory Buffer - with token limits
	fmt.Println("\n" + separator)
	fmt.Println("=== Chat Memory Buffer ===")
	fmt.Println(separator)

	chatBuffer := memory.NewChatMemoryBuffer(
		memory.WithTokenLimit(100), // Low limit for demonstration
	)

	fmt.Println("\nChat memory buffer enforces token limits.")
	fmt.Printf("Token limit: %d\n", chatBuffer.TokenLimit())

	// Add many messages to exceed the token limit
	longConversation := []llm.ChatMessage{
		llm.NewUserMessage("Tell me about machine learning."),
		llm.NewAssistantMessage("Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed."),
		llm.NewUserMessage("What are the main types?"),
		llm.NewAssistantMessage("The main types are supervised learning, unsupervised learning, and reinforcement learning. Each has different use cases and approaches."),
		llm.NewUserMessage("Can you explain supervised learning?"),
		llm.NewAssistantMessage("Supervised learning uses labeled data to train models. The algorithm learns to map inputs to outputs based on example input-output pairs."),
		llm.NewUserMessage("What about neural networks?"),
		llm.NewAssistantMessage("Neural networks are computing systems inspired by biological neural networks. They consist of layers of interconnected nodes that process information."),
	}

	for _, msg := range longConversation {
		if err := chatBuffer.Put(ctx, msg); err != nil {
			log.Printf("Error adding message: %v", err)
		}
	}

	fmt.Printf("\nAdded %d messages to chat buffer\n", len(longConversation))

	// Get messages within token limit
	limitedHistory, err := chatBuffer.Get(ctx, "")
	if err != nil {
		log.Printf("Error getting limited history: %v", err)
	} else {
		fmt.Printf("\nRetrieved %d messages (within token limit):\n", len(limitedHistory))
		for i, msg := range limitedHistory {
			fmt.Printf("  %d. [%s] %s\n", i+1, msg.Role, truncate(msg.Content, 40))
		}
	}

	// Get all messages (ignoring limit)
	allHistory, err := chatBuffer.GetAll(ctx)
	if err != nil {
		log.Printf("Error getting all history: %v", err)
	} else {
		fmt.Printf("\nTotal messages stored: %d\n", len(allHistory))
	}

	// 3. Memory with custom tokenizer
	fmt.Println("\n" + separator)
	fmt.Println("=== Custom Tokenizer ===")
	fmt.Println(separator)

	// Custom tokenizer that counts words
	wordTokenizer := func(text string) int {
		return len(strings.Fields(text))
	}

	customBuffer := memory.NewChatMemoryBuffer(
		memory.WithTokenLimit(20), // 20 words
		memory.WithTokenizer(wordTokenizer),
	)

	fmt.Println("\nUsing word-based tokenizer (counts words instead of characters)")
	fmt.Printf("Token limit: %d words\n", customBuffer.TokenLimit())

	testMessages := []llm.ChatMessage{
		llm.NewUserMessage("Hello there!"),
		llm.NewAssistantMessage("Hi! How can I help you today?"),
		llm.NewUserMessage("I need help with my code."),
		llm.NewAssistantMessage("Sure, I'd be happy to help. What programming language are you using?"),
	}

	for _, msg := range testMessages {
		if err := customBuffer.Put(ctx, msg); err != nil {
			log.Printf("Error adding message: %v", err)
		}
	}

	customHistory, err := customBuffer.Get(ctx, "")
	if err != nil {
		log.Printf("Error getting custom history: %v", err)
	} else {
		fmt.Printf("\nRetrieved %d messages within word limit:\n", len(customHistory))
		for i, msg := range customHistory {
			wordCount := len(strings.Fields(msg.Content))
			fmt.Printf("  %d. [%s] (%d words) %s\n", i+1, msg.Role, wordCount, truncate(msg.Content, 40))
		}
	}

	// 4. Memory operations
	fmt.Println("\n" + separator)
	fmt.Println("=== Memory Operations ===")
	fmt.Println(separator)

	opMemory := memory.NewSimpleMemory()

	// Put single message
	fmt.Println("\n1. Put single message:")
	err = opMemory.Put(ctx, llm.NewUserMessage("First message"))
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Println("   Added: 'First message'")
	}

	// Put multiple messages
	fmt.Println("\n2. Put multiple messages:")
	err = opMemory.PutMessages(ctx, []llm.ChatMessage{
		llm.NewAssistantMessage("Response to first"),
		llm.NewUserMessage("Second message"),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Println("   Added 2 more messages")
	}

	history, _ = opMemory.GetAll(ctx)
	fmt.Printf("   Total messages: %d\n", len(history))

	// Set (replace all)
	fmt.Println("\n3. Set (replace all):")
	err = opMemory.Set(ctx, []llm.ChatMessage{
		llm.NewUserMessage("New conversation"),
		llm.NewAssistantMessage("Starting fresh!"),
	})
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Println("   Replaced with 2 new messages")
	}

	history, _ = opMemory.GetAll(ctx)
	fmt.Printf("   Total messages: %d\n", len(history))

	// Reset (clear all)
	fmt.Println("\n4. Reset (clear all):")
	err = opMemory.Reset(ctx)
	if err != nil {
		log.Printf("Error: %v", err)
	} else {
		fmt.Println("   Cleared all messages")
	}

	history, _ = opMemory.GetAll(ctx)
	fmt.Printf("   Total messages: %d\n", len(history))

	// 5. Memory with chat store key
	fmt.Println("\n" + separator)
	fmt.Println("=== Multiple Conversations ===")
	fmt.Println(separator)

	fmt.Println("\nUsing different chat store keys for separate conversations:")

	conv1Memory := memory.NewSimpleMemory(
		memory.WithChatStoreKey("conversation_1"),
	)
	conv2Memory := memory.NewSimpleMemory(
		memory.WithChatStoreKey("conversation_2"),
	)

	// Add messages to conversation 1
	conv1Memory.Put(ctx, llm.NewUserMessage("Hello from conversation 1"))
	conv1Memory.Put(ctx, llm.NewAssistantMessage("Hi! This is conversation 1"))

	// Add messages to conversation 2
	conv2Memory.Put(ctx, llm.NewUserMessage("Hello from conversation 2"))
	conv2Memory.Put(ctx, llm.NewAssistantMessage("Hi! This is conversation 2"))

	conv1History, _ := conv1Memory.GetAll(ctx)
	conv2History, _ := conv2Memory.GetAll(ctx)

	fmt.Printf("\nConversation 1 (%s): %d messages\n", conv1Memory.ChatStoreKey(), len(conv1History))
	for _, msg := range conv1History {
		fmt.Printf("  [%s] %s\n", msg.Role, msg.Content)
	}

	fmt.Printf("\nConversation 2 (%s): %d messages\n", conv2Memory.ChatStoreKey(), len(conv2History))
	for _, msg := range conv2History {
		fmt.Printf("  [%s] %s\n", msg.Role, msg.Content)
	}

	// 6. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nMemory Types:")
	fmt.Println("  - SimpleMemory: Stores all messages without limits")
	fmt.Println("  - ChatMemoryBuffer: Enforces token limits, trims old messages")
	fmt.Println()
	fmt.Println("Key Operations:")
	fmt.Println("  - Put: Add a single message")
	fmt.Println("  - PutMessages: Add multiple messages")
	fmt.Println("  - Get: Retrieve messages (may apply limits)")
	fmt.Println("  - GetAll: Retrieve all messages")
	fmt.Println("  - Set: Replace entire history")
	fmt.Println("  - Reset: Clear all messages")
	fmt.Println()
	fmt.Println("Configuration:")
	fmt.Println("  - Token limit: Maximum tokens to return")
	fmt.Println("  - Tokenizer: Function to count tokens")
	fmt.Println("  - Chat store key: Separate different conversations")

	fmt.Println("\n=== Basic Memory Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
