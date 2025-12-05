// Package main demonstrates streaming with chat engines.
// This example corresponds to Python's customization/streaming/chat_engine_condense_question_stream_response.ipynb
package main

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/aqua777/go-llamaindex/chatengine"
	"github.com/aqua777/go-llamaindex/llm"
	"github.com/aqua777/go-llamaindex/memory"
	"github.com/aqua777/go-llamaindex/rag/retriever"
	"github.com/aqua777/go-llamaindex/schema"
)

// MockRetriever simulates a retriever for demonstration.
type MockRetriever struct {
	*retriever.BaseRetriever
	documents map[string]string
}

// NewMockRetriever creates a mock retriever with predefined documents.
func NewMockRetriever(docs map[string]string) *MockRetriever {
	return &MockRetriever{
		BaseRetriever: retriever.NewBaseRetriever(),
		documents:     docs,
	}
}

// Retrieve implements retriever.Retriever.
func (m *MockRetriever) Retrieve(ctx context.Context, query schema.QueryBundle) ([]schema.NodeWithScore, error) {
	queryLower := strings.ToLower(query.QueryString)
	var results []schema.NodeWithScore

	for keyword, content := range m.documents {
		if strings.Contains(queryLower, strings.ToLower(keyword)) {
			results = append(results, schema.NodeWithScore{
				Node:  schema.Node{ID: keyword, Text: content},
				Score: 0.9,
			})
		}
	}

	// Return at least one result for demo
	if len(results) == 0 && len(m.documents) > 0 {
		for keyword, content := range m.documents {
			results = append(results, schema.NodeWithScore{
				Node:  schema.Node{ID: keyword, Text: content},
				Score: 0.5,
			})
			break
		}
	}

	return results, nil
}

func main() {
	ctx := context.Background()

	fmt.Println("=== Chat Engine Streaming Demo ===")
	fmt.Println("\nDemonstrates streaming responses from chat engines.")

	separator := strings.Repeat("=", 60)

	// Create LLM
	llmInstance := llm.NewOpenAILLM("", "", "")

	// Create mock knowledge base
	knowledgeBase := map[string]string{
		"go": "Go (Golang) is a statically typed, compiled programming language designed at Google. " +
			"It features garbage collection, structural typing, and CSP-style concurrency with goroutines and channels.",
		"goroutine": "A goroutine is a lightweight thread managed by the Go runtime. " +
			"Goroutines are created with the 'go' keyword and can communicate via channels.",
		"channel": "Channels are typed conduits through which you can send and receive values with the channel operator. " +
			"They provide synchronization and communication between goroutines.",
		"interface": "An interface in Go is a type that specifies a set of method signatures. " +
			"A type implements an interface by implementing its methods. Interfaces enable polymorphism in Go.",
	}

	mockRetriever := NewMockRetriever(knowledgeBase)

	// 1. SimpleChatEngine streaming
	fmt.Println("\n" + separator)
	fmt.Println("=== SimpleChatEngine Streaming ===")
	fmt.Println(separator)

	simpleChatEngine := chatengine.NewSimpleChatEngine(
		chatengine.WithSimpleChatEngineLLM(llmInstance),
		chatengine.WithSimpleChatEngineMemory(memory.NewChatMemoryBuffer(memory.WithTokenLimit(2000))),
		chatengine.WithSimpleChatEngineSystemPrompt("You are a helpful Go programming assistant. Be concise."),
	)

	fmt.Println("\nSimpleChatEngine with streaming:")

	message1 := "What is Go programming language?"
	fmt.Printf("\nUser: %s\n", message1)
	fmt.Print("Assistant: ")

	streamResp, err := simpleChatEngine.StreamChat(ctx, message1)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamResp.ResponseChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// Follow-up question (uses memory)
	message2 := "What are its main features?"
	fmt.Printf("\nUser: %s\n", message2)
	fmt.Print("Assistant: ")

	streamResp, err = simpleChatEngine.StreamChat(ctx, message2)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamResp.ResponseChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// 2. ContextChatEngine streaming
	fmt.Println("\n" + separator)
	fmt.Println("=== ContextChatEngine Streaming ===")
	fmt.Println(separator)

	contextChatEngine := chatengine.NewContextChatEngine(
		chatengine.WithContextChatEngineLLM(llmInstance),
		chatengine.WithContextChatEngineRetriever(mockRetriever),
		chatengine.WithContextChatEngineMemory(memory.NewChatMemoryBuffer(memory.WithTokenLimit(2000))),
		chatengine.WithContextChatEngineSystemPrompt(
			"You are a Go programming expert. Answer questions based on the provided context. Be concise.",
		),
	)

	fmt.Println("\nContextChatEngine with RAG streaming:")

	message3 := "What are goroutines and how do they work?"
	fmt.Printf("\nUser: %s\n", message3)
	fmt.Print("Assistant: ")

	streamResp, err = contextChatEngine.StreamChat(ctx, message3)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamResp.ResponseChan {
			fmt.Print(token)
		}
		fmt.Println()

		// Show source nodes
		if len(streamResp.SourceNodes) > 0 {
			fmt.Println("\nSource nodes used:")
			for _, node := range streamResp.SourceNodes {
				fmt.Printf("  - [%.2f] %s\n", node.Score, truncate(node.Node.Text, 60))
			}
		}
	}

	// 3. CondensePlusContextChatEngine streaming
	fmt.Println("\n" + separator)
	fmt.Println("=== CondensePlusContextChatEngine Streaming ===")
	fmt.Println(separator)

	condenseChatEngine := chatengine.NewCondensePlusContextChatEngine(
		chatengine.WithCondensePlusContextLLM(llmInstance),
		chatengine.WithCondensePlusContextRetriever(mockRetriever),
		chatengine.WithCondensePlusContextMemory(memory.NewChatMemoryBuffer(memory.WithTokenLimit(2000))),
		chatengine.WithCondensePlusContextSystemPrompt(
			"You are a Go programming expert. Answer questions based on the provided context.",
		),
	)

	fmt.Println("\nCondensePlusContextChatEngine with question condensing:")

	// First question
	message4 := "Tell me about channels in Go"
	fmt.Printf("\nUser: %s\n", message4)
	fmt.Print("Assistant: ")

	streamResp, err = condenseChatEngine.StreamChat(ctx, message4)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamResp.ResponseChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// Follow-up (will be condensed with context)
	message5 := "How do they relate to goroutines?"
	fmt.Printf("\nUser: %s\n", message5)
	fmt.Println("(This question will be condensed with conversation history)")
	fmt.Print("Assistant: ")

	streamResp, err = condenseChatEngine.StreamChat(ctx, message5)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		for token := range streamResp.ResponseChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// 4. Streaming with Consume()
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming with Consume() ===")
	fmt.Println(separator)

	fmt.Println("\nUsing Consume() to get full response:")

	message6 := "What is an interface in Go?"
	fmt.Printf("\nUser: %s\n", message6)

	streamResp, err = contextChatEngine.StreamChat(ctx, message6)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		// Consume blocks until streaming is complete
		fullResponse := streamResp.Consume()
		fmt.Printf("Full response: %s\n", truncate(fullResponse, 200))
		fmt.Printf("IsDone: %v\n", streamResp.IsDone())
	}

	// 5. Streaming with progress tracking
	fmt.Println("\n" + separator)
	fmt.Println("=== Streaming with Progress Tracking ===")
	fmt.Println(separator)

	message7 := "Explain Go's error handling approach"
	fmt.Printf("\nUser: %s\n", message7)
	fmt.Println("Streaming with progress:")

	streamResp, err = simpleChatEngine.StreamChat(ctx, message7)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
	} else {
		var fullResponse strings.Builder
		tokenCount := 0
		startTime := time.Now()

		for token := range streamResp.ResponseChan {
			fullResponse.WriteString(token)
			tokenCount++

			// Update progress
			elapsed := time.Since(startTime)
			fmt.Printf("\r[Tokens: %d, Time: %.1fs]", tokenCount, elapsed.Seconds())
		}

		elapsed := time.Since(startTime)
		fmt.Printf("\r[Complete: %d tokens in %.2fs]                    \n",
			tokenCount, elapsed.Seconds())
		fmt.Printf("\nResponse: %s\n", truncate(fullResponse.String(), 150))
	}

	// 6. Streaming conversation loop
	fmt.Println("\n" + separator)
	fmt.Println("=== Multi-turn Streaming Conversation ===")
	fmt.Println(separator)

	// Reset the engine for fresh conversation
	_ = contextChatEngine.Reset(ctx)

	questions := []string{
		"What is Go?",
		"What are its concurrency features?",
		"Give me a simple example",
	}

	fmt.Println("\nMulti-turn conversation with streaming:")

	for i, question := range questions {
		fmt.Printf("\n--- Turn %d ---\n", i+1)
		fmt.Printf("User: %s\n", question)
		fmt.Print("Assistant: ")

		streamResp, err := contextChatEngine.StreamChat(ctx, question)
		if err != nil {
			fmt.Printf("Error: %v\n", err)
			continue
		}

		for token := range streamResp.ResponseChan {
			fmt.Print(token)
		}
		fmt.Println()
	}

	// 7. Summary
	fmt.Println("\n" + separator)
	fmt.Println("=== Summary ===")
	fmt.Println(separator)

	fmt.Println("\nChat Engine Streaming Features:")
	fmt.Println("  - StreamChat(): Returns StreamingChatResponse")
	fmt.Println("  - ResponseChan: Channel for streaming tokens")
	fmt.Println("  - Consume(): Block and get full response")
	fmt.Println("  - IsDone(): Check if streaming complete")
	fmt.Println("  - SourceNodes: Retrieved context (for RAG engines)")
	fmt.Println()
	fmt.Println("Chat Engine Types:")
	fmt.Println("  - SimpleChatEngine: Basic chat with memory")
	fmt.Println("  - ContextChatEngine: RAG-based chat")
	fmt.Println("  - CondensePlusContextChatEngine: Question condensing + RAG")
	fmt.Println()
	fmt.Println("Use Cases:")
	fmt.Println("  - Interactive chat interfaces")
	fmt.Println("  - Real-time response display")
	fmt.Println("  - RAG-powered assistants")
	fmt.Println("  - Multi-turn conversations")

	fmt.Println("\n=== Chat Engine Streaming Demo Complete ===")
}

// truncate truncates a string to the specified length.
func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
